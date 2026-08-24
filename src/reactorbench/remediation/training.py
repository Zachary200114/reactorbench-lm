"""Deterministic compact-target fitting with safe interruption recovery.

This module accepts only already verified tokenized examples and provenance hashes.
It does not locate or load datasets.  Durable recovery state is a closed inventory
of canonical JSON and safetensors; pickle and arbitrary optimizer objects are never
accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import resource
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import torch
from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator
from safetensors.torch import load_file, save_file
from torch import Tensor

from reactorbench.model.checkpoint import load_checkpoint, save_checkpoint
from reactorbench.model.config import TransformerConfig
from reactorbench.model.transformer import (
    TransformerLM,
    exact_parameter_count,
    initialized_model,
)
from reactorbench.schemas.base import ContractModel, canonical_json_bytes, canonical_sha256
from reactorbench.tokenizer import TokenizerArtifactManifest

from .config import RemediationTraining
from .sampling import task_balanced_batch_indices
from .serialization import (
    CompactTokenizedExample,
    compact_batch_tensors,
    supervised_causal_loss,
)

TRAINING_CONTRACT_VERSION: Literal["0.3.0"] = "0.3.0"
TRAINING_STATE_VERSION: Literal["0.3.0"] = "0.3.0"
STATE_MANIFEST_FILENAME = "manifest.json"
STATE_MODEL_FILENAME = "model.safetensors"
STATE_BEST_MODEL_FILENAME = "best_model.safetensors"
STATE_OPTIMIZER_FILENAME = "optimizer.safetensors"
STATE_FILE_INVENTORY = (
    STATE_BEST_MODEL_FILENAME,
    STATE_MANIFEST_FILENAME,
    STATE_MODEL_FILENAME,
    STATE_OPTIMIZER_FILENAME,
)
MAX_STATE_MANIFEST_BYTES = 1024 * 1024
MAX_STATE_TENSOR_FILE_BYTES = 512 * 1024 * 1024
MAX_STATE_FILES_TOTAL_BYTES = 1536 * 1024 * 1024
MAX_RETAINED_DURABLE_STATES = 2
MAX_SAFETENSORS_HEADER_BYTES = 1024 * 1024
MAX_SERIALIZED_RNG_BYTES = 1024 * 1024
MAX_EXAMPLES = 1_000_000
MAX_BATCH_SIZE = 4096
MAX_SAFE_INTEGER = (1 << 63) - 1
_TRAINING_RNG_LOCK = threading.Lock()
_STATE_DIRECTORY_PATTERN = re.compile(r"^state-step-([0-9]{8})$")
_STATE_LOCK_PATTERN = re.compile(r"^\.state-step-[0-9]{8}\.lock$")
_STATE_TEMPORARY_PATTERN = re.compile(r"^\.state-step-[0-9]{8}\.tmp-[A-Za-z0-9_-]{1,64}$")

Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
CandidateId = Annotated[
    StrictStr,
    Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
]
type SamplingStrategy = Literal["uniform_control", "task_balanced"]
type EvaluationCallback = Callable[[TransformerLM, int, float], float]
type ProgressCallback = Callable[["TrainingProgress"], None]
type StopRequested = Callable[[int], bool]
type MonotonicClock = Callable[[], float]


class TrainingError(RuntimeError):
    """Safe public failure raised by the compact-target fitting core."""


class DeviceResolution(ContractModel):
    requested: Literal["cpu", "mps"]
    resolved: Literal["cpu", "mps"]
    fallback_used: StrictBool

    @model_validator(mode="after")
    def fallback_flag_matches(self) -> DeviceResolution:
        if self.fallback_used != (self.requested == "mps" and self.resolved == "cpu"):
            raise ValueError("device fallback flag is inconsistent")
        return self


class TrainingEvaluationPoint(ContractModel):
    step: StrictInt = Field(ge=0, le=50_000)
    validation_nll: StrictFloat = Field(ge=0.0, allow_inf_nan=False)
    selection_score: StrictFloat = Field(ge=0.0, allow_inf_nan=False)


class TrainingProgress(ContractModel):
    event: Literal["evaluation", "durable_checkpoint", "final_checkpoint", "stopped"]
    step: StrictInt = Field(ge=0, le=50_000)
    total_steps: StrictInt = Field(ge=1, le=50_000)
    validation_nll: StrictFloat | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    selection_score: StrictFloat | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    checkpoint_name: StrictStr | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )

    @model_validator(mode="after")
    def event_fields_match(self) -> TrainingProgress:
        if self.step > self.total_steps:
            raise ValueError("progress step exceeds total steps")
        if self.event == "evaluation" and (
            self.validation_nll is None or self.selection_score is None
        ):
            raise ValueError("evaluation progress requires both measurements")
        if self.event in {"durable_checkpoint", "final_checkpoint", "stopped"} and (
            self.checkpoint_name is None
        ):
            raise ValueError("checkpoint progress requires a safe checkpoint name")
        return self


class TrainingStateFile(ContractModel):
    filename: Literal[
        "best_model.safetensors",
        "model.safetensors",
        "optimizer.safetensors",
    ]
    sha256: Sha256
    size_bytes: StrictInt = Field(ge=1, le=MAX_STATE_TENSOR_FILE_BYTES)


class TrainingStateManifest(ContractModel):
    artifact_version: Literal["0.3.0"] = TRAINING_STATE_VERSION
    candidate_id: CandidateId
    sampling_strategy: SamplingStrategy
    source_commit: StrictStr = Field(pattern=r"^[0-9a-f]{7,64}$")
    device: DeviceResolution
    step: StrictInt = Field(ge=1, le=50_000)
    total_steps: StrictInt = Field(ge=1, le=50_000)
    sampler_cursor: StrictInt = Field(ge=1, le=MAX_SAFE_INTEGER)
    batch_size: StrictInt = Field(ge=1, le=MAX_BATCH_SIZE)
    vocab_size: StrictInt = Field(ge=8, le=65_536)
    training_config_sha256: Sha256
    model_config_sha256: Sha256
    train_inventory_sha256: Sha256
    validation_inventory_sha256: Sha256
    train_tokenized_sha256: Sha256
    validation_tokenized_sha256: Sha256
    tokenizer_manifest_sha256: Sha256
    initial_validation_nll: StrictFloat = Field(ge=0.0, allow_inf_nan=False)
    best_step: StrictInt = Field(ge=0, le=50_000)
    best_validation_nll: StrictFloat = Field(ge=0.0, allow_inf_nan=False)
    best_selection_score: StrictFloat = Field(ge=0.0, allow_inf_nan=False)
    final_training_nll: StrictFloat = Field(ge=0.0, allow_inf_nan=False)
    validation_curve: tuple[TrainingEvaluationPoint, ...] = Field(min_length=1, max_length=50_001)
    scored_target_tokens: StrictInt = Field(ge=1, le=MAX_SAFE_INTEGER)
    elapsed_seconds: StrictFloat = Field(ge=0.0, allow_inf_nan=False)
    process_peak_rss_bytes: StrictInt = Field(ge=1)
    mps_peak_current_allocated_bytes: StrictInt = Field(ge=0)
    mps_peak_driver_allocated_bytes: StrictInt = Field(ge=0)
    optimizer_parameter_names: tuple[StrictStr, ...] = Field(min_length=1, max_length=100_000)
    files: tuple[TrainingStateFile, TrainingStateFile, TrainingStateFile]
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def state_is_self_consistent(self) -> TrainingStateManifest:
        if self.step > self.total_steps:
            raise ValueError("durable state step exceeds total training steps")
        if self.sampler_cursor != self.step * self.batch_size:
            raise ValueError("durable state sampler cursor is inconsistent")
        if len(self.optimizer_parameter_names) != len(set(self.optimizer_parameter_names)):
            raise ValueError("optimizer parameter names must be unique")
        names = tuple(item.filename for item in self.files)
        if names != (
            STATE_BEST_MODEL_FILENAME,
            STATE_MODEL_FILENAME,
            STATE_OPTIMIZER_FILENAME,
        ):
            raise ValueError("durable state tensor inventory is not canonical")
        if sum(item.size_bytes for item in self.files) > MAX_STATE_FILES_TOTAL_BYTES:
            raise ValueError("durable state tensor inventory exceeds its total byte bound")
        steps = tuple(point.step for point in self.validation_curve)
        if steps != tuple(sorted(set(steps))) or steps[0] != 0 or steps[-1] > self.step:
            raise ValueError("durable validation curve steps are not canonical")
        best = min(
            self.validation_curve,
            key=lambda point: (
                point.selection_score,
                point.validation_nll,
                point.step,
            ),
        )
        if (
            self.best_step,
            self.best_validation_nll,
            self.best_selection_score,
        ) != (best.step, best.validation_nll, best.selection_score):
            raise ValueError("durable best-state selection is inconsistent")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("durable state manifest checksum mismatch")
        return self


class CompactTrainingResult(ContractModel):
    result_version: Literal["0.3.0"] = TRAINING_CONTRACT_VERSION
    status: Literal["completed"] = "completed"
    candidate_id: CandidateId
    sampling_strategy: SamplingStrategy
    source_commit: StrictStr = Field(pattern=r"^[0-9a-f]{7,64}$")
    device: DeviceResolution
    parameter_count: StrictInt = Field(ge=1)
    vocab_size: StrictInt = Field(ge=8, le=65_536)
    train_example_count: StrictInt = Field(ge=1, le=MAX_EXAMPLES)
    validation_example_count: StrictInt = Field(ge=1, le=MAX_EXAMPLES)
    training_steps: StrictInt = Field(ge=1, le=50_000)
    selected_step: StrictInt = Field(ge=0, le=50_000)
    initial_validation_nll: StrictFloat = Field(ge=0.0, allow_inf_nan=False)
    selected_validation_nll: StrictFloat = Field(ge=0.0, allow_inf_nan=False)
    selected_score: StrictFloat = Field(ge=0.0, allow_inf_nan=False)
    final_training_nll: StrictFloat = Field(ge=0.0, allow_inf_nan=False)
    validation_curve: tuple[TrainingEvaluationPoint, ...] = Field(min_length=1, max_length=50_001)
    training_config_sha256: Sha256
    model_config_sha256: Sha256
    train_inventory_sha256: Sha256
    validation_inventory_sha256: Sha256
    train_tokenized_sha256: Sha256
    validation_tokenized_sha256: Sha256
    tokenizer_manifest_sha256: Sha256
    elapsed_seconds: StrictFloat = Field(gt=0.0, allow_inf_nan=False)
    scored_target_tokens: StrictInt = Field(ge=1, le=MAX_SAFE_INTEGER)
    target_tokens_per_second: StrictFloat = Field(gt=0.0, allow_inf_nan=False)
    process_peak_rss_bytes: StrictInt = Field(ge=1)
    mps_peak_current_allocated_bytes: StrictInt = Field(ge=0)
    mps_peak_driver_allocated_bytes: StrictInt = Field(ge=0)
    durable_state_count: StrictInt = Field(
        ge=1,
        le=MAX_RETAINED_DURABLE_STATES,
        description="Committed durable training states retained when fitting completed.",
    )
    checkpoint_manifest_sha256: Sha256
    checkpoint_weights_sha256: Sha256
    checkpoint_size_bytes: StrictInt = Field(ge=1)
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def result_is_checksum_bound_and_optimal(self) -> CompactTrainingResult:
        if self.selected_step > self.training_steps:
            raise ValueError("selected step exceeds completed training")
        steps = tuple(point.step for point in self.validation_curve)
        if steps != tuple(sorted(set(steps))) or steps[0] != 0 or steps[-1] != self.training_steps:
            raise ValueError("result validation curve does not cover the training boundary")
        best = min(
            self.validation_curve,
            key=lambda point: (
                point.selection_score,
                point.validation_nll,
                point.step,
            ),
        )
        if (self.selected_step, self.selected_validation_nll, self.selected_score) != (
            best.step,
            best.validation_nll,
            best.selection_score,
        ):
            raise ValueError(
                "result did not select lower score, lower validation NLL, then earlier step"
            )
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("compact training result checksum mismatch")
        return self


class CompactTrainingStopped(ContractModel):
    result_version: Literal["0.3.0"] = TRAINING_CONTRACT_VERSION
    status: Literal["stopped_resumable"] = "stopped_resumable"
    candidate_id: CandidateId
    sampling_strategy: SamplingStrategy
    source_commit: StrictStr = Field(pattern=r"^[0-9a-f]{7,64}$")
    device: DeviceResolution
    completed_steps: StrictInt = Field(ge=1, le=50_000)
    total_steps: StrictInt = Field(ge=1, le=50_000)
    batch_size: StrictInt = Field(ge=1, le=MAX_BATCH_SIZE)
    sampler_cursor: StrictInt = Field(ge=1, le=MAX_SAFE_INTEGER)
    durable_state_name: StrictStr = Field(pattern=r"^state-step-[0-9]{8}$")
    durable_state_manifest_sha256: Sha256
    training_config_sha256: Sha256
    model_config_sha256: Sha256
    train_inventory_sha256: Sha256
    validation_inventory_sha256: Sha256
    train_tokenized_sha256: Sha256
    validation_tokenized_sha256: Sha256
    tokenizer_manifest_sha256: Sha256
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def stopped_state_is_consistent(self) -> CompactTrainingStopped:
        if self.completed_steps >= self.total_steps:
            raise ValueError("a completed training run cannot be marked resumably stopped")
        if self.sampler_cursor != self.completed_steps * self.batch_size:
            raise ValueError("stopped sampler cursor is inconsistent")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("stopped training result checksum mismatch")
        return self


type CompactTrainingOutcome = CompactTrainingResult | CompactTrainingStopped


@dataclass(frozen=True, slots=True)
class _LoadedTrainingState:
    manifest: TrainingStateManifest
    model: TransformerLM
    optimizer: torch.optim.AdamW
    best_model_state: dict[str, Tensor]
    cpu_rng_state: Tensor
    mps_rng_state: Tensor | None


@dataclass(frozen=True, slots=True)
class _CommittedTrainingState:
    directory: Path
    manifest: TrainingStateManifest


def _optimizer_parameter_tensor_count(config: TransformerConfig) -> int:
    tensors_per_affine = 2 if config.bias else 1
    embeddings = 2
    block_tensors = config.layers * 6 * tensors_per_affine
    final_norm = tensors_per_affine
    untied_output = 0 if config.tie_embeddings else 1
    return embeddings + block_tensors + final_norm + untied_output


def durable_training_state_upper_bound_bytes(
    model_config: TransformerConfig,
    *,
    vocab_size: int,
) -> int:
    """Return the enforced float32 byte ceiling for one committed training state.

    The ceiling includes the current and best model snapshots, both AdamW moment
    tensors, bounded scalar/RNG/header overhead, and the canonical JSON manifest.
    Tied output embeddings are counted twice because safetensors receives cloned
    state-dict entries even though the architecture counts the shared parameter once.
    """

    if type(model_config) is not TransformerConfig:
        raise TypeError("storage budgeting requires an exact TransformerConfig")
    if type(vocab_size) is not int or not 8 <= vocab_size <= 65_536:
        raise ValueError("vocab_size must be an integer in [8, 65536]")
    parameter_count = exact_parameter_count(model_config, vocab_size=vocab_size)
    duplicated_tied_elements = vocab_size * model_config.width if model_config.tie_embeddings else 0
    parameter_tensors = _optimizer_parameter_tensor_count(model_config)
    tensor_payload = 16 * parameter_count + 8 * duplicated_tied_elements + 8 * parameter_tensors
    bounded_overhead = (
        3 * MAX_SAFETENSORS_HEADER_BYTES + 2 * MAX_SERIALIZED_RNG_BYTES + MAX_STATE_MANIFEST_BYTES
    )
    return tensor_payload + bounded_overhead


def selected_checkpoint_upper_bound_bytes(
    model_config: TransformerConfig,
    *,
    vocab_size: int,
) -> int:
    """Return the float32 byte ceiling for the separately published checkpoint."""

    if type(model_config) is not TransformerConfig:
        raise TypeError("storage budgeting requires an exact TransformerConfig")
    if type(vocab_size) is not int or not 8 <= vocab_size <= 65_536:
        raise ValueError("vocab_size must be an integer in [8, 65536]")
    parameter_count = exact_parameter_count(model_config, vocab_size=vocab_size)
    duplicated_tied_elements = vocab_size * model_config.width if model_config.tie_embeddings else 0
    return (
        4 * (parameter_count + duplicated_tied_elements)
        + MAX_SAFETENSORS_HEADER_BYTES
        + MAX_STATE_MANIFEST_BYTES
    )


def resolve_training_device(training: RemediationTraining) -> DeviceResolution:
    """Resolve the configured device with an explicit, reportable CPU fallback."""

    if type(training) is not RemediationTraining:
        raise TypeError("training must be an exact RemediationTraining contract")
    if training.device == "mps" and torch.backends.mps.is_available():
        return DeviceResolution(requested="mps", resolved="mps", fallback_used=False)
    if training.device == "mps" and not training.allow_cpu_fallback:
        raise TrainingError("configured MPS is unavailable and CPU fallback is disabled")
    return DeviceResolution(
        requested=training.device,
        resolved="cpu",
        fallback_used=training.device == "mps",
    )


def _bounded_integer(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an exact integer in [{minimum}, {maximum}]")
    return value


def _validated_sampling_strategy(value: object) -> SamplingStrategy:
    if type(value) is not str or value not in {"uniform_control", "task_balanced"}:
        raise ValueError("sampling_strategy is not supported")
    return cast(SamplingStrategy, value)


def _rank(seed: int, epoch: int, example_id: str, index: int) -> tuple[int, str, int]:
    digest = hashlib.sha256()
    for value in ("uniform-control", str(seed), str(epoch), example_id, str(index)):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big"), example_id, index


def uniform_control_batch_indices(
    records: tuple[CompactTokenizedExample, ...],
    *,
    batch_size: int,
    seed: int,
    cursor: int,
) -> tuple[int, ...]:
    """Select a deterministic uniform stream without NumPy or process RNG state."""

    _validate_tokenized_inventory(records, name="uniform control")
    batch_size = _bounded_integer(batch_size, name="batch_size", minimum=1, maximum=MAX_BATCH_SIZE)
    seed = _bounded_integer(seed, name="seed", minimum=0, maximum=4_294_967_295)
    cursor = _bounded_integer(cursor, name="cursor", minimum=0, maximum=MAX_SAFE_INTEGER)
    count = len(records)
    permutations: dict[int, tuple[int, ...]] = {}
    selected: list[int] = []
    for position in range(cursor, cursor + batch_size):
        epoch, offset = divmod(position, count)
        if epoch not in permutations:
            permutations[epoch] = tuple(
                sorted(
                    range(count),
                    key=lambda index: _rank(seed, epoch, records[index].example_id, index),
                )
            )
        selected.append(permutations[epoch][offset])
    return tuple(selected)


def _validate_tokenized_inventory(
    records: tuple[CompactTokenizedExample, ...], *, name: str
) -> None:
    if type(records) is not tuple or not records:
        raise ValueError(f"{name} records must be a non-empty exact tuple")
    if len(records) > MAX_EXAMPLES:
        raise ValueError(f"{name} record inventory exceeds its bound")
    if any(type(record) is not CompactTokenizedExample for record in records):
        raise TypeError(f"{name} records must be exact CompactTokenizedExample objects")
    identifiers = tuple(record.example_id for record in records)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{name} example IDs must be unique")
    for record in records:
        if (
            not record.token_ids
            or len(record.token_ids) != len(record.target_mask)
            or record.target_token_count < 1
            or record.prompt_tokens_retained < 1
            or record.prompt_token_count < record.prompt_tokens_retained
            or len(record.token_ids) != record.prompt_tokens_retained + record.target_token_count
            or record.target_mask
            != (*([False] * record.prompt_tokens_retained), *([True] * record.target_token_count))
            or record.prompt_truncated
            != (record.prompt_tokens_retained < record.prompt_token_count)
        ):
            raise ValueError(f"{name} record has an invalid supervised token boundary")


def tokenized_inventory_sha256(records: tuple[CompactTokenizedExample, ...]) -> str:
    """Hash the one canonical tokenized-example representation used by every consumer."""

    _validate_tokenized_inventory(records, name="tokenized checksum")
    return canonical_sha256(
        tuple(
            (
                record.example_id,
                record.task_name,
                record.group_id,
                record.token_ids,
                record.target_mask,
                record.prompt_token_count,
                record.target_token_count,
                record.prompt_tokens_retained,
                record.prompt_truncated,
            )
            for record in records
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("training state contains a duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"training state contains a non-finite constant: {value}")

    decoded = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )
    if type(decoded) is not dict:
        raise ValueError("training state manifest must contain one JSON object")
    return decoded


def _model_state_cpu(model: TransformerLM) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }


def _parameter_inventory(model: TransformerLM) -> tuple[tuple[str, Tensor], ...]:
    return tuple((name, parameter) for name, parameter in model.named_parameters())


def _optimizer_state_tensors(
    model: TransformerLM,
    optimizer: torch.optim.AdamW,
    *,
    include_mps_rng: bool,
) -> dict[str, Tensor]:
    parameters = _parameter_inventory(model)
    group_parameters = tuple(
        cast(Tensor, parameter)
        for group in optimizer.param_groups
        for parameter in cast(list[object], group["params"])
    )
    if tuple(parameter for _name, parameter in parameters) != group_parameters:
        raise TrainingError("optimizer parameter order differs from the model inventory")
    tensors: dict[str, Tensor] = {"rng.cpu": torch.get_rng_state().contiguous().clone()}
    if include_mps_rng:
        tensors["rng.mps"] = torch.mps.get_rng_state().cpu().contiguous().clone()
    for index, (_name, parameter) in enumerate(parameters):
        state = optimizer.state.get(parameter)
        if state is None or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise TrainingError("AdamW state inventory is incomplete or unexpected")
        for state_name in ("step", "exp_avg", "exp_avg_sq"):
            tensor = state[state_name]
            if type(tensor) is not Tensor:
                raise TrainingError("AdamW state contains a non-tensor value")
            tensors[f"optimizer.{index:06d}.{state_name}"] = (
                tensor.detach().cpu().contiguous().clone()
            )
    return tensors


def _write_safetensors(path: Path, tensors: dict[str, Tensor], *, metadata: str) -> None:
    save_file(tensors, str(path), metadata={"format": metadata})
    if not 0 < path.stat().st_size <= MAX_STATE_TENSOR_FILE_BYTES:
        raise TrainingError("training-state safetensors exceeds its byte bound")
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _manifest_with_checksum(draft: TrainingStateManifest) -> TrainingStateManifest:
    values = draft.model_dump(mode="python", round_trip=True, exclude={"checksum_sha256"})
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return TrainingStateManifest(**values, checksum_sha256=checksum)


def _save_training_state(
    *,
    state_root: Path,
    candidate_id: str,
    sampling_strategy: SamplingStrategy,
    source_commit: str,
    device_resolution: DeviceResolution,
    step: int,
    training: RemediationTraining,
    vocab_size: int,
    training_config_sha256: str,
    model_config_sha256: str,
    train_inventory_sha256: str,
    validation_inventory_sha256: str,
    train_tokenized_sha256: str,
    validation_tokenized_sha256: str,
    tokenizer_manifest_sha256: str,
    initial_validation_nll: float,
    best_step: int,
    best_validation_nll: float,
    best_selection_score: float,
    final_training_nll: float,
    validation_curve: tuple[TrainingEvaluationPoint, ...],
    scored_target_tokens: int,
    elapsed_seconds: float,
    process_peak_rss_bytes: int,
    peak_current: int,
    peak_driver: int,
    model: TransformerLM,
    optimizer: torch.optim.AdamW,
    best_model_state: dict[str, Tensor],
    include_mps_rng: bool,
) -> tuple[Path, TrainingStateManifest]:
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()) or any(
        tensor.dtype != torch.float32 for tensor in best_model_state.values()
    ):
        raise TrainingError("durable training storage budgeting requires float32 tensors")
    name = f"state-step-{step:08d}"
    target = state_root / name
    lock = state_root / f".{name}.lock"
    if target.exists() or target.is_symlink() or lock.exists() or lock.is_symlink():
        raise FileExistsError("durable training state already exists")
    lock_descriptor: int | None = None
    temporary: Path | None = None
    try:
        lock_descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(lock_descriptor, b"reactorbench-training-state-lock\n")
        os.fsync(lock_descriptor)
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=state_root))
        model_path = temporary / STATE_MODEL_FILENAME
        best_path = temporary / STATE_BEST_MODEL_FILENAME
        optimizer_path = temporary / STATE_OPTIMIZER_FILENAME
        _write_safetensors(
            model_path,
            _model_state_cpu(model),
            metadata="reactorbench-remediation-model-v0.3.0",
        )
        _write_safetensors(
            best_path,
            {key: tensor.cpu().contiguous().clone() for key, tensor in best_model_state.items()},
            metadata="reactorbench-remediation-best-model-v0.3.0",
        )
        _write_safetensors(
            optimizer_path,
            _optimizer_state_tensors(model, optimizer, include_mps_rng=include_mps_rng),
            metadata="reactorbench-remediation-adamw-v0.3.0",
        )
        files = tuple(
            TrainingStateFile(
                filename=cast(Any, path.name),
                sha256=_sha256(path),
                size_bytes=path.stat().st_size,
            )
            for path in (best_path, model_path, optimizer_path)
        )
        draft = TrainingStateManifest.model_construct(
            candidate_id=candidate_id,
            sampling_strategy=sampling_strategy,
            source_commit=source_commit,
            device=device_resolution,
            step=step,
            total_steps=training.steps,
            sampler_cursor=step * training.batch_size,
            batch_size=training.batch_size,
            vocab_size=vocab_size,
            training_config_sha256=training_config_sha256,
            model_config_sha256=model_config_sha256,
            train_inventory_sha256=train_inventory_sha256,
            validation_inventory_sha256=validation_inventory_sha256,
            train_tokenized_sha256=train_tokenized_sha256,
            validation_tokenized_sha256=validation_tokenized_sha256,
            tokenizer_manifest_sha256=tokenizer_manifest_sha256,
            initial_validation_nll=initial_validation_nll,
            best_step=best_step,
            best_validation_nll=best_validation_nll,
            best_selection_score=best_selection_score,
            final_training_nll=final_training_nll,
            validation_curve=validation_curve,
            scored_target_tokens=scored_target_tokens,
            elapsed_seconds=elapsed_seconds,
            process_peak_rss_bytes=process_peak_rss_bytes,
            mps_peak_current_allocated_bytes=peak_current,
            mps_peak_driver_allocated_bytes=peak_driver,
            optimizer_parameter_names=tuple(
                name for name, _parameter in _parameter_inventory(model)
            ),
            files=files,
            checksum_sha256="0" * 64,
        )
        manifest = _manifest_with_checksum(draft)
        payload = canonical_json_bytes(manifest.model_dump(mode="json", round_trip=True)) + b"\n"
        if len(payload) > MAX_STATE_MANIFEST_BYTES:
            raise TrainingError("durable training-state manifest exceeds its byte bound")
        actual_state_bytes = len(payload) + sum(item.size_bytes for item in manifest.files)
        storage_bound = durable_training_state_upper_bound_bytes(
            model.config,
            vocab_size=vocab_size,
        )
        if actual_state_bytes > storage_bound:
            raise TrainingError("durable training state exceeds its deterministic byte budget")
        manifest_path = temporary / STATE_MANIFEST_FILENAME
        with manifest_path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory_descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        if target.exists() or target.is_symlink():
            raise FileExistsError("durable training state appeared during publication")
        os.rename(temporary, target)
        temporary = None
        parent_descriptor = os.open(state_root, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return target, manifest
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                pass
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def _read_state_manifest(directory: Path) -> TrainingStateManifest:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("durable training state must be a regular non-symlink directory")
    inventory = {path.name for path in directory.iterdir()}
    if inventory != set(STATE_FILE_INVENTORY):
        raise ValueError("durable training state contains an unexpected file inventory")
    paths = tuple(directory / name for name in STATE_FILE_INVENTORY)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("durable training state contains a symlink or non-file")
    manifest_path = directory / STATE_MANIFEST_FILENAME
    if not 0 < manifest_path.stat().st_size <= MAX_STATE_MANIFEST_BYTES:
        raise ValueError("durable training-state manifest exceeds its byte bound")
    payload = manifest_path.read_bytes()
    _strict_json_object(payload)
    manifest = TrainingStateManifest.model_validate_json(payload, strict=True)
    canonical = canonical_json_bytes(manifest.model_dump(mode="json", round_trip=True)) + b"\n"
    if payload != canonical:
        raise ValueError("durable training-state manifest is not canonical JSON")
    for file in manifest.files:
        path = directory / file.filename
        if path.stat().st_size != file.size_bytes or _sha256(path) != file.sha256:
            raise ValueError("durable training-state tensor checksum or size mismatch")
    return manifest


def _committed_training_states(
    state_root: Path,
    *,
    candidate_id: str,
) -> tuple[_CommittedTrainingState, ...]:
    """Verify every committed state while leaving known in-progress entries untouched."""

    if state_root.is_symlink() or not state_root.is_dir():
        raise ValueError("durable state root must be a regular non-symlink directory")
    resolved_root = state_root.resolve(strict=True)
    states: list[_CommittedTrainingState] = []
    for entry in sorted(resolved_root.iterdir(), key=lambda path: path.name):
        state_match = _STATE_DIRECTORY_PATTERN.fullmatch(entry.name)
        if state_match is not None:
            if entry.is_symlink() or not entry.is_dir():
                raise ValueError("committed durable state entry is a symlink or non-directory")
            resolved_entry = entry.resolve(strict=True)
            if resolved_entry.parent != resolved_root:
                raise ValueError("committed durable state escapes its bounded candidate root")
            manifest = _read_state_manifest(resolved_entry)
            recorded_step = int(state_match.group(1))
            if manifest.step != recorded_step or manifest.candidate_id != candidate_id:
                raise ValueError("committed durable state identity is inconsistent")
            states.append(_CommittedTrainingState(directory=resolved_entry, manifest=manifest))
            continue

        if _STATE_LOCK_PATTERN.fullmatch(entry.name) is not None:
            if entry.is_symlink() or not entry.is_file():
                raise ValueError("durable state lock is a symlink or non-file")
            continue
        if _STATE_TEMPORARY_PATTERN.fullmatch(entry.name) is not None:
            if entry.is_symlink() or not entry.is_dir():
                raise ValueError("durable temporary state is a symlink or non-directory")
            continue
        raise ValueError("durable state root contains an unexpected entry")

    return tuple(
        sorted(
            states,
            key=lambda state: (state.manifest.step, state.directory.name),
        )
    )


def _retain_newest_training_states(
    *,
    state_root: Path,
    candidate_id: str,
    current_directory: Path,
    current_manifest: TrainingStateManifest,
) -> tuple[_CommittedTrainingState, ...]:
    """Prune only verified obsolete states after the new state is durably committed."""

    resolved_root = state_root.resolve(strict=True)
    if current_directory.is_symlink():
        raise TrainingError("current durable state is unsafe for retention")
    resolved_current = current_directory.resolve(strict=True)
    if resolved_current.parent != resolved_root:
        raise TrainingError("current durable state escapes its bounded candidate root")

    states = _committed_training_states(resolved_root, candidate_id=candidate_id)
    current = next(
        (state for state in states if state.directory == resolved_current),
        None,
    )
    if current is None or current.manifest != current_manifest:
        raise TrainingError("current durable state failed retention verification")
    if states[-1].directory != resolved_current:
        raise TrainingError("newly committed durable state is not the newest valid state")

    obsolete = states[:-MAX_RETAINED_DURABLE_STATES]
    for state in obsolete:
        if state.directory == resolved_current:
            raise TrainingError("retention attempted to remove the current durable state")
        if (
            state.directory.is_symlink()
            or state.directory.resolve(strict=True).parent != resolved_root
        ):
            raise TrainingError("obsolete durable state failed path-containment verification")
        observed = _read_state_manifest(state.directory)
        if observed != state.manifest or observed.candidate_id != candidate_id:
            raise TrainingError("obsolete durable state changed before retention")
        if not shutil.rmtree.avoids_symlink_attacks:
            raise TrainingError("platform cannot safely retire a durable state")
        parent_descriptor = os.open(resolved_root, os.O_RDONLY)
        try:
            shutil.rmtree(state.directory.name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
        except OSError as error:
            raise TrainingError("verified obsolete durable state could not be retired") from error
        finally:
            os.close(parent_descriptor)

    retained = _committed_training_states(resolved_root, candidate_id=candidate_id)
    if (
        not 1 <= len(retained) <= MAX_RETAINED_DURABLE_STATES
        or retained[-1].directory != resolved_current
    ):
        raise TrainingError("durable-state retention did not reach its bounded safe state")
    return retained


_RESUME_LINEAGE_FIELDS = (
    "candidate_id",
    "sampling_strategy",
    "source_commit",
    "device",
    "total_steps",
    "batch_size",
    "vocab_size",
    "training_config_sha256",
    "model_config_sha256",
    "train_inventory_sha256",
    "validation_inventory_sha256",
    "train_tokenized_sha256",
    "validation_tokenized_sha256",
    "tokenizer_manifest_sha256",
    "optimizer_parameter_names",
)


def latest_committed_training_state(
    state_root: Path,
    *,
    candidate_id: str,
) -> Path | None:
    """Return the newest verified state while safely ignoring known crash remnants."""

    states = _committed_training_states(state_root, candidate_id=candidate_id)
    return None if not states else states[-1].directory


def retire_superseded_training_states(
    source_root: Path,
    *,
    candidate_id: str,
    successor_directory: Path,
) -> int:
    """Retire verified committed states only after a bound successor is durable.

    Known lock and temporary crash remnants remain untouched for forensic inspection.
    The successor may be in a later non-overwriting attempt, but it must carry the
    same frozen training lineage and be at least as advanced as every retired state.
    """

    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("retired training-state root must be a regular directory")
    if (
        successor_directory.is_symlink()
        or successor_directory.parent.is_symlink()
        or not successor_directory.is_dir()
    ):
        raise ValueError("successor training state must be a regular directory")
    resolved_source = source_root.resolve(strict=True)
    resolved_successor = successor_directory.resolve(strict=True)
    if resolved_successor.parent == resolved_source:
        raise ValueError("successor training state must be outside the retired root")
    successor = _read_state_manifest(resolved_successor)
    if successor.candidate_id != candidate_id:
        raise ValueError("successor training state belongs to another candidate")

    states = _committed_training_states(resolved_source, candidate_id=candidate_id)
    if not states:
        return 0
    for state in states:
        if state.manifest.step > successor.step or any(
            getattr(state.manifest, field) != getattr(successor, field)
            for field in _RESUME_LINEAGE_FIELDS
        ):
            raise TrainingError("successor training state does not supersede its source")
        if (
            state.manifest.step == successor.step
            and state.manifest.checksum_sha256 != successor.checksum_sha256
        ):
            raise TrainingError("equal-step successor training state differs from its source")

    retired = 0
    for state in states:
        if state.directory.is_symlink() or state.directory.resolve(strict=True).parent != (
            resolved_source
        ):
            raise TrainingError("superseded durable state failed containment verification")
        if _read_state_manifest(state.directory) != state.manifest:
            raise TrainingError("superseded durable state changed before retirement")
        if not shutil.rmtree.avoids_symlink_attacks:
            raise TrainingError("platform cannot safely retire a durable state")
        parent_descriptor = os.open(resolved_source, os.O_RDONLY)
        try:
            shutil.rmtree(state.directory.name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
        except OSError as error:
            raise TrainingError("verified superseded durable state could not be retired") from error
        finally:
            os.close(parent_descriptor)
        retired += 1
    if _committed_training_states(resolved_source, candidate_id=candidate_id):
        raise TrainingError("superseded durable-state retirement remained incomplete")
    return retired


def _restore_model_state(model: TransformerLM, path: Path) -> dict[str, Tensor]:
    try:
        state = load_file(str(path), device="cpu")
        expected = set(model.state_dict())
        if set(state) != expected:
            raise ValueError("training-state model tensor inventory differs from the architecture")
        model.load_state_dict(state, strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("training-state model safetensors could not be loaded") from error
    return {name: tensor.cpu().contiguous().clone() for name, tensor in state.items()}


def _restore_optimizer_state(
    model: TransformerLM,
    optimizer: torch.optim.AdamW,
    path: Path,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor | None]:
    try:
        tensors = load_file(str(path), device="cpu")
    except (OSError, RuntimeError) as error:
        raise ValueError("training-state optimizer safetensors could not be loaded") from error
    parameters = _parameter_inventory(model)
    expected = {"rng.cpu"}
    if device.type == "mps":
        expected.add("rng.mps")
    for index in range(len(parameters)):
        expected.update(
            {
                f"optimizer.{index:06d}.step",
                f"optimizer.{index:06d}.exp_avg",
                f"optimizer.{index:06d}.exp_avg_sq",
            }
        )
    if set(tensors) != expected:
        raise ValueError("training-state optimizer tensor inventory is unexpected")
    for index, (_name, parameter) in enumerate(parameters):
        step = tensors[f"optimizer.{index:06d}.step"]
        exp_avg = tensors[f"optimizer.{index:06d}.exp_avg"]
        exp_avg_sq = tensors[f"optimizer.{index:06d}.exp_avg_sq"]
        if (
            step.numel() != 1
            or exp_avg.shape != parameter.shape
            or exp_avg_sq.shape != parameter.shape
        ):
            raise ValueError("training-state AdamW tensor shape mismatch")
        if exp_avg.dtype != parameter.dtype or exp_avg_sq.dtype != parameter.dtype:
            raise ValueError("training-state AdamW tensor dtype mismatch")
        optimizer.state[parameter] = {
            "step": step.cpu().clone(),
            "exp_avg": exp_avg.to(device=device).clone(),
            "exp_avg_sq": exp_avg_sq.to(device=device).clone(),
        }
    cpu_rng = tensors["rng.cpu"]
    if cpu_rng.dtype != torch.uint8 or cpu_rng.ndim != 1:
        raise ValueError("training-state CPU RNG tensor is invalid")
    mps_rng = tensors.get("rng.mps")
    if mps_rng is not None and (mps_rng.dtype != torch.uint8 or mps_rng.ndim != 1):
        raise ValueError("training-state MPS RNG tensor is invalid")
    return cpu_rng.clone(), None if mps_rng is None else mps_rng.clone()


def _load_training_state(
    *,
    directory: Path,
    expected_root: Path,
    candidate_id: str,
    sampling_strategy: SamplingStrategy,
    source_commit: str,
    device_resolution: DeviceResolution,
    training: RemediationTraining,
    model_config: TransformerConfig,
    vocab_size: int,
    training_config_sha256: str,
    model_config_sha256: str,
    train_inventory_sha256: str,
    validation_inventory_sha256: str,
    train_tokenized_sha256: str,
    validation_tokenized_sha256: str,
    tokenizer_manifest_sha256: str,
    device: torch.device,
) -> _LoadedTrainingState:
    if directory.is_symlink() or directory.resolve(strict=True).parent != expected_root:
        raise ValueError("resume state must be one direct non-symlink child of the state root")
    manifest = _read_state_manifest(directory)
    if directory.name != f"state-step-{manifest.step:08d}":
        raise ValueError("resume state directory name does not match its recorded step")
    expected_bindings = (
        (manifest.candidate_id, candidate_id),
        (manifest.sampling_strategy, sampling_strategy),
        (manifest.source_commit, source_commit),
        (manifest.device, device_resolution),
        (manifest.total_steps, training.steps),
        (manifest.batch_size, training.batch_size),
        (manifest.vocab_size, vocab_size),
        (manifest.training_config_sha256, training_config_sha256),
        (manifest.model_config_sha256, model_config_sha256),
        (manifest.train_inventory_sha256, train_inventory_sha256),
        (manifest.validation_inventory_sha256, validation_inventory_sha256),
        (manifest.train_tokenized_sha256, train_tokenized_sha256),
        (manifest.validation_tokenized_sha256, validation_tokenized_sha256),
        (manifest.tokenizer_manifest_sha256, tokenizer_manifest_sha256),
    )
    if any(observed != expected for observed, expected in expected_bindings):
        raise ValueError("resume state does not match the frozen training inputs")
    model = initialized_model(model_config, vocab_size=vocab_size, seed=training.seed).to(device)
    _restore_model_state(model, directory / STATE_MODEL_FILENAME)
    best_state = _restore_model_state(
        initialized_model(model_config, vocab_size=vocab_size, seed=training.seed),
        directory / STATE_BEST_MODEL_FILENAME,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    cpu_rng, mps_rng = _restore_optimizer_state(
        model,
        optimizer,
        directory / STATE_OPTIMIZER_FILENAME,
        device=device,
    )
    names = tuple(name for name, _parameter in _parameter_inventory(model))
    if names != manifest.optimizer_parameter_names:
        raise ValueError("resume optimizer parameter inventory differs from the manifest")
    return _LoadedTrainingState(
        manifest=manifest,
        model=model,
        optimizer=optimizer,
        best_model_state=best_state,
        cpu_rng_state=cpu_rng,
        mps_rng_state=mps_rng,
    )


def _device(resolution: DeviceResolution) -> torch.device:
    return torch.device(resolution.resolved)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def compact_validation_nll(
    model: TransformerLM,
    records: tuple[CompactTokenizedExample, ...],
    *,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[float, int]:
    """Return finite target-token-weighted NLL for verified compact examples."""

    if type(model) is not TransformerLM:
        raise TypeError("model must be an exact TransformerLM")
    _validate_tokenized_inventory(records, name="validation")
    batch_size = _bounded_integer(batch_size, name="batch_size", minimum=1, maximum=MAX_BATCH_SIZE)
    was_training = model.training
    weighted_loss = 0.0
    target_tokens = 0
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                input_ids, attention_mask, target_mask = compact_batch_tensors(
                    batch, context_length=context_length
                )
                tokens = int(target_mask[:, 1:].sum().item())
                loss = supervised_causal_loss(
                    model,
                    input_ids.to(device),
                    attention_mask.to(device),
                    target_mask.to(device),
                )
                value = float(loss.item())
                if not math.isfinite(value) or value < 0.0:
                    raise TrainingError("validation produced a non-finite NLL")
                weighted_loss += value * tokens
                target_tokens += tokens
        _synchronize(device)
    finally:
        model.train(was_training)
    if target_tokens < 1:
        raise TrainingError("validation produced no scorable target tokens")
    result = weighted_loss / target_tokens
    if not math.isfinite(result) or result < 0.0:
        raise TrainingError("validation produced a non-finite aggregate NLL")
    return result, target_tokens


def _preserve_rng_call[ResultT](callback: Callable[[], ResultT], *, include_mps: bool) -> ResultT:
    cpu_state = torch.get_rng_state().clone()
    mps_state = torch.mps.get_rng_state().clone() if include_mps else None
    try:
        return callback()
    finally:
        torch.set_rng_state(cpu_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def _evaluation_score(
    callback: EvaluationCallback | None,
    *,
    model: TransformerLM,
    step: int,
    validation_nll: float,
    include_mps_rng: bool,
) -> float:
    if callback is None:
        return validation_nll
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            value = _preserve_rng_call(
                lambda: callback(model, step, validation_nll),
                include_mps=include_mps_rng,
            )
    finally:
        model.train(was_training)
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise TrainingError("evaluation callback must return a finite non-negative float")
    return value


def _notify(
    callback: ProgressCallback | None,
    event: TrainingProgress,
    *,
    include_mps_rng: bool,
) -> None:
    if callback is not None:
        _preserve_rng_call(lambda: callback(event), include_mps=include_mps_rng)


def _stop_requested(callback: StopRequested | None, *, step: int, include_mps_rng: bool) -> bool:
    if callback is None:
        return False
    value = _preserve_rng_call(lambda: callback(step), include_mps=include_mps_rng)
    if type(value) is not bool:
        raise TrainingError("stop_requested callback must return an exact boolean")
    return value


def _result_with_checksum(draft: CompactTrainingResult) -> CompactTrainingResult:
    values = draft.model_dump(mode="python", round_trip=True, exclude={"checksum_sha256"})
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return CompactTrainingResult(**values, checksum_sha256=checksum)


def _stopped_with_checksum(draft: CompactTrainingStopped) -> CompactTrainingStopped:
    values = draft.model_dump(mode="python", round_trip=True, exclude={"checksum_sha256"})
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return CompactTrainingStopped(**values, checksum_sha256=checksum)


def _validated_hash(value: str, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _elapsed(clock: MonotonicClock, *, started: float, offset: float) -> float:
    current = clock()
    if type(current) is not float or not math.isfinite(current) or current < started:
        raise TrainingError("monotonic clock returned an invalid timestamp")
    return offset + (current - started)


def train_compact_model(
    *,
    candidate_id: str,
    sampling_strategy: SamplingStrategy,
    model_config: TransformerConfig,
    training: RemediationTraining,
    vocab_size: int,
    tokenizer_manifest: TokenizerArtifactManifest,
    train_examples: tuple[CompactTokenizedExample, ...],
    validation_examples: tuple[CompactTokenizedExample, ...],
    train_inventory_sha256: str,
    validation_inventory_sha256: str,
    output_directory: Path,
    durable_state_root: Path,
    source_commit: str,
    resume_state_directory: Path | None = None,
    evaluation_callback: EvaluationCallback | None = None,
    progress_callback: ProgressCallback | None = None,
    stop_requested: StopRequested | None = None,
    monotonic_clock: MonotonicClock = time.perf_counter,
) -> CompactTrainingOutcome:
    """Fit one candidate or return a checksum-bound safe resumable stop."""

    if type(model_config) is not TransformerConfig or type(training) is not RemediationTraining:
        raise TypeError("training requires exact model and remediation configuration contracts")
    if type(tokenizer_manifest) is not TokenizerArtifactManifest:
        raise TypeError("training requires an exact tokenizer manifest")
    sampling_strategy = _validated_sampling_strategy(sampling_strategy)
    if type(vocab_size) is not int or not 8 <= vocab_size <= 65_536:
        raise ValueError("vocab_size must be an integer in [8, 65536]")
    if vocab_size != tokenizer_manifest.actual_vocab_size:
        raise ValueError("vocab_size differs from the frozen tokenizer manifest")
    if (
        type(candidate_id) is not str
        or not 1 <= len(candidate_id) <= 96
        or not candidate_id[0].isalnum()
        or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-"
            for character in candidate_id
        )
    ):
        raise ValueError("candidate_id violates its strict identifier contract")
    if (
        type(source_commit) is not str
        or not 7 <= len(source_commit) <= 64
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("source_commit must be a lowercase hexadecimal Git revision")
    if not isinstance(output_directory, Path) or not isinstance(durable_state_root, Path):
        raise TypeError("training output paths must be pathlib.Path objects")
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError("final checkpoint output must be a new path")
    if (
        not 1 <= len(output_directory.name) <= 128
        or not output_directory.name[0].isalnum()
        or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in output_directory.name
        )
    ):
        raise ValueError("final checkpoint directory name is unsafe")
    if durable_state_root.is_symlink() or not durable_state_root.is_dir():
        raise ValueError("durable state root must be an existing non-symlink directory")
    state_root = durable_state_root.resolve(strict=True)
    if resume_state_directory is not None and not isinstance(resume_state_directory, Path):
        raise TypeError("resume_state_directory must be a pathlib.Path or None")
    committed_states = _committed_training_states(state_root, candidate_id=candidate_id)
    if resume_state_directory is None:
        if committed_states:
            raise FileExistsError(
                "durable training state already exists; resume from the newest state"
            )
    else:
        if resume_state_directory.is_symlink():
            raise ValueError("resume state must be a direct non-symlink child of the state root")
        resolved_resume = resume_state_directory.resolve(strict=True)
        if resolved_resume.parent != state_root:
            raise ValueError("resume state must be a direct non-symlink child of the state root")
        if not committed_states or committed_states[-1].directory != resolved_resume:
            raise ValueError("resume state must be the newest surviving valid durable state")
        resume_state_directory = resolved_resume

    _validate_tokenized_inventory(train_examples, name="training")
    _validate_tokenized_inventory(validation_examples, name="validation")
    if {item.example_id for item in train_examples} & {
        item.example_id for item in validation_examples
    }:
        raise ValueError("training and validation example IDs must be disjoint")
    if any(len(item.token_ids) > model_config.context_length for item in train_examples):
        raise ValueError("training example exceeds the model context")
    if any(len(item.token_ids) > model_config.context_length for item in validation_examples):
        raise ValueError("validation example exceeds the model context")
    if any(
        token < 0 or token >= vocab_size
        for item in (*train_examples, *validation_examples)
        for token in item.token_ids
    ):
        raise ValueError("tokenized inventory contains an out-of-vocabulary ID")

    train_inventory_sha256 = _validated_hash(train_inventory_sha256, name="train_inventory_sha256")
    validation_inventory_sha256 = _validated_hash(
        validation_inventory_sha256, name="validation_inventory_sha256"
    )
    training_hash = canonical_sha256(training.model_dump(mode="json", round_trip=True))
    model_hash = canonical_sha256(model_config.model_dump(mode="json", round_trip=True))
    train_tokenized_hash = tokenized_inventory_sha256(train_examples)
    validation_tokenized_hash = tokenized_inventory_sha256(validation_examples)
    tokenizer_hash = tokenizer_manifest.checksum_sha256
    resolution = resolve_training_device(training)
    device = _device(resolution)
    include_mps_rng = device.type == "mps"
    parameter_count = exact_parameter_count(model_config, vocab_size=vocab_size)

    if not _TRAINING_RNG_LOCK.acquire(blocking=False):
        raise TrainingError("another RNG-isolated training run is already active")
    try:
        caller_cpu_rng = torch.get_rng_state().clone()
        caller_mps_rng = torch.mps.get_rng_state().clone() if include_mps_rng else None
        started = monotonic_clock()
        if type(started) is not float or not math.isfinite(started):
            raise TrainingError("monotonic clock returned an invalid starting timestamp")
    except BaseException:
        _TRAINING_RNG_LOCK.release()
        raise
    durable_state_count = len(committed_states)
    try:
        if resume_state_directory is None:
            torch.manual_seed(training.seed)
            if include_mps_rng:
                torch.mps.manual_seed(training.seed)
            model = initialized_model(model_config, vocab_size=vocab_size, seed=training.seed).to(
                device
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
            )
            initial_nll, _ = compact_validation_nll(
                model,
                validation_examples,
                batch_size=training.batch_size,
                context_length=model_config.context_length,
                device=device,
            )
            initial_score = _evaluation_score(
                evaluation_callback,
                model=model,
                step=0,
                validation_nll=initial_nll,
                include_mps_rng=include_mps_rng,
            )
            curve = [
                TrainingEvaluationPoint(
                    step=0,
                    validation_nll=initial_nll,
                    selection_score=initial_score,
                )
            ]
            best_step = 0
            best_nll = initial_nll
            best_score = initial_score
            best_state = _model_state_cpu(model)
            final_training_nll = initial_nll
            scored_target_tokens = 0
            completed_step = 0
            sampler_cursor = 0
            elapsed_offset = 0.0
            peak_current = peak_driver = 0
            _notify(
                progress_callback,
                TrainingProgress(
                    event="evaluation",
                    step=0,
                    total_steps=training.steps,
                    validation_nll=initial_nll,
                    selection_score=initial_score,
                ),
                include_mps_rng=include_mps_rng,
            )
        else:
            loaded = _load_training_state(
                directory=resume_state_directory,
                expected_root=state_root,
                candidate_id=candidate_id,
                sampling_strategy=sampling_strategy,
                source_commit=source_commit,
                device_resolution=resolution,
                training=training,
                model_config=model_config,
                vocab_size=vocab_size,
                training_config_sha256=training_hash,
                model_config_sha256=model_hash,
                train_inventory_sha256=train_inventory_sha256,
                validation_inventory_sha256=validation_inventory_sha256,
                train_tokenized_sha256=train_tokenized_hash,
                validation_tokenized_sha256=validation_tokenized_hash,
                tokenizer_manifest_sha256=tokenizer_hash,
                device=device,
            )
            manifest = loaded.manifest
            model = loaded.model
            optimizer = loaded.optimizer
            best_state = loaded.best_model_state
            torch.set_rng_state(loaded.cpu_rng_state)
            if include_mps_rng:
                if loaded.mps_rng_state is None:
                    raise ValueError("MPS resume state lacks its RNG tensor")
                torch.mps.set_rng_state(loaded.mps_rng_state)
            curve = list(manifest.validation_curve)
            initial_nll = manifest.initial_validation_nll
            best_step = manifest.best_step
            best_nll = manifest.best_validation_nll
            best_score = manifest.best_selection_score
            final_training_nll = manifest.final_training_nll
            scored_target_tokens = manifest.scored_target_tokens
            completed_step = manifest.step
            sampler_cursor = manifest.sampler_cursor
            elapsed_offset = manifest.elapsed_seconds
            peak_current = manifest.mps_peak_current_allocated_bytes
            peak_driver = manifest.mps_peak_driver_allocated_bytes

        allocated_parameters = sum(parameter.numel() for parameter in model.parameters())
        if allocated_parameters != parameter_count:
            raise TrainingError("allocated parameter count differs from the exact formula")

        model.train()
        last_state: tuple[Path, TrainingStateManifest] | None = None
        for step in range(completed_step + 1, training.steps + 1):
            if sampling_strategy == "uniform_control":
                indices = uniform_control_batch_indices(
                    train_examples,
                    batch_size=training.batch_size,
                    seed=training.seed,
                    cursor=sampler_cursor,
                )
            else:
                indices = task_balanced_batch_indices(
                    train_examples,
                    batch_size=training.batch_size,
                    seed=training.seed,
                    step=sampler_cursor // training.batch_size,
                )
            batch = tuple(train_examples[index] for index in indices)
            input_ids, attention_mask, target_mask = compact_batch_tensors(
                batch, context_length=model_config.context_length
            )
            scored_target_tokens += int(target_mask[:, 1:].sum().item())
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_causal_loss(
                model,
                input_ids.to(device),
                attention_mask.to(device),
                target_mask.to(device),
            )
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
            optimizer.step()
            final_training_nll = float(loss.detach().item())
            if not math.isfinite(final_training_nll) or final_training_nll < 0.0:
                raise TrainingError("training produced a non-finite NLL")
            sampler_cursor += training.batch_size
            completed_step = step
            if device.type == "mps":
                peak_current = max(peak_current, int(torch.mps.current_allocated_memory()))
                peak_driver = max(peak_driver, int(torch.mps.driver_allocated_memory()))

            if step % training.evaluation_interval == 0:
                validation_nll, _ = compact_validation_nll(
                    model,
                    validation_examples,
                    batch_size=training.batch_size,
                    context_length=model_config.context_length,
                    device=device,
                )
                score = _evaluation_score(
                    evaluation_callback,
                    model=model,
                    step=step,
                    validation_nll=validation_nll,
                    include_mps_rng=include_mps_rng,
                )
                point = TrainingEvaluationPoint(
                    step=step,
                    validation_nll=validation_nll,
                    selection_score=score,
                )
                curve.append(point)
                if (score, validation_nll, step) < (best_score, best_nll, best_step):
                    best_step = step
                    best_nll = validation_nll
                    best_score = score
                    best_state = _model_state_cpu(model)
                _notify(
                    progress_callback,
                    TrainingProgress(
                        event="evaluation",
                        step=step,
                        total_steps=training.steps,
                        validation_nll=validation_nll,
                        selection_score=score,
                    ),
                    include_mps_rng=include_mps_rng,
                )
                model.train()

            scheduled_checkpoint = step % training.durable_checkpoint_interval == 0
            should_stop = step < training.steps and _stop_requested(
                stop_requested,
                step=step,
                include_mps_rng=include_mps_rng,
            )
            if scheduled_checkpoint or should_stop:
                elapsed = _elapsed(
                    monotonic_clock,
                    started=started,
                    offset=elapsed_offset,
                )
                last_state = _save_training_state(
                    state_root=state_root,
                    candidate_id=candidate_id,
                    sampling_strategy=sampling_strategy,
                    source_commit=source_commit,
                    device_resolution=resolution,
                    step=step,
                    training=training,
                    vocab_size=vocab_size,
                    training_config_sha256=training_hash,
                    model_config_sha256=model_hash,
                    train_inventory_sha256=train_inventory_sha256,
                    validation_inventory_sha256=validation_inventory_sha256,
                    train_tokenized_sha256=train_tokenized_hash,
                    validation_tokenized_sha256=validation_tokenized_hash,
                    tokenizer_manifest_sha256=tokenizer_hash,
                    initial_validation_nll=initial_nll,
                    best_step=best_step,
                    best_validation_nll=best_nll,
                    best_selection_score=best_score,
                    final_training_nll=final_training_nll,
                    validation_curve=tuple(curve),
                    scored_target_tokens=scored_target_tokens,
                    elapsed_seconds=elapsed,
                    process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    peak_current=peak_current,
                    peak_driver=peak_driver,
                    model=model,
                    optimizer=optimizer,
                    best_model_state=best_state,
                    include_mps_rng=include_mps_rng,
                )
                state_path, state_manifest = last_state
                retained_states = _retain_newest_training_states(
                    state_root=state_root,
                    candidate_id=candidate_id,
                    current_directory=state_path,
                    current_manifest=state_manifest,
                )
                durable_state_count = len(retained_states)
                _notify(
                    progress_callback,
                    TrainingProgress(
                        event="durable_checkpoint",
                        step=step,
                        total_steps=training.steps,
                        checkpoint_name=state_path.name,
                    ),
                    include_mps_rng=include_mps_rng,
                )
                if should_stop:
                    _notify(
                        progress_callback,
                        TrainingProgress(
                            event="stopped",
                            step=step,
                            total_steps=training.steps,
                            checkpoint_name=state_path.name,
                        ),
                        include_mps_rng=include_mps_rng,
                    )
                    draft = CompactTrainingStopped.model_construct(
                        candidate_id=candidate_id,
                        sampling_strategy=sampling_strategy,
                        source_commit=source_commit,
                        device=resolution,
                        completed_steps=step,
                        total_steps=training.steps,
                        batch_size=training.batch_size,
                        sampler_cursor=sampler_cursor,
                        durable_state_name=state_path.name,
                        durable_state_manifest_sha256=state_manifest.checksum_sha256,
                        training_config_sha256=training_hash,
                        model_config_sha256=model_hash,
                        train_inventory_sha256=train_inventory_sha256,
                        validation_inventory_sha256=validation_inventory_sha256,
                        train_tokenized_sha256=train_tokenized_hash,
                        validation_tokenized_sha256=validation_tokenized_hash,
                        tokenizer_manifest_sha256=tokenizer_hash,
                        checksum_sha256="0" * 64,
                    )
                    return _stopped_with_checksum(draft)

        _synchronize(device)
        elapsed = _elapsed(monotonic_clock, started=started, offset=elapsed_offset)
        if elapsed <= 0.0:
            elapsed = float.fromhex("0x1.0p-52")
        model.load_state_dict(best_state, strict=True)
        model.eval()
        checkpoint = save_checkpoint(
            model,
            output_directory=output_directory,
            tokenizer_manifest=tokenizer_manifest,
            source_commit=source_commit,
            seed=training.seed,
            training_steps=best_step,
            initial_loss=initial_nll,
            final_loss=best_nll,
        )
        checkpoint_bytes = (
            checkpoint.weights_size_bytes
            + (output_directory / STATE_MANIFEST_FILENAME).stat().st_size
        )
        if checkpoint_bytes > selected_checkpoint_upper_bound_bytes(
            model_config,
            vocab_size=vocab_size,
        ):
            raise TrainingError("selected checkpoint exceeds its deterministic byte budget")
        reloaded, reloaded_manifest = load_checkpoint(
            output_directory,
            expected_manifest_sha256=checkpoint.checksum_sha256,
            expected_tokenizer_sha256=tokenizer_hash,
            device=device,
        )
        reloaded_nll, _ = compact_validation_nll(
            reloaded,
            validation_examples,
            batch_size=training.batch_size,
            context_length=model_config.context_length,
            device=device,
        )
        if reloaded_manifest != checkpoint or reloaded_nll != best_nll:
            raise TrainingError("immutable selected checkpoint failed exact reload verification")
        _notify(
            progress_callback,
            TrainingProgress(
                event="final_checkpoint",
                step=training.steps,
                total_steps=training.steps,
                checkpoint_name=output_directory.name,
            ),
            include_mps_rng=include_mps_rng,
        )
        draft_result = CompactTrainingResult.model_construct(
            candidate_id=candidate_id,
            sampling_strategy=sampling_strategy,
            source_commit=source_commit,
            device=resolution,
            parameter_count=parameter_count,
            vocab_size=vocab_size,
            train_example_count=len(train_examples),
            validation_example_count=len(validation_examples),
            training_steps=training.steps,
            selected_step=best_step,
            initial_validation_nll=initial_nll,
            selected_validation_nll=best_nll,
            selected_score=best_score,
            final_training_nll=final_training_nll,
            validation_curve=tuple(curve),
            training_config_sha256=training_hash,
            model_config_sha256=model_hash,
            train_inventory_sha256=train_inventory_sha256,
            validation_inventory_sha256=validation_inventory_sha256,
            train_tokenized_sha256=train_tokenized_hash,
            validation_tokenized_sha256=validation_tokenized_hash,
            tokenizer_manifest_sha256=tokenizer_hash,
            elapsed_seconds=elapsed,
            scored_target_tokens=scored_target_tokens,
            target_tokens_per_second=scored_target_tokens / elapsed,
            process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            mps_peak_current_allocated_bytes=peak_current,
            mps_peak_driver_allocated_bytes=peak_driver,
            durable_state_count=durable_state_count,
            checkpoint_manifest_sha256=checkpoint.checksum_sha256,
            checkpoint_weights_sha256=checkpoint.weights_sha256,
            checkpoint_size_bytes=checkpoint.weights_size_bytes,
            checksum_sha256="0" * 64,
        )
        return _result_with_checksum(draft_result)
    finally:
        try:
            torch.set_rng_state(caller_cpu_rng)
            if caller_mps_rng is not None:
                torch.mps.set_rng_state(caller_mps_rng)
        finally:
            _TRAINING_RNG_LOCK.release()


__all__ = [
    "MAX_RETAINED_DURABLE_STATES",
    "TRAINING_CONTRACT_VERSION",
    "TRAINING_STATE_VERSION",
    "CompactTrainingOutcome",
    "CompactTrainingResult",
    "CompactTrainingStopped",
    "DeviceResolution",
    "EvaluationCallback",
    "MonotonicClock",
    "ProgressCallback",
    "SamplingStrategy",
    "StopRequested",
    "TrainingError",
    "TrainingEvaluationPoint",
    "TrainingProgress",
    "TrainingStateFile",
    "TrainingStateManifest",
    "compact_validation_nll",
    "durable_training_state_upper_bound_bytes",
    "latest_committed_training_state",
    "resolve_training_device",
    "retire_superseded_training_states",
    "selected_checkpoint_upper_bound_bytes",
    "tokenized_inventory_sha256",
    "train_compact_model",
    "uniform_control_batch_indices",
]
