"""Deterministic Phase 4 tokenizer and tiny-shard Transformer smoke gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

import numpy
import pydantic
import safetensors
import sentencepiece
import torch
from pydantic import Field, StrictFloat, model_validator
from safetensors.torch import load_file, save, save_file

from reactorbench.dataset import (
    ArtifactWriter,
    VerifiedDevelopmentCandidateArtifact,
    verify_development_candidate_artifact,
)
from reactorbench.dataset.review import HumanReviewRecord
from reactorbench.model import (
    Phase4Config,
    TransformerLM,
    causal_language_model_loss,
    exact_parameter_count,
    initialized_model,
    load_checkpoint,
    resolve_project_path,
    save_checkpoint,
)
from reactorbench.schemas.base import ContractModel, canonical_json_bytes, canonical_sha256
from reactorbench.tokenizer import (
    PAD_ID,
    ProjectTokenizer,
    approved_training_corpus,
    train_tokenizer,
)

MAX_SMOKE_INPUT_BYTES = 8 * 1024 * 1024
MAX_SMOKE_REPORT_BYTES = 1024 * 1024


class ModelTierParameterCounts(ContractModel):
    smoke: int = Field(strict=True, ge=1)
    pilot: int = Field(strict=True, ge=1)
    main: int = Field(strict=True, ge=1)

    @model_validator(mode="after")
    def counts_are_monotonic(self) -> ModelTierParameterCounts:
        if not self.smoke < self.pilot < self.main:
            raise ValueError("model tier parameter counts must increase monotonically")
        return self


class DependencyVersions(ContractModel):
    numpy: str = Field(min_length=1, max_length=32)
    pydantic: str = Field(min_length=1, max_length=32)
    torch: str = Field(min_length=1, max_length=32)
    sentencepiece: str = Field(min_length=1, max_length=32)
    safetensors: str = Field(min_length=1, max_length=32)


class SmokeRunReport(ContractModel):
    report_version: Literal["0.1.0"] = "0.1.0"
    run_status: Literal["phase4_smoke_passed"] = "phase4_smoke_passed"
    source_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    dependency_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase4_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    smoke_inputs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_logits_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_versions: DependencyVersions
    parameter_counts: ModelTierParameterCounts
    device: Literal["cpu", "mps"]
    document_count: int = Field(strict=True, ge=1)
    batch_size: int = Field(strict=True, ge=1)
    sequence_length: int = Field(strict=True, ge=2)
    target_tokens_per_step: int = Field(strict=True, ge=1)
    training_steps: int = Field(strict=True, ge=1)
    initial_loss: StrictFloat
    final_loss: StrictFloat
    loss_reduction_fraction: StrictFloat
    loss_curve: tuple[StrictFloat, ...]
    elapsed_seconds: StrictFloat
    tokens_per_second: StrictFloat
    causal_mask_verified: Literal[True]
    deterministic_evaluation_verified: Literal[True]
    checkpoint_reload_verified: Literal[True]
    tiny_shard_overfit_verified: Literal[True]
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def numbers_and_checksum_are_valid(self) -> SmokeRunReport:
        numbers = (
            self.initial_loss,
            self.final_loss,
            self.loss_reduction_fraction,
            self.elapsed_seconds,
            self.tokens_per_second,
            *self.loss_curve,
        )
        if any(not math.isfinite(number) or number < 0.0 for number in numbers):
            raise ValueError("smoke report measurements must be finite and non-negative")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("smoke report checksum mismatch")
        return self


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError("JSON artifact contains a duplicate key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON artifact contains non-finite data: {value}")

    decoded = json.loads(
        payload.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject_constant
    )
    if type(decoded) is not dict:
        raise ValueError("JSON artifact must contain one object")
    return decoded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        while chunk := artifact.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(save({"logits": tensor.detach().cpu().contiguous()})).hexdigest()


def _load_approved_candidate(
    config: Phase4Config, project_root: Path
) -> tuple[VerifiedDevelopmentCandidateArtifact, HumanReviewRecord]:
    dataset_root = resolve_project_path(project_root, config.phase4.dataset_root, must_exist=True)
    writer = ArtifactWriter(dataset_root)
    verified = verify_development_candidate_artifact(
        writer, relative_directory=config.phase4.dataset_artifact_name
    )
    approval_path = resolve_project_path(
        project_root, config.phase4.postrender_approval_record, must_exist=True
    )
    if approval_path.is_symlink() or not approval_path.is_file():
        raise ValueError("post-render approval record must be a regular file")
    approval_payload = approval_path.read_bytes()
    _strict_json_object(approval_payload)
    approval = HumanReviewRecord.model_validate_json(approval_payload)
    return verified, approval


def _device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("the configured MPS device is unavailable")
    return torch.device(name)


def _smoke_batch(
    tokenizer: ProjectTokenizer,
    documents: tuple[str, ...],
    *,
    document_count: int,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = documents[:document_count]
    if len(selected) != document_count:
        raise ValueError("smoke document count exceeds the approved training corpus")
    rows: list[tuple[int, ...]] = []
    for document in selected:
        encoded = tokenizer.encode(document)
        if len(encoded) < 2:
            raise ValueError("smoke document produced fewer than two tokens")
        rows.append(encoded[:context_length])
    sequence_length = min(context_length, max(len(row) for row in rows))
    input_ids = torch.full((len(rows), sequence_length), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for index, row in enumerate(rows):
        visible = row[:sequence_length]
        input_ids[index, : len(visible)] = torch.tensor(visible, dtype=torch.long)
        attention_mask[index, : len(visible)] = True
    return input_ids, attention_mask


def _causal_mask_probe(model: TransformerLM, input_ids: torch.Tensor) -> bool:
    if input_ids.shape[1] < 3:
        raise ValueError("causal-mask probe requires at least three tokens")
    model.eval()
    probe = input_ids[:1, : min(8, input_ids.shape[1])].clone()
    changed = probe.clone()
    changed[:, -1] = (changed[:, -1] + 1) % model.vocab_size
    with torch.no_grad():
        original_logits = model(probe)
        changed_logits = model(changed)
    return torch.equal(original_logits[:, :-1, :], changed_logits[:, :-1, :])


def _report(
    *,
    config: Phase4Config,
    source_commit: str,
    dependency_lock_sha256: str,
    candidate_sha256: str,
    corpus_sha256: str,
    tokenizer_sha256: str,
    checkpoint_sha256: str,
    smoke_inputs_sha256: str,
    logits_sha256: str,
    counts: ModelTierParameterCounts,
    sequence_length: int,
    target_tokens: int,
    initial_loss: float,
    final_loss: float,
    loss_curve: tuple[float, ...],
    elapsed_seconds: float,
) -> SmokeRunReport:
    reduction = (initial_loss - final_loss) / initial_loss
    tokens_seen = target_tokens * config.smoke_training.steps
    draft = SmokeRunReport.model_construct(
        source_commit=source_commit,
        dependency_lock_sha256=dependency_lock_sha256,
        phase4_config_sha256=canonical_sha256(config.model_dump(mode="json", round_trip=True)),
        dataset_candidate_sha256=candidate_sha256,
        corpus_sha256=corpus_sha256,
        tokenizer_manifest_sha256=tokenizer_sha256,
        checkpoint_manifest_sha256=checkpoint_sha256,
        smoke_inputs_sha256=smoke_inputs_sha256,
        evaluation_logits_sha256=logits_sha256,
        dependency_versions=DependencyVersions(
            numpy=numpy.__version__,
            pydantic=pydantic.__version__,
            torch=torch.__version__,
            sentencepiece=sentencepiece.__version__,
            safetensors=safetensors.__version__,
        ),
        parameter_counts=counts,
        device=config.smoke_training.device,
        document_count=config.smoke_training.document_count,
        batch_size=config.smoke_training.batch_size,
        sequence_length=sequence_length,
        target_tokens_per_step=target_tokens,
        training_steps=config.smoke_training.steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        loss_reduction_fraction=reduction,
        loss_curve=loss_curve,
        elapsed_seconds=elapsed_seconds,
        tokens_per_second=tokens_seen / elapsed_seconds,
        causal_mask_verified=True,
        deterministic_evaluation_verified=True,
        checkpoint_reload_verified=True,
        tiny_shard_overfit_verified=True,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return SmokeRunReport(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
        checksum_sha256=checksum,
    )


def run_phase4_smoke(
    config: Phase4Config,
    *,
    project_root: Path,
    source_commit: str,
) -> SmokeRunReport:
    """Run the complete non-overwriting Phase 4 correctness milestone."""

    if type(config) is not Phase4Config:
        raise TypeError("config must be an exact Phase4Config")
    if (
        type(source_commit) is not str
        or not 7 <= len(source_commit) <= 64
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("source_commit must be a lowercase hexadecimal Git revision")
    verified, approval = _load_approved_candidate(config, project_root)
    corpus = approved_training_corpus(verified, approval)
    lock_path = resolve_project_path(project_root, "uv.lock", must_exist=True)
    dependency_lock_sha256 = _sha256(lock_path)
    run_root = resolve_project_path(project_root, config.phase4.run_root, must_exist=False)
    run_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    if run_root.is_symlink() or not run_root.is_dir():
        raise ValueError("run root must be a regular directory")
    output = run_root / config.phase4.run_name
    if output.exists() or output.is_symlink():
        raise FileExistsError("Phase 4 run output already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=run_root))
    try:
        tokenizer_manifest = train_tokenizer(
            corpus, config.tokenizer, output_directory=temporary / "tokenizer"
        )
        tokenizer = ProjectTokenizer.load(
            temporary / "tokenizer",
            expected_checksum=tokenizer_manifest.checksum_sha256,
        )
        counts = ModelTierParameterCounts(
            smoke=exact_parameter_count(config.smoke_model, vocab_size=tokenizer.vocab_size),
            pilot=exact_parameter_count(config.pilot_model, vocab_size=tokenizer.vocab_size),
            main=exact_parameter_count(config.main_model, vocab_size=tokenizer.vocab_size),
        )
        input_ids, attention_mask = _smoke_batch(
            tokenizer,
            corpus.documents,
            document_count=config.smoke_training.document_count,
            context_length=config.smoke_model.context_length,
        )
        if input_ids.shape[0] != config.smoke_training.batch_size:
            raise ValueError("smoke document count and batch size must match")
        smoke_input_path = temporary / "smoke-inputs.safetensors"
        save_file(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            str(smoke_input_path),
            metadata={"format": "reactorbench-smoke-input-v0.1.0"},
        )
        if smoke_input_path.stat().st_size > MAX_SMOKE_INPUT_BYTES:
            raise ValueError("smoke input artifact exceeds its size limit")
        smoke_input_sha256 = _sha256(smoke_input_path)

        device = _device(config.smoke_training.device)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        model = initialized_model(
            config.smoke_model,
            vocab_size=tokenizer.vocab_size,
            seed=config.smoke_training.seed,
        ).to(device)
        if sum(parameter.numel() for parameter in model.parameters()) != counts.smoke:
            raise RuntimeError("allocated smoke model parameter count differs from the formula")
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        model.eval()
        with torch.no_grad():
            initial_loss = float(
                causal_language_model_loss(model, input_ids, attention_mask=attention_mask).item()
            )
        if not _causal_mask_probe(model, input_ids):
            raise RuntimeError("causal-mask probe failed before training")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.smoke_training.learning_rate,
            weight_decay=config.smoke_training.weight_decay,
        )
        losses: list[float] = [initial_loss]
        started = time.perf_counter()
        model.train()
        interval = max(1, config.smoke_training.steps // 12)
        for step in range(1, config.smoke_training.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = causal_language_model_loss(model, input_ids, attention_mask=attention_mask)
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.smoke_training.gradient_clip_norm
            )
            optimizer.step()
            if step % interval == 0 or step == config.smoke_training.steps:
                losses.append(float(loss.detach().item()))
        elapsed = time.perf_counter() - started
        model.eval()
        with torch.no_grad():
            final_loss = float(
                causal_language_model_loss(model, input_ids, attention_mask=attention_mask).item()
            )
            logits_one = model(input_ids, attention_mask)
            logits_two = model(input_ids, attention_mask)
        if not torch.equal(logits_one, logits_two):
            raise RuntimeError("deterministic evaluation produced unequal logits")
        reduction = (initial_loss - final_loss) / initial_loss
        if (
            final_loss > config.smoke_training.maximum_final_loss
            or reduction < config.smoke_training.minimum_loss_reduction_fraction
        ):
            raise RuntimeError("tiny-shard overfit acceptance threshold failed")

        checkpoint = save_checkpoint(
            model,
            output_directory=temporary / "checkpoint",
            tokenizer_manifest=tokenizer_manifest,
            source_commit=source_commit,
            seed=config.smoke_training.seed,
            training_steps=config.smoke_training.steps,
            initial_loss=initial_loss,
            final_loss=final_loss,
        )
        reloaded, reloaded_manifest = load_checkpoint(
            temporary / "checkpoint",
            expected_manifest_sha256=checkpoint.checksum_sha256,
            expected_tokenizer_sha256=tokenizer_manifest.checksum_sha256,
            device=device,
        )
        with torch.no_grad():
            reloaded_logits = reloaded(input_ids, attention_mask)
        if reloaded_manifest != checkpoint or not torch.equal(logits_one, reloaded_logits):
            raise RuntimeError("checkpoint save/reload equivalence failed")
        logits_sha256 = _tensor_sha256(logits_one)
        target_tokens = int(attention_mask[:, 1:].sum().item())
        report = _report(
            config=config,
            source_commit=source_commit,
            dependency_lock_sha256=dependency_lock_sha256,
            candidate_sha256=verified.candidate.checksum_sha256,
            corpus_sha256=corpus.manifest.corpus_sha256,
            tokenizer_sha256=tokenizer_manifest.checksum_sha256,
            checkpoint_sha256=checkpoint.checksum_sha256,
            smoke_inputs_sha256=smoke_input_sha256,
            logits_sha256=logits_sha256,
            counts=counts,
            sequence_length=input_ids.shape[1],
            target_tokens=target_tokens,
            initial_loss=initial_loss,
            final_loss=final_loss,
            loss_curve=tuple(losses),
            elapsed_seconds=elapsed,
        )
        (temporary / "report.json").write_bytes(
            canonical_json_bytes(report.model_dump(mode="json", round_trip=True)) + b"\n"
        )
        os.rename(temporary, output)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_phase4_run(
    config: Phase4Config,
    *,
    project_root: Path,
) -> SmokeRunReport:
    """Independently re-parse, re-hash, and re-evaluate a completed smoke run."""

    verified, approval = _load_approved_candidate(config, project_root)
    corpus = approved_training_corpus(verified, approval)
    run_root = resolve_project_path(project_root, config.phase4.run_root, must_exist=True)
    run_directory = run_root / config.phase4.run_name
    if run_directory.is_symlink() or not run_directory.is_dir():
        raise ValueError("Phase 4 run directory is missing or unsafe")
    if {path.name for path in run_directory.iterdir()} != {
        "checkpoint",
        "report.json",
        "smoke-inputs.safetensors",
        "tokenizer",
    }:
        raise ValueError("Phase 4 run contains an unexpected file inventory")
    report_path = run_directory / "report.json"
    input_path = run_directory / "smoke-inputs.safetensors"
    if report_path.stat().st_size > MAX_SMOKE_REPORT_BYTES:
        raise ValueError("Phase 4 report exceeds its size limit")
    report_payload = report_path.read_bytes()
    _strict_json_object(report_payload)
    report = SmokeRunReport.model_validate_json(report_payload)
    if report.phase4_config_sha256 != canonical_sha256(
        config.model_dump(mode="json", round_trip=True)
    ):
        raise ValueError("Phase 4 report is bound to a different configuration")
    lock_path = resolve_project_path(project_root, "uv.lock", must_exist=True)
    if report.dependency_lock_sha256 != _sha256(lock_path):
        raise ValueError("Phase 4 report is bound to a different dependency lockfile")
    if report.dataset_candidate_sha256 != verified.candidate.checksum_sha256:
        raise ValueError("Phase 4 report is bound to a different candidate")
    if report.corpus_sha256 != corpus.manifest.corpus_sha256:
        raise ValueError("Phase 4 report is bound to a different training corpus")
    if input_path.is_symlink() or input_path.stat().st_size > MAX_SMOKE_INPUT_BYTES:
        raise ValueError("smoke input artifact is unsafe or oversized")
    if _sha256(input_path) != report.smoke_inputs_sha256:
        raise ValueError("smoke input checksum mismatch")
    tokenizer = ProjectTokenizer.load(
        run_directory / "tokenizer",
        expected_checksum=report.tokenizer_manifest_sha256,
    )
    device = _device(report.device)
    model, _manifest = load_checkpoint(
        run_directory / "checkpoint",
        expected_manifest_sha256=report.checkpoint_manifest_sha256,
        expected_tokenizer_sha256=tokenizer.manifest.checksum_sha256,
        device=device,
    )
    tensors = load_file(str(input_path), device=str(device))
    if set(tensors) != {"attention_mask", "input_ids"}:
        raise ValueError("smoke input artifact has unexpected tensors")
    input_ids = tensors["input_ids"]
    attention_mask = tensors["attention_mask"]
    with torch.no_grad():
        logits_one = model(input_ids, attention_mask)
        logits_two = model(input_ids, attention_mask)
    if not torch.equal(logits_one, logits_two):
        raise ValueError("independent evaluation is nondeterministic")
    if _tensor_sha256(logits_one) != report.evaluation_logits_sha256:
        raise ValueError("independent evaluation logits do not match the report")
    return report


__all__ = [
    "DependencyVersions",
    "ModelTierParameterCounts",
    "SmokeRunReport",
    "run_phase4_smoke",
    "verify_phase4_run",
]
