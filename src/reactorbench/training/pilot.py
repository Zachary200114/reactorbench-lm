"""Phase 5 baseline suite and validation-selected Transformer pilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import Field, StrictFloat, model_validator

from reactorbench.dataset import ArtifactWriter, verify_development_candidate_artifact
from reactorbench.dataset.review import HumanReviewRecord
from reactorbench.evaluation.baselines import BaselineResult, run_preregistered_baselines
from reactorbench.evaluation.config import Phase5Config, TransformerTrainingConfig
from reactorbench.evaluation.data import ExperimentData, materialize_experiment_data
from reactorbench.evaluation.serialization import (
    TokenizedExample,
    batch_tensors,
    supervised_causal_loss,
    tokenize_example,
)
from reactorbench.model import (
    Phase4Config,
    TransformerConfig,
    TransformerLM,
    exact_parameter_count,
    initialized_model,
    load_checkpoint,
    load_phase4_config,
    resolve_project_path,
    save_checkpoint,
)
from reactorbench.schemas.base import ContractModel, canonical_json_bytes, canonical_sha256
from reactorbench.tokenizer import ProjectTokenizer, TokenizerArtifactManifest

from .smoke import SmokeRunReport, verify_phase4_run

MAX_REPORT_BYTES = 4 * 1024 * 1024


class ValidationPoint(ContractModel):
    step: int = Field(strict=True, ge=0)
    target_nll: StrictFloat

    @model_validator(mode="after")
    def nll_is_finite(self) -> ValidationPoint:
        if not math.isfinite(self.target_nll) or self.target_nll < 0.0:
            raise ValueError("validation NLL must be finite and non-negative")
        return self


class TransformerPilotResult(ContractModel):
    tier: Literal["smaller_transformer", "pilot_transformer"]
    device: Literal["cpu", "mps"]
    parameter_count: int = Field(strict=True, ge=1)
    context_length: int = Field(strict=True, ge=8)
    batch_size: int = Field(strict=True, ge=1)
    training_steps: int = Field(strict=True, ge=1)
    selected_step: int = Field(strict=True, ge=0)
    initial_validation_nll: StrictFloat
    selected_validation_nll: StrictFloat
    validation_nll_reduction_fraction: StrictFloat
    final_training_nll: StrictFloat
    validation_curve: tuple[ValidationPoint, ...]
    elapsed_seconds: StrictFloat
    scored_target_tokens: int = Field(strict=True, ge=1)
    target_tokens_per_second: StrictFloat
    process_peak_rss_bytes: int = Field(strict=True, ge=1)
    mps_peak_current_allocated_bytes: int = Field(strict=True, ge=0)
    mps_peak_driver_allocated_bytes: int = Field(strict=True, ge=0)
    checkpoint_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_size_bytes: int = Field(strict=True, ge=1)

    @model_validator(mode="after")
    def measurements_are_consistent(self) -> TransformerPilotResult:
        numbers = (
            self.initial_validation_nll,
            self.selected_validation_nll,
            self.validation_nll_reduction_fraction,
            self.final_training_nll,
            self.elapsed_seconds,
            self.target_tokens_per_second,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in numbers):
            raise ValueError("Transformer pilot measurements must be finite and non-negative")
        if self.selected_step not in {point.step for point in self.validation_curve}:
            raise ValueError("selected step must be one of the validation checkpoints")
        best = min(self.validation_curve, key=lambda point: (point.target_nll, point.step))
        if self.selected_step != best.step or self.selected_validation_nll != best.target_nll:
            raise ValueError("selected checkpoint must use the lowest validation NLL")
        return self


class Phase5RunReport(ContractModel):
    report_version: Literal["0.1.0"] = "0.1.0"
    run_status: Literal["phase5_pilot_passed"] = "phase5_pilot_passed"
    source_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    phase5_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase4_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    experiment_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    train_example_count: int = Field(strict=True, ge=1)
    validation_example_count: int = Field(strict=True, ge=1)
    prohibited_split_example_count: Literal[0]
    truncated_train_prompts_smaller: int = Field(strict=True, ge=0)
    truncated_validation_prompts_smaller: int = Field(strict=True, ge=0)
    truncated_train_prompts_pilot: int = Field(strict=True, ge=0)
    truncated_validation_prompts_pilot: int = Field(strict=True, ge=0)
    baseline_results: tuple[BaselineResult, ...]
    transformer_results: tuple[TransformerPilotResult, TransformerPilotResult]
    minimum_validation_nll_reduction_fraction: StrictFloat
    validation_only_selection_verified: Literal[True]
    all_baselines_completed: Literal[True]
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def acceptance_and_checksum_are_valid(self) -> Phase5RunReport:
        expected_baselines = {
            ("majority_frequency", "fault_family"),
            ("majority_frequency", "next_action"),
            ("majority_frequency", "continue_log"),
            ("deterministic_keyword_rules", "fault_family"),
            ("deterministic_keyword_rules", "next_action"),
            ("word_ngram_suffix", "continue_log"),
            ("token_trigram_additive", "target_language_modeling"),
            ("bag_of_words_logistic_regression", "fault_family"),
            ("gru_sequence_classifier", "fault_family"),
            ("gru_sequence_classifier", "continue_log"),
        }
        actual = {(item.baseline_name, item.task_name) for item in self.baseline_results}
        if actual != expected_baselines or len(self.baseline_results) != len(expected_baselines):
            raise ValueError("Phase 5 report does not contain the exact baseline inventory")
        if tuple(result.tier for result in self.transformer_results) != (
            "smaller_transformer",
            "pilot_transformer",
        ):
            raise ValueError("Transformer results must use canonical tier order")
        if any(
            result.validation_nll_reduction_fraction
            < self.minimum_validation_nll_reduction_fraction
            for result in self.transformer_results
        ):
            raise ValueError("Transformer validation improvement threshold failed")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("Phase 5 report checksum mismatch")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        while chunk := artifact.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError("Phase 5 JSON contains a duplicate key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"Phase 5 JSON contains non-finite data: {value}")

    decoded = json.loads(
        payload.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject_constant
    )
    if type(decoded) is not dict:
        raise ValueError("Phase 5 JSON must contain one object")
    return decoded


def _load_phase5_inputs(
    config: Phase5Config, project_root: Path
) -> tuple[
    Phase4Config,
    SmokeRunReport,
    TokenizerArtifactManifest,
    ProjectTokenizer,
    ExperimentData,
    str,
]:
    phase4_path = resolve_project_path(
        project_root, config.phase5.phase4_config_path, must_exist=True
    )
    phase4_config = load_phase4_config(phase4_path)
    phase4_report = verify_phase4_run(phase4_config, project_root=project_root)
    dataset_root = resolve_project_path(
        project_root, phase4_config.phase4.dataset_root, must_exist=True
    )
    verified = verify_development_candidate_artifact(
        ArtifactWriter(dataset_root),
        relative_directory=phase4_config.phase4.dataset_artifact_name,
    )
    approval_path = resolve_project_path(
        project_root, phase4_config.phase4.postrender_approval_record, must_exist=True
    )
    approval_payload = approval_path.read_bytes()
    _strict_json_object(approval_payload)
    approval = HumanReviewRecord.model_validate_json(approval_payload)
    if approval.decision != "approved":
        raise ValueError("Phase 3 post-render approval is not approved")
    phase4_run = resolve_project_path(project_root, config.phase5.phase4_run_path, must_exist=True)
    tokenizer = ProjectTokenizer.load(
        phase4_run / "tokenizer",
        expected_checksum=phase4_report.tokenizer_manifest_sha256,
    )
    data = materialize_experiment_data(
        verified,
        maximum_prompt_utf8_bytes=config.serialization.maximum_prompt_utf8_bytes,
    )
    if len(data.train) != 630 or len(data.validation) != 252:
        raise ValueError("approved Phase 5 train/validation inventory changed unexpectedly")
    return (
        phase4_config,
        phase4_report,
        tokenizer.manifest,
        tokenizer,
        data,
        verified.candidate.checksum_sha256,
    )


def _tokenize_inventory(
    records: tuple[Any, ...],
    tokenizer: ProjectTokenizer,
    config: Phase5Config,
    *,
    context_length: int,
) -> tuple[TokenizedExample, ...]:
    return tuple(
        tokenize_example(
            record,
            tokenizer,
            config.serialization,
            context_length=context_length,
        )
        for record in records
    )


def _device(config: TransformerTrainingConfig) -> torch.device:
    if config.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if config.device == "mps" and not config.allow_cpu_fallback:
        raise RuntimeError("configured MPS device is unavailable and fallback is disabled")
    return torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _validation_nll(
    model: TransformerLM,
    records: tuple[TokenizedExample, ...],
    *,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> float:
    model.eval()
    weighted_loss = 0.0
    target_tokens = 0
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            input_ids, attention_mask, target_mask = batch_tensors(
                batch, context_length=context_length
            )
            count = int(target_mask[:, 1:].sum().item())
            loss = supervised_causal_loss(
                model,
                input_ids.to(device),
                attention_mask.to(device),
                target_mask.to(device),
            )
            weighted_loss += float(loss.item()) * count
            target_tokens += count
    _synchronize(device)
    return weighted_loss / target_tokens


def _train_transformer(
    *,
    tier: Literal["smaller_transformer", "pilot_transformer"],
    model_config: TransformerConfig,
    training_config: TransformerTrainingConfig,
    tokenizer_manifest: TokenizerArtifactManifest,
    tokenizer: ProjectTokenizer,
    data: ExperimentData,
    phase5_config: Phase5Config,
    output_directory: Path,
    source_commit: str,
) -> TransformerPilotResult:
    train = _tokenize_inventory(
        data.train, tokenizer, phase5_config, context_length=model_config.context_length
    )
    validation = _tokenize_inventory(
        data.validation, tokenizer, phase5_config, context_length=model_config.context_length
    )
    device = _device(training_config)
    model = initialized_model(
        model_config, vocab_size=tokenizer.vocab_size, seed=training_config.seed
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != exact_parameter_count(model_config, vocab_size=tokenizer.vocab_size):
        raise RuntimeError("allocated Transformer parameter count is incorrect")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    initial_validation = _validation_nll(
        model,
        validation,
        batch_size=training_config.batch_size,
        context_length=model_config.context_length,
        device=device,
    )
    points: list[ValidationPoint] = [ValidationPoint(step=0, target_nll=initial_validation)]
    best_step = 0
    best_nll = initial_validation
    best_state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }
    rng = np.random.default_rng(training_config.seed)
    scored_tokens = 0
    final_train = initial_validation
    peak_current = 0
    peak_driver = 0
    _synchronize(device)
    started = time.perf_counter()
    model.train()
    for step in range(1, training_config.steps + 1):
        indices = rng.choice(len(train), size=training_config.batch_size, replace=False)
        batch = tuple(train[int(index)] for index in indices)
        input_ids, attention_mask, target_mask = batch_tensors(
            batch, context_length=model_config.context_length
        )
        scored_tokens += int(target_mask[:, 1:].sum().item())
        optimizer.zero_grad(set_to_none=True)
        loss = supervised_causal_loss(
            model,
            input_ids.to(device),
            attention_mask.to(device),
            target_mask.to(device),
        )
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
        optimizer.step()
        final_train = float(loss.detach().item())
        if device.type == "mps":
            peak_current = max(peak_current, int(torch.mps.current_allocated_memory()))
            peak_driver = max(peak_driver, int(torch.mps.driver_allocated_memory()))
        if step % training_config.evaluation_interval == 0:
            value = _validation_nll(
                model,
                validation,
                batch_size=training_config.batch_size,
                context_length=model_config.context_length,
                device=device,
            )
            point = ValidationPoint(step=step, target_nll=value)
            points.append(point)
            if (value, step) < (best_nll, best_step):
                best_nll = value
                best_step = step
                best_state = {
                    name: tensor.detach().cpu().contiguous().clone()
                    for name, tensor in model.state_dict().items()
                }
            model.train()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state, strict=True)
    model.eval()
    checkpoint = save_checkpoint(
        model,
        output_directory=output_directory,
        tokenizer_manifest=tokenizer_manifest,
        source_commit=source_commit,
        seed=training_config.seed,
        training_steps=best_step,
        initial_loss=initial_validation,
        final_loss=best_nll,
    )
    reloaded, _manifest = load_checkpoint(
        output_directory,
        expected_manifest_sha256=checkpoint.checksum_sha256,
        expected_tokenizer_sha256=tokenizer_manifest.checksum_sha256,
        device=device,
    )
    reloaded_nll = _validation_nll(
        reloaded,
        validation,
        batch_size=training_config.batch_size,
        context_length=model_config.context_length,
        device=device,
    )
    if reloaded_nll != best_nll:
        raise RuntimeError("selected checkpoint reload changed validation NLL")
    reduction = (initial_validation - best_nll) / initial_validation
    actual_device: Literal["cpu", "mps"] = "mps" if device.type == "mps" else "cpu"
    return TransformerPilotResult(
        tier=tier,
        device=actual_device,
        parameter_count=parameter_count,
        context_length=model_config.context_length,
        batch_size=training_config.batch_size,
        training_steps=training_config.steps,
        selected_step=best_step,
        initial_validation_nll=initial_validation,
        selected_validation_nll=best_nll,
        validation_nll_reduction_fraction=reduction,
        final_training_nll=final_train,
        validation_curve=tuple(points),
        elapsed_seconds=elapsed,
        scored_target_tokens=scored_tokens,
        target_tokens_per_second=scored_tokens / elapsed,
        process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        mps_peak_current_allocated_bytes=peak_current,
        mps_peak_driver_allocated_bytes=peak_driver,
        checkpoint_manifest_sha256=checkpoint.checksum_sha256,
        checkpoint_weights_sha256=checkpoint.weights_sha256,
        checkpoint_size_bytes=checkpoint.weights_size_bytes,
    )


def run_phase5_pilot(
    config: Phase5Config,
    *,
    project_root: Path,
    source_commit: str,
) -> Phase5RunReport:
    if type(config) is not Phase5Config:
        raise TypeError("config must be an exact Phase5Config")
    if (
        type(source_commit) is not str
        or not 7 <= len(source_commit) <= 64
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("source_commit must be a lowercase hexadecimal Git revision")
    phase4, phase4_report, tokenizer_manifest, tokenizer, data, candidate_sha256 = (
        _load_phase5_inputs(config, project_root)
    )
    run_root = resolve_project_path(project_root, config.phase5.run_root, must_exist=False)
    run_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    output = run_root / config.phase5.run_name
    if output.exists() or output.is_symlink():
        raise FileExistsError("Phase 5 run output already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=run_root))
    try:
        pilot_train = _tokenize_inventory(
            data.train, tokenizer, config, context_length=phase4.pilot_model.context_length
        )
        pilot_validation = _tokenize_inventory(
            data.validation,
            tokenizer,
            config,
            context_length=phase4.pilot_model.context_length,
        )
        baselines = run_preregistered_baselines(
            data,
            tokenizer,
            config.baselines,
            tokenized_train=pilot_train,
            tokenized_validation=pilot_validation,
        )
        smaller = _train_transformer(
            tier="smaller_transformer",
            model_config=phase4.smoke_model,
            training_config=config.smaller_transformer,
            tokenizer_manifest=tokenizer_manifest,
            tokenizer=tokenizer,
            data=data,
            phase5_config=config,
            output_directory=temporary / "smaller-checkpoint",
            source_commit=source_commit,
        )
        pilot = _train_transformer(
            tier="pilot_transformer",
            model_config=phase4.pilot_model,
            training_config=config.pilot_transformer,
            tokenizer_manifest=tokenizer_manifest,
            tokenizer=tokenizer,
            data=data,
            phase5_config=config,
            output_directory=temporary / "pilot-checkpoint",
            source_commit=source_commit,
        )
        lock_sha256 = _sha256(resolve_project_path(project_root, "uv.lock", must_exist=True))
        draft = Phase5RunReport.model_construct(
            source_commit=source_commit,
            phase5_config_sha256=canonical_sha256(config.model_dump(mode="json", round_trip=True)),
            dependency_lock_sha256=lock_sha256,
            phase4_report_sha256=phase4_report.checksum_sha256,
            dataset_candidate_sha256=candidate_sha256,
            tokenizer_manifest_sha256=tokenizer_manifest.checksum_sha256,
            experiment_inventory_sha256=data.inventory_sha256,
            train_example_count=len(data.train),
            validation_example_count=len(data.validation),
            prohibited_split_example_count=0,
            truncated_train_prompts_smaller=sum(
                tokenize_example(
                    item,
                    tokenizer,
                    config.serialization,
                    context_length=phase4.smoke_model.context_length,
                ).truncated_prompt
                for item in data.train
            ),
            truncated_validation_prompts_smaller=sum(
                tokenize_example(
                    item,
                    tokenizer,
                    config.serialization,
                    context_length=phase4.smoke_model.context_length,
                ).truncated_prompt
                for item in data.validation
            ),
            truncated_train_prompts_pilot=sum(item.truncated_prompt for item in pilot_train),
            truncated_validation_prompts_pilot=sum(
                item.truncated_prompt for item in pilot_validation
            ),
            baseline_results=baselines,
            transformer_results=(smaller, pilot),
            minimum_validation_nll_reduction_fraction=(
                config.acceptance.minimum_validation_nll_reduction_fraction
            ),
            validation_only_selection_verified=True,
            all_baselines_completed=True,
            checksum_sha256="0" * 64,
        )
        checksum = canonical_sha256(
            draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        report = Phase5RunReport(
            **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
            checksum_sha256=checksum,
        )
        report_path = temporary / "report.json"
        report_path.write_bytes(
            canonical_json_bytes(report.model_dump(mode="json", round_trip=True)) + b"\n"
        )
        if report_path.stat().st_size > config.acceptance.maximum_report_bytes:
            raise ValueError("Phase 5 report exceeds its configured size limit")
        size = sum(path.stat().st_size for path in temporary.rglob("*") if path.is_file())
        if size > config.acceptance.maximum_run_bytes:
            raise ValueError("Phase 5 run exceeds its configured size limit")
        os.rename(temporary, output)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_phase5_run(config: Phase5Config, *, project_root: Path) -> Phase5RunReport:
    phase4, phase4_report, tokenizer_manifest, tokenizer, data, candidate_sha256 = (
        _load_phase5_inputs(config, project_root)
    )
    run_root = resolve_project_path(project_root, config.phase5.run_root, must_exist=True)
    directory = run_root / config.phase5.run_name
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Phase 5 run directory is missing or unsafe")
    if {path.name for path in directory.iterdir()} != {
        "report.json",
        "smaller-checkpoint",
        "pilot-checkpoint",
    }:
        raise ValueError("Phase 5 run contains an unexpected file inventory")
    report_path = directory / "report.json"
    if (
        report_path.is_symlink()
        or report_path.stat().st_size > config.acceptance.maximum_report_bytes
    ):
        raise ValueError("Phase 5 report is unsafe or oversized")
    payload = report_path.read_bytes()
    _strict_json_object(payload)
    report = Phase5RunReport.model_validate_json(payload)
    if report.phase5_config_sha256 != canonical_sha256(
        config.model_dump(mode="json", round_trip=True)
    ):
        raise ValueError("Phase 5 report is bound to a different config")
    if report.phase4_report_sha256 != phase4_report.checksum_sha256:
        raise ValueError("Phase 5 report is bound to a different Phase 4 run")
    if report.dataset_candidate_sha256 != candidate_sha256:
        raise ValueError("Phase 5 report is bound to a different candidate")
    if report.tokenizer_manifest_sha256 != tokenizer_manifest.checksum_sha256:
        raise ValueError("Phase 5 report is bound to a different tokenizer")
    if report.experiment_inventory_sha256 != data.inventory_sha256:
        raise ValueError("Phase 5 report is bound to a different experiment inventory")
    if report.dependency_lock_sha256 != _sha256(
        resolve_project_path(project_root, "uv.lock", must_exist=True)
    ):
        raise ValueError("Phase 5 report is bound to a different dependency lock")
    for result, model_config, name in (
        (report.transformer_results[0], phase4.smoke_model, "smaller-checkpoint"),
        (report.transformer_results[1], phase4.pilot_model, "pilot-checkpoint"),
    ):
        if result.context_length != model_config.context_length:
            raise ValueError("Phase 5 checkpoint context differs from reviewed model tier")
        _model, manifest = load_checkpoint(
            directory / name,
            expected_manifest_sha256=result.checkpoint_manifest_sha256,
            expected_tokenizer_sha256=tokenizer.manifest.checksum_sha256,
            device=torch.device(result.device),
        )
        if (
            manifest.weights_sha256 != result.checkpoint_weights_sha256
            or manifest.weights_size_bytes != result.checkpoint_size_bytes
        ):
            raise ValueError("Phase 5 checkpoint metadata differs from its report")
    return report


__all__ = [
    "Phase5RunReport",
    "TransformerPilotResult",
    "ValidationPoint",
    "run_phase5_pilot",
    "verify_phase5_run",
]
