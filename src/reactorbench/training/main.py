"""Phase 6 validation selection and one-time held-out evaluation.

The two public gates deliberately separate model selection from held-out access.
``run_phase6_selection`` may read only IID train/validation records.  The evaluation
gate verifies the immutable selection artifact and owner-approved golden packet
before materializing any test record.
"""

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

from reactorbench.dataset import (
    ArtifactWriter,
    infer_projection_view,
    project_trajectory,
    verify_development_candidate_artifact,
)
from reactorbench.dataset.catalog import AliasFamily, TemplateFamily
from reactorbench.dataset.renderer import render_model_input
from reactorbench.evaluation.baselines import (
    BaselineResult,
    _token_ngram_language_model,
    run_preregistered_baselines,
)
from reactorbench.evaluation.config import (
    Phase5Config,
    Phase6Config,
    TransformerTrainingConfig,
    load_phase5_config,
)
from reactorbench.evaluation.data import (
    ExperimentData,
    ExperimentExample,
    materialize_experiment_data,
    materialize_phase6_data,
)
from reactorbench.evaluation.decoding import DecodedPrediction, greedy_decode_predictions
from reactorbench.evaluation.golden import (
    load_golden_review_packet,
    load_golden_review_record,
    verify_golden_review,
)
from reactorbench.evaluation.metrics import (
    BootstrapInterval,
    CalibrationMetrics,
    ClassificationMetrics,
    LanguageModelMetrics,
    SetF1Metrics,
    bootstrap_mean_interval,
    calibration_metrics,
    classification_metrics,
    language_model_metrics,
    set_f1_metrics,
)
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
from reactorbench.schemas.enums import DiagnosisStatus, SplitName, TaskName
from reactorbench.schemas.provenance import ProvenanceRecord
from reactorbench.schemas.target import FaultDiagnosisTarget, NextActionTarget
from reactorbench.schemas.trajectory import StructuredTrajectory
from reactorbench.simulator import generate_trace
from reactorbench.tokenizer import ProjectTokenizer, TokenizerArtifactManifest

from .pilot import ValidationPoint, verify_phase5_run
from .smoke import verify_phase4_run

MAX_REPORT_BYTES = 16 * 1024 * 1024
MAX_PREDICTIONS_BYTES = 64 * 1024 * 1024
SELECTION_SUFFIX = "-selection"
TEST_SPLITS = (
    SplitName.IID_TEST,
    SplitName.TEMPLATE_TEST,
    SplitName.COMPONENT_TEST,
    SplitName.SEVERITY_TEST,
    SplitName.COMPOSITION_TEST,
    SplitName.COUNTERFACTUAL_TEST,
    SplitName.NOISE_TEST,
)


class Phase6ModelResult(ContractModel):
    experiment_id: Literal[
        "E3_main_transformer",
        "E5_renderer_diversity_ablation",
        "E6_abstention_ablation",
    ]
    device: Literal["cpu", "mps"]
    parameter_count: int = Field(strict=True, ge=1)
    train_example_count: int = Field(strict=True, ge=1)
    validation_example_count: int = Field(strict=True, ge=1)
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
    def result_is_consistent(self) -> Phase6ModelResult:
        values = (
            self.initial_validation_nll,
            self.selected_validation_nll,
            self.validation_nll_reduction_fraction,
            self.final_training_nll,
            self.elapsed_seconds,
            self.target_tokens_per_second,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Phase 6 model measurements must be finite and non-negative")
        best = min(self.validation_curve, key=lambda point: (point.target_nll, point.step))
        if (self.selected_step, self.selected_validation_nll) != (best.step, best.target_nll):
            raise ValueError("Phase 6 checkpoint selection is not validation-optimal")
        return self


class Phase6SelectionReport(ContractModel):
    report_version: Literal["0.1.0"] = "0.1.0"
    run_status: Literal["phase6_selection_passed"] = "phase6_selection_passed"
    source_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    phase6_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase5_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_records_materialized: Literal[False]
    test_predictions_generated: Literal[False]
    all_ablation_selection_complete: Literal[True]
    e7_status: Literal["not_applicable_no_compound_iid_train_rows"]
    results: tuple[Phase6ModelResult, Phase6ModelResult, Phase6ModelResult]
    selection_thresholds_passed: bool
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def order_and_checksum_match(self) -> Phase6SelectionReport:
        if tuple(result.experiment_id for result in self.results) != (
            "E3_main_transformer",
            "E5_renderer_diversity_ablation",
            "E6_abstention_ablation",
        ):
            raise ValueError("Phase 6 selection result order is not canonical")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("Phase 6 selection report checksum mismatch")
        return self


class ModelSplitEvaluation(ContractModel):
    experiment_id: str = Field(min_length=2, max_length=80)
    split_name: SplitName
    sample_count: int = Field(strict=True, ge=1)
    scored_sample_count: int = Field(strict=True, ge=0)
    insufficient_context_by_design: int = Field(strict=True, ge=0)
    language_model: LanguageModelMetrics | None
    classification: tuple[ClassificationMetrics, ...]

    @model_validator(mode="after")
    def counts_match(self) -> ModelSplitEvaluation:
        if self.scored_sample_count + self.insufficient_context_by_design != self.sample_count:
            raise ValueError("split scored and insufficient-context counts do not match")
        if (self.scored_sample_count == 0) != (self.language_model is None):
            raise ValueError("language-model metric presence does not match scored samples")
        return self


class BaselineSplitResult(ContractModel):
    split_name: SplitName
    result: BaselineResult


class MainPredictionMetrics(ContractModel):
    split_name: SplitName
    sample_count: int = Field(strict=True, ge=1)
    parse_success_rate: StrictFloat
    schema_validity_rate: StrictFloat
    exact_match_rate: StrictFloat
    exact_match_interval: BootstrapInterval
    evidence: SetF1Metrics | None
    calibration: CalibrationMetrics
    no_fault_sample_count: int = Field(strict=True, ge=0)
    no_fault_false_positive_rate: StrictFloat
    required_abstention_sample_count: int = Field(strict=True, ge=0)
    required_abstention_accuracy: StrictFloat

    @model_validator(mode="after")
    def rates_are_probabilities(self) -> MainPredictionMetrics:
        rates = (
            self.parse_success_rate,
            self.schema_validity_rate,
            self.exact_match_rate,
            self.no_fault_false_positive_rate,
            self.required_abstention_accuracy,
        )
        if any(not math.isfinite(rate) or not 0.0 <= rate <= 1.0 for rate in rates):
            raise ValueError("prediction rates must be finite probabilities")
        return self


class FailureGalleryEntry(ContractModel):
    category: str = Field(min_length=1, max_length=80)
    split_name: SplitName
    example_id: str = Field(min_length=1, max_length=128)
    task_name: TaskName
    expected_target: str = Field(min_length=1, max_length=65_536)
    predicted_target: str = Field(min_length=1, max_length=65_536)


class ExperimentDecodedPrediction(ContractModel):
    experiment_id: str = Field(min_length=2, max_length=80)
    prediction: DecodedPrediction
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def checksum_matches(self) -> ExperimentDecodedPrediction:
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("comparison prediction checksum mismatch")
        return self


class HeldoutAccessRecord(ContractModel):
    record_version: Literal["0.1.0"] = "0.1.0"
    access_status: Literal["authorized_single_phase6_evaluation"] = (
        "authorized_single_phase6_evaluation"
    )
    source_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    phase6_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    golden_review_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    access_count: Literal[1]
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def checksum_matches(self) -> HeldoutAccessRecord:
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("held-out access record checksum mismatch")
        return self


class Phase6EvaluationReport(ContractModel):
    report_version: Literal["0.1.0"] = "0.1.0"
    run_status: Literal["phase6_evaluation_complete"] = "phase6_evaluation_complete"
    source_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    phase6_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    golden_packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    golden_review_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    heldout_access_count: Literal[1]
    heldout_access_after_selection: Literal[True]
    test_example_count: Literal[894]
    experiment_matrix: tuple[str, ...]
    baseline_results: tuple[BaselineSplitResult, ...]
    model_split_results: tuple[ModelSplitEvaluation, ...]
    main_prediction_metrics: tuple[MainPredictionMetrics, ...]
    golden_case_count: Literal[15]
    golden_example_count: int = Field(strict=True, ge=15)
    golden_exact_match_rate: StrictFloat
    acceptance_checks: tuple[tuple[str, bool], ...]
    negative_results: tuple[str, ...]
    failure_gallery: tuple[FailureGalleryEntry, ...]
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    comparison_predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    golden_predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def checksum_matches(self) -> Phase6EvaluationReport:
        if tuple(name for name, _passed in self.acceptance_checks) != tuple(
            sorted(name for name, _passed in self.acceptance_checks)
        ):
            raise ValueError("acceptance checks must use canonical order")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("Phase 6 evaluation report checksum mismatch")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(value: str) -> str:
    if (
        type(value) is not str
        or not 7 <= len(value) <= 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("source_commit must be a lowercase hexadecimal Git revision")
    return value


def _device(config: TransformerTrainingConfig) -> torch.device:
    if config.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if config.device == "mps" and not config.allow_cpu_fallback:
        raise RuntimeError("configured MPS device is unavailable and fallback is disabled")
    return torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _safe_tokenize(
    records: tuple[ExperimentExample, ...],
    tokenizer: ProjectTokenizer,
    config: Phase5Config,
    *,
    context_length: int,
) -> tuple[tuple[TokenizedExample, ...], tuple[str, ...]]:
    tokenized: list[TokenizedExample] = []
    insufficient: list[str] = []
    for record in records:
        try:
            tokenized.append(
                tokenize_example(
                    record, tokenizer, config.serialization, context_length=context_length
                )
            )
        except ValueError as error:
            if str(error) != "complete target does not fit the configured model context":
                raise
            insufficient.append(record.example_id)
    return tuple(tokenized), tuple(insufficient)


def _nll(
    model: TransformerLM,
    records: tuple[TokenizedExample, ...],
    *,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[float, int]:
    if not records:
        raise ValueError("NLL requires at least one scorable record")
    weighted = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            input_ids, attention_mask, target_mask = batch_tensors(
                batch, context_length=context_length
            )
            tokens = int(target_mask[:, 1:].sum().item())
            loss = supervised_causal_loss(
                model,
                input_ids.to(device),
                attention_mask.to(device),
                target_mask.to(device),
            )
            weighted += float(loss.item()) * tokens
            count += tokens
    _synchronize(device)
    return weighted / count, count


def _train_model(
    *,
    experiment_id: Literal[
        "E3_main_transformer",
        "E5_renderer_diversity_ablation",
        "E6_abstention_ablation",
    ],
    model_config: TransformerConfig,
    training: TransformerTrainingConfig,
    tokenizer: ProjectTokenizer,
    tokenizer_manifest: TokenizerArtifactManifest,
    data: ExperimentData,
    serialization_config: Phase5Config,
    output: Path,
    source_commit: str,
) -> Phase6ModelResult:
    train, train_insufficient = _safe_tokenize(
        data.train,
        tokenizer,
        serialization_config,
        context_length=model_config.context_length,
    )
    validation, validation_insufficient = _safe_tokenize(
        data.validation,
        tokenizer,
        serialization_config,
        context_length=model_config.context_length,
    )
    if train_insufficient or validation_insufficient:
        raise ValueError("training and validation targets must fit the frozen context")
    device = _device(training)
    model = initialized_model(model_config, vocab_size=tokenizer.vocab_size, seed=training.seed).to(
        device
    )
    parameters = exact_parameter_count(model_config, vocab_size=tokenizer.vocab_size)
    if sum(parameter.numel() for parameter in model.parameters()) != parameters:
        raise RuntimeError("allocated parameter count differs from the exact formula")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    initial, _ = _nll(
        model,
        validation,
        batch_size=training.batch_size,
        context_length=model_config.context_length,
        device=device,
    )
    points = [ValidationPoint(step=0, target_nll=initial)]
    best_step, best_nll = 0, initial
    best_state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }
    rng = np.random.default_rng(training.seed)
    scored_tokens = 0
    final_train = initial
    peak_current = peak_driver = 0
    _synchronize(device)
    started = time.perf_counter()
    model.train()
    for step in range(1, training.steps + 1):
        indices = rng.choice(len(train), size=training.batch_size, replace=False)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
        optimizer.step()
        final_train = float(loss.detach().item())
        if device.type == "mps":
            peak_current = max(peak_current, int(torch.mps.current_allocated_memory()))
            peak_driver = max(peak_driver, int(torch.mps.driver_allocated_memory()))
        if step % training.evaluation_interval == 0:
            value, _ = _nll(
                model,
                validation,
                batch_size=training.batch_size,
                context_length=model_config.context_length,
                device=device,
            )
            points.append(ValidationPoint(step=step, target_nll=value))
            if (value, step) < (best_nll, best_step):
                best_step, best_nll = step, value
                best_state = {
                    name: tensor.detach().cpu().contiguous().clone()
                    for name, tensor in model.state_dict().items()
                }
            model.train()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state, strict=True)
    checkpoint = save_checkpoint(
        model,
        output_directory=output,
        tokenizer_manifest=tokenizer_manifest,
        source_commit=source_commit,
        seed=training.seed,
        training_steps=best_step,
        initial_loss=initial,
        final_loss=best_nll,
    )
    reloaded, _manifest = load_checkpoint(
        output,
        expected_manifest_sha256=checkpoint.checksum_sha256,
        expected_tokenizer_sha256=tokenizer_manifest.checksum_sha256,
        device=device,
    )
    reloaded_nll, _ = _nll(
        reloaded,
        validation,
        batch_size=training.batch_size,
        context_length=model_config.context_length,
        device=device,
    )
    if reloaded_nll != best_nll:
        raise RuntimeError("checkpoint reload changed validation NLL")
    actual_device: Literal["cpu", "mps"] = "mps" if device.type == "mps" else "cpu"
    return Phase6ModelResult(
        experiment_id=experiment_id,
        device=actual_device,
        parameter_count=parameters,
        train_example_count=len(data.train),
        validation_example_count=len(data.validation),
        context_length=model_config.context_length,
        batch_size=training.batch_size,
        training_steps=training.steps,
        selected_step=best_step,
        initial_validation_nll=initial,
        selected_validation_nll=best_nll,
        validation_nll_reduction_fraction=(initial - best_nll) / initial,
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


def _load_inputs(
    config: Phase6Config, project_root: Path
) -> tuple[Phase5Config, Phase4Config, Any, Any, ProjectTokenizer, str]:
    phase5_path = resolve_project_path(
        project_root, config.phase6.phase5_config_path, must_exist=True
    )
    phase5 = load_phase5_config(phase5_path)
    if (
        canonical_sha256(phase5.model_dump(mode="json", round_trip=True))
        != config.phase6.phase5_config_sha256
    ):
        raise ValueError("Phase 5 config differs from the Phase 6 freeze")
    phase5_report = verify_phase5_run(phase5, project_root=project_root)
    if phase5_report.checksum_sha256 != config.phase6.phase5_report_sha256:
        raise ValueError("Phase 5 report differs from the Phase 6 freeze")
    phase4_path = resolve_project_path(
        project_root, config.phase6.phase4_config_path, must_exist=True
    )
    phase4 = load_phase4_config(phase4_path)
    phase4_report = verify_phase4_run(phase4, project_root=project_root)
    dataset_root = resolve_project_path(project_root, phase4.phase4.dataset_root, must_exist=True)
    verified = verify_development_candidate_artifact(
        ArtifactWriter(dataset_root), relative_directory=phase4.phase4.dataset_artifact_name
    )
    phase4_run = resolve_project_path(project_root, phase5.phase5.phase4_run_path, must_exist=True)
    tokenizer = ProjectTokenizer.load(
        phase4_run / "tokenizer", expected_checksum=phase4_report.tokenizer_manifest_sha256
    )
    return phase5, phase4, phase5_report, verified, tokenizer, verified.candidate.checksum_sha256


def _ablation_training_sets(verified: Any) -> tuple[set[str], set[str]]:
    renders = {item.render_id: item for item in verified.candidate.rendered_candidates}
    renderer_ids: set[str] = set()
    abstention_ids: set[str] = set()
    for example in verified.candidate.task_examples:
        if example.split_name is not SplitName.IID_TRAIN:
            continue
        aliases = {renders[render_id].alias_family_id for render_id in example.prompt_render_ids}
        if aliases and aliases.issubset({AliasFamily.CANONICAL, AliasFamily.SHORT}):
            renderer_ids.add(example.example_id)
        target = example.task_target.target
        payload = target.model_dump(mode="json", round_trip=True)
        if (
            payload.get("diagnosis_status") != DiagnosisStatus.UNRESOLVED.value
            and payload.get("immediate_action") != "INSUFFICIENT_EVIDENCE"
        ):
            abstention_ids.add(example.example_id)
    if not renderer_ids or not abstention_ids:
        raise ValueError("ablation training filter produced an empty dataset")
    return renderer_ids, abstention_ids


def _selection_directory(config: Phase6Config, project_root: Path) -> Path:
    root = resolve_project_path(project_root, config.phase6.run_root, must_exist=False)
    return root / f"{config.phase6.run_name}{SELECTION_SUFFIX}"


def run_phase6_selection(
    config: Phase6Config, *, project_root: Path, source_commit: str
) -> Phase6SelectionReport:
    if type(config) is not Phase6Config:
        raise TypeError("config must be an exact Phase6Config")
    source_commit = _source_commit(source_commit)
    phase5, phase4, phase5_report, verified, tokenizer, candidate_sha = _load_inputs(
        config, project_root
    )
    data = materialize_experiment_data(
        verified, maximum_prompt_utf8_bytes=phase5.serialization.maximum_prompt_utf8_bytes
    )
    renderer_ids, abstention_ids = _ablation_training_sets(verified)
    renderer_data = ExperimentData(
        train=tuple(item for item in data.train if item.example_id in renderer_ids),
        validation=data.validation,
        inventory_sha256=canonical_sha256(tuple(sorted(renderer_ids))),
    )
    abstention_data = ExperimentData(
        train=tuple(item for item in data.train if item.example_id in abstention_ids),
        validation=data.validation,
        inventory_sha256=canonical_sha256(tuple(sorted(abstention_ids))),
    )
    root = resolve_project_path(project_root, config.phase6.run_root, must_exist=False)
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    output = _selection_directory(config, project_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Phase 6 selection output already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=root))
    try:
        main = _train_model(
            experiment_id="E3_main_transformer",
            model_config=config.model,
            training=config.training,
            tokenizer=tokenizer,
            tokenizer_manifest=tokenizer.manifest,
            data=data,
            serialization_config=phase5,
            output=temporary / "main-checkpoint",
            source_commit=source_commit,
        )
        ablation_model = phase4.pilot_model.model_copy(update={"context_length": 512})
        renderer_training = config.training.model_copy(
            update={
                "seed": config.ablations.renderer_diversity_seed,
                "steps": config.ablations.control_steps,
                "batch_size": config.ablations.batch_size,
            }
        )
        abstention_training = renderer_training.model_copy(
            update={"seed": config.ablations.abstention_seed}
        )
        renderer = _train_model(
            experiment_id="E5_renderer_diversity_ablation",
            model_config=ablation_model,
            training=renderer_training,
            tokenizer=tokenizer,
            tokenizer_manifest=tokenizer.manifest,
            data=renderer_data,
            serialization_config=phase5,
            output=temporary / "renderer-ablation-checkpoint",
            source_commit=source_commit,
        )
        abstention = _train_model(
            experiment_id="E6_abstention_ablation",
            model_config=ablation_model,
            training=abstention_training,
            tokenizer=tokenizer,
            tokenizer_manifest=tokenizer.manifest,
            data=abstention_data,
            serialization_config=phase5,
            output=temporary / "abstention-ablation-checkpoint",
            source_commit=source_commit,
        )
        smaller_nll = phase5_report.transformer_results[0].selected_validation_nll
        thresholds = (
            main.validation_nll_reduction_fraction
            >= config.selection.minimum_validation_nll_reduction_fraction
            and main.selected_validation_nll <= config.selection.maximum_selected_validation_nll
            and (smaller_nll - main.selected_validation_nll) / smaller_nll
            >= config.selection.minimum_relative_nll_improvement_over_smaller
        )
        draft = Phase6SelectionReport.model_construct(
            source_commit=source_commit,
            phase6_config_sha256=canonical_sha256(config.model_dump(mode="json", round_trip=True)),
            phase5_report_sha256=phase5_report.checksum_sha256,
            dataset_candidate_sha256=candidate_sha,
            tokenizer_manifest_sha256=tokenizer.manifest.checksum_sha256,
            training_inventory_sha256=data.inventory_sha256,
            test_records_materialized=False,
            test_predictions_generated=False,
            all_ablation_selection_complete=True,
            e7_status="not_applicable_no_compound_iid_train_rows",
            results=(main, renderer, abstention),
            selection_thresholds_passed=thresholds,
            checksum_sha256="0" * 64,
        )
        checksum = canonical_sha256(
            draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        report = Phase6SelectionReport(
            **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
            checksum_sha256=checksum,
        )
        (temporary / "selection-report.json").write_bytes(
            canonical_json_bytes(report.model_dump(mode="json", round_trip=True)) + b"\n"
        )
        os.rename(temporary, output)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_report[T: ContractModel](
    path: Path, model: type[T], maximum: int = MAX_REPORT_BYTES
) -> T:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError("report path is missing, unsafe, or oversized")
    payload = path.read_bytes()
    _strict_json(payload)
    return model.model_validate_json(payload)


def _strict_json(payload: bytes) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError("Phase 6 JSON contains a duplicate key")
            result[key] = value
        return result

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Phase 6 JSON contains non-finite data: {value}")
        ),
    )


def verify_phase6_selection(config: Phase6Config, *, project_root: Path) -> Phase6SelectionReport:
    phase5, _phase4, phase5_report, verified, tokenizer, candidate_sha = _load_inputs(
        config, project_root
    )
    data = materialize_experiment_data(
        verified, maximum_prompt_utf8_bytes=phase5.serialization.maximum_prompt_utf8_bytes
    )
    directory = _selection_directory(config, project_root)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Phase 6 selection directory is missing or unsafe")
    expected = {
        "selection-report.json",
        "main-checkpoint",
        "renderer-ablation-checkpoint",
        "abstention-ablation-checkpoint",
    }
    if {item.name for item in directory.iterdir()} != expected:
        raise ValueError("Phase 6 selection inventory is unexpected")
    report = _load_report(directory / "selection-report.json", Phase6SelectionReport)
    if report.phase6_config_sha256 != canonical_sha256(
        config.model_dump(mode="json", round_trip=True)
    ):
        raise ValueError("selection report is bound to another Phase 6 config")
    if report.phase5_report_sha256 != phase5_report.checksum_sha256:
        raise ValueError("selection report is bound to another Phase 5 report")
    if (
        report.dataset_candidate_sha256 != candidate_sha
        or report.training_inventory_sha256 != data.inventory_sha256
    ):
        raise ValueError("selection report is bound to another dataset inventory")
    for result, name in zip(
        report.results,
        ("main-checkpoint", "renderer-ablation-checkpoint", "abstention-ablation-checkpoint"),
        strict=True,
    ):
        _model, manifest = load_checkpoint(
            directory / name,
            expected_manifest_sha256=result.checkpoint_manifest_sha256,
            expected_tokenizer_sha256=tokenizer.manifest.checksum_sha256,
            device=torch.device(result.device),
        )
        if manifest.weights_sha256 != result.checkpoint_weights_sha256:
            raise ValueError("selection checkpoint weights differ from the report")
    return report


def _raw_dataset_freeze(config: Phase6Config, phase4: Phase4Config, project_root: Path) -> None:
    directory = (
        resolve_project_path(project_root, phase4.phase4.dataset_root, must_exist=True)
        / phase4.phase4.dataset_artifact_name
    )
    expected = {
        "split-manifest.jsonl": config.test_freeze.split_manifest_raw_sha256,
        "task-examples.jsonl": config.test_freeze.task_examples_raw_sha256,
    }
    for name, digest in expected.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"frozen {name} differs from the preregistration")


def _classification_results(
    records: tuple[ExperimentExample, ...], predictions: tuple[DecodedPrediction, ...]
) -> tuple[ClassificationMetrics, ...]:
    by_id = {prediction.example_id: prediction for prediction in predictions}
    results: list[ClassificationMetrics] = []
    for task in (TaskName.FAULT_FAMILY, TaskName.NEXT_ACTION, TaskName.CONTINUE_LOG):
        selected = tuple(record for record in records if record.task_name is task)
        if not selected:
            continue
        truth = tuple(str(record.classification_label) for record in selected)
        predicted = tuple(
            by_id[record.example_id].classification_label or "__INVALID__" for record in selected
        )
        results.append(classification_metrics(task, truth, predicted))
    return tuple(results)


def _evaluate_model_split(
    experiment_id: str,
    model: TransformerLM,
    records: tuple[ExperimentExample, ...],
    *,
    tokenizer: ProjectTokenizer,
    phase5: Phase5Config,
    batch_size: int,
    device: torch.device,
    predictions: tuple[DecodedPrediction, ...] | None = None,
) -> ModelSplitEvaluation:
    tokenized, insufficient = _safe_tokenize(
        records, tokenizer, phase5, context_length=model.config.context_length
    )
    metric = None
    if tokenized:
        value, target_tokens = _nll(
            model,
            tokenized,
            batch_size=batch_size,
            context_length=model.config.context_length,
            device=device,
        )
        metric = language_model_metrics(
            sample_count=len(tokenized),
            target_token_count=target_tokens,
            negative_log_likelihood=value,
        )
    classification = () if predictions is None else _classification_results(records, predictions)
    return ModelSplitEvaluation(
        experiment_id=experiment_id,
        split_name=records[0].split_name,
        sample_count=len(records),
        scored_sample_count=len(tokenized),
        insufficient_context_by_design=len(insufficient),
        language_model=metric,
        classification=classification,
    )


def _target_object(text: str) -> dict[str, Any]:
    value = json.loads(text)
    if type(value) is not dict:
        raise ValueError("target JSON is not an object")
    return value


def _evidence_slots(value: dict[str, Any]) -> tuple[str, ...]:
    slots = value.get("evidence_slots", ())
    if type(slots) is not list:
        return ()
    return tuple(str(item) for item in slots)


def _main_metrics(
    split: SplitName,
    records: tuple[ExperimentExample, ...],
    predictions: tuple[DecodedPrediction, ...],
    config: Phase6Config,
) -> MainPredictionMetrics:
    by_id = {prediction.example_id: prediction for prediction in predictions}
    ordered = tuple(by_id[record.example_id] for record in records)
    exact = tuple(
        prediction.predicted_target_json == record.target_text
        for record, prediction in zip(records, ordered, strict=True)
    )
    evidence_records = tuple(
        record for record in records if record.task_name is TaskName.EXTRACT_EVIDENCE
    )
    evidence = None
    if evidence_records:
        evidence = set_f1_metrics(
            tuple(
                _evidence_slots(_target_object(record.target_text)) for record in evidence_records
            ),
            tuple(
                _evidence_slots(_target_object(predicted)) if predicted is not None else ()
                for record in evidence_records
                for predicted in (by_id[record.example_id].predicted_target_json,)
            ),
        )
    no_fault = tuple(
        record
        for record in records
        if record.task_name is TaskName.FAULT_FAMILY
        and _target_object(record.target_text).get("diagnosis_status") == "NO_FAULT"
    )
    false_positives = sum(
        (by_id[record.example_id].classification_label or "").startswith("DIAGNOSED:")
        for record in no_fault
    )
    required = tuple(
        record
        for record in records
        if (
            _target_object(record.target_text).get("diagnosis_status") == "UNRESOLVED"
            or _target_object(record.target_text).get("immediate_action") == "INSUFFICIENT_EVIDENCE"
        )
    )
    required_correct = sum(
        by_id[record.example_id].predicted_target_json == record.target_text for record in required
    )
    return MainPredictionMetrics(
        split_name=split,
        sample_count=len(records),
        parse_success_rate=sum(item.json_parse_success for item in ordered) / len(ordered),
        schema_validity_rate=sum(item.schema_valid for item in ordered) / len(ordered),
        exact_match_rate=sum(exact) / len(exact),
        exact_match_interval=bootstrap_mean_interval(
            tuple(float(item) for item in exact),
            resamples=config.evaluation.bootstrap_resamples,
            seed=config.evaluation.bootstrap_seed,
            confidence_level=config.evaluation.confidence_level,
        ),
        evidence=evidence,
        calibration=calibration_metrics(
            exact,
            tuple(float(item.confidence) for item in ordered),
            bin_count=config.decoder.calibration_bins,
            selective_coverage=config.evaluation.selective_risk_coverage,
        ),
        no_fault_sample_count=len(no_fault),
        no_fault_false_positive_rate=false_positives / len(no_fault) if no_fault else 0.0,
        required_abstention_sample_count=len(required),
        required_abstention_accuracy=required_correct / len(required) if required else 1.0,
    )


def _failure_category(record: ExperimentExample, prediction: DecodedPrediction) -> str:
    if prediction.generation_truncated:
        return "generation_truncated"
    if not prediction.json_parse_success:
        return "invalid_json"
    if not prediction.schema_valid:
        return "schema_invalid"
    truth = _target_object(record.target_text)
    predicted = _target_object(prediction.predicted_target_json or "{}")
    if (
        truth.get("diagnosis_status") == "UNRESOLVED"
        and predicted.get("diagnosis_status") != "UNRESOLVED"
    ):
        return "failed_abstention"
    if record.task_name is TaskName.FAULT_FAMILY:
        return "fault_family_error"
    if record.task_name is TaskName.NEXT_ACTION:
        return "next_action_error"
    if record.task_name is TaskName.EXTRACT_EVIDENCE:
        return "evidence_error"
    if record.task_name is TaskName.CONTINUE_LOG:
        return "continuation_error"
    if record.task_name is TaskName.COUNTERFACTUAL_COMPARE:
        return "counterfactual_error"
    return "summary_error"


def _failure_gallery(
    records: tuple[ExperimentExample, ...], predictions: tuple[DecodedPrediction, ...]
) -> tuple[FailureGalleryEntry, ...]:
    by_id = {prediction.example_id: prediction for prediction in predictions}
    selected: dict[str, FailureGalleryEntry] = {}
    for record in records:
        prediction = by_id[record.example_id]
        if prediction.predicted_target_json == record.target_text:
            continue
        category = _failure_category(record, prediction)
        if category not in selected:
            selected[category] = FailureGalleryEntry(
                category=category,
                split_name=record.split_name,
                example_id=record.example_id,
                task_name=record.task_name,
                expected_target=record.target_text,
                predicted_target=prediction.predicted_target_json
                or prediction.generated_text
                or "<empty>",
            )
    return tuple(selected[key] for key in sorted(selected))


def _acceptance_checks(
    config: Phase6Config,
    *,
    baselines: tuple[BaselineSplitResult, ...],
    model_results: tuple[ModelSplitEvaluation, ...],
    prediction_metrics: tuple[MainPredictionMetrics, ...],
    selection_thresholds_passed: bool,
) -> tuple[tuple[str, bool], ...]:
    """Apply the frozen acceptance policy to one complete evaluation result graph."""

    checks = {
        "golden_suite_approved": True,
        "heldout_access_after_selection": True,
        "main_selection_thresholds": selection_thresholds_passed,
        "parse_success": all(
            result.parse_success_rate >= config.evaluation.minimum_parse_success_rate
            for result in prediction_metrics
        ),
        "schema_validity": all(
            result.schema_validity_rate >= config.evaluation.minimum_schema_validity_rate
            for result in prediction_metrics
        ),
        "evidence_f1": all(
            result.evidence is None or result.evidence.f1 >= config.evaluation.minimum_evidence_f1
            for result in prediction_metrics
        ),
        "no_fault_false_positive": all(
            result.no_fault_false_positive_rate
            <= config.evaluation.maximum_no_fault_false_positive_rate
            for result in prediction_metrics
        ),
        "required_abstention": all(
            result.required_abstention_accuracy
            >= config.evaluation.minimum_required_abstention_accuracy
            for result in prediction_metrics
        ),
        "calibration": all(
            result.calibration.expected_calibration_error
            <= config.evaluation.maximum_expected_calibration_error
            for result in prediction_metrics
        ),
        "selective_risk": all(
            result.calibration.selective_risk <= config.evaluation.maximum_selective_risk
            for result in prediction_metrics
        ),
    }
    for split in TEST_SPLITS:
        if split is SplitName.COMPOSITION_TEST:
            continue
        main_result = next(
            item
            for item in model_results
            if item.experiment_id == "E3_main_transformer" and item.split_name is split
        )
        main_by_task = {item.task_name: item for item in main_result.classification}
        split_baselines = tuple(item.result for item in baselines if item.split_name is split)
        for task, margin in (
            (
                TaskName.FAULT_FAMILY,
                config.evaluation.minimum_fault_macro_f1_margin_over_best_simple,
            ),
            (
                TaskName.NEXT_ACTION,
                config.evaluation.minimum_next_action_macro_f1_margin_over_best_simple,
            ),
        ):
            if task not in main_by_task:
                continue
            simple = tuple(
                item.classification.macro_f1
                for item in split_baselines
                if item.classification is not None
                and item.classification.task_name is task
                and item.baseline_name
                in {
                    "majority_frequency",
                    "deterministic_keyword_rules",
                    "bag_of_words_logistic_regression",
                }
            )
            checks[f"{split.value}:{task.value}:simple_margin"] = bool(
                simple and main_by_task[task].macro_f1 >= max(simple) + margin
            )
        if TaskName.CONTINUE_LOG in main_by_task:
            checks[f"{split.value}:continue_log:minimum_macro_f1"] = (
                main_by_task[TaskName.CONTINUE_LOG].macro_f1
                >= config.evaluation.minimum_continue_log_macro_f1
            )
        trigram = next(
            (
                item.language_model
                for item in split_baselines
                if item.baseline_name == "token_trigram_additive"
            ),
            None,
        )
        if trigram is not None and main_result.language_model is not None:
            checks[f"{split.value}:target_nll:trigram_fraction"] = (
                main_result.language_model.negative_log_likelihood
                <= config.evaluation.maximum_target_nll_fraction_of_trigram
                * trigram.negative_log_likelihood
            )
    return tuple(sorted(checks.items()))


def _write_jsonl(path: Path, values: tuple[ContractModel, ...]) -> str:
    payload = b"".join(
        canonical_json_bytes(value.model_dump(mode="json", round_trip=True)) + b"\n"
        for value in values
    )
    if len(payload) > MAX_PREDICTIONS_BYTES:
        raise ValueError("prediction artifact exceeds the configured safety bound")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _comparison_prediction(
    experiment_id: str, prediction: DecodedPrediction
) -> ExperimentDecodedPrediction:
    draft = ExperimentDecodedPrediction.model_construct(
        experiment_id=experiment_id,
        prediction=prediction,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return ExperimentDecodedPrediction(
        experiment_id=experiment_id,
        prediction=prediction,
        checksum_sha256=checksum,
    )


def _read_predictions(path: Path, *, expected_count: int) -> tuple[DecodedPrediction, ...]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_PREDICTIONS_BYTES:
        raise ValueError("prediction artifact is missing, unsafe, or oversized")
    records: list[DecodedPrediction] = []
    for line in path.read_bytes().splitlines():
        if not line:
            raise ValueError("prediction artifact contains an empty JSONL row")
        _strict_json(line)
        records.append(DecodedPrediction.model_validate_json(line))
    if len(records) != expected_count or len({item.example_id for item in records}) != len(records):
        raise ValueError("prediction artifact count or example IDs are invalid")
    return tuple(records)


def _read_comparison_predictions(
    path: Path, *, expected_count: int
) -> tuple[ExperimentDecodedPrediction, ...]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_PREDICTIONS_BYTES:
        raise ValueError("comparison prediction artifact is missing, unsafe, or oversized")
    lines = path.read_bytes().splitlines()
    if any(not line for line in lines):
        raise ValueError("comparison prediction artifact contains an empty JSONL row")
    for line in lines:
        _strict_json(line)
    records = tuple(ExperimentDecodedPrediction.model_validate_json(line) for line in lines)
    keys = tuple((item.experiment_id, item.prediction.example_id) for item in records)
    if len(records) != expected_count or len(keys) != len(set(keys)):
        raise ValueError("comparison prediction count or composite IDs are invalid")
    return records


def _heldout_access_path(config: Phase6Config, project_root: Path) -> Path:
    root = resolve_project_path(project_root, config.phase6.run_root, must_exist=False)
    return root / f"{config.phase6.run_name}-heldout-access.json"


def _authorize_heldout_access(
    config: Phase6Config,
    *,
    project_root: Path,
    source_commit: str,
    selection: Phase6SelectionReport,
    golden_record_sha256: str,
) -> HeldoutAccessRecord:
    draft = HeldoutAccessRecord.model_construct(
        source_commit=source_commit,
        phase6_config_sha256=canonical_sha256(config.model_dump(mode="json", round_trip=True)),
        selection_report_sha256=selection.checksum_sha256,
        golden_review_record_sha256=golden_record_sha256,
        access_count=1,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    expected = HeldoutAccessRecord(
        source_commit=source_commit,
        phase6_config_sha256=canonical_sha256(config.model_dump(mode="json", round_trip=True)),
        selection_report_sha256=selection.checksum_sha256,
        golden_review_record_sha256=golden_record_sha256,
        access_count=1,
        checksum_sha256=checksum,
    )
    path = _heldout_access_path(config, project_root)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    if path.exists() or path.is_symlink():
        observed = _load_report(path, HeldoutAccessRecord, maximum=1024 * 1024)
        if observed != expected:
            raise ValueError("held-out access is already bound to another evaluation")
        return observed
    path.write_bytes(
        canonical_json_bytes(expected.model_dump(mode="json", round_trip=True)) + b"\n"
    )
    return expected


def _baseline_split(
    train: tuple[ExperimentExample, ...],
    records: tuple[ExperimentExample, ...],
    *,
    tokenizer: ProjectTokenizer,
    phase5: Phase5Config,
) -> tuple[BaselineResult, ...]:
    data = ExperimentData(
        train=train,
        validation=records,
        inventory_sha256=canonical_sha256(tuple(item.example_id for item in train + records)),
    )
    token_train, train_missing = _safe_tokenize(
        train, tokenizer, phase5, context_length=phase5.serialization.model_context_length
    )
    token_test, _test_missing = _safe_tokenize(
        records, tokenizer, phase5, context_length=phase5.serialization.model_context_length
    )
    if train_missing:
        raise ValueError("baseline training target exceeds context")
    if not token_test:
        return ()
    present_tasks = {record.task_name for record in records}
    required_classification = {
        TaskName.FAULT_FAMILY,
        TaskName.NEXT_ACTION,
        TaskName.CONTINUE_LOG,
    }
    if not required_classification.issubset(present_tasks):
        return (
            _token_ngram_language_model(
                token_train,
                token_test,
                order=phase5.baselines.ngram_order,
                alpha=phase5.baselines.ngram_additive_smoothing,
                vocab_size=tokenizer.vocab_size,
            ),
        )
    return run_preregistered_baselines(
        data,
        tokenizer,
        phase5.baselines,
        tokenized_train=token_train,
        tokenized_validation=token_test,
    )


def _event_order_records(records: tuple[ExperimentExample, ...]) -> tuple[ExperimentExample, ...]:
    transformed: list[ExperimentExample] = []
    for record in records:
        lines = record.prompt_text.splitlines()
        output: list[str] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            marker = line[:7] if line.startswith("[T+") and len(line) >= 7 else None
            if marker is None:
                output.append(line)
                index += 1
                continue
            end = index + 1
            while end < len(lines) and lines[end].startswith(marker):
                end += 1
            output.extend(reversed(lines[index:end]))
            index = end
        transformed.append(
            ExperimentExample(
                example_id=record.example_id,
                split_name=record.split_name,
                task_name=record.task_name,
                prompt_text="\n".join(output),
                target_text=record.target_text,
                classification_label=record.classification_label,
                source_checksum_sha256=record.source_checksum_sha256,
            )
        )
    return tuple(transformed)


def _golden_label(task: TaskName, target: Any) -> str | None:
    if task is TaskName.FAULT_FAMILY:
        if type(target) is not FaultDiagnosisTarget:
            raise TypeError("golden fault target has the wrong exact type")
        if target.diagnosis_status is not DiagnosisStatus.DIAGNOSED:
            return target.diagnosis_status.value
        return "DIAGNOSED:" + "+".join(label.value for label in target.fault_labels)
    if task is TaskName.NEXT_ACTION:
        if type(target) is not NextActionTarget:
            raise TypeError("golden action target has the wrong exact type")
        return target.immediate_action.value
    return None


def _golden_examples(config: Phase6Config, project_root: Path) -> tuple[ExperimentExample, ...]:
    packet = load_golden_review_packet(
        resolve_project_path(project_root, config.phase6.golden_packet_path, must_exist=True)
    )
    examples: list[ExperimentExample] = []
    for case in packet.cases:
        trace = generate_trace(case.scenario)
        trace_payload = {
            "scenario": trace.scenario.model_dump(mode="json", round_trip=True),
            "latent_states": tuple(
                state.model_dump(mode="json", round_trip=True) for state in trace.latent_states
            ),
            "observations": tuple(
                frame.model_dump(mode="json", round_trip=True) for frame in trace.observations
            ),
            "events": tuple(
                event.model_dump(mode="json", round_trip=True) for event in trace.events
            ),
            "targets": trace.targets.model_dump(mode="json", round_trip=True),
        }
        if (
            canonical_sha256(trace_payload) != case.trace_sha256
            or trace.targets != case.expected_targets
        ):
            raise ValueError(f"{case.case_id} regenerated trace differs from owner-approved truth")
        decision_tick = trace.targets.decisions[-1].decision_tick
        for task in (
            TaskName.FAULT_FAMILY,
            TaskName.EXTRACT_EVIDENCE,
            TaskName.NEXT_ACTION,
            TaskName.INCIDENT_SUMMARY,
        ):
            trajectory_id = f"golden-{case.case_id.casefold()}-{task.value.replace('_', '-')}"
            provenance = ProvenanceRecord(
                dataset_version="0.1.0",
                generator_commit=packet.generator_commit,
                renderer_version="0.1.0",
                seed=trace.scenario.seed,
                trajectory_id=trajectory_id,
                scenario_id=trace.scenario.scenario_id,
                plant_variant_id=trace.scenario.plant_variant_id,
                fault_family_ids=tuple(
                    injection.fault_family for injection in trace.scenario.fault_injections
                ),
                template_family_ids=("compact-log-v1",),
                split_name=SplitName.IID_TEST,
                task_name=task,
            )
            trajectory = StructuredTrajectory(
                trajectory_id=trajectory_id,
                scenario_id=trace.scenario.scenario_id,
                scenario=trace.scenario,
                provenance=provenance,
                latent_states=trace.latent_states,
                observations=trace.observations,
                events=trace.events,
                targets=trace.targets,
            )
            projection = project_trajectory(
                trajectory,
                decision_tick=decision_tick,
                task_name=task,
                view=infer_projection_view(trajectory),
            )
            rendered = render_model_input(
                projection.model_input,
                template_family=TemplateFamily.COMPACT_LOG,
                alias_family=AliasFamily.CANONICAL,
                split_name=SplitName.IID_TEST,
            )
            target = projection.task_target.target
            examples.append(
                ExperimentExample(
                    example_id=f"golden:{case.case_id}:{task.value}",
                    split_name=SplitName.IID_TEST,
                    task_name=task,
                    prompt_text=rendered.text,
                    target_text=canonical_json_bytes(
                        target.model_dump(mode="json", round_trip=True)
                    ).decode("utf-8"),
                    classification_label=_golden_label(task, target),
                    source_checksum_sha256=projection.checksum_sha256,
                )
            )
    if len(examples) != 60:
        raise RuntimeError("golden projection must produce exactly sixty examples")
    return tuple(examples)


def run_phase6_evaluation(
    config: Phase6Config, *, project_root: Path, source_commit: str
) -> Phase6EvaluationReport:
    if type(config) is not Phase6Config:
        raise TypeError("config must be an exact Phase6Config")
    source_commit = _source_commit(source_commit)
    selection = verify_phase6_selection(config, project_root=project_root)
    if selection.source_commit != source_commit:
        raise ValueError("selection and evaluation must use the same implementation commit")
    phase5, phase4, phase5_report, verified, tokenizer, _candidate_sha = _load_inputs(
        config, project_root
    )
    packet_path = resolve_project_path(
        project_root, config.phase6.golden_packet_path, must_exist=True
    )
    record_path = resolve_project_path(
        project_root, config.phase6.golden_review_record_path, must_exist=True
    )
    packet = load_golden_review_packet(packet_path)
    review = load_golden_review_record(record_path)
    verify_golden_review(packet, review, expected_packet_sha256=config.phase6.golden_packet_sha256)
    _authorize_heldout_access(
        config,
        project_root=project_root,
        source_commit=source_commit,
        selection=selection,
        golden_record_sha256=review.record_sha256,
    )
    _raw_dataset_freeze(config, phase4, project_root)
    # This is the single held-out materialization point in the Phase 6 workflow.
    data = materialize_phase6_data(
        verified,
        freeze=config.test_freeze,
        maximum_prompt_utf8_bytes=phase5.serialization.maximum_prompt_utf8_bytes,
    )
    selection_dir = _selection_directory(config, project_root)
    main, _ = load_checkpoint(
        selection_dir / "main-checkpoint",
        expected_manifest_sha256=selection.results[0].checkpoint_manifest_sha256,
        expected_tokenizer_sha256=tokenizer.manifest.checksum_sha256,
        device=torch.device(selection.results[0].device),
    )
    renderer, _ = load_checkpoint(
        selection_dir / "renderer-ablation-checkpoint",
        expected_manifest_sha256=selection.results[1].checkpoint_manifest_sha256,
        expected_tokenizer_sha256=tokenizer.manifest.checksum_sha256,
        device=torch.device(selection.results[1].device),
    )
    abstention, _ = load_checkpoint(
        selection_dir / "abstention-ablation-checkpoint",
        expected_manifest_sha256=selection.results[2].checkpoint_manifest_sha256,
        expected_tokenizer_sha256=tokenizer.manifest.checksum_sha256,
        device=torch.device(selection.results[2].device),
    )
    phase5_dir = (
        resolve_project_path(project_root, phase5.phase5.run_root, must_exist=True)
        / phase5.phase5.run_name
    )
    smaller, _ = load_checkpoint(
        phase5_dir / "smaller-checkpoint",
        expected_manifest_sha256=phase5_report.transformer_results[0].checkpoint_manifest_sha256,
        expected_tokenizer_sha256=tokenizer.manifest.checksum_sha256,
        device=torch.device(phase5_report.transformer_results[0].device),
    )
    pilot, _ = load_checkpoint(
        phase5_dir / "pilot-checkpoint",
        expected_manifest_sha256=phase5_report.transformer_results[1].checkpoint_manifest_sha256,
        expected_tokenizer_sha256=tokenizer.manifest.checksum_sha256,
        device=torch.device(phase5_report.transformer_results[1].device),
    )
    train = data.by_split[SplitName.IID_TRAIN]
    baselines: list[BaselineSplitResult] = []
    model_results: list[ModelSplitEvaluation] = []
    prediction_metrics: list[MainPredictionMetrics] = []
    predictions: list[DecodedPrediction] = []
    comparison_predictions: list[ExperimentDecodedPrediction] = []
    for split in TEST_SPLITS:
        records = data.by_split[split]
        baselines.extend(
            BaselineSplitResult(split_name=split, result=result)
            for result in _baseline_split(train, records, tokenizer=tokenizer, phase5=phase5)
        )
        decoded = greedy_decode_predictions(
            main,
            tokenizer,
            records,
            phase5.serialization,
            maximum_generated_tokens=config.decoder.maximum_generated_target_tokens,
            batch_size=config.training.batch_size,
            device=torch.device(selection.results[0].device),
        )
        predictions.extend(decoded)
        prediction_metrics.append(_main_metrics(split, records, decoded, config))
        model_results.append(
            _evaluate_model_split(
                "E3_main_transformer",
                main,
                records,
                tokenizer=tokenizer,
                phase5=phase5,
                batch_size=config.training.batch_size,
                device=torch.device(selection.results[0].device),
                predictions=decoded,
            )
        )
        for experiment_id, model, result in (
            ("E2_smaller_transformer", smaller, phase5_report.transformer_results[0]),
            ("E2_pilot_transformer", pilot, phase5_report.transformer_results[1]),
            ("E5_renderer_diversity_ablation", renderer, selection.results[1]),
            ("E6_abstention_ablation", abstention, selection.results[2]),
        ):
            comparison_decoded = greedy_decode_predictions(
                model,
                tokenizer,
                records,
                phase5.serialization,
                maximum_generated_tokens=config.decoder.maximum_generated_target_tokens,
                batch_size=result.batch_size,
                device=torch.device(result.device),
            )
            comparison_predictions.extend(
                _comparison_prediction(experiment_id, item) for item in comparison_decoded
            )
            model_results.append(
                _evaluate_model_split(
                    experiment_id,
                    model,
                    records,
                    tokenizer=tokenizer,
                    phase5=phase5,
                    batch_size=result.batch_size,
                    device=torch.device(result.device),
                    predictions=comparison_decoded,
                )
            )
        reordered = _event_order_records(records)
        reordered_decoded = greedy_decode_predictions(
            main,
            tokenizer,
            reordered,
            phase5.serialization,
            maximum_generated_tokens=config.decoder.maximum_generated_target_tokens,
            batch_size=config.training.batch_size,
            device=torch.device(selection.results[0].device),
        )
        comparison_predictions.extend(
            _comparison_prediction("E4_event_order_ablation", item) for item in reordered_decoded
        )
        model_results.append(
            _evaluate_model_split(
                "E4_event_order_ablation",
                main,
                reordered,
                tokenizer=tokenizer,
                phase5=phase5,
                batch_size=config.training.batch_size,
                device=torch.device(selection.results[0].device),
                predictions=reordered_decoded,
            )
        )
    golden_examples = _golden_examples(config, project_root)
    golden_predictions = greedy_decode_predictions(
        main,
        tokenizer,
        golden_examples,
        phase5.serialization,
        maximum_generated_tokens=config.decoder.maximum_generated_target_tokens,
        batch_size=config.training.batch_size,
        device=torch.device(selection.results[0].device),
    )
    golden_exact = sum(
        prediction.predicted_target_json == record.target_text
        for record, prediction in zip(golden_examples, golden_predictions, strict=True)
    ) / len(golden_examples)
    all_records = tuple(record for split in TEST_SPLITS for record in data.by_split[split])
    gallery = _failure_gallery(all_records, tuple(predictions))
    checks = _acceptance_checks(
        config,
        baselines=tuple(baselines),
        model_results=tuple(model_results),
        prediction_metrics=tuple(prediction_metrics),
        selection_thresholds_passed=selection.selection_thresholds_passed,
    )
    negative = tuple(name for name, passed in checks if not passed)
    run_root = resolve_project_path(project_root, config.phase6.run_root, must_exist=False)
    output = run_root / config.phase6.run_name
    if output.exists() or output.is_symlink():
        raise FileExistsError("Phase 6 evaluation output already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=run_root))
    try:
        predictions_sha = _write_jsonl(temporary / "predictions.jsonl", tuple(predictions))
        comparison_sha = _write_jsonl(
            temporary / "comparison-predictions.jsonl", tuple(comparison_predictions)
        )
        golden_sha = _write_jsonl(temporary / "golden-predictions.jsonl", tuple(golden_predictions))
        draft = Phase6EvaluationReport.model_construct(
            source_commit=source_commit,
            phase6_config_sha256=canonical_sha256(config.model_dump(mode="json", round_trip=True)),
            selection_report_sha256=selection.checksum_sha256,
            golden_packet_sha256=packet.packet_sha256,
            golden_review_record_sha256=review.record_sha256,
            heldout_access_count=1,
            heldout_access_after_selection=True,
            test_example_count=894,
            experiment_matrix=config.experiments.required,
            baseline_results=tuple(baselines),
            model_split_results=tuple(model_results),
            main_prediction_metrics=tuple(prediction_metrics),
            golden_case_count=15,
            golden_example_count=len(golden_examples),
            golden_exact_match_rate=golden_exact,
            acceptance_checks=checks,
            negative_results=negative,
            failure_gallery=gallery,
            predictions_sha256=predictions_sha,
            comparison_predictions_sha256=comparison_sha,
            golden_predictions_sha256=golden_sha,
            checksum_sha256="0" * 64,
        )
        checksum = canonical_sha256(
            draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        report = Phase6EvaluationReport(
            **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
            checksum_sha256=checksum,
        )
        (temporary / "report.json").write_bytes(
            canonical_json_bytes(report.model_dump(mode="json", round_trip=True)) + b"\n"
        )
        if (temporary / "report.json").stat().st_size > MAX_REPORT_BYTES:
            raise ValueError("Phase 6 report exceeds its size limit")
        os.rename(temporary, output)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_phase6_evaluation(config: Phase6Config, *, project_root: Path) -> Phase6EvaluationReport:
    selection = verify_phase6_selection(config, project_root=project_root)
    directory = (
        resolve_project_path(project_root, config.phase6.run_root, must_exist=True)
        / config.phase6.run_name
    )
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Phase 6 evaluation directory is missing or unsafe")
    expected = {
        "report.json",
        "predictions.jsonl",
        "comparison-predictions.jsonl",
        "golden-predictions.jsonl",
    }
    if {item.name for item in directory.iterdir()} != expected:
        raise ValueError("Phase 6 evaluation inventory is unexpected")
    report = _load_report(directory / "report.json", Phase6EvaluationReport)
    if report.selection_report_sha256 != selection.checksum_sha256:
        raise ValueError("evaluation report is bound to another selection report")
    if _sha256(directory / "predictions.jsonl") != report.predictions_sha256:
        raise ValueError("prediction artifact checksum mismatch")
    if _sha256(directory / "comparison-predictions.jsonl") != report.comparison_predictions_sha256:
        raise ValueError("comparison prediction artifact checksum mismatch")
    if _sha256(directory / "golden-predictions.jsonl") != report.golden_predictions_sha256:
        raise ValueError("golden prediction artifact checksum mismatch")
    _read_predictions(directory / "predictions.jsonl", expected_count=report.test_example_count)
    _read_comparison_predictions(
        directory / "comparison-predictions.jsonl",
        expected_count=report.test_example_count * 5,
    )
    _read_predictions(
        directory / "golden-predictions.jsonl", expected_count=report.golden_example_count
    )
    access = _load_report(
        _heldout_access_path(config, project_root), HeldoutAccessRecord, maximum=1024 * 1024
    )
    if (
        access.selection_report_sha256 != selection.checksum_sha256
        or access.golden_review_record_sha256 != report.golden_review_record_sha256
    ):
        raise ValueError("held-out access record differs from the evaluation report")
    if report.phase6_config_sha256 != canonical_sha256(
        config.model_dump(mode="json", round_trip=True)
    ):
        raise ValueError("evaluation report is bound to another Phase 6 config")
    return report


__all__ = [
    "BaselineSplitResult",
    "ExperimentDecodedPrediction",
    "HeldoutAccessRecord",
    "MainPredictionMetrics",
    "ModelSplitEvaluation",
    "Phase6EvaluationReport",
    "Phase6ModelResult",
    "Phase6SelectionReport",
    "run_phase6_evaluation",
    "run_phase6_selection",
    "verify_phase6_evaluation",
    "verify_phase6_selection",
]
