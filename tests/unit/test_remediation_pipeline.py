"""Focused tests for the concrete, development-only remediation pipeline adapter."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Literal, cast

import pytest
import torch
from pydantic import BaseModel, ValidationError

import reactorbench.remediation.data as remediation_data
import reactorbench.remediation.pipeline as pipeline
from reactorbench.dataset.catalog import AliasFamily, TemplateFamily
from reactorbench.evaluation.compact import CompactTargetContext, compact_output_contract
from reactorbench.model import TransformerConfig, TransformerLM
from reactorbench.model.checkpoint import CheckpointManifest
from reactorbench.remediation.acceptance import (
    DevelopmentArtifactBinding,
    DevelopmentView,
    V03AcceptanceResult,
)
from reactorbench.remediation.config import (
    PIPELINE_STAGES,
    SHADOW_VIEWS,
    PipelineConfig,
    RemediationTraining,
    RemediationView,
    SemanticSelectionPolicy,
    config_sha256,
    load_pipeline_config,
    load_v03_config,
)
from reactorbench.remediation.data import (
    FrozenV03IIDMaterial,
    RemediationExample,
    SafeDevelopmentDataset,
    TaskScopedStructuredFingerprint,
)
from reactorbench.remediation.decoding import (
    CompactPathPrediction,
    DecodePath,
    DualPathCompactPrediction,
)
from reactorbench.remediation.inventory import (
    CompactInventoryReport,
    CounterfactualCapExtensionReport,
)
from reactorbench.remediation.metrics import (
    SemanticEvaluationReport,
    canonical_prediction_jsonl_bytes,
    prediction_artifact_byte_sha256,
)
from reactorbench.remediation.orchestration import (
    ArtifactReference,
    PipelineEngine,
    PipelineStageError,
    PipelineState,
    PipelineStore,
    StageAction,
    StageContext,
    StageOutcome,
    StageStatus,
)
from reactorbench.remediation.progress import (
    ProgressEventKind,
    ProgressMetric,
    ProgressSnapshot,
    ProgressState,
)
from reactorbench.remediation.selection import SemanticSelectionManifest
from reactorbench.remediation.serialization import CompactTokenizedExample
from reactorbench.remediation.training import (
    CompactTrainingResult,
    DeviceResolution,
    TrainingProgress,
    durable_training_state_upper_bound_bytes,
    selected_checkpoint_upper_bound_bytes,
    tokenized_inventory_sha256,
)
from reactorbench.schemas.base import canonical_json_bytes, canonical_sha256
from reactorbench.schemas.enums import ActionLabel, SplitName, TaskName
from reactorbench.schemas.target import NextActionTarget
from reactorbench.tokenizer import ProjectTokenizer

SOURCE_COMMIT = "abcdef0"
FULL_SOURCE_COMMIT = "a" * 40
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def test_semantic_checkpoint_selector_uses_floor_before_validation_nll() -> None:
    root = Path(__file__).resolve().parents[2]
    historical = load_v03_config(
        root / "configs/experiments/phase6-remediation-v0.3.2-focused.toml"
    ).selection
    hierarchical = load_v03_config(
        root / "configs/experiments/phase6-remediation-v0.3.3-hierarchical.toml"
    ).selection

    assert pipeline._semantic_checkpoint_selection_score(historical, composite=0.75) == 0.25
    assert pipeline._semantic_checkpoint_selection_score(hierarchical, composite=0.80) == 0.0
    assert pipeline._semantic_checkpoint_selection_score(hierarchical, composite=0.70) == 1.05

    historical_candidates = (
        pipeline.CandidateScore(
            candidate_id="higher-semantic",
            checkpoint_manifest_sha256=HASH_A,
            semantic_composite=0.80,
            selected_validation_nll=0.40,
            selected_step=10,
            evaluation_report_sha256=HASH_B,
        ),
        pipeline.CandidateScore(
            candidate_id="lower-nll",
            checkpoint_manifest_sha256=HASH_C,
            semantic_composite=0.79,
            selected_validation_nll=0.10,
            selected_step=5,
            evaluation_report_sha256=HASH_D,
        ),
    )
    assert (
        pipeline._select_v03_candidate(historical, historical_candidates).candidate_id
        == "higher-semantic"
    )
    assert (
        pipeline._select_v03_candidate(hierarchical, historical_candidates[:1]).candidate_id
        == "higher-semantic"
    )
    with pytest.raises(pipeline.PipelineExecutionError, match="single frozen candidate"):
        pipeline._select_v03_candidate(hierarchical, historical_candidates)

    missing_floor = SemanticSelectionPolicy.model_construct(
        **hierarchical.model_dump(mode="python", exclude={"minimum_checkpoint_semantic_composite"}),
        minimum_checkpoint_semantic_composite=None,
    )
    with pytest.raises(pipeline.PipelineExecutionError, match="lacks its frozen floor"):
        pipeline._semantic_checkpoint_selection_score(missing_floor, composite=0.75)
    with pytest.raises(ValueError, match="finite probability"):
        pipeline._semantic_checkpoint_selection_score(hierarchical, composite=float("nan"))
    with pytest.raises(TypeError, match="exact semantic policy"):
        pipeline._semantic_checkpoint_selection_score(
            cast(SemanticSelectionPolicy, object()), composite=0.75
        )


def test_targeted_v02_prefix_reuse_reopens_checksum_bound_evidence() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_pipeline_config(
        project_root / "configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-01.toml"
    )
    evidence = pipeline.verify_v02_prefix_reuse(project_root, config)

    assert len(evidence) == 21
    assert len(evidence) == len(set(evidence))
    assert all(path.is_file() and not path.is_symlink() for path in evidence)

    assert config.reuse_v02_prefix is not None
    bad_policy = config.reuse_v02_prefix.model_copy(update={"source_run_manifest_sha256": "0" * 64})
    bad_config = config.model_copy(update={"reuse_v02_prefix": bad_policy})
    with pytest.raises(pipeline.PipelineExecutionError, match="provenance"):
        pipeline.verify_v02_prefix_reuse(project_root, bad_config)


def _config(run_name: str = "pipeline-test") -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "pipeline_version": "0.4.0",
            "run_name": run_name,
            "run_root": "work",
            "v02_config_path": "configs/v02.toml",
            "v02_config_sha256": HASH_A,
            "v03_config_path": "configs/v03.toml",
            "v03_config_sha256": HASH_B,
            "v04_config_path": "configs/v04.toml",
            "v04_config_sha256": HASH_C,
            "stage_order": list(PIPELINE_STAGES),
            "heartbeat_interval_seconds": 5,
            "maximum_status_bytes": 64 * 1024,
            "maximum_event_log_bytes": 1024 * 1024,
            "maximum_pipeline_seconds": 3600,
            "maximum_run_bytes": 1024 * 1024,
            "maximum_process_rss_bytes": 256 * 1024**2,
            "stop_before_final_evaluation": True,
        }
    )


def _store(root: Path, config: PipelineConfig) -> PipelineStore:
    return PipelineStore.create(
        project_root=root,
        config=config,
        pipeline_config_sha256=config_sha256(config),
        source_commit=SOURCE_COMMIT,
        command=("python", "-m", "reactorbench.remediation"),
    )


def _success(_context: StageContext) -> StageOutcome:
    return StageOutcome(summary="Development stage completed.")


def _actions(overrides: Mapping[str, StageAction] | None = None) -> dict[str, StageAction]:
    selected = {} if overrides is None else dict(overrides)
    return {stage: selected.get(stage, _success) for stage in PIPELINE_STAGES}


def _engine(
    root: Path,
    config: PipelineConfig,
    store: PipelineStore,
    *,
    overrides: Mapping[str, StageAction] | None = None,
    stop_requested: Callable[[], bool] = lambda: False,
) -> PipelineEngine:
    return PipelineEngine(
        project_root=root,
        config=config,
        store=store,
        actions=_actions(overrides),
        stop_requested=stop_requested,
    )


def _terminal_state(root: Path, kind: str) -> tuple[PipelineConfig, PipelineState]:
    config = _config(run_name=f"terminal-{kind}")
    store = _store(root, config)
    if kind == "completed":
        state = _engine(root, config, store).run()
    elif kind == "blocked":

        def blocked(_context: StageContext) -> StageOutcome:
            return StageOutcome(summary="Development gate blocked.", advancement_allowed=False)

        state = _engine(
            root,
            config,
            store,
            overrides={PIPELINE_STAGES[1]: blocked},
        ).run()
    elif kind == "failed":

        def failed(_context: StageContext) -> StageOutcome:
            raise RuntimeError("private failure detail")

        with pytest.raises(PipelineStageError, match="failed safely"):
            _engine(
                root,
                config,
                store,
                overrides={PIPELINE_STAGES[1]: failed},
            ).run()
        state = store.load_state()
    elif kind == "attempted-stop":

        def stopped(_context: StageContext) -> StageOutcome:
            raise KeyboardInterrupt

        state = _engine(
            root,
            config,
            store,
            overrides={PIPELINE_STAGES[1]: stopped},
        ).run()
    elif kind == "pre-stage-stop":
        state = _engine(
            root,
            config,
            store,
            stop_requested=lambda: True,
        ).run()
    elif kind == "prefix-stop":
        decisions = iter((False, True))
        state = _engine(
            root,
            config,
            store,
            stop_requested=lambda: next(decisions),
        ).run()
    else:
        raise AssertionError("unknown terminal test state")
    return config, state


@pytest.mark.parametrize(
    ("kind", "status", "prefix", "stage_statuses"),
    [
        ("completed", "completed", len(PIPELINE_STAGES), ("completed",) * 16),
        ("blocked", "blocked", 1, ("completed", "blocked")),
        ("failed", "failed", 1, ("completed", "failed")),
        ("attempted-stop", "stopped", 1, ("completed", "stopped")),
        ("pre-stage-stop", "stopped", 0, ()),
        ("prefix-stop", "stopped", 1, ("completed",)),
    ],
)
def test_terminal_review_bundle_covers_every_terminal_shape_and_is_idempotent(
    tmp_path: Path,
    kind: str,
    status: str,
    prefix: int,
    stage_statuses: tuple[str, ...],
) -> None:
    config, state = _terminal_state(tmp_path, kind)

    first = pipeline.write_terminal_review_bundle(
        project_root=tmp_path,
        config=config,
        source_commit=SOURCE_COMMIT,
        state=state,
    )
    manifest_before = first.manifest_path.read_bytes()
    summary_before = first.summary_path.read_bytes()
    second = pipeline.write_terminal_review_bundle(
        project_root=tmp_path,
        config=config,
        source_commit=SOURCE_COMMIT,
        state=state,
    )

    assert first == second
    assert first.manifest_path.read_bytes() == manifest_before
    assert first.summary_path.read_bytes() == summary_before
    manifest = cast(pipeline.TerminalReviewBundleManifest, first.manifest)
    assert manifest.pipeline_status == status
    assert manifest.completed_prefix_length == prefix
    assert tuple(item.status for item in manifest.stages) == stage_statuses
    assert manifest.final_evaluation_accessed is False
    assert b"Final evaluation accessed: `no`" in summary_before
    assert not (first.manifest_path.parent / pipeline.FINAL_ACCESS_LEDGER_FILENAME).exists()


def test_terminal_review_rejects_mismatch_symlink_and_final_access_artifact(
    tmp_path: Path,
) -> None:
    config, state = _terminal_state(tmp_path, "pre-stage-stop")
    output = pipeline.write_terminal_review_bundle(
        project_root=tmp_path,
        config=config,
        source_commit=SOURCE_COMMIT,
        state=state,
    )
    output.summary_path.write_text("changed\n", encoding="ascii")
    with pytest.raises(pipeline.PipelineExecutionError, match="differs"):
        pipeline.write_terminal_review_bundle(
            project_root=tmp_path,
            config=config,
            source_commit=SOURCE_COMMIT,
            state=state,
        )

    other_root = tmp_path / "other"
    other_root.mkdir()
    other_config, other_state = _terminal_state(other_root, "pre-stage-stop")
    run = other_root / other_config.run_root / other_config.run_name
    (run / pipeline.FINAL_ACCESS_LEDGER_FILENAME).write_text("{}\n", encoding="ascii")
    with pytest.raises(pipeline.PipelineExecutionError, match="final-access"):
        pipeline.write_terminal_review_bundle(
            project_root=other_root,
            config=other_config,
            source_commit=SOURCE_COMMIT,
            state=other_state,
        )

    third_root = tmp_path / "third"
    third_root.mkdir()
    third_config, third_state = _terminal_state(third_root, "pre-stage-stop")
    third_run = third_root / third_config.run_root / third_config.run_name
    (third_run / pipeline.TERMINAL_REVIEW_DIRECTORY).symlink_to(tmp_path)
    with pytest.raises(pipeline.PipelineExecutionError, match="root is unsafe"):
        pipeline.write_terminal_review_bundle(
            project_root=third_root,
            config=third_config,
            source_commit=SOURCE_COMMIT,
            state=third_state,
        )


def test_stage_action_builder_runs_read_only_preflight_and_returns_immutable_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    calls: list[str] = []
    fake_inputs = cast(pipeline._ExecutionInputs, object())

    def verify(root: Path, *, source_commit: str, run_root: str) -> str:
        assert root == tmp_path.resolve()
        assert source_commit == FULL_SOURCE_COMMIT
        assert run_root == config.run_root
        calls.append("source")
        return source_commit

    def load(*, project_root: Path, config: PipelineConfig) -> pipeline._ExecutionInputs:
        assert project_root == tmp_path.resolve()
        calls.append("inputs")
        return fake_inputs

    before = tuple(tmp_path.iterdir())
    monkeypatch.setattr(pipeline, "_verify_runner_source", verify)
    monkeypatch.setattr(pipeline, "_load_execution_inputs", load)

    actions = pipeline.build_stage_actions(tmp_path, config, FULL_SOURCE_COMMIT)

    assert calls == ["source", "inputs"]
    assert tuple(actions) == PIPELINE_STAGES
    assert all(callable(action) for action in actions.values())
    assert (
        cast(SimpleNamespace, actions["preflight"]).__self__.__class__ is pipeline._PipelineRuntime
    )
    assert tuple(tmp_path.iterdir()) == before
    assert not (tmp_path / config.run_root).exists()
    with pytest.raises(TypeError):
        cast(dict[str, StageAction], actions)["preflight"] = _success

    monkeypatch.setattr(pipeline, "_verify_runner_source", lambda *_args, **_kwargs: HASH_A)
    with pytest.raises(pipeline.PipelineExecutionError, match="full Git commit"):
        pipeline.build_stage_actions(tmp_path, config, FULL_SOURCE_COMMIT)


def test_source_verifier_uses_only_trusted_absolute_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, ...]] = []

    def run(arguments: tuple[str, ...], *, project_root: Path) -> str:
        observed.append(arguments)
        assert project_root == tmp_path
        assert arguments[0] == "/usr/bin/git"
        if arguments[1:3] == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if arguments[1:3] == ("rev-parse", "HEAD"):
            return FULL_SOURCE_COMMIT
        return ""

    monkeypatch.setattr(pipeline, "_run_process", run)

    assert (
        pipeline._verify_runner_source(
            tmp_path,
            source_commit=FULL_SOURCE_COMMIT,
            run_root="work",
        )
        == FULL_SOURCE_COMMIT
    )
    assert len(observed) == 3


@pytest.mark.parametrize("confirmation", [False, True])
def test_final_evaluation_adapter_is_an_explicit_nonmutating_lock(
    tmp_path: Path,
    confirmation: bool,
) -> None:
    config = _config()
    before = tuple(tmp_path.iterdir())

    with pytest.raises(
        pipeline.FinalEvaluationBlockedError,
        match="distinct final-evidence executor",
    ):
        pipeline.run_final_evaluation(
            project_root=tmp_path,
            config=config,
            source_commit=FULL_SOURCE_COMMIT,
            explicit_confirmation=confirmation,
        )

    assert tuple(tmp_path.iterdir()) == before
    assert not tuple(tmp_path.rglob(pipeline.FINAL_ACCESS_LEDGER_FILENAME))
    with pytest.raises(TypeError, match="exact boolean"):
        pipeline.run_final_evaluation(
            project_root=tmp_path,
            config=config,
            source_commit=FULL_SOURCE_COMMIT,
            explicit_confirmation=cast(bool, 1),
        )


def _candidate(
    candidate_id: str,
    *,
    passed: bool,
    worst: float,
    iid: float,
    context_length: int,
    hash_character: str,
) -> pipeline.V04CandidateEvaluation:
    checksum = hash_character * 64
    return pipeline.V04CandidateEvaluation(
        candidate_id=candidate_id,
        context_length=context_length,
        checkpoint_manifest_sha256=checksum,
        iid_report_sha256=checksum,
        iid_acceptance_sha256=checksum,
        shadow_reports=tuple((view, checksum) for view in SHADOW_VIEWS),
        v04_acceptance_sha256=checksum,
        all_required_gates_passed=passed,
        worst_view_semantic_composite=worst,
        iid_semantic_composite=iid,
    )


def test_v04_candidate_ranking_uses_frozen_gate_worst_iid_and_context_order() -> None:
    control = _candidate(
        "control",
        passed=True,
        worst=0.60,
        iid=0.70,
        context_length=512,
        hash_character="a",
    )
    variant = _candidate(
        "variant",
        passed=False,
        worst=0.99,
        iid=0.99,
        context_length=1024,
        hash_character="b",
    )
    assert pipeline._select_v04_candidate((control, variant)) is control

    variant = variant.model_copy(
        update={
            "all_required_gates_passed": True,
            "worst_view_semantic_composite": 0.61,
            "iid_semantic_composite": 0.60,
        }
    )
    assert pipeline._select_v04_candidate((control, variant)) is variant

    variant = variant.model_copy(
        update={
            "worst_view_semantic_composite": 0.60,
            "iid_semantic_composite": 0.71,
        }
    )
    assert pipeline._select_v04_candidate((control, variant)) is variant

    variant = variant.model_copy(
        update={
            "iid_semantic_composite": 0.70,
        }
    )
    assert pipeline._select_v04_candidate((control, variant)) is control


def test_semantic_report_composite_reconstructs_supported_frozen_formula() -> None:
    def metric(estimate: float, *, support: int = 1) -> SimpleNamespace:
        return SimpleNamespace(estimate=estimate, support=support)

    metrics = SimpleNamespace(
        constrained_schema_validity_rate=metric(1.0),
        fault_family_macro_f1=metric(0.75),
        next_action_macro_f1=metric(0.75),
        continuation_macro_f1=metric(0.0, support=0),
        evidence_f1=metric(0.75),
        required_abstention_accuracy=metric(0.0, support=0),
        no_fault_false_positive_rate=metric(0.25),
        expected_calibration_error=metric(0.25),
        selective_risk_at_80_percent_coverage=metric(0.25),
    )
    report = SemanticEvaluationReport.model_construct(
        constrained=SimpleNamespace(schema_validity_rate=1.0, exact_match_rate=0.75),
        view_metrics=SimpleNamespace(metrics=metrics),
    )
    assert pipeline._semantic_report_composite(report) == 0.75

    mismatched = report.model_copy(
        update={"constrained": SimpleNamespace(schema_validity_rate=0.5, exact_match_rate=0.75)}
    )
    with pytest.raises(ValueError, match="schema-validity fields disagree"):
        pipeline._semantic_report_composite(mismatched)

    metrics.constrained_schema_validity_rate = metric(0.5)
    assert pipeline._semantic_report_composite(mismatched) == 0.0


def test_semantic_report_scope_rejects_self_bound_subset_and_manifest_rebinding() -> None:
    artifacts = SimpleNamespace(
        source_commit=SOURCE_COMMIT,
        config_sha256=HASH_A,
        dataset_manifest_sha256=HASH_B,
        tokenizer_manifest_sha256=HASH_C,
        output_contract_sha256=HASH_D,
        checkpoint_sha256=HASH_E,
        prediction_artifact_sha256=HASH_A,
        comparator_artifact_sha256=HASH_B,
    )
    report = SemanticEvaluationReport.model_construct(
        evaluation_view=RemediationView.SHADOW_RENDERER,
        example_count=12,
        predictions_sha256=HASH_A,
        baseline_report_sha256=HASH_B,
        view_metrics=SimpleNamespace(
            view=DevelopmentView.RENDERER_SHADOW,
            sample_count=12,
            artifacts=artifacts,
        ),
    )

    def require_scope(candidate: SemanticEvaluationReport) -> None:
        pipeline._require_semantic_report_scope(
            candidate,
            view=RemediationView.SHADOW_RENDERER,
            example_count=12,
            dataset_manifest_sha256=HASH_B,
            source_commit=SOURCE_COMMIT,
            config_sha256_value=HASH_A,
            tokenizer_manifest_sha256=HASH_C,
            output_contract_sha256=HASH_D,
            checkpoint_manifest_sha256=HASH_E,
        )

    require_scope(report)

    subset = report.model_copy(update={"example_count": 11})
    with pytest.raises(pipeline.PipelineExecutionError, match="exact frozen development scope"):
        require_scope(subset)

    rebound_artifacts = SimpleNamespace(**vars(artifacts))
    rebound_artifacts.dataset_manifest_sha256 = HASH_C
    rebound = report.model_copy(
        update={
            "view_metrics": SimpleNamespace(
                view=DevelopmentView.RENDERER_SHADOW,
                sample_count=12,
                artifacts=rebound_artifacts,
            )
        }
    )
    with pytest.raises(pipeline.PipelineExecutionError, match="exact frozen development scope"):
        require_scope(rebound)


def _pilot_report(
    batch_sizes: tuple[int, ...],
    *,
    resolved_device: Literal["cpu", "mps"] = "mps",
    resolved_devices: tuple[Literal["cpu", "mps"], ...] | None = None,
) -> pipeline.V04PilotReport:
    devices = (
        (resolved_device,) * len(batch_sizes) if resolved_devices is None else resolved_devices
    )
    if len(devices) != len(batch_sizes):
        raise ValueError("pilot test device inventory must cover every batch")
    measurements = tuple(
        pipeline.V04PilotMeasurement(
            batch_size=batch_size,
            training_result_sha256=HASH_A,
            training_config_sha256=HASH_B,
            model_config_sha256=HASH_C,
            train_tokenized_sha256=HASH_D,
            validation_tokenized_sha256=HASH_E,
            tokenizer_manifest_sha256=HASH_A,
            checkpoint_manifest_sha256=HASH_B,
            device=DeviceResolution(
                requested="mps",
                resolved=observed_device,
                fallback_used=observed_device == "cpu",
            ),
            train_example_count=100,
            validation_example_count=40,
            pilot_train_example_count=3,
            pilot_validation_example_count=3,
            train_length_inventory_sha256=HASH_B,
            validation_length_inventory_sha256=HASH_C,
            maximum_train_sequence_tokens=1024,
            maximum_validation_sequence_tokens=900,
            mean_train_sequence_tokens=700.0,
            mean_validation_sequence_tokens=650.0,
            maximum_train_sequence_exercised=True,
            finite_loss=True,
            checkpoint_reloaded=True,
            elapsed_seconds=1.0,
            process_peak_rss_bytes=1,
        )
        for batch_size, observed_device in zip(batch_sizes, devices, strict=True)
    )
    draft = pipeline.V04PilotReport.model_construct(
        candidate_id="v04-context-1024",
        requested_device="mps",
        required_resolved_device="mps",
        mandatory_batch_resolved_device=devices[-1] if devices else None,
        prompt_truncation_rate=668 / 882,
        v03_train_prompt_truncation_rate=0.5,
        material_truncation_threshold=0.1,
        activated=True,
        measurements=measurements,
        passed=all(device == "mps" for device in devices),
        checksum_sha256="0" * 64,
    )
    return pipeline._bound_model(draft, pipeline.V04PilotReport)


def test_v04_candidate_training_refuses_incomplete_and_cpu_fallback_pilots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    inputs = cast(
        pipeline._ExecutionInputs,
        SimpleNamespace(
            v04=SimpleNamespace(training=SimpleNamespace(batch_size=4)),
        ),
    )
    runtime = pipeline._PipelineRuntime(
        project_root=tmp_path,
        config=config,
        source_commit=SOURCE_COMMIT,
        inputs=inputs,
    )
    monkeypatch.setattr(runtime, "_start", lambda _context: SOURCE_COMMIT)
    monkeypatch.setattr(pipeline, "_upstream_attempt", lambda *_args: tmp_path)
    complete_pilot = _pilot_report((1, 2, 4))
    cpu_fallback_pilot = _pilot_report((1, 2, 4), resolved_device="cpu")
    mixed_device_pilot = _pilot_report(
        (1, 2, 4),
        resolved_devices=("cpu", "mps", "mps"),
    )
    incomplete_pilot = pipeline.V04PilotReport.model_construct(
        **complete_pilot.model_dump(mode="python", exclude={"measurements"}),
        measurements=complete_pilot.measurements[:2],
    )

    selected_pilot = [incomplete_pilot]

    def read(_path: Path, model_type: type[object], **_kwargs: object) -> object:
        if model_type is pipeline.V04PilotReport:
            return selected_pilot[0]
        return SimpleNamespace(selected_candidate_id="control")

    monkeypatch.setattr(pipeline, "_read_contract", read)
    monkeypatch.setattr(
        pipeline,
        "_verify_v04_pilot_training_evidence",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(pipeline.PipelineExecutionError, match="lacks a passing MPS pilot"):
        runtime.v04_candidate_training(
            cast(StageContext, SimpleNamespace(source_commit=SOURCE_COMMIT))
        )

    selected_pilot[0] = cpu_fallback_pilot
    with pytest.raises(pipeline.PipelineExecutionError, match="lacks a passing MPS pilot"):
        runtime.v04_candidate_training(
            cast(StageContext, SimpleNamespace(source_commit=SOURCE_COMMIT))
        )

    selected_pilot[0] = mixed_device_pilot
    with pytest.raises(pipeline.PipelineExecutionError, match="lacks a passing MPS pilot"):
        runtime.v04_candidate_training(
            cast(StageContext, SimpleNamespace(source_commit=SOURCE_COMMIT))
        )

    assert {item.batch_size for item in complete_pilot.measurements} == {1, 2, 4}
    assert complete_pilot.passed is True
    assert complete_pilot.measurements[-1].device.resolved == "mps"
    assert cpu_fallback_pilot.passed is False
    assert cpu_fallback_pilot.mandatory_batch_resolved_device == "cpu"
    assert cpu_fallback_pilot.measurements[-1].device.fallback_used is True
    assert mixed_device_pilot.mandatory_batch_resolved_device == "mps"
    assert mixed_device_pilot.measurements[0].device.resolved == "cpu"
    assert mixed_device_pilot.measurements[-1].device.resolved == "mps"
    assert mixed_device_pilot.passed is False
    with pytest.raises(ValidationError, match="batches 1, 2, and 4"):
        _pilot_report((1, 2))

    legacy_measurement = complete_pilot.measurements[0].model_dump(mode="python")
    legacy_measurement.pop("device")
    with pytest.raises(ValidationError, match="device"):
        pipeline.V04PilotMeasurement.model_validate(legacy_measurement, strict=True)
    legacy_report = complete_pilot.model_dump(mode="python")
    legacy_report.pop("requested_device")
    legacy_report.pop("required_resolved_device")
    legacy_report.pop("mandatory_batch_resolved_device")
    with pytest.raises(
        ValidationError,
        match=r"requested_device|required_resolved_device|mandatory_batch_resolved_device",
    ):
        pipeline.V04PilotReport.model_validate(legacy_report, strict=True)


def test_verified_v04_candidate_inventory_binds_control_and_optional_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_config = TransformerConfig(
        model_version="0.3.0",
        layers=2,
        width=32,
        heads=4,
        context_length=512,
        feed_forward_multiplier=2,
        dropout=0.0,
        tie_embeddings=True,
        bias=True,
    )
    variant_config = control_config.model_copy(
        update={"model_version": "0.4.0", "context_length": 1024}
    )
    runtime = pipeline._PipelineRuntime(
        project_root=tmp_path,
        config=_config("candidate-inventory"),
        source_commit=SOURCE_COMMIT,
        inputs=cast(
            pipeline._ExecutionInputs,
            SimpleNamespace(
                tokenizer=object(),
                v02=SimpleNamespace(model=control_config),
                v04=SimpleNamespace(longer_context_model=variant_config),
            ),
        ),
    )
    monkeypatch.setattr(pipeline, "_upstream_attempt", lambda *_args: tmp_path)
    selection = SimpleNamespace(
        selected_candidate_id="control",
        selected_checkpoint_manifest_sha256=HASH_A,
    )
    monkeypatch.setattr(pipeline, "_read_contract", lambda *_args, **_kwargs: selection)
    control_result = SimpleNamespace(
        candidate_id="control",
        checkpoint_manifest_sha256=HASH_A,
        checksum_sha256=HASH_C,
    )
    monkeypatch.setattr(pipeline, "_load_training_result", lambda *_args: control_result)
    control_model = SimpleNamespace(config=control_config)
    control_checkpoint = SimpleNamespace(checksum_sha256=HASH_A)
    monkeypatch.setattr(
        pipeline,
        "_load_candidate_checkpoint",
        lambda *_args: (control_model, control_checkpoint, torch.device("cpu")),
    )
    variant_report = SimpleNamespace(
        activated=True,
        candidate_id="variant",
        checkpoint_manifest_sha256=HASH_B,
    )
    variant_result = SimpleNamespace(candidate_id="variant")
    variant_model = SimpleNamespace(config=variant_config)
    variant_checkpoint = SimpleNamespace(checksum_sha256=HASH_B)
    monkeypatch.setattr(
        runtime,
        "_v04_checkpoint",
        lambda _context: (
            variant_report,
            variant_result,
            variant_model,
            variant_checkpoint,
            torch.device("cpu"),
        ),
    )
    verified_variants: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        runtime,
        "_require_v04_variant_training_provenance",
        lambda _context, *evidence: verified_variants.append(evidence),
    )

    assert runtime._verified_v04_candidate_inventory(cast(StageContext, SimpleNamespace())) == (
        ("control", 512, HASH_A),
        ("variant", 1024, HASH_B),
    )
    assert verified_variants == [
        (variant_report, variant_result, variant_checkpoint),
    ]

    variant_model.config = control_config
    with pytest.raises(pipeline.PipelineExecutionError, match="variant identity"):
        runtime._verified_v04_candidate_inventory(cast(StageContext, SimpleNamespace()))


def test_v04_variant_training_provenance_rebuilds_exact_1024_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TransformerConfig(
        model_version="0.4.0",
        layers=2,
        width=32,
        heads=4,
        context_length=1024,
        feed_forward_multiplier=2,
        dropout=0.0,
        tie_embeddings=True,
        bias=True,
    )
    training = RemediationTraining(
        seed=17,
        device="mps",
        allow_cpu_fallback=True,
        steps=20,
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        evaluation_interval=10,
        durable_checkpoint_interval=10,
    )
    train_examples = (
        SimpleNamespace(
            example_id="train",
            checksum_sha256=HASH_A,
            view=RemediationView.IID_TRAIN,
        ),
    )
    validation_examples = tuple(
        SimpleNamespace(
            example_id=f"validation-{index:02d}",
            checksum_sha256=hashlib.sha256(f"validation-{index}".encode()).hexdigest(),
            view=RemediationView.IID_VALIDATION,
        )
        for index in range(48)
    )
    iid = SimpleNamespace(examples=(*train_examples, *validation_examples))
    encoded = {
        item.example_id: CompactTokenizedExample(
            example_id=item.example_id,
            task_name=TaskName.FAULT_FAMILY,
            group_id=f"group-{item.example_id}",
            token_ids=(1, 2),
            target_mask=(False, True),
            prompt_token_count=1,
            target_token_count=1,
            prompt_tokens_retained=1,
            prompt_truncated=False,
        )
        for item in (*train_examples, *validation_examples)
    }
    tokenizer = SimpleNamespace(
        manifest=SimpleNamespace(checksum_sha256=HASH_C),
    )
    inputs = cast(
        pipeline._ExecutionInputs,
        SimpleNamespace(
            v03_dataset_config=object(),
            frozen_data_source_commit=SOURCE_COMMIT,
            v03=SimpleNamespace(
                augmentation=SimpleNamespace(
                    train_template_families=("template",),
                    train_alias_families=("alias",),
                    renderer_variants_per_projection=3,
                    include_insufficient_evidence_views=True,
                ),
                semantic_selection_example_limit=48,
            ),
            v04=SimpleNamespace(
                pilot=SimpleNamespace(candidate_id="v04-context-1024"),
                longer_context_model=model,
                training=training,
            ),
            tokenizer=tokenizer,
            generation_caps={},
        ),
    )
    runtime = pipeline._PipelineRuntime(
        project_root=tmp_path,
        config=_config("v04-variant-provenance"),
        source_commit=SOURCE_COMMIT,
        inputs=inputs,
    )
    monkeypatch.setattr(pipeline, "_load_stage_dataset", lambda *_args: iid)
    monkeypatch.setattr(
        pipeline,
        "build_frozen_v03_iid_material",
        lambda *_args, **_kwargs: SimpleNamespace(dataset=iid),
    )
    monkeypatch.setattr(pipeline, "_upstream_attempt", lambda *_args: tmp_path)
    monkeypatch.setattr(
        pipeline,
        "_read_contract",
        lambda *_args, **_kwargs: SimpleNamespace(checksum_sha256=HASH_D),
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_semantic_selection_examples",
        lambda *_args: validation_examples,
    )
    monkeypatch.setattr(
        pipeline,
        "_tokenize_examples",
        lambda examples, *_args, **_kwargs: tuple(encoded[item.example_id] for item in examples),
    )
    train_tokenized = tuple(encoded[item.example_id] for item in train_examples)
    validation_tokenized = tuple(encoded[item.example_id] for item in validation_examples)
    result = SimpleNamespace(
        candidate_id="v04-context-1024",
        checksum_sha256=HASH_E,
        sampling_strategy="task_balanced",
        source_commit=SOURCE_COMMIT,
        training_steps=training.steps,
        training_config_sha256=canonical_sha256(training.model_dump(mode="json", round_trip=True)),
        model_config_sha256=canonical_sha256(model.model_dump(mode="json", round_trip=True)),
        tokenizer_manifest_sha256=HASH_C,
        train_example_count=1,
        validation_example_count=48,
        train_inventory_sha256=canonical_sha256((("train", HASH_A),)),
        validation_inventory_sha256=canonical_sha256(
            tuple((item.example_id, item.checksum_sha256) for item in validation_examples)
        ),
        train_tokenized_sha256=pipeline._tokenized_inventory_sha256(train_tokenized),
        validation_tokenized_sha256=pipeline._tokenized_inventory_sha256(validation_tokenized),
        checkpoint_manifest_sha256=HASH_B,
        checkpoint_weights_sha256=HASH_D,
        checkpoint_size_bytes=4096,
        selected_step=10,
        initial_validation_nll=1.25,
        selected_validation_nll=0.5,
        parameter_count=321,
        vocab_size=64,
    )
    checkpoint = SimpleNamespace(
        checksum_sha256=HASH_B,
        transformer_config=model,
        vocab_size=64,
        parameter_count=321,
        tokenizer_manifest_sha256=HASH_C,
        source_commit=SOURCE_COMMIT,
        seed=training.seed,
        training_steps=10,
        initial_loss=1.25,
        final_loss=0.5,
        weights_sha256=HASH_D,
        weights_size_bytes=4096,
    )
    report = SimpleNamespace(
        activated=True,
        reused_v03_candidate=False,
        source_stage="v04_candidate_training",
        candidate_id="v04-context-1024",
        training_result_sha256=HASH_E,
    )
    context = cast(StageContext, SimpleNamespace(source_commit=SOURCE_COMMIT))

    runtime._require_v04_variant_training_provenance(
        context,
        cast(pipeline.V04CandidateTrainingReport, report),
        cast(CompactTrainingResult, result),
        cast(CheckpointManifest, checkpoint),
    )

    original_validation_inventory = result.validation_inventory_sha256
    result.validation_inventory_sha256 = HASH_A
    with pytest.raises(pipeline.PipelineExecutionError, match="exact frozen inputs"):
        runtime._require_v04_variant_training_provenance(
            context,
            cast(pipeline.V04CandidateTrainingReport, report),
            cast(CompactTrainingResult, result),
            cast(CheckpointManifest, checkpoint),
        )
    result.validation_inventory_sha256 = original_validation_inventory
    result.training_config_sha256 = HASH_A
    with pytest.raises(pipeline.PipelineExecutionError, match="exact frozen inputs"):
        runtime._require_v04_variant_training_provenance(
            context,
            cast(pipeline.V04CandidateTrainingReport, report),
            cast(CompactTrainingResult, result),
            cast(CheckpointManifest, checkpoint),
        )


def test_v04_candidate_consumer_reopens_and_binds_each_pilot_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TransformerConfig(
        model_version="0.4.0",
        layers=2,
        width=32,
        heads=4,
        context_length=1024,
        feed_forward_multiplier=2,
        dropout=0.0,
        tie_embeddings=True,
        bias=True,
    )
    training = RemediationTraining(
        seed=7,
        device="mps",
        allow_cpu_fallback=True,
        steps=20,
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        evaluation_interval=5,
        durable_checkpoint_interval=5,
    )
    inputs = cast(
        pipeline._ExecutionInputs,
        SimpleNamespace(
            v03=SimpleNamespace(
                augmentation=SimpleNamespace(
                    train_template_families=("t",),
                    train_alias_families=("a",),
                    renderer_variants_per_projection=1,
                    include_insufficient_evidence_views=True,
                )
            ),
            v03_dataset_config=object(),
            frozen_data_source_commit=SOURCE_COMMIT,
            generation_caps={},
            v02=SimpleNamespace(model=model.model_copy(update={"context_length": 512})),
            v04=SimpleNamespace(
                training=training,
                pilot=SimpleNamespace(
                    candidate_id="v04-context-1024",
                    steps=10,
                    batch_sizes=(1, 2, 4),
                ),
                longer_context_model=model,
                variants=SimpleNamespace(material_prompt_truncation_rate=0.1),
            ),
            tokenizer=SimpleNamespace(manifest=SimpleNamespace(checksum_sha256=HASH_A)),
        ),
    )
    raw_train: list[RemediationExample] = []
    raw_validation: list[RemediationExample] = []
    encoded: dict[str, CompactTokenizedExample] = {}
    short_train: list[CompactTokenizedExample] = []
    for task_index, task_name in enumerate(TaskName):
        for view, destination, base_length in (
            (RemediationView.IID_TRAIN, raw_train, 100),
            (RemediationView.IID_VALIDATION, raw_validation, 80),
        ):
            for variant, offset in (("short", 0), ("long", 40)):
                example_id = f"{view.value}-{task_name.value}-{variant}"
                group_id = f"group-{example_id}"
                destination.append(
                    cast(
                        RemediationExample,
                        SimpleNamespace(
                            example_id=example_id,
                            task_name=task_name,
                            group_id=group_id,
                            view=view,
                            checksum_sha256=hashlib.sha256(example_id.encode()).hexdigest(),
                        ),
                    )
                )
                length = base_length + task_index + offset
                tokenized = CompactTokenizedExample(
                    example_id=example_id,
                    task_name=task_name,
                    group_id=group_id,
                    token_ids=tuple(range(length)),
                    target_mask=(*([False] * (length - 1)), True),
                    prompt_token_count=length if variant == "long" else length - 1,
                    target_token_count=1,
                    prompt_tokens_retained=length - 1,
                    prompt_truncated=variant == "long",
                )
                encoded[example_id] = tokenized
                if view is RemediationView.IID_TRAIN and variant == "short":
                    short_train.append(tokenized)
    all_raw = (*raw_train, *raw_validation)
    monkeypatch.setattr(
        pipeline,
        "build_frozen_v03_iid_material",
        lambda *_args, **_kwargs: SimpleNamespace(dataset=SimpleNamespace(examples=all_raw)),
    )
    monkeypatch.setattr(
        pipeline,
        "_tokenize_examples",
        lambda examples, *_args, **_kwargs: tuple(encoded[item.example_id] for item in examples),
    )
    longer_train = tuple(encoded[item.example_id] for item in raw_train)
    longer_validation = tuple(encoded[item.example_id] for item in raw_validation)
    pilot_train, pilot_train_tokenized = pipeline._longest_pilot_examples_per_task(
        tuple(raw_train),
        longer_train,
    )
    pilot_validation, pilot_validation_tokenized = pipeline._longest_pilot_examples_per_task(
        tuple(raw_validation),
        longer_validation,
    )
    train_tokenized_sha256 = pipeline._tokenized_inventory_sha256(pilot_train_tokenized)
    validation_tokenized_sha256 = pipeline._tokenized_inventory_sha256(pilot_validation_tokenized)
    model_sha256 = canonical_sha256(model.model_dump(mode="json", round_trip=True))
    original = _pilot_report((1, 2, 4))
    measurements: list[pipeline.V04PilotMeasurement] = []
    reopened: dict[str, SimpleNamespace] = {}
    checkpoints: dict[str, SimpleNamespace] = {}
    for measurement in original.measurements:
        pilot_training = pipeline._v04_pilot_training_config(
            inputs,
            batch_size=measurement.batch_size,
        )
        training_sha256 = canonical_sha256(pilot_training.model_dump(mode="json", round_trip=True))
        bound_measurement = measurement.model_copy(
            update={
                "training_config_sha256": training_sha256,
                "model_config_sha256": model_sha256,
                "tokenizer_manifest_sha256": HASH_A,
                "train_tokenized_sha256": train_tokenized_sha256,
                "validation_tokenized_sha256": validation_tokenized_sha256,
                "train_example_count": len(raw_train),
                "validation_example_count": len(raw_validation),
                "pilot_train_example_count": len(pilot_train),
                "pilot_validation_example_count": len(pilot_validation),
                "train_length_inventory_sha256": pipeline._sequence_length_inventory_sha256(
                    longer_train
                ),
                "validation_length_inventory_sha256": (
                    pipeline._sequence_length_inventory_sha256(longer_validation)
                ),
                "maximum_train_sequence_tokens": max(len(item.token_ids) for item in longer_train),
                "maximum_validation_sequence_tokens": max(
                    len(item.token_ids) for item in longer_validation
                ),
                "mean_train_sequence_tokens": sum(len(item.token_ids) for item in longer_train)
                / len(longer_train),
                "mean_validation_sequence_tokens": sum(
                    len(item.token_ids) for item in longer_validation
                )
                / len(longer_validation),
                "elapsed_seconds": 1.25,
                "process_peak_rss_bytes": 2048,
            }
        )
        measurements.append(bound_measurement)
        candidate_id = f"{original.candidate_id}-b{measurement.batch_size}"
        reopened[candidate_id] = SimpleNamespace(
            candidate_id=candidate_id,
            sampling_strategy="task_balanced",
            training_steps=10,
            source_commit=SOURCE_COMMIT,
            checksum_sha256=bound_measurement.training_result_sha256,
            training_config_sha256=training_sha256,
            model_config_sha256=model_sha256,
            train_tokenized_sha256=bound_measurement.train_tokenized_sha256,
            validation_tokenized_sha256=bound_measurement.validation_tokenized_sha256,
            tokenizer_manifest_sha256=HASH_A,
            checkpoint_manifest_sha256=bound_measurement.checkpoint_manifest_sha256,
            checkpoint_weights_sha256=HASH_C,
            checkpoint_size_bytes=4096,
            device=bound_measurement.device,
            train_example_count=bound_measurement.pilot_train_example_count,
            validation_example_count=bound_measurement.pilot_validation_example_count,
            train_inventory_sha256=canonical_sha256(
                tuple((item.example_id, item.checksum_sha256) for item in pilot_train)
            ),
            validation_inventory_sha256=canonical_sha256(
                tuple((item.example_id, item.checksum_sha256) for item in pilot_validation)
            ),
            selected_step=10,
            elapsed_seconds=1.25,
            process_peak_rss_bytes=2048,
        )
        checkpoints[candidate_id] = SimpleNamespace(
            checksum_sha256=bound_measurement.checkpoint_manifest_sha256,
            transformer_config=model,
            tokenizer_manifest_sha256=HASH_A,
            source_commit=SOURCE_COMMIT,
            training_steps=10,
            weights_sha256=HASH_C,
            weights_size_bytes=4096,
        )
    draft = pipeline.V04PilotReport.model_construct(
        **original.model_dump(
            mode="python",
            exclude={"measurements", "checksum_sha256"},
        ),
        measurements=tuple(measurements),
        checksum_sha256="0" * 64,
    )
    pilot = pipeline._bound_model(draft, pipeline.V04PilotReport)
    loaded_candidates: list[str] = []

    def load_result(_attempt: Path, candidate_id: str) -> object:
        loaded_candidates.append(candidate_id)
        return reopened[candidate_id]

    monkeypatch.setattr(pipeline, "_load_training_result", load_result)
    monkeypatch.setattr(
        pipeline,
        "_read_contract",
        lambda path, model_type, **_kwargs: (
            checkpoints[path.parent.name.removeprefix("checkpoint-")]
            if model_type is CheckpointManifest
            else pytest.fail(f"unexpected contract type: {model_type}")
        ),
    )
    pipeline._verify_v04_pilot_training_evidence(
        tmp_path,
        pilot,
        inputs,
        source_commit=SOURCE_COMMIT,
    )
    assert loaded_candidates == [
        "v04-context-1024-b1",
        "v04-context-1024-b2",
        "v04-context-1024-b4",
    ]

    original_encoded = dict(encoded)
    encoded.update(
        {example_id: replace(item, prompt_truncated=False) for example_id, item in encoded.items()}
    )
    with pytest.raises(pipeline.PipelineExecutionError, match="activation differs"):
        pipeline._verify_v04_pilot_training_evidence(
            tmp_path,
            pilot,
            inputs,
            source_commit=SOURCE_COMMIT,
        )
    encoded.clear()
    encoded.update(original_encoded)

    inactive_draft = pipeline.V04PilotReport.model_construct(
        **pilot.model_dump(
            mode="python",
            exclude={
                "activated",
                "mandatory_batch_resolved_device",
                "measurements",
                "passed",
                "v03_train_prompt_truncation_rate",
                "checksum_sha256",
            },
        ),
        v03_train_prompt_truncation_rate=0.0,
        activated=False,
        mandatory_batch_resolved_device=None,
        measurements=(),
        passed=True,
        checksum_sha256="0" * 64,
    )
    inactive = pipeline._bound_model(inactive_draft, pipeline.V04PilotReport)
    with pytest.raises(pipeline.PipelineExecutionError, match="activation differs"):
        pipeline._verify_v04_pilot_training_evidence(
            tmp_path,
            inactive,
            inputs,
            source_commit=SOURCE_COMMIT,
        )

    short_train_sha256 = pipeline._tokenized_inventory_sha256(tuple(short_train))
    tampered_measurements = tuple(
        item.model_copy(update={"train_tokenized_sha256": short_train_sha256})
        for item in pilot.measurements
    )
    tampered_draft = pipeline.V04PilotReport.model_construct(
        **pilot.model_dump(mode="python", exclude={"measurements", "checksum_sha256"}),
        measurements=tampered_measurements,
        checksum_sha256="0" * 64,
    )
    tampered = pipeline._bound_model(tampered_draft, pipeline.V04PilotReport)
    for result in reopened.values():
        result.train_tokenized_sha256 = short_train_sha256
    with pytest.raises(pipeline.PipelineExecutionError, match="independently reopened"):
        pipeline._verify_v04_pilot_training_evidence(
            tmp_path,
            tampered,
            inputs,
            source_commit=SOURCE_COMMIT,
        )

    for result in reopened.values():
        result.train_tokenized_sha256 = train_tokenized_sha256
    reopened["v04-context-1024-b2"].train_tokenized_sha256 = HASH_A
    with pytest.raises(pipeline.PipelineExecutionError, match="independently reopened"):
        pipeline._verify_v04_pilot_training_evidence(
            tmp_path,
            pilot,
            inputs,
            source_commit=SOURCE_COMMIT,
        )


def _bound_v02_report(
    *,
    prompt_truncation_count: int = pipeline.V02_FROZEN_PROMPT_TRUNCATION_COUNT,
    advancement_allowed: bool = True,
) -> pipeline.V02DevelopmentGateReport:
    draft = pipeline.V02DevelopmentGateReport.model_construct(
        inventory_report_sha256=HASH_A,
        prediction_manifest_sha256=HASH_B,
        training_result_sha256=HASH_C,
        checkpoint_manifest_sha256=HASH_D,
        checkpoint_weights_sha256=HASH_E,
        example_count=252,
        constrained_parse_rate=1.0,
        constrained_schema_validity_rate=1.0,
        constrained_exact_semantic_match_rate=0.5,
        constrained_mean_latency_seconds=0.01,
        unconstrained_parse_rate=0.0,
        unconstrained_schema_validity_rate=0.0,
        unconstrained_exact_semantic_match_rate=0.25,
        unconstrained_mean_latency_seconds=0.02,
        generation_cap_exhaustion_rate=0.0,
        process_peak_rss_bytes=1024,
        mps_peak_current_allocated_bytes=0,
        mps_peak_driver_allocated_bytes=0,
        checkpoint_size_bytes=2048,
        inventory_example_count=882,
        prompt_truncation_count=prompt_truncation_count,
        prompt_truncation_rate=prompt_truncation_count / 882,
        target_fit_rate=1.0,
        round_trip_rate=1.0,
        reachability_rate=1.0,
        task_footer_retained_rate=1.0,
        cap_exhaustion_target_rate=0.0,
        advancement_allowed=advancement_allowed,
        checksum_sha256="0" * 64,
    )
    return pipeline._bound_model(draft, pipeline.V02DevelopmentGateReport)


def test_v02_gate_requires_exact_d073_inventory_and_never_claims_materiality() -> None:
    report = _bound_v02_report()

    assert report.advancement_allowed is True
    assert report.prompt_truncation_materially_lower is False
    assert report.v04_context_pilot_required is True
    with pytest.raises(ValidationError, match="advancement differs"):
        _bound_v02_report(prompt_truncation_count=667, advancement_allowed=True)

    payload = report.model_dump(mode="json", round_trip=True)
    payload["constrained_exact_semantic_match_rate"] = 0.75
    with pytest.raises(ValidationError, match=r"v0\.2 development gate checksum mismatch"):
        pipeline.V02DevelopmentGateReport.model_validate_json(
            canonical_json_bytes(payload), strict=True
        )


class _FakeProgress:
    def __init__(self) -> None:
        self.metric = ProgressMetric(name="validation_nll", value=0.5)
        self.reports: list[dict[str, object]] = []
        self.checkpoints: list[dict[str, object]] = []

    def snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(latest_metric=self.metric)

    def report(self, **values: object) -> None:
        self.reports.append(values)

    def checkpoint(self, **values: object) -> None:
        self.checkpoints.append(values)


def test_training_progress_exposes_durable_stopped_and_final_checkpoint_paths(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    attempt = run / "stages/07-v03_candidate_training/attempt-0001"
    durable = attempt / "training-state/control/state-step-00000003"
    final = attempt / "checkpoint-control"
    durable.mkdir(parents=True)
    final.mkdir()
    progress = _FakeProgress()
    context = cast(
        StageContext,
        SimpleNamespace(run_directory=run, attempt_directory=attempt, progress=progress),
    )
    callback = pipeline._training_progress_callback(context, "control")

    callback(
        TrainingProgress(
            event="durable_checkpoint",
            step=3,
            total_steps=10,
            checkpoint_name=durable.name,
        )
    )
    callback(
        TrainingProgress(
            event="stopped",
            step=3,
            total_steps=10,
            checkpoint_name=durable.name,
        )
    )
    callback(
        TrainingProgress(
            event="final_checkpoint",
            step=10,
            total_steps=10,
            checkpoint_name=final.name,
        )
    )

    assert [item["completed_units"] for item in progress.reports] == [3, 3, 10]
    assert all(item["latest_metric"] == progress.metric for item in progress.reports)
    assert [item["checkpoint"] for item in progress.checkpoints] == [
        durable.relative_to(run).as_posix(),
        durable.relative_to(run).as_posix(),
        final.relative_to(run).as_posix(),
    ]


def test_prediction_writer_uses_the_shared_canonical_byte_contract(tmp_path: Path) -> None:
    run = tmp_path / "run"
    attempt = run / "attempt"
    attempt.mkdir(parents=True)
    examples = tuple(
        RemediationExample.model_construct(
            example_id=identifier,
            view=RemediationView.IID_VALIDATION,
            checksum_sha256=checksum,
        )
        for identifier, checksum in (("example:b", HASH_B), ("example:a", HASH_A))
    )
    predictions = tuple(
        DualPathCompactPrediction.model_construct(
            example_id=identifier,
            example_checksum_sha256=example_checksum,
            checksum_sha256=prediction_checksum,
        )
        for identifier, example_checksum, prediction_checksum in (
            ("example:b", HASH_B, HASH_D),
            ("example:a", HASH_A, HASH_C),
        )
    )
    context = cast(
        StageContext,
        SimpleNamespace(run_directory=run, attempt_directory=attempt),
    )

    manifest, _artifacts = pipeline._write_predictions(
        context,
        stem="canonical-predictions",
        view=RemediationView.IID_VALIDATION,
        examples=examples,
        predictions=predictions,
    )
    ordered = tuple(sorted(predictions, key=lambda item: item.example_id))
    expected = canonical_prediction_jsonl_bytes(ordered)

    assert (attempt / "canonical-predictions.jsonl").read_bytes() == expected
    assert manifest.predictions_size_bytes == len(expected)
    assert manifest.predictions_sha256 == prediction_artifact_byte_sha256(ordered)


def test_prediction_reader_reconstructs_exact_artifact_and_rejects_byte_drift(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    attempt = run / "attempt"
    attempt.mkdir(parents=True)
    example = RemediationExample.model_construct(
        example_id="example:calibration",
        view=RemediationView.IID_VALIDATION,
        task_name=TaskName.FAULT_FAMILY,
        checksum_sha256=HASH_A,
    )

    def path_result(path: DecodePath) -> CompactPathPrediction:
        values = {
            "path": path,
            "task_name": TaskName.FAULT_FAMILY,
            "generation_cap": 16,
            "prompt_token_count": 10,
            "prompt_tokens_retained": 10,
            "prompt_truncated": False,
            "generated_token_ids": (4,),
            "generated_token_count": 1,
            "selected_token_count": 2,
            "generated_text": "invalid",
            "eos_emitted": True,
            "generation_cap_exhausted": False,
            "compact_parse_success": False,
            "schema_valid": False,
            "canonical_target_json": None,
            "selected_token_geometric_mean_probability": 0.5,
            "elapsed_seconds": 0.01,
            "used_cache": True,
        }
        draft = CompactPathPrediction.model_construct(**values, checksum_sha256="0" * 64)
        return CompactPathPrediction(
            **values,
            checksum_sha256=canonical_sha256(
                draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
            ),
        )

    dual_values = {
        "example_id": example.example_id,
        "example_checksum_sha256": example.checksum_sha256,
        "task_name": example.task_name,
        "model_config_sha256": HASH_B,
        "tokenizer_manifest_sha256": HASH_C,
        "generation_caps_sha256": HASH_D,
        "unconstrained": path_result(DecodePath.UNCONSTRAINED),
        "constrained": path_result(DecodePath.CONSTRAINED),
    }
    dual_draft = DualPathCompactPrediction.model_construct(**dual_values, checksum_sha256="0" * 64)
    prediction = DualPathCompactPrediction(
        **dual_values,
        checksum_sha256=canonical_sha256(
            dual_draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        ),
    )
    context = cast(
        StageContext,
        SimpleNamespace(run_directory=run, attempt_directory=attempt),
    )
    pipeline._write_predictions(
        context,
        stem="calibration-predictions",
        view=RemediationView.IID_VALIDATION,
        examples=(example,),
        predictions=(prediction,),
    )
    manifest, reopened = pipeline._read_predictions(
        manifest_path=attempt / "calibration-predictions-manifest.json",
        predictions_path=attempt / "calibration-predictions.jsonl",
        view=RemediationView.IID_VALIDATION,
        examples=(example,),
    )
    assert reopened == (prediction,)
    assert manifest.predictions_sha256 == prediction_artifact_byte_sha256(reopened)

    predictions_path = attempt / "calibration-predictions.jsonl"
    predictions_path.write_bytes(predictions_path.read_bytes().replace(b"invalid", b"changed"))
    with pytest.raises(pipeline.PipelineExecutionError, match="byte checksum"):
        pipeline._read_predictions(
            manifest_path=attempt / "calibration-predictions-manifest.json",
            predictions_path=predictions_path,
            view=RemediationView.IID_VALIDATION,
            examples=(example,),
        )


def test_targeted_gate_reconstructs_acceptance_and_rejects_every_derived_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score = pipeline.CandidateScore(
        candidate_id="targeted",
        checkpoint_manifest_sha256=HASH_A,
        semantic_composite=0.75,
        selected_validation_nll=0.2,
        selected_step=100,
        evaluation_report_sha256=HASH_B,
    )
    selection_draft = pipeline.CandidateSelectionReport.model_construct(
        selection_manifest_sha256=HASH_C,
        candidates=(score,),
        selected_candidate_id="targeted",
        selected_checkpoint_manifest_sha256=HASH_A,
        checksum_sha256="0" * 64,
    )
    selection = pipeline._bound_model(selection_draft, pipeline.CandidateSelectionReport)
    calibration_selection = SimpleNamespace(checksum_sha256=HASH_B)
    calibration_manifest = SimpleNamespace(
        checksum_sha256=HASH_C,
        predictions_sha256=HASH_D,
    )
    calibration_examples = tuple(
        SimpleNamespace(
            example_id=f"calibration-{index:03d}",
            checksum_sha256=HASH_A,
            canonical_target_json="target" if index % 2 else "other",
        )
        for index in reversed(range(56))
    )
    calibration_predictions = tuple(
        SimpleNamespace(
            example_id=f"calibration-{index:03d}",
            example_checksum_sha256=HASH_A,
            constrained=SimpleNamespace(
                selected_token_geometric_mean_probability=0.6,
                canonical_target_json="target",
            ),
        )
        for index in range(56)
    )
    assert calibration_examples[0].example_id != calibration_predictions[0].example_id
    observations = pipeline._calibration_observations_by_identity(
        cast(tuple[RemediationExample, ...], calibration_examples),
        cast(tuple[DualPathCompactPrediction, ...], calibration_predictions),
    )
    calibration = pipeline.fit_temperature(
        observations,
        calibration_selection_manifest_sha256=HASH_B,
        calibration_prediction_manifest_sha256=HASH_C,
        calibration_predictions_sha256=HASH_D,
        selected_checkpoint_manifest_sha256=HASH_A,
    )
    gate_examples = (SimpleNamespace(example_id="gate"),)
    raw_manifest = SimpleNamespace(predictions_sha256=HASH_E)
    raw_predictions = (SimpleNamespace(marker="original"),)
    baseline = SimpleNamespace(checksum_sha256=HASH_B)
    artifacts = DevelopmentArtifactBinding(
        source_commit=SOURCE_COMMIT,
        config_sha256=HASH_A,
        dataset_manifest_sha256=HASH_B,
        tokenizer_manifest_sha256=HASH_C,
        output_contract_sha256=HASH_D,
        checkpoint_sha256=HASH_A,
        prediction_artifact_sha256=HASH_E,
        comparator_artifact_sha256=HASH_B,
    )
    raw_metrics = SimpleNamespace(name="raw")
    calibrated_metrics = SimpleNamespace(name="calibrated")
    saved_raw = SimpleNamespace(
        predictions_sha256=HASH_E,
        checksum_sha256=HASH_C,
        view_metrics=raw_metrics,
    )
    saved_calibrated = SimpleNamespace(
        predictions_sha256=HASH_E,
        checksum_sha256=HASH_D,
        view_metrics=calibrated_metrics,
    )
    tampered = SimpleNamespace(
        predictions_sha256=HASH_E,
        checksum_sha256=HASH_A,
        view_metrics=SimpleNamespace(name="tampered"),
    )

    def evaluate(*_args: object, **kwargs: object) -> object:
        predictions = cast(tuple[SimpleNamespace, ...], kwargs["predictions"])
        if predictions[0].marker != "original":
            return tampered
        return saved_calibrated if kwargs.get("confidence_transform") is not None else saved_raw

    monkeypatch.setattr(pipeline, "evaluate_semantic_predictions", evaluate)
    monkeypatch.setattr(
        pipeline,
        "_semantic_reports_differ_only_in_confidence",
        lambda *_args: True,
    )
    acceptance = SimpleNamespace(advancement_allowed=True, checksum_sha256=HASH_E)
    observed_acceptance_metrics: list[object] = []

    def accept(metrics: object) -> object:
        observed_acceptance_metrics.append(metrics)
        return acceptance

    monkeypatch.setattr(pipeline, "evaluate_v03_acceptance", accept)

    def reconstruct(
        *,
        calibration_predictions_override: tuple[object, ...] = calibration_predictions,
        raw_predictions_override: tuple[object, ...] = raw_predictions,
        raw_report: object = saved_raw,
        calibrated_report: object = saved_calibrated,
    ) -> tuple[object, pipeline.TargetedV03GateBinding]:
        return pipeline._reconstruct_targeted_v03_gate(
            selection=selection,
            calibration_selection=cast(object, calibration_selection),
            calibration=calibration,
            calibration_examples=cast(tuple[RemediationExample, ...], calibration_examples),
            calibration_prediction_manifest=cast(
                pipeline.PredictionArtifactManifest, calibration_manifest
            ),
            calibration_predictions=cast(
                tuple[DualPathCompactPrediction, ...], calibration_predictions_override
            ),
            gate_examples=cast(tuple[RemediationExample, ...], gate_examples),
            raw_prediction_manifest=cast(pipeline.PredictionArtifactManifest, raw_manifest),
            raw_predictions=cast(tuple[DualPathCompactPrediction, ...], raw_predictions_override),
            raw_baseline=cast(object, baseline),
            expected_artifacts=artifacts,
            saved_raw_evaluation=cast(SemanticEvaluationReport, raw_report),
            saved_calibrated_evaluation=cast(SemanticEvaluationReport, calibrated_report),
        )

    reconstructed_acceptance, binding = reconstruct()
    assert reconstructed_acceptance is acceptance
    assert observed_acceptance_metrics == [calibrated_metrics]
    assert binding.raw_prediction_artifact_sha256 == HASH_E
    assert binding.calibrated_prediction_artifact_sha256 == HASH_E
    assert binding.outputs_bit_exact is True
    assert binding.thresholds_unchanged is True

    changed_calibration = (
        SimpleNamespace(
            example_id=calibration_predictions[0].example_id,
            example_checksum_sha256=HASH_A,
            constrained=SimpleNamespace(
                selected_token_geometric_mean_probability=0.99,
                canonical_target_json="target",
            ),
        ),
        *calibration_predictions[1:],
    )
    with pytest.raises(pipeline.PipelineExecutionError, match="temperature calibration differs"):
        reconstruct(calibration_predictions_override=changed_calibration)
    with pytest.raises(pipeline.PipelineExecutionError, match="gate reports differ"):
        reconstruct(raw_predictions_override=(SimpleNamespace(marker="changed"),))
    with pytest.raises(pipeline.PipelineExecutionError, match="gate reports differ"):
        reconstruct(raw_report=tampered)
    with pytest.raises(pipeline.PipelineExecutionError, match="gate reports differ"):
        reconstruct(calibrated_report=tampered)


def test_v03_development_evaluation_selects_on_subset_but_gates_on_full_iid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    selection_example = SimpleNamespace(view=RemediationView.IID_VALIDATION)
    full_validation = (
        selection_example,
        SimpleNamespace(view=RemediationView.IID_VALIDATION),
        SimpleNamespace(view=RemediationView.IID_VALIDATION),
    )
    train = (SimpleNamespace(view=RemediationView.IID_TRAIN),)
    dataset = SimpleNamespace(examples=(*train, *full_validation))
    candidates = (
        SimpleNamespace(candidate_id="control"),
        SimpleNamespace(candidate_id="balanced"),
    )
    inputs = SimpleNamespace(
        v03=SimpleNamespace(
            candidates=candidates,
            selection=load_v03_config(
                Path(__file__).resolve().parents[2]
                / "configs/experiments/phase6-remediation-v0.3.2-focused.toml"
            ).selection,
        ),
        tokenizer=object(),
    )
    runtime = pipeline._PipelineRuntime(
        project_root=tmp_path,
        config=config,
        source_commit=SOURCE_COMMIT,
        inputs=cast(pipeline._ExecutionInputs, inputs),
    )
    monkeypatch.setattr(runtime, "_start", lambda _context: SOURCE_COMMIT)
    monkeypatch.setattr(runtime, "_finish", lambda _context, outcome: outcome)
    monkeypatch.setattr(pipeline, "config_sha256", lambda _config: HASH_A)
    monkeypatch.setattr(pipeline, "_upstream_attempt", lambda *_args: tmp_path)
    monkeypatch.setattr(pipeline, "load_safe_development_artifact", lambda _path: dataset)
    monkeypatch.setattr(
        pipeline,
        "_read_contract",
        lambda *_args, **_kwargs: SimpleNamespace(checksum_sha256=HASH_C),
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_semantic_selection_examples",
        lambda *_args: (selection_example,),
    )

    def result(_attempt: Path, candidate_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            selected_validation_nll=0.2 if candidate_id == "balanced" else 0.3,
            selected_step=2,
        )

    monkeypatch.setattr(pipeline, "_load_training_result", result)

    def checkpoint(
        _attempt: Path,
        candidate_id: str,
        _result: object,
        _tokenizer: object,
    ) -> tuple[object, object, torch.device]:
        checksum = HASH_D if candidate_id == "balanced" else HASH_E
        return object(), SimpleNamespace(checksum_sha256=checksum), torch.device("cpu")

    monkeypatch.setattr(pipeline, "_load_candidate_checkpoint", checkpoint)
    evaluated_sizes: list[int] = []
    evaluated_stems: list[str] = []

    def evaluate(*_args: object, **kwargs: object) -> tuple[object, None, tuple[object], tuple[()]]:
        examples = cast(tuple[object, ...], kwargs["evaluation_examples"])
        evaluated_sizes.append(len(examples))
        evaluated_stems.append(cast(str, kwargs["stem"]))
        score = 0.8 if len(examples) == 1 and len(evaluated_sizes) == 2 else 0.5
        prediction = SimpleNamespace(score=score)
        evaluation = SimpleNamespace(
            checksum_sha256=hashlib.sha256(str(len(evaluated_sizes)).encode()).hexdigest(),
            score=score,
        )
        return evaluation, None, (prediction,), ()

    monkeypatch.setattr(pipeline, "_evaluate_candidate_view", evaluate)
    monkeypatch.setattr(
        pipeline,
        "_semantic_report_composite",
        lambda evaluation: evaluation.score,
    )
    written: list[tuple[str, object]] = []

    def artifact(_context: object, filename: str, model: object) -> ArtifactReference:
        written.append((filename, model))
        return ArtifactReference(
            relative_path=f"stages/{filename}",
            sha256=hashlib.sha256(filename.encode()).hexdigest(),
            size_bytes=1,
        )

    monkeypatch.setattr(pipeline, "_contract_artifact", artifact)
    context = cast(StageContext, SimpleNamespace())

    runtime.v03_development_evaluation(context)

    assert evaluated_sizes == [1, 1, 3]
    assert evaluated_stems[-1] == "v03-selected-full-iid"


def test_v03_gate_rederives_complete_candidate_ranking_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TransformerConfig(
        model_version="0.3.0",
        layers=2,
        width=32,
        heads=4,
        context_length=512,
        feed_forward_multiplier=2,
        dropout=0.0,
        tie_embeddings=True,
        bias=True,
    )
    training = RemediationTraining(
        seed=5,
        device="cpu",
        allow_cpu_fallback=True,
        steps=10,
        batch_size=2,
        learning_rate=0.001,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        evaluation_interval=5,
        durable_checkpoint_interval=5,
    )
    candidate_policies = (
        SimpleNamespace(candidate_id="control", sampling="uniform_control", seed=7),
        SimpleNamespace(candidate_id="balanced", sampling="task_balanced", seed=11),
    )
    inputs = cast(
        pipeline._ExecutionInputs,
        SimpleNamespace(
            v02=SimpleNamespace(model=model),
                v03=SimpleNamespace(
                    training=training,
                    candidates=candidate_policies,
                    selection=load_v03_config(
                        Path(__file__).resolve().parents[2]
                        / "configs/experiments/phase6-remediation-v0.3.2-focused.toml"
                    ).selection,
                ),
            tokenizer=SimpleNamespace(manifest=SimpleNamespace(checksum_sha256=HASH_C)),
            generation_caps={},
            compact_contract_sha256=HASH_D,
        ),
    )
    runtime = pipeline._PipelineRuntime(
        project_root=tmp_path,
        config=_config("v03-gate-provenance"),
        source_commit=SOURCE_COMMIT,
        inputs=inputs,
    )
    monkeypatch.setattr(runtime, "_start", lambda _context: SOURCE_COMMIT)
    monkeypatch.setattr(runtime, "_finish", lambda _context, outcome: outcome)
    monkeypatch.setattr(pipeline, "_upstream_attempt", lambda *_args: tmp_path)
    train_example = SimpleNamespace(
        example_id="train",
        checksum_sha256=HASH_A,
        view=RemediationView.IID_TRAIN,
    )
    validation_example = SimpleNamespace(
        example_id="validation",
        checksum_sha256=HASH_B,
        view=RemediationView.IID_VALIDATION,
    )
    iid_dataset = SimpleNamespace(
        examples=(train_example, validation_example),
        manifest=SimpleNamespace(checksum_sha256=HASH_B, dataset_version="0.3.0"),
    )
    monkeypatch.setattr(pipeline, "_load_stage_dataset", lambda *_args: iid_dataset)
    frozen_selection = SimpleNamespace(checksum_sha256=HASH_E)
    monkeypatch.setattr(
        pipeline,
        "resolve_semantic_selection_examples",
        lambda *_args: (validation_example,),
    )
    monkeypatch.setattr(
        pipeline,
        "_subset_dataset",
        lambda *_args, **_kwargs: SimpleNamespace(manifest=SimpleNamespace(checksum_sha256=HASH_D)),
    )
    train_tokenized = CompactTokenizedExample(
        example_id="train",
        task_name=TaskName.FAULT_FAMILY,
        group_id="train-group",
        token_ids=(1, 2),
        target_mask=(False, True),
        prompt_token_count=1,
        target_token_count=1,
        prompt_tokens_retained=1,
        prompt_truncated=False,
    )
    validation_tokenized = replace(
        train_tokenized,
        example_id="validation",
        group_id="validation-group",
    )
    monkeypatch.setattr(
        pipeline,
        "_tokenize_examples",
        lambda examples, *_args, **_kwargs: (
            (train_tokenized,)
            if examples[0].view is RemediationView.IID_TRAIN
            else (validation_tokenized,)
        ),
    )
    monkeypatch.setattr(pipeline, "config_sha256", lambda _config: HASH_E)

    scores = (
        pipeline.CandidateScore(
            candidate_id="balanced",
            checkpoint_manifest_sha256=HASH_B,
            semantic_composite=0.5,
            selected_validation_nll=0.2,
            selected_step=2,
            evaluation_report_sha256=HASH_D,
        ),
        pipeline.CandidateScore(
            candidate_id="control",
            checkpoint_manifest_sha256=HASH_A,
            semantic_composite=0.75,
            selected_validation_nll=0.3,
            selected_step=3,
            evaluation_report_sha256=HASH_C,
        ),
    )

    def selection_report(
        candidate_scores: tuple[pipeline.CandidateScore, ...],
        selected_id: str,
        selected_checkpoint: str,
    ) -> pipeline.CandidateSelectionReport:
        draft = pipeline.CandidateSelectionReport.model_construct(
            selection_manifest_sha256=HASH_E,
            candidates=candidate_scores,
            selected_candidate_id=selected_id,
            selected_checkpoint_manifest_sha256=selected_checkpoint,
            checksum_sha256="0" * 64,
        )
        return pipeline._bound_model(draft, pipeline.CandidateSelectionReport)

    current_selection = [selection_report(scores, "control", HASH_A)]

    def artifacts(checkpoint_sha256: str, dataset_sha256: str) -> SimpleNamespace:
        return SimpleNamespace(
            source_commit=SOURCE_COMMIT,
            config_sha256=HASH_E,
            dataset_manifest_sha256=dataset_sha256,
            tokenizer_manifest_sha256=HASH_C,
            output_contract_sha256=HASH_D,
            checkpoint_sha256=checkpoint_sha256,
            prediction_artifact_sha256=HASH_A,
            comparator_artifact_sha256=HASH_B,
        )

    reports = {
        "balanced": SimpleNamespace(
            evaluation_view=RemediationView.IID_VALIDATION,
            example_count=1,
            predictions_sha256=HASH_A,
            baseline_report_sha256=HASH_B,
            checksum_sha256=HASH_D,
            derived_score=0.5,
            view_metrics=SimpleNamespace(
                view=DevelopmentView.IID_VALIDATION,
                sample_count=1,
                artifacts=artifacts(HASH_B, HASH_D),
            ),
        ),
        "control": SimpleNamespace(
            evaluation_view=RemediationView.IID_VALIDATION,
            example_count=1,
            predictions_sha256=HASH_A,
            baseline_report_sha256=HASH_B,
            checksum_sha256=HASH_C,
            derived_score=0.75,
            view_metrics=SimpleNamespace(
                view=DevelopmentView.IID_VALIDATION,
                sample_count=1,
                artifacts=artifacts(HASH_A, HASH_D),
            ),
        ),
    }
    full_report = SimpleNamespace(
        evaluation_view=RemediationView.IID_VALIDATION,
        example_count=1,
        predictions_sha256=HASH_A,
        baseline_report_sha256=HASH_B,
        checksum_sha256=HASH_E,
        view_metrics=SimpleNamespace(
            view=DevelopmentView.IID_VALIDATION,
            sample_count=1,
            artifacts=artifacts(HASH_A, HASH_B),
        ),
    )
    model_sha256 = canonical_sha256(model.model_dump(mode="json", round_trip=True))
    train_inventory_sha256 = canonical_sha256((("train", HASH_A),))
    validation_inventory_sha256 = canonical_sha256((("validation", HASH_B),))
    train_tokenized_sha256 = pipeline._tokenized_inventory_sha256((train_tokenized,))
    validation_tokenized_sha256 = pipeline._tokenized_inventory_sha256((validation_tokenized,))
    results = {
        "balanced": SimpleNamespace(
            candidate_id="balanced",
            sampling_strategy="task_balanced",
            source_commit=SOURCE_COMMIT,
            training_steps=10,
            parameter_count=123,
            vocab_size=32,
            checkpoint_manifest_sha256=HASH_B,
            training_config_sha256=canonical_sha256(
                pipeline._with_training_seed(training, 11).model_dump(mode="json", round_trip=True)
            ),
            model_config_sha256=model_sha256,
            tokenizer_manifest_sha256=HASH_C,
            train_example_count=1,
            validation_example_count=1,
            train_inventory_sha256=train_inventory_sha256,
            validation_inventory_sha256=validation_inventory_sha256,
            train_tokenized_sha256=train_tokenized_sha256,
            validation_tokenized_sha256=validation_tokenized_sha256,
            initial_validation_nll=1.2,
            selected_validation_nll=0.2,
            selected_score=0.5,
            selected_step=2,
            checkpoint_weights_sha256=HASH_D,
            checkpoint_size_bytes=2048,
        ),
        "control": SimpleNamespace(
            candidate_id="control",
            sampling_strategy="uniform_control",
            source_commit=SOURCE_COMMIT,
            training_steps=10,
            parameter_count=123,
            vocab_size=32,
            checkpoint_manifest_sha256=HASH_A,
            training_config_sha256=canonical_sha256(
                pipeline._with_training_seed(training, 7).model_dump(mode="json", round_trip=True)
            ),
            model_config_sha256=model_sha256,
            tokenizer_manifest_sha256=HASH_C,
            train_example_count=1,
            validation_example_count=1,
            train_inventory_sha256=train_inventory_sha256,
            validation_inventory_sha256=validation_inventory_sha256,
            train_tokenized_sha256=train_tokenized_sha256,
            validation_tokenized_sha256=validation_tokenized_sha256,
            initial_validation_nll=1.3,
            selected_validation_nll=0.3,
            selected_score=0.25,
            selected_step=3,
            checkpoint_weights_sha256=HASH_E,
            checkpoint_size_bytes=2048,
        ),
    }
    checkpoints = {
        candidate_id: SimpleNamespace(
            checksum_sha256=result.checkpoint_manifest_sha256,
            transformer_config=model,
            vocab_size=result.vocab_size,
            parameter_count=result.parameter_count,
            tokenizer_manifest_sha256=HASH_C,
            source_commit=SOURCE_COMMIT,
            seed=11 if candidate_id == "balanced" else 7,
            training_steps=result.selected_step,
            initial_loss=result.initial_validation_nll,
            final_loss=result.selected_validation_nll,
            weights_sha256=result.checkpoint_weights_sha256,
            weights_size_bytes=result.checkpoint_size_bytes,
        )
        for candidate_id, result in results.items()
    }

    def read(path: Path, model_type: type[object], **_kwargs: object) -> object:
        if model_type is pipeline.CandidateSelectionReport:
            return current_selection[0]
        if model_type is SemanticSelectionManifest:
            return frozen_selection
        if model_type is CheckpointManifest:
            candidate_id = path.parent.name.removeprefix("checkpoint-")
            return checkpoints[candidate_id]
        if model_type is SemanticEvaluationReport:
            if path.name == "v03-selected-full-iid-semantic-evaluation.json":
                return full_report
            candidate_id = path.name.removesuffix("-semantic-evaluation.json")
            return reports[candidate_id]
        raise AssertionError(f"unexpected contract read: {model_type}")

    monkeypatch.setattr(pipeline, "_read_contract", read)
    monkeypatch.setattr(
        pipeline,
        "_load_training_result",
        lambda _attempt, candidate_id: results[candidate_id],
    )
    monkeypatch.setattr(
        pipeline,
        "_semantic_report_composite",
        lambda report: report.derived_score,
    )
    acceptance = SimpleNamespace(advancement_allowed=True, checksum_sha256=HASH_E)
    monkeypatch.setattr(pipeline, "evaluate_v03_acceptance", lambda _metrics: acceptance)
    monkeypatch.setattr(pipeline, "_contract_artifact", lambda *_args, **_kwargs: _artifact("a"))
    context = cast(StageContext, SimpleNamespace(source_commit=SOURCE_COMMIT))

    outcome = runtime.v03_gate(context)
    assert outcome.advancement_allowed is True

    tampered_balanced = scores[0].model_copy(update={"semantic_composite": 0.9})
    current_selection[0] = selection_report(
        (tampered_balanced, scores[1]),
        "balanced",
        HASH_B,
    )
    with pytest.raises(pipeline.PipelineExecutionError, match="candidate ranking differs"):
        runtime.v03_gate(context)

    current_selection[0] = selection_report(scores, "control", HASH_A)
    results["control"].selected_step = 4
    with pytest.raises(pipeline.PipelineExecutionError, match="candidate ranking differs"):
        runtime.v03_gate(context)

    results["control"].selected_step = 3
    results["control"].validation_inventory_sha256 = HASH_C
    with pytest.raises(pipeline.PipelineExecutionError, match="candidate ranking differs"):
        runtime.v03_gate(context)

    results["control"].validation_inventory_sha256 = validation_inventory_sha256
    full_report.view_metrics.artifacts.dataset_manifest_sha256 = HASH_C
    with pytest.raises(pipeline.PipelineExecutionError, match="full-IID evaluation differs"):
        runtime.v03_gate(context)

    full_report.view_metrics.artifacts.dataset_manifest_sha256 = HASH_B
    checkpoints["control"].seed = 8
    with pytest.raises(pipeline.PipelineExecutionError, match="candidate ranking differs"):
        runtime.v03_gate(context)

    checkpoints["control"].seed = 7
    results["control"].selected_score = 0.3
    with pytest.raises(pipeline.PipelineExecutionError, match="candidate ranking differs"):
        runtime.v03_gate(context)


def test_v04_evaluation_compares_control_and_variant_on_full_iid_and_all_shadows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    train = (SimpleNamespace(view=RemediationView.IID_TRAIN),)
    iid_validation = (
        SimpleNamespace(view=RemediationView.IID_VALIDATION),
        SimpleNamespace(view=RemediationView.IID_VALIDATION),
    )
    iid = SimpleNamespace(examples=(*train, *iid_validation))
    shadow_examples = tuple(SimpleNamespace(view=view) for view in SHADOW_VIEWS)
    shadow = SimpleNamespace(examples=shadow_examples)
    selection_rule = (
        "all_gates_then_highest_min_view_composite_then_iid_composite_then_shorter_context"
    )
    inputs = SimpleNamespace(
        v02=SimpleNamespace(model=SimpleNamespace(context_length=512)),
        v03=object(),
        v04=SimpleNamespace(
            variants=SimpleNamespace(context_candidate_selection_rule=selection_rule),
            longer_context_model=SimpleNamespace(context_length=1024),
        ),
        tokenizer=object(),
    )
    runtime = pipeline._PipelineRuntime(
        project_root=tmp_path,
        config=config,
        source_commit=SOURCE_COMMIT,
        inputs=cast(pipeline._ExecutionInputs, inputs),
    )
    monkeypatch.setattr(runtime, "_start", lambda _context: SOURCE_COMMIT)
    monkeypatch.setattr(runtime, "_finish", lambda _context, outcome: outcome)
    monkeypatch.setattr(pipeline, "config_sha256", lambda _config: HASH_A)
    evaluation_dataset_loads: list[str] = []

    def load_evaluation_dataset(
        _context: object,
        _config: object,
        stage: str,
        _name: str,
    ) -> object:
        evaluation_dataset_loads.append(stage)
        return iid if stage == "v03_data_audit" else shadow

    monkeypatch.setattr(pipeline, "_load_stage_dataset", load_evaluation_dataset)
    upstream_calls: list[str] = []

    def upstream(_context: object, _config: object, stage: str) -> Path:
        upstream_calls.append(stage)
        return tmp_path / stage

    monkeypatch.setattr(pipeline, "_upstream_attempt", upstream)
    monkeypatch.setattr(
        pipeline,
        "_read_contract",
        lambda *_args, **_kwargs: SimpleNamespace(selected_candidate_id="control"),
    )
    control_result = SimpleNamespace()
    monkeypatch.setattr(pipeline, "_load_training_result", lambda *_args: control_result)
    control_model = SimpleNamespace(config=SimpleNamespace(context_length=512))
    control_checkpoint = SimpleNamespace(checksum_sha256=HASH_D)
    monkeypatch.setattr(
        pipeline,
        "_load_candidate_checkpoint",
        lambda *_args: (control_model, control_checkpoint, torch.device("cpu")),
    )
    variant_model = SimpleNamespace(config=SimpleNamespace(context_length=1024))
    variant_checkpoint = SimpleNamespace(checksum_sha256=HASH_E)
    monkeypatch.setattr(
        runtime,
        "_v04_checkpoint",
        lambda _context: (
            SimpleNamespace(activated=True, candidate_id="variant"),
            SimpleNamespace(),
            variant_model,
            variant_checkpoint,
            torch.device("cpu"),
        ),
    )
    evaluation_calls: list[tuple[int, RemediationView, int]] = []

    def evaluate(*_args: object, **kwargs: object) -> tuple[object, None, tuple[object], tuple[()]]:
        model = cast(SimpleNamespace, kwargs["model"])
        view = cast(RemediationView, kwargs["view"])
        examples = cast(tuple[object, ...], kwargs["evaluation_examples"])
        evaluation_calls.append((model.config.context_length, view, len(examples)))
        score = 0.8 if model.config.context_length == 512 else 0.7
        checksum = hashlib.sha256(repr(evaluation_calls[-1]).encode()).hexdigest()
        return (
            SimpleNamespace(
                checksum_sha256=checksum,
                view_metrics=SimpleNamespace(),
                score=score,
            ),
            None,
            (SimpleNamespace(score=score),),
            (),
        )

    monkeypatch.setattr(pipeline, "_evaluate_candidate_view", evaluate)
    monkeypatch.setattr(
        pipeline,
        "_semantic_report_composite",
        lambda evaluation: evaluation.score,
    )
    iid_acceptances: list[object] = []

    def iid_acceptance(_metrics: object) -> object:
        result = SimpleNamespace(checksum_sha256=HASH_B)
        iid_acceptances.append(result)
        return result

    monkeypatch.setattr(pipeline, "evaluate_v03_acceptance", iid_acceptance)
    v04_inputs: list[object] = []

    def v04_acceptance(current_iid: object, _shadow: object) -> object:
        v04_inputs.append(current_iid)
        return SimpleNamespace(
            checksum_sha256=HASH_C,
            advancement_allowed=current_iid is iid_acceptances[0],
        )

    monkeypatch.setattr(pipeline, "evaluate_v04_acceptance", v04_acceptance)
    written: list[tuple[str, object]] = []

    def artifact(_context: object, filename: str, model: object) -> ArtifactReference:
        written.append((filename, model))
        return ArtifactReference(
            relative_path=f"stages/{filename}",
            sha256=hashlib.sha256(filename.encode()).hexdigest(),
            size_bytes=1,
        )

    monkeypatch.setattr(pipeline, "_contract_artifact", artifact)

    runtime.v04_shadow_evaluation(cast(StageContext, SimpleNamespace()))

    assert len(evaluation_calls) == 2 * (1 + len(SHADOW_VIEWS))
    assert evaluation_dataset_loads == ["v03_data_audit", "v04_shadow_freeze"]
    for context_length in (512, 1024):
        candidate_calls = tuple(call for call in evaluation_calls if call[0] == context_length)
        assert candidate_calls[0] == (
            context_length,
            RemediationView.IID_VALIDATION,
            len(iid_validation),
        )
        assert tuple(call[1] for call in candidate_calls[1:]) == SHADOW_VIEWS
    assert v04_inputs == iid_acceptances
    assert "v03_gate" not in upstream_calls
    indices = tuple(model for name, model in written if name == "v04-evaluation-index.json")
    assert len(indices) == 1
    index = cast(pipeline.V04EvaluationIndex, indices[0])
    assert len(index.candidates) == 2
    assert index.selected_candidate_id == "control"


def test_public_pipeline_contracts_are_explicitly_exported() -> None:
    required = {
        "build_stage_actions",
        "write_terminal_review_bundle",
        "run_final_evaluation",
        "FinalEvaluationBlockedError",
    }
    assert required <= set(pipeline.__all__)
    assert callable(pipeline.build_stage_actions)
    assert isinstance(MappingProxyType({}), Mapping)


def _artifact(name: str) -> ArtifactReference:
    return ArtifactReference(
        relative_path=f"stages/{name}",
        sha256=hashlib.sha256(name.encode()).hexdigest(),
        size_bytes=1,
    )


def _structured_records_for_views(
    *,
    collision: tuple[RemediationView, RemediationView] | None = None,
) -> tuple[TaskScopedStructuredFingerprint, ...]:
    collision_fingerprint = hashlib.sha256(b"metamorphic-collision").hexdigest()
    return tuple(
        TaskScopedStructuredFingerprint(
            example_id=f"different-id-{view.value}",
            view=view,
            task_name=TaskName.FAULT_FAMILY,
            structured_fingerprint_sha256=(
                collision_fingerprint
                if collision is not None and view in collision
                else hashlib.sha256(f"structured-{view.value}".encode()).hexdigest()
            ),
        )
        for view in RemediationView
    )


def _metamorphic_dataset_pair() -> tuple[object, object]:
    examples = tuple(
        SimpleNamespace(
            example_id=f"different-id-{view.value}",
            view=view,
            group_id=f"different-group-{view.value}",
            checksum_sha256=hashlib.sha256(f"different-example-{view.value}".encode()).hexdigest(),
            prompt_sha256=hashlib.sha256(f"different-prompt-{view.value}".encode()).hexdigest(),
        )
        for view in RemediationView
    )
    iid = SimpleNamespace(
        examples=examples[:2],
        manifest=SimpleNamespace(checksum_sha256=HASH_A),
    )
    shadow = SimpleNamespace(
        examples=examples[2:],
        manifest=SimpleNamespace(checksum_sha256=HASH_B),
    )
    return iid, shadow


def _runtime_context(tmp_path: Path) -> StageContext:
    run = tmp_path / "work" / "runtime"
    attempt = run / "stages/01-preflight/attempt-0001"
    attempt.mkdir(parents=True)
    return cast(
        StageContext,
        SimpleNamespace(
            project_root=tmp_path,
            run_directory=run,
            attempt_directory=attempt,
            source_commit=SOURCE_COMMIT,
            progress=_FakeProgress(),
            stop_requested=lambda: False,
        ),
    )


def test_task_scoped_structured_separation_allows_same_input_for_different_tasks() -> None:
    fingerprint = hashlib.sha256(b"shared-structured-input").hexdigest()
    records = (
        TaskScopedStructuredFingerprint(
            example_id="train-example",
            view=RemediationView.IID_TRAIN,
            task_name=TaskName.FAULT_FAMILY,
            structured_fingerprint_sha256=fingerprint,
        ),
        TaskScopedStructuredFingerprint(
            example_id="validation-example",
            view=RemediationView.IID_VALIDATION,
            task_name=TaskName.NEXT_ACTION,
            structured_fingerprint_sha256=fingerprint,
        ),
    )

    report = pipeline._task_scoped_structured_separation(
        records,
        views=(RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION),
    )
    reordered = pipeline._task_scoped_structured_separation(
        tuple(reversed(records)),
        views=(RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION),
    )

    assert report.passed
    assert report.overlap_count == 0
    assert report.checksum_sha256 == reordered.checksum_sha256


@pytest.mark.parametrize(
    "collision",
    [
        (RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION),
        (RemediationView.IID_VALIDATION, RemediationView.SHADOW_RENDERER),
    ],
)
def test_structured_gate_rejects_cross_view_match_despite_rendered_differences(
    collision: tuple[RemediationView, RemediationView],
) -> None:
    """Different IDs/prompts/example checksums cannot hide structured reuse."""

    iid, shadow = _metamorphic_dataset_pair()
    structured = pipeline._task_scoped_structured_separation(
        _structured_records_for_views(collision=collision),
        views=tuple(RemediationView),
    )
    report = pipeline._development_separation_report(
        cast(SafeDevelopmentDataset, iid),
        cast(SafeDevelopmentDataset, shadow),
        structured,
    )

    assert report.group_overlap_count == 0
    assert report.example_checksum_overlap_count == 0
    assert report.prompt_checksum_overlap_count == 0
    assert report.structured_separation.overlap_count == 1
    assert not report.passed
    assert any(
        item.first_view is collision[0]
        and item.second_view is collision[1]
        and item.overlap_count == 1
        for item in report.structured_separation.pairwise_overlaps
    )


def test_structured_inventory_checksum_and_regenerated_iid_are_tamper_evident() -> None:
    structured = pipeline._task_scoped_structured_separation(
        _structured_records_for_views(),
        views=tuple(RemediationView),
    )
    payload = structured.model_dump(mode="json", round_trip=True)
    cast(list[dict[str, object]], payload["inventories"])[0]["inventory_sha256"] = HASH_E
    with pytest.raises(ValidationError, match="structured separation checksum mismatch"):
        pipeline.TaskScopedStructuredSeparationReport.model_validate_json(
            canonical_json_bytes(payload), strict=True
        )

    committed = cast(SafeDevelopmentDataset, SimpleNamespace(identity="committed"))
    pipeline._require_exact_regenerated_iid(committed, committed)
    with pytest.raises(pipeline.PipelineExecutionError, match="regenerated IID material"):
        pipeline._require_exact_regenerated_iid(
            cast(SafeDevelopmentDataset, SimpleNamespace(identity="tampered")),
            committed,
        )


def test_v03_cap_compatibility_bridge_binds_exact_counterfactual_evidence() -> None:
    class CountedRows:
        def __init__(self, rows: tuple[object, ...], reported_length: int) -> None:
            self.rows = rows
            self.reported_length = reported_length

        def __iter__(self) -> Iterator[object]:
            return iter(self.rows)

        def __len__(self) -> int:
            return self.reported_length

    counterfactuals = tuple(
        SimpleNamespace(
            example_id=f"counterfactual-{index:03d}",
            checksum_sha256=hashlib.sha256(f"cf-{index}".encode()).hexdigest(),
            view=(RemediationView.IID_TRAIN if index < 40 else RemediationView.IID_VALIDATION),
            task_name=TaskName.COUNTERFACTUAL_COMPARE,
            prompt_sha256=hashlib.sha256(f"cf-prompt-{index}".encode()).hexdigest(),
            canonical_target_json=f'{{"counterfactual":{index}}}',
        )
        for index in range(55)
    )
    retained = tuple(
        SimpleNamespace(
            example_id=f"retained-{index:03d}",
            checksum_sha256=hashlib.sha256(f"retained-{index}".encode()).hexdigest(),
            view=RemediationView.IID_TRAIN,
            task_name=TaskName.FAULT_FAMILY,
            prompt_sha256=hashlib.sha256(f"prompt-{index}".encode()).hexdigest(),
            canonical_target_json="{}",
        )
        for index in range(24)
    )
    removed = tuple(
        SimpleNamespace(
            example_id=f"removed-{index:03d}",
            checksum_sha256=hashlib.sha256(f"removed-{index}".encode()).hexdigest(),
            view=RemediationView.IID_TRAIN,
            task_name=TaskName.FAULT_FAMILY,
            prompt_sha256=hashlib.sha256(f"prompt-{index}".encode()).hexdigest(),
            canonical_target_json="{}",
        )
        for index in range(24)
    )
    raw_dataset = SimpleNamespace(
        examples=CountedRows((*counterfactuals, *retained, *removed), 5_859),
        manifest=SimpleNamespace(
            checksum_sha256=HASH_A,
            inventory_sha256=HASH_B,
        ),
    )
    deduplicated_dataset = SimpleNamespace(
        examples=CountedRows((*counterfactuals, *retained), 5_835),
        manifest=SimpleNamespace(
            checksum_sha256=HASH_C,
            inventory_sha256=HASH_D,
        ),
    )
    material = object.__new__(FrozenV03IIDMaterial)
    object.__setattr__(material, "raw_dataset", raw_dataset)
    object.__setattr__(material, "dataset", deduplicated_dataset)
    object.__setattr__(material, "structured_fingerprints", ())
    frozen = CounterfactualCapExtensionReport.model_construct(
        dataset_manifest_sha256=HASH_A,
        checksum_sha256=HASH_E,
    )
    deduplicated = CounterfactualCapExtensionReport.model_construct(
        dataset_manifest_sha256=HASH_C,
        checksum_sha256="f" * 64,
    )

    report = pipeline._v03_cap_compatibility_report(
        material,
        frozen_cap=frozen,
        raw_cap=frozen,
        deduplicated_cap=deduplicated,
    )

    assert report.passed
    assert report.raw_example_count == 5_859
    assert report.deduplicated_example_count == 5_835
    assert report.removed_example_count == 24
    assert len(report.removed_examples) == 24
    assert tuple(item.removed_example_id for item in report.removed_examples) == tuple(
        f"removed-{index:03d}" for index in range(24)
    )
    assert tuple(item.retained_example_id for item in report.removed_examples) == tuple(
        f"retained-{index:03d}" for index in range(24)
    )
    assert report.counterfactual_train_count == 40
    assert report.counterfactual_validation_count == 15
    assert (
        report.raw_counterfactual_inventory_sha256
        == report.deduplicated_counterfactual_inventory_sha256
    )
    assert (
        report.raw_counterfactual_evidence_sha256
        == report.deduplicated_counterfactual_evidence_sha256
    )
    with pytest.raises(pipeline.PipelineExecutionError, match="did not reproduce"):
        pipeline._v03_cap_compatibility_report(
            material,
            frozen_cap=frozen,
            raw_cap=deduplicated,
            deduplicated_cap=deduplicated,
        )
    payload = report.model_dump(mode="python", round_trip=True)
    payload["deduplicated_counterfactual_inventory_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="compatibility state"):
        pipeline.V03CounterfactualCapCompatibilityReport.model_validate(payload)
    payload = report.model_dump(mode="python", round_trip=True)
    cast(list[dict[str, object]], payload["removed_examples"])[0]["retained_prompt_sha256"] = HASH_A
    with pytest.raises(ValidationError, match="removed-row binding"):
        pipeline.V03CounterfactualCapCompatibilityReport.model_validate(payload)


def test_evidence_removal_deduplication_is_deterministic_and_conflict_safe() -> None:
    first = cast(
        RemediationExample,
        SimpleNamespace(
            example_id="example-a",
            task_name=TaskName.NEXT_ACTION,
            prompt_sha256=HASH_A,
            canonical_target_json='{"action":"insufficient"}',
        ),
    )
    duplicate = cast(
        RemediationExample,
        SimpleNamespace(
            example_id="example-b",
            task_name=TaskName.NEXT_ACTION,
            prompt_sha256=HASH_A,
            canonical_target_json='{"action":"insufficient"}',
        ),
    )
    before = ((duplicate, HASH_B), (first, HASH_B))
    after = remediation_data._deduplicate_evidence_removal_examples(before)

    assert len(before) == 2
    assert len(after) == 1
    assert after[0][0].example_id == "example-a"
    assert remediation_data._deduplicate_evidence_removal_examples(tuple(reversed(before))) == after

    conflicting = cast(
        RemediationExample,
        SimpleNamespace(
            example_id="example-c",
            task_name=TaskName.NEXT_ACTION,
            prompt_sha256=HASH_A,
            canonical_target_json='{"action":"different"}',
        ),
    )
    with pytest.raises(ValueError, match="conflicting compact targets"):
        remediation_data._deduplicate_evidence_removal_examples(
            ((first, HASH_B), (conflicting, HASH_B))
        )


def test_remediation_example_rejects_rebound_classification_label_tamper() -> None:
    context = CompactTargetContext(
        task_name=TaskName.NEXT_ACTION,
        visible_fact_refs=("o-0000",),
    )
    example = remediation_data._make_example(
        view=RemediationView.IID_TRAIN,
        source_split=SplitName.IID_TRAIN,
        source_record_ids=("projection:test",),
        parent_record_sha256=HASH_A,
        group_id="source:test:next_action",
        prompt_text="Fictional observation o-0000.",
        template_family=TemplateFamily.COMPACT_LOG,
        alias_family=AliasFamily.CANONICAL,
        context=context,
        target=NextActionTarget(immediate_action=ActionLabel.CONTINUE_MONITORING),
        augmentation="none",
    )
    assert example.classification_label == ActionLabel.CONTINUE_MONITORING.value

    payload = example.model_dump(mode="python", exclude={"checksum_sha256"})
    payload["classification_label"] = ActionLabel.INSUFFICIENT_EVIDENCE.value
    payload["checksum_sha256"] = canonical_sha256(payload)
    with pytest.raises(ValidationError, match="classification label differs"):
        RemediationExample.model_validate(payload)


def test_v02_development_selection_callback_uses_only_structural_iid_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _runtime_context(tmp_path)
    train = SimpleNamespace(view=RemediationView.IID_TRAIN)
    validation = SimpleNamespace(view=RemediationView.IID_VALIDATION)
    dataset = SimpleNamespace(examples=(train, validation))
    inputs = cast(
        pipeline._ExecutionInputs,
        SimpleNamespace(
            v02=SimpleNamespace(model=object(), training=object()),
            tokenizer=object(),
            generation_caps={},
        ),
    )
    runtime = pipeline._PipelineRuntime(
        project_root=tmp_path,
        config=_config("v02-structural-selection"),
        source_commit=SOURCE_COMMIT,
        inputs=inputs,
    )
    monkeypatch.setattr(runtime.guard, "stop_required", lambda _context: False)
    monkeypatch.setattr(runtime, "_start", lambda _context: SOURCE_COMMIT)
    monkeypatch.setattr(runtime, "_finish", lambda _context, outcome: outcome)
    monkeypatch.setattr(pipeline, "_load_stage_dataset", lambda *_args: dataset)
    captured: dict[str, object] = {}

    def run_training(*_args: object, **kwargs: object) -> tuple[object, tuple[object, ...]]:
        captured.update(kwargs)
        return SimpleNamespace(selected_validation_nll=9.0), ()

    monkeypatch.setattr(pipeline, "_run_training", run_training)
    runtime.v02_development_training(context)
    assert captured["validation_examples"] == (validation,)
    callback = cast(Callable[[TransformerLM, int, float], float], captured["evaluation_callback"])

    good_path = SimpleNamespace(
        compact_parse_success=True,
        schema_valid=True,
        generation_cap_exhausted=False,
    )
    bad_path = SimpleNamespace(
        compact_parse_success=False,
        schema_valid=False,
        generation_cap_exhausted=True,
    )
    decode_results = [
        (SimpleNamespace(constrained=bad_path, unconstrained=bad_path),),
        (SimpleNamespace(constrained=good_path, unconstrained=good_path),),
    ]

    def decode(
        _model: object,
        _tokenizer: object,
        examples: tuple[object, ...],
        **_kwargs: object,
    ) -> object:
        assert examples == (validation,)
        return decode_results.pop(0)

    monkeypatch.setattr(pipeline, "_decode_examples", decode)
    model = cast(TransformerLM, torch.nn.Linear(1, 1))
    bad_score = callback(model, 1, 0.01)
    good_score = callback(model, 2, 9.0)

    assert bad_score == 1.0
    assert good_score == 0.0
    # The training contract selects (score, NLL, step), so structure outranks NLL.
    assert min((bad_score, 0.01, 1), (good_score, 9.0, 2)) == (good_score, 9.0, 2)


def test_each_stage_rechecks_source_before_loading_scientific_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _runtime_context(tmp_path)
    runtime = pipeline._PipelineRuntime(
        project_root=tmp_path,
        config=_config("stage-source-recheck"),
        source_commit=SOURCE_COMMIT,
        inputs=cast(
            pipeline._ExecutionInputs,
            SimpleNamespace(v02_dataset_config=object(), frozen_data_source_commit=SOURCE_COMMIT),
        ),
    )
    monkeypatch.setattr(runtime.guard, "enforce_start", lambda _context: None)
    source_is_dirty = [False]

    def verify(*_args: object, **_kwargs: object) -> str:
        if source_is_dirty[0]:
            raise pipeline.PipelineExecutionError(
                "runner Git worktree contains uncommitted source changes"
            )
        return SOURCE_COMMIT

    monkeypatch.setattr(pipeline, "_verify_runner_source", verify)
    assert runtime._start(context) == SOURCE_COMMIT
    source_is_dirty[0] = True
    scientific_input_loaded = [False]

    def forbidden_builder(*_args: object, **_kwargs: object) -> None:
        scientific_input_loaded[0] = True

    monkeypatch.setattr(pipeline, "build_safe_development_dataset", forbidden_builder)
    with pytest.raises(pipeline.PipelineExecutionError, match="uncommitted source changes"):
        runtime.v02_inventory_and_caps(context)
    assert not scientific_input_loaded[0]


def test_runtime_executes_v02_v03_and_shadow_freeze_happy_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the complete pre-v0.4 development graph without scientific work."""

    context = _runtime_context(tmp_path)
    inventory = SimpleNamespace(
        checksum_sha256=HASH_A,
        example_count=882,
        prompt_truncation_count=668,
        prompt_truncation_rate=668 / 882,
        target_fit_rate=1.0,
        round_trip_rate=1.0,
        reachability_rate=1.0,
        task_footer_retained_rate=1.0,
        cap_exhaustion_target_rate=0.0,
    )
    counterfactual = SimpleNamespace(checksum_sha256=HASH_B)
    tokenizer = SimpleNamespace(manifest=SimpleNamespace(checksum_sha256=HASH_C))
    train = SimpleNamespace(
        example_id="train-example",
        view=RemediationView.IID_TRAIN,
        task_name=TaskName.FAULT_FAMILY,
        group_id="train-group",
        checksum_sha256=HASH_A,
        prompt_sha256=HASH_B,
    )
    validation = SimpleNamespace(
        example_id="validation-example",
        view=RemediationView.IID_VALIDATION,
        task_name=TaskName.FAULT_FAMILY,
        group_id="validation-group",
        checksum_sha256=HASH_C,
        prompt_sha256=HASH_D,
        canonical_target_json="{}",
    )
    iid = SimpleNamespace(
        examples=(train, validation),
        manifest=SimpleNamespace(checksum_sha256=HASH_A),
    )
    shadow_examples = tuple(
        SimpleNamespace(
            example_id=f"example-{view.value}",
            view=view,
            task_name=TaskName.FAULT_FAMILY,
            group_id=f"shadow-{view.value}",
            checksum_sha256=hashlib.sha256(f"example-{view.value}".encode()).hexdigest(),
            prompt_sha256=hashlib.sha256(f"prompt-{view.value}".encode()).hexdigest(),
        )
        for view in SHADOW_VIEWS
    )
    shadow = SimpleNamespace(
        examples=shadow_examples,
        manifest=SimpleNamespace(checksum_sha256=HASH_B),
    )
    candidate = SimpleNamespace(candidate_id="control", sampling="task_balanced", seed=7)
    inputs = cast(
        pipeline._ExecutionInputs,
        SimpleNamespace(
            v02=SimpleNamespace(model=object(), training=object(), inventory=object()),
            v03=SimpleNamespace(
                training=object(),
                candidates=(candidate,),
                selection=load_v03_config(
                    Path(__file__).resolve().parents[2]
                    / "configs/experiments/phase6-remediation-v0.3.2-focused.toml"
                ).selection,
                augmentation=SimpleNamespace(
                    train_template_families=("a",),
                    train_alias_families=("b",),
                    renderer_variants_per_projection=1,
                    include_insufficient_evidence_views=True,
                ),
            ),
            v04=object(),
            v02_dataset_config=object(),
            v03_dataset_config=object(),
            tokenizer=tokenizer,
            compact_contract_sha256=HASH_D,
            frozen_v02_inventory=inventory,
            frozen_v03_counterfactual_cap=counterfactual,
            frozen_data_source_commit=SOURCE_COMMIT,
            generation_caps={},
        ),
    )
    runtime = pipeline._PipelineRuntime(
        project_root=tmp_path,
        config=_config("runtime"),
        source_commit=SOURCE_COMMIT,
        inputs=inputs,
    )
    monkeypatch.setattr(runtime.guard, "stop_required", lambda _context: False)
    monkeypatch.setattr(runtime, "_start", lambda _context: SOURCE_COMMIT)
    monkeypatch.setattr(runtime, "_finish", lambda _context, outcome: outcome)
    monkeypatch.setattr(pipeline, "config_sha256", lambda _value: HASH_E)
    monkeypatch.setattr(pipeline, "_verify_runner_source", lambda *_args, **_kwargs: SOURCE_COMMIT)
    monkeypatch.setattr(
        pipeline,
        "_contract_artifact",
        lambda _context, filename, _model: _artifact(filename),
    )
    monkeypatch.setattr(
        pipeline,
        "_directory_artifacts",
        lambda directory, **_kwargs: (_artifact(f"{directory.name}/file"),),
    )
    built_views: list[tuple[RemediationView, ...]] = []

    def build_dataset(*_args: object, **kwargs: object) -> object:
        views = cast(tuple[RemediationView, ...], kwargs["views"])
        built_views.append(views)
        return shadow if views == SHADOW_VIEWS else iid

    monkeypatch.setattr(pipeline, "build_safe_development_dataset", build_dataset)

    def build_dataset_with_fingerprints(
        *_args: object, **kwargs: object
    ) -> tuple[object, tuple[TaskScopedStructuredFingerprint, ...]]:
        dataset = build_dataset(*_args, **kwargs)
        examples = cast(tuple[SimpleNamespace, ...], cast(SimpleNamespace, dataset).examples)
        fingerprints = tuple(
            TaskScopedStructuredFingerprint(
                example_id=cast(str, item.example_id),
                view=cast(RemediationView, item.view),
                task_name=cast(TaskName, item.task_name),
                structured_fingerprint_sha256=hashlib.sha256(
                    f"structured-{item.view.value}".encode()
                ).hexdigest(),
            )
            for item in examples
        )
        return dataset, fingerprints

    monkeypatch.setattr(
        pipeline,
        "build_safe_development_dataset_with_structured_fingerprints",
        build_dataset_with_fingerprints,
    )
    iid_with_fingerprints = build_dataset_with_fingerprints(
        views=(RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION)
    )
    frozen_iid_calls: list[str] = []

    def build_frozen_iid(*_args: object, **_kwargs: object) -> object:
        frozen_iid_calls.append("iid")
        return SimpleNamespace(
            raw_dataset=iid,
            dataset=iid_with_fingerprints[0],
            structured_fingerprints=iid_with_fingerprints[1],
        )

    monkeypatch.setattr(pipeline, "build_frozen_v03_iid_material", build_frozen_iid)

    def write_dataset(_dataset: object, directory: Path) -> None:
        directory.mkdir()

    monkeypatch.setattr(pipeline, "write_safe_development_artifact", write_dataset)
    monkeypatch.setattr(pipeline, "measure_compact_inventory", lambda *_args: inventory)
    monkeypatch.setattr(pipeline, "_load_stage_dataset", lambda *_args: iid)
    monkeypatch.setattr(pipeline, "_smoke_model_config", lambda model: model)
    monkeypatch.setattr(pipeline, "_smoke_training_config", lambda training: training)
    training_result = SimpleNamespace(
        final_training_nll=0.4,
        selected_validation_nll=0.3,
        selected_step=2,
        checksum_sha256=HASH_D,
        checkpoint_manifest_sha256=HASH_E,
        checkpoint_weights_sha256=HASH_A,
        checkpoint_size_bytes=2048,
        process_peak_rss_bytes=1024,
        mps_peak_current_allocated_bytes=0,
        mps_peak_driver_allocated_bytes=0,
    )
    training_calls: list[str] = []

    def run_training(*_args: object, **kwargs: object) -> tuple[object, tuple[ArtifactReference]]:
        training_calls.append(cast(str, kwargs["candidate_id"]))
        callback = kwargs.get("evaluation_callback")
        if callback is not None:
            cast(Callable[[TransformerLM, int, float], float], callback)(
                cast(TransformerLM, torch.nn.Linear(1, 1)),
                1,
                0.5,
            )
        return training_result, (_artifact(f"training-{kwargs['candidate_id']}"),)

    monkeypatch.setattr(pipeline, "_run_training", run_training)
    monkeypatch.setattr(
        pipeline,
        "_load_candidate_checkpoint",
        lambda *_args: (object(), SimpleNamespace(checksum_sha256=HASH_E), torch.device("cpu")),
    )
    monkeypatch.setattr(pipeline, "_upstream_attempt", lambda *_args: tmp_path)
    monkeypatch.setattr(pipeline, "_load_training_result", lambda *_args: training_result)
    constrained = SimpleNamespace(
        compact_parse_success=True,
        schema_valid=True,
        generation_cap_exhausted=False,
        canonical_target_json="{}",
        elapsed_seconds=0.01,
    )
    unconstrained = SimpleNamespace(
        compact_parse_success=False,
        schema_valid=False,
        generation_cap_exhausted=False,
        canonical_target_json=None,
        elapsed_seconds=0.02,
    )
    prediction = SimpleNamespace(
        example_id="validation-example",
        constrained=constrained,
        unconstrained=unconstrained,
    )
    monkeypatch.setattr(pipeline, "_decode_examples", lambda *_args, **_kwargs: (prediction,))
    monkeypatch.setattr(pipeline, "semantic_composite_score", lambda *_args, **_kwargs: 0.75)
    monkeypatch.setattr(
        pipeline,
        "_write_predictions",
        lambda *_args, **_kwargs: (
            SimpleNamespace(checksum_sha256=HASH_C),
            (_artifact("prediction-manifest"), _artifact("predictions")),
        ),
    )
    audit = SimpleNamespace(passed=True)
    selection = SimpleNamespace(selected_example_count=48, checksum_sha256=HASH_C)
    monkeypatch.setattr(pipeline, "audit_safe_development_dataset", lambda _dataset: audit)
    monkeypatch.setattr(
        pipeline,
        "measure_counterfactual_cap_extension",
        lambda *_args: counterfactual,
    )
    monkeypatch.setattr(
        pipeline,
        "_v03_cap_compatibility_report",
        lambda *_args, **_kwargs: SimpleNamespace(passed=True, removed_example_count=24),
    )
    monkeypatch.setattr(pipeline, "build_semantic_selection_manifest", lambda *_args: selection)
    monkeypatch.setattr(pipeline, "load_safe_development_artifact", lambda _path: iid)
    monkeypatch.setattr(
        pipeline,
        "_read_contract",
        lambda *_args, **_kwargs: SimpleNamespace(view_metrics=object()),
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_semantic_selection_examples",
        lambda *_args: (validation,),
    )
    monkeypatch.setattr(pipeline, "_with_training_seed", lambda training, _seed: training)
    acceptance = SimpleNamespace(advancement_allowed=True, checksum_sha256=HASH_A)
    monkeypatch.setattr(pipeline, "evaluate_v03_acceptance", lambda _metrics: acceptance)
    monkeypatch.setattr(
        runtime,
        "v03_gate",
        lambda _context: StageOutcome(summary="v0.3 gate covered by focused provenance tests."),
    )

    outcomes = (
        runtime.preflight(context),
        runtime.v02_inventory_and_caps(context),
        runtime.v02_smoke(context),
        runtime.v02_development_training(context),
        runtime.v02_development_gate(context),
        runtime.v03_data_audit(context),
        runtime.v03_smoke(context),
        runtime.v03_candidate_training(context),
        runtime.v03_gate(context),
    )
    monkeypatch.setattr(pipeline, "_load_stage_dataset", lambda *_args: iid)
    # Shadow freeze uses the distinct dataset returned by the builder and the IID loader.
    shadow_outcome = runtime.v04_shadow_freeze(context)

    assert all(outcome.advancement_allowed for outcome in (*outcomes, shadow_outcome))
    assert built_views == [
        (RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION),
        (RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION),
        SHADOW_VIEWS,
    ]
    assert frozen_iid_calls == ["iid", "iid"]
    assert training_calls == ["v02-smoke", "v02-development", "v03-smoke", "control"]


def test_v04_native_mps_candidate_training_uses_only_iid_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _runtime_context(tmp_path)
    train_examples = tuple(
        SimpleNamespace(
            example_id=f"train-{task.value}",
            view=RemediationView.IID_TRAIN,
            task_name=task,
            group_id=f"train-group-{task.value}",
        )
        for task in TaskName
    )
    validation_examples = tuple(
        SimpleNamespace(
            example_id=f"validation-{task.value}",
            view=RemediationView.IID_VALIDATION,
            task_name=task,
            group_id=f"validation-group-{task.value}",
        )
        for task in TaskName
    )
    validation = validation_examples[0]
    iid = SimpleNamespace(examples=(*train_examples, *validation_examples))
    shadow = SimpleNamespace(examples=tuple(SimpleNamespace(view=view) for view in SHADOW_VIEWS))
    base_training = SimpleNamespace(
        seed=1,
        device="mps",
        allow_cpu_fallback=True,
        steps=4,
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
    )
    inputs = cast(
        pipeline._ExecutionInputs,
        SimpleNamespace(
            v02=SimpleNamespace(model=SimpleNamespace(context_length=512)),
            v03=object(),
            v04=SimpleNamespace(
                variants=SimpleNamespace(material_prompt_truncation_rate=0.1),
                pilot=SimpleNamespace(candidate_id="long", steps=10, batch_sizes=(1, 2, 4)),
                training=base_training,
                longer_context_model=SimpleNamespace(context_length=1024),
            ),
            tokenizer=object(),
            frozen_v02_inventory=SimpleNamespace(
                example_count=882,
                prompt_truncation_count=668,
                prompt_truncation_rate=668 / 882,
            ),
            generation_caps={},
        ),
    )
    runtime = pipeline._PipelineRuntime(
        project_root=tmp_path,
        config=_config("runtime"),
        source_commit=SOURCE_COMMIT,
        inputs=inputs,
    )
    monkeypatch.setattr(runtime.guard, "stop_required", lambda _context: False)
    monkeypatch.setattr(runtime, "_start", lambda _context: SOURCE_COMMIT)
    monkeypatch.setattr(runtime, "_finish", lambda _context, outcome: outcome)
    monkeypatch.setattr(pipeline, "_load_stage_dataset", lambda *_args: iid)
    truncate_prompts = [True]

    def tokenize(
        examples: tuple[SimpleNamespace, ...],
        _tokenizer: object,
        *,
        context_length: int,
        generation_caps: object,
    ) -> tuple[CompactTokenizedExample, ...]:
        del generation_caps
        return tuple(
            CompactTokenizedExample(
                example_id=cast(str, example.example_id),
                task_name=cast(TaskName, example.task_name),
                group_id=cast(str, example.group_id),
                token_ids=tuple(range(20 + index)),
                target_mask=(*([False] * (19 + index)), True),
                prompt_token_count=(
                    20 + index if truncate_prompts[0] and context_length == 512 else 19 + index
                ),
                target_token_count=1,
                prompt_tokens_retained=19 + index,
                prompt_truncated=truncate_prompts[0] if context_length == 512 else False,
            )
            for index, example in enumerate(examples)
        )

    monkeypatch.setattr(pipeline, "_tokenize_examples", tokenize)
    pilot_train_tokenized = tokenize(
        train_examples,
        object(),
        context_length=1024,
        generation_caps={},
    )
    pilot_validation_tokenized = tokenize(
        validation_examples,
        object(),
        context_length=1024,
        generation_caps={},
    )
    result = SimpleNamespace(
        checksum_sha256=HASH_A,
        checkpoint_manifest_sha256=HASH_B,
        training_config_sha256=HASH_C,
        model_config_sha256=HASH_D,
        tokenizer_manifest_sha256=HASH_E,
        device=DeviceResolution(requested="mps", resolved="mps", fallback_used=False),
        train_example_count=len(train_examples),
        validation_example_count=len(validation_examples),
        train_tokenized_sha256=pipeline._tokenized_inventory_sha256(pilot_train_tokenized),
        validation_tokenized_sha256=pipeline._tokenized_inventory_sha256(
            pilot_validation_tokenized
        ),
        elapsed_seconds=1.0,
        process_peak_rss_bytes=1,
    )
    training_calls: list[tuple[str, int]] = []

    def run_candidate_training(
        *_args: object, **kwargs: object
    ) -> tuple[object, tuple[ArtifactReference, ...]]:
        training = cast(SimpleNamespace, kwargs["training"])
        training_calls.append((cast(str, kwargs["candidate_id"]), cast(int, training.batch_size)))
        callback = kwargs.get("evaluation_callback")
        if callback is not None:
            cast(Callable[[TransformerLM, int, float], float], callback)(
                cast(TransformerLM, torch.nn.Linear(1, 1)),
                1,
                0.5,
            )
        return result, (_artifact(f"training-{kwargs['candidate_id']}"),)

    monkeypatch.setattr(pipeline, "_run_training", run_candidate_training)
    monkeypatch.setattr(pipeline, "_load_candidate_checkpoint", lambda *_args: (None, None, None))
    written: dict[str, object] = {}

    def contract(_context: object, filename: str, model: object) -> ArtifactReference:
        written[filename] = model
        return _artifact(filename)

    monkeypatch.setattr(pipeline, "_contract_artifact", contract)

    pilot_outcome = runtime.v04_pilot(context)
    pilot = cast(pipeline.V04PilotReport, written["v04-pilot.json"])
    assert pilot.activated is True
    assert pilot.requested_device == "mps"
    assert pilot.mandatory_batch_resolved_device == "mps"
    assert tuple(item.batch_size for item in pilot.measurements) == (1, 2, 4)
    assert training_calls == [("long-b1", 1), ("long-b2", 2), ("long-b4", 4)]
    assert pilot_outcome.advancement_allowed is True

    truncate_prompts[0] = False
    inactive_outcome = runtime.v04_pilot(context)
    inactive_pilot_report = cast(pipeline.V04PilotReport, written["v04-pilot.json"])
    assert inactive_pilot_report.activated is False
    assert inactive_pilot_report.measurements == ()
    assert inactive_pilot_report.mandatory_batch_resolved_device is None
    assert inactive_pilot_report.passed is True
    assert inactive_outcome.advancement_allowed is True
    assert training_calls == [("long-b1", 1), ("long-b2", 2), ("long-b4", 4)]

    truncate_prompts[0] = True
    result.device = DeviceResolution(requested="mps", resolved="cpu", fallback_used=True)
    cpu_fallback_outcome = runtime.v04_pilot(context)
    cpu_fallback_report = cast(pipeline.V04PilotReport, written["v04-pilot.json"])
    assert cpu_fallback_report.requested_device == "mps"
    assert cpu_fallback_report.mandatory_batch_resolved_device == "cpu"
    assert cpu_fallback_report.passed is False
    assert cpu_fallback_outcome.advancement_allowed is False
    result.device = DeviceResolution(requested="mps", resolved="mps", fallback_used=False)

    monkeypatch.setattr(pipeline, "_upstream_attempt", lambda *_args: tmp_path)
    selection = SimpleNamespace(selected_candidate_id="control")

    def read(path: Path, model_type: type[object], **_kwargs: object) -> object:
        if model_type is pipeline.V04PilotReport:
            return pilot
        if model_type is pipeline.CandidateSelectionReport:
            return selection
        return SimpleNamespace()

    monkeypatch.setattr(pipeline, "_read_contract", read)
    candidate_dataset_loads: list[str] = []

    def load_candidate_dataset(
        _context: object,
        _config: object,
        stage: str,
        _name: str,
    ) -> object:
        candidate_dataset_loads.append(stage)
        return shadow if stage == "v04_shadow_freeze" else iid

    monkeypatch.setattr(pipeline, "_load_stage_dataset", load_candidate_dataset)
    monkeypatch.setattr(
        pipeline,
        "resolve_semantic_selection_examples",
        lambda *_args: (validation,),
    )
    decoded_selection_views: list[tuple[RemediationView, ...]] = []

    def decode_selection(
        _model: object,
        _tokenizer: object,
        examples: tuple[object, ...],
        **_kwargs: object,
    ) -> tuple[object, ...]:
        decoded_selection_views.append(
            tuple(cast(RemediationView, cast(SimpleNamespace, item).view) for item in examples)
        )
        return (object(),)

    monkeypatch.setattr(pipeline, "_decode_examples", decode_selection)
    monkeypatch.setattr(pipeline, "semantic_composite_score", lambda *_args, **_kwargs: 0.75)
    verified_pilots: list[tuple[Path, object]] = []
    monkeypatch.setattr(
        pipeline,
        "_verify_v04_pilot_training_evidence",
        lambda attempt, report, _inputs, **_kwargs: verified_pilots.append((attempt, report)),
    )
    training_calls.clear()
    active_outcome = runtime.v04_candidate_training(context)
    active_report = cast(
        pipeline.V04CandidateTrainingReport,
        written["v04-candidate-training.json"],
    )
    assert active_report.activated is True
    assert active_report.source_stage == "v04_candidate_training"
    assert training_calls == [("long", 4)]
    assert candidate_dataset_loads == ["v03_data_audit"]
    assert decoded_selection_views == [(RemediationView.IID_VALIDATION,)]
    assert active_outcome.advancement_allowed is True
    assert verified_pilots == [(tmp_path, pilot)]

    inactive = SimpleNamespace(activated=False, measurements=())

    def read_inactive(path: Path, model_type: type[object], **_kwargs: object) -> object:
        return inactive if model_type is pipeline.V04PilotReport else selection

    monkeypatch.setattr(pipeline, "_read_contract", read_inactive)
    monkeypatch.setattr(pipeline, "_load_training_result", lambda *_args: result)
    inactive_outcome = runtime.v04_candidate_training(context)
    inactive_report = cast(
        pipeline.V04CandidateTrainingReport,
        written["v04-candidate-training.json"],
    )
    assert inactive_report.reused_v03_candidate is True
    assert inactive_report.source_stage == "v03_candidate_training"
    assert inactive_outcome.advancement_allowed is True


def test_v04_gate_and_complete_review_publish_bound_development_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _runtime_context(tmp_path)
    inputs = cast(
        pipeline._ExecutionInputs,
        SimpleNamespace(
            v04=object(),
            tokenizer=SimpleNamespace(manifest=SimpleNamespace(checksum_sha256=HASH_B)),
            compact_contract_sha256=HASH_C,
        ),
    )
    runtime = pipeline._PipelineRuntime(
        project_root=tmp_path,
        config=_config("runtime"),
        source_commit=SOURCE_COMMIT,
        inputs=inputs,
    )
    monkeypatch.setattr(runtime, "_start", lambda _context: SOURCE_COMMIT)
    monkeypatch.setattr(runtime, "_finish", lambda _context, outcome: outcome)
    monkeypatch.setattr(pipeline, "_upstream_attempt", lambda *_args: tmp_path)
    iid_dataset = SimpleNamespace(
        examples=(
            SimpleNamespace(view=RemediationView.IID_TRAIN),
            SimpleNamespace(view=RemediationView.IID_VALIDATION),
        ),
        manifest=SimpleNamespace(dataset_version="0.3.0"),
    )
    shadow_dataset = SimpleNamespace(
        examples=tuple(SimpleNamespace(view=view) for view in SHADOW_VIEWS),
        manifest=SimpleNamespace(dataset_version="0.4.0"),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_stage_dataset",
        lambda _context, _config, stage, _name: (
            iid_dataset if stage == "v03_data_audit" else shadow_dataset
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_subset_dataset",
        lambda *_args, **_kwargs: SimpleNamespace(manifest=SimpleNamespace(checksum_sha256=HASH_D)),
    )
    monkeypatch.setattr(pipeline, "config_sha256", lambda _config: HASH_E)
    candidate = _candidate(
        "control",
        passed=True,
        worst=0.7,
        iid=0.8,
        context_length=512,
        hash_character="a",
    )
    index_draft = pipeline.V04EvaluationIndex.model_construct(
        candidates=(candidate,),
        selected_candidate_id=candidate.candidate_id,
        checkpoint_manifest_sha256=candidate.checkpoint_manifest_sha256,
        iid_report_sha256=candidate.iid_report_sha256,
        shadow_reports=candidate.shadow_reports,
        v04_acceptance_sha256=candidate.v04_acceptance_sha256,
        checksum_sha256="0" * 64,
    )
    index = pipeline._bound_model(index_draft, pipeline.V04EvaluationIndex)
    current_index = [index]
    artifact_binding = SimpleNamespace(
        checkpoint_sha256=candidate.checkpoint_manifest_sha256,
    )
    iid_metrics = SimpleNamespace(artifacts=artifact_binding)
    iid_report = SimpleNamespace(
        evaluation_view=RemediationView.IID_VALIDATION,
        checksum_sha256=candidate.iid_report_sha256,
        view_metrics=iid_metrics,
    )
    iid_acceptance = SimpleNamespace(
        checksum_sha256=candidate.iid_acceptance_sha256,
        advancement_allowed=True,
        view_metrics=iid_metrics,
    )
    shadow_reports = {
        view: SimpleNamespace(
            evaluation_view=view,
            checksum_sha256=checksum,
            view_metrics=SimpleNamespace(artifacts=artifact_binding),
        )
        for view, checksum in candidate.shadow_reports
    }
    candidate_acceptance = SimpleNamespace(
        checksum_sha256=index.v04_acceptance_sha256,
        advancement_allowed=True,
        v03_result=iid_acceptance,
        shadow_view_metrics=tuple(shadow_reports[view].view_metrics for view in SHADOW_VIEWS),
    )
    expected_iid_acceptance = [iid_acceptance]
    expected_candidate_acceptance = [candidate_acceptance]

    def read_gate(path: Path, model_type: type[object], **_kwargs: object) -> object:
        if model_type is pipeline.V04EvaluationIndex:
            return current_index[0]
        if model_type is SemanticEvaluationReport:
            if "-iid-semantic-evaluation" in path.name:
                return iid_report
            return next(
                report
                for view, report in shadow_reports.items()
                if f"-{view.value}-semantic-evaluation" in path.name
            )
        if model_type is V03AcceptanceResult:
            return iid_acceptance
        return candidate_acceptance

    monkeypatch.setattr(pipeline, "_read_contract", read_gate)
    monkeypatch.setattr(
        runtime,
        "_verified_v04_candidate_inventory",
        lambda _context: (
            (
                candidate.candidate_id,
                candidate.context_length,
                candidate.checkpoint_manifest_sha256,
            ),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_semantic_report_composite",
        lambda report: (
            0.8
            if report is iid_report
            else 0.7
            if report is shadow_reports[SHADOW_VIEWS[0]]
            else 0.8
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_require_semantic_report_scope",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_v03_acceptance",
        lambda _metrics: expected_iid_acceptance[0],
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_v04_acceptance",
        lambda _iid, _shadows: expected_candidate_acceptance[0],
    )
    written: dict[str, object] = {}

    def contract(_context: object, filename: str, model: object) -> ArtifactReference:
        written[filename] = model
        return _artifact(filename)

    monkeypatch.setattr(pipeline, "_contract_artifact", contract)
    gate = runtime.v04_gate_and_final_policy_freeze(context)
    policy = cast(
        pipeline.FinalEvaluationPolicyFreeze,
        written["final-evaluation-policy.json"],
    )
    assert gate.advancement_allowed is True
    assert policy.status == "locked_pending_owner_reviewed_fresh_extension"

    expected_candidate_acceptance[0] = SimpleNamespace(
        checksum_sha256=HASH_B,
        advancement_allowed=False,
        v03_result=iid_acceptance,
        shadow_view_metrics=tuple(shadow_reports[view].view_metrics for view in SHADOW_VIEWS),
    )
    with pytest.raises(pipeline.PipelineExecutionError, match="immutable evaluation evidence"):
        runtime.v04_gate_and_final_policy_freeze(context)
    expected_candidate_acceptance[0] = candidate_acceptance

    tampered_candidate = candidate.model_copy(
        update={"worst_view_semantic_composite": 0.9, "iid_semantic_composite": 0.9}
    )
    tampered_index_draft = pipeline.V04EvaluationIndex.model_construct(
        candidates=(tampered_candidate,),
        selected_candidate_id=tampered_candidate.candidate_id,
        checkpoint_manifest_sha256=tampered_candidate.checkpoint_manifest_sha256,
        iid_report_sha256=tampered_candidate.iid_report_sha256,
        shadow_reports=tampered_candidate.shadow_reports,
        v04_acceptance_sha256=tampered_candidate.v04_acceptance_sha256,
        checksum_sha256="0" * 64,
    )
    current_index[0] = pipeline._bound_model(
        tampered_index_draft,
        pipeline.V04EvaluationIndex,
    )
    with pytest.raises(pipeline.PipelineExecutionError, match="immutable evaluation evidence"):
        runtime.v04_gate_and_final_policy_freeze(context)
    current_index[0] = index

    relabelled = candidate.model_copy(update={"candidate_id": "relabelled", "context_length": 1024})
    relabelled_draft = pipeline.V04EvaluationIndex.model_construct(
        candidates=(relabelled,),
        selected_candidate_id=relabelled.candidate_id,
        checkpoint_manifest_sha256=relabelled.checkpoint_manifest_sha256,
        iid_report_sha256=relabelled.iid_report_sha256,
        shadow_reports=relabelled.shadow_reports,
        v04_acceptance_sha256=relabelled.v04_acceptance_sha256,
        checksum_sha256="0" * 64,
    )
    current_index[0] = pipeline._bound_model(relabelled_draft, pipeline.V04EvaluationIndex)
    with pytest.raises(pipeline.PipelineExecutionError, match="candidate inventory"):
        runtime.v04_gate_and_final_policy_freeze(context)
    current_index[0] = index

    original_shadow = shadow_reports[SHADOW_VIEWS[0]]
    shadow_reports[SHADOW_VIEWS[0]] = SimpleNamespace(
        evaluation_view=original_shadow.evaluation_view,
        checksum_sha256=HASH_B,
        view_metrics=original_shadow.view_metrics,
    )
    with pytest.raises(pipeline.PipelineExecutionError, match="immutable evaluation evidence"):
        runtime.v04_gate_and_final_policy_freeze(context)
    shadow_reports[SHADOW_VIEWS[0]] = original_shadow

    records = tuple(
        SimpleNamespace(
            name=stage,
            status=StageStatus.COMPLETED,
            latest_attempt_path=f"stages/{ordinal:02d}-{stage}/attempt-0001",
        )
        for ordinal, stage in enumerate(PIPELINE_STAGES, start=1)
    )
    state = SimpleNamespace(stages=records)
    monkeypatch.setattr(
        pipeline,
        "PipelineStore",
        lambda *_args, **_kwargs: SimpleNamespace(load_state=lambda: state),
    )
    monkeypatch.setattr(
        pipeline,
        "_stage_completion_outcome",
        lambda _run, record_path: _artifact(f"{record_path}/outcome.json"),
    )

    def read_review(path: Path, model_type: type[object], **_kwargs: object) -> object:
        assert model_type is pipeline.FinalEvaluationPolicyFreeze
        return policy

    monkeypatch.setattr(pipeline, "_read_contract", read_review)
    review = runtime.review_bundle(context)
    manifest = cast(pipeline.ReviewBundleManifest, written["review-bundle.json"])

    assert review.advancement_allowed is True
    assert len(manifest.stages) == len(PIPELINE_STAGES) - 1
    assert manifest.final_policy_sha256 == policy.checksum_sha256
    assert (context.attempt_directory / "REVIEW_BUNDLE.md").is_file()


def _future_policy(*, passed: bool = True) -> pipeline.FinalEvaluationPolicyFreeze:
    draft = pipeline.FinalEvaluationPolicyFreeze.model_construct(
        v04_acceptance_sha256=HASH_A,
        development_gate_passed=passed,
        status=(
            "locked_pending_owner_reviewed_fresh_extension"
            if passed
            else "locked_development_gate_failed"
        ),
        checksum_sha256="0" * 64,
    )
    return pipeline._bound_model(draft, pipeline.FinalEvaluationPolicyFreeze)


def _complete_review(policy: pipeline.FinalEvaluationPolicyFreeze) -> pipeline.ReviewBundleManifest:
    bindings = tuple(
        pipeline.ReviewStageBinding(stage=stage, outcome=_artifact(f"{stage}/outcome.json"))
        for stage in PIPELINE_STAGES[:-1]
    )
    draft = pipeline.ReviewBundleManifest.model_construct(
        run_name="future-review",
        source_commit=SOURCE_COMMIT,
        pipeline_config_sha256=HASH_B,
        stages=bindings,
        final_policy_sha256=policy.checksum_sha256,
        summary_relative_path="stages/16-review_bundle/attempt-0001/REVIEW_BUNDLE.md",
        summary_sha256=HASH_C,
        checksum_sha256="0" * 64,
    )
    return pipeline._bound_model(draft, pipeline.ReviewBundleManifest)


def test_future_final_contracts_validate_data_only_without_accessing_payloads() -> None:
    created = "2026-08-23T00:00:00+00:00"
    policy = _future_policy()
    review_bundle = _complete_review(policy)
    manifest_draft = pipeline.FreshFinalExtensionManifest.model_construct(
        manifest_version="future-1.0.0",
        extension_id="fresh-extension-01",
        created_at=created,
        generated_after_policy_sha256=policy.checksum_sha256,
        final_dataset_config_sha256=HASH_A,
        frozen_final_payload_relative_path="sealed/frozen-payload.jsonl",
        frozen_final_payload_sha256=HASH_B,
        fresh_extension_payload_relative_path="sealed/fresh-extension.jsonl",
        fresh_extension_payload_sha256=HASH_C,
        case_ids=("FRESH-001", "FRESH-002"),
        historical_case_ids=(),
        checksum_sha256="0" * 64,
    )
    manifest = pipeline._bound_model(manifest_draft, pipeline.FreshFinalExtensionManifest)
    manifest_payload = manifest.model_dump(mode="python", round_trip=True)
    with pytest.raises(ValidationError, match="path escapes"):
        pipeline.FreshFinalExtensionManifest.model_validate(
            {**manifest_payload, "fresh_extension_payload_relative_path": "../escape.jsonl"},
            strict=True,
        )
    with pytest.raises(ValidationError, match="historical G01-G15"):
        pipeline.FreshFinalExtensionManifest.model_validate(
            {**manifest_payload, "case_ids": ("G01",)},
            strict=True,
        )
    with pytest.raises(ValidationError, match="checksum"):
        pipeline.FreshFinalExtensionManifest.model_validate(
            {**manifest_payload, "checksum_sha256": HASH_A},
            strict=True,
        )
    extension_review_draft = pipeline.FreshExtensionReview.model_construct(
        review_version="future-1.0.0",
        fresh_extension_manifest_sha256=manifest.checksum_sha256,
        owner_review_record_sha256=HASH_D,
        owner_approved=True,
        generated_after_development_freeze=True,
        historical_payload_used=False,
        checksum_sha256="0" * 64,
    )
    extension_review = pipeline._bound_model(
        extension_review_draft,
        pipeline.FreshExtensionReview,
    )
    with pytest.raises(ValidationError, match="checksum"):
        pipeline.FreshExtensionReview.model_validate(
            {
                **extension_review.model_dump(mode="python", round_trip=True),
                "checksum_sha256": HASH_A,
            },
            strict=True,
        )
    request_draft = pipeline.FinalEvaluationRequest.model_construct(
        request_version="future-1.0.0",
        policy_sha256=policy.checksum_sha256,
        review_bundle_sha256=review_bundle.checksum_sha256,
        fresh_extension_review_sha256=extension_review.checksum_sha256,
        explicit_confirmation=True,
        one_access_nonce_sha256=HASH_E,
        checksum_sha256="0" * 64,
    )
    request = pipeline._bound_model(request_draft, pipeline.FinalEvaluationRequest)
    authorization = pipeline.verify_final_evaluation_prerequisites(
        policy=policy,
        review_bundle=review_bundle,
        fresh_extension_review=extension_review,
        request=request,
    )
    assert authorization.policy_sha256 == policy.checksum_sha256
    assert authorization.fresh_extension_manifest_sha256 == manifest.checksum_sha256

    for status, completed_at, result_sha, failure_code in (
        ("claimed", None, None, None),
        ("completed", created, HASH_A, None),
        ("failed", created, None, "safe_failure"),
    ):
        ledger_draft = pipeline.FinalAccessLedger.model_construct(
            ledger_version="future-1.0.0",
            status=cast(Literal["claimed", "completed", "failed"], status),
            source_commit=SOURCE_COMMIT,
            authorization_sha256=authorization.checksum_sha256,
            claimed_at=created,
            completed_at=completed_at,
            result_sha256=result_sha,
            failure_code=failure_code,
            checksum_sha256="0" * 64,
        )
        ledger = pipeline._bound_model(ledger_draft, pipeline.FinalAccessLedger)
        assert ledger.status == status
        invalid_payload = ledger.model_dump(mode="python", round_trip=True)
        invalid_payload["checksum_sha256"] = HASH_B
        with pytest.raises(ValidationError, match="checksum"):
            pipeline.FinalAccessLedger.model_validate(invalid_payload, strict=True)

    result_draft = pipeline.FinalEvaluationResult.model_construct(
        result_version="future-1.0.0",
        authorization_sha256=authorization.checksum_sha256,
        source_commit=SOURCE_COMMIT,
        final_dataset_config_sha256=HASH_A,
        frozen_final_payload_sha256=HASH_B,
        fresh_extension_payload_sha256=HASH_C,
        selected_checkpoint_sha256=HASH_D,
        final_acceptance_sha256=HASH_E,
        final_acceptance_passed=True,
        completed_at=created,
        review_summary_relative_path="sealed/FINAL_REVIEW.md",
        review_summary_sha256=HASH_A,
        checksum_sha256="0" * 64,
    )
    result = pipeline._bound_model(result_draft, pipeline.FinalEvaluationResult)
    assert result.completed_at == created
    with pytest.raises(ValidationError, match="checksum"):
        pipeline.FinalEvaluationResult.model_validate(
            {
                **result.model_dump(mode="python", round_trip=True),
                "checksum_sha256": HASH_A,
            },
            strict=True,
        )
    with pytest.raises(ValidationError, match="checksum"):
        pipeline.FinalEvaluationRequest.model_validate(
            {
                **request.model_dump(mode="python", round_trip=True),
                "checksum_sha256": HASH_A,
            },
            strict=True,
        )
    with pytest.raises(ValidationError, match="checksum"):
        pipeline.FinalEvaluationAuthorization.model_validate(
            {
                **authorization.model_dump(mode="python", round_trip=True),
                "checksum_sha256": HASH_A,
            },
            strict=True,
        )

    with pytest.raises(pipeline.FinalEvaluationBlockedError, match="development gate"):
        pipeline.verify_final_evaluation_prerequisites(
            policy=_future_policy(passed=False),
            review_bundle=review_bundle,
            fresh_extension_review=extension_review,
            request=request,
        )
    mismatched = request.model_copy(update={"policy_sha256": HASH_A})
    with pytest.raises(pipeline.FinalEvaluationBlockedError, match="checksums differ"):
        pipeline.verify_final_evaluation_prerequisites(
            policy=policy,
            review_bundle=review_bundle,
            fresh_extension_review=extension_review,
            request=mismatched,
        )
    with pytest.raises(TypeError, match="exact review contracts"):
        pipeline.verify_final_evaluation_prerequisites(
            policy=cast(pipeline.FinalEvaluationPolicyFreeze, object()),
            review_bundle=review_bundle,
            fresh_extension_review=extension_review,
            request=request,
        )


def test_file_contract_path_and_stop_helpers_are_checksum_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert pipeline._canonical_utc("2026-08-23T00:00:00+00:00").year == 2026
    for invalid in ("bad", "2026-08-23T00:00:00", "2026-08-23T00:00:00.123+00:00"):
        with pytest.raises(ValueError, match="timestamp"):
            pipeline._canonical_utc(invalid)
    with pytest.raises(ValueError, match="duplicate key"):
        pipeline._strict_json(b'{"a":1,"a":2}')
    with pytest.raises(ValueError, match="non-finite"):
        pipeline._strict_json(b'{"a":NaN}')

    root = tmp_path.resolve()
    config = _config("stop-helper")
    run = root / config.run_root / config.run_name
    run.mkdir(parents=True)
    stop_path = pipeline.pipeline_stop_file(project_root=root, config=config)
    stop_requested = pipeline.build_stop_requested(project_root=root, config=config)
    assert stop_requested() is False
    request = pipeline.request_pipeline_stop(stop_path)
    assert stop_requested() is True
    assert pipeline._read_contract(stop_path, pipeline.PipelineStopRequest) == request
    real_open = os.open
    real_fsync = os.fsync
    opened_directories: dict[int, Path] = {}
    fsynced_directories: list[Path] = []

    def tracked_open(path: str | Path, flags: int, mode: int = 0o777) -> int:
        descriptor = real_open(path, flags, mode)
        candidate = Path(path)
        if flags == os.O_RDONLY and candidate.is_dir():
            opened_directories[descriptor] = candidate
        return descriptor

    def tracked_fsync(descriptor: int) -> None:
        directory = opened_directories.get(descriptor)
        if directory is not None:
            fsynced_directories.append(directory)
        real_fsync(descriptor)

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "fsync", tracked_fsync)
    archived = pipeline.archive_pipeline_stop(stop_path)
    assert archived is not None
    assert archived.is_file()
    assert fsynced_directories[-2:] == [archived.parent, stop_path.parent]
    assert stop_requested() is False
    assert pipeline.archive_pipeline_stop(stop_path) is None

    archived_payload = archived.read_bytes()
    pipeline._write_bytes(stop_path, archived_payload)
    fsynced_directories.clear()
    assert pipeline.archive_pipeline_stop(stop_path) == archived
    assert not stop_path.exists()
    assert fsynced_directories == [stop_path.parent]

    pipeline._write_bytes(stop_path, archived_payload)
    conflicting_draft = pipeline.PipelineStopRequest.model_construct(
        requested_at="2026-08-23T00:00:00+00:00",
        process_id=1,
        checksum_sha256="0" * 64,
    )
    conflicting = pipeline._bound_model(conflicting_draft, pipeline.PipelineStopRequest)
    archived.write_bytes(
        canonical_json_bytes(conflicting.model_dump(mode="json", round_trip=True)) + b"\n"
    )
    with pytest.raises(FileExistsError, match="conflicting"):
        pipeline.archive_pipeline_stop(stop_path)
    assert stop_path.is_file()
    with pytest.raises(FileExistsError, match="new regular file"):
        pipeline._write_bytes(archived, b"replacement")

    artifact_dir = run / "artifacts"
    nested = artifact_dir / "nested"
    nested.mkdir(parents=True)
    payload = nested / "payload.bin"
    pipeline._write_bytes(payload, b"payload")
    reference = pipeline._artifact_reference(payload, run_directory=run)
    assert reference.relative_path == "artifacts/nested/payload.bin"
    assert pipeline._run_size_bytes(run) >= len(b"payload")
    assert pipeline._directory_artifacts(artifact_dir, run_directory=run) == (reference,)

    source_file = root / "inputs" / "config.toml"
    source_file.parent.mkdir()
    source_file.write_text("version = 1\n", encoding="ascii")
    assert pipeline._safe_input_path(root, "inputs/config.toml", kind="file") == source_file
    assert pipeline._safe_input_path(root, "inputs", kind="directory") == source_file.parent
    for unsafe in ("", "../config.toml", "heldout/config.toml", "inputs\\config.toml"):
        with pytest.raises(ValueError, match=r"pipeline input|prohibited"):
            pipeline._safe_input_path(root, unsafe, kind="file")
    with pytest.raises(ValueError, match="missing"):
        pipeline._safe_input_path(root, "inputs/missing.toml", kind="file")


def test_compact_snapshot_verifier_reproduces_the_complete_manifest(tmp_path: Path) -> None:
    snapshot = tmp_path / "compact"
    snapshot.mkdir()
    contract = snapshot / "contract.json"
    readme = snapshot / "README.md"
    contract.write_bytes(canonical_json_bytes(compact_output_contract()) + b"\n")
    readme.write_text("# Frozen compact contract\n", encoding="ascii")
    files = {
        "README.md": hashlib.sha256(readme.read_bytes()).hexdigest(),
        "contract.json": hashlib.sha256(contract.read_bytes()).hexdigest(),
    }
    manifest = {
        "contract_version": "0.2.0",
        "files": files,
        "manifest_version": "0.1.0",
        "snapshot_sha256": canonical_sha256({"files": files, "contract_version": "0.2.0"}),
    }
    (snapshot / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")

    assert pipeline._verify_compact_contract(contract) == files["contract.json"]
    (snapshot / "manifest.json").write_bytes(
        canonical_json_bytes({**manifest, "files": {}}) + b"\n"
    )
    with pytest.raises(pipeline.PipelineExecutionError, match="inventory"):
        pipeline._verify_compact_contract(contract)


def test_execution_input_loader_checks_every_frozen_provenance_edge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("input-loader")
    tokenizer_path = "artifacts/tokenizer"
    contract_path = "artifacts/compact/contract.json"
    v02 = SimpleNamespace(
        paths=SimpleNamespace(
            tokenizer_path=tokenizer_path,
            compact_contract_path=contract_path,
            dataset_config_path="configs/dataset-v02.toml",
        ),
        inventory_report_path="reports/v02.json",
        inventory_report_checksum_sha256=HASH_D,
    )
    v03 = SimpleNamespace(
        paths=SimpleNamespace(
            tokenizer_path=tokenizer_path,
            compact_contract_path=contract_path,
            dataset_config_path="configs/dataset-v03.toml",
        ),
        baseline_config_path="configs/baseline.toml",
        baseline_config_sha256=HASH_E,
        counterfactual_cap_report_path="reports/v03.json",
        counterfactual_cap_report_checksum_sha256=HASH_B,
    )
    v04 = SimpleNamespace(
        compact_contract_path=contract_path,
        development_dataset_config_path=v03.paths.dataset_config_path,
        final_access=SimpleNamespace(
            automatically_run_final_evaluation=False,
            require_ready_marker=True,
            require_owner_review=True,
            require_explicit_confirm_flag=True,
            one_access_only=True,
            historical_golden_packet_permitted=False,
        ),
    )
    fake_paths = {
        config.v02_config_path: tmp_path / "v02.toml",
        config.v03_config_path: tmp_path / "v03.toml",
        config.v04_config_path: tmp_path / "v04.toml",
        tokenizer_path: tmp_path / "tokenizer",
        contract_path: tmp_path / "contract.json",
        v02.paths.dataset_config_path: tmp_path / "dataset-v02.toml",
        v03.paths.dataset_config_path: tmp_path / "dataset-v03.toml",
        v03.baseline_config_path: tmp_path / "baseline.toml",
        v02.inventory_report_path: tmp_path / "inventory.json",
        v03.counterfactual_cap_report_path: tmp_path / "counterfactual.json",
    }
    monkeypatch.setattr(
        pipeline,
        "_safe_input_path",
        lambda _root, relative, **_kwargs: fake_paths[relative],
    )
    monkeypatch.setattr(pipeline, "load_v02_config", lambda _path: v02)
    monkeypatch.setattr(pipeline, "load_v03_config", lambda _path: v03)
    monkeypatch.setattr(pipeline, "load_v04_config", lambda _path: v04)

    def config_digest(value: object) -> str:
        return {id(v02): HASH_A, id(v03): HASH_B, id(v04): HASH_C}[id(value)]

    monkeypatch.setattr(pipeline, "config_sha256", config_digest)
    tokenizer = SimpleNamespace(manifest=SimpleNamespace(checksum_sha256=HASH_C))
    monkeypatch.setattr(
        pipeline,
        "ProjectTokenizer",
        SimpleNamespace(load=lambda _path: tokenizer),
    )
    monkeypatch.setattr(pipeline, "_verify_compact_contract", lambda _path: HASH_A)
    v02_dataset = SimpleNamespace(dataset=SimpleNamespace(dataset_version="0.2.0"))
    v03_dataset = SimpleNamespace(dataset=SimpleNamespace(dataset_version="0.3.0"))
    monkeypatch.setattr(
        pipeline,
        "load_development_dataset_config",
        lambda path: (
            v02_dataset if path == fake_paths[v02.paths.dataset_config_path] else v03_dataset
        ),
    )
    baseline = SimpleNamespace(model_dump=lambda **_kwargs: {"baseline": "frozen"})
    monkeypatch.setattr(
        pipeline,
        "load_phase5_config",
        lambda _path: SimpleNamespace(baselines=baseline),
    )
    original_sha = canonical_sha256
    monkeypatch.setattr(
        pipeline,
        "canonical_sha256",
        lambda value: HASH_E if value == {"baseline": "frozen"} else original_sha(value),
    )
    inventory = SimpleNamespace(
        checksum_sha256=HASH_D,
        source_commit=SOURCE_COMMIT,
        tokenizer_manifest_sha256=HASH_C,
    )
    counterfactual = SimpleNamespace(
        checksum_sha256=HASH_B,
        base_inventory_report_sha256=HASH_D,
        source_commit=SOURCE_COMMIT,
        tokenizer_manifest_sha256=HASH_C,
    )

    def read(_path: Path, model_type: type[object], **_kwargs: object) -> object:
        return inventory if model_type is CompactInventoryReport else counterfactual

    monkeypatch.setattr(pipeline, "_read_contract", read)

    loaded = pipeline._load_execution_inputs(project_root=tmp_path, config=config)

    assert id(loaded.v02) == id(v02)
    assert id(loaded.v03) == id(v03)
    assert id(loaded.v04) == id(v04)
    assert id(loaded.tokenizer) == id(tokenizer)
    assert loaded.compact_contract_sha256 == HASH_A
    assert loaded.frozen_data_source_commit == SOURCE_COMMIT

    different_v04_dataset_path = "configs/dataset-v04-different.toml"
    fake_paths[different_v04_dataset_path] = tmp_path / "dataset-v04-different.toml"
    v04.development_dataset_config_path = different_v04_dataset_path
    with pytest.raises(pipeline.PipelineExecutionError, match="dataset recipe differs"):
        pipeline._load_execution_inputs(project_root=tmp_path, config=config)


def test_training_and_evaluation_helpers_preserve_resume_and_artifact_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    stage = run / "stages/03-training"
    prior = stage / "attempt-0001/training-state/control/state-step-00000002"
    prior.mkdir(parents=True)
    (prior / "state.json").write_text("{}\n", encoding="ascii")
    attempt = stage / "attempt-0002"
    attempt.mkdir()
    context = cast(
        StageContext,
        SimpleNamespace(
            run_directory=run,
            attempt_directory=attempt,
            source_commit=SOURCE_COMMIT,
            progress=_FakeProgress(),
        ),
    )

    def latest_state(root: Path, *, candidate_id: str) -> Path | None:
        assert candidate_id == "control"
        states = tuple(sorted(root.glob("state-step-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]")))
        return None if not states else states[-1].resolve(strict=True)

    def retire_states(
        root: Path,
        *,
        candidate_id: str,
        successor_directory: Path,
    ) -> int:
        assert candidate_id == "control"
        assert successor_directory.is_dir()
        states = tuple(root.glob("state-step-*"))
        for state in states:
            for child in state.iterdir():
                child.unlink()
            state.rmdir()
        return len(states)

    monkeypatch.setattr(pipeline, "latest_committed_training_state", latest_state)
    monkeypatch.setattr(pipeline, "retire_superseded_training_states", retire_states)
    assert pipeline._latest_resume_source(context, "control") == prior
    copied_root = attempt / "copied"
    copied_root.mkdir()
    copied = pipeline._copy_resume_state(prior, copied_root)
    assert (copied / "state.json").read_text(encoding="ascii") == "{}\n"

    train_example = RemediationExample.model_construct(
        example_id="example:train",
        view=RemediationView.IID_TRAIN,
        checksum_sha256=HASH_A,
    )
    validation_example = RemediationExample.model_construct(
        example_id="example:validation",
        view=RemediationView.IID_VALIDATION,
        checksum_sha256=HASH_B,
    )
    tokenizer = SimpleNamespace(
        vocab_size=32,
        manifest=SimpleNamespace(checksum_sha256=HASH_C),
    )
    inputs = cast(
        pipeline._ExecutionInputs,
        SimpleNamespace(tokenizer=tokenizer, generation_caps={}),
    )
    tokenized = SimpleNamespace(example_id="tokenized")
    monkeypatch.setattr(
        pipeline,
        "_tokenize_examples",
        lambda *_args, **_kwargs: (tokenized,),
    )
    model_config = TransformerConfig(
        model_version="0.3.0",
        layers=1,
        width=16,
        heads=4,
        context_length=8,
        feed_forward_multiplier=2,
        dropout=0.0,
        tie_embeddings=True,
        bias=True,
    )
    training = RemediationTraining(
        seed=7,
        device="cpu",
        allow_cpu_fallback=True,
        steps=2,
        batch_size=1,
        learning_rate=0.001,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        evaluation_interval=1,
        durable_checkpoint_interval=1,
    )
    result = CompactTrainingResult.model_construct(
        checksum_sha256=HASH_D,
        checkpoint_manifest_sha256=HASH_E,
    )
    observed_resume: list[Path | None] = []

    def train_model(**kwargs: object) -> object:
        checkpoint = cast(Path, kwargs["output_directory"])
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"safe")
        observed_resume.append(cast(Path | None, kwargs["resume_state_directory"]))
        return result

    monkeypatch.setattr(pipeline, "train_compact_model", train_model)
    projected_reservations: list[int] = []
    guard_polls: list[str] = []

    def stop_required(_context: object) -> bool:
        guard_polls.append("poll")
        return False

    guard = cast(
        pipeline._ResourceGuard,
        SimpleNamespace(
            stop_required=stop_required,
            enforce_projected_write=lambda _context, *, reservation_bytes: (
                projected_reservations.append(reservation_bytes)
            ),
        ),
    )

    outcome, artifacts = pipeline._run_training(
        context,
        guard=guard,
        inputs=inputs,
        candidate_id="control",
        sampling_strategy="uniform_control",
        model_config=model_config,
        training=training,
        train_examples=(train_example,),
        validation_examples=(validation_example,),
        evaluation_callback=None,
    )

    assert outcome is result
    assert observed_resume
    assert observed_resume[0] is not None
    assert len(projected_reservations) == 2
    state_bound = durable_training_state_upper_bound_bytes(
        model_config,
        vocab_size=tokenizer.vocab_size,
    )
    checkpoint_bound = selected_checkpoint_upper_bound_bytes(
        model_config,
        vocab_size=tokenizer.vocab_size,
    )
    assert projected_reservations == [
        state_bound,
        2 * state_bound + checkpoint_bound,
    ]
    assert not prior.exists()
    assert {item.relative_path for item in artifacts} == {
        "stages/03-training/attempt-0002/checkpoint-control/model.safetensors",
        "stages/03-training/attempt-0002/training-control.json",
    }

    scoped = SimpleNamespace(manifest=SimpleNamespace(checksum_sha256=HASH_A))
    baseline = SimpleNamespace(checksum_sha256=HASH_B)
    prediction_manifest = SimpleNamespace(predictions_sha256=HASH_C)
    predictions = (SimpleNamespace(),)
    evaluation = SimpleNamespace(checksum_sha256=HASH_D)
    monkeypatch.setattr(pipeline, "_subset_dataset", lambda *_args, **_kwargs: scoped)
    baseline_poll_counts: list[tuple[int, int]] = []

    def run_baselines(*_args: object, **kwargs: object) -> object:
        before = len(guard_polls)
        callback = cast(Callable[[], bool], kwargs["stop_requested"])
        assert callback() is False
        baseline_poll_counts.append((before, len(guard_polls)))
        return baseline

    monkeypatch.setattr(pipeline, "run_remediation_baselines", run_baselines)
    monkeypatch.setattr(pipeline, "_decode_examples", lambda *_args, **_kwargs: predictions)
    monkeypatch.setattr(
        pipeline,
        "_write_predictions",
        lambda *_args, **_kwargs: (
            prediction_manifest,
            (_artifact("prediction-manifest"), _artifact("predictions")),
        ),
    )
    bound_artifacts: list[object] = []

    def evaluate(**kwargs: object) -> object:
        bound_artifacts.append(kwargs["artifacts"])
        return evaluation

    monkeypatch.setattr(pipeline, "evaluate_semantic_predictions", evaluate)
    monkeypatch.setattr(
        pipeline,
        "_contract_artifact",
        lambda _context, filename, _model: _artifact(filename),
    )
    model = SimpleNamespace(config=SimpleNamespace(context_length=512))
    checkpoint = SimpleNamespace(checksum_sha256=HASH_E)
    evaluated, observed_baseline, observed_predictions, observed_artifacts = (
        pipeline._evaluate_candidate_view(
            context,
            guard=guard,
            inputs=cast(
                pipeline._ExecutionInputs,
                SimpleNamespace(
                    tokenizer=tokenizer,
                    generation_caps={},
                    baseline_config=object(),
                    compact_contract_sha256=HASH_D,
                ),
            ),
            config_sha256_value=HASH_A,
            dataset=cast(
                SafeDevelopmentDataset,
                SimpleNamespace(manifest=SimpleNamespace(dataset_version="0.3.0")),
            ),
            train_examples=(train_example,),
            evaluation_examples=(validation_example,),
            view=RemediationView.IID_VALIDATION,
            model=cast(TransformerLM, model),
            checkpoint_manifest=cast(CheckpointManifest, checkpoint),
            device=torch.device("cpu"),
            stem="evaluation",
        )
    )
    binding = cast(DevelopmentArtifactBinding, bound_artifacts[0])
    assert evaluated.checksum_sha256 == HASH_D
    assert observed_baseline.checksum_sha256 == HASH_B
    assert len(observed_predictions) == len(predictions)
    assert len(observed_artifacts) == 4
    assert binding.tokenizer_manifest_sha256 == HASH_C
    assert binding.prediction_artifact_sha256 == HASH_C
    assert baseline_poll_counts == [(1, 2)]
    assert len(guard_polls) > baseline_poll_counts[0][1]
    messages = tuple(item["message"] for item in cast(_FakeProgress, context.progress).reports)
    assert "Baseline comparator evaluation started." in messages
    assert "Baseline comparator evaluation completed." in messages


def test_cross_attempt_resume_retention_is_globally_bounded_and_preserves_remnants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    stage = run / "stages/07-v03_candidate_training"
    initial_root = stage / "attempt-0001/training-state/control"
    initial_state = initial_root / "state-step-00000001"
    initial_state.mkdir(parents=True)
    (initial_state / "state.json").write_text("{}\n", encoding="ascii")

    def latest_state(root: Path, *, candidate_id: str) -> Path | None:
        assert candidate_id == "control"
        states = tuple(
            sorted(
                path
                for path in root.iterdir()
                if re.fullmatch(r"state-step-[0-9]{8}", path.name) is not None
            )
        )
        return None if not states else states[-1].resolve(strict=True)

    def retire_states(
        root: Path,
        *,
        candidate_id: str,
        successor_directory: Path,
    ) -> int:
        assert candidate_id == "control"
        assert successor_directory.is_dir()
        states = tuple(
            path
            for path in root.iterdir()
            if re.fullmatch(r"state-step-[0-9]{8}", path.name) is not None
        )
        for state in states:
            for child in state.iterdir():
                child.unlink()
            state.rmdir()
        return len(states)

    monkeypatch.setattr(pipeline, "latest_committed_training_state", latest_state)
    monkeypatch.setattr(pipeline, "retire_superseded_training_states", retire_states)
    guard = cast(
        pipeline._ResourceGuard,
        SimpleNamespace(enforce_projected_write=lambda *_args, **_kwargs: None),
    )

    for attempt_number in range(2, 13):
        attempt = stage / f"attempt-{attempt_number:04d}"
        destination_root = attempt / "training-state/control"
        destination_root.mkdir(parents=True)
        progress = _FakeProgress()
        context = cast(
            StageContext,
            SimpleNamespace(
                run_directory=run,
                attempt_directory=attempt,
                progress=progress,
            ),
        )

        copied = pipeline._prepare_resume_state(
            context,
            guard=guard,
            candidate_id="control",
            destination_root=destination_root,
            state_upper_bound_bytes=1024,
        )

        assert copied is not None
        assert copied.is_dir()
        assert len(progress.checkpoints) == 1
        committed = tuple(stage.glob("attempt-*/training-state/control/state-step-*"))
        assert committed == (copied,)
        (destination_root / f".state-step-{attempt_number:08d}.lock").write_text(
            "forensic\n", encoding="ascii"
        )
        (destination_root / f".state-step-{attempt_number:08d}.tmp-resume").mkdir()

    remnants = tuple(stage.glob("attempt-*/training-state/control/.state-step-*"))
    assert len(remnants) == 22
    assert all(path.exists() for path in remnants)


def test_resume_copy_and_retirement_failures_leave_a_successor_or_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    stage = run / "stages/07-v03_candidate_training"
    source_root = stage / "attempt-0001/training-state/control"
    source = source_root / "state-step-00000003"
    source.mkdir(parents=True)
    (source / "state.json").write_text("{}\n", encoding="ascii")
    attempt = stage / "attempt-0002"
    destination_root = attempt / "training-state/control"
    destination_root.mkdir(parents=True)
    progress = _FakeProgress()
    context = cast(
        StageContext,
        SimpleNamespace(
            run_directory=run,
            attempt_directory=attempt,
            progress=progress,
        ),
    )

    def latest_state(root: Path, *, candidate_id: str) -> Path | None:
        assert candidate_id == "control"
        candidate = root / "state-step-00000003"
        return candidate.resolve(strict=True) if candidate.is_dir() else None

    guard = cast(
        pipeline._ResourceGuard,
        SimpleNamespace(enforce_projected_write=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(pipeline, "latest_committed_training_state", latest_state)
    monkeypatch.setattr(
        shutil,
        "copytree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("simulated copy failure")),
    )
    with pytest.raises(OSError, match="simulated copy failure"):
        pipeline._prepare_resume_state(
            context,
            guard=guard,
            candidate_id="control",
            destination_root=destination_root,
            state_upper_bound_bytes=1024,
        )
    assert source.is_dir()
    assert (destination_root / ".state-step-00000003.tmp-resume").is_dir()
    assert not progress.checkpoints

    # A fresh non-overwriting attempt can copy successfully. If retirement then
    # fails, both the checkpointed successor and its source remain available.
    monkeypatch.undo()
    attempt = stage / "attempt-0003"
    destination_root = attempt / "training-state/control"
    destination_root.mkdir(parents=True)
    progress = _FakeProgress()
    context = cast(
        StageContext,
        SimpleNamespace(
            run_directory=run,
            attempt_directory=attempt,
            progress=progress,
        ),
    )
    monkeypatch.setattr(pipeline, "latest_committed_training_state", latest_state)
    monkeypatch.setattr(
        pipeline,
        "retire_superseded_training_states",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("simulated retirement failure")),
    )
    with pytest.raises(OSError, match="simulated retirement failure"):
        pipeline._prepare_resume_state(
            context,
            guard=guard,
            candidate_id="control",
            destination_root=destination_root,
            state_upper_bound_bytes=1024,
        )
    successor = destination_root / source.name
    assert source.is_dir()
    assert successor.is_dir()
    assert cast(str, progress.checkpoints[0]["checkpoint"]).endswith(source.name)


def test_targeted_resume_publishes_binding_before_state_and_recovers_after_copy_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    stage = run / "stages/07-v03_candidate_training"
    source_root = stage / "attempt-0001/training-state/targeted"
    source_state = source_root / "state-step-00000003"
    source_state.mkdir(parents=True)
    (source_state / "state.json").write_text("{}\n", encoding="ascii")
    binding = pipeline.bind_targeted_sampling(
        candidate_id="targeted",
        training_config_sha256=HASH_A,
        train_inventory_sha256=HASH_B,
        train_tokenized_sha256=HASH_C,
        sampling_metadata_inventory_sha256=HASH_D,
    )
    pipeline.ensure_targeted_sampling_binding(source_root, binding, create_if_missing=True)

    def latest_state(root: Path, *, candidate_id: str) -> Path | None:
        assert candidate_id == "targeted"
        candidate = root / "state-step-00000003"
        return candidate.resolve(strict=True) if candidate.is_dir() else None

    guard = cast(
        pipeline._ResourceGuard,
        SimpleNamespace(enforce_projected_write=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(pipeline, "latest_committed_training_state", latest_state)
    monkeypatch.setattr(
        pipeline,
        "retire_superseded_training_states",
        lambda *_args, **_kwargs: 0,
    )
    original_copy = pipeline._copy_resume_state
    monkeypatch.setattr(
        pipeline,
        "_copy_resume_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("copy crash")),
    )

    failed_attempt = stage / "attempt-0002"
    failed_destination = failed_attempt / "training-state/targeted"
    failed_destination.mkdir(parents=True)
    failed_context = cast(
        StageContext,
        SimpleNamespace(
            run_directory=run,
            attempt_directory=failed_attempt,
            progress=_FakeProgress(),
        ),
    )
    with pytest.raises(OSError, match="copy crash"):
        pipeline._prepare_resume_state(
            failed_context,
            guard=guard,
            candidate_id="targeted",
            destination_root=failed_destination,
            state_upper_bound_bytes=1024,
            targeted_binding=binding,
        )
    assert (failed_destination / pipeline.TARGETED_SAMPLING_BINDING_FILENAME).is_file()
    assert not tuple(failed_destination.glob("state-step-*"))

    monkeypatch.setattr(pipeline, "_copy_resume_state", original_copy)
    next_attempt = stage / "attempt-0003"
    next_destination = next_attempt / "training-state/targeted"
    next_destination.mkdir(parents=True)
    next_context = cast(
        StageContext,
        SimpleNamespace(
            run_directory=run,
            attempt_directory=next_attempt,
            progress=_FakeProgress(),
        ),
    )
    copied = pipeline._prepare_resume_state(
        next_context,
        guard=guard,
        candidate_id="targeted",
        destination_root=next_destination,
        state_upper_bound_bytes=1024,
        targeted_binding=binding,
    )
    assert copied == (next_destination / source_state.name).resolve(strict=True)


def test_dataset_tokenization_decode_smoke_and_checkpoint_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_example = RemediationExample.model_construct(
        example_id="example:train",
        view=RemediationView.IID_TRAIN,
        task_name=TaskName.FAULT_FAMILY,
        checksum_sha256=HASH_A,
    )
    validation_example = RemediationExample.model_construct(
        example_id="example:validation",
        view=RemediationView.IID_VALIDATION,
        task_name=TaskName.FAULT_FAMILY,
        checksum_sha256=HASH_B,
    )
    source_dataset = cast(
        SafeDevelopmentDataset,
        SimpleNamespace(
            manifest=SimpleNamespace(
                source_commit=SOURCE_COMMIT,
                dataset_config_sha256=HASH_C,
            )
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "SafeDevelopmentDataset",
        lambda **values: SimpleNamespace(**values),
    )
    subset = pipeline._subset_dataset(
        source_dataset,
        (validation_example, train_example),
        dataset_version="0.3.0",
    )
    assert tuple(item.example_id for item in subset.examples) == (
        "example:train",
        "example:validation",
    )
    assert subset.manifest.example_count == 2
    with pytest.raises(ValueError, match="non-empty and unique"):
        pipeline._subset_dataset(source_dataset, (), dataset_version="0.3.0")

    tokenized = CompactTokenizedExample(
        example_id="example:train",
        task_name=TaskName.FAULT_FAMILY,
        group_id="group:1",
        token_ids=(1, 2),
        target_mask=(False, True),
        prompt_token_count=1,
        target_token_count=1,
        prompt_tokens_retained=1,
        prompt_truncated=False,
    )
    assert pipeline._tokenized_inventory_sha256((tokenized,)) == tokenized_inventory_sha256(
        (tokenized,)
    )
    monkeypatch.setattr(pipeline, "tokenize_compact_example", lambda *_args, **_kwargs: tokenized)
    observed_tokenized = pipeline._tokenize_examples(
        (train_example,),
        cast(ProjectTokenizer, SimpleNamespace()),
        context_length=512,
        generation_caps={},
    )
    assert len(observed_tokenized) == 1
    assert id(observed_tokenized[0]) == id(tokenized)

    decode_calls: list[int] = []

    def decode(
        _model: object,
        _tokenizer: object,
        examples: tuple[object, ...],
        **_kwargs: object,
    ) -> tuple[DualPathCompactPrediction, ...]:
        decode_calls.append(len(examples))
        return cast(
            tuple[DualPathCompactPrediction, ...],
            tuple(SimpleNamespace(example_id=str(index)) for index, _item in enumerate(examples)),
        )

    monkeypatch.setattr(pipeline, "decode_compact_examples", decode)
    many_examples = tuple(
        train_example.model_copy(update={"example_id": f"decode-{index}"}) for index in range(5)
    )
    decode_progress: list[tuple[int, int]] = []
    decoded = pipeline._decode_examples(
        cast(TransformerLM, object()),
        cast(ProjectTokenizer, SimpleNamespace()),
        many_examples,
        generation_caps={},
        device=torch.device("cpu"),
        progress_callback=lambda completed, total: decode_progress.append((completed, total)),
    )
    assert len(decoded) == len(many_examples)
    assert decode_calls == [1, 1, 1, 1, 1]
    assert decode_progress == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
    assert (
        pipeline._decode_examples(
            cast(TransformerLM, object()),
            cast(ProjectTokenizer, SimpleNamespace()),
            (),
            generation_caps={},
            device=torch.device("cpu"),
        )
        == ()
    )
    calls_before_duplicate = tuple(decode_calls)
    with pytest.raises(ValueError, match="globally unique"):
        pipeline._decode_examples(
            cast(TransformerLM, object()),
            cast(ProjectTokenizer, SimpleNamespace()),
            (train_example, train_example),
            generation_caps={},
            device=torch.device("cpu"),
        )
    assert tuple(decode_calls) == calls_before_duplicate

    durable_progress = _FakeProgress()
    guarded_context = cast(
        StageContext,
        SimpleNamespace(progress=durable_progress),
    )
    guarded = pipeline._guarded_decode_examples(
        guarded_context,
        guard=cast(
            pipeline._ResourceGuard,
            SimpleNamespace(stop_required=lambda _context: False),
        ),
        model=cast(TransformerLM, object()),
        tokenizer=cast(ProjectTokenizer, SimpleNamespace()),
        examples=tuple(
            train_example.model_copy(update={"example_id": f"guarded-{index}"})
            for index in range(35)
        ),
        generation_caps={},
        device=torch.device("cpu"),
        progress_message="Bounded decoding progress.",
    )
    assert len(guarded) == 35
    assert [item["completed_units"] for item in durable_progress.reports] == [
        0,
        16,
        32,
        35,
    ]
    assert all(item["total_units"] == 35 for item in durable_progress.reports)

    smoke_train, smoke_validation = pipeline._smoke_examples(subset)
    assert smoke_train == (train_example,)
    assert smoke_validation == (validation_example,)
    with pytest.raises(ValueError, match="IID train and validation"):
        pipeline._smoke_examples(
            cast(SafeDevelopmentDataset, SimpleNamespace(examples=(train_example,)))
        )

    base_model = TransformerConfig(
        model_version="0.3.0",
        layers=4,
        width=128,
        heads=4,
        context_length=512,
        feed_forward_multiplier=4,
        dropout=0.1,
        tie_embeddings=True,
        bias=True,
    )
    smoke_model = pipeline._smoke_model_config(base_model)
    assert (smoke_model.layers, smoke_model.width, smoke_model.context_length) == (2, 64, 512)
    base_training = RemediationTraining(
        seed=9,
        device="mps",
        allow_cpu_fallback=False,
        steps=20,
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
        evaluation_interval=5,
        durable_checkpoint_interval=5,
    )
    smoke_training = pipeline._smoke_training_config(base_training)
    assert smoke_training.device == "cpu"
    assert smoke_training.steps == pipeline.SMOKE_STEPS
    assert pipeline._with_training_seed(base_training, 33).seed == 33

    training_result = cast(
        CompactTrainingResult,
        SimpleNamespace(
            device=SimpleNamespace(resolved="mps"),
            checkpoint_manifest_sha256=HASH_D,
        ),
    )
    tokenizer = SimpleNamespace(manifest=SimpleNamespace(checksum_sha256=HASH_C))
    loaded_model = cast(
        TransformerLM,
        SimpleNamespace(parameters=lambda: iter((SimpleNamespace(device=torch.device("mps:0")),))),
    )
    loaded_manifest = cast(
        CheckpointManifest,
        SimpleNamespace(checksum_sha256=HASH_D),
    )
    observed_checkpoint: list[dict[str, object]] = []

    def load_checkpoint(path: Path, **kwargs: object) -> tuple[TransformerLM, CheckpointManifest]:
        observed_checkpoint.append({"path": path, **kwargs})
        return loaded_model, loaded_manifest

    monkeypatch.setattr(pipeline, "load_checkpoint", load_checkpoint)
    model, manifest, device = pipeline._load_candidate_checkpoint(
        tmp_path,
        "control",
        training_result,
        cast(ProjectTokenizer, tokenizer),
    )
    assert model is loaded_model
    assert manifest.checksum_sha256 == HASH_D
    assert device == torch.device("mps:0")
    assert observed_checkpoint[0]["device"] == torch.device("mps")
    assert observed_checkpoint[0]["expected_tokenizer_sha256"] == HASH_C

    mismatched_model = cast(
        TransformerLM,
        SimpleNamespace(parameters=lambda: iter((SimpleNamespace(device=torch.device("cpu")),))),
    )
    monkeypatch.setattr(
        pipeline,
        "load_checkpoint",
        lambda *_args, **_kwargs: (mismatched_model, loaded_manifest),
    )
    with pytest.raises(
        pipeline.PipelineExecutionError,
        match="differs from its resolved device",
    ):
        pipeline._load_candidate_checkpoint(
            tmp_path,
            "control",
            training_result,
            cast(ProjectTokenizer, tokenizer),
        )


def test_view_evaluation_stops_after_one_of_1024_examples_without_partial_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = tmp_path / "run/stages/evaluation/attempt-0001"
    attempt.mkdir(parents=True)
    progress = _FakeProgress()
    context = cast(
        StageContext,
        SimpleNamespace(
            run_directory=tmp_path / "run",
            attempt_directory=attempt,
            source_commit=SOURCE_COMMIT,
            progress=progress,
        ),
    )
    train = RemediationExample.model_construct(
        example_id="train",
        view=RemediationView.IID_TRAIN,
        checksum_sha256=HASH_A,
    )
    validation = RemediationExample.model_construct(
        example_id="validation",
        view=RemediationView.IID_VALIDATION,
        checksum_sha256=HASH_B,
    )
    tokenizer = cast(
        ProjectTokenizer,
        SimpleNamespace(manifest=SimpleNamespace(checksum_sha256=HASH_C)),
    )
    inputs = cast(
        pipeline._ExecutionInputs,
        SimpleNamespace(
            tokenizer=tokenizer,
            generation_caps={},
            baseline_config=object(),
            compact_contract_sha256=HASH_D,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_subset_dataset",
        lambda *_args, **_kwargs: SimpleNamespace(manifest=SimpleNamespace(checksum_sha256=HASH_A)),
    )
    monkeypatch.setattr(pipeline, "_tokenize_examples", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        pipeline,
        "run_remediation_baselines",
        lambda *_args, **_kwargs: SimpleNamespace(checksum_sha256=HASH_B),
    )
    decode_calls: list[int] = []

    def decode(
        _model: object,
        _tokenizer: object,
        examples: tuple[object, ...],
        **_kwargs: object,
    ) -> tuple[DualPathCompactPrediction, ...]:
        decode_calls.append(len(examples))
        return cast(
            tuple[DualPathCompactPrediction, ...],
            (SimpleNamespace(example_id="validation"),),
        )

    monkeypatch.setattr(pipeline, "decode_compact_examples", decode)
    monkeypatch.setattr(
        pipeline,
        "_contract_artifact",
        lambda *_args, **_kwargs: pytest.fail("partial contract artifact was written"),
    )
    monkeypatch.setattr(
        pipeline,
        "_write_predictions",
        lambda *_args, **_kwargs: pytest.fail("partial predictions were written"),
    )

    def stop_required(_context: object) -> bool:
        return bool(decode_calls)

    guard = cast(
        pipeline._ResourceGuard,
        SimpleNamespace(stop_required=stop_required),
    )
    model = cast(
        TransformerLM,
        SimpleNamespace(config=SimpleNamespace(context_length=512)),
    )
    checkpoint = cast(
        CheckpointManifest,
        SimpleNamespace(checksum_sha256=HASH_E),
    )

    with pytest.raises(KeyboardInterrupt):
        pipeline._evaluate_candidate_view(
            context,
            guard=guard,
            inputs=inputs,
            config_sha256_value=HASH_A,
            dataset=cast(
                SafeDevelopmentDataset,
                SimpleNamespace(manifest=SimpleNamespace(dataset_version="0.3.0")),
            ),
            train_examples=(train,),
            evaluation_examples=tuple(
                validation.model_copy(update={"example_id": f"validation-{index:04d}"})
                for index in range(1024)
            ),
            view=RemediationView.IID_VALIDATION,
            model=model,
            checkpoint_manifest=checkpoint,
            device=torch.device("cpu"),
            stem="stopped-view",
        )

    assert decode_calls == [1]
    decode_reports = tuple(
        item
        for item in progress.reports
        if item.get("message") == "Model evaluation decoding in progress."
    )
    assert [item["completed_units"] for item in decode_reports] == [0]
    assert all(item["total_units"] == 1024 for item in decode_reports)
    assert tuple(attempt.iterdir()) == ()


def test_v04_pilot_selects_and_exercises_the_longest_row_per_task() -> None:
    raw_examples: list[RemediationExample] = []
    tokenized_examples: list[CompactTokenizedExample] = []
    for task_index, task_name in enumerate(TaskName):
        for variant, length in (("short", 12 + task_index), ("long", 80 + task_index)):
            example_id = f"{variant}-{task_name.value}"
            group_id = f"group-{variant}-{task_name.value}"
            raw_examples.append(
                cast(
                    RemediationExample,
                    SimpleNamespace(
                        example_id=example_id,
                        task_name=task_name,
                        group_id=group_id,
                    ),
                )
            )
            tokenized_examples.append(
                CompactTokenizedExample(
                    example_id=example_id,
                    task_name=task_name,
                    group_id=group_id,
                    token_ids=tuple(range(length)),
                    target_mask=tuple(False for _ in range(length)),
                    prompt_token_count=length - 1,
                    target_token_count=1,
                    prompt_tokens_retained=length - 1,
                    prompt_truncated=False,
                )
            )

    raw = tuple(raw_examples)
    tokenized = tuple(tokenized_examples)
    selected_raw, selected_tokenized = pipeline._longest_pilot_examples_per_task(
        raw,
        tokenized,
    )
    assert tuple(item.example_id for item in selected_raw) == tuple(
        f"long-{task_name.value}" for task_name in TaskName
    )
    assert max(len(item.token_ids) for item in selected_tokenized) == max(
        len(item.token_ids) for item in tokenized
    )
    assert pipeline._sequence_length_inventory_sha256(tokenized) == (
        pipeline._sequence_length_inventory_sha256(tuple(reversed(tokenized)))
    )
    assert all(
        pipeline._pilot_exercises_global_maximum(
            selected_tokenized,
            batch_size=batch_size,
            seed=6401,
            steps=10,
        )
        for batch_size in (1, 2, 4)
    )

    with pytest.raises(ValueError, match="align"):
        pipeline._longest_pilot_examples_per_task(raw, tokenized[:-1])


def test_source_control_and_resource_boundaries_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "git version" in pipeline._run_process(
        (pipeline.TRUSTED_GIT, "--version"),
        project_root=tmp_path,
    )
    with pytest.raises(TypeError, match="trusted Git"):
        pipeline._run_process(("git", "--version"), project_root=tmp_path)

    outputs = iter((str(tmp_path), FULL_SOURCE_COMMIT, "?? work/run/output.json"))
    monkeypatch.setattr(pipeline, "_run_process", lambda *_args, **_kwargs: next(outputs))
    assert (
        pipeline._verify_runner_source(
            tmp_path,
            source_commit=FULL_SOURCE_COMMIT,
            run_root="work",
        )
        == FULL_SOURCE_COMMIT
    )
    for status, message in (
        ("?", "malformed"),
        ("R  old.py -> new.py", "uncommitted"),
        (" M src/module.py", "uncommitted"),
    ):
        outputs = iter((str(tmp_path), FULL_SOURCE_COMMIT, status))
        monkeypatch.setattr(
            pipeline,
            "_run_process",
            lambda *_args, _outputs=outputs, **_kwargs: next(_outputs),
        )
        with pytest.raises(pipeline.PipelineExecutionError, match=message):
            pipeline._verify_runner_source(
                tmp_path,
                source_commit=FULL_SOURCE_COMMIT,
                run_root="work",
            )

    run = tmp_path / "run"
    run.mkdir()
    (run / "one.bin").write_bytes(b"one")
    nested = run / "nested"
    nested.mkdir()
    (nested / "two.bin").write_bytes(b"two")
    assert pipeline._run_size_bytes(run) == 6
    (run / "unsafe-link").symlink_to(tmp_path)
    with pytest.raises(pipeline.PipelineResourceLimitError, match="symlink"):
        pipeline._run_size_bytes(run)
    (run / "unsafe-link").unlink()

    config = _config("resource")
    guard = pipeline._ResourceGuard(config)
    monkeypatch.setattr(pipeline, "_process_peak_rss_bytes", lambda: 1)
    monkeypatch.setattr(
        pipeline,
        "PipelineStore",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("resource guard must not use wall-clock pipeline creation")
        ),
    )
    active_elapsed = [1.0]

    def progress_snapshot() -> ProgressSnapshot:
        return ProgressSnapshot(
            sequence=1,
            event_kind=ProgressEventKind.RESUMED,
            state=ProgressState.RUNNING,
            timestamp_utc=datetime(2000, 1, 1, tzinfo=UTC),
            stage="v04_candidate_training",
            elapsed_seconds=active_elapsed[0],
            message="progress reporting resumed",
        )

    context = cast(
        StageContext,
        SimpleNamespace(
            run_directory=run,
            stop_requested=lambda: False,
            progress=SimpleNamespace(snapshot=progress_snapshot),
        ),
    )
    # A decades-old wall timestamp does not consume active execution budget.
    assert guard._elapsed_seconds(context) == 1.0
    assert guard.resource_stop_required(context, force=True) is False
    assert guard.resource_stop_required(context, force=False) is False
    guard.enforce_projected_write(
        context,
        reservation_bytes=config.maximum_run_bytes - 7,
    )
    with pytest.raises(pipeline.PipelineResourceLimitError, match="lacks capacity"):
        guard.enforce_projected_write(
            context,
            reservation_bytes=config.maximum_run_bytes - 6,
        )
    with pytest.raises(TypeError, match="positive integer"):
        guard.enforce_projected_write(context, reservation_bytes=0)
    active_elapsed[0] = float(config.maximum_pipeline_seconds)
    assert guard.resource_stop_required(context, force=True) is True
    with pytest.raises(KeyboardInterrupt):
        guard.enforce_start(context)
    with pytest.raises(pipeline.PipelineResourceLimitError, match="resource boundary"):
        guard.enforce_end(context)
    bad_context = cast(
        StageContext,
        SimpleNamespace(run_directory=run, stop_requested=lambda: 1),
    )
    with pytest.raises(TypeError, match="exact boolean"):
        guard.stop_required(bad_context)


def test_contract_validators_reject_semantic_and_checksum_drift(tmp_path: Path) -> None:
    def reject(
        model: BaseModel,
        model_type: type[BaseModel],
        message: str,
        **updates: object,
    ) -> None:
        payload = model.model_dump(mode="python", round_trip=True)
        payload.update(updates)
        with pytest.raises(ValidationError, match=message):
            model_type.model_validate(payload, strict=True)

    stop_draft = pipeline.PipelineStopRequest.model_construct(
        requested_at="2026-08-23T00:00:00+00:00",
        process_id=1,
        checksum_sha256="0" * 64,
    )
    stop = pipeline._bound_model(stop_draft, pipeline.PipelineStopRequest)
    reject(stop, pipeline.PipelineStopRequest, "checksum", checksum_sha256=HASH_A)

    preflight_draft = pipeline.ExecutionPreflightReport.model_construct(
        runner_source_commit=SOURCE_COMMIT,
        runner_worktree_clean=True,
        pipeline_config_sha256=HASH_A,
        v02_config_sha256=HASH_B,
        v03_config_sha256=HASH_C,
        v04_config_sha256=HASH_D,
        frozen_data_source_commit=SOURCE_COMMIT,
        tokenizer_manifest_sha256=HASH_A,
        compact_contract_sha256=HASH_B,
        v02_inventory_report_sha256=HASH_C,
        v03_counterfactual_cap_report_sha256=HASH_D,
        final_evaluation_automatic=False,
        development_only=True,
        checksum_sha256="0" * 64,
    )
    preflight = pipeline._bound_model(preflight_draft, pipeline.ExecutionPreflightReport)
    reject(preflight, pipeline.ExecutionPreflightReport, "checksum", checksum_sha256=HASH_A)

    prediction_draft = pipeline.PredictionArtifactManifest.model_construct(
        view=RemediationView.IID_VALIDATION,
        example_count=1,
        example_inventory_sha256=HASH_A,
        prediction_inventory_sha256=HASH_B,
        predictions_sha256=HASH_C,
        predictions_size_bytes=1,
        checksum_sha256="0" * 64,
    )
    prediction = pipeline._bound_model(prediction_draft, pipeline.PredictionArtifactManifest)
    reject(prediction, pipeline.PredictionArtifactManifest, "checksum", checksum_sha256=HASH_A)

    structured_separation = pipeline._task_scoped_structured_separation(
        _structured_records_for_views(),
        views=tuple(RemediationView),
    )
    separation_draft = pipeline.DevelopmentSeparationReport.model_construct(
        iid_dataset_manifest_sha256=HASH_A,
        shadow_dataset_manifest_sha256=HASH_B,
        iid_example_count=1,
        shadow_example_count=1,
        group_overlap_count=0,
        example_checksum_overlap_count=0,
        prompt_checksum_overlap_count=0,
        structured_separation=structured_separation,
        passed=True,
        checksum_sha256="0" * 64,
    )
    separation = pipeline._bound_model(separation_draft, pipeline.DevelopmentSeparationReport)
    reject(separation, pipeline.DevelopmentSeparationReport, "pass state", passed=False)
    reject(separation, pipeline.DevelopmentSeparationReport, "checksum", checksum_sha256=HASH_A)

    scores = (
        pipeline.CandidateScore(
            candidate_id="a",
            checkpoint_manifest_sha256=HASH_A,
            semantic_composite=0.6,
            selected_validation_nll=0.4,
            selected_step=2,
            evaluation_report_sha256=HASH_C,
        ),
        pipeline.CandidateScore(
            candidate_id="b",
            checkpoint_manifest_sha256=HASH_B,
            semantic_composite=0.5,
            selected_validation_nll=0.3,
            selected_step=1,
            evaluation_report_sha256=HASH_D,
        ),
    )
    selection_draft = pipeline.CandidateSelectionReport.model_construct(
        selection_manifest_sha256=HASH_E,
        candidates=scores,
        selected_candidate_id="a",
        selected_checkpoint_manifest_sha256=HASH_A,
        checksum_sha256="0" * 64,
    )
    selection = pipeline._bound_model(selection_draft, pipeline.CandidateSelectionReport)
    reject(
        selection, pipeline.CandidateSelectionReport, "unique and sorted", candidates=scores[::-1]
    )
    reject(
        selection, pipeline.CandidateSelectionReport, "frozen ranking", selected_candidate_id="b"
    )
    reject(selection, pipeline.CandidateSelectionReport, "checksum", checksum_sha256=HASH_A)

    pilot = _pilot_report((1, 2, 4))
    reject(pilot, pipeline.V04PilotReport, "D-073", prompt_truncation_rate=0.5)
    reject(pilot, pipeline.V04PilotReport, "activation", activated=False)
    reject(
        pilot, pipeline.V04PilotReport, "batches 1, 2, and 4", measurements=pilot.measurements[:2]
    )
    reject(pilot, pipeline.V04PilotReport, "checksum", checksum_sha256=HASH_A)
    inactive_pilot_draft = pipeline.V04PilotReport.model_construct(
        candidate_id="v04-context-1024",
        requested_device="mps",
        required_resolved_device="mps",
        mandatory_batch_resolved_device=None,
        prompt_truncation_rate=668 / 882,
        v03_train_prompt_truncation_rate=0.05,
        material_truncation_threshold=0.1,
        activated=False,
        measurements=(),
        passed=True,
        checksum_sha256="0" * 64,
    )
    inactive_pilot = pipeline._bound_model(inactive_pilot_draft, pipeline.V04PilotReport)
    reject(
        inactive_pilot,
        pipeline.V04PilotReport,
        "two measured conditions",
        activated=True,
    )

    training_draft = pipeline.V04CandidateTrainingReport.model_construct(
        activated=True,
        candidate_id="long",
        reused_v03_candidate=False,
        source_stage="v04_candidate_training",
        training_result_sha256=HASH_A,
        checkpoint_manifest_sha256=HASH_B,
        checksum_sha256="0" * 64,
    )
    training_report = pipeline._bound_model(
        training_draft,
        pipeline.V04CandidateTrainingReport,
    )
    reject(
        training_report, pipeline.V04CandidateTrainingReport, "source", reused_v03_candidate=True
    )
    reject(training_report, pipeline.V04CandidateTrainingReport, "checksum", checksum_sha256=HASH_A)

    candidate = _candidate(
        "control",
        passed=True,
        worst=0.6,
        iid=0.7,
        context_length=512,
        hash_character="a",
    )
    reject(candidate, pipeline.V04CandidateEvaluation, "shadow order", shadow_reports=())
    index_draft = pipeline.V04EvaluationIndex.model_construct(
        candidates=(candidate,),
        selected_candidate_id="control",
        checkpoint_manifest_sha256=candidate.checkpoint_manifest_sha256,
        iid_report_sha256=candidate.iid_report_sha256,
        shadow_reports=candidate.shadow_reports,
        v04_acceptance_sha256=candidate.v04_acceptance_sha256,
        checksum_sha256="0" * 64,
    )
    index = pipeline._bound_model(index_draft, pipeline.V04EvaluationIndex)
    reject(index, pipeline.V04EvaluationIndex, "shadow view order", shadow_reports=())
    reject(index, pipeline.V04EvaluationIndex, "selected evidence", selected_candidate_id="other")
    reject(index, pipeline.V04EvaluationIndex, "checksum", checksum_sha256=HASH_B)

    policy = _future_policy()
    reject(
        policy,
        pipeline.FinalEvaluationPolicyFreeze,
        "status differs",
        status="locked_development_gate_failed",
    )
    reject(policy, pipeline.FinalEvaluationPolicyFreeze, "checksum", checksum_sha256=HASH_A)
    review = _complete_review(policy)
    wrong_stages = (
        review.stages[0].model_copy(update={"stage": "wrong"}),
        *review.stages[1:],
    )
    reject(review, pipeline.ReviewBundleManifest, "frozen graph", stages=wrong_stages)
    reject(review, pipeline.ReviewBundleManifest, "checksum", checksum_sha256=HASH_A)

    terminal_stage = pipeline.TerminalReviewStage(
        stage="preflight",
        status="completed",
        latest_attempt_path="stages/01-preflight/attempt-0001",
        outcome=_artifact("preflight/outcome.json"),
    )
    reject(terminal_stage, pipeline.TerminalReviewStage, "unsafe", latest_attempt_path="../escape")
    reject(terminal_stage, pipeline.TerminalReviewStage, "lacks", outcome=None)
    reject(terminal_stage, pipeline.TerminalReviewStage, "cannot publish", status="failed")

    config, state = _terminal_state(tmp_path, "pre-stage-stop")
    terminal = cast(
        pipeline.TerminalReviewBundleManifest,
        pipeline.write_terminal_review_bundle(
            project_root=tmp_path,
            config=config,
            source_commit=SOURCE_COMMIT,
            state=state,
        ).manifest,
    )
    reject(
        terminal,
        pipeline.TerminalReviewBundleManifest,
        "completed-prefix",
        completed_prefix_length=len(PIPELINE_STAGES),
    )
    reject(terminal, pipeline.TerminalReviewBundleManifest, "checksum", checksum_sha256=HASH_A)


def test_runtime_binding_and_v04_checkpoint_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = cast(object, SimpleNamespace())
    runtime = pipeline._PipelineRuntime(
        project_root=tmp_path,
        config=_config("runtime-binding"),
        source_commit=SOURCE_COMMIT,
        inputs=cast(
            pipeline._ExecutionInputs,
            SimpleNamespace(tokenizer=tokenizer),
        ),
    )
    guard_calls: list[str] = []
    runtime.guard = cast(
        pipeline._ResourceGuard,
        SimpleNamespace(
            enforce_start=lambda _context: guard_calls.append("start"),
            enforce_end=lambda _context: guard_calls.append("end"),
        ),
    )
    context = cast(
        StageContext,
        SimpleNamespace(project_root=tmp_path, source_commit=SOURCE_COMMIT),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_runner_source",
        lambda *_args, **_kwargs: SOURCE_COMMIT,
    )
    runtime._start(context)
    outcome = StageOutcome(summary="bound")
    assert runtime._finish(context, outcome) is outcome
    assert guard_calls == ["start", "end"]
    other_root = tmp_path / "other"
    other_root.mkdir()
    with pytest.raises(pipeline.PipelineExecutionError, match="project root"):
        runtime._start(
            cast(
                StageContext,
                SimpleNamespace(project_root=other_root, source_commit=SOURCE_COMMIT),
            )
        )
    with pytest.raises(pipeline.PipelineExecutionError, match="source commit"):
        runtime._start(
            cast(
                StageContext,
                SimpleNamespace(project_root=tmp_path, source_commit="bbbbbbb"),
            )
        )

    report = SimpleNamespace(
        reused_v03_candidate=False,
        candidate_id="long",
        training_result_sha256=HASH_A,
        checkpoint_manifest_sha256=HASH_B,
    )
    result = SimpleNamespace(
        checksum_sha256=HASH_A,
        checkpoint_manifest_sha256=HASH_B,
    )
    model = object()
    checkpoint = SimpleNamespace(checksum_sha256=HASH_B)
    monkeypatch.setattr(pipeline, "_upstream_attempt", lambda *_args: tmp_path)
    monkeypatch.setattr(pipeline, "_read_contract", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(pipeline, "_load_training_result", lambda *_args: result)
    monkeypatch.setattr(
        pipeline,
        "_load_candidate_checkpoint",
        lambda *_args: (model, checkpoint, torch.device("cpu")),
    )

    observed = runtime._v04_checkpoint(cast(StageContext, SimpleNamespace()))
    assert id(observed[0]) == id(report)
    assert id(observed[1]) == id(result)
    assert id(observed[2]) == id(model)
    assert id(observed[3]) == id(checkpoint)
    assert observed[4].type == "cpu"

    report.training_result_sha256 = HASH_C
    with pytest.raises(pipeline.PipelineExecutionError, match="binding differs"):
        runtime._v04_checkpoint(cast(StageContext, SimpleNamespace()))


def test_low_level_path_and_resume_rejection_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="must be a string"):
        pipeline._canonical_utc(cast(str, 1))
    with pytest.raises(ValueError, match="missing, unsafe, or oversized"):
        pipeline._read_contract(tmp_path / "missing.json", pipeline.PipelineStopRequest)

    stop_draft = pipeline.PipelineStopRequest.model_construct(
        requested_at="2026-08-23T00:00:00+00:00",
        process_id=1,
        checksum_sha256="0" * 64,
    )
    stop = pipeline._bound_model(stop_draft, pipeline.PipelineStopRequest)
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(
        b" " + canonical_json_bytes(stop.model_dump(mode="json", round_trip=True)) + b"\n"
    )
    with pytest.raises(ValueError, match="not canonical"):
        pipeline._read_contract(noncanonical, pipeline.PipelineStopRequest)

    run = tmp_path / "run"
    real = run / "real"
    real.mkdir(parents=True)
    payload = real / "payload.bin"
    payload.write_bytes(b"payload")
    alias = run / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="traverses a symlink"):
        pipeline._artifact_reference(alias / "payload.bin", run_directory=run)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="escapes the run"):
        pipeline._artifact_reference(outside, run_directory=run)
    with pytest.raises(ValueError, match="regular file"):
        pipeline._artifact_reference(run / "missing.bin", run_directory=run)
    with pytest.raises(ValueError, match="directory is unsafe"):
        pipeline._directory_artifacts(run / "missing", run_directory=run)
    with pytest.raises(ValueError, match="contains a symlink"):
        pipeline._directory_artifacts(run, run_directory=run)

    input_real = tmp_path / "input-real"
    input_real.mkdir()
    (input_real / "config.toml").write_text("version = 1\n", encoding="ascii")
    input_alias = tmp_path / "input-alias"
    input_alias.symlink_to(input_real, target_is_directory=True)
    with pytest.raises(ValueError, match="traverses a symlink"):
        pipeline._safe_input_path(tmp_path, "input-alias/config.toml", kind="file")
    with pytest.raises(TypeError, match="project/config"):
        pipeline.pipeline_stop_file(
            project_root=tmp_path,
            config=cast(PipelineConfig, object()),
        )
    with pytest.raises(TypeError, match="canonical marker"):
        pipeline.request_pipeline_stop(tmp_path / "wrong")
    with pytest.raises(TypeError, match="canonical marker"):
        pipeline.archive_pipeline_stop(tmp_path / "wrong")

    invalid_attempt = run / "stages/training/current"
    invalid_attempt.mkdir(parents=True)
    invalid_context = cast(StageContext, SimpleNamespace(attempt_directory=invalid_attempt))
    with pytest.raises(ValueError, match="attempt name"):
        pipeline._latest_resume_source(invalid_context, "control")
    empty_stage = run / "stages/empty"
    current = empty_stage / "attempt-0001"
    current.mkdir(parents=True)
    empty_context = cast(StageContext, SimpleNamespace(attempt_directory=current))
    assert pipeline._latest_resume_source(empty_context, "control") is None
    prior_root = empty_stage / "attempt-0000/training-state/control"
    prior_root.mkdir(parents=True)
    (prior_root / ".state-step-00000001.lock").write_text("forensic\n", encoding="ascii")
    (prior_root / ".state-step-00000001.tmp-resume").mkdir()
    assert pipeline._latest_resume_source(empty_context, "control") is None
    unknown = prior_root / "unexpected.bin"
    unknown.write_bytes(b"unsafe")
    with pytest.raises(ValueError, match="unexpected entry"):
        pipeline._latest_resume_source(empty_context, "control")
    unknown.unlink()
    unsafe_state = prior_root / "state-step-00000002"
    unsafe_state.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink or non-directory"):
        pipeline._latest_resume_source(empty_context, "control")
    unsafe_state.unlink()
    with pytest.raises(ValueError, match="regular state directory"):
        pipeline._copy_resume_state(tmp_path / "missing-state", tmp_path / "destination")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="safe"),
    )
    with pytest.raises(pipeline.PipelineExecutionError, match="failed safely"):
        pipeline._run_process((pipeline.TRUSTED_GIT, "--version"), project_root=tmp_path)
