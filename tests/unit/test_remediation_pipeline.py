"""Focused tests for the concrete, development-only remediation pipeline adapter."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Literal, cast

import pytest
import torch
from pydantic import BaseModel, ValidationError

import reactorbench.remediation.pipeline as pipeline
from reactorbench.evaluation.compact import compact_output_contract
from reactorbench.model import TransformerConfig, TransformerLM
from reactorbench.model.checkpoint import CheckpointManifest
from reactorbench.remediation.acceptance import DevelopmentArtifactBinding
from reactorbench.remediation.config import (
    PIPELINE_STAGES,
    SHADOW_VIEWS,
    PipelineConfig,
    RemediationTraining,
    RemediationView,
    config_sha256,
)
from reactorbench.remediation.data import RemediationExample, SafeDevelopmentDataset
from reactorbench.remediation.decoding import MAX_DECODE_BATCH_SIZE, DualPathCompactPrediction
from reactorbench.remediation.inventory import CompactInventoryReport
from reactorbench.remediation.metrics import (
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
from reactorbench.remediation.progress import ProgressMetric
from reactorbench.remediation.serialization import CompactTokenizedExample
from reactorbench.remediation.training import CompactTrainingResult, TrainingProgress
from reactorbench.schemas.base import canonical_json_bytes, canonical_sha256
from reactorbench.schemas.enums import TaskName
from reactorbench.tokenizer import ProjectTokenizer

SOURCE_COMMIT = "abcdef0"
FULL_SOURCE_COMMIT = "a" * 40
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


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


def _pilot_report(batch_sizes: tuple[int, ...]) -> pipeline.V04PilotReport:
    measurements = tuple(
        pipeline.V04PilotMeasurement(
            batch_size=batch_size,
            training_result_sha256=HASH_A,
            finite_loss=True,
            checkpoint_reloaded=True,
            elapsed_seconds=1.0,
            process_peak_rss_bytes=1,
        )
        for batch_size in batch_sizes
    )
    draft = pipeline.V04PilotReport.model_construct(
        candidate_id="v04-context-1024",
        prompt_truncation_rate=668 / 882,
        v03_train_prompt_truncation_rate=0.5,
        material_truncation_threshold=0.1,
        activated=True,
        measurements=measurements,
        passed=True,
        checksum_sha256="0" * 64,
    )
    return pipeline._bound_model(draft, pipeline.V04PilotReport)


def test_v04_candidate_training_requires_the_main_batch_to_pass_the_pilot(
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
    monkeypatch.setattr(runtime, "_start", lambda _context: None)
    monkeypatch.setattr(pipeline, "_upstream_attempt", lambda *_args: tmp_path)
    complete_pilot = _pilot_report((1, 2, 4))
    incomplete_pilot = pipeline.V04PilotReport.model_construct(
        **complete_pilot.model_dump(mode="python", exclude={"measurements"}),
        measurements=complete_pilot.measurements[:2],
    )

    def read(_path: Path, model_type: type[object], **_kwargs: object) -> object:
        if model_type is pipeline.V04PilotReport:
            return incomplete_pilot
        return SimpleNamespace(selected_candidate_id="control")

    monkeypatch.setattr(pipeline, "_read_contract", read)

    with pytest.raises(pipeline.PipelineExecutionError, match="lacks a passing MPS pilot"):
        runtime.v04_candidate_training(cast(StageContext, SimpleNamespace()))

    assert {item.batch_size for item in complete_pilot.measurements} == {1, 2, 4}
    with pytest.raises(ValidationError, match="batches 1, 2, and 4"):
        _pilot_report((1, 2))


def _bound_v02_report(
    *,
    prompt_truncation_count: int = pipeline.V02_FROZEN_PROMPT_TRUNCATION_COUNT,
    advancement_allowed: bool = True,
) -> pipeline.V02DevelopmentGateReport:
    draft = pipeline.V02DevelopmentGateReport.model_construct(
        inventory_report_sha256=HASH_A,
        prediction_manifest_sha256=HASH_B,
        example_count=252,
        constrained_parse_rate=1.0,
        constrained_schema_validity_rate=1.0,
        unconstrained_parse_rate=0.0,
        unconstrained_schema_validity_rate=0.0,
        generation_cap_exhaustion_rate=0.0,
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
        v03=SimpleNamespace(candidates=candidates),
        tokenizer=object(),
    )
    runtime = pipeline._PipelineRuntime(
        project_root=tmp_path,
        config=config,
        source_commit=SOURCE_COMMIT,
        inputs=cast(pipeline._ExecutionInputs, inputs),
    )
    monkeypatch.setattr(runtime, "_start", lambda _context: None)
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
            checksum_sha256=hashlib.sha256(str(len(evaluated_sizes)).encode()).hexdigest()
        )
        return evaluation, None, (prediction,), ()

    monkeypatch.setattr(pipeline, "_evaluate_candidate_view", evaluate)
    monkeypatch.setattr(
        pipeline,
        "semantic_composite_score",
        lambda _examples, predictions: predictions[0].score,
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
    monkeypatch.setattr(runtime, "_start", lambda _context: None)
    monkeypatch.setattr(runtime, "_finish", lambda _context, outcome: outcome)
    monkeypatch.setattr(pipeline, "config_sha256", lambda _config: HASH_A)
    monkeypatch.setattr(
        pipeline,
        "_load_stage_dataset",
        lambda _context, _config, stage, _name: iid if stage == "v03_data_audit" else shadow,
    )
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
            SimpleNamespace(checksum_sha256=checksum, view_metrics=SimpleNamespace()),
            None,
            (SimpleNamespace(score=score),),
            (),
        )

    monkeypatch.setattr(pipeline, "_evaluate_candidate_view", evaluate)
    monkeypatch.setattr(
        pipeline,
        "semantic_composite_score",
        lambda _examples, predictions: predictions[0].score,
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
        ),
    )


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
        view=RemediationView.IID_TRAIN,
        group_id="train-group",
        checksum_sha256=HASH_A,
        prompt_sha256=HASH_B,
    )
    validation = SimpleNamespace(
        view=RemediationView.IID_VALIDATION,
        group_id="validation-group",
        checksum_sha256=HASH_C,
        prompt_sha256=HASH_D,
    )
    iid = SimpleNamespace(
        examples=(train, validation),
        manifest=SimpleNamespace(checksum_sha256=HASH_A),
    )
    shadow_examples = tuple(
        SimpleNamespace(
            view=view,
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
    monkeypatch.setattr(runtime, "_start", lambda _context: None)
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
    )
    unconstrained = SimpleNamespace(
        compact_parse_success=False,
        schema_valid=False,
        generation_cap_exhausted=False,
    )
    prediction = SimpleNamespace(constrained=constrained, unconstrained=unconstrained)
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
    assert training_calls == ["v02-smoke", "v02-development", "v03-smoke", "control"]


def test_v04_pilot_and_candidate_training_execute_both_source_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _runtime_context(tmp_path)
    train = SimpleNamespace(view=RemediationView.IID_TRAIN)
    validation = SimpleNamespace(view=RemediationView.IID_VALIDATION)
    iid = SimpleNamespace(examples=(train, validation))
    shadow = SimpleNamespace(examples=tuple(SimpleNamespace(view=view) for view in SHADOW_VIEWS))
    base_training = SimpleNamespace(
        seed=1,
        device="cpu",
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
                pilot=SimpleNamespace(candidate_id="long", steps=2, batch_sizes=(1, 2, 4)),
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
    monkeypatch.setattr(runtime, "_start", lambda _context: None)
    monkeypatch.setattr(runtime, "_finish", lambda _context, outcome: outcome)
    monkeypatch.setattr(pipeline, "_load_stage_dataset", lambda *_args: iid)
    monkeypatch.setattr(
        pipeline,
        "_tokenize_examples",
        lambda *_args, **_kwargs: (
            SimpleNamespace(prompt_truncated=True),
            SimpleNamespace(prompt_truncated=False),
        ),
    )
    monkeypatch.setattr(pipeline, "_smoke_examples", lambda _dataset: ((train,), (validation,)))
    result = SimpleNamespace(
        checksum_sha256=HASH_A,
        checkpoint_manifest_sha256=HASH_B,
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
    assert tuple(item.batch_size for item in pilot.measurements) == (1, 2, 4)
    assert training_calls == [("long-b1", 1), ("long-b2", 2), ("long-b4", 4)]
    assert pilot_outcome.advancement_allowed is True

    monkeypatch.setattr(pipeline, "_upstream_attempt", lambda *_args: tmp_path)
    selection = SimpleNamespace(selected_candidate_id="control")

    def read(path: Path, model_type: type[object], **_kwargs: object) -> object:
        if model_type is pipeline.V04PilotReport:
            return pilot
        if model_type is pipeline.CandidateSelectionReport:
            return selection
        return SimpleNamespace()

    monkeypatch.setattr(pipeline, "_read_contract", read)
    monkeypatch.setattr(
        pipeline,
        "_load_stage_dataset",
        lambda _context, _config, stage, _name: shadow if stage == "v04_shadow_freeze" else iid,
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_semantic_selection_examples",
        lambda *_args: (validation,),
    )
    monkeypatch.setattr(pipeline, "_decode_examples", lambda *_args, **_kwargs: (object(),))
    monkeypatch.setattr(pipeline, "semantic_composite_score", lambda *_args, **_kwargs: 0.75)
    training_calls.clear()
    active_outcome = runtime.v04_candidate_training(context)
    active_report = cast(
        pipeline.V04CandidateTrainingReport,
        written["v04-candidate-training.json"],
    )
    assert active_report.activated is True
    assert active_report.source_stage == "v04_candidate_training"
    assert training_calls == [("long", 4)]
    assert active_outcome.advancement_allowed is True

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
    inputs = cast(pipeline._ExecutionInputs, SimpleNamespace())
    runtime = pipeline._PipelineRuntime(
        project_root=tmp_path,
        config=_config("runtime"),
        source_commit=SOURCE_COMMIT,
        inputs=inputs,
    )
    monkeypatch.setattr(runtime, "_start", lambda _context: None)
    monkeypatch.setattr(runtime, "_finish", lambda _context, outcome: outcome)
    monkeypatch.setattr(pipeline, "_upstream_attempt", lambda *_args: tmp_path)
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
    acceptance = SimpleNamespace(
        checksum_sha256=index.v04_acceptance_sha256,
        advancement_allowed=True,
    )

    def read_gate(path: Path, model_type: type[object], **_kwargs: object) -> object:
        if model_type is pipeline.V04EvaluationIndex:
            return index
        return acceptance

    monkeypatch.setattr(pipeline, "_read_contract", read_gate)
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


def test_file_contract_path_and_stop_helpers_are_checksum_bound(tmp_path: Path) -> None:
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
    archived = pipeline.archive_pipeline_stop(stop_path)
    assert archived is not None
    assert archived.is_file()
    assert stop_requested() is False
    assert pipeline.archive_pipeline_stop(stop_path) is None
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
    assert pipeline._latest_resume_source(context, "control") == prior
    copied = pipeline._copy_resume_state(prior, attempt / "copied")
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
    guard = cast(pipeline._ResourceGuard, SimpleNamespace(stop_required=lambda _context: False))

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
    monkeypatch.setattr(pipeline, "run_remediation_baselines", lambda *_args, **_kwargs: baseline)
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

    tokenized = SimpleNamespace(
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
    assert (
        len(pipeline._tokenized_inventory_sha256((cast(CompactTokenizedExample, tokenized),))) == 64
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
    many_examples = (train_example,) * (MAX_DECODE_BATCH_SIZE + 1)
    decoded = pipeline._decode_examples(
        cast(TransformerLM, object()),
        cast(ProjectTokenizer, SimpleNamespace()),
        many_examples,
        generation_caps={},
        device=torch.device("cpu"),
    )
    assert len(decoded) == len(many_examples)
    assert decode_calls == [MAX_DECODE_BATCH_SIZE, 1]

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
            device=SimpleNamespace(resolved="cpu"),
            checkpoint_manifest_sha256=HASH_D,
        ),
    )
    tokenizer = SimpleNamespace(manifest=SimpleNamespace(checksum_sha256=HASH_C))
    loaded_model = object()
    loaded_manifest = SimpleNamespace(checksum_sha256=HASH_D)
    observed_checkpoint: list[dict[str, object]] = []

    def load_checkpoint(path: Path, **kwargs: object) -> tuple[object, object]:
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
    assert device.type == "cpu"
    assert observed_checkpoint[0]["expected_tokenizer_sha256"] == HASH_C


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
    state = SimpleNamespace(created_at="2026-08-23T00:00:00+00:00")
    monkeypatch.setattr(
        pipeline,
        "PipelineStore",
        lambda *_args, **_kwargs: SimpleNamespace(load_state=lambda: state),
    )
    monkeypatch.setattr(pipeline, "_process_peak_rss_bytes", lambda: 1)
    context = cast(
        StageContext,
        SimpleNamespace(
            run_directory=run,
            stop_requested=lambda: False,
        ),
    )
    assert guard._elapsed_seconds(context) >= 0.0
    assert guard.resource_stop_required(context, force=True) is True
    assert guard.resource_stop_required(context, force=False) is False
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

    separation_draft = pipeline.DevelopmentSeparationReport.model_construct(
        iid_dataset_manifest_sha256=HASH_A,
        shadow_dataset_manifest_sha256=HASH_B,
        iid_example_count=1,
        shadow_example_count=1,
        group_overlap_count=0,
        example_checksum_overlap_count=0,
        prompt_checksum_overlap_count=0,
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
    with pytest.raises(ValueError, match="regular state directory"):
        pipeline._copy_resume_state(tmp_path / "missing-state", tmp_path / "destination")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="safe"),
    )
    with pytest.raises(pipeline.PipelineExecutionError, match="failed safely"):
        pipeline._run_process((pipeline.TRUSTED_GIT, "--version"), project_root=tmp_path)
