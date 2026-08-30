"""Focused crash-safety tests for the generic remediation stage engine."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

import reactorbench.remediation.orchestration as orchestration
from reactorbench.remediation.config import PIPELINE_STAGES, PipelineConfig, config_sha256
from reactorbench.remediation.orchestration import (
    ArtifactReference,
    PipelineBusyError,
    PipelineEngine,
    PipelineError,
    PipelineStageError,
    PipelineState,
    PipelineStore,
    RunManifest,
    StageAction,
    StageContext,
    StageMetric,
    StageOutcome,
    StageRecord,
    StageStatus,
    command_tuple,
)
from reactorbench.remediation.progress import (
    ProgressEventKind,
    ProgressIOError,
    ProgressReporter,
    ProgressSnapshot,
    ProgressState,
)
from reactorbench.schemas.base import canonical_json_bytes

SOURCE_COMMIT = "abcdef0"
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _config(**changes: object) -> PipelineConfig:
    payload: dict[str, object] = {
        "pipeline_version": "0.4.0",
        "run_name": "test-run",
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
    payload.update(changes)
    return PipelineConfig.model_validate(payload)


def _store(project_root: Path, config: PipelineConfig | None = None) -> PipelineStore:
    selected = _config() if config is None else config
    return PipelineStore.create(
        project_root=project_root,
        config=selected,
        pipeline_config_sha256=config_sha256(selected),
        source_commit=SOURCE_COMMIT,
        command=("python", "-m", "reactorbench.remediation"),
    )


def _artifact(context: StageContext, name: str, payload: bytes) -> ArtifactReference:
    path = context.attempt_directory / name
    path.write_bytes(payload)
    return ArtifactReference(
        relative_path=path.relative_to(context.run_directory).as_posix(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _successful_action(stage: str, calls: list[tuple[str, str]]) -> StageAction:
    def action(context: StageContext) -> StageOutcome:
        calls.append((stage, context.attempt_directory.name))
        artifact = _artifact(context, "result.txt", stage.encode("ascii"))
        return StageOutcome(
            summary=f"{stage} completed.",
            artifacts=(artifact,),
            metrics=(StageMetric(name="loss", value=1.0, unit="nll"),),
        )

    return action


def _actions(
    calls: list[tuple[str, str]], overrides: Mapping[str, StageAction] | None = None
) -> dict[str, StageAction]:
    replacements = {} if overrides is None else dict(overrides)
    return {
        stage: replacements.get(stage, _successful_action(stage, calls))
        for stage in PIPELINE_STAGES
    }


def _engine(
    project_root: Path,
    store: PipelineStore,
    config: PipelineConfig,
    actions: Mapping[str, StageAction],
    *,
    stop_requested: Callable[[], bool] = lambda: False,
) -> PipelineEngine:
    return PipelineEngine(
        project_root=project_root,
        config=config,
        store=store,
        actions=actions,
        stop_requested=stop_requested,
    )


def _progress_events(store: PipelineStore) -> tuple[ProgressSnapshot, ...]:
    records = (store.run_directory / "progress.jsonl").read_bytes().splitlines()
    return tuple(ProgressSnapshot.model_validate_json(record, strict=True) for record in records)


def _rewrite_canonical(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def test_full_run_uses_exact_order_markers_progress_and_idempotent_return(
    tmp_path: Path,
) -> None:
    config = _config()
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []
    engine = _engine(tmp_path, store, config, _actions(calls))

    result = engine.run()

    assert result.status == "completed"
    assert [stage for stage, _ in calls] == list(PIPELINE_STAGES)
    assert all(attempt == "attempt-0001" for _, attempt in calls)
    assert all(record.status is StageStatus.COMPLETED for record in result.stages)
    assert all(record.artifact_count == 1 for record in result.stages)
    assert all(record.metric_count == 1 for record in result.stages)
    for ordinal, stage in enumerate(PIPELINE_STAGES):
        root = store.run_directory / "stages" / f"{ordinal:02d}-{stage}"
        assert (root / "attempt-0001" / "outcome.json").is_file()
        assert (root / "completed.json").is_file()

    events = _progress_events(store)
    assert events[0].event_kind is ProgressEventKind.STARTED
    assert events[-1].event_kind is ProgressEventKind.COMPLETED
    assert events[-1].state is ProgressState.COMPLETED
    assert sum(event.event_kind is ProgressEventKind.PROGRESS for event in events) == len(
        PIPELINE_STAGES
    )
    assert sum(event.event_kind is ProgressEventKind.CHECKPOINT for event in events) == len(
        PIPELINE_STAGES
    )
    final_status = ProgressSnapshot.model_validate_json(
        (store.run_directory / "status.json").read_bytes(), strict=True
    )
    assert final_status == events[-1]

    state_bytes = store.state_path.read_bytes()
    event_bytes = (store.run_directory / "progress.jsonl").read_bytes()
    assert engine.run() == result
    assert calls == [(stage, "attempt-0001") for stage in PIPELINE_STAGES]
    assert store.state_path.read_bytes() == state_bytes
    assert (store.run_directory / "progress.jsonl").read_bytes() == event_bytes


def test_dry_run_is_read_only_and_validates_exact_runtime_boundaries(tmp_path: Path) -> None:
    config = _config()
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []
    engine = _engine(tmp_path, store, config, _actions(calls))
    state_before = store.state_path.read_bytes()

    audited = engine.run(dry_run=True)

    assert audited.status == "ready"
    assert calls == []
    assert store.state_path.read_bytes() == state_before
    assert not (store.run_directory / "progress.jsonl").exists()
    with pytest.raises(TypeError, match="exact boolean"):
        engine.run(dry_run=cast(bool, 1))


def test_diagnostic_sweep_records_both_scientific_failures_and_finishes_graph(
    tmp_path: Path,
) -> None:
    config = _config(
        run_name="phase6-remediation-v0.4.0-targeted-05-diagnostic-01",
        diagnostic_mode="collect_scientific_failures",
    )
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []

    def scientific_failure(context: StageContext) -> StageOutcome:
        stage = context.attempt_directory.parent.name.split("-", 1)[1]
        calls.append((stage, context.attempt_directory.name))
        return StageOutcome(
            summary="Scientific acceptance threshold did not pass.",
            advancement_allowed=False,
        )

    actions = _actions(
        calls,
        {
            "v03_gate": scientific_failure,
            "v04_gate_and_final_policy_freeze": scientific_failure,
        },
    )
    result = _engine(tmp_path, store, config, actions).run()

    assert result.status == "diagnostic_completed"
    assert result.stages[9].status is StageStatus.SCIENTIFIC_FAILED
    assert result.stages[14].status is StageStatus.SCIENTIFIC_FAILED
    assert result.stages[15].status is StageStatus.COMPLETED
    assert len(calls) == len(PIPELINE_STAGES)
    events = _progress_events(store)
    assert events[-1].state is ProgressState.COMPLETED
    assert "Diagnostic sweep completed" in events[-1].message


def test_diagnostic_sweep_does_not_bypass_non_allowlisted_failure(tmp_path: Path) -> None:
    config = _config(
        run_name="phase6-remediation-v0.4.0-targeted-05-diagnostic-01",
        diagnostic_mode="collect_scientific_failures",
    )
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []

    def blocked(context: StageContext) -> StageOutcome:
        calls.append(("v04_pilot", context.attempt_directory.name))
        return StageOutcome(summary="Pilot feasibility gate blocked.", advancement_allowed=False)

    result = _engine(
        tmp_path,
        store,
        config,
        _actions(calls, {"v04_pilot": blocked}),
    ).run()

    assert result.status == "blocked"
    assert result.stages[11].status is StageStatus.BLOCKED
    assert result.stages[12].status is StageStatus.PENDING


def test_diagnostic_sweep_stops_on_an_engineering_exception(tmp_path: Path) -> None:
    config = _config(
        run_name="phase6-remediation-v0.4.0-targeted-05-diagnostic-01",
        diagnostic_mode="collect_scientific_failures",
    )
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []

    def engineering_failure(context: StageContext) -> StageOutcome:
        calls.append(("v03_gate", context.attempt_directory.name))
        raise RuntimeError("simulated internal stage failure")

    with pytest.raises(PipelineStageError, match="internal stage callback failed"):
        _engine(
            tmp_path,
            store,
            config,
            _actions(calls, {"v03_gate": engineering_failure}),
        ).run()

    result = store.load_state()
    assert result.status == "failed"
    assert result.stages[9].status is StageStatus.FAILED
    assert result.stages[10].status is StageStatus.PENDING


def test_create_and_engine_reject_config_drift_order_drift_and_existing_run(
    tmp_path: Path,
) -> None:
    config = _config()
    with pytest.raises(ValueError, match="config checksum"):
        PipelineStore.create(
            project_root=tmp_path,
            config=config,
            pipeline_config_sha256="0" * 64,
            source_commit=SOURCE_COMMIT,
            command=("python",),
        )
    store = _store(tmp_path, config)
    with pytest.raises(FileExistsError, match="already exists"):
        _store(tmp_path, config)

    calls: list[tuple[str, str]] = []
    reversed_actions = dict(reversed(tuple(_actions(calls).items())))
    with pytest.raises(ValueError, match="frozen stage order"):
        _engine(tmp_path, store, config, reversed_actions)

    changed = _config(maximum_pipeline_seconds=3601)
    with pytest.raises(ValueError, match="different source or configuration"):
        _engine(tmp_path, store, changed, _actions(calls)).run(dry_run=True)
    assert calls == []


def test_crash_after_marker_before_state_update_recovers_without_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []
    actions = _actions(calls)
    original_write_state = store.write_state
    crashed = False

    def crash_after_marker(candidate: PipelineState) -> PipelineState:
        nonlocal crashed
        marker = store.run_directory / "stages/00-preflight/completed.json"
        if not crashed and marker.exists() and candidate.current_stage is None:
            crashed = True
            raise KeyboardInterrupt
        return original_write_state(candidate)

    monkeypatch.setattr(store, "write_state", crash_after_marker)
    with pytest.raises(KeyboardInterrupt):
        _engine(tmp_path, store, config, actions).run()
    marker_path = store.run_directory / "stages/00-preflight/completed.json"
    marker_before = marker_path.read_bytes()
    interrupted = store.load_state()
    assert interrupted.current_stage == "preflight"
    assert interrupted.stages[0].status is StageStatus.RUNNING
    assert calls == [("preflight", "attempt-0001")]

    monkeypatch.setattr(store, "write_state", original_write_state)
    resumed = _engine(tmp_path, store, config, actions).run()

    assert resumed.status == "completed"
    assert resumed.interruption_count == 1
    assert calls.count(("preflight", "attempt-0001")) == 1
    assert [stage for stage, _ in calls] == list(PIPELINE_STAGES)
    assert marker_path.read_bytes() == marker_before
    assert not (marker_path.parent / "attempt-0002").exists()
    assert ProgressEventKind.RESUMED in {event.event_kind for event in _progress_events(store)}


def test_keyboard_interrupt_stops_and_resume_uses_new_attempt(tmp_path: Path) -> None:
    config = _config()
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []
    first = True

    def interrupt_once(context: StageContext) -> StageOutcome:
        nonlocal first
        calls.append(("preflight", context.attempt_directory.name))
        if first:
            first = False
            raise KeyboardInterrupt
        return StageOutcome(summary="preflight resumed.")

    actions = _actions(calls, {"preflight": interrupt_once})
    stopped = _engine(tmp_path, store, config, actions).run()
    assert stopped.status == "stopped"
    assert stopped.stages[0].status is StageStatus.STOPPED
    assert stopped.stages[0].attempt_count == 1
    assert not (store.run_directory / "stages/00-preflight/completed.json").exists()

    resumed = _engine(tmp_path, store, config, actions).run()
    assert resumed.status == "completed"
    assert resumed.stages[0].attempt_count == 2
    assert calls[:2] == [("preflight", "attempt-0001"), ("preflight", "attempt-0002")]
    kinds = [event.event_kind for event in _progress_events(store)]
    assert ProgressEventKind.STOPPED in kinds
    assert ProgressEventKind.RESUMED in kinds


def test_stop_before_stage_is_resumable_without_consuming_attempt(tmp_path: Path) -> None:
    config = _config()
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []
    should_stop = True

    def stop_requested() -> bool:
        return should_stop

    stopped = _engine(
        tmp_path,
        store,
        config,
        _actions(calls),
        stop_requested=stop_requested,
    ).run()
    assert stopped.status == "stopped"
    assert stopped.stages[0].status is StageStatus.PENDING
    assert calls == []

    should_stop = False
    completed = _engine(
        tmp_path,
        store,
        config,
        _actions(calls),
        stop_requested=stop_requested,
    ).run()
    assert completed.status == "completed"
    assert calls[0] == ("preflight", "attempt-0001")


def test_failed_action_has_safe_error_and_can_retry_in_new_attempt(tmp_path: Path) -> None:
    config = _config()
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []
    unsafe_detail = "/private/sensitive/model-path"

    def unsafe_failure(context: StageContext) -> StageOutcome:
        calls.append(("preflight", context.attempt_directory.name))
        raise RuntimeError(unsafe_detail)

    with pytest.raises(PipelineStageError) as captured:
        _engine(
            tmp_path,
            store,
            config,
            _actions(calls, {"preflight": unsafe_failure}),
        ).run()
    assert unsafe_detail not in str(captured.value)
    failed = store.load_state()
    assert failed.status == "failed"
    assert failed.stages[0].summary == ("Stage failed safely: an internal stage callback failed.")
    diagnostic_path = (
        store.run_directory / "stages/00-preflight/attempt-0001/failure-diagnostic.json"
    )
    diagnostic = json.loads(diagnostic_path.read_bytes())
    assert diagnostic["stage"] == "preflight"
    assert diagnostic["public_category"] == failed.stages[0].summary
    assert diagnostic["exception_chain"] == ["builtins.RuntimeError"]
    assert unsafe_detail not in diagnostic_path.read_text(encoding="ascii")
    assert len(diagnostic["fingerprint_sha256"]) == 64
    assert len(diagnostic["checksum_sha256"]) == 64

    completed = _engine(tmp_path, store, config, _actions(calls)).run()
    assert completed.status == "completed"
    assert calls[:2] == [("preflight", "attempt-0001"), ("preflight", "attempt-0002")]


def test_failed_state_is_durable_before_progress_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    store = _store(tmp_path, config)
    private_detail = "/private/sensitive/progress-path"

    def unsafe_failure(context: StageContext) -> StageOutcome:
        del context
        raise ValueError(private_detail)

    def fail_progress(
        self: ProgressReporter, *, message: str = "pipeline failed safely"
    ) -> ProgressSnapshot:
        del self, message
        raise ProgressIOError(private_detail)

    monkeypatch.setattr(ProgressReporter, "fail", fail_progress)
    with pytest.raises(ProgressIOError):
        _engine(
            tmp_path,
            store,
            config,
            _actions([], {"preflight": unsafe_failure}),
        ).run()

    failed = store.load_state()
    assert failed.status == "failed"
    assert failed.current_stage is None
    assert failed.stages[0].status is StageStatus.FAILED
    assert failed.stages[0].summary == (
        "Stage failed safely: contract or boundary validation failed."
    )
    assert private_detail not in store.state_path.read_text(encoding="utf-8")


def test_blocked_outcome_commits_marker_and_never_runs_later_stages(tmp_path: Path) -> None:
    config = _config()
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []

    def blocked(context: StageContext) -> StageOutcome:
        calls.append(("preflight", context.attempt_directory.name))
        return StageOutcome(summary="Development gate blocked.", advancement_allowed=False)

    engine = _engine(tmp_path, store, config, _actions(calls, {"preflight": blocked}))
    result = engine.run()
    assert result.status == "blocked"
    assert result.stages[0].status is StageStatus.BLOCKED
    assert all(stage.status is StageStatus.PENDING for stage in result.stages[1:])
    assert calls == [("preflight", "attempt-0001")]
    assert (store.run_directory / "stages/00-preflight/completed.json").is_file()

    state_bytes = store.state_path.read_bytes()
    assert engine.run() == result
    assert store.state_path.read_bytes() == state_bytes
    assert calls == [("preflight", "attempt-0001")]


def test_orphan_attempt_is_preserved_and_next_attempt_never_overwrites(tmp_path: Path) -> None:
    config = _config()
    store = _store(tmp_path, config)
    stage_root = store.run_directory / "stages/00-preflight"
    orphan = stage_root / "attempt-0001"
    orphan.mkdir(parents=True)
    sentinel = orphan / "sentinel.txt"
    sentinel.write_text("preserve", encoding="ascii")
    calls: list[tuple[str, str]] = []

    result = _engine(tmp_path, store, config, _actions(calls)).run()

    assert result.status == "completed"
    assert result.stages[0].attempt_count == 2
    assert calls[0] == ("preflight", "attempt-0002")
    assert sentinel.read_text(encoding="ascii") == "preserve"


def test_lock_rejects_concurrent_engine_and_unsafe_lock_path(tmp_path: Path) -> None:
    config = _config()
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []
    engine = _engine(tmp_path, store, config, _actions(calls))

    with store.exclusive_lock(), pytest.raises(PipelineBusyError, match="another"):
        engine.run(dry_run=True)

    store.lock_path.unlink()
    store.lock_path.symlink_to(store.state_path)
    with pytest.raises(ValueError, match="lock path is unsafe"):
        engine.run(dry_run=True)


@pytest.mark.parametrize("target", ["outcome", "marker", "progress", "state"])
def test_checksum_canonical_and_progress_tampering_fail_closed(tmp_path: Path, target: str) -> None:
    config = _config()
    store = _store(tmp_path, config)
    _engine(tmp_path, store, config, _actions([])).run()
    if target == "outcome":
        path = store.run_directory / "stages/00-preflight/attempt-0001/outcome.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif target == "marker":
        path = store.run_directory / "stages/00-preflight/completed.json"
        payload = json.loads(path.read_bytes())
        payload["attempt"] = 2
        _rewrite_canonical(path, payload)
    elif target == "progress":
        path = store.run_directory / "status.json"
        payload = json.loads(path.read_bytes())
        payload["message"] = "tampered"
        _rewrite_canonical(path, payload)
    else:
        path = store.state_path
        payload = json.loads(path.read_bytes())
        payload["status"] = "ready"
        _rewrite_canonical(path, payload)

    with pytest.raises((ValueError, ValidationError)):
        _engine(tmp_path, store, config, _actions([])).run(dry_run=True)


def test_missing_marker_symlink_artifact_and_future_stage_output_are_rejected(
    tmp_path: Path,
) -> None:
    config = _config()
    store = _store(tmp_path, config)
    _engine(tmp_path, store, config, _actions([])).run()
    marker = store.run_directory / "stages/00-preflight/completed.json"
    marker.unlink()
    with pytest.raises(ValueError, match="missing its completion marker"):
        _engine(tmp_path, store, config, _actions([])).run(dry_run=True)

    other_root = tmp_path / "other"
    other_root.mkdir()
    second_config = _config(run_name="second-run")
    second = _store(tmp_path, second_config)
    future = second.run_directory / f"stages/01-{PIPELINE_STAGES[1]}"
    future.mkdir()
    with pytest.raises(ValueError, match="beyond the first incomplete"):
        _engine(tmp_path, second, second_config, _actions([])).run(dry_run=True)


def test_artifact_must_be_regular_and_inside_attempt(tmp_path: Path) -> None:
    config = _config()
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []

    def outside_artifact(context: StageContext) -> StageOutcome:
        calls.append(("preflight", context.attempt_directory.name))
        payload = b"outside"
        path = context.run_directory / "outside.txt"
        path.write_bytes(payload)
        reference = ArtifactReference(
            relative_path="outside.txt",
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )
        return StageOutcome(summary="invalid artifact.", artifacts=(reference,))

    with pytest.raises(PipelineStageError):
        _engine(
            tmp_path,
            store,
            config,
            _actions(calls, {"preflight": outside_artifact}),
        ).run()
    assert store.load_state().status == "failed"


def test_atomic_state_replace_failure_preserves_previous_canonical_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    store = _store(tmp_path, config)
    original = store.state_path.read_bytes()
    original_replace = os.replace

    def fail_state_replace(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == store.state_path:
            raise OSError("simulated replace failure")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_state_replace)
    with pytest.raises(PipelineError, match="atomic pipeline output"):
        store.write_state(store.load_state())
    assert store.state_path.read_bytes() == original
    assert list(store.run_directory.glob(".pipeline-state.json.tmp-*")) == []


def test_contracts_reject_unsafe_messages_inconsistent_state_and_bad_callbacks(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError):
        StageOutcome(summary="unsafe\nmessage")
    config = _config()
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []
    engine = _engine(
        tmp_path,
        store,
        config,
        _actions(calls),
        stop_requested=cast(Callable[[], bool], lambda: 1),
    )
    with pytest.raises(TypeError, match="exact boolean"):
        engine.run()


def test_serialized_contracts_reject_noncanonical_shapes_and_inconsistent_records(
    tmp_path: Path,
) -> None:
    for unsafe_path in (
        "",
        "/absolute",
        "../escape",
        "bad\\path",
        "caf\N{LATIN SMALL LETTER E WITH ACUTE}",
    ):
        with pytest.raises(ValidationError):
            ArtifactReference(relative_path=unsafe_path, sha256=HASH_A, size_bytes=1)

    artifact_a = ArtifactReference(relative_path="a.txt", sha256=HASH_A, size_bytes=1)
    artifact_b = ArtifactReference(relative_path="b.txt", sha256=HASH_B, size_bytes=1)
    metric_a = StageMetric(name="a", value=1.0, unit="nll")
    metric_b = StageMetric(name="b", value=2.0, unit="nll")
    invalid_outcomes = (
        {"summary": "x", "warnings": tuple("warning" for _ in range(257))},
        {"summary": "x", "artifacts": (artifact_a, artifact_a)},
        {"summary": "x", "artifacts": (artifact_b, artifact_a)},
        {"summary": "x", "metrics": (metric_a, metric_a)},
        {"summary": "x", "metrics": (metric_b, metric_a)},
        {"summary": "x", "warnings": ("unsafe\twarning",)},
    )
    for outcome_payload in invalid_outcomes:
        with pytest.raises(ValidationError):
            StageOutcome.model_validate(outcome_payload)

    started = "2026-08-23T12:00:00+00:00"
    completed = "2026-08-23T12:00:01+00:00"
    base: dict[str, object] = {
        "name": "preflight",
        "ordinal": 0,
        "status": StageStatus.RUNNING,
        "attempt_count": 1,
        "started_at": started,
        "latest_attempt_path": "stages/00-preflight/attempt-0001",
        "artifact_count": 0,
        "metric_count": 0,
    }
    invalid_records = (
        {**base, "name": "wrong"},
        {**base, "completed_at": "2026-08-23T11:59:59+00:00"},
        {
            **base,
            "status": StageStatus.PENDING,
            "attempt_count": 1,
        },
        {**base, "attempt_count": 0},
        {**base, "summary": "already terminal"},
        {**base, "status": StageStatus.STOPPED},
        {
            **base,
            "status": StageStatus.COMPLETED,
            "completed_at": completed,
            "summary": "done",
        },
        {
            **base,
            "status": StageStatus.FAILED,
            "completed_at": completed,
            "summary": "failed",
            "advancement_allowed": True,
        },
        {
            **base,
            "status": StageStatus.COMPLETED,
            "completed_at": completed,
            "summary": "done",
            "advancement_allowed": False,
        },
        {
            **base,
            "status": StageStatus.BLOCKED,
            "completed_at": completed,
            "summary": "blocked",
            "advancement_allowed": True,
        },
        {**base, "started_at": "not-a-time"},
        {**base, "started_at": "2026-08-23T12:00:00"},
        {**base, "started_at": "2026-08-23T12:00:00.100000+00:00"},
    )
    for record_payload in invalid_records:
        with pytest.raises(ValidationError):
            StageRecord.model_validate(record_payload)

    store = _store(tmp_path, _config())
    state = store.load_state()
    completed_record = StageRecord(
        name=PIPELINE_STAGES[1],
        ordinal=1,
        status=StageStatus.COMPLETED,
        attempt_count=1,
        started_at=started,
        completed_at=completed,
        summary="done",
        latest_attempt_path=f"stages/01-{PIPELINE_STAGES[1]}/attempt-0001",
        advancement_allowed=True,
        artifact_count=0,
        metric_count=0,
    )
    stopped_record = StageRecord(
        name=PIPELINE_STAGES[1],
        ordinal=1,
        status=StageStatus.STOPPED,
        attempt_count=1,
        started_at=started,
        completed_at=completed,
        summary="stopped",
        latest_attempt_path=f"stages/01-{PIPELINE_STAGES[1]}/attempt-0001",
        artifact_count=0,
        metric_count=0,
    )
    invalid_states = (
        state.model_copy(update={"stages": state.stages[:-1]}),
        state.model_copy(update={"updated_at": "2000-01-01T00:00:00+00:00"}),
        state.model_copy(update={"current_stage": "preflight"}),
        state.model_copy(
            update={
                "status": "running",
                "stages": (state.stages[0], completed_record, *state.stages[2:]),
            }
        ),
        state.model_copy(
            update={
                "status": "stopped",
                "stages": (state.stages[0], stopped_record, *state.stages[2:]),
            }
        ),
        state.model_copy(update={"status": "completed"}),
    )
    for candidate in invalid_states:
        with pytest.raises(ValidationError):
            orchestration._bound_state(candidate)

    state_payload = state.model_dump(mode="json", round_trip=True)
    state_payload["checksum_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="checksum mismatch"):
        PipelineState.model_validate_json(canonical_json_bytes(state_payload), strict=True)

    manifest = store.load_manifest()
    manifest_payload = manifest.model_dump(mode="json", round_trip=True)
    manifest_payload["command"] = []
    manifest_payload["checksum_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    with pytest.raises(ValidationError, match="command is invalid"):
        RunManifest.model_validate_json(canonical_json_bytes(manifest_payload), strict=True)
    manifest_payload = manifest.model_dump(mode="json", round_trip=True)
    manifest_payload["checksum_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="checksum mismatch"):
        RunManifest.model_validate_json(canonical_json_bytes(manifest_payload), strict=True)


def test_abrupt_process_loss_without_marker_is_retried_in_a_new_attempt(tmp_path: Path) -> None:
    class AbruptProcessLoss(BaseException):
        pass

    config = _config()
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []

    def abrupt(context: StageContext) -> StageOutcome:
        calls.append(("preflight", context.attempt_directory.name))
        raise AbruptProcessLoss

    with pytest.raises(AbruptProcessLoss):
        _engine(
            tmp_path,
            store,
            config,
            _actions(calls, {"preflight": abrupt}),
        ).run()
    crashed = store.load_state()
    assert crashed.current_stage == "preflight"
    assert crashed.stages[0].status is StageStatus.RUNNING

    resumed = _engine(tmp_path, store, config, _actions(calls)).run()
    assert resumed.status == "completed"
    assert resumed.interruption_count == 1
    assert resumed.stages[0].attempt_count == 2
    assert calls[:2] == [("preflight", "attempt-0001"), ("preflight", "attempt-0002")]


def test_blocked_marker_crash_recovers_terminal_gate_without_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []

    def blocked(context: StageContext) -> StageOutcome:
        calls.append(("preflight", context.attempt_directory.name))
        return StageOutcome(summary="Gate remained blocked.", advancement_allowed=False)

    actions = _actions(calls, {"preflight": blocked})
    original_write_state = store.write_state

    def crash_after_marker(candidate: PipelineState) -> PipelineState:
        marker = store.run_directory / "stages/00-preflight/completed.json"
        if marker.exists() and candidate.current_stage is None:
            raise KeyboardInterrupt
        return original_write_state(candidate)

    monkeypatch.setattr(store, "write_state", crash_after_marker)
    with pytest.raises(KeyboardInterrupt):
        _engine(tmp_path, store, config, actions).run()
    monkeypatch.setattr(store, "write_state", original_write_state)

    recovered = _engine(tmp_path, store, config, actions).run()
    assert recovered.status == "blocked"
    assert recovered.stages[0].status is StageStatus.BLOCKED
    assert recovered.interruption_count == 1
    assert calls == [("preflight", "attempt-0001")]
    assert _progress_events(store)[-1].state is ProgressState.STOPPED


def test_completed_state_recovers_progress_if_completion_event_was_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []
    original_complete = ProgressReporter.complete

    def crash_before_progress_completion(
        self: ProgressReporter, *, message: str = "pipeline completed"
    ) -> ProgressSnapshot:
        del self, message
        raise KeyboardInterrupt

    monkeypatch.setattr(ProgressReporter, "complete", crash_before_progress_completion)
    with pytest.raises(KeyboardInterrupt):
        _engine(tmp_path, store, config, _actions(calls)).run()
    assert store.load_state().status == "completed"
    assert _progress_events(store)[-1].state is ProgressState.FAILED

    monkeypatch.setattr(ProgressReporter, "complete", original_complete)
    recovered = _engine(tmp_path, store, config, _actions(calls)).run()
    assert recovered.status == "completed"
    assert _progress_events(store)[-1].state is ProgressState.COMPLETED
    assert len(calls) == len(PIPELINE_STAGES)


def test_committed_stage_replays_missing_progress_checkpoint_without_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []
    original_checkpoint = ProgressReporter.checkpoint

    def crash_before_checkpoint_event(
        self: ProgressReporter,
        *,
        checkpoint: str,
        message: str = "checkpoint saved",
        stage: str | None = None,
    ) -> ProgressSnapshot:
        del self, checkpoint, message, stage
        raise KeyboardInterrupt

    monkeypatch.setattr(ProgressReporter, "checkpoint", crash_before_checkpoint_event)
    with pytest.raises(KeyboardInterrupt):
        _engine(tmp_path, store, config, _actions(calls)).run()
    state = store.load_state()
    assert state.stages[0].status is StageStatus.COMPLETED
    assert state.stages[1].status is StageStatus.PENDING
    assert calls == [("preflight", "attempt-0001")]

    monkeypatch.setattr(ProgressReporter, "checkpoint", original_checkpoint)
    resumed = _engine(tmp_path, store, config, _actions(calls)).run()
    assert resumed.status == "completed"
    assert calls.count(("preflight", "attempt-0001")) == 1
    preflight_marker = "stages/00-preflight/completed.json"
    checkpoint_events = [
        event
        for event in _progress_events(store)
        if event.event_kind is ProgressEventKind.CHECKPOINT
        and event.latest_checkpoint == preflight_marker
    ]
    assert len(checkpoint_events) == 1


def test_invalid_callback_result_attempt_limit_and_progress_corruption_fail_closed(
    tmp_path: Path,
) -> None:
    config = _config()
    invalid_store = _store(tmp_path, config)

    def invalid_result(context: StageContext) -> StageOutcome:
        del context
        return cast(StageOutcome, object())

    with pytest.raises(PipelineStageError):
        _engine(
            tmp_path,
            invalid_store,
            config,
            _actions([], {"preflight": invalid_result}),
        ).run()

    limit_config = _config(run_name="limit-run")
    limit_store = _store(tmp_path, limit_config)
    stage_root = limit_store.run_directory / "stages/00-preflight"
    stage_root.mkdir()
    for attempt in range(1, 101):
        (stage_root / f"attempt-{attempt:04d}").mkdir()
    with pytest.raises(PipelineError, match="attempt limit"):
        _engine(tmp_path, limit_store, limit_config, _actions([])).run()

    progress_config = _config(run_name="progress-run")
    progress_store = _store(tmp_path, progress_config)
    _engine(tmp_path, progress_store, progress_config, _actions([])).run()
    (progress_store.run_directory / "status.json").unlink()
    with pytest.raises(ValueError, match="progress evidence is incomplete"):
        _engine(tmp_path, progress_store, progress_config, _actions([])).run(dry_run=True)


def test_store_engine_constructor_boundaries_and_command_capture(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="run_directory"):
        PipelineStore(cast(Path, "not-a-path"), maximum_state_bytes=1024)
    with pytest.raises(ValueError, match="maximum state"):
        PipelineStore(tmp_path, maximum_state_bytes=1)
    config = _config()
    with pytest.raises(TypeError, match="project_root"):
        PipelineStore.create(
            project_root=cast(Path, "not-a-path"),
            config=config,
            pipeline_config_sha256=config_sha256(config),
            source_commit=SOURCE_COMMIT,
            command=("python",),
        )
    store = _store(tmp_path, config)
    calls: list[tuple[str, str]] = []
    with pytest.raises(TypeError, match="project_root"):
        _engine(cast(Path, "not-a-path"), store, config, _actions(calls))
    bad_actions = _actions(calls)
    bad_actions["preflight"] = cast(StageAction, object())
    with pytest.raises(TypeError, match="callbacks"):
        _engine(tmp_path, store, config, bad_actions)
    captured = command_tuple()
    assert captured[0]
    assert isinstance(captured, tuple)


def test_symlinks_unknown_stage_entries_and_duplicate_json_are_rejected(tmp_path: Path) -> None:
    linked_config = _config(run_name="linked-run")
    linked_store = _store(tmp_path, linked_config)
    external = tmp_path / "external-stage"
    external.mkdir()
    (linked_store.run_directory / "stages/00-preflight").symlink_to(
        external, target_is_directory=True
    )
    with pytest.raises(ValueError, match="unexpected entry"):
        _engine(tmp_path, linked_store, linked_config, _actions([])).run(dry_run=True)

    marker_config = _config(run_name="marker-run")
    marker_store = _store(tmp_path, marker_config)
    _engine(tmp_path, marker_store, marker_config, _actions([])).run()
    marker = marker_store.run_directory / "stages/00-preflight/completed.json"
    marker_target = marker_store.run_directory / "run-manifest.json"
    marker.unlink()
    marker.symlink_to(marker_target)
    with pytest.raises(ValueError, match="completion marker is unsafe"):
        _engine(tmp_path, marker_store, marker_config, _actions([])).run(dry_run=True)

    json_config = _config(run_name="json-run")
    json_store = _store(tmp_path, json_config)
    json_store.state_path.write_bytes(b'{"state_version":"0.4.0","state_version":"0.4.0"}\n')
    with pytest.raises(ValueError, match="duplicate key"):
        _engine(tmp_path, json_store, json_config, _actions([])).run(dry_run=True)
