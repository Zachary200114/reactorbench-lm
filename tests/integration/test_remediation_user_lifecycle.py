"""Real process-boundary coverage for the user-operated remediation lifecycle."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from reactorbench.remediation import pipeline
from reactorbench.remediation.config import PIPELINE_STAGES, PipelineConfig
from reactorbench.remediation.orchestration import PipelineState, StageStatus
from reactorbench.remediation.progress import (
    ProgressEventKind,
    ProgressSnapshot,
    ProgressState,
)
from reactorbench.schemas.base import canonical_json_bytes

PROJECT_ROOT = Path(__file__).parents[2].resolve()
HELPER = PROJECT_ROOT / "tests/helpers/remediation_lifecycle_runner.py"
SOURCE_COMMIT = "abcdef0123456789"
RUN_NAME = "lifecycle-run"
SIMULATED_STAGE = "v02_inventory_and_caps"
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _pipeline_config() -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "pipeline_version": "0.4.0",
            "run_name": RUN_NAME,
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
            "maximum_run_bytes": 4 * 1024 * 1024,
            "maximum_process_rss_bytes": 512 * 1024**2,
            "stop_before_final_evaluation": True,
        }
    )


def _temporary_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src/reactorbench/remediation").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'lifecycle-test'\n",
        encoding="ascii",
    )
    (root / "frozen.toml").write_text("frozen = true\n", encoding="ascii")
    return root.resolve(strict=True)


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    source_root = str(PROJECT_ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_root if not existing else source_root + os.pathsep + existing
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _command(project_root: Path, action: str) -> tuple[str, ...]:
    return (sys.executable, str(HELPER), str(project_root), action)


def _run_command(
    project_root: Path,
    action: str,
    *,
    timeout: float = 8.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed interpreter and repository test helper
        _command(project_root, action),
        cwd=PROJECT_ROOT,
        env=_environment(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _wait_for_file(path: Path, process: subprocess.Popen[str], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and process.poll() is None:
        if path.is_file():
            return
        time.sleep(0.01)
    assert path.is_file(), "simulated stage did not publish its durable checkpoint"


def _assert_no_traceback(*streams: str) -> None:
    combined = "\n".join(streams)
    assert "Traceback (most recent call last)" not in combined
    assert 'tests/helpers/remediation_lifecycle_runner.py"' not in combined


def _load_state(path: Path) -> PipelineState:
    return PipelineState.model_validate_json(path.read_bytes(), strict=True)


def test_start_status_stop_and_resume_cross_real_process_boundaries(tmp_path: Path) -> None:
    root = _temporary_project(tmp_path)
    run_directory = root / "work" / RUN_NAME
    interrupted_attempt = run_directory / "stages/01-v02_inventory_and_caps/attempt-0001"
    durable_checkpoint = interrupted_attempt / "lifecycle-checkpoint.json"
    starter: subprocess.Popen[str] = subprocess.Popen(  # noqa: S603
        _command(root, "start"),
        cwd=PROJECT_ROOT,
        env=_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    starter_output = ""
    starter_errors = ""
    try:
        _wait_for_file(durable_checkpoint, starter)

        status = _run_command(root, "status")
        assert status.returncode == 0
        assert "Pipeline status: running" in status.stdout
        assert f"Next stage: {SIMULATED_STAGE}" in status.stdout

        status_payload = (run_directory / "status.json").read_bytes()
        snapshot = ProgressSnapshot.model_validate_json(status_payload, strict=True)
        assert snapshot.event_kind in {
            ProgressEventKind.PROGRESS,
            ProgressEventKind.HEARTBEAT,
        }
        assert snapshot.state is ProgressState.RUNNING
        assert snapshot.stage == SIMULATED_STAGE
        assert snapshot.completed_units == 2
        assert snapshot.total_units == 5
        assert snapshot.latest_checkpoint == (
            "stages/01-v02_inventory_and_caps/attempt-0001/lifecycle-checkpoint.json"
        )
        assert status_payload == (
            canonical_json_bytes(snapshot.model_dump(mode="json", round_trip=True)) + b"\n"
        )
        assert f"Latest progress: {snapshot.event_kind.value} at {SIMULATED_STAGE}" in status.stdout
        assert "Work completed: 2/5" in status.stdout

        stop = _run_command(root, "stop")
        assert stop.returncode == 0
        assert "Stop requested" in stop.stdout
        starter_output, starter_errors = starter.communicate(timeout=5.0)
    finally:
        if starter.poll() is None:
            starter.kill()
            starter_output, starter_errors = starter.communicate(timeout=2.0)

    assert starter.returncode == 8
    assert "Pipeline returned status: stopped" in starter_output
    assert starter_errors == ""
    _assert_no_traceback(
        status.stdout,
        status.stderr,
        stop.stdout,
        stop.stderr,
        starter_output,
        starter_errors,
    )

    checkpoint_payload = json.loads(durable_checkpoint.read_text(encoding="ascii"))
    assert checkpoint_payload == {
        "completed_units": 2,
        "stage": SIMULATED_STAGE,
        "total_units": 5,
    }
    stopped_state = _load_state(run_directory / "pipeline-state.json")
    assert stopped_state.status == "stopped"
    assert stopped_state.stages[0].status is StageStatus.COMPLETED
    assert stopped_state.stages[0].attempt_count == 1
    assert stopped_state.stages[1].status is StageStatus.STOPPED
    assert stopped_state.stages[1].attempt_count == 1
    first_stage_marker = run_directory / "stages/00-preflight/completed.json"
    first_stage_marker_before_resume = first_stage_marker.read_bytes()

    terminal_manifest = run_directory / "terminal-review/review-bundle.json"
    terminal_summary = run_directory / "terminal-review/review-summary.md"
    assert terminal_manifest.is_file()
    assert terminal_summary.is_file()
    assert json.loads(terminal_manifest.read_text(encoding="ascii"))["status"] == "stopped"
    assert "Terminal review manifest:" in starter_output
    assert "Terminal review summary:" in starter_output

    resume = _run_command(root, "resume", timeout=10.0)
    assert resume.returncode == 0
    assert "Previous stop request archived." in resume.stdout
    assert "Pipeline returned status: completed" in resume.stdout
    assert resume.stderr == ""
    _assert_no_traceback(resume.stdout, resume.stderr)

    completed_state = _load_state(run_directory / "pipeline-state.json")
    assert completed_state.status == "completed"
    assert len(completed_state.stages) == 16
    assert all(stage.status is StageStatus.COMPLETED for stage in completed_state.stages)
    assert completed_state.stages[0].attempt_count == 1
    assert completed_state.stages[1].attempt_count == 2
    assert first_stage_marker.read_bytes() == first_stage_marker_before_resume
    assert not (first_stage_marker.parent / "attempt-0002").exists()

    resumed_attempt = run_directory / "stages/01-v02_inventory_and_caps/attempt-0002"
    resume_proof = json.loads(
        (resumed_attempt / "resumed-from-checkpoint.json").read_text(encoding="ascii")
    )
    assert resume_proof == {
        "completed_units": 2,
        "source_checkpoint": (
            "stages/01-v02_inventory_and_caps/attempt-0001/lifecycle-checkpoint.json"
        ),
        "stage": SIMULATED_STAGE,
    }
    assert len(list(first_stage_marker.parent.glob("attempt-*/action-invocation.json"))) == 1
    assert len(list(resumed_attempt.parent.glob("attempt-*/action-invocation.json"))) == 2
    assert (run_directory / "stop-requests/acknowledged-stop-request").is_file()

    events = [
        ProgressSnapshot.model_validate_json(line, strict=True)
        for line in (run_directory / "progress.jsonl").read_bytes().splitlines()
    ]
    event_kinds = [event.event_kind for event in events]
    assert ProgressEventKind.STOPPED in event_kinds
    assert ProgressEventKind.RESUMED in event_kinds
    assert event_kinds[-1] is ProgressEventKind.COMPLETED
    assert any(
        event.event_kind is ProgressEventKind.CHECKPOINT
        and event.latest_checkpoint
        == "stages/01-v02_inventory_and_caps/attempt-0001/lifecycle-checkpoint.json"
        for event in events
    )


def test_actual_final_evaluation_guard_stays_locked_despite_plausible_markers(
    tmp_path: Path,
) -> None:
    project_root = tmp_path.resolve(strict=True)
    config = _pipeline_config()
    run_directory = project_root / config.run_root / config.run_name
    run_directory.mkdir(parents=True)
    for filename in (
        pipeline.FINAL_EVALUATION_READY_FILENAME,
        pipeline.OWNER_REVIEW_APPROVED_FILENAME,
        pipeline.FRESH_EXTENSION_MANIFEST_FILENAME,
    ):
        (run_directory / filename).write_text(
            '{"status":"approved","version":"0.4.0"}\n',
            encoding="ascii",
        )

    ledger = run_directory / pipeline.FINAL_ACCESS_LEDGER_FILENAME
    with pytest.raises(
        pipeline.FinalEvaluationBlockedError,
        match="distinct final-evidence executor",
    ):
        pipeline.run_final_evaluation(
            project_root=project_root,
            config=config,
            source_commit=SOURCE_COMMIT,
            explicit_confirmation=True,
        )

    assert not ledger.exists()
    assert not (run_directory / pipeline.FINAL_RESULT_FILENAME).exists()
    assert not (run_directory / pipeline.FINAL_REVIEW_FILENAME).exists()
