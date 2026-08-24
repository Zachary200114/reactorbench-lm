from __future__ import annotations

import ast
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from reactorbench.remediation.progress import (
    MAX_HEARTBEAT_INTERVAL_SECONDS,
    MAX_MESSAGE_CHARACTERS,
    MIN_HEARTBEAT_INTERVAL_SECONDS,
    PROGRESS_EVENT_LOG_FILENAME,
    PROGRESS_STATUS_FILENAME,
    ProgressError,
    ProgressEventKind,
    ProgressExistsError,
    ProgressIOError,
    ProgressMetric,
    ProgressReporter,
    ProgressSnapshot,
    ProgressState,
    ProgressStateError,
)
from reactorbench.schemas.base import canonical_json_bytes


@dataclass
class FakeClock:
    wall: datetime = datetime(2026, 8, 23, 12, 0, tzinfo=UTC)
    tick: float = 100.0

    def now_utc(self) -> datetime:
        return self.wall

    def monotonic(self) -> float:
        return self.tick

    def advance(self, seconds: float) -> None:
        self.tick += seconds
        self.wall += timedelta(seconds=seconds)


class RecordingConsole(StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def _snapshot(**overrides: object) -> ProgressSnapshot:
    values: dict[str, object] = {
        "sequence": 1,
        "event_kind": ProgressEventKind.PROGRESS,
        "state": ProgressState.RUNNING,
        "timestamp_utc": datetime(2026, 8, 23, tzinfo=UTC),
        "stage": "training v0.3",
        "stage_index": 2,
        "stage_total": 7,
        "completed_units": 10,
        "total_units": 20,
        "elapsed_seconds": 12.5,
        "message": "training is active",
        "latest_metric": ProgressMetric(name="loss", value=1.25),
        "latest_checkpoint": "checkpoints/step-10.safetensors",
        "eta_seconds": 30.0,
    }
    values.update(overrides)
    return ProgressSnapshot.model_validate(values, strict=True)


def _read_events(path: Path) -> tuple[ProgressSnapshot, ...]:
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    records: list[ProgressSnapshot] = []
    for line in raw.splitlines():
        record = ProgressSnapshot.model_validate_json(line, strict=True)
        assert line == canonical_json_bytes(record.model_dump(mode="json", round_trip=True))
        records.append(record)
    return tuple(records)


def test_progress_snapshot_is_strict_bounded_and_relationship_checked() -> None:
    assert _snapshot().latest_metric == ProgressMetric(name="loss", value=1.25)

    invalid: tuple[dict[str, object], ...] = (
        {"message": "unsafe\nmessage"},
        {"message": "x" * (MAX_MESSAGE_CHARACTERS + 1)},
        {"stage_index": 8, "stage_total": 7},
        {"completed_units": 21, "total_units": 20},
        {"stage_index": None, "stage_total": 7},
        {"completed_units": None, "total_units": 20},
        {"event_kind": ProgressEventKind.COMPLETED, "state": ProgressState.RUNNING},
        {"latest_checkpoint": "../escaped.safetensors"},
        {"latest_checkpoint": "/absolute/checkpoint.safetensors"},
        {"latest_checkpoint": "checkpoints\\windows.safetensors"},
        {"latest_checkpoint": "checkpoints/café.safetensors"},
        {"latest_checkpoint": "checkpoints/unsafe\nname.safetensors"},
        {"eta_seconds": float("inf")},
        {"timestamp_utc": datetime(2026, 8, 23, tzinfo=UTC).replace(tzinfo=None)},
        {
            "timestamp_utc": datetime(
                2026,
                8,
                23,
                tzinfo=timezone(timedelta(hours=-4)),
            )
        },
    )
    for overrides in invalid:
        with pytest.raises(ValidationError):
            _snapshot(**overrides)

    payload = _snapshot().model_dump(mode="python", round_trip=True)
    payload["unknown"] = "rejected"
    with pytest.raises(ValidationError):
        ProgressSnapshot.model_validate(payload, strict=True)


@pytest.mark.parametrize(
    "interval",
    [
        MIN_HEARTBEAT_INTERVAL_SECONDS - 0.1,
        MAX_HEARTBEAT_INTERVAL_SECONDS + 0.1,
        float("nan"),
    ],
)
def test_heartbeat_interval_is_strict_and_bounded(tmp_path: Path, interval: float) -> None:
    with pytest.raises(ValueError, match="between 5 and 60"):
        ProgressReporter(tmp_path, heartbeat_interval_seconds=interval)
    assert ProgressReporter(tmp_path, heartbeat_interval_seconds=5).status_path == (
        tmp_path / PROGRESS_STATUS_FILENAME
    )


def test_reporter_constructor_rejects_unsafe_paths_flags_and_stages(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match=r"pathlib\.Path"):
        ProgressReporter(cast(Path, "not-a-path"))

    missing = tmp_path / "missing"
    with pytest.raises(ProgressError, match="existing non-symlink"):
        ProgressReporter(missing)

    regular_file = tmp_path / "regular-file"
    regular_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ProgressError, match="existing non-symlink"):
        ProgressReporter(regular_file)

    linked_directory = tmp_path / "linked-directory"
    linked_directory.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ProgressError, match="existing non-symlink"):
        ProgressReporter(linked_directory)

    with pytest.raises(TypeError, match="must be a number"):
        ProgressReporter(tmp_path, heartbeat_interval_seconds=True)
    with pytest.raises(TypeError, match="exact booleans"):
        ProgressReporter(tmp_path, background_heartbeat=cast(bool, 1))
    for invalid_stage in ("", "/absolute", "bad/stage", "bad\nstage"):
        with pytest.raises(ValueError, match="stage"):
            ProgressReporter(tmp_path, initial_stage=invalid_stage)
    with pytest.raises(TypeError, match="stage must be a string"):
        ProgressReporter(tmp_path, initial_stage=cast(str, 3))


def test_reporter_writes_flushed_console_canonical_log_and_atomic_status(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    console = RecordingConsole()
    reporter = ProgressReporter(
        tmp_path,
        initial_stage="training v0.3",
        heartbeat_interval_seconds=5.0,
        clock=clock,
        console=console,
        background_heartbeat=False,
    )

    with reporter:
        clock.advance(2.0)
        progress = reporter.report(
            message="epoch update",
            stage_index=2,
            stage_total=7,
            completed_units=25,
            total_units=100,
            latest_metric=ProgressMetric(name="loss", value=1.5),
            latest_checkpoint="checkpoints/step-25.safetensors",
            eta_seconds=60.0,
        )
        assert progress.elapsed_seconds == 2.0
        clock.advance(1.0)
        checkpoint = reporter.checkpoint(checkpoint="checkpoints/step-30.safetensors")
        assert checkpoint.latest_metric == ProgressMetric(name="loss", value=1.5)
        assert checkpoint.latest_checkpoint == "checkpoints/step-30.safetensors"
        assert checkpoint.eta_seconds == 60.0
        final = reporter.complete()

    assert final.state is ProgressState.COMPLETED
    assert reporter.snapshot() == final
    events = _read_events(tmp_path / PROGRESS_EVENT_LOG_FILENAME)
    assert tuple(event.event_kind for event in events) == (
        ProgressEventKind.STARTED,
        ProgressEventKind.PROGRESS,
        ProgressEventKind.CHECKPOINT,
        ProgressEventKind.COMPLETED,
    )
    assert (tmp_path / PROGRESS_STATUS_FILENAME).read_bytes() == (
        canonical_json_bytes(final.model_dump(mode="json", round_trip=True)) + b"\n"
    )
    assert not tuple(tmp_path.glob(".status.json.*.tmp"))
    assert console.flush_count == len(events)
    console_lines = console.getvalue().splitlines()
    assert len(console_lines) == len(events)
    assert all(line.startswith("[2026-08-23T12:00:") for line in console_lines)
    assert "work=25/100" in console_lines[1]
    assert "metric=loss:1.5" in console_lines[1]
    assert "checkpoint=checkpoints/step-25.safetensors" in console_lines[1]
    assert "eta_seconds=60.0" in console_lines[1]


def test_fake_clock_heartbeat_is_due_at_interval_and_preserves_snapshot(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    with ProgressReporter(
        tmp_path,
        heartbeat_interval_seconds=5.0,
        clock=clock,
        console=RecordingConsole(),
        background_heartbeat=False,
    ) as reporter:
        reporter.report(
            message="working",
            stage="calibration v0.4",
            completed_units=3,
            total_units=10,
            latest_metric=ProgressMetric(name="ece", value=0.2),
            latest_checkpoint="checkpoints/calibration.safetensors",
            eta_seconds=15.0,
        )
        clock.advance(4.999)
        assert reporter.heartbeat_if_due() is None
        clock.advance(0.001)
        heartbeat = reporter.heartbeat_if_due()
        assert heartbeat is not None
        assert heartbeat.event_kind is ProgressEventKind.HEARTBEAT
        assert heartbeat.stage == "calibration v0.4"
        assert heartbeat.completed_units == 3
        assert heartbeat.total_units == 10
        assert heartbeat.latest_metric == ProgressMetric(name="ece", value=0.2)
        assert heartbeat.latest_checkpoint == "checkpoints/calibration.safetensors"
        assert heartbeat.eta_seconds == 15.0
        assert reporter.heartbeat_if_due() is None
        reporter.stop()


def test_normal_context_exit_records_stopped_and_public_paths(tmp_path: Path) -> None:
    reporter = ProgressReporter(
        tmp_path,
        heartbeat_interval_seconds=5.0,
        console=RecordingConsole(),
        background_heartbeat=False,
    )
    assert reporter.event_log_path == tmp_path / PROGRESS_EVENT_LOG_FILENAME
    assert reporter.status_path == tmp_path / PROGRESS_STATUS_FILENAME
    with reporter:
        reporter.report(message="still running")
    final = reporter.snapshot()
    assert final.event_kind is ProgressEventKind.STOPPED
    assert final.message == "progress reporting stopped"


def test_direct_failure_uses_curated_bounded_message(tmp_path: Path) -> None:
    with ProgressReporter(
        tmp_path,
        heartbeat_interval_seconds=5.0,
        console=RecordingConsole(),
        background_heartbeat=False,
    ) as reporter:
        final = reporter.fail(message="validation gate failed")
    assert final.event_kind is ProgressEventKind.FAILED
    assert final.message == "validation gate failed"


def test_background_heartbeat_runs_and_context_shutdown_joins_thread(tmp_path: Path) -> None:
    console = RecordingConsole()
    reporter = ProgressReporter(
        tmp_path,
        heartbeat_interval_seconds=5.0,
        console=console,
        background_heartbeat=True,
    )
    # Exercise the validated production loop without making the unit test wait five seconds.
    reporter._heartbeat_interval_seconds = 0.02
    with reporter:
        deadline = time.monotonic() + 1.0
        while reporter.snapshot().event_kind is not ProgressEventKind.HEARTBEAT:
            if time.monotonic() >= deadline:
                pytest.fail("background heartbeat was not emitted")
            time.sleep(0.005)
        thread = reporter._thread
        assert thread is not None
        assert thread.is_alive()
        reporter.complete()
    assert thread is not None
    assert not thread.is_alive()


def test_context_exception_records_safe_failure_without_leaking_exception_text(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="sensitive detail"):
        with ProgressReporter(
            tmp_path,
            heartbeat_interval_seconds=5.0,
            console=RecordingConsole(),
            background_heartbeat=False,
        ):
            raise RuntimeError("sensitive detail /private/internal/path")

    final = _read_events(tmp_path / PROGRESS_EVENT_LOG_FILENAME)[-1]
    assert final.event_kind is ProgressEventKind.FAILED
    assert final.message == "pipeline context exited with an error"
    assert "sensitive" not in (tmp_path / PROGRESS_EVENT_LOG_FILENAME).read_text(encoding="utf-8")


def test_resume_is_explicit_append_only_and_rejects_completed_output(tmp_path: Path) -> None:
    first_console = RecordingConsole()
    with ProgressReporter(
        tmp_path,
        heartbeat_interval_seconds=5.0,
        console=first_console,
        background_heartbeat=False,
    ) as first:
        first.report(message="first process")
        first.stop()
    prefix = (tmp_path / PROGRESS_EVENT_LOG_FILENAME).read_bytes()

    with pytest.raises(ProgressExistsError, match="explicit resume"):
        with ProgressReporter(
            tmp_path,
            heartbeat_interval_seconds=5.0,
            console=RecordingConsole(),
            background_heartbeat=False,
        ):
            pass

    with ProgressReporter(
        tmp_path,
        heartbeat_interval_seconds=5.0,
        console=RecordingConsole(),
        background_heartbeat=False,
        resume=True,
    ) as resumed:
        assert resumed.snapshot().event_kind is ProgressEventKind.RESUMED
        assert resumed.snapshot().sequence == 4
        resumed.complete()

    combined = (tmp_path / PROGRESS_EVENT_LOG_FILENAME).read_bytes()
    assert combined.startswith(prefix)
    events = _read_events(tmp_path / PROGRESS_EVENT_LOG_FILENAME)
    assert tuple(event.sequence for event in events) == tuple(range(1, len(events) + 1))
    assert events[-1].state is ProgressState.COMPLETED

    with pytest.raises(ProgressStateError, match="cannot be resumed"):
        with ProgressReporter(
            tmp_path,
            heartbeat_interval_seconds=5.0,
            console=RecordingConsole(),
            background_heartbeat=False,
            resume=True,
        ):
            pass


def test_resume_rejects_mismatched_status_and_event_log(tmp_path: Path) -> None:
    with ProgressReporter(
        tmp_path,
        heartbeat_interval_seconds=5.0,
        console=RecordingConsole(),
        background_heartbeat=False,
    ) as reporter:
        reporter.stop()
    status_path = tmp_path / PROGRESS_STATUS_FILENAME
    status = ProgressSnapshot.model_validate_json(status_path.read_bytes(), strict=True)
    changed = status.model_copy(update={"message": "tampered status"})
    status_path.write_bytes(canonical_json_bytes(changed.model_dump(mode="json")) + b"\n")

    with pytest.raises(ProgressError, match="does not match"):
        with ProgressReporter(
            tmp_path,
            heartbeat_interval_seconds=5.0,
            console=RecordingConsole(),
            background_heartbeat=False,
            resume=True,
        ):
            pass


def test_resume_rejects_incomplete_symlink_nonregular_and_malformed_outputs(
    tmp_path: Path,
) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / PROGRESS_EVENT_LOG_FILENAME).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProgressError, match="requires both"):
        with ProgressReporter(incomplete, resume=True, background_heartbeat=False):
            pass

    symlinked = tmp_path / "symlinked"
    symlinked.mkdir()
    target = symlinked / "target"
    target.write_text("{}\n", encoding="utf-8")
    (symlinked / PROGRESS_EVENT_LOG_FILENAME).symlink_to(target)
    (symlinked / PROGRESS_STATUS_FILENAME).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProgressError, match="symbolic links"):
        with ProgressReporter(symlinked, resume=True, background_heartbeat=False):
            pass

    nonregular = tmp_path / "nonregular"
    nonregular.mkdir()
    (nonregular / PROGRESS_EVENT_LOG_FILENAME).mkdir()
    (nonregular / PROGRESS_STATUS_FILENAME).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProgressError, match="regular files"):
        with ProgressReporter(nonregular, resume=True, background_heartbeat=False):
            pass

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / PROGRESS_EVENT_LOG_FILENAME).write_text("{}\n", encoding="utf-8")
    (malformed / PROGRESS_STATUS_FILENAME).write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ProgressError, match="strict contract"):
        with ProgressReporter(malformed, resume=True, background_heartbeat=False):
            pass


def test_resume_rejects_noncanonical_status_and_unterminated_event(tmp_path: Path) -> None:
    snapshot = _snapshot(
        event_kind=ProgressEventKind.STOPPED,
        state=ProgressState.STOPPED,
        message="stopped",
    )
    canonical = canonical_json_bytes(snapshot.model_dump(mode="json", round_trip=True)) + b"\n"

    noncanonical = tmp_path / "noncanonical"
    noncanonical.mkdir()
    (noncanonical / PROGRESS_EVENT_LOG_FILENAME).write_bytes(canonical)
    noncanonical_status = (
        json.dumps(
            snapshot.model_dump(mode="json", round_trip=True),
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    (noncanonical / PROGRESS_STATUS_FILENAME).write_bytes(noncanonical_status)
    with pytest.raises(ProgressError, match="not canonical"):
        with ProgressReporter(noncanonical, resume=True, background_heartbeat=False):
            pass

    unterminated = tmp_path / "unterminated"
    unterminated.mkdir()
    (unterminated / PROGRESS_EVENT_LOG_FILENAME).write_bytes(canonical[:-1])
    (unterminated / PROGRESS_STATUS_FILENAME).write_bytes(canonical)
    with pytest.raises(ProgressError, match="final newline"):
        with ProgressReporter(unterminated, resume=True, background_heartbeat=False):
            pass


def test_terminal_state_rejects_more_events_and_snapshot_survives_close(tmp_path: Path) -> None:
    reporter = ProgressReporter(
        tmp_path,
        heartbeat_interval_seconds=5.0,
        console=RecordingConsole(),
        background_heartbeat=False,
    )
    with pytest.raises(ProgressStateError, match="has not started"):
        reporter.snapshot()
    with reporter:
        final = reporter.complete()
        with pytest.raises(ProgressStateError, match="not running"):
            reporter.report(message="must be rejected")
        with pytest.raises(ProgressStateError, match="not running"):
            reporter.heartbeat_if_due()
    assert reporter.snapshot() == final
    with pytest.raises(ProgressStateError, match="more than once"):
        with reporter:
            pass


def test_progress_module_has_no_training_or_global_rng_imports() -> None:
    source_path = Path(__file__).parents[2] / "src/reactorbench/remediation/progress.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.partition(".")[0])
    assert imported_roots.isdisjoint({"numpy", "torch", "random"})


def test_atomic_status_replacement_fsyncs_files_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fsync = os.fsync
    fsynced: list[int] = []

    def recording_fsync(descriptor: int) -> None:
        fsynced.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr("reactorbench.remediation.progress.os.fsync", recording_fsync)
    with ProgressReporter(
        tmp_path,
        heartbeat_interval_seconds=5.0,
        console=RecordingConsole(),
        background_heartbeat=False,
    ) as reporter:
        reporter.complete()
    # Each event fsyncs its append descriptor, status temporary file, and directory.
    assert len(fsynced) >= 6
    assert not tuple(tmp_path.glob(".status.json.*.tmp"))


def test_progress_io_failures_use_safe_messages_and_clean_temporary_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    append_failure = tmp_path / "append-failure"
    append_failure.mkdir()
    monkeypatch.setattr("reactorbench.remediation.progress.os.write", lambda *_args: 0)
    with pytest.raises(ProgressIOError, match="append the progress event safely"):
        with ProgressReporter(append_failure, background_heartbeat=False):
            pass
    monkeypatch.undo()

    status_failure = tmp_path / "status-failure"
    status_failure.mkdir()

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("internal path must remain hidden")

    monkeypatch.setattr("reactorbench.remediation.progress.os.replace", fail_replace)
    with pytest.raises(ProgressIOError, match="replace progress status safely") as error:
        with ProgressReporter(status_failure, background_heartbeat=False):
            pass
    assert "internal path" not in str(error.value)
    assert not tuple(status_failure.glob(".status.json.*.tmp"))


def test_console_and_clock_failures_are_safe_and_monotonic(tmp_path: Path) -> None:
    class BrokenConsole(RecordingConsole):
        def flush(self) -> None:
            raise ValueError("private console detail")

    console_failure = tmp_path / "console-failure"
    console_failure.mkdir()
    with pytest.raises(ProgressIOError, match="console event") as console_error:
        with ProgressReporter(
            console_failure,
            console=BrokenConsole(),
            background_heartbeat=False,
        ):
            pass
    assert "private console" not in str(console_error.value)

    backward = tmp_path / "backward"
    backward.mkdir()
    clock = FakeClock()
    with ProgressReporter(
        backward,
        heartbeat_interval_seconds=5.0,
        clock=clock,
        console=RecordingConsole(),
        background_heartbeat=False,
    ) as reporter:
        clock.advance(2.0)
        reporter.report(message="advanced")
        clock.tick -= 1.0
        with pytest.raises(ProgressError, match="moved backwards"):
            reporter.report(message="must fail")
        clock.tick += 1.0
