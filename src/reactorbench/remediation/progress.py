"""Durable, bounded progress reporting for user-operated remediation runs.

The reporter intentionally depends only on the Python standard library and the
project's strict schema primitives.  Every successful event is appended as one
canonical JSONL record, reflected through an atomically replaced ``status.json``,
and printed as one immediately flushed, human-readable console line.
"""

from __future__ import annotations

import math
import os
import stat
import sys
import tempfile
import threading
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Protocol, TextIO

from pydantic import Field, StrictFloat, StrictInt, StrictStr, model_validator

from reactorbench.schemas.base import ContractModel, canonical_json_bytes

PROGRESS_CONTRACT_VERSION: Literal["0.1.0"] = "0.1.0"
PROGRESS_EVENT_LOG_FILENAME = "progress.jsonl"
PROGRESS_STATUS_FILENAME = "status.json"

MIN_HEARTBEAT_INTERVAL_SECONDS = 5.0
MAX_HEARTBEAT_INTERVAL_SECONDS = 60.0
MAX_MESSAGE_CHARACTERS = 320
MAX_STAGE_CHARACTERS = 80
MAX_CHECKPOINT_CHARACTERS = 240
MAX_STATUS_BYTES = 16 * 1024
MAX_DURATION_SECONDS = 366 * 24 * 60 * 60

SafeMessage = Annotated[
    StrictStr,
    Field(
        min_length=1,
        max_length=MAX_MESSAGE_CHARACTERS,
        pattern=r"^[ -~]+$",
    ),
]
StageName = Annotated[
    StrictStr,
    Field(
        min_length=1,
        max_length=MAX_STAGE_CHARACTERS,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._ -]*$",
    ),
]
MetricName = Annotated[
    StrictStr,
    Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z][A-Za-z0-9._-]*$",
    ),
]
NonNegativeDuration = Annotated[
    StrictFloat,
    Field(ge=0.0, le=MAX_DURATION_SECONDS, allow_inf_nan=False),
]


class ProgressError(RuntimeError):
    """Base class for progress-reporting failures with safe public messages."""


class ProgressExistsError(ProgressError):
    """Raised instead of reusing progress evidence without explicit resume intent."""


class ProgressStateError(ProgressError):
    """Raised when an operation is incompatible with the reporter lifecycle."""


class ProgressIOError(ProgressError):
    """Raised when durable progress output cannot be written safely."""


class ProgressEventKind(StrEnum):
    STARTED = "started"
    RESUMED = "resumed"
    PROGRESS = "progress"
    HEARTBEAT = "heartbeat"
    CHECKPOINT = "checkpoint"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class ProgressState(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


_EVENT_STATES: dict[ProgressEventKind, ProgressState] = {
    ProgressEventKind.STARTED: ProgressState.RUNNING,
    ProgressEventKind.RESUMED: ProgressState.RUNNING,
    ProgressEventKind.PROGRESS: ProgressState.RUNNING,
    ProgressEventKind.HEARTBEAT: ProgressState.RUNNING,
    ProgressEventKind.CHECKPOINT: ProgressState.RUNNING,
    ProgressEventKind.COMPLETED: ProgressState.COMPLETED,
    ProgressEventKind.FAILED: ProgressState.FAILED,
    ProgressEventKind.STOPPED: ProgressState.STOPPED,
}


class ProgressMetric(ContractModel):
    """One latest scalar metric suitable for status displays and review bundles."""

    name: MetricName
    value: StrictFloat = Field(allow_inf_nan=False)


class ProgressSnapshot(ContractModel):
    """The complete bounded state written for every progress event."""

    contract_version: Literal["0.1.0"] = PROGRESS_CONTRACT_VERSION
    sequence: StrictInt = Field(ge=1)
    event_kind: ProgressEventKind
    state: ProgressState
    timestamp_utc: datetime
    stage: StageName
    stage_index: StrictInt | None = Field(default=None, ge=1, le=10_000)
    stage_total: StrictInt | None = Field(default=None, ge=1, le=10_000)
    completed_units: StrictInt | None = Field(default=None, ge=0, le=2_147_483_647)
    total_units: StrictInt | None = Field(default=None, ge=1, le=2_147_483_647)
    elapsed_seconds: NonNegativeDuration
    message: SafeMessage
    latest_metric: ProgressMetric | None = None
    latest_checkpoint: StrictStr | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_CHECKPOINT_CHARACTERS,
    )
    eta_seconds: NonNegativeDuration | None = None

    @model_validator(mode="after")
    def fields_are_consistent_and_safe(self) -> ProgressSnapshot:
        utc_offset = self.timestamp_utc.utcoffset()
        if self.timestamp_utc.tzinfo is None or utc_offset is None:
            raise ValueError("timestamp_utc must be timezone-aware")
        if utc_offset.total_seconds() != 0:
            raise ValueError("timestamp_utc must use UTC")
        if (self.stage_index is None) != (self.stage_total is None):
            raise ValueError("stage index and total must be provided together")
        if self.stage_index is not None and self.stage_total is not None:
            if self.stage_index > self.stage_total:
                raise ValueError("stage index must not exceed stage total")
        if (self.completed_units is None) != (self.total_units is None):
            raise ValueError("completed and total units must be provided together")
        if self.completed_units is not None and self.total_units is not None:
            if self.completed_units > self.total_units:
                raise ValueError("completed units must not exceed total units")
        if self.state is not _EVENT_STATES[self.event_kind]:
            raise ValueError("event kind does not match progress state")
        if self.latest_checkpoint is not None:
            _validate_checkpoint_reference(self.latest_checkpoint)
        return self


class ProgressClock(Protocol):
    """Injectable wall and monotonic time source used by deterministic tests."""

    def now_utc(self) -> datetime:
        """Return the current timezone-aware UTC wall time."""

    def monotonic(self) -> float:
        """Return a nondecreasing monotonic timestamp in seconds."""


class SystemProgressClock:
    """Production clock with no dependency on process-global random state."""

    def now_utc(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        import time

        return time.monotonic()


def _validate_checkpoint_reference(value: str) -> None:
    if not value or "\\" in value or "\x00" in value or not value.isascii():
        raise ValueError("latest_checkpoint must be a safe POSIX relative path")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or candidate == PurePosixPath(".")
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("latest_checkpoint must be a safe POSIX relative path")
    if any(character < " " or character > "~" for character in value):
        raise ValueError("latest_checkpoint must contain printable ASCII only")


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("progress write made no forward progress")
        offset += written


class ProgressReporter:
    """Write durable progress evidence and guarantee periodic heartbeat attempts.

    Use the reporter as a context manager.  The background heartbeat is enabled by
    default and emits whenever no other event has completed during the configured
    5--60 second interval.  ``heartbeat_if_due`` exposes the same decision for
    deterministic fake-clock tests and synchronous integrations.
    """

    def __init__(
        self,
        output_directory: Path,
        *,
        initial_stage: str = "initializing",
        heartbeat_interval_seconds: float = 30.0,
        clock: ProgressClock | None = None,
        console: TextIO | None = None,
        background_heartbeat: bool = True,
        resume: bool = False,
    ) -> None:
        if not isinstance(output_directory, Path):
            raise TypeError("output_directory must be a pathlib.Path")
        if output_directory.is_symlink() or not output_directory.is_dir():
            raise ProgressError("progress output must be an existing non-symlink directory")
        if isinstance(heartbeat_interval_seconds, bool) or not isinstance(
            heartbeat_interval_seconds, (int, float)
        ):
            raise TypeError("heartbeat_interval_seconds must be a number")
        heartbeat_interval = float(heartbeat_interval_seconds)
        if not math.isfinite(heartbeat_interval) or not (
            MIN_HEARTBEAT_INTERVAL_SECONDS <= heartbeat_interval <= MAX_HEARTBEAT_INTERVAL_SECONDS
        ):
            raise ValueError("heartbeat interval must be between 5 and 60 seconds")
        if type(background_heartbeat) is not bool or type(resume) is not bool:
            raise TypeError("background_heartbeat and resume must be exact booleans")

        # Validate user-facing text through the same strict contract used for output.
        self._initial_stage = _validate_stage(initial_stage)
        self._output_directory = output_directory.resolve(strict=True)
        self._event_log_path = self._output_directory / PROGRESS_EVENT_LOG_FILENAME
        self._status_path = self._output_directory / PROGRESS_STATUS_FILENAME
        self._heartbeat_interval_seconds = heartbeat_interval
        self._clock = SystemProgressClock() if clock is None else clock
        self._console = sys.stdout if console is None else console
        self._background_heartbeat = background_heartbeat
        self._resume = resume

        self._condition = threading.Condition(threading.RLock())
        self._thread: threading.Thread | None = None
        self._active = False
        self._closed = False
        self._sequence = 0
        self._snapshot: ProgressSnapshot | None = None
        self._started_monotonic = 0.0
        self._elapsed_offset = 0.0
        self._last_emit_monotonic = 0.0
        self._background_error: BaseException | None = None

    @property
    def event_log_path(self) -> Path:
        return self._event_log_path

    @property
    def status_path(self) -> Path:
        return self._status_path

    def __enter__(self) -> ProgressReporter:
        with self._condition:
            if self._closed or self._active:
                raise ProgressStateError("progress reporter cannot be entered more than once")
            previous = self._preflight_outputs()
            now = self._monotonic_now()
            self._started_monotonic = now
            self._last_emit_monotonic = now
            self._active = True
            if previous is None:
                stage = self._initial_stage
                kind = ProgressEventKind.STARTED
                message = "progress reporting started"
            else:
                self._sequence = previous.sequence
                self._snapshot = previous
                self._elapsed_offset = previous.elapsed_seconds
                stage = previous.stage
                kind = ProgressEventKind.RESUMED
                message = "progress reporting resumed"
            try:
                self._emit_locked(
                    kind=kind,
                    stage=stage,
                    message=message,
                    preserve_details=previous is not None,
                )
            except BaseException:
                self._active = False
                self._closed = True
                raise
            if self._background_heartbeat:
                self._thread = threading.Thread(
                    target=self._heartbeat_loop,
                    name="reactorbench-progress-heartbeat",
                    daemon=True,
                )
                self._thread.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> Literal[False]:
        del exception, traceback
        reporting_error: BaseException | None = None
        with self._condition:
            if self._active and self._current_state() is ProgressState.RUNNING:
                kind = (
                    ProgressEventKind.FAILED
                    if exception_type is not None
                    else ProgressEventKind.STOPPED
                )
                message = (
                    "pipeline context exited with an error"
                    if exception_type is not None
                    else "progress reporting stopped"
                )
                try:
                    self._emit_locked(
                        kind=kind,
                        stage=self._current_stage(),
                        message=message,
                        preserve_details=True,
                    )
                except BaseException as error:  # Preserve an original pipeline exception.
                    reporting_error = error
            self._active = False
            self._closed = True
            self._condition.notify_all()
        self._join_heartbeat()
        if reporting_error is not None and exception_type is None:
            raise reporting_error
        return False

    def snapshot(self) -> ProgressSnapshot:
        """Return the latest immutable snapshot without touching output files."""

        with self._condition:
            if self._snapshot is None:
                raise ProgressStateError("progress reporter has not started")
            return self._snapshot

    def report(
        self,
        *,
        message: str,
        stage: str | None = None,
        stage_index: int | None = None,
        stage_total: int | None = None,
        completed_units: int | None = None,
        total_units: int | None = None,
        latest_metric: ProgressMetric | None = None,
        latest_checkpoint: str | None = None,
        eta_seconds: float | None = None,
    ) -> ProgressSnapshot:
        """Record one explicit running-state progress update."""

        return self._emit(
            kind=ProgressEventKind.PROGRESS,
            stage=self._current_stage() if stage is None else stage,
            message=message,
            stage_index=stage_index,
            stage_total=stage_total,
            completed_units=completed_units,
            total_units=total_units,
            latest_metric=latest_metric,
            latest_checkpoint=latest_checkpoint,
            eta_seconds=eta_seconds,
        )

    def checkpoint(
        self,
        *,
        checkpoint: str,
        message: str = "checkpoint saved",
        stage: str | None = None,
    ) -> ProgressSnapshot:
        """Record one checkpoint while retaining the latest metric and ETA."""

        return self._emit(
            kind=ProgressEventKind.CHECKPOINT,
            stage=self._current_stage() if stage is None else stage,
            message=message,
            latest_checkpoint=checkpoint,
            preserve_metric_and_eta=True,
        )

    def complete(self, *, message: str = "pipeline completed") -> ProgressSnapshot:
        """Record successful completion and prevent further progress events."""

        return self._emit(
            kind=ProgressEventKind.COMPLETED,
            stage=self._current_stage(),
            message=message,
            preserve_details=True,
        )

    def fail(self, *, message: str = "pipeline failed safely") -> ProgressSnapshot:
        """Record a failure using bounded, curated public text."""

        return self._emit(
            kind=ProgressEventKind.FAILED,
            stage=self._current_stage(),
            message=message,
            preserve_details=True,
        )

    def stop(self, *, message: str = "pipeline stopped safely") -> ProgressSnapshot:
        """Record an intentional resumable stop."""

        return self._emit(
            kind=ProgressEventKind.STOPPED,
            stage=self._current_stage(),
            message=message,
            preserve_details=True,
        )

    def heartbeat_if_due(self) -> ProgressSnapshot | None:
        """Emit one heartbeat if the interval has elapsed; otherwise return ``None``."""

        with self._condition:
            self._require_running_locked()
            now = self._monotonic_now()
            if now - self._last_emit_monotonic < self._heartbeat_interval_seconds:
                return None
            return self._emit_locked(
                kind=ProgressEventKind.HEARTBEAT,
                stage=self._current_stage(),
                message="pipeline remains active",
                preserve_details=True,
                monotonic_now=now,
            )

    def _emit(
        self,
        *,
        kind: ProgressEventKind,
        stage: str,
        message: str,
        stage_index: int | None = None,
        stage_total: int | None = None,
        completed_units: int | None = None,
        total_units: int | None = None,
        latest_metric: ProgressMetric | None = None,
        latest_checkpoint: str | None = None,
        eta_seconds: float | None = None,
        preserve_details: bool = False,
        preserve_metric_and_eta: bool = False,
    ) -> ProgressSnapshot:
        with self._condition:
            self._require_running_locked()
            snapshot = self._emit_locked(
                kind=kind,
                stage=stage,
                message=message,
                stage_index=stage_index,
                stage_total=stage_total,
                completed_units=completed_units,
                total_units=total_units,
                latest_metric=latest_metric,
                latest_checkpoint=latest_checkpoint,
                eta_seconds=eta_seconds,
                preserve_details=preserve_details,
                preserve_metric_and_eta=preserve_metric_and_eta,
            )
            if snapshot.state is not ProgressState.RUNNING:
                self._active = False
                self._condition.notify_all()
            return snapshot

    def _emit_locked(
        self,
        *,
        kind: ProgressEventKind,
        stage: str,
        message: str,
        stage_index: int | None = None,
        stage_total: int | None = None,
        completed_units: int | None = None,
        total_units: int | None = None,
        latest_metric: ProgressMetric | None = None,
        latest_checkpoint: str | None = None,
        eta_seconds: float | None = None,
        preserve_details: bool = False,
        preserve_metric_and_eta: bool = False,
        monotonic_now: float | None = None,
    ) -> ProgressSnapshot:
        current = self._snapshot
        if preserve_details and current is not None:
            stage_index = current.stage_index
            stage_total = current.stage_total
            completed_units = current.completed_units
            total_units = current.total_units
            latest_metric = current.latest_metric
            latest_checkpoint = current.latest_checkpoint
            eta_seconds = current.eta_seconds
        elif preserve_metric_and_eta and current is not None:
            stage_index = current.stage_index
            stage_total = current.stage_total
            completed_units = current.completed_units
            total_units = current.total_units
            latest_metric = current.latest_metric
            eta_seconds = current.eta_seconds

        current_monotonic = self._monotonic_now() if monotonic_now is None else monotonic_now
        elapsed = self._elapsed_offset + (current_monotonic - self._started_monotonic)
        snapshot = ProgressSnapshot(
            sequence=self._sequence + 1,
            event_kind=kind,
            state=_EVENT_STATES[kind],
            timestamp_utc=self._wall_now(),
            stage=stage,
            stage_index=stage_index,
            stage_total=stage_total,
            completed_units=completed_units,
            total_units=total_units,
            elapsed_seconds=float(elapsed),
            message=message,
            latest_metric=latest_metric,
            latest_checkpoint=latest_checkpoint,
            eta_seconds=eta_seconds,
        )
        payload = canonical_json_bytes(snapshot.model_dump(mode="json", round_trip=True)) + b"\n"
        if len(payload) > MAX_STATUS_BYTES:
            raise ProgressError("progress snapshot exceeds the configured byte limit")

        self._append_event(payload)
        # Preserve the durable sequence even if a later status or console write fails.
        self._sequence = snapshot.sequence
        self._snapshot = snapshot
        self._last_emit_monotonic = current_monotonic
        self._write_status_atomically(payload)
        self._write_console(snapshot)
        self._condition.notify_all()
        return snapshot

    def _preflight_outputs(self) -> ProgressSnapshot | None:
        for path in (self._event_log_path, self._status_path):
            if path.is_symlink():
                raise ProgressError("progress output files must not be symbolic links")
        event_exists = self._event_log_path.exists()
        status_exists = self._status_path.exists()
        if not self._resume:
            if event_exists or status_exists:
                raise ProgressExistsError(
                    "progress output already exists; explicit resume is required"
                )
            return None
        if not event_exists or not status_exists:
            raise ProgressError("resume requires both progress output files")
        if not self._event_log_path.is_file() or not self._status_path.is_file():
            raise ProgressError("progress output paths must be regular files")
        status = self._load_snapshot_file(self._status_path)
        final_event = self._load_last_event()
        if status != final_event:
            raise ProgressError("progress status does not match the final event")
        if status.state is ProgressState.COMPLETED:
            raise ProgressStateError("completed progress output cannot be resumed")
        return status

    def _load_snapshot_file(self, path: Path) -> ProgressSnapshot:
        try:
            if path.stat().st_size > MAX_STATUS_BYTES:
                raise ProgressError("progress status is not a bounded canonical record")
            payload = path.read_bytes()
        except OSError as error:
            raise ProgressIOError("could not read progress status safely") from error
        if not payload.endswith(b"\n") or len(payload) > MAX_STATUS_BYTES:
            raise ProgressError("progress status is not a bounded canonical record")
        try:
            snapshot = ProgressSnapshot.model_validate_json(payload[:-1], strict=True)
        except ValueError as error:
            raise ProgressError("progress status violates its strict contract") from error
        expected = canonical_json_bytes(snapshot.model_dump(mode="json", round_trip=True)) + b"\n"
        if payload != expected:
            raise ProgressError("progress status is not canonical JSON")
        return snapshot

    def _load_last_event(self) -> ProgressSnapshot:
        try:
            size = self._event_log_path.stat().st_size
            if size <= 0:
                raise ProgressError("progress event log is empty")
            with self._event_log_path.open("rb") as stream:
                stream.seek(max(0, size - MAX_STATUS_BYTES))
                tail = stream.read(MAX_STATUS_BYTES)
        except OSError as error:
            raise ProgressIOError("could not read the final progress event safely") from error
        if not tail.endswith(b"\n"):
            raise ProgressError("progress event log lacks its final newline")
        lines = tail[:-1].split(b"\n")
        if not lines or not lines[-1]:
            raise ProgressError("progress event log lacks a final record")
        record = lines[-1] + b"\n"
        if len(record) > MAX_STATUS_BYTES:
            raise ProgressError("final progress event exceeds the configured byte limit")
        try:
            snapshot = ProgressSnapshot.model_validate_json(record[:-1], strict=True)
        except ValueError as error:
            raise ProgressError("final progress event violates its strict contract") from error
        expected = canonical_json_bytes(snapshot.model_dump(mode="json", round_trip=True)) + b"\n"
        if record != expected:
            raise ProgressError("final progress event is not canonical JSON")
        return snapshot

    def _append_event(self, payload: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(self._event_log_path, flags, 0o600)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("progress event target is not a regular file")
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        except OSError as error:
            raise ProgressIOError("could not append the progress event safely") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _write_status_atomically(self, payload: bytes) -> None:
        descriptor: int | None = None
        temporary: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{PROGRESS_STATUS_FILENAME}.",
                suffix=".tmp",
                dir=self._output_directory,
            )
            temporary = Path(name)
            os.fchmod(descriptor, 0o600)
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, self._status_path)
            temporary = None
            directory_descriptor = os.open(self._output_directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError as error:
            raise ProgressIOError("could not replace progress status safely") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def _write_console(self, snapshot: ProgressSnapshot) -> None:
        timestamp = snapshot.timestamp_utc.isoformat().replace("+00:00", "Z")
        fields = [
            f"[{timestamp}]",
            f"event={snapshot.event_kind.value}",
            f"stage={snapshot.stage}",
            f"message={snapshot.message}",
        ]
        if snapshot.completed_units is not None and snapshot.total_units is not None:
            fields.append(f"work={snapshot.completed_units}/{snapshot.total_units}")
        if snapshot.latest_metric is not None:
            fields.append(
                f"metric={snapshot.latest_metric.name}:{snapshot.latest_metric.value:.8g}"
            )
        if snapshot.latest_checkpoint is not None:
            fields.append(f"checkpoint={snapshot.latest_checkpoint}")
        if snapshot.eta_seconds is not None:
            fields.append(f"eta_seconds={snapshot.eta_seconds:.1f}")
        try:
            self._console.write(" | ".join(fields) + "\n")
            self._console.flush()
        except (OSError, ValueError) as error:
            raise ProgressIOError("could not write the progress console event") from error

    def _heartbeat_loop(self) -> None:
        try:
            with self._condition:
                while self._active:
                    now = self._monotonic_now()
                    remaining = self._heartbeat_interval_seconds - (now - self._last_emit_monotonic)
                    if remaining > 0.0:
                        self._condition.wait(timeout=remaining)
                        continue
                    self._emit_locked(
                        kind=ProgressEventKind.HEARTBEAT,
                        stage=self._current_stage(),
                        message="pipeline remains active",
                        preserve_details=True,
                        monotonic_now=now,
                    )
        except BaseException as error:
            with self._condition:
                self._background_error = error
                self._condition.notify_all()

    def _join_heartbeat(self) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise ProgressStateError("progress heartbeat did not stop cleanly")

    def _require_running_locked(self) -> None:
        if self._background_error is not None:
            raise ProgressIOError(
                "background progress heartbeat failed"
            ) from self._background_error
        if not self._active or self._closed or self._current_state() is not ProgressState.RUNNING:
            raise ProgressStateError("progress reporter is not running")

    def _current_state(self) -> ProgressState:
        return ProgressState.RUNNING if self._snapshot is None else self._snapshot.state

    def _current_stage(self) -> str:
        return self._initial_stage if self._snapshot is None else self._snapshot.stage

    def _wall_now(self) -> datetime:
        try:
            value = self._clock.now_utc()
        except BaseException as error:
            raise ProgressError("progress clock could not provide wall time") from error
        if not isinstance(value, datetime):
            raise ProgressError("progress clock must return timezone-aware UTC wall time")
        utc_offset = value.utcoffset()
        if value.tzinfo is None or utc_offset is None:
            raise ProgressError("progress clock must return timezone-aware UTC wall time")
        if utc_offset.total_seconds() != 0:
            raise ProgressError("progress clock must return UTC wall time")
        return value.astimezone(UTC)

    def _monotonic_now(self) -> float:
        try:
            value = self._clock.monotonic()
        except BaseException as error:
            raise ProgressError("progress clock could not provide monotonic time") from error
        if type(value) is not float or not math.isfinite(value):
            raise ProgressError("progress clock must return a finite float")
        if self._active and value < max(self._started_monotonic, self._last_emit_monotonic):
            raise ProgressError("progress clock moved backwards")
        return value


def _validate_stage(value: str) -> str:
    # Mirror the strict serialized field at the runtime API boundary.
    if not isinstance(value, str):
        raise TypeError("stage must be a string")
    if not (1 <= len(value) <= MAX_STAGE_CHARACTERS):
        raise ValueError("stage length is outside the configured bounds")
    if not value[0].isalnum() or not value[0].isascii():
        raise ValueError("stage must begin with an ASCII letter or digit")
    if any(
        not (character.isascii() and (character.isalnum() or character in "._ -"))
        for character in value
    ):
        raise ValueError("stage contains unsafe characters")
    return value


__all__ = [
    "MAX_HEARTBEAT_INTERVAL_SECONDS",
    "MIN_HEARTBEAT_INTERVAL_SECONDS",
    "PROGRESS_CONTRACT_VERSION",
    "PROGRESS_EVENT_LOG_FILENAME",
    "PROGRESS_STATUS_FILENAME",
    "ProgressClock",
    "ProgressError",
    "ProgressEventKind",
    "ProgressExistsError",
    "ProgressIOError",
    "ProgressMetric",
    "ProgressReporter",
    "ProgressSnapshot",
    "ProgressState",
    "ProgressStateError",
    "SystemProgressClock",
]
