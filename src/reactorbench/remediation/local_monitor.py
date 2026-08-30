"""Local-only macOS monitor for the owner-operated Phase 6 remediation rerun.

This module intentionally uses only the Python standard library. The native macOS
window invokes it as a narrow JSON controller because the project Python lacks
``_tkinter`` and the importable system Tk 8.5 renders blank on this Mac. Every project
operation still goes through a fixed shell wrapper that enforces the checkout's
``.venv`` and frozen remediation configuration.

The monitor does not read scientific JSON artifacts directly. It parses only the
bounded, validated output of ``check_phase6_status.sh`` and fails closed on drift.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Final, Protocol

RUN_NAME: Final = "phase6-remediation-v0.4.0-targeted-04"
TOTAL_STAGES: Final = 16
MAX_COMMAND_OUTPUT_BYTES: Final = 32 * 1024
MAX_ACTIVITY_ENTRIES: Final = 100
MAX_ACTIVITY_CHARACTERS: Final = 32 * 1024

PIPELINE_STAGES: Final = (
    "preflight",
    "v02_inventory_and_caps",
    "v02_smoke",
    "v02_development_training",
    "v02_development_gate",
    "v03_data_audit",
    "v03_smoke",
    "v03_candidate_training",
    "v03_development_evaluation",
    "v03_gate",
    "v04_shadow_freeze",
    "v04_pilot",
    "v04_candidate_training",
    "v04_shadow_evaluation",
    "v04_gate_and_final_policy_freeze",
    "review_bundle",
)
VERSION_STAGE_GROUPS: Final = (
    ("Setup", 0, 1),
    ("v0.2", 1, 5),
    ("v0.3", 5, 10),
    ("v0.4", 10, 15),
    ("Finalization", 15, 16),
)
PROGRESS_ONLY_STAGES: Final = ("pipeline_start", "pipeline_resume")

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{7,64}$")
_STAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,79}$")
_LATEST_PROGRESS_PATTERN = re.compile(
    r"^(started|resumed|progress|heartbeat|checkpoint|completed|failed|stopped) "
    r"at ([A-Za-z0-9][A-Za-z0-9._ -]{0,79}) \(event ([1-9][0-9]*)\)$"
)
_POSITION_PATTERN = re.compile(r"^([1-9][0-9]*)/([1-9][0-9]*)$")
_WORK_PATTERN = re.compile(r"^([0-9]+)/([1-9][0-9]*)$")
_METRIC_PATTERN = re.compile(
    r"^([A-Za-z][A-Za-z0-9._-]{0,63})="
    r"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)$"
)


class _StringEnum(str, Enum):  # noqa: UP042 - the trusted macOS Tk host uses Python 3.9
    pass


class MonitorState(_StringEnum):
    """User-facing states. Invalid evidence is represented as a fail-closed failure."""

    NOT_STARTED = "Not started"
    RUNNING = "Running"
    STOPPED = "Stopped"
    BLOCKED = "Blocked"
    FAILED = "Failed"
    COMPLETED = "Completed"


class GuiAction(_StringEnum):
    """The complete allowlist of subprocess actions available to the monitor."""

    DRY_RUN = "dry-run"
    STATUS = "status"
    START = "start"
    STOP = "stop"
    RESUME = "resume"
    OPEN_FINDER = "open-finder"


@dataclass(frozen=True)
class TrustedPaths:
    """Paths derived only from this reviewed source file, never from GUI input."""

    project_root: Path
    run_directory: Path
    dry_run_wrapper: Path
    status_wrapper: Path
    start_wrapper: Path
    stop_wrapper: Path
    resume_wrapper: Path


@dataclass(frozen=True)
class CommandOutcome:
    """Bounded result of one short allowlisted command."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class MonitorStatus:
    """Strict display model derived from validated command output."""

    state: MonitorState
    verified: bool
    run_exists: bool
    pipeline_status: str
    run_name: str
    source_commit: str
    current_stage: str
    next_stage: str
    stage_index: int | None
    stage_total: int | None
    completed_units: int | None
    total_units: int | None
    latest_message: str
    latest_event: str
    event_sequence: int | None
    elapsed_seconds: float | None
    eta_seconds: float | None
    metric_name: str | None
    metric_value: float | None
    latest_checkpoint: str | None
    interruptions: int
    stop_requested: bool
    latest_update_utc: str | None
    safe_error: str | None = None

    @classmethod
    def not_started(cls, source_commit: str) -> MonitorStatus:
        return cls(
            state=MonitorState.NOT_STARTED,
            verified=True,
            run_exists=False,
            pipeline_status="not_started",
            run_name=RUN_NAME,
            source_commit=source_commit,
            current_stage="preflight",
            next_stage="preflight",
            stage_index=None,
            stage_total=TOTAL_STAGES,
            completed_units=None,
            total_units=None,
            latest_message="The non-overwriting rerun has not been created.",
            latest_event="not started",
            event_sequence=None,
            elapsed_seconds=0.0,
            eta_seconds=None,
            metric_name=None,
            metric_value=None,
            latest_checkpoint=None,
            interruptions=0,
            stop_requested=False,
            latest_update_utc=None,
        )

    @classmethod
    def failed_verification(cls, message: str) -> MonitorStatus:
        return cls(
            state=MonitorState.FAILED,
            verified=False,
            run_exists=False,
            pipeline_status="unverified",
            run_name=RUN_NAME,
            source_commit="Unavailable",
            current_stage="Unavailable",
            next_stage="Unavailable",
            stage_index=None,
            stage_total=TOTAL_STAGES,
            completed_units=None,
            total_units=None,
            latest_message=message,
            latest_event="verification refused",
            event_sequence=None,
            elapsed_seconds=None,
            eta_seconds=None,
            metric_name=None,
            metric_value=None,
            latest_checkpoint=None,
            interruptions=0,
            stop_requested=False,
            latest_update_utc=None,
            safe_error=message,
        )

    @property
    def overall_percent(self) -> float:
        if self.state is MonitorState.COMPLETED:
            return 100.0
        if self.state is MonitorState.NOT_STARTED or self.stage_index is None:
            return 0.0
        stage_fraction = 0.0
        if self.completed_units is not None and self.total_units is not None:
            stage_fraction = self.completed_units / self.total_units
        return min(100.0, max(0.0, ((self.stage_index - 1 + stage_fraction) / TOTAL_STAGES) * 100))

    @property
    def work_percent(self) -> float:
        if self.completed_units is None or self.total_units is None:
            return 0.0
        return min(100.0, max(0.0, (self.completed_units / self.total_units) * 100))

    @property
    def version_progress(self) -> tuple[str, int, int, float]:
        """Return the active version's stage position and bounded completion percent."""

        if self.state is MonitorState.COMPLETED:
            return ("Complete", 1, 1, 100.0)
        if self.stage_index is None:
            return ("Setup", 0, 1, 0.0)
        stage_offset = self.stage_index - 1
        stage_fraction = 0.0
        if self.completed_units is not None and self.total_units is not None:
            stage_fraction = self.completed_units / self.total_units
        for name, first_offset, end_offset in VERSION_STAGE_GROUPS:
            if first_offset <= stage_offset < end_offset:
                stage_total = end_offset - first_offset
                stage_index = stage_offset - first_offset + 1
                percent = ((stage_index - 1 + stage_fraction) / stage_total) * 100
                return (name, stage_index, stage_total, min(100.0, max(0.0, percent)))
        raise AssertionError("current stage is outside the frozen version groups")

    def concise_summary(self) -> str:
        stage = self.current_stage
        if self.stage_index is not None and self.stage_total is not None:
            stage = f"Stage {self.stage_index} of {self.stage_total}: {stage}"
        work = "work unavailable"
        if self.completed_units is not None and self.total_units is not None:
            work = f"work {self.completed_units} of {self.total_units}"
        return (
            f"{self.state.value} | {stage} | {work} | {self.latest_message} | run {self.run_name}"
        )


@dataclass(frozen=True)
class ButtonPolicy:
    """State-derived enablement for every potentially mutating control."""

    dry_run: bool
    start: bool
    refresh: bool
    stop: bool
    resume: bool
    open_finder: bool
    copy_status: bool


class ProcessHandle(Protocol):
    """Minimal detached child-process surface used by production and tests."""

    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None: ...


CommandRunner = Callable[[GuiAction], CommandOutcome]
CommitReader = Callable[[], str]
LongProcessSpawner = Callable[[tuple[str, ...]], ProcessHandle]


class MonitorError(RuntimeError):
    """A bounded local-monitor error that is safe to summarize."""


class StatusParseError(MonitorError):
    """Validated status output changed or was malformed."""


class AlreadyActiveError(MonitorError):
    """The current GUI already owns a live detached launcher handle."""


def _trusted_paths() -> TrustedPaths:
    source = Path(__file__)
    if source.is_symlink() or not source.is_file():
        raise MonitorError("The local monitor source path is unsafe.")
    try:
        resolved_source = source.resolve(strict=True)
        project_root = resolved_source.parents[3]
    except (OSError, IndexError):
        raise MonitorError("The ReactorBench-LM checkout could not be located safely.") from None
    if project_root.is_symlink() or not project_root.is_dir():
        raise MonitorError("The ReactorBench-LM checkout is unsafe.")
    if not (project_root / "pyproject.toml").is_file():
        raise MonitorError("The ReactorBench-LM checkout marker is unavailable.")
    scripts = project_root / "scripts"
    wrappers = {
        "dry_run": scripts / "run_phase6_pipeline.sh",
        "status": scripts / "check_phase6_status.sh",
        "start": scripts / "run_phase6_pipeline.sh",
        "stop": scripts / "stop_phase6_pipeline.sh",
        "resume": scripts / "resume_phase6_pipeline.sh",
    }
    for wrapper in wrappers.values():
        if wrapper.is_symlink() or not wrapper.is_file() or not os.access(wrapper, os.X_OK):
            raise MonitorError("A required Phase 6 wrapper is missing or unsafe.")
    run_directory = project_root / "runs" / RUN_NAME
    return TrustedPaths(
        project_root=project_root,
        run_directory=run_directory,
        dry_run_wrapper=wrappers["dry_run"],
        status_wrapper=wrappers["status"],
        start_wrapper=wrappers["start"],
        stop_wrapper=wrappers["stop"],
        resume_wrapper=wrappers["resume"],
    )


def command_for(action: GuiAction) -> tuple[str, ...]:
    """Return one fixed command. Strings, paths, and caller-supplied arguments are refused."""

    if type(action) is not GuiAction:
        raise TypeError("action must be a GuiAction")
    paths = _trusted_paths()
    if action is GuiAction.DRY_RUN:
        return (str(paths.dry_run_wrapper), "--dry-run")
    if action is GuiAction.STATUS:
        return (str(paths.status_wrapper),)
    if action is GuiAction.START:
        return ("/usr/bin/caffeinate", "-i", str(paths.start_wrapper))
    if action is GuiAction.STOP:
        return (str(paths.stop_wrapper),)
    if action is GuiAction.RESUME:
        return ("/usr/bin/caffeinate", "-i", str(paths.resume_wrapper))
    if action is GuiAction.OPEN_FINDER:
        return ("/usr/bin/open", str(paths.run_directory))
    raise AssertionError("unreachable GUI action")


def _bounded_decode(payload: bytes, *, maximum_bytes: int = MAX_COMMAND_OUTPUT_BYTES) -> str:
    if type(payload) is not bytes or type(maximum_bytes) is not int or maximum_bytes < 128:
        raise TypeError("bounded output requires bytes and a valid maximum")
    marker = b"\n[output truncated safely]"
    truncated = len(payload) > maximum_bytes
    bounded = payload[: maximum_bytes - len(marker)] if truncated else payload
    text = bounded.decode("utf-8", errors="replace")
    safe = "".join(
        character if character in "\n\t" or character >= " " else "�" for character in text
    )
    if truncated:
        safe = safe.rstrip() + marker.decode("ascii")
    return safe


def _run_short_command(action: GuiAction) -> CommandOutcome:
    if type(action) is not GuiAction:
        raise TypeError("action must be a GuiAction")
    if action not in {GuiAction.DRY_RUN, GuiAction.STATUS, GuiAction.STOP, GuiAction.OPEN_FINDER}:
        raise ValueError("action is not a short allowlisted operation")
    command = command_for(action)
    timeout = 120.0 if action is GuiAction.DRY_RUN else 15.0
    try:
        result = subprocess.run(  # noqa: S603 - command_for returns a closed argv allowlist
            command,
            cwd=_trusted_paths().project_root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return CommandOutcome(
            returncode=5,
            stdout="",
            stderr="The local operation exceeded its safe time limit.",
            timed_out=True,
        )
    except OSError:
        return CommandOutcome(
            returncode=5,
            stdout="",
            stderr="The local operation could not be started safely.",
        )
    return CommandOutcome(
        returncode=result.returncode,
        stdout=_bounded_decode(result.stdout),
        stderr=_bounded_decode(result.stderr),
    )


def _read_current_commit() -> str:
    try:
        result = subprocess.run(
            ("/usr/bin/git", "rev-parse", "--verify", "HEAD^{commit}"),
            cwd=_trusted_paths().project_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "Unavailable"
    try:
        commit = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError:
        return "Unavailable"
    if result.returncode != 0 or _COMMIT_PATTERN.fullmatch(commit) is None:
        return "Unavailable"
    return commit


def _spawn_detached(command: tuple[str, ...]) -> ProcessHandle:
    return subprocess.Popen(  # noqa: S603 - command comes only from command_for
        command,
        cwd=_trusted_paths().project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )


class LongRunLauncher:
    """Launch Start/Resume detached; monitor closure never signals the child."""

    def __init__(self, spawner: LongProcessSpawner = _spawn_detached) -> None:
        self._spawner = spawner
        self._process: ProcessHandle | None = None

    def has_active_process(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def launch(self, action: GuiAction) -> int:
        if action not in {GuiAction.START, GuiAction.RESUME}:
            raise ValueError("only Start and Resume are long-running actions")
        if self.has_active_process():
            raise AlreadyActiveError("A pipeline launcher from this monitor is already active.")
        process = self._spawner(command_for(action))
        self._process = process
        return process.pid

    def window_closed(self) -> None:
        """Deliberately do not terminate, wait for, or otherwise signal the child."""


def button_policy(
    status: MonitorStatus,
    *,
    operation_in_progress: bool,
    local_launcher_active: bool,
) -> ButtonPolicy:
    if type(operation_in_progress) is not bool or type(local_launcher_active) is not bool:
        raise TypeError("button-policy flags must be exact booleans")
    available = status.verified and not operation_in_progress
    return ButtonPolicy(
        dry_run=available and not local_launcher_active,
        start=(
            available
            and status.state is MonitorState.NOT_STARTED
            and not status.run_exists
            and not local_launcher_active
        ),
        refresh=not operation_in_progress,
        stop=(
            available
            and status.state is MonitorState.RUNNING
            and status.pipeline_status in {"ready", "running"}
            and not status.stop_requested
        ),
        resume=(
            available
            and status.state is MonitorState.STOPPED
            and status.pipeline_status == "stopped"
            and not local_launcher_active
        ),
        open_finder=available and status.run_exists,
        copy_status=available,
    )


def _required(mapping: dict[str, str], name: str) -> str:
    try:
        return mapping[name]
    except KeyError:
        raise StatusParseError("Verified status output omitted a required field.") from None


def _nonnegative_int(value: str, *, field: str) -> int:
    if not value.isascii() or not value.isdigit():
        raise StatusParseError(f"Verified status {field} is invalid.")
    parsed = int(value)
    if not 0 <= parsed <= 2_147_483_647:
        raise StatusParseError(f"Verified status {field} is outside its bound.")
    return parsed


def _nonnegative_float(value: str, *, field: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise StatusParseError(f"Verified status {field} is invalid.") from None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 366 * 24 * 60 * 60:
        raise StatusParseError(f"Verified status {field} is outside its bound.")
    return parsed


def _safe_checkpoint(value: str) -> str:
    if not value.isascii() or "\\" in value or "\x00" in value:
        raise StatusParseError("Verified checkpoint reference is unsafe.")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or candidate == PurePosixPath(".")
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise StatusParseError("Verified checkpoint reference is unsafe.")
    return value


def parse_status_output(output: str) -> MonitorStatus:
    """Parse only the documented, bounded status command format."""

    if type(output) is not str:
        raise TypeError("status output must be text")
    encoded = output.encode("utf-8")
    if not encoded or len(encoded) > MAX_COMMAND_OUTPUT_BYTES:
        raise StatusParseError("Verified status output is empty or oversized.")
    if any(
        character not in "\n\t" and (character < " " or character > "~") for character in output
    ):
        raise StatusParseError("Verified status output contains unsafe characters.")
    lines = output.splitlines()
    if not 6 <= len(lines) <= 20:
        raise StatusParseError("Verified status output has an unexpected shape.")
    values: dict[str, str] = {}
    allowed = {
        "Pipeline status",
        "Run",
        "Source commit",
        "Next stage",
        "Interruptions",
        "Progress reporter",
        "Latest progress",
        "Message",
        "Stage position",
        "Work completed",
        "Elapsed seconds",
        "Latest metric",
        "Estimated seconds remaining",
        "Latest checkpoint",
        "Latest verified update UTC",
        "Stop request",
    }
    for line in lines:
        if ": " not in line:
            raise StatusParseError("Verified status output contains an invalid line.")
        key, value = line.split(": ", 1)
        if key not in allowed or key in values or not value:
            raise StatusParseError("Verified status output contains an unknown or duplicate field.")
        values[key] = value

    pipeline_status = _required(values, "Pipeline status")
    if pipeline_status not in {"ready", "running", "completed", "failed", "stopped", "blocked"}:
        raise StatusParseError("Verified pipeline state is unknown.")
    run_name = _required(values, "Run")
    if run_name != RUN_NAME:
        raise StatusParseError("Verified status belongs to a different run.")
    source_commit = _required(values, "Source commit")
    if _COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise StatusParseError("Verified source commit is invalid.")
    next_stage = _required(values, "Next stage")
    if next_stage not in {*PIPELINE_STAGES, "none"}:
        raise StatusParseError("Verified next stage is outside the frozen graph.")
    interruptions = _nonnegative_int(_required(values, "Interruptions"), field="interruptions")
    stop_value = _required(values, "Stop request")
    if stop_value not in {"none", "pending"}:
        raise StatusParseError("Verified stop-request state is invalid.")

    reporter_not_started = values.get("Progress reporter") == "not started"
    if "Progress reporter" in values and not reporter_not_started:
        raise StatusParseError("Verified progress-reporter state is invalid.")

    current_stage = next_stage
    latest_event = "not started"
    event_sequence: int | None = None
    stage_index: int | None = None
    stage_total: int | None = None
    completed_units: int | None = None
    total_units: int | None = None
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    metric_name: str | None = None
    metric_value: float | None = None
    latest_checkpoint: str | None = None
    latest_update: str | None = None
    message = "Run created; progress reporting has not started."

    if not reporter_not_started:
        progress_match = _LATEST_PROGRESS_PATTERN.fullmatch(_required(values, "Latest progress"))
        if progress_match is None:
            raise StatusParseError("Verified latest-progress field is invalid.")
        latest_event, current_stage, sequence = progress_match.groups()
        if current_stage not in {*PIPELINE_STAGES, *PROGRESS_ONLY_STAGES}:
            raise StatusParseError("Verified current stage is outside the frozen graph.")
        event_sequence = _nonnegative_int(sequence, field="event sequence")
        if event_sequence < 1:
            raise StatusParseError("Verified event sequence is invalid.")
        message = _required(values, "Message")
        if not 1 <= len(message) <= 320 or not _STAGE_PATTERN.fullmatch(current_stage):
            raise StatusParseError("Verified progress text is outside its bound.")
        if "Stage position" in values:
            position_match = _POSITION_PATTERN.fullmatch(values["Stage position"])
            if position_match is None:
                raise StatusParseError("Verified stage position is invalid.")
            stage_index, stage_total = (int(item) for item in position_match.groups())
            if stage_total != TOTAL_STAGES or not 1 <= stage_index <= stage_total:
                raise StatusParseError("Verified stage position differs from the frozen graph.")
        if "Work completed" in values:
            work_match = _WORK_PATTERN.fullmatch(values["Work completed"])
            if work_match is None:
                raise StatusParseError("Verified work progress is invalid.")
            completed_units, total_units = (int(item) for item in work_match.groups())
            if completed_units > total_units or total_units > 2_147_483_647:
                raise StatusParseError("Verified work progress is outside its bound.")
        elapsed_seconds = _nonnegative_float(
            _required(values, "Elapsed seconds"), field="elapsed time"
        )
        if "Estimated seconds remaining" in values:
            eta_seconds = _nonnegative_float(
                values["Estimated seconds remaining"], field="estimated time"
            )
        if "Latest metric" in values:
            metric_match = _METRIC_PATTERN.fullmatch(values["Latest metric"])
            if metric_match is None:
                raise StatusParseError("Verified metric is invalid.")
            metric_name, metric_text = metric_match.groups()
            metric_value = float(metric_text)
            if not math.isfinite(metric_value):
                raise StatusParseError("Verified metric is not finite.")
        if "Latest checkpoint" in values:
            latest_checkpoint = _safe_checkpoint(values["Latest checkpoint"])
        latest_update = _required(values, "Latest verified update UTC")
        try:
            parsed_update = datetime.fromisoformat(latest_update)
        except ValueError:
            raise StatusParseError("Verified update timestamp is invalid.") from None
        offset = parsed_update.utcoffset()
        if parsed_update.tzinfo is None or offset is None or offset.total_seconds() != 0:
            raise StatusParseError("Verified update timestamp must use UTC.")

    if reporter_not_started and any(
        key in values
        for key in {
            "Latest progress",
            "Message",
            "Stage position",
            "Work completed",
            "Elapsed seconds",
            "Latest metric",
            "Estimated seconds remaining",
            "Latest checkpoint",
            "Latest verified update UTC",
        }
    ):
        raise StatusParseError("Verified status mixed incompatible reporter states.")

    monitor_state = {
        "ready": MonitorState.RUNNING,
        "running": MonitorState.RUNNING,
        "completed": MonitorState.COMPLETED,
        "failed": MonitorState.FAILED,
        "stopped": MonitorState.STOPPED,
        "blocked": MonitorState.BLOCKED,
    }[pipeline_status]
    if monitor_state is MonitorState.COMPLETED:
        stage_index = TOTAL_STAGES
        stage_total = TOTAL_STAGES
        current_stage = "review_bundle"

    return MonitorStatus(
        state=monitor_state,
        verified=True,
        run_exists=True,
        pipeline_status=pipeline_status,
        run_name=run_name,
        source_commit=source_commit,
        current_stage=current_stage,
        next_stage=next_stage,
        stage_index=stage_index,
        stage_total=stage_total,
        completed_units=completed_units,
        total_units=total_units,
        latest_message=message,
        latest_event=latest_event,
        event_sequence=event_sequence,
        elapsed_seconds=elapsed_seconds,
        eta_seconds=eta_seconds,
        metric_name=metric_name,
        metric_value=metric_value,
        latest_checkpoint=latest_checkpoint,
        interruptions=interruptions,
        stop_requested=stop_value == "pending",
        latest_update_utc=latest_update,
    )


class StatusReader:
    """Read only the fixed validated status command and map missing state safely."""

    def __init__(
        self,
        runner: CommandRunner = _run_short_command,
        commit_reader: CommitReader = _read_current_commit,
    ) -> None:
        self._runner = runner
        self._commit_reader = commit_reader

    def read(self) -> MonitorStatus:
        outcome = self._runner(GuiAction.STATUS)
        if outcome.returncode == 0:
            try:
                return parse_status_output(outcome.stdout)
            except (StatusParseError, UnicodeError):
                return MonitorStatus.failed_verification(
                    "Status evidence failed strict validation; controls are disabled."
                )
        if (
            outcome.returncode == 3
            and outcome.stdout == ""
            and outcome.stderr.strip()
            == "Command refused: Pipeline run does not exist; use the start command."
        ):
            return MonitorStatus.not_started(self._commit_reader())
        return MonitorStatus.failed_verification(
            f"Verified status is unavailable (safe exit code {outcome.returncode}); "
            "controls are disabled."
        )


class ActivityBuffer:
    """Small in-memory log with both entry and character bounds."""

    def __init__(self) -> None:
        self._entries: list[str] = []

    def append(self, message: str) -> None:
        safe = "".join(
            character if character == "\t" or character >= " " else " "
            for character in str(message)
        ).strip()
        if not safe:
            return
        entry = safe[:640]
        self._entries.append(entry)
        self._entries = self._entries[-MAX_ACTIVITY_ENTRIES:]
        while sum(len(item) + 1 for item in self._entries) > MAX_ACTIVITY_CHARACTERS:
            self._entries.pop(0)

    def text(self) -> str:
        return "\n".join(self._entries)

    def entries(self) -> tuple[str, ...]:
        return tuple(self._entries)


def _safe_action_message(action: GuiAction, outcome: CommandOutcome) -> str:
    if outcome.returncode == 0:
        return {
            GuiAction.DRY_RUN: "Readiness check passed without starting the rerun.",
            GuiAction.STOP: "Safe-stop request was accepted.",
            GuiAction.OPEN_FINDER: "Run folder opened in Finder.",
        }.get(action, "Local operation completed.")
    meanings = {
        2: "invalid operation request",
        3: "required local state is missing",
        4: "frozen configuration or environment was refused",
        5: "local integrity verification failed",
        6: "another pipeline process is active",
        7: "a managed stage failed safely",
        8: "the run stopped at a safe boundary",
        9: "a scientific gate blocked later work",
        10: "final evaluation remains locked",
        130: "the outer command was interrupted",
    }
    meaning = meanings.get(outcome.returncode, "the operation failed safely")
    return f"Local operation did not complete: {meaning} (exit {outcome.returncode})."


def _smoke() -> int:
    status = StatusReader().read()
    print(status.concise_summary())
    return 0 if status.verified else 5


def _status_payload(status: MonitorStatus) -> dict[str, object]:
    policy = button_policy(
        status,
        operation_in_progress=False,
        local_launcher_active=False,
    )
    version_name, version_stage_index, version_stage_total, version_percent = (
        status.version_progress
    )
    return {
        "completed_units": status.completed_units,
        "current_stage": status.current_stage,
        "elapsed_seconds": status.elapsed_seconds,
        "eta_seconds": status.eta_seconds,
        "event_sequence": status.event_sequence,
        "interruptions": status.interruptions,
        "latest_checkpoint": status.latest_checkpoint,
        "latest_event": status.latest_event,
        "latest_message": status.latest_message,
        "latest_update_utc": status.latest_update_utc,
        "metric_name": status.metric_name,
        "metric_value": status.metric_value,
        "next_stage": status.next_stage,
        "overall_percent": status.overall_percent,
        "pipeline_status": status.pipeline_status,
        "policy": {
            "copy_status": policy.copy_status,
            "dry_run": policy.dry_run,
            "open_finder": policy.open_finder,
            "refresh": policy.refresh,
            "resume": policy.resume,
            "start": policy.start,
            "stop": policy.stop,
        },
        "run_exists": status.run_exists,
        "run_name": status.run_name,
        "safe_error": status.safe_error,
        "source_commit": status.source_commit,
        "stage_index": status.stage_index,
        "stage_total": status.stage_total,
        "state": status.state.value,
        "stop_requested": status.stop_requested,
        "summary": status.concise_summary(),
        "total_units": status.total_units,
        "verified": status.verified,
        "version_name": version_name,
        "version_percent": version_percent,
        "version_stage_index": version_stage_index,
        "version_stage_total": version_stage_total,
        "work_percent": status.work_percent,
    }


def _emit_json(payload: dict[str, object]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(encoded.encode("ascii")) > MAX_COMMAND_OUTPUT_BYTES:
        raise MonitorError("Local monitor output exceeded its safe bound.")
    print(encoded, flush=True)


def _snapshot_json() -> int:
    status = StatusReader().read()
    _emit_json(_status_payload(status))
    return 0 if status.verified else 5


def _operation_json(action: GuiAction) -> int:
    if action in {GuiAction.START, GuiAction.RESUME}:
        try:
            pid = LongRunLauncher().launch(action)
        except (AlreadyActiveError, MonitorError, OSError, RuntimeError, ValueError):
            _emit_json(
                {
                    "kind": action.value,
                    "message": "The detached pipeline launcher could not start safely.",
                    "ok": False,
                    "pid": None,
                    "returncode": 5,
                }
            )
            return 5
        _emit_json(
            {
                "kind": action.value,
                "message": f"Detached {action.value} launcher began as local process {pid}.",
                "ok": True,
                "pid": pid,
                "returncode": 0,
            }
        )
        return 0
    outcome = _run_short_command(action)
    _emit_json(
        {
            "kind": action.value,
            "message": _safe_action_message(action, outcome),
            "ok": outcome.returncode == 0,
            "pid": None,
            "returncode": outcome.returncode,
        }
    )
    return outcome.returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="open_phase6_progress_gui",
        description="Serve the fixed local Phase 6 AppKit monitor controller.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--smoke",
        action="store_true",
        help="verify status integration without opening a window or starting work",
    )
    action.add_argument("--snapshot-json", action="store_true", help=argparse.SUPPRESS)
    action.add_argument("--readiness-check", action="store_true", help=argparse.SUPPRESS)
    action.add_argument("--start-detached", action="store_true", help=argparse.SUPPRESS)
    action.add_argument("--request-stop", action="store_true", help=argparse.SUPPRESS)
    action.add_argument("--resume-detached", action="store_true", help=argparse.SUPPRESS)
    action.add_argument("--open-finder", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.smoke:
        return _smoke()
    if arguments.snapshot_json:
        return _snapshot_json()
    if arguments.readiness_check:
        return _operation_json(GuiAction.DRY_RUN)
    if arguments.start_detached:
        return _operation_json(GuiAction.START)
    if arguments.request_stop:
        return _operation_json(GuiAction.STOP)
    if arguments.resume_detached:
        return _operation_json(GuiAction.RESUME)
    if arguments.open_finder:
        return _operation_json(GuiAction.OPEN_FINDER)
    raise AssertionError("unreachable monitor controller action")


if __name__ == "__main__":
    raise SystemExit(main())
