"""Fail-closed user command layer for the Phase 6 remediation pipeline.

Routine commands operate only on the frozen development pipeline.  This module does
not import, open, or dispatch the final evaluation or historical golden-suite code.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Protocol, TextIO, cast

from pydantic import ValidationError

from reactorbench.schemas.base import canonical_json_bytes

from .config import (
    PIPELINE_STAGES,
    PipelineConfig,
    config_sha256,
    load_pipeline_config,
    load_v02_config,
    load_v03_config,
    load_v04_config,
)
from .orchestration import (
    PipelineBusyError,
    PipelineEngine,
    PipelineError,
    PipelineStageError,
    PipelineState,
    PipelineStore,
    StageAction,
)
from .progress import (
    MAX_STATUS_BYTES,
    PROGRESS_EVENT_LOG_FILENAME,
    PROGRESS_STATUS_FILENAME,
    ProgressError,
    ProgressSnapshot,
)

DEFAULT_PIPELINE_CONFIG = "configs/experiments/phase6-remediation-pipeline-v0.4.0.toml"
FINAL_READY_MARKER = "FINAL_EVALUATION_READY.json"
OWNER_REVIEW_MARKER = "OWNER_REVIEW_APPROVED.json"
FRESH_EXTENSION_MARKER = "FRESH_EXTENSION_MANIFEST.json"
FINAL_ACCESS_LEDGER = "FINAL_EVALUATION_ACCESS.json"
MAX_ARGUMENT_CHARACTERS = 512
MAX_CONTROL_FILE_BYTES = 64 * 1024


class ExitCode(IntEnum):
    """Stable process outcomes used by the shell wrappers."""

    OK = 0
    USAGE = 2
    NOT_FOUND = 3
    CONFIGURATION = 4
    STATE = 5
    BUSY = 6
    PIPELINE_FAILED = 7
    STOPPED = 8
    BLOCKED = 9
    FINAL_EVALUATION_LOCKED = 10
    INTERRUPTED = 130


class CliFailure(RuntimeError):
    """One bounded, user-safe command refusal."""

    def __init__(self, message: str, exit_code: ExitCode) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class LoadedPipeline:
    """Strict frozen configuration plus its canonical checksum and path."""

    config: PipelineConfig
    config_path: Path
    checksum_sha256: str


class StageActionBuilder(Protocol):
    def __call__(
        self, *, project_root: Path, config: PipelineConfig, source_commit: str
    ) -> Mapping[str, StageAction]: ...


class StopPathBuilder(Protocol):
    def __call__(self, *, project_root: Path, config: PipelineConfig) -> Path: ...


class StopRequester(Protocol):
    def __call__(self, path: Path) -> None: ...


class StopArchiver(Protocol):
    def __call__(self, path: Path) -> Path | None: ...


class TerminalReviewOutput(Protocol):
    manifest_path: Path
    summary_path: Path


class TerminalReviewWriter(Protocol):
    def __call__(
        self,
        *,
        project_root: Path,
        config: PipelineConfig,
        source_commit: str,
        state: PipelineState,
    ) -> TerminalReviewOutput: ...


class FinalEvaluationRunner(Protocol):
    def __call__(
        self,
        *,
        project_root: Path,
        config: PipelineConfig,
        source_commit: str,
        explicit_confirmation: bool,
    ) -> object: ...


def _safe_relative_argument(value: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > MAX_ARGUMENT_CHARACTERS
        or "\\" in value
        or "\x00" in value
        or not value.isascii()
    ):
        raise CliFailure("Configuration path is not a safe relative path.", ExitCode.USAGE)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path == PurePosixPath(".")
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CliFailure("Configuration path is not a safe relative path.", ExitCode.USAGE)
    return value


def _safe_project_file(project_root: Path, relative_path: str) -> Path:
    safe = _safe_relative_argument(relative_path)
    candidate = project_root.joinpath(*PurePosixPath(safe).parts)
    cursor = project_root
    for part in PurePosixPath(safe).parts:
        cursor /= part
        if cursor.is_symlink():
            raise CliFailure("A required project file is a symbolic link.", ExitCode.CONFIGURATION)
    if not candidate.is_file():
        raise CliFailure("A required project file is missing.", ExitCode.NOT_FOUND)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise CliFailure(
            "A required project file cannot be resolved safely.", ExitCode.STATE
        ) from None
    if not resolved.is_relative_to(project_root):
        raise CliFailure(
            "A required project file escapes the project root.", ExitCode.CONFIGURATION
        )
    return candidate


def find_project_root(start: Path | None = None) -> Path:
    """Locate and verify the source checkout without trusting the working directory."""

    candidate = Path(__file__).parents[3] if start is None else start
    if not isinstance(candidate, Path):
        raise TypeError("start must be a pathlib.Path")
    if candidate.is_symlink() or not candidate.is_dir():
        raise CliFailure("Project root is missing or unsafe.", ExitCode.NOT_FOUND)
    try:
        root = candidate.resolve(strict=True)
    except OSError:
        raise CliFailure("Project root cannot be resolved safely.", ExitCode.NOT_FOUND) from None
    marker = root / "pyproject.toml"
    source_package = root / "src/reactorbench/remediation"
    if (
        marker.is_symlink()
        or not marker.is_file()
        or source_package.is_symlink()
        or not source_package.is_dir()
    ):
        raise CliFailure(
            "Directory is not a trusted ReactorBench-LM checkout.", ExitCode.CONFIGURATION
        )
    return root


def _require_project_venv(project_root: Path) -> None:
    environment = project_root / ".venv"
    executable = Path(sys.executable)
    if environment.is_symlink() or not environment.is_dir():
        raise CliFailure("Project .venv is missing or unsafe.", ExitCode.CONFIGURATION)
    try:
        environment_resolved = environment.resolve(strict=True)
        prefix_resolved = Path(sys.prefix).resolve(strict=True)
        executable.relative_to(environment)
    except (OSError, ValueError):
        raise CliFailure(
            "Run this command with the project .venv Python interpreter.",
            ExitCode.CONFIGURATION,
        ) from None
    if prefix_resolved != environment_resolved:
        raise CliFailure(
            "Run this command with the project .venv Python interpreter.",
            ExitCode.CONFIGURATION,
        )


def _load_configuration(project_root: Path, relative_path: str) -> LoadedPipeline:
    path = _safe_project_file(project_root, relative_path)
    try:
        config = load_pipeline_config(path)
        referenced = (
            (
                config.v02_config_path,
                config.v02_config_sha256,
                load_v02_config,
            ),
            (
                config.v03_config_path,
                config.v03_config_sha256,
                load_v03_config,
            ),
            (
                config.v04_config_path,
                config.v04_config_sha256,
                load_v04_config,
            ),
        )
        for referenced_path, expected_checksum, loader in referenced:
            model = loader(_safe_project_file(project_root, referenced_path))
            if config_sha256(model) != expected_checksum:
                raise CliFailure(
                    "A referenced remediation config checksum does not match the freeze.",
                    ExitCode.CONFIGURATION,
                )
    except CliFailure:
        raise
    except (OSError, ValueError, ValidationError):
        raise CliFailure(
            "Frozen remediation configuration failed strict validation.",
            ExitCode.CONFIGURATION,
        ) from None
    if config.stop_before_final_evaluation is not True:
        raise CliFailure(
            "Pipeline configuration does not preserve the final-evaluation stop boundary.",
            ExitCode.CONFIGURATION,
        )
    return LoadedPipeline(
        config=config,
        config_path=path,
        checksum_sha256=config_sha256(config),
    )


def _source_commit(project_root: Path) -> str:
    """Read the exact local commit without executing a shell."""

    git = Path("/usr/bin/git")
    if git.is_symlink() or not git.is_file():
        raise CliFailure("Trusted Git executable is unavailable.", ExitCode.CONFIGURATION)
    try:
        result = subprocess.run(  # noqa: S603 - fixed absolute executable and argv
            (str(git), "rev-parse", "--verify", "HEAD^{commit}"),
            cwd=project_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        raise CliFailure("Could not verify the local Git commit.", ExitCode.STATE) from None
    commit = result.stdout.strip()
    if (
        result.returncode != 0
        or not 7 <= len(commit) <= 64
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise CliFailure("Could not verify the local Git commit.", ExitCode.STATE)
    return commit


def _pipeline_module() -> object:
    try:
        return import_module("reactorbench.remediation.pipeline")
    except ImportError:
        raise CliFailure(
            "Pipeline implementation is unavailable.", ExitCode.CONFIGURATION
        ) from None


def _build_stage_actions(
    *, project_root: Path, config: PipelineConfig, source_commit: str
) -> Mapping[str, StageAction]:
    builder = cast(StageActionBuilder, getattr(_pipeline_module(), "build_stage_actions", None))
    if not callable(builder):
        raise CliFailure("Pipeline action factory is unavailable.", ExitCode.CONFIGURATION)
    actions = builder(
        project_root=project_root,
        config=config,
        source_commit=source_commit,
    )
    if tuple(actions) != PIPELINE_STAGES or any(
        not callable(action) for action in actions.values()
    ):
        raise CliFailure("Pipeline actions differ from the frozen graph.", ExitCode.CONFIGURATION)
    return actions


def _pipeline_stop_file(*, project_root: Path, config: PipelineConfig) -> Path:
    builder = cast(StopPathBuilder, getattr(_pipeline_module(), "pipeline_stop_file", None))
    if not callable(builder):
        raise CliFailure("Pipeline stop control is unavailable.", ExitCode.CONFIGURATION)
    path = builder(project_root=project_root, config=config)
    expected_run = project_root / config.run_root / config.run_name
    if not isinstance(path, Path) or path.parent != expected_run or path.name != "STOP_REQUESTED":
        raise CliFailure("Pipeline stop path violates its fixed boundary.", ExitCode.CONFIGURATION)
    return path


def _request_pipeline_stop(path: Path) -> None:
    requester = cast(StopRequester, getattr(_pipeline_module(), "request_pipeline_stop", None))
    if not callable(requester):
        raise CliFailure("Pipeline stop control is unavailable.", ExitCode.CONFIGURATION)
    requester(path)


def _archive_pipeline_stop(path: Path) -> Path | None:
    archiver = cast(StopArchiver, getattr(_pipeline_module(), "archive_pipeline_stop", None))
    if not callable(archiver):
        raise CliFailure("Pipeline stop archive control is unavailable.", ExitCode.CONFIGURATION)
    return archiver(path)


def _write_terminal_review_bundle(
    *,
    project_root: Path,
    config: PipelineConfig,
    source_commit: str,
    state: PipelineState,
) -> TerminalReviewOutput:
    writer = cast(
        TerminalReviewWriter,
        getattr(_pipeline_module(), "write_terminal_review_bundle", None),
    )
    if not callable(writer):
        raise CliFailure("Terminal review writer is unavailable.", ExitCode.CONFIGURATION)
    return writer(
        project_root=project_root,
        config=config,
        source_commit=source_commit,
        state=state,
    )


def _run_final_evaluation(
    *,
    project_root: Path,
    config: PipelineConfig,
    source_commit: str,
    explicit_confirmation: bool,
) -> object:
    module = _pipeline_module()
    runner = cast(FinalEvaluationRunner, getattr(module, "run_final_evaluation", None))
    blocked_error = getattr(module, "FinalEvaluationBlockedError", None)
    if (
        not callable(runner)
        or not isinstance(blocked_error, type)
        or not issubclass(blocked_error, RuntimeError)
    ):
        raise CliFailure("Final-evaluation executor is unavailable.", ExitCode.CONFIGURATION)
    try:
        return runner(
            project_root=project_root,
            config=config,
            source_commit=source_commit,
            explicit_confirmation=explicit_confirmation,
        )
    except RuntimeError as error:
        if isinstance(error, blocked_error):
            raise CliFailure(
                "Final-evaluation prerequisites failed strict verification; access remains locked.",
                ExitCode.FINAL_EVALUATION_LOCKED,
            ) from None
        raise CliFailure("Final evaluation failed safely.", ExitCode.STATE) from None


def _stop_requested(path: Path) -> bool:
    if path.is_symlink():
        raise CliFailure("Pipeline stop marker is unsafe.", ExitCode.STATE)
    if not path.exists():
        return False
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_CONTROL_FILE_BYTES:
        raise CliFailure("Pipeline stop marker is invalid.", ExitCode.STATE)
    return True


def _run_directory(project_root: Path, config: PipelineConfig) -> Path:
    expected = project_root / config.run_root / config.run_name
    if expected.is_symlink():
        raise CliFailure("Pipeline run directory is a symbolic link.", ExitCode.STATE)
    return expected


def _existing_store(
    loaded: LoadedPipeline,
    *,
    project_root: Path,
    source_commit: str,
) -> tuple[PipelineStore, PipelineState]:
    run_directory = _run_directory(project_root, loaded.config)
    if not run_directory.is_dir():
        raise CliFailure("Pipeline run does not exist; use the start command.", ExitCode.NOT_FOUND)
    store = PipelineStore(
        run_directory,
        maximum_state_bytes=loaded.config.maximum_status_bytes,
    )
    try:
        manifest = store.load_manifest()
        state = store.load_state()
    except (OSError, ValueError, ValidationError):
        raise CliFailure("Pipeline run state failed strict verification.", ExitCode.STATE) from None
    if (
        manifest.pipeline_config_sha256 != loaded.checksum_sha256
        or state.pipeline_config_sha256 != loaded.checksum_sha256
        or manifest.source_commit != source_commit
        or state.source_commit != source_commit
        or manifest.run_name != loaded.config.run_name
        or state.run_name != loaded.config.run_name
    ):
        raise CliFailure(
            "Pipeline run belongs to a different config or source commit.",
            ExitCode.CONFIGURATION,
        )
    return store, state


def _progress_snapshot(store: PipelineStore, config: PipelineConfig) -> ProgressSnapshot | None:
    path = store.run_directory / PROGRESS_STATUS_FILENAME
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise CliFailure("Progress status is unsafe.", ExitCode.STATE)
    try:
        size = path.stat().st_size
        if not 0 < size <= min(config.maximum_status_bytes, MAX_STATUS_BYTES):
            raise ValueError
        payload = path.read_bytes()
        snapshot = ProgressSnapshot.model_validate_json(payload, strict=True)
        expected = canonical_json_bytes(snapshot.model_dump(mode="json", round_trip=True)) + b"\n"
        if payload != expected:
            raise ValueError
    except (OSError, ValueError, ValidationError):
        raise CliFailure("Progress status failed strict verification.", ExitCode.STATE) from None
    event_path = store.run_directory / PROGRESS_EVENT_LOG_FILENAME
    if event_path.is_symlink() or not event_path.is_file():
        raise CliFailure("Progress event log is missing or unsafe.", ExitCode.STATE)
    try:
        event_size = event_path.stat().st_size
        if not 0 < event_size <= config.maximum_event_log_bytes:
            raise ValueError
        read_size = min(event_size, MAX_STATUS_BYTES + 1)
        with event_path.open("rb") as stream:
            stream.seek(event_size - read_size)
            tail = stream.read(read_size)
        if not tail.endswith(b"\n"):
            raise ValueError
        if read_size < event_size:
            boundary = tail.find(b"\n")
            if boundary < 0:
                raise ValueError
            tail = tail[boundary + 1 :]
        records = tail.splitlines(keepends=True)
        if not records or len(records[-1]) > MAX_STATUS_BYTES:
            raise ValueError
        final_event_bytes = records[-1]
        final_event = ProgressSnapshot.model_validate_json(final_event_bytes, strict=True)
        expected_event = (
            canonical_json_bytes(final_event.model_dump(mode="json", round_trip=True)) + b"\n"
        )
        if final_event_bytes != expected_event or final_event != snapshot:
            raise ValueError
    except (OSError, ValueError, ValidationError):
        raise CliFailure(
            "Progress status does not match the verified event-log tail.",
            ExitCode.STATE,
        ) from None
    return snapshot


def _exit_for_state(state: PipelineState) -> ExitCode:
    if state.status == "completed":
        return ExitCode.OK
    if state.status == "blocked":
        return ExitCode.BLOCKED
    if state.status == "stopped":
        return ExitCode.STOPPED
    if state.status == "failed":
        return ExitCode.PIPELINE_FAILED
    return ExitCode.STATE


def _next_stage(state: PipelineState) -> str:
    for stage in state.stages:
        if stage.status.value != "completed":
            return stage.name
    return "none"


def _print_status(
    state: PipelineState,
    snapshot: ProgressSnapshot | None,
    *,
    output: TextIO,
) -> None:
    print(f"Pipeline status: {state.status}", file=output)
    print(f"Run: {state.run_name}", file=output)
    print(f"Source commit: {state.source_commit}", file=output)
    print(f"Next stage: {_next_stage(state)}", file=output)
    print(f"Interruptions: {state.interruption_count}", file=output)
    if snapshot is None:
        print("Progress reporter: not started", file=output)
        return
    print(
        f"Latest progress: {snapshot.event_kind.value} at {snapshot.stage} "
        f"(event {snapshot.sequence})",
        file=output,
    )
    print(f"Message: {snapshot.message}", file=output)
    if snapshot.stage_index is not None and snapshot.stage_total is not None:
        print(f"Stage position: {snapshot.stage_index}/{snapshot.stage_total}", file=output)
    if snapshot.completed_units is not None and snapshot.total_units is not None:
        print(f"Work completed: {snapshot.completed_units}/{snapshot.total_units}", file=output)
    print(f"Elapsed seconds: {snapshot.elapsed_seconds:.1f}", file=output)
    if snapshot.latest_metric is not None:
        print(
            f"Latest metric: {snapshot.latest_metric.name}={snapshot.latest_metric.value:.8g}",
            file=output,
        )
    if snapshot.eta_seconds is not None:
        print(f"Estimated seconds remaining: {snapshot.eta_seconds:.1f}", file=output)
    if snapshot.latest_checkpoint is not None:
        print(f"Latest checkpoint: {snapshot.latest_checkpoint}", file=output)


def _verified_review_path(path: Path, *, run_directory: Path) -> Path:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or not path.is_relative_to(run_directory)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValueError("terminal review output is not a safe regular file")
    cursor = run_directory
    for part in path.relative_to(run_directory).parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("terminal review output traverses a symbolic link")
    resolved_run = run_directory.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_run):
        raise ValueError("terminal review output escapes its run directory")
    return resolved


def _emit_terminal_review_bundle(
    *,
    project_root: Path,
    loaded: LoadedPipeline,
    source_commit: str,
    state: PipelineState,
    output: TextIO,
) -> None:
    try:
        review = _write_terminal_review_bundle(
            project_root=project_root,
            config=loaded.config,
            source_commit=source_commit,
            state=state,
        )
        run_directory = _run_directory(project_root, loaded.config)
        manifest = _verified_review_path(review.manifest_path, run_directory=run_directory)
        summary = _verified_review_path(review.summary_path, run_directory=run_directory)
    except (CliFailure, OSError, RuntimeError, TypeError, ValueError):
        print(
            "Terminal review bundle was unavailable; durable pipeline state was preserved.",
            file=output,
        )
        return
    print(f"Terminal review manifest: {manifest.relative_to(project_root)}", file=output)
    print(f"Terminal review summary: {summary.relative_to(project_root)}", file=output)


def _make_engine(
    *,
    project_root: Path,
    loaded: LoadedPipeline,
    store: PipelineStore,
    source_commit: str,
    actions: Mapping[str, StageAction],
) -> PipelineEngine:
    stop_path = _pipeline_stop_file(project_root=project_root, config=loaded.config)
    return PipelineEngine(
        project_root=project_root,
        config=loaded.config,
        store=store,
        actions=actions,
        stop_requested=lambda: _stop_requested(stop_path),
    )


def _start_or_resume(
    command: str,
    *,
    project_root: Path,
    loaded: LoadedPipeline,
    source_commit: str,
    output: TextIO,
) -> ExitCode:
    actions = _build_stage_actions(
        project_root=project_root,
        config=loaded.config,
        source_commit=source_commit,
    )
    if command == "start":
        run_directory = _run_directory(project_root, loaded.config)
        if run_directory.exists() or run_directory.is_symlink():
            raise CliFailure(
                "Pipeline run already exists; use the resume command.",
                ExitCode.STATE,
            )
        store = PipelineStore.create(
            project_root=project_root,
            config=loaded.config,
            pipeline_config_sha256=loaded.checksum_sha256,
            source_commit=source_commit,
            command=(sys.executable, "-m", "reactorbench.remediation", "start"),
        )
        print(f"Starting development pipeline: {loaded.config.run_name}", file=output)
    else:
        store, state = _existing_store(
            loaded,
            project_root=project_root,
            source_commit=source_commit,
        )
        stop_path = _pipeline_stop_file(project_root=project_root, config=loaded.config)
        try:
            with store.exclusive_lock():
                archived = _archive_pipeline_stop(stop_path)
        except PipelineBusyError:
            raise
        except (OSError, ValueError):
            raise CliFailure(
                "Existing stop request could not be archived safely.", ExitCode.STATE
            ) from None
        print(f"Resuming development pipeline from status: {state.status}", file=output)
        if archived is not None:
            print("Previous stop request archived.", file=output)
    engine = _make_engine(
        project_root=project_root,
        loaded=loaded,
        store=store,
        source_commit=source_commit,
        actions=actions,
    )
    try:
        state = engine.run()
    except PipelineStageError:
        try:
            failed_state = store.load_state()
        except (OSError, ValueError, ValidationError):
            failed_state = None
        if failed_state is not None and failed_state.status == "failed":
            _emit_terminal_review_bundle(
                project_root=project_root,
                loaded=loaded,
                source_commit=source_commit,
                state=failed_state,
                output=output,
            )
        raise
    print(f"Pipeline returned status: {state.status}", file=output)
    print(f"Recorded next stage: {_next_stage(state)}", file=output)
    if state.status in {"blocked", "stopped", "failed"}:
        _emit_terminal_review_bundle(
            project_root=project_root,
            loaded=loaded,
            source_commit=source_commit,
            state=state,
            output=output,
        )
    return _exit_for_state(state)


def _dry_run(
    *,
    project_root: Path,
    loaded: LoadedPipeline,
    source_commit: str,
    output: TextIO,
) -> ExitCode:
    actions = _build_stage_actions(
        project_root=project_root,
        config=loaded.config,
        source_commit=source_commit,
    )
    run_directory = _run_directory(project_root, loaded.config)
    print("Dry run: no training, data generation, or evaluation will execute.", file=output)
    print(f"Config checksum: {loaded.checksum_sha256}", file=output)
    print(f"Source commit: {source_commit}", file=output)
    print(f"Frozen stages: {len(PIPELINE_STAGES)}", file=output)
    if not run_directory.exists() and not run_directory.is_symlink():
        print("Run state: not created; start would create a new run.", file=output)
        return ExitCode.OK
    store, _ = _existing_store(
        loaded,
        project_root=project_root,
        source_commit=source_commit,
    )
    state = _make_engine(
        project_root=project_root,
        loaded=loaded,
        store=store,
        source_commit=source_commit,
        actions=actions,
    ).run(dry_run=True)
    print(f"Run state verified without changes: {state.status}", file=output)
    print(f"Next stage: {_next_stage(state)}", file=output)
    return ExitCode.OK


def _status(
    *,
    project_root: Path,
    loaded: LoadedPipeline,
    source_commit: str,
    output: TextIO,
) -> ExitCode:
    store, state = _existing_store(
        loaded,
        project_root=project_root,
        source_commit=source_commit,
    )
    _print_status(state, _progress_snapshot(store, loaded.config), output=output)
    stop_path = _pipeline_stop_file(project_root=project_root, config=loaded.config)
    if _stop_requested(stop_path):
        print("Stop request: pending", file=output)
    else:
        print("Stop request: none", file=output)
    return ExitCode.OK


def _stop(
    *,
    project_root: Path,
    loaded: LoadedPipeline,
    source_commit: str,
    output: TextIO,
) -> ExitCode:
    _, state = _existing_store(loaded, project_root=project_root, source_commit=source_commit)
    if state.status not in {"ready", "running"}:
        raise CliFailure(
            "Pipeline is not active; no new stop request was written.",
            ExitCode.STATE,
        )
    stop_path = _pipeline_stop_file(project_root=project_root, config=loaded.config)
    if _stop_requested(stop_path):
        print("A stop request is already pending.", file=output)
        return ExitCode.OK
    try:
        _request_pipeline_stop(stop_path)
    except FileExistsError:
        print("A stop request is already pending.", file=output)
        return ExitCode.OK
    except (OSError, ValueError):
        raise CliFailure("Stop request could not be written safely.", ExitCode.STATE) from None
    print("Stop requested. The active stage will stop at its next safe boundary.", file=output)
    return ExitCode.OK


def _final_evaluation(
    *,
    project_root: Path,
    loaded: LoadedPipeline,
    source_commit: str,
    confirmed: bool,
    historical_golden: bool,
    output: TextIO,
) -> ExitCode:
    if historical_golden:
        raise CliFailure(
            "Historical golden evidence is prohibited for the fresh final evaluation.",
            ExitCode.FINAL_EVALUATION_LOCKED,
        )
    if not confirmed:
        raise CliFailure(
            "Final evaluation requires the explicit confirmation flag.",
            ExitCode.FINAL_EVALUATION_LOCKED,
        )
    try:
        store, state = _existing_store(
            loaded,
            project_root=project_root,
            source_commit=source_commit,
        )
    except CliFailure:
        raise CliFailure(
            "Verified development run evidence is unavailable; final evaluation remains locked.",
            ExitCode.FINAL_EVALUATION_LOCKED,
        ) from None
    if state.status != "completed":
        raise CliFailure(
            "Development pipeline is not complete; final evaluation remains locked.",
            ExitCode.FINAL_EVALUATION_LOCKED,
        )
    for name in (FINAL_READY_MARKER, OWNER_REVIEW_MARKER, FRESH_EXTENSION_MARKER):
        marker = store.run_directory / name
        if (
            marker.is_symlink()
            or not marker.is_file()
            or not 0 < marker.stat().st_size <= MAX_CONTROL_FILE_BYTES
        ):
            raise CliFailure(
                "Future ready, owner-review, and fresh-extension markers are absent "
                "or unsafe; final evaluation remains locked.",
                ExitCode.FINAL_EVALUATION_LOCKED,
            )
    ledger = store.run_directory / FINAL_ACCESS_LEDGER
    if ledger.exists() or ledger.is_symlink():
        raise CliFailure(
            "A final-access record already exists; repeat final access is prohibited.",
            ExitCode.FINAL_EVALUATION_LOCKED,
        )
    _run_final_evaluation(
        project_root=project_root,
        config=loaded.config,
        source_commit=source_commit,
        explicit_confirmation=confirmed,
    )
    print("Final evaluation completed and the one-access record was committed.", file=output)
    return ExitCode.OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reactorbench.remediation",
        description="Operate the frozen ReactorBench-LM development remediation pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "status", "resume", "stop", "dry-run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", default=DEFAULT_PIPELINE_CONFIG)
    evaluation = subparsers.add_parser("final-evaluation")
    evaluation.add_argument("--config", default=DEFAULT_PIPELINE_CONFIG)
    evaluation.add_argument("--confirm-final-evaluation", action="store_true")
    evaluation.add_argument("--historical-golden", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    project_root: Path | None = None,
    enforce_venv: bool = True,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one bounded CLI command and return its stable process exit code."""

    if type(enforce_venv) is not bool:
        raise TypeError("enforce_venv must be an exact boolean")
    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    arguments = _parser().parse_args(argv)
    try:
        root = find_project_root(project_root)
        if enforce_venv:
            _require_project_venv(root)
        loaded = _load_configuration(root, cast(str, arguments.config))
        source_commit = _source_commit(root)
        command = cast(str, arguments.command)
        if command in {"start", "resume"}:
            code = _start_or_resume(
                command,
                project_root=root,
                loaded=loaded,
                source_commit=source_commit,
                output=output,
            )
        elif command == "status":
            code = _status(
                project_root=root,
                loaded=loaded,
                source_commit=source_commit,
                output=output,
            )
        elif command == "stop":
            code = _stop(
                project_root=root,
                loaded=loaded,
                source_commit=source_commit,
                output=output,
            )
        elif command == "dry-run":
            code = _dry_run(
                project_root=root,
                loaded=loaded,
                source_commit=source_commit,
                output=output,
            )
        else:
            code = _final_evaluation(
                project_root=root,
                loaded=loaded,
                source_commit=source_commit,
                confirmed=cast(bool, arguments.confirm_final_evaluation),
                historical_golden=cast(bool, arguments.historical_golden),
                output=output,
            )
        return int(code)
    except CliFailure as error:
        print(f"Command refused: {error}", file=errors)
        return int(error.exit_code)
    except PipelineBusyError:
        print("Command refused: another pipeline process is active.", file=errors)
        return int(ExitCode.BUSY)
    except PipelineStageError:
        print("Pipeline stage failed safely; inspect status before resuming.", file=errors)
        return int(ExitCode.PIPELINE_FAILED)
    except KeyboardInterrupt:
        print(
            "Interrupted. Durable state was preserved; use resume after checking status.",
            file=errors,
        )
        return int(ExitCode.INTERRUPTED)
    except (
        FileExistsError,
        PipelineError,
        ProgressError,
        OSError,
        ValueError,
        ValidationError,
        TypeError,
    ):
        print("Command refused: local state or boundary validation failed safely.", file=errors)
        return int(ExitCode.STATE)


__all__ = [
    "DEFAULT_PIPELINE_CONFIG",
    "CliFailure",
    "ExitCode",
    "LoadedPipeline",
    "find_project_root",
    "main",
]
