"""Crash-safe stage orchestration for the user-operated remediation pipeline.

This module owns only local run state.  Scientific stage implementations are injected
as callbacks, which keeps resume and filesystem behavior independently testable and
prevents the orchestration layer from opening any dataset or evaluation artifact.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal

import torch
from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator

from reactorbench.schemas.base import ContractModel, canonical_json_bytes, canonical_sha256

from .config import PIPELINE_STAGES, PipelineConfig, config_sha256
from .progress import (
    MAX_STATUS_BYTES,
    PROGRESS_EVENT_LOG_FILENAME,
    PROGRESS_STATUS_FILENAME,
    ProgressEventKind,
    ProgressReporter,
    ProgressSnapshot,
    ProgressState,
)

MAX_ERROR_MESSAGE_BYTES = 2_048
MAX_STAGE_ARTIFACTS = 256
MAX_STAGE_METRICS = 256
MAX_STAGE_WARNINGS = 256
MAX_STAGE_ATTEMPTS = 100
MAX_COMPLETION_MARKER_BYTES = 16 * 1024
MAX_STAGE_OUTCOME_BYTES = 1024 * 1024
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class PipelineError(RuntimeError):
    """Base class for bounded, user-safe orchestration failures."""


class PipelineBusyError(PipelineError):
    """Raised when another process owns the run lock."""


class PipelineStageError(PipelineError):
    """Raised after a stage callback or boundary operation fails safely."""


def _safe_stage_failure_message(error: Exception) -> str:
    """Map internal failures to bounded public categories without leaking details."""

    error_name = type(error).__name__
    if error_name == "PipelineResourceLimitError":
        return "Stage failed safely: a configured resource boundary was exceeded."
    if error_name == "PipelineExecutionError":
        return "Stage failed safely: a scientific integrity check failed."
    if isinstance(error, FileExistsError):
        return "Stage failed safely: a protected non-overwriting output already exists."
    if isinstance(error, OSError):
        return "Stage failed safely: a local input/output operation failed."
    if isinstance(error, (TypeError, ValueError)):
        return "Stage failed safely: contract or boundary validation failed."
    return "Stage failed safely: an internal stage callback failed."


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError("artifact paths must be non-empty POSIX relative paths")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ValueError("artifact paths must stay inside the run directory")
    if any(not part or not part.isascii() or not part.isprintable() for part in path.parts):
        raise ValueError("artifact paths must use printable ASCII components")
    return value


def _safe_message(value: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("status messages must be non-empty and trimmed")
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_ERROR_MESSAGE_BYTES or any(
        character < " " or character > "~" for character in value
    ):
        raise ValueError("status message exceeds its safety boundary")
    return value


def _validated_timestamp(value: str) -> str:
    if type(value) is not str:
        raise ValueError("pipeline timestamps must be strings")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("pipeline timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("pipeline timestamps must use UTC")
    if value != parsed.astimezone(UTC).isoformat(timespec="seconds"):
        raise ValueError("pipeline timestamps must use canonical second precision")
    return value


def _atomic_write(path: Path, payload: bytes, *, replace: bool) -> None:
    if path.is_symlink() or path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("atomic output parent is unsafe")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("atomic output temporary path already exists")
    if not replace and path.exists():
        raise FileExistsError("immutable output already exists")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # A hard-link publish fails atomically if the immutable destination
            # appeared after preflight. Both names reference the same fsynced inode.
            os.link(temporary, path)
            temporary.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError:
        raise PipelineError("atomic pipeline output could not be written safely") from None
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _strict_json(payload: bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError("pipeline JSON contains a duplicate key")
            result[key] = value
        return result

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"pipeline JSON contains non-finite data: {value}")
        ),
    )


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    BLOCKED = "blocked"


class ArtifactReference(ContractModel):
    relative_path: str
    sha256: Sha256
    size_bytes: StrictInt = Field(ge=1, le=4 * 1024**3)

    @model_validator(mode="after")
    def path_is_contained(self) -> ArtifactReference:
        _safe_relative_path(self.relative_path)
        return self


class StageMetric(ContractModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_.-]*$")
    value: StrictFloat = Field(allow_inf_nan=False)
    unit: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_/%.-]+$")


class StageOutcome(ContractModel):
    summary: str
    advancement_allowed: StrictBool = True
    artifacts: tuple[ArtifactReference, ...] = ()
    metrics: tuple[StageMetric, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def payload_is_bounded_and_canonical(self) -> StageOutcome:
        _safe_message(self.summary)
        if (
            len(self.artifacts) > MAX_STAGE_ARTIFACTS
            or len(self.metrics) > MAX_STAGE_METRICS
            or len(self.warnings) > MAX_STAGE_WARNINGS
        ):
            raise ValueError("stage outcome exceeds its item bound")
        if len({item.relative_path for item in self.artifacts}) != len(self.artifacts):
            raise ValueError("stage artifacts must be unique")
        if tuple(item.relative_path for item in self.artifacts) != tuple(
            sorted(item.relative_path for item in self.artifacts)
        ):
            raise ValueError("stage artifacts must use canonical path order")
        if len({item.name for item in self.metrics}) != len(self.metrics):
            raise ValueError("stage metrics must be unique")
        if tuple(item.name for item in self.metrics) != tuple(
            sorted(item.name for item in self.metrics)
        ):
            raise ValueError("stage metrics must use canonical name order")
        if any(_safe_message(warning) != warning for warning in self.warnings):
            raise ValueError("stage warnings are unsafe")
        return self


class StageRecord(ContractModel):
    name: str
    ordinal: StrictInt = Field(ge=0, lt=len(PIPELINE_STAGES))
    status: StageStatus
    attempt_count: StrictInt = Field(ge=0, le=MAX_STAGE_ATTEMPTS)
    started_at: str | None = None
    completed_at: str | None = None
    summary: str | None = None
    latest_attempt_path: str | None = None
    advancement_allowed: StrictBool | None = None
    artifact_count: StrictInt = Field(ge=0, le=MAX_STAGE_ARTIFACTS)
    metric_count: StrictInt = Field(ge=0, le=MAX_STAGE_METRICS)

    @model_validator(mode="after")
    def state_shape_is_valid(self) -> StageRecord:
        if self.name != PIPELINE_STAGES[self.ordinal]:
            raise ValueError("stage name and ordinal differ from the frozen graph")
        if self.latest_attempt_path is not None:
            _safe_relative_path(self.latest_attempt_path)
        if self.summary is not None:
            _safe_message(self.summary)
        if self.started_at is not None:
            _validated_timestamp(self.started_at)
        if self.completed_at is not None:
            _validated_timestamp(self.completed_at)
        if (
            self.started_at is not None
            and self.completed_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError("stage completion precedes its start")
        if self.status is StageStatus.PENDING:
            if any(
                (
                    self.attempt_count,
                    self.started_at,
                    self.completed_at,
                    self.summary,
                    self.latest_attempt_path,
                    self.advancement_allowed,
                    self.artifact_count,
                    self.metric_count,
                )
            ):
                raise ValueError("pending stage cannot contain attempt state")
            return self
        if self.attempt_count < 1 or self.started_at is None or self.latest_attempt_path is None:
            raise ValueError("started stage must identify an attempt directory")
        if self.status is StageStatus.RUNNING:
            if (
                self.completed_at is not None
                or self.summary is not None
                or self.advancement_allowed is not None
                or self.artifact_count
                or self.metric_count
            ):
                raise ValueError("running stage contains terminal state")
            return self
        if self.completed_at is None or self.summary is None:
            raise ValueError("terminal stage lacks a completion result")
        if self.status in {StageStatus.COMPLETED, StageStatus.BLOCKED}:
            if self.advancement_allowed is None:
                raise ValueError("terminal gate stage lacks an advancement result")
        elif self.advancement_allowed is not None or self.artifact_count or self.metric_count:
            raise ValueError("unsuccessful stage cannot publish an outcome")
        if self.status is StageStatus.COMPLETED and self.advancement_allowed is not True:
            raise ValueError("completed stage must permit advancement")
        if self.status is StageStatus.BLOCKED and self.advancement_allowed is not False:
            raise ValueError("blocked stage must deny advancement")
        return self


class PipelineState(ContractModel):
    state_version: Literal["0.4.0"] = "0.4.0"
    run_name: str
    pipeline_config_sha256: Sha256
    source_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    status: Literal["ready", "running", "completed", "failed", "stopped", "blocked"]
    current_stage: str | None
    stages: tuple[StageRecord, ...]
    interruption_count: StrictInt = Field(ge=0, le=10_000)
    created_at: str
    updated_at: str
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def graph_and_checksum_match(self) -> PipelineState:
        if tuple(stage.name for stage in self.stages) != PIPELINE_STAGES:
            raise ValueError("pipeline state does not contain the frozen stage graph")
        _validated_timestamp(self.created_at)
        _validated_timestamp(self.updated_at)
        if self.updated_at < self.created_at:
            raise ValueError("pipeline update precedes creation")
        running = tuple(stage.name for stage in self.stages if stage.status is StageStatus.RUNNING)
        if len(running) > 1 or (running[0] if running else None) != self.current_stage:
            raise ValueError("pipeline current-stage pointer is inconsistent")
        active = tuple(
            (index, stage)
            for index, stage in enumerate(self.stages)
            if stage.status is not StageStatus.COMPLETED
        )
        if active:
            first_index = active[0][0]
            if any(
                stage.status is StageStatus.COMPLETED for stage in self.stages[first_index + 1 :]
            ):
                raise ValueError("pipeline stages are not a contiguous completed prefix")
        non_pending = tuple(stage for _, stage in active if stage.status is not StageStatus.PENDING)
        if len(non_pending) > 1 or (non_pending and active and non_pending[0] is not active[0][1]):
            raise ValueError("pipeline has work beyond its first incomplete stage")
        status_counts = {
            status: sum(stage.status is status for stage in self.stages) for status in StageStatus
        }
        if self.status == "ready":
            valid_status = status_counts[StageStatus.PENDING] == len(PIPELINE_STAGES)
        elif self.status == "completed":
            valid_status = status_counts[StageStatus.COMPLETED] == len(PIPELINE_STAGES)
        elif self.status == "blocked":
            valid_status = status_counts[StageStatus.BLOCKED] == 1
        elif self.status == "failed":
            valid_status = status_counts[StageStatus.FAILED] == 1
        elif self.status == "stopped":
            valid_status = (
                not running
                and not status_counts[StageStatus.BLOCKED]
                and not status_counts[StageStatus.FAILED]
                and status_counts[StageStatus.STOPPED] <= 1
            )
        else:
            valid_status = (
                not status_counts[StageStatus.BLOCKED]
                and not status_counts[StageStatus.FAILED]
                and not status_counts[StageStatus.STOPPED]
            )
        if not valid_status:
            raise ValueError("pipeline status does not match its stage states")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("pipeline state checksum mismatch")
        return self


class RunManifest(ContractModel):
    manifest_version: Literal["0.4.0"] = "0.4.0"
    run_name: str
    source_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    pipeline_config_sha256: Sha256
    v02_config_sha256: Sha256
    v03_config_sha256: Sha256
    v04_config_sha256: Sha256
    command: tuple[str, ...]
    python_version: str
    torch_version: str
    platform: str
    mps_built: StrictBool
    mps_available: StrictBool
    created_at: str
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def checksum_matches(self) -> RunManifest:
        if not self.command or any(
            type(item) is not str or not item or len(item.encode("utf-8")) > 4096
            for item in self.command
        ):
            raise ValueError("run command is invalid")
        _validated_timestamp(self.created_at)
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("run manifest checksum mismatch")
        return self


class StageCompletionMarker(ContractModel):
    """Immutable commit record used to recover a finished stage after a crash."""

    marker_version: Literal["0.4.0"] = "0.4.0"
    run_name: str
    pipeline_config_sha256: Sha256
    source_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    stage: str
    ordinal: StrictInt = Field(ge=0, lt=len(PIPELINE_STAGES))
    attempt: StrictInt = Field(ge=1, le=MAX_STAGE_ATTEMPTS)
    attempt_relative_path: str
    outcome: ArtifactReference
    completed_at: str
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def boundary_is_exact(self) -> StageCompletionMarker:
        if self.stage != PIPELINE_STAGES[self.ordinal]:
            raise ValueError("completion marker differs from the frozen stage graph")
        _safe_relative_path(self.attempt_relative_path)
        _validated_timestamp(self.completed_at)
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("completion marker checksum mismatch")
        return self


class StageContext:
    """Narrow filesystem/progress surface supplied to one stage callback."""

    def __init__(
        self,
        *,
        project_root: Path,
        run_directory: Path,
        attempt_directory: Path,
        source_commit: str,
        progress: ProgressReporter,
        stop_requested: Callable[[], bool],
    ) -> None:
        self.project_root = project_root
        self.run_directory = run_directory
        self.attempt_directory = attempt_directory
        self.source_commit = source_commit
        self.progress = progress
        self.stop_requested = stop_requested


StageAction = Callable[[StageContext], StageOutcome]


def _bound_state(state: PipelineState) -> PipelineState:
    payload = state.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    payload["checksum_sha256"] = canonical_sha256(payload)
    return PipelineState.model_validate_json(canonical_json_bytes(payload), strict=True)


def _bound_manifest(manifest: RunManifest) -> RunManifest:
    payload = manifest.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    payload["checksum_sha256"] = canonical_sha256(payload)
    return RunManifest.model_validate_json(canonical_json_bytes(payload), strict=True)


def _bound_completion_marker(marker: StageCompletionMarker) -> StageCompletionMarker:
    payload = marker.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    payload["checksum_sha256"] = canonical_sha256(payload)
    return StageCompletionMarker.model_validate_json(canonical_json_bytes(payload), strict=True)


def _write_model(path: Path, model: ContractModel, *, replace: bool) -> None:
    _atomic_write(
        path,
        canonical_json_bytes(model.model_dump(mode="json", round_trip=True)) + b"\n",
        replace=replace,
    )


def _read_model[ModelT: ContractModel](
    path: Path, model_type: type[ModelT], maximum: int
) -> ModelT:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError("pipeline state artifact is missing, unsafe, or oversized")
    payload = path.read_bytes()
    _strict_json(payload)
    model = model_type.model_validate_json(payload, strict=True)
    expected = canonical_json_bytes(model.model_dump(mode="json", round_trip=True)) + b"\n"
    if payload != expected:
        raise ValueError("pipeline state artifact is not canonical JSON")
    return model


def _artifact_reference(path: Path, *, run_directory: Path) -> ArtifactReference:
    root = run_directory.resolve(strict=True)
    try:
        lexical_relative = path.relative_to(run_directory)
    except ValueError as error:
        raise ValueError("stage outcome references an unsafe artifact") from error
    cursor = run_directory
    for part in lexical_relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("stage outcome references an unsafe artifact")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not path.is_file():
        raise ValueError("stage outcome references an unsafe artifact")
    relative = resolved.relative_to(root).as_posix()
    return ArtifactReference(
        relative_path=relative,
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
    )


class PipelineStore:
    """Non-overwriting run creation plus strict state verification."""

    def __init__(self, run_directory: Path, *, maximum_state_bytes: int) -> None:
        if not isinstance(run_directory, Path):
            raise TypeError("run_directory must be a Path")
        if type(maximum_state_bytes) is not int or not 1024 <= maximum_state_bytes <= 4 * 1024**2:
            raise ValueError("maximum state bytes is invalid")
        self.run_directory = run_directory
        self.maximum_state_bytes = maximum_state_bytes
        self.manifest_path = run_directory / "run-manifest.json"
        self.state_path = run_directory / "pipeline-state.json"
        self.lock_path = run_directory / ".pipeline.lock"

    @classmethod
    def create(
        cls,
        *,
        project_root: Path,
        config: PipelineConfig,
        pipeline_config_sha256: str,
        source_commit: str,
        command: tuple[str, ...],
    ) -> PipelineStore:
        if not isinstance(project_root, Path):
            raise TypeError("project_root must be a Path")
        if project_root.is_symlink() or not project_root.is_dir():
            raise ValueError("project root is unsafe")
        resolved_project_root = project_root.resolve(strict=True)
        if pipeline_config_sha256 != config_sha256(config):
            raise ValueError("pipeline config checksum does not match the strict configuration")
        run_root = resolved_project_root / config.run_root
        if not run_root.is_relative_to(resolved_project_root):
            raise ValueError("pipeline run root escapes the project")
        run_root.mkdir(parents=True, exist_ok=True, mode=0o750)
        if (
            run_root.is_symlink()
            or not run_root.is_dir()
            or run_root.resolve(strict=True) != run_root
        ):
            raise ValueError("pipeline run root is unsafe")
        run_directory = run_root / config.run_name
        if run_directory.exists() or run_directory.is_symlink():
            raise FileExistsError("pipeline run already exists; use the resume command")
        run_directory.mkdir(mode=0o750)
        (run_directory / "stages").mkdir(mode=0o750)
        store = cls(run_directory, maximum_state_bytes=config.maximum_status_bytes)
        created = _utc_now()
        manifest = _bound_manifest(
            RunManifest.model_construct(
                run_name=config.run_name,
                source_commit=source_commit,
                pipeline_config_sha256=pipeline_config_sha256,
                v02_config_sha256=config.v02_config_sha256,
                v03_config_sha256=config.v03_config_sha256,
                v04_config_sha256=config.v04_config_sha256,
                command=command,
                python_version=platform.python_version(),
                torch_version=torch.__version__,
                platform=platform.platform(),
                mps_built=bool(torch.backends.mps.is_built()),
                mps_available=bool(torch.backends.mps.is_available()),
                created_at=created,
                checksum_sha256="0" * 64,
            )
        )
        state = _bound_state(
            PipelineState.model_construct(
                run_name=config.run_name,
                pipeline_config_sha256=pipeline_config_sha256,
                source_commit=source_commit,
                status="ready",
                current_stage=None,
                stages=tuple(
                    StageRecord(
                        name=name,
                        ordinal=index,
                        status=StageStatus.PENDING,
                        attempt_count=0,
                        artifact_count=0,
                        metric_count=0,
                    )
                    for index, name in enumerate(PIPELINE_STAGES)
                ),
                interruption_count=0,
                created_at=created,
                updated_at=created,
                checksum_sha256="0" * 64,
            )
        )
        _write_model(store.manifest_path, manifest, replace=False)
        _write_model(store.state_path, state, replace=False)
        descriptor = os.open(store.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        os.close(descriptor)
        directory_descriptor = os.open(run_directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return store

    def load_manifest(self) -> RunManifest:
        return _read_model(self.manifest_path, RunManifest, self.maximum_state_bytes)

    def load_state(self) -> PipelineState:
        return _read_model(self.state_path, PipelineState, self.maximum_state_bytes)

    def write_state(self, state: PipelineState) -> PipelineState:
        bound = _bound_state(state)
        _write_model(self.state_path, bound, replace=True)
        return bound

    @contextmanager
    def exclusive_lock(self) -> Iterator[None]:
        if self.lock_path.is_symlink() or not self.lock_path.is_file():
            raise ValueError("pipeline lock path is unsafe")
        with self.lock_path.open("rb") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise PipelineBusyError("another remediation pipeline process is active") from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class PipelineEngine:
    """Execute the frozen graph and recover only checksum-bound stage commits."""

    def __init__(
        self,
        *,
        project_root: Path,
        config: PipelineConfig,
        store: PipelineStore,
        actions: Mapping[str, StageAction],
        stop_requested: Callable[[], bool],
    ) -> None:
        if not isinstance(project_root, Path):
            raise TypeError("project_root must be a Path")
        if project_root.is_symlink() or not project_root.is_dir():
            raise ValueError("project root is unsafe")
        if not isinstance(config, PipelineConfig) or not isinstance(store, PipelineStore):
            raise TypeError("config and store must use the remediation contracts")
        if tuple(actions) != PIPELINE_STAGES:
            raise ValueError("pipeline actions must match the frozen stage order")
        if any(not callable(action) for action in actions.values()) or not callable(stop_requested):
            raise TypeError("pipeline callbacks must be callable")
        self.project_root = project_root.resolve(strict=True)
        self.config = config
        self.store = store
        self.actions: Mapping[str, StageAction] = MappingProxyType(dict(actions))
        self.stop_requested = stop_requested
        self.pipeline_config_sha256 = config_sha256(config)

    def _verify_binding(self, state: PipelineState, manifest: RunManifest) -> None:
        expected_run_directory = self.project_root / self.config.run_root / self.config.run_name
        if (
            state.run_name != self.config.run_name
            or manifest.run_name != self.config.run_name
            or state.pipeline_config_sha256 != self.pipeline_config_sha256
            or manifest.pipeline_config_sha256 != self.pipeline_config_sha256
            or state.pipeline_config_sha256 != manifest.pipeline_config_sha256
            or state.source_commit != manifest.source_commit
            or manifest.v02_config_sha256 != self.config.v02_config_sha256
            or manifest.v03_config_sha256 != self.config.v03_config_sha256
            or manifest.v04_config_sha256 != self.config.v04_config_sha256
            or state.created_at != manifest.created_at
        ):
            raise ValueError("pipeline run is bound to different source or configuration")
        if (
            self.store.run_directory.is_symlink()
            or not self.store.run_directory.is_dir()
            or self.store.run_directory.resolve(strict=True) != expected_run_directory
        ):
            raise ValueError("pipeline store differs from the configured run directory")
        stages_root = self.store.run_directory / "stages"
        if stages_root.is_symlink() or not stages_root.is_dir():
            raise ValueError("pipeline stage root is unsafe")

    def _stage_root(self, ordinal: int, stage_name: str) -> Path:
        return self.store.run_directory / "stages" / f"{ordinal:02d}-{stage_name}"

    def _attempt_relative_path(self, ordinal: int, stage_name: str, attempt: int) -> str:
        return f"stages/{ordinal:02d}-{stage_name}/attempt-{attempt:04d}"

    def _attempt_numbers(self, stage_root: Path) -> tuple[int, ...]:
        if not stage_root.exists() and not stage_root.is_symlink():
            return ()
        if stage_root.is_symlink() or not stage_root.is_dir():
            raise ValueError("stage output root is unsafe")
        attempts: list[int] = []
        for child in stage_root.iterdir():
            if child.name == "completed.json":
                if child.is_symlink() or not child.is_file():
                    raise ValueError("stage completion marker is unsafe")
                continue
            match = re.fullmatch(r"attempt-([0-9]{4})", child.name)
            if match is None or child.is_symlink() or not child.is_dir():
                raise ValueError("stage root contains an unexpected entry")
            attempt = int(match.group(1))
            if not 1 <= attempt <= MAX_STAGE_ATTEMPTS:
                raise ValueError("stage attempt directory is outside its bound")
            attempts.append(attempt)
        if len(attempts) != len(set(attempts)):
            raise ValueError("stage attempt directories are ambiguous")
        return tuple(sorted(attempts))

    def _validate_outcome_artifacts(
        self, outcome: StageOutcome, *, attempt_directory: Path
    ) -> None:
        attempt_root = attempt_directory.resolve(strict=True)
        for artifact in outcome.artifacts:
            path = self.store.run_directory / artifact.relative_path
            observed = _artifact_reference(path, run_directory=self.store.run_directory)
            if not observed.relative_path.startswith(
                attempt_directory.relative_to(self.store.run_directory).as_posix() + "/"
            ) or not path.resolve(strict=True).is_relative_to(attempt_root):
                raise ValueError("stage artifacts must stay inside their immutable attempt")
            if observed != artifact:
                raise ValueError("stage artifact changed before its boundary commit")

    def _load_boundary(
        self,
        record: StageRecord,
        manifest: RunManifest,
    ) -> tuple[StageCompletionMarker, StageOutcome, Path]:
        stage_root = self._stage_root(record.ordinal, record.name)
        marker_path = stage_root / "completed.json"
        marker = _read_model(
            marker_path,
            StageCompletionMarker,
            min(self.store.maximum_state_bytes, MAX_COMPLETION_MARKER_BYTES),
        )
        expected_attempt_path = self._attempt_relative_path(
            record.ordinal, record.name, record.attempt_count
        )
        expected_outcome_path = f"{expected_attempt_path}/outcome.json"
        if (
            marker.run_name != self.config.run_name
            or marker.pipeline_config_sha256 != self.pipeline_config_sha256
            or marker.source_commit != manifest.source_commit
            or marker.stage != record.name
            or marker.ordinal != record.ordinal
            or marker.attempt != record.attempt_count
            or marker.attempt_relative_path != expected_attempt_path
            or marker.outcome.relative_path != expected_outcome_path
            or record.latest_attempt_path != expected_attempt_path
        ):
            raise ValueError("stage completion marker does not match its pipeline state")
        attempt_directory = self.store.run_directory / expected_attempt_path
        if attempt_directory.is_symlink() or not attempt_directory.is_dir():
            raise ValueError("completion marker attempt directory is unsafe")
        outcome_path = self.store.run_directory / marker.outcome.relative_path
        observed_outcome = _artifact_reference(outcome_path, run_directory=self.store.run_directory)
        if observed_outcome != marker.outcome:
            raise ValueError("stage outcome differs from its completion marker")
        outcome = _read_model(outcome_path, StageOutcome, MAX_STAGE_OUTCOME_BYTES)
        self._validate_outcome_artifacts(outcome, attempt_directory=attempt_directory)
        return marker, outcome, marker_path

    def _audit_boundaries(
        self, state: PipelineState, manifest: RunManifest
    ) -> dict[int, tuple[StageCompletionMarker, StageOutcome, Path]]:
        stages_root = self.store.run_directory / "stages"
        expected_roots = {
            f"{ordinal:02d}-{name}": ordinal for ordinal, name in enumerate(PIPELINE_STAGES)
        }
        present_roots: set[int] = set()
        for entry in stages_root.iterdir():
            ordinal = expected_roots.get(entry.name)
            if ordinal is None or entry.is_symlink() or not entry.is_dir():
                raise ValueError("pipeline stages directory contains an unexpected entry")
            present_roots.add(ordinal)
        first_incomplete = next(
            (
                ordinal
                for ordinal, record in enumerate(state.stages)
                if record.status is not StageStatus.COMPLETED
            ),
            len(PIPELINE_STAGES),
        )
        if any(ordinal > first_incomplete for ordinal in present_roots):
            raise ValueError("stage outputs exist beyond the first incomplete stage")

        boundaries: dict[int, tuple[StageCompletionMarker, StageOutcome, Path]] = {}
        for ordinal, record in enumerate(state.stages):
            stage_root = self._stage_root(ordinal, record.name)
            attempts = self._attempt_numbers(stage_root)
            if record.attempt_count:
                if record.attempt_count not in attempts:
                    raise ValueError("pipeline state refers to a missing stage attempt")
                expected_attempt_path = self._attempt_relative_path(
                    ordinal, record.name, record.attempt_count
                )
                if record.latest_attempt_path != expected_attempt_path:
                    raise ValueError("pipeline state attempt path is not canonical")
            marker_path = stage_root / "completed.json"
            marker_exists = marker_path.exists() or marker_path.is_symlink()
            if marker_exists:
                if record.status is StageStatus.PENDING:
                    raise ValueError("pending stage has an impossible completion marker")
                boundary = self._load_boundary(record, manifest)
                marker, outcome, _ = boundary
                boundaries[ordinal] = boundary
                if record.status in {StageStatus.COMPLETED, StageStatus.BLOCKED}:
                    expected_status = (
                        StageStatus.COMPLETED
                        if outcome.advancement_allowed
                        else StageStatus.BLOCKED
                    )
                    if (
                        record.status is not expected_status
                        or record.completed_at != marker.completed_at
                        or record.summary != outcome.summary
                        or record.advancement_allowed is not outcome.advancement_allowed
                        or record.artifact_count != len(outcome.artifacts)
                        or record.metric_count != len(outcome.metrics)
                    ):
                        raise ValueError("terminal stage state differs from its immutable outcome")
            elif record.status in {StageStatus.COMPLETED, StageStatus.BLOCKED}:
                raise ValueError("terminal stage is missing its completion marker")
            if (
                record.status in {StageStatus.COMPLETED, StageStatus.BLOCKED}
                and attempts
                and (attempts[-1] != record.attempt_count)
            ):
                raise ValueError("terminal stage contains an uncommitted later attempt")
        return boundaries

    def _load_progress_evidence(
        self,
    ) -> tuple[ProgressSnapshot | None, frozenset[str]]:
        status_path = self.store.run_directory / PROGRESS_STATUS_FILENAME
        event_path = self.store.run_directory / PROGRESS_EVENT_LOG_FILENAME
        status_exists = status_path.exists() or status_path.is_symlink()
        event_exists = event_path.exists() or event_path.is_symlink()
        if not status_exists and not event_exists:
            return None, frozenset()
        if status_exists != event_exists:
            raise ValueError("progress evidence is incomplete")
        if (
            status_path.is_symlink()
            or event_path.is_symlink()
            or not status_path.is_file()
            or not event_path.is_file()
        ):
            raise ValueError("progress evidence is unsafe")
        status = _read_model(
            status_path,
            ProgressSnapshot,
            min(self.config.maximum_status_bytes, MAX_STATUS_BYTES),
        )
        event_size = event_path.stat().st_size
        if not 0 < event_size <= self.config.maximum_event_log_bytes:
            raise ValueError("progress event log exceeds its configured bound")
        checkpoint_paths: set[str] = set()
        final_event: ProgressSnapshot | None = None
        expected_sequence = 1
        with event_path.open("rb") as stream:
            while record := stream.readline(MAX_STATUS_BYTES + 1):
                if not record.endswith(b"\n") or len(record) > MAX_STATUS_BYTES:
                    raise ValueError("progress event log contains an invalid record")
                _strict_json(record)
                event = ProgressSnapshot.model_validate_json(record, strict=True)
                expected = (
                    canonical_json_bytes(event.model_dump(mode="json", round_trip=True)) + b"\n"
                )
                if record != expected or event.sequence != expected_sequence:
                    raise ValueError("progress event log is noncanonical or out of sequence")
                if event.event_kind is ProgressEventKind.CHECKPOINT:
                    if event.latest_checkpoint is None:
                        raise ValueError("progress checkpoint event lacks its checkpoint path")
                    checkpoint_paths.add(event.latest_checkpoint)
                final_event = event
                expected_sequence += 1
        if final_event is None or final_event != status:
            raise ValueError("progress status does not match its final event")
        return status, frozenset(checkpoint_paths)

    def _stop_is_requested(self) -> bool:
        requested = self.stop_requested()
        if type(requested) is not bool:
            raise TypeError("stop_requested must return an exact boolean")
        return requested

    def _commit_boundary(
        self,
        state: PipelineState,
        ordinal: int,
        marker: StageCompletionMarker,
        outcome: StageOutcome,
        *,
        recovered_from_running: bool,
    ) -> PipelineState:
        terminal = StageStatus.COMPLETED if outcome.advancement_allowed else StageStatus.BLOCKED
        stages = list(state.stages)
        stages[ordinal] = stages[ordinal].model_copy(
            update={
                "status": terminal,
                "completed_at": marker.completed_at,
                "summary": outcome.summary,
                "advancement_allowed": outcome.advancement_allowed,
                "artifact_count": len(outcome.artifacts),
                "metric_count": len(outcome.metrics),
            }
        )
        return self.store.write_state(
            state.model_copy(
                update={
                    "status": "running" if outcome.advancement_allowed else "blocked",
                    "current_stage": None,
                    "stages": tuple(stages),
                    "interruption_count": state.interruption_count + int(recovered_from_running),
                    "updated_at": max(state.updated_at, marker.completed_at, _utc_now()),
                }
            )
        )

    def _mark_interrupted(self, state: PipelineState) -> PipelineState:
        if state.current_stage is None:
            return state
        ordinal = PIPELINE_STAGES.index(state.current_stage)
        record = state.stages[ordinal]
        completed = _utc_now()
        stages = list(state.stages)
        stages[ordinal] = record.model_copy(
            update={
                "status": StageStatus.STOPPED,
                "completed_at": completed,
                "summary": "Previous process stopped before a committed stage boundary.",
            }
        )
        return self.store.write_state(
            state.model_copy(
                update={
                    "status": "stopped",
                    "current_stage": None,
                    "stages": tuple(stages),
                    "interruption_count": state.interruption_count + 1,
                    "updated_at": completed,
                }
            )
        )

    def _prepare_attempt(self, record: StageRecord) -> tuple[int, Path]:
        stage_root = self._stage_root(record.ordinal, record.name)
        stages_root = stage_root.parent
        if not stage_root.exists() and not stage_root.is_symlink():
            try:
                stage_root.mkdir(mode=0o750)
                descriptor = os.open(stages_root, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError:
                raise PipelineError("stage output root could not be created safely") from None
        attempts = self._attempt_numbers(stage_root)
        attempt = max((record.attempt_count, *attempts), default=0) + 1
        if attempt > MAX_STAGE_ATTEMPTS:
            raise PipelineError("stage attempt limit exceeded")
        attempt_directory = stage_root / f"attempt-{attempt:04d}"
        try:
            attempt_directory.mkdir(mode=0o750)
            descriptor = os.open(stage_root, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            raise PipelineError("stage attempt could not be created without overwrite") from None
        return attempt, attempt_directory

    def _progress_reporter(
        self, state: PipelineState, snapshot: ProgressSnapshot | None
    ) -> ProgressReporter:
        return ProgressReporter(
            self.store.run_directory,
            initial_stage="pipeline_resume" if state.status != "ready" else "pipeline_start",
            heartbeat_interval_seconds=float(self.config.heartbeat_interval_seconds),
            resume=snapshot is not None,
        )

    def _finish_terminal_progress(
        self,
        state: PipelineState,
        snapshot: ProgressSnapshot | None,
        missing_checkpoints: tuple[tuple[StageCompletionMarker, StageOutcome, Path], ...],
    ) -> None:
        target_completed = state.status == "completed"
        if target_completed and snapshot is not None and snapshot.state is ProgressState.COMPLETED:
            return
        if (
            not target_completed
            and snapshot is not None
            and snapshot.state is ProgressState.STOPPED
            and not missing_checkpoints
        ):
            return
        with self._progress_reporter(state, snapshot) as progress:
            for marker, outcome, marker_path in missing_checkpoints:
                progress.checkpoint(
                    checkpoint=marker_path.relative_to(self.store.run_directory).as_posix(),
                    stage=marker.stage,
                    message=outcome.summary,
                )
            if target_completed:
                progress.complete(message="All development pipeline stages completed.")
            else:
                progress.stop(message="Scientific gate did not pass; later stages were not run.")

    def run(self, *, dry_run: bool = False) -> PipelineState:
        """Run or strictly audit the pipeline; dry runs never mutate run evidence."""

        if type(dry_run) is not bool:
            raise TypeError("dry_run must be an exact boolean")
        with self.store.exclusive_lock():
            manifest = self.store.load_manifest()
            state = self.store.load_state()
            self._verify_binding(state, manifest)
            boundaries = self._audit_boundaries(state, manifest)
            progress_snapshot, checkpoint_paths = self._load_progress_evidence()
            all_stage_records_completed = all(
                record.status is StageStatus.COMPLETED for record in state.stages
            )
            if progress_snapshot is not None and progress_snapshot.state is ProgressState.COMPLETED:
                if not all_stage_records_completed:
                    raise ValueError("completed progress conflicts with incomplete pipeline state")
                if any(
                    state.stages[ordinal].status in {StageStatus.COMPLETED, StageStatus.BLOCKED}
                    and boundary[2].relative_to(self.store.run_directory).as_posix()
                    not in checkpoint_paths
                    for ordinal, boundary in boundaries.items()
                ):
                    raise ValueError("completed progress omits a committed stage checkpoint")
            if dry_run:
                return state

            recovered: tuple[StageCompletionMarker, StageOutcome, Path] | None = None
            if state.current_stage is not None:
                interrupted_index = PIPELINE_STAGES.index(state.current_stage)
                recovered = boundaries.get(interrupted_index)
                if recovered is None:
                    state = self._mark_interrupted(state)
                else:
                    marker, outcome, _ = recovered
                    state = self._commit_boundary(
                        state,
                        interrupted_index,
                        marker,
                        outcome,
                        recovered_from_running=True,
                    )
            else:
                first_incomplete = next(
                    (
                        ordinal
                        for ordinal, record in enumerate(state.stages)
                        if record.status is not StageStatus.COMPLETED
                    ),
                    None,
                )
                if (
                    first_incomplete is not None
                    and state.stages[first_incomplete].status
                    in {StageStatus.RUNNING, StageStatus.FAILED, StageStatus.STOPPED}
                    and first_incomplete in boundaries
                ):
                    recovered = boundaries[first_incomplete]
                    marker, outcome, _ = recovered
                    state = self._commit_boundary(
                        state,
                        first_incomplete,
                        marker,
                        outcome,
                        recovered_from_running=False,
                    )

            missing_checkpoints = tuple(
                boundary
                for ordinal, boundary in sorted(boundaries.items())
                if state.stages[ordinal].status in {StageStatus.COMPLETED, StageStatus.BLOCKED}
                and boundary[2].relative_to(self.store.run_directory).as_posix()
                not in checkpoint_paths
            )
            if all(record.status is StageStatus.COMPLETED for record in state.stages):
                if state.status != "completed":
                    state = self.store.write_state(
                        state.model_copy(
                            update={
                                "status": "completed",
                                "current_stage": None,
                                "updated_at": _utc_now(),
                            }
                        )
                    )
                self._finish_terminal_progress(state, progress_snapshot, missing_checkpoints)
                return state
            if state.status == "blocked":
                self._finish_terminal_progress(state, progress_snapshot, missing_checkpoints)
                return state
            if progress_snapshot is not None and progress_snapshot.state is ProgressState.COMPLETED:
                raise ValueError("completed progress cannot resume an incomplete pipeline")

            with self._progress_reporter(state, progress_snapshot) as progress:
                for marker, outcome, marker_path in missing_checkpoints:
                    progress.checkpoint(
                        checkpoint=marker_path.relative_to(self.store.run_directory).as_posix(),
                        stage=marker.stage,
                        message=outcome.summary,
                    )
                for ordinal, stage_name in enumerate(PIPELINE_STAGES):
                    current = state.stages[ordinal]
                    if current.status is StageStatus.COMPLETED:
                        continue
                    if self._stop_is_requested():
                        state = self.store.write_state(
                            state.model_copy(
                                update={
                                    "status": "stopped",
                                    "current_stage": None,
                                    "updated_at": _utc_now(),
                                }
                            )
                        )
                        progress.stop(message="Stop requested before beginning the next stage.")
                        return state
                    attempt, attempt_directory = self._prepare_attempt(current)
                    stage_root = attempt_directory.parent
                    started = _utc_now()
                    stages = list(state.stages)
                    stages[ordinal] = StageRecord(
                        name=stage_name,
                        ordinal=ordinal,
                        status=StageStatus.RUNNING,
                        attempt_count=attempt,
                        started_at=started,
                        latest_attempt_path=attempt_directory.relative_to(
                            self.store.run_directory
                        ).as_posix(),
                        artifact_count=0,
                        metric_count=0,
                    )
                    state = self.store.write_state(
                        state.model_copy(
                            update={
                                "status": "running",
                                "current_stage": stage_name,
                                "stages": tuple(stages),
                                "updated_at": started,
                            }
                        )
                    )
                    progress.report(
                        stage=stage_name,
                        stage_index=ordinal + 1,
                        stage_total=len(PIPELINE_STAGES),
                        message="Stage started.",
                    )
                    context = StageContext(
                        project_root=self.project_root,
                        run_directory=self.store.run_directory,
                        attempt_directory=attempt_directory,
                        source_commit=manifest.source_commit,
                        progress=progress,
                        stop_requested=self.stop_requested,
                    )
                    try:
                        outcome = self.actions[stage_name](context)
                        if type(outcome) is not StageOutcome:
                            raise TypeError("stage callback returned an invalid outcome")
                        self._validate_outcome_artifacts(
                            outcome, attempt_directory=attempt_directory
                        )
                        outcome_path = attempt_directory / "outcome.json"
                        _write_model(outcome_path, outcome, replace=False)
                        outcome_ref = _artifact_reference(
                            outcome_path, run_directory=self.store.run_directory
                        )
                        completed = _utc_now()
                        marker = _bound_completion_marker(
                            StageCompletionMarker.model_construct(
                                run_name=self.config.run_name,
                                pipeline_config_sha256=self.pipeline_config_sha256,
                                source_commit=manifest.source_commit,
                                stage=stage_name,
                                ordinal=ordinal,
                                attempt=attempt,
                                attempt_relative_path=attempt_directory.relative_to(
                                    self.store.run_directory
                                ).as_posix(),
                                outcome=outcome_ref,
                                completed_at=completed,
                                checksum_sha256="0" * 64,
                            )
                        )
                        marker_path = stage_root / "completed.json"
                        _write_model(marker_path, marker, replace=False)
                    except KeyboardInterrupt:
                        completed = _utc_now()
                        stages = list(state.stages)
                        stages[ordinal] = stages[ordinal].model_copy(
                            update={
                                "status": StageStatus.STOPPED,
                                "completed_at": completed,
                                "summary": "Interrupted before a committed stage boundary.",
                            }
                        )
                        state = self.store.write_state(
                            state.model_copy(
                                update={
                                    "status": "stopped",
                                    "current_stage": None,
                                    "stages": tuple(stages),
                                    "interruption_count": state.interruption_count + 1,
                                    "updated_at": completed,
                                }
                            )
                        )
                        progress.stop(message="Stage interrupted before its durable boundary.")
                        return state
                    except Exception as error:
                        message = _safe_stage_failure_message(error)
                        completed = _utc_now()
                        stages = list(state.stages)
                        stages[ordinal] = stages[ordinal].model_copy(
                            update={
                                "status": StageStatus.FAILED,
                                "completed_at": completed,
                                "summary": message,
                            }
                        )
                        self.store.write_state(
                            state.model_copy(
                                update={
                                    "status": "failed",
                                    "current_stage": None,
                                    "stages": tuple(stages),
                                    "updated_at": completed,
                                }
                            )
                        )
                        # Persist the terminal state before reporting progress. If the
                        # reporter itself fails (for example, disk exhaustion), a later
                        # status/resume still sees an unambiguous failed stage.
                        progress.fail(message=message)
                        raise PipelineStageError(message) from None
                    state = self._commit_boundary(
                        state,
                        ordinal,
                        marker,
                        outcome,
                        recovered_from_running=False,
                    )
                    progress.checkpoint(
                        checkpoint=marker_path.relative_to(self.store.run_directory).as_posix(),
                        message=outcome.summary,
                    )
                    if not outcome.advancement_allowed:
                        progress.stop(
                            message="Scientific gate did not pass; later stages were not run."
                        )
                        return state
                state = self.store.write_state(
                    state.model_copy(
                        update={
                            "status": "completed",
                            "current_stage": None,
                            "updated_at": _utc_now(),
                        }
                    )
                )
                progress.complete(message="All development pipeline stages completed.")
                return state


def command_tuple() -> tuple[str, ...]:
    """Capture the local invocation without executing or normalizing shell text."""

    return (sys.executable, *sys.argv)


__all__ = [
    "ArtifactReference",
    "PipelineBusyError",
    "PipelineEngine",
    "PipelineError",
    "PipelineStageError",
    "PipelineState",
    "PipelineStore",
    "RunManifest",
    "StageAction",
    "StageCompletionMarker",
    "StageContext",
    "StageMetric",
    "StageOutcome",
    "StageRecord",
    "StageStatus",
    "command_tuple",
]
