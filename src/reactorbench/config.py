"""Strict, deterministic, file-backed project configuration.

Configuration is intentionally loaded only from an explicit TOML path. Environment
variables and arbitrary command-line overrides are not part of this boundary.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RESOLVED_CONFIG_FILENAME: Final = "resolved-config.json"
_RUN_NAME_PATTERN: Final = r"^[a-z0-9][a-z0-9._-]{0,63}$"


def _ensure_project_relative(path: Path) -> Path:
    """Return a normalized project-relative path or reject it."""
    raw_path = str(path)
    windows_path = PureWindowsPath(raw_path)

    if not raw_path or raw_path.strip() != raw_path or "\x00" in raw_path:
        raise ValueError("path must be a non-empty, normalized string")
    if path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError("path must be project-relative")
    if path == Path(".") or ".." in path.parts:
        raise ValueError("path must remain within the project root")
    if any(part.startswith("~") for part in path.parts):
        raise ValueError("home-directory expansion is not permitted")

    return path


class ProjectPaths(BaseModel):
    """Project-relative roots for generated or derived output."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)

    output_root: Path = Path("runs")
    dataset_root: Path = Path("data/generated")
    checkpoint_root: Path = Path("checkpoints")
    artifact_root: Path = Path("artifacts")

    @field_validator("*", mode="before")
    @classmethod
    def parse_project_relative_path(cls, value: object) -> Path:
        if isinstance(value, Path):
            path = value
        elif isinstance(value, str):
            path = Path(value)
        else:
            raise ValueError("path values must be strings")
        return _ensure_project_relative(path)


class ExecutionConfig(BaseModel):
    """Runtime-independent reproducibility settings."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    seed: int = Field(ge=0, le=4_294_967_295)
    deterministic: bool = True
    device: Literal["auto", "cpu", "mps"] = "auto"
    max_workers: int = Field(default=1, ge=1, le=64)


class ProjectConfig(BaseModel):
    """Versioned top-level ReactorBench-LM configuration contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1"]
    project_name: Literal["ReactorBench-LM"]
    run_name: str = Field(pattern=_RUN_NAME_PATTERN)
    paths: ProjectPaths
    execution: ExecutionConfig


@dataclass(frozen=True, slots=True)
class RunDirectory:
    """A newly reserved run directory and its immutable config snapshot."""

    path: Path
    config_path: Path
    config_sha256: str


def load_project_config(config_path: Path) -> ProjectConfig:
    """Load and validate an explicit TOML configuration file."""
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)
    return ProjectConfig.model_validate(raw_config)


def canonical_config_bytes(config: ProjectConfig) -> bytes:
    """Serialize a fully resolved config in a stable, hashable representation."""
    payload = config.model_dump(mode="json")
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{serialized}\n".encode()


def config_sha256(config: ProjectConfig) -> str:
    """Return the SHA-256 digest of the canonical resolved configuration."""
    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()


def resolve_project_path(project_root: Path, relative_path: Path) -> Path:
    """Resolve a configured path while enforcing containment in the project root."""
    root = project_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)

    candidate = (root / _ensure_project_relative(relative_path)).resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise ValueError("resolved path escapes the project root")
    return candidate


def create_run_directory(config: ProjectConfig, project_root: Path) -> RunDirectory:
    """Atomically reserve a new run directory and snapshot its resolved config.

    Existing run directories are never reused or overwritten. A failed snapshot write
    intentionally leaves the reserved directory in place so the run name cannot be
    mistaken for a complete, reusable run.
    """
    output_root = resolve_project_path(project_root, config.paths.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = resolve_project_path(project_root, config.paths.output_root)
    if not output_root.is_dir():
        raise NotADirectoryError(output_root)

    run_directory = output_root / config.run_name
    run_directory.mkdir(mode=0o750, exist_ok=False)

    snapshot = canonical_config_bytes(config)
    snapshot_path = run_directory / RESOLVED_CONFIG_FILENAME
    with snapshot_path.open("xb") as snapshot_file:
        snapshot_file.write(snapshot)

    return RunDirectory(
        path=run_directory,
        config_path=snapshot_path,
        config_sha256=hashlib.sha256(snapshot).hexdigest(),
    )
