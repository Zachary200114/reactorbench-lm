"""Read-only access to reviewed configuration and schema package resources."""

from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Final

_PACKAGE_DATA_DIRECTORY: Final = "_data"
_SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _reviewed_resource(*relative_parts: str, directory: bool) -> Traversable:
    packaged = files("reactorbench").joinpath(_PACKAGE_DATA_DIRECTORY, *relative_parts)
    if packaged.is_dir() if directory else packaged.is_file():
        return packaged

    source_tree = _SOURCE_ROOT.joinpath(*relative_parts)
    if source_tree.is_dir() if directory else source_tree.is_file():
        return source_tree

    resource_name = "/".join(relative_parts)
    raise FileNotFoundError(f"reviewed ReactorBench resource is unavailable: {resource_name}")


def default_config_resource() -> Traversable:
    """Return the reviewed default TOML configuration as a resource."""

    return _reviewed_resource("configs", "default.toml", directory=False)


def phase4_smoke_config_resource() -> Traversable:
    """Return the reviewed Phase 4 tokenizer/model/smoke configuration."""

    return _reviewed_resource("configs", "model", "phase4-smoke-v0.1.0.toml", directory=False)


def canonical_schema_snapshot_resource() -> Traversable:
    """Return the current reviewed Aster schema snapshot directory as a resource."""

    return _reviewed_resource("schemas", "aster", "v0", directory=True)


def canonical_dataset_schema_snapshot_resource() -> Traversable:
    """Return the current reviewed dataset schema snapshot directory as a resource."""

    return _reviewed_resource("schemas", "dataset", "v0", directory=True)


__all__ = [
    "canonical_dataset_schema_snapshot_resource",
    "canonical_schema_snapshot_resource",
    "default_config_resource",
    "phase4_smoke_config_resource",
]
