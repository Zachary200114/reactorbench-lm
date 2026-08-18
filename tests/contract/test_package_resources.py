from __future__ import annotations

import tomllib
from importlib.resources.abc import Traversable
from pathlib import Path

from reactorbench.resources import (
    canonical_schema_snapshot_resource,
    default_config_resource,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "default.toml"
SNAPSHOT_DIRECTORY = ROOT / "schemas" / "aster" / "v0"


def _resource_file_tree(directory: Traversable) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    pending = [("", directory)]
    while pending:
        prefix, current = pending.pop()
        for child in current.iterdir():
            relative_name = f"{prefix}/{child.name}" if prefix else child.name
            if child.is_dir():
                pending.append((relative_name, child))
            elif child.is_file():
                files[relative_name] = child.read_bytes()
    return files


def _source_file_tree(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_resource_api_reads_the_root_reviewed_assets_without_drift() -> None:
    assert default_config_resource().read_bytes() == CONFIG_PATH.read_bytes()
    assert _resource_file_tree(canonical_schema_snapshot_resource()) == _source_file_tree(
        SNAPSHOT_DIRECTORY
    )


def test_distribution_configuration_packages_canonical_root_assets() -> None:
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    hatch_targets = pyproject["tool"]["hatch"]["build"]["targets"]
    sdist_includes = set(hatch_targets["sdist"]["include"])
    assert {"/configs", "/schemas", "/src"} <= sdist_includes
    assert "/tests" not in sdist_includes
    assert hatch_targets["wheel"]["force-include"] == {
        "configs/default.toml": "reactorbench/_data/configs/default.toml",
        "schemas/aster/v0": "reactorbench/_data/schemas/aster/v0",
    }
