"""Tests for the strict project configuration boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reactorbench.config import (
    RESOLVED_CONFIG_FILENAME,
    ProjectConfig,
    canonical_config_bytes,
    config_sha256,
    create_run_directory,
    load_project_config,
    resolve_project_path,
)


def _config_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1",
        "project_name": "ReactorBench-LM",
        "run_name": "unit-test-run",
        "paths": {
            "output_root": "runs",
            "dataset_root": "data/generated",
            "checkpoint_root": "checkpoints",
            "artifact_root": "artifacts",
        },
        "execution": {
            "seed": 1729,
            "deterministic": True,
            "device": "cpu",
            "max_workers": 1,
        },
    }
    payload.update(overrides)
    return payload


def _write_config(path: Path, *, extra_line: str = "") -> None:
    path.write_text(
        f"""\
schema_version = "1"
project_name = "ReactorBench-LM"
run_name = "unit-test-run"
{extra_line}
[paths]
output_root = "runs"
dataset_root = "data/generated"
checkpoint_root = "checkpoints"
artifact_root = "artifacts"

[execution]
seed = 1729
deterministic = true
device = "cpu"
max_workers = 1
""",
        encoding="utf-8",
    )


def test_load_project_config_accepts_reviewed_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path)

    config = load_project_config(config_path)

    assert config.schema_version == "1"
    assert config.execution.seed == 1729
    assert config.paths.output_root == Path("runs")


def test_unknown_configuration_field_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, extra_line='unexpected = "rejected"\n')

    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_project_config(config_path)


@pytest.mark.parametrize(
    "invalid_path",
    ["/outside", "../outside", "runs/../../outside", "~/outside", "C:\\outside"],
)
def test_output_paths_must_be_project_relative(invalid_path: str) -> None:
    paths: dict[str, object] = {
        "output_root": invalid_path,
        "dataset_root": "data/generated",
        "checkpoint_root": "checkpoints",
        "artifact_root": "artifacts",
    }
    payload = _config_payload(paths=paths)

    with pytest.raises(ValidationError, match="path"):
        ProjectConfig.model_validate(payload)


def test_environment_does_not_override_reviewed_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path)
    monkeypatch.setenv("REACTORBENCH_SEED", "999999")
    monkeypatch.setenv("REACTORBENCH_OUTPUT_ROOT", "/untrusted")

    config = load_project_config(config_path)

    assert config.execution.seed == 1729
    assert config.paths.output_root == Path("runs")


def test_canonical_serialization_and_hash_ignore_mapping_order() -> None:
    first = ProjectConfig.model_validate(_config_payload())
    reversed_payload = dict(reversed(list(_config_payload().items())))
    second = ProjectConfig.model_validate(reversed_payload)

    assert canonical_config_bytes(first) == canonical_config_bytes(second)
    assert config_sha256(first) == config_sha256(second)
    assert len(config_sha256(first)) == 64


def test_resolve_project_path_rejects_existing_symlink_escape(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    outside_root = tmp_path / "outside"
    project_root.mkdir()
    outside_root.mkdir()
    (project_root / "redirect").symlink_to(outside_root, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes"):
        resolve_project_path(project_root, Path("redirect/results"))


def test_create_run_directory_snapshots_config_and_never_overwrites(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = ProjectConfig.model_validate(_config_payload())

    run = create_run_directory(config, project_root)

    assert run.path == project_root / "runs" / "unit-test-run"
    assert run.config_path == run.path / RESOLVED_CONFIG_FILENAME
    assert run.config_path.read_bytes() == canonical_config_bytes(config)
    assert json.loads(run.config_path.read_text(encoding="utf-8"))["execution"]["seed"] == 1729
    assert run.config_sha256 == config_sha256(config)

    with pytest.raises(FileExistsError):
        create_run_directory(config, project_root)
