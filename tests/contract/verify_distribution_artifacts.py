"""Verify built distributions and an isolated, no-network wheel installation."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST_DIRECTORY = ROOT / "dist"
PACKAGE_DATA_PREFIX = "reactorbench/_data"
DATASET_GUARD_PREFIX = "reactorbench/dataset/resources"


def _project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    version = pyproject["project"]["version"]
    if not isinstance(version, str):
        raise TypeError("project.version must be a string")
    return version


def _expected_resource_files() -> dict[str, bytes]:
    expected = {
        f"{PACKAGE_DATA_PREFIX}/configs/default.toml": (
            ROOT / "configs" / "default.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/model/phase4-smoke-v0.1.0.toml": (
            ROOT / "configs" / "model" / "phase4-smoke-v0.1.0.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/dataset/development-v0.1.0.toml": (
            ROOT / "configs" / "dataset" / "development-v0.1.0.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/dataset/remediation-development-v0.3.0.toml": (
            ROOT / "configs" / "dataset" / "remediation-development-v0.3.0.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/dataset/remediation-final-v0.4.0.toml": (
            ROOT / "configs" / "dataset" / "remediation-final-v0.4.0.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/experiments/phase5-pilot-v0.1.0.toml": (
            ROOT / "configs" / "experiments" / "phase5-pilot-v0.1.0.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/experiments/phase6-main-v0.1.0.toml": (
            ROOT / "configs" / "experiments" / "phase6-main-v0.1.0.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/experiments/phase6-remediation-v0.2.0.toml": (
            ROOT / "configs" / "experiments" / "phase6-remediation-v0.2.0.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/experiments/phase6-remediation-v0.3.0.toml": (
            ROOT / "configs" / "experiments" / "phase6-remediation-v0.3.0.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/experiments/phase6-remediation-v0.3.1-targeted.toml": (
            ROOT / "configs" / "experiments" / "phase6-remediation-v0.3.1-targeted.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/experiments/phase6-remediation-v0.3.2-focused.toml": (
            ROOT / "configs" / "experiments" / "phase6-remediation-v0.3.2-focused.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/experiments/phase6-remediation-v0.3.3-hierarchical.toml": (
            ROOT / "configs" / "experiments" / "phase6-remediation-v0.3.3-hierarchical.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/experiments/phase6-remediation-v0.3.4-fault-boosted.toml": (
            ROOT / "configs" / "experiments" / "phase6-remediation-v0.3.4-fault-boosted.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/experiments/phase6-remediation-v0.4.0.toml": (
            ROOT / "configs" / "experiments" / "phase6-remediation-v0.4.0.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/experiments/phase6-remediation-pipeline-v0.4.0.toml": (
            ROOT / "configs" / "experiments" / "phase6-remediation-pipeline-v0.4.0.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/experiments/"
        "phase6-remediation-pipeline-v0.4.0-rerun-01.toml": (
            ROOT / "configs" / "experiments" / "phase6-remediation-pipeline-v0.4.0-rerun-01.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/experiments/"
        "phase6-remediation-pipeline-v0.4.0-rerun-02.toml": (
            ROOT / "configs" / "experiments" / "phase6-remediation-pipeline-v0.4.0-rerun-02.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/experiments/"
        "phase6-remediation-pipeline-v0.4.0-rerun-03.toml": (
            ROOT / "configs" / "experiments" / "phase6-remediation-pipeline-v0.4.0-rerun-03.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/experiments/"
        "phase6-remediation-pipeline-v0.4.0-targeted-01.toml": (
            ROOT / "configs" / "experiments" / "phase6-remediation-pipeline-v0.4.0-targeted-01.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/experiments/"
        "phase6-remediation-pipeline-v0.4.0-targeted-02.toml": (
            ROOT / "configs" / "experiments" / "phase6-remediation-pipeline-v0.4.0-targeted-02.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/experiments/"
        "phase6-remediation-pipeline-v0.4.0-targeted-03.toml": (
            ROOT / "configs" / "experiments" / "phase6-remediation-pipeline-v0.4.0-targeted-03.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/configs/experiments/"
        "phase6-remediation-pipeline-v0.4.0-targeted-04.toml": (
            ROOT / "configs" / "experiments" / "phase6-remediation-pipeline-v0.4.0-targeted-04.toml"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/docs/model/PHASE6_REMEDIATION_RUNBOOK.md": (
            ROOT / "docs" / "model" / "PHASE6_REMEDIATION_RUNBOOK.md"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/docs/model/PHASE6_V02_INVENTORY.json": (
            ROOT / "docs" / "model" / "PHASE6_V02_INVENTORY.json"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/docs/model/PHASE6_V03_COUNTERFACTUAL_CAP.json": (
            ROOT / "docs" / "model" / "PHASE6_V03_COUNTERFACTUAL_CAP.json"
        ).read_bytes(),
        f"{PACKAGE_DATA_PREFIX}/golden/golden-suite-v0.1.0.json": (
            ROOT / "golden" / "golden-suite-v0.1.0.json"
        ).read_bytes(),
    }
    for snapshot_family in ("aster", "compact-output", "dataset"):
        snapshot_root = ROOT / "schemas" / snapshot_family / "v0"
        for path in snapshot_root.rglob("*"):
            if path.is_file():
                relative_path = path.relative_to(snapshot_root).as_posix()
                packaged_path = (
                    f"{PACKAGE_DATA_PREFIX}/schemas/{snapshot_family}/v0/{relative_path}"
                )
                expected[packaged_path] = path.read_bytes()
    for script_name in (
        "check_phase6_status.sh",
        "open_phase6_progress_gui.sh",
        "phase6_monitor_controller.sh",
        "resume_phase6_pipeline.sh",
        "run_phase6_evaluation.sh",
        "run_phase6_pipeline.sh",
        "stop_phase6_pipeline.sh",
    ):
        expected[f"{PACKAGE_DATA_PREFIX}/scripts/{script_name}"] = (
            ROOT / "scripts" / script_name
        ).read_bytes()
    guard_root = ROOT / "src" / "reactorbench" / "dataset" / "resources"
    for path in guard_root.rglob("*"):
        if path.is_file():
            relative_path = path.relative_to(guard_root).as_posix()
            expected[f"{DATASET_GUARD_PREFIX}/{relative_path}"] = path.read_bytes()
    return expected


def _verify_wheel(wheel_path: Path, expected_resources: dict[str, bytes]) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        members = wheel.namelist()
        if len(members) != len(set(members)):
            raise AssertionError("wheel contains duplicate archive members")
        packaged_resources = {
            member
            for member in members
            if member.startswith((f"{PACKAGE_DATA_PREFIX}/", f"{DATASET_GUARD_PREFIX}/"))
        }
        if packaged_resources != set(expected_resources):
            raise AssertionError("wheel package resources do not match reviewed root assets")
        for member, expected_bytes in expected_resources.items():
            if wheel.read(member) != expected_bytes:
                raise AssertionError(f"wheel resource drifted from its reviewed source: {member}")


def _verify_sdist(
    sdist_path: Path,
    *,
    version: str,
    expected_resources: dict[str, bytes],
) -> None:
    archive_root = f"reactorbench_lm-{version}"
    expected_sources: dict[str, bytes] = {}
    for resource_member, content in expected_resources.items():
        if resource_member.startswith(f"{PACKAGE_DATA_PREFIX}/"):
            source_path = resource_member.removeprefix(f"{PACKAGE_DATA_PREFIX}/")
        elif resource_member.startswith(f"{DATASET_GUARD_PREFIX}/"):
            source_path = f"src/{resource_member}"
        else:
            raise AssertionError(f"unknown packaged resource root: {resource_member}")
        expected_sources[f"{archive_root}/{source_path}"] = content
    with tarfile.open(sdist_path, mode="r:gz") as sdist:
        members = sdist.getmembers()
        member_names = [member.name for member in members]
        if len(member_names) != len(set(member_names)):
            raise AssertionError("sdist contains duplicate archive members")
        if any(
            name == f"{archive_root}/tests" or name.startswith(f"{archive_root}/tests/")
            for name in member_names
        ):
            raise AssertionError(
                "sdist must not ship tests without the complete research fixture tree"
            )
        members_by_name = {member.name: member for member in members}
        for member_name, expected_bytes in expected_sources.items():
            archive_member = members_by_name.get(member_name)
            if archive_member is None:
                raise AssertionError(f"sdist is missing a wheel source asset: {member_name}")
            extracted = sdist.extractfile(archive_member)
            if extracted is None or extracted.read() != expected_bytes:
                raise AssertionError(f"sdist asset drifted from its reviewed source: {member_name}")


def _verify_isolated_install(wheel_path: Path) -> None:
    verification = """
from importlib.resources import as_file
import json
from pathlib import Path
import sys

installation = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(installation))

import reactorbench
from reactorbench.dataset.content_guard import guard_manifest
from reactorbench.dataset import dataset_schema_documents, load_dataset_snapshot
from reactorbench.config import load_project_config
from reactorbench.resources import (
    canonical_dataset_schema_snapshot_resource,
    canonical_schema_snapshot_resource,
    compact_output_contract_resource,
    default_config_resource,
    development_dataset_config_resource,
    golden_suite_resource,
    phase4_smoke_config_resource,
    phase5_pilot_config_resource,
    phase6_main_config_resource,
    phase6_remediation_development_dataset_config_resource,
    phase6_remediation_final_dataset_config_resource,
    phase6_remediation_hierarchical_pipeline_config_resource,
    phase6_remediation_hierarchical_v03_config_resource,
    phase6_remediation_pipeline_config_resource,
    phase6_remediation_runbook_resource,
    phase6_remediation_script_resource,
    phase6_remediation_v02_config_resource,
    phase6_remediation_v03_config_resource,
    phase6_remediation_v04_config_resource,
    phase6_v02_inventory_report_resource,
    phase6_v03_counterfactual_cap_report_resource,
)
from reactorbench.schemas import load_snapshot, schema_documents

package_path = Path(reactorbench.__file__).resolve()
assert package_path.is_relative_to(installation)
with as_file(default_config_resource()) as config_path:
    assert load_project_config(config_path).project_name == "ReactorBench-LM"
with as_file(phase4_smoke_config_resource()) as config_path:
    assert config_path.read_bytes().startswith(b'[phase4]')
with as_file(phase5_pilot_config_resource()) as config_path:
    assert config_path.read_bytes().startswith(b'[phase5]')
with as_file(phase6_main_config_resource()) as config_path:
    assert config_path.read_bytes().startswith(b'[phase6]')
with as_file(development_dataset_config_resource()) as config_path:
    assert config_path.read_bytes().startswith(b'[dataset]')
with as_file(phase6_remediation_development_dataset_config_resource()) as config_path:
    assert b'dataset_version = "0.3.0"' in config_path.read_bytes()
with as_file(phase6_remediation_final_dataset_config_resource()) as config_path:
    assert b'dataset_version = "0.4.0"' in config_path.read_bytes()
with as_file(phase6_remediation_v02_config_resource()) as config_path:
    assert config_path.read_bytes().startswith(b'iteration_version = "0.2.0"')
with as_file(phase6_remediation_v03_config_resource()) as config_path:
    assert config_path.read_bytes().startswith(b'iteration_version = "0.3.0"')
with as_file(phase6_remediation_hierarchical_v03_config_resource()) as config_path:
    assert config_path.read_bytes().startswith(b'iteration_version = "0.3.0"')
    assert b'policy_version = "0.3.3-hierarchical"' in config_path.read_bytes()
with as_file(phase6_remediation_v04_config_resource()) as config_path:
    assert config_path.read_bytes().startswith(b'iteration_version = "0.4.0"')
with as_file(phase6_remediation_pipeline_config_resource()) as config_path:
    assert config_path.read_bytes().startswith(b'pipeline_version = "0.4.0"')
with as_file(phase6_remediation_hierarchical_pipeline_config_resource()) as config_path:
    assert b'run_name = "phase6-remediation-v0.4.0-targeted-03"' in config_path.read_bytes()
with as_file(phase6_v02_inventory_report_resource()) as report_path:
    assert json.loads(report_path.read_bytes())["report_version"] == "0.2.0"
with as_file(phase6_v03_counterfactual_cap_report_resource()) as report_path:
    assert json.loads(report_path.read_bytes())["report_version"] == "0.3.0"
with as_file(phase6_remediation_runbook_resource()) as runbook_path:
    assert runbook_path.read_bytes().startswith(b'# Phase 6 remediation local runbook')
for script_name in (
    'check_phase6_status.sh',
    'open_phase6_progress_gui.sh',
    'phase6_monitor_controller.sh',
    'resume_phase6_pipeline.sh',
    'run_phase6_evaluation.sh',
    'run_phase6_pipeline.sh',
    'stop_phase6_pipeline.sh',
):
    with as_file(phase6_remediation_script_resource(script_name)) as script_path:
        assert script_path.read_bytes().startswith(b'#!/')
with as_file(golden_suite_resource()) as golden_path:
    assert b'"packet_sha256"' in golden_path.read_bytes()
with as_file(canonical_schema_snapshot_resource()) as snapshot_path:
    documents, _manifest = load_snapshot(snapshot_path)
    assert documents == schema_documents()
with as_file(canonical_dataset_schema_snapshot_resource()) as snapshot_path:
    documents, _manifest, _contract = load_dataset_snapshot(snapshot_path)
    assert documents == dataset_schema_documents()
with as_file(compact_output_contract_resource()) as contract_path:
    assert b'"contract_version":"0.2.0"' in contract_path.joinpath("contract.json").read_bytes()
guard = guard_manifest()
assert len(str(guard["denylist_sha256"])) == 64
assert len(str(guard["fingerprints_sha256"])) == 64
"""
    with tempfile.TemporaryDirectory(prefix="reactorbench-wheel-") as temporary_directory:
        installation = Path(temporary_directory) / "site-packages"
        base_python = Path(sys.executable).resolve()
        if not base_python.is_file():
            raise FileNotFoundError("current Python executable cannot be resolved")
        subprocess.run(  # noqa: S603
            [
                str(base_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--no-deps",
                "--no-index",
                "--target",
                str(installation),
                str(wheel_path),
            ],
            check=True,
        )
        subprocess.run(  # noqa: S603
            [sys.executable, "-I", "-c", verification, str(installation)],
            check=True,
            cwd=temporary_directory,
        )


def main() -> None:
    version = _project_version()
    wheel_path = DIST_DIRECTORY / f"reactorbench_lm-{version}-py3-none-any.whl"
    sdist_path = DIST_DIRECTORY / f"reactorbench_lm-{version}.tar.gz"
    if not wheel_path.is_file() or not sdist_path.is_file():
        raise FileNotFoundError("expected wheel and sdist artifacts; run the build target first")

    expected_resources = _expected_resource_files()
    _verify_wheel(wheel_path, expected_resources)
    _verify_sdist(sdist_path, version=version, expected_resources=expected_resources)
    _verify_isolated_install(wheel_path)


if __name__ == "__main__":
    main()
