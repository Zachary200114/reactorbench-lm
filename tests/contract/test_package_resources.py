from __future__ import annotations

import tomllib
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

import pytest

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
    phase6_remediation_fault_boosted_pipeline_config_resource,
    phase6_remediation_fault_boosted_v03_config_resource,
    phase6_remediation_final_dataset_config_resource,
    phase6_remediation_focused_pipeline_config_resource,
    phase6_remediation_focused_v03_config_resource,
    phase6_remediation_hierarchical_pipeline_config_resource,
    phase6_remediation_hierarchical_v03_config_resource,
    phase6_remediation_pipeline_config_resource,
    phase6_remediation_runbook_resource,
    phase6_remediation_script_resource,
    phase6_remediation_targeted_pipeline_config_resource,
    phase6_remediation_targeted_v03_config_resource,
    phase6_remediation_task_weighted_pipeline_config_resource,
    phase6_remediation_task_weighted_v03_config_resource,
    phase6_remediation_v02_config_resource,
    phase6_remediation_v03_config_resource,
    phase6_remediation_v04_config_resource,
    phase6_v02_inventory_report_resource,
    phase6_v03_counterfactual_cap_report_resource,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "default.toml"
PHASE4_CONFIG_PATH = ROOT / "configs" / "model" / "phase4-smoke-v0.1.0.toml"
PHASE5_CONFIG_PATH = ROOT / "configs" / "experiments" / "phase5-pilot-v0.1.0.toml"
PHASE6_CONFIG_PATH = ROOT / "configs" / "experiments" / "phase6-main-v0.1.0.toml"
GOLDEN_SUITE_PATH = ROOT / "golden" / "golden-suite-v0.1.0.json"
SNAPSHOT_DIRECTORY = ROOT / "schemas" / "aster" / "v0"
DATASET_SNAPSHOT_DIRECTORY = ROOT / "schemas" / "dataset" / "v0"
COMPACT_OUTPUT_DIRECTORY = ROOT / "schemas" / "compact-output" / "v0"
DATASET_GUARD_DIRECTORY = ROOT / "src" / "reactorbench" / "dataset" / "resources"
PHASE6_SCRIPT_NAMES = (
    "check_phase6_status.sh",
    "open_phase6_progress_gui.sh",
    "phase6_monitor_controller.sh",
    "replay_phase6_targeted03_gate.sh",
    "resume_phase6_pipeline.sh",
    "run_phase6_evaluation.sh",
    "run_phase6_pipeline.sh",
    "stop_phase6_pipeline.sh",
)


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
    assert phase4_smoke_config_resource().read_bytes() == PHASE4_CONFIG_PATH.read_bytes()
    assert phase5_pilot_config_resource().read_bytes() == PHASE5_CONFIG_PATH.read_bytes()
    assert phase6_main_config_resource().read_bytes() == PHASE6_CONFIG_PATH.read_bytes()
    assert (
        development_dataset_config_resource().read_bytes()
        == (ROOT / "configs/dataset/development-v0.1.0.toml").read_bytes()
    )
    assert (
        phase6_remediation_development_dataset_config_resource().read_bytes()
        == (ROOT / "configs/dataset/remediation-development-v0.3.0.toml").read_bytes()
    )
    assert (
        phase6_remediation_final_dataset_config_resource().read_bytes()
        == (ROOT / "configs/dataset/remediation-final-v0.4.0.toml").read_bytes()
    )
    assert (
        phase6_remediation_v02_config_resource().read_bytes()
        == (ROOT / "configs/experiments/phase6-remediation-v0.2.0.toml").read_bytes()
    )
    assert (
        phase6_remediation_v03_config_resource().read_bytes()
        == (ROOT / "configs/experiments/phase6-remediation-v0.3.0.toml").read_bytes()
    )
    assert (
        phase6_remediation_v04_config_resource().read_bytes()
        == (ROOT / "configs/experiments/phase6-remediation-v0.4.0.toml").read_bytes()
    )
    assert (
        phase6_remediation_pipeline_config_resource().read_bytes()
        == (
            ROOT / "configs/experiments/phase6-remediation-pipeline-v0.4.0-rerun-03.toml"
        ).read_bytes()
    )
    assert (
        phase6_remediation_targeted_v03_config_resource().read_bytes()
        == (ROOT / "configs/experiments/phase6-remediation-v0.3.1-targeted.toml").read_bytes()
    )
    assert (
        phase6_remediation_targeted_pipeline_config_resource().read_bytes()
        == (
            ROOT / "configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-01.toml"
        ).read_bytes()
    )
    assert (
        phase6_remediation_focused_v03_config_resource().read_bytes()
        == (ROOT / "configs/experiments/phase6-remediation-v0.3.2-focused.toml").read_bytes()
    )
    assert (
        phase6_remediation_focused_pipeline_config_resource().read_bytes()
        == (
            ROOT / "configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-02.toml"
        ).read_bytes()
    )
    assert (
        phase6_remediation_hierarchical_v03_config_resource().read_bytes()
        == (ROOT / "configs/experiments/phase6-remediation-v0.3.3-hierarchical.toml").read_bytes()
    )
    assert (
        phase6_remediation_hierarchical_pipeline_config_resource().read_bytes()
        == (
            ROOT / "configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-03.toml"
        ).read_bytes()
    )
    assert (
        phase6_remediation_fault_boosted_v03_config_resource().read_bytes()
        == (ROOT / "configs/experiments/phase6-remediation-v0.3.4-fault-boosted.toml").read_bytes()
    )
    assert (
        phase6_remediation_fault_boosted_pipeline_config_resource().read_bytes()
        == (
            ROOT / "configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-04.toml"
        ).read_bytes()
    )
    assert (
        phase6_remediation_task_weighted_v03_config_resource().read_bytes()
        == (ROOT / "configs/experiments/phase6-remediation-v0.3.5-task-weighted.toml").read_bytes()
    )
    assert (
        phase6_remediation_task_weighted_pipeline_config_resource().read_bytes()
        == (
            ROOT / "configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-05.toml"
        ).read_bytes()
    )
    assert (
        phase6_v02_inventory_report_resource().read_bytes()
        == (ROOT / "docs/model/PHASE6_V02_INVENTORY.json").read_bytes()
    )
    assert (
        phase6_v03_counterfactual_cap_report_resource().read_bytes()
        == (ROOT / "docs/model/PHASE6_V03_COUNTERFACTUAL_CAP.json").read_bytes()
    )
    assert (
        phase6_remediation_runbook_resource().read_bytes()
        == (ROOT / "docs/model/PHASE6_REMEDIATION_RUNBOOK.md").read_bytes()
    )
    for script_name in PHASE6_SCRIPT_NAMES:
        assert (
            phase6_remediation_script_resource(script_name).read_bytes()
            == (ROOT / "scripts" / script_name).read_bytes()
        )
    assert golden_suite_resource().read_bytes() == GOLDEN_SUITE_PATH.read_bytes()
    assert _resource_file_tree(canonical_schema_snapshot_resource()) == _source_file_tree(
        SNAPSHOT_DIRECTORY
    )
    assert _resource_file_tree(canonical_dataset_schema_snapshot_resource()) == _source_file_tree(
        DATASET_SNAPSHOT_DIRECTORY
    )
    assert _resource_file_tree(compact_output_contract_resource()) == _source_file_tree(
        COMPACT_OUTPUT_DIRECTORY
    )


def test_dataset_guard_resources_are_importlib_readable_without_drift() -> None:
    packaged = files("reactorbench.dataset.resources")
    assert _resource_file_tree(packaged) == _source_file_tree(DATASET_GUARD_DIRECTORY)


def test_remediation_script_resource_rejects_non_allowlisted_names() -> None:
    assert (
        phase6_remediation_script_resource("replay_phase6_targeted03_gate.sh").read_bytes()
        == (ROOT / "scripts/replay_phase6_targeted03_gate.sh").read_bytes()
    )
    with pytest.raises(ValueError, match="not an allowlisted"):
        phase6_remediation_script_resource("../phase6_rescore_v0_1_1.py")


def test_remediation_runbook_freezes_the_user_operated_safety_workflow() -> None:
    runbook = phase6_remediation_runbook_resource().read_text(encoding="utf-8")
    for command in (
        "./scripts/run_phase6_pipeline.sh",
        "./scripts/open_phase6_progress_gui.sh",
        "./scripts/check_phase6_status.sh",
        "./scripts/replay_phase6_targeted03_gate.sh",
        "./scripts/stop_phase6_pipeline.sh",
        "./scripts/resume_phase6_pipeline.sh",
        "./scripts/run_phase6_evaluation.sh --confirm-final-evaluation",
    ):
        assert command in runbook
    assert "heartbeat every **30 seconds**" in runbook
    assert "planning estimates, not measured" in runbook
    assert "historical G01" in runbook
    assert "G15 packet is prohibited" in runbook
    assert "engineering evidence, not proof" in runbook
    assert "Do not remove `STOP_REQUESTED` yourself" in runbook
    assert "batch sizes **1, 2, and 4**" in runbook
    assert "requires it to equal" in runbook
    assert "final complete event in `progress.jsonl`" in runbook
    assert "`8` a managed stage stop or interrupt" in runbook
    assert "`130` only a keyboard interrupt" in runbook
    assert "intentionally unimplemented and locked" in runbook
    assert "stages/15-review_bundle/attempt-*" in runbook


def test_distribution_configuration_packages_canonical_root_assets() -> None:
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    hatch_targets = pyproject["tool"]["hatch"]["build"]["targets"]
    sdist_includes = set(hatch_targets["sdist"]["include"])
    assert {
        "/configs",
        "/docs/model/PHASE6_REMEDIATION_RUNBOOK.md",
        "/docs/model/PHASE6_V02_INVENTORY.json",
        "/docs/model/PHASE6_V03_COUNTERFACTUAL_CAP.json",
        "/golden",
        "/schemas",
        "/src",
        *(f"/scripts/{script_name}" for script_name in PHASE6_SCRIPT_NAMES),
    } <= sdist_includes
    assert "/tests" not in sdist_includes
    assert hatch_targets["wheel"]["force-include"] == {
        "configs/default.toml": "reactorbench/_data/configs/default.toml",
        "configs/model/phase4-smoke-v0.1.0.toml": (
            "reactorbench/_data/configs/model/phase4-smoke-v0.1.0.toml"
        ),
        "configs/dataset/development-v0.1.0.toml": (
            "reactorbench/_data/configs/dataset/development-v0.1.0.toml"
        ),
        "configs/dataset/remediation-development-v0.3.0.toml": (
            "reactorbench/_data/configs/dataset/remediation-development-v0.3.0.toml"
        ),
        "configs/dataset/remediation-final-v0.4.0.toml": (
            "reactorbench/_data/configs/dataset/remediation-final-v0.4.0.toml"
        ),
        "configs/experiments/phase5-pilot-v0.1.0.toml": (
            "reactorbench/_data/configs/experiments/phase5-pilot-v0.1.0.toml"
        ),
        "configs/experiments/phase6-main-v0.1.0.toml": (
            "reactorbench/_data/configs/experiments/phase6-main-v0.1.0.toml"
        ),
        "configs/experiments/phase6-remediation-v0.2.0.toml": (
            "reactorbench/_data/configs/experiments/phase6-remediation-v0.2.0.toml"
        ),
        "configs/experiments/phase6-remediation-v0.3.0.toml": (
            "reactorbench/_data/configs/experiments/phase6-remediation-v0.3.0.toml"
        ),
        "configs/experiments/phase6-remediation-v0.3.1-targeted.toml": (
            "reactorbench/_data/configs/experiments/phase6-remediation-v0.3.1-targeted.toml"
        ),
        "configs/experiments/phase6-remediation-v0.3.2-focused.toml": (
            "reactorbench/_data/configs/experiments/phase6-remediation-v0.3.2-focused.toml"
        ),
        "configs/experiments/phase6-remediation-v0.3.3-hierarchical.toml": (
            "reactorbench/_data/configs/experiments/phase6-remediation-v0.3.3-hierarchical.toml"
        ),
        "configs/experiments/phase6-remediation-v0.3.4-fault-boosted.toml": (
            "reactorbench/_data/configs/experiments/phase6-remediation-v0.3.4-fault-boosted.toml"
        ),
        "configs/experiments/phase6-remediation-v0.3.5-task-weighted.toml": (
            "reactorbench/_data/configs/experiments/phase6-remediation-v0.3.5-task-weighted.toml"
        ),
        "configs/experiments/phase6-remediation-v0.4.0.toml": (
            "reactorbench/_data/configs/experiments/phase6-remediation-v0.4.0.toml"
        ),
        "configs/experiments/phase6-remediation-pipeline-v0.4.0.toml": (
            "reactorbench/_data/configs/experiments/phase6-remediation-pipeline-v0.4.0.toml"
        ),
        "configs/experiments/phase6-remediation-pipeline-v0.4.0-rerun-01.toml": (
            "reactorbench/_data/configs/experiments/"
            "phase6-remediation-pipeline-v0.4.0-rerun-01.toml"
        ),
        "configs/experiments/phase6-remediation-pipeline-v0.4.0-rerun-02.toml": (
            "reactorbench/_data/configs/experiments/"
            "phase6-remediation-pipeline-v0.4.0-rerun-02.toml"
        ),
        "configs/experiments/phase6-remediation-pipeline-v0.4.0-rerun-03.toml": (
            "reactorbench/_data/configs/experiments/"
            "phase6-remediation-pipeline-v0.4.0-rerun-03.toml"
        ),
        "configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-01.toml": (
            "reactorbench/_data/configs/experiments/"
            "phase6-remediation-pipeline-v0.4.0-targeted-01.toml"
        ),
        "configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-02.toml": (
            "reactorbench/_data/configs/experiments/"
            "phase6-remediation-pipeline-v0.4.0-targeted-02.toml"
        ),
        "configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-03.toml": (
            "reactorbench/_data/configs/experiments/"
            "phase6-remediation-pipeline-v0.4.0-targeted-03.toml"
        ),
        "configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-04.toml": (
            "reactorbench/_data/configs/experiments/"
            "phase6-remediation-pipeline-v0.4.0-targeted-04.toml"
        ),
        "configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-05.toml": (
            "reactorbench/_data/configs/experiments/"
            "phase6-remediation-pipeline-v0.4.0-targeted-05.toml"
        ),
        "docs/model/PHASE6_REMEDIATION_RUNBOOK.md": (
            "reactorbench/_data/docs/model/PHASE6_REMEDIATION_RUNBOOK.md"
        ),
        "docs/model/PHASE6_V02_INVENTORY.json": (
            "reactorbench/_data/docs/model/PHASE6_V02_INVENTORY.json"
        ),
        "docs/model/PHASE6_V03_COUNTERFACTUAL_CAP.json": (
            "reactorbench/_data/docs/model/PHASE6_V03_COUNTERFACTUAL_CAP.json"
        ),
        "golden/golden-suite-v0.1.0.json": ("reactorbench/_data/golden/golden-suite-v0.1.0.json"),
        "schemas/aster/v0": "reactorbench/_data/schemas/aster/v0",
        "schemas/compact-output/v0": "reactorbench/_data/schemas/compact-output/v0",
        "schemas/dataset/v0": "reactorbench/_data/schemas/dataset/v0",
        **{
            f"scripts/{script_name}": f"reactorbench/_data/scripts/{script_name}"
            for script_name in PHASE6_SCRIPT_NAMES
        },
    }
