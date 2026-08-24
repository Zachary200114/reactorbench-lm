"""Read-only access to reviewed configuration and schema package resources."""

from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Final

_PACKAGE_DATA_DIRECTORY: Final = "_data"
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_PHASE6_REMEDIATION_SCRIPTS: Final = frozenset(
    {
        "check_phase6_status.sh",
        "resume_phase6_pipeline.sh",
        "run_phase6_evaluation.sh",
        "run_phase6_pipeline.sh",
        "stop_phase6_pipeline.sh",
    }
)


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


def phase5_pilot_config_resource() -> Traversable:
    """Return the reviewed Phase 5 baseline and pilot configuration."""

    return _reviewed_resource("configs", "experiments", "phase5-pilot-v0.1.0.toml", directory=False)


def phase6_main_config_resource() -> Traversable:
    """Return the pilot-informed frozen Phase 6 experiment configuration."""

    return _reviewed_resource("configs", "experiments", "phase6-main-v0.1.0.toml", directory=False)


def phase6_remediation_v02_config_resource() -> Traversable:
    """Return the reviewed Phase 6 v0.2 output-reliability configuration."""

    return _reviewed_resource(
        "configs", "experiments", "phase6-remediation-v0.2.0.toml", directory=False
    )


def phase6_remediation_v03_config_resource() -> Traversable:
    """Return the reviewed Phase 6 v0.3 semantic-learning configuration."""

    return _reviewed_resource(
        "configs", "experiments", "phase6-remediation-v0.3.0.toml", directory=False
    )


def phase6_remediation_v04_config_resource() -> Traversable:
    """Return the reviewed Phase 6 v0.4 generalization configuration."""

    return _reviewed_resource(
        "configs", "experiments", "phase6-remediation-v0.4.0.toml", directory=False
    )


def phase6_remediation_pipeline_config_resource() -> Traversable:
    """Return the reviewed default Phase 6 remediation rerun configuration."""

    return _reviewed_resource(
        "configs",
        "experiments",
        "phase6-remediation-pipeline-v0.4.0-rerun-01.toml",
        directory=False,
    )


def development_dataset_config_resource() -> Traversable:
    """Return the reviewed v0.1 development-dataset configuration."""

    return _reviewed_resource("configs", "dataset", "development-v0.1.0.toml", directory=False)


def phase6_remediation_development_dataset_config_resource() -> Traversable:
    """Return the reviewed v0.3 remediation development-dataset configuration."""

    return _reviewed_resource(
        "configs", "dataset", "remediation-development-v0.3.0.toml", directory=False
    )


def phase6_remediation_final_dataset_config_resource() -> Traversable:
    """Return the reviewed v0.4 fresh-final dataset configuration."""

    return _reviewed_resource(
        "configs", "dataset", "remediation-final-v0.4.0.toml", directory=False
    )


def phase6_v02_inventory_report_resource() -> Traversable:
    """Return the reviewed v0.2 development-only target inventory report."""

    return _reviewed_resource("docs", "model", "PHASE6_V02_INVENTORY.json", directory=False)


def phase6_v03_counterfactual_cap_report_resource() -> Traversable:
    """Return the reviewed v0.3 counterfactual generation-cap report."""

    return _reviewed_resource(
        "docs", "model", "PHASE6_V03_COUNTERFACTUAL_CAP.json", directory=False
    )


def phase6_remediation_runbook_resource() -> Traversable:
    """Return the operator-facing local Phase 6 remediation runbook."""

    return _reviewed_resource("docs", "model", "PHASE6_REMEDIATION_RUNBOOK.md", directory=False)


def phase6_remediation_script_resource(script_name: str) -> Traversable:
    """Return one allowlisted user-operated Phase 6 shell wrapper."""

    if script_name not in _PHASE6_REMEDIATION_SCRIPTS:
        raise ValueError("script name is not an allowlisted Phase 6 remediation wrapper")
    return _reviewed_resource("scripts", script_name, directory=False)


def golden_suite_resource() -> Traversable:
    """Return the checksum-bound developmental G01-G15 owner-review packet."""

    return _reviewed_resource("golden", "golden-suite-v0.1.0.json", directory=False)


def canonical_schema_snapshot_resource() -> Traversable:
    """Return the current reviewed Aster schema snapshot directory as a resource."""

    return _reviewed_resource("schemas", "aster", "v0", directory=True)


def canonical_dataset_schema_snapshot_resource() -> Traversable:
    """Return the current reviewed dataset schema snapshot directory as a resource."""

    return _reviewed_resource("schemas", "dataset", "v0", directory=True)


def compact_output_contract_resource() -> Traversable:
    """Return the developmental v0.2 compact-output contract directory."""

    return _reviewed_resource("schemas", "compact-output", "v0", directory=True)


__all__ = [
    "canonical_dataset_schema_snapshot_resource",
    "canonical_schema_snapshot_resource",
    "compact_output_contract_resource",
    "default_config_resource",
    "development_dataset_config_resource",
    "golden_suite_resource",
    "phase4_smoke_config_resource",
    "phase5_pilot_config_resource",
    "phase6_main_config_resource",
    "phase6_remediation_development_dataset_config_resource",
    "phase6_remediation_final_dataset_config_resource",
    "phase6_remediation_pipeline_config_resource",
    "phase6_remediation_runbook_resource",
    "phase6_remediation_script_resource",
    "phase6_remediation_v02_config_resource",
    "phase6_remediation_v03_config_resource",
    "phase6_remediation_v04_config_resource",
    "phase6_v02_inventory_report_resource",
    "phase6_v03_counterfactual_cap_report_resource",
]
