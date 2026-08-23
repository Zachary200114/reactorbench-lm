from __future__ import annotations

import hashlib
import inspect
import json
import tomllib
from pathlib import Path

from reactorbench.evaluation.compact import (
    COMPACT_TARGET_VERSION,
    COMPACT_WIRE_PREFIX,
    MAX_COMPACT_TARGET_BYTES,
    CompactTargetConstraint,
    CompactTargetContext,
    compact_output_contract,
)
from reactorbench.schemas.base import canonical_json_bytes, canonical_sha256
from reactorbench.schemas.enums import TaskName

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DIRECTORY = ROOT / "schemas" / "compact-output" / "v0"


def test_committed_compact_contract_is_canonical_and_matches_source() -> None:
    contract_path = CONTRACT_DIRECTORY / "contract.json"
    expected = compact_output_contract()
    assert contract_path.read_bytes() == canonical_json_bytes(expected) + b"\n"
    assert expected["contract_version"] == COMPACT_TARGET_VERSION
    assert expected["wire_prefix"] == COMPACT_WIRE_PREFIX
    assert expected["max_utf8_bytes"] == MAX_COMPACT_TARGET_BYTES
    assert expected["status"] == "developmental"
    assert expected["frozen"] is False
    task_fields = expected["task_fields"]
    assert isinstance(task_fields, dict)
    assert set(task_fields) == {task.value for task in TaskName}


def test_compact_manifest_binds_the_exact_non_manifest_inventory() -> None:
    manifest_path = CONTRACT_DIRECTORY / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest_path.read_bytes() == canonical_json_bytes(manifest) + b"\n"
    assert manifest["contract_version"] == COMPACT_TARGET_VERSION
    assert manifest["manifest_version"] == "0.1.0"
    expected_files = {"README.md", "contract.json"}
    assert set(manifest["files"]) == expected_files
    for filename, expected_digest in manifest["files"].items():
        payload = (CONTRACT_DIRECTORY / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_digest
    assert manifest["snapshot_sha256"] == canonical_sha256(
        {
            "files": manifest["files"],
            "contract_version": manifest["contract_version"],
        }
    )


def test_compact_schema_snapshot_is_included_in_wheel_configuration() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    forced = configuration["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert forced["schemas/compact-output/v0"] == ("reactorbench/_data/schemas/compact-output/v0")


def test_decoder_context_and_constructor_cannot_receive_hidden_truth() -> None:
    assert set(CompactTargetContext.model_fields) == {
        "contract_version",
        "task_name",
        "visible_fact_refs",
        "counterfactual_visible_fact_refs",
    }
    forbidden = {
        "target",
        "fault_labels",
        "latent_state",
        "provenance",
        "scenario_id",
        "source_event_ids",
        "future_events",
    }
    assert forbidden.isdisjoint(CompactTargetContext.model_fields)
    constructor_parameters = set(inspect.signature(CompactTargetConstraint).parameters)
    assert constructor_parameters == {"context", "maximum_generated_tokens"}
    next_token_parameters = set(
        inspect.signature(CompactTargetConstraint.allowed_next_token_ids).parameters
    )
    assert next_token_parameters == {"self", "tokenizer", "generated_token_ids"}
