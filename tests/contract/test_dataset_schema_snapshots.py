from __future__ import annotations

import json
from pathlib import Path

import pytest

from reactorbench.dataset.contracts import DATASET_CONTRACT_VERSION
from reactorbench.dataset.schema_export import (
    DATASET_MANIFEST_FILENAME,
    DATASET_SCHEMA_MODELS,
    DATASET_SNAPSHOT_CONTRACT_FILENAME,
    dataset_schema_documents,
    dataset_snapshot_contract,
    dataset_snapshot_manifest,
    export_dataset_json_schemas,
    load_dataset_snapshot,
)
from reactorbench.schemas.base import SCHEMA_VERSION, canonical_json_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIRECTORY = ROOT / "schemas" / "dataset" / "v0"


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def test_dataset_snapshot_contract_fixes_the_exact_public_roots() -> None:
    contract = dataset_snapshot_contract()
    assert contract == {
        "schema_version": SCHEMA_VERSION,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "status": "developmental",
        "frozen": False,
        "schema_dialect": "https://json-schema.org/draft/2020-12/schema",
        "schema_namespace": "urn:reactorbench:dataset",
        "root_exports": [
            {"filename": filename, "model": model.__name__}
            for filename, model in DATASET_SCHEMA_MODELS
        ],
    }


def test_dataset_schema_documents_and_manifest_are_deterministic() -> None:
    first_documents = dataset_schema_documents()
    second_documents = dataset_schema_documents()
    assert first_documents == second_documents
    assert dataset_snapshot_manifest(first_documents) == dataset_snapshot_manifest(second_documents)
    assert tuple(first_documents) == tuple(filename for filename, _model in DATASET_SCHEMA_MODELS)
    for filename, document in first_documents.items():
        assert document["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert document["$id"] == f"urn:reactorbench:dataset:0.1.0:{filename}"
        assert document["additionalProperties"] is False
        assert document["type"] == "object"


def test_committed_dataset_snapshot_matches_public_contracts() -> None:
    documents, manifest, contract = load_dataset_snapshot(SNAPSHOT_DIRECTORY)
    assert documents == dataset_schema_documents()
    assert contract == dataset_snapshot_contract()
    assert manifest == dataset_snapshot_manifest(documents, contract=contract)


def test_dataset_snapshot_export_is_byte_stable_and_self_verifying(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = export_dataset_json_schemas(first)
    second_manifest = export_dataset_json_schemas(second)

    assert first_manifest == second_manifest
    filenames = (
        *first_manifest["files"],
        DATASET_MANIFEST_FILENAME,
        DATASET_SNAPSHOT_CONTRACT_FILENAME,
    )
    for filename in filenames:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert load_dataset_snapshot(first) == load_dataset_snapshot(second)


def test_dataset_snapshot_manifest_binds_the_root_descriptor() -> None:
    manifest = dataset_snapshot_manifest()
    contract_hash = canonical_sha256(dataset_snapshot_contract())
    assert manifest["snapshot_contract_sha256"] == contract_hash
    assert manifest["snapshot_hash"] == canonical_sha256(
        {"files": manifest["files"], "snapshot_contract_sha256": contract_hash}
    )


def test_dataset_snapshot_rejects_descriptor_drift_even_when_rehashed(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = export_dataset_json_schemas(snapshot)
    contract_path = snapshot / DATASET_SNAPSHOT_CONTRACT_FILENAME
    contract = json.loads(contract_path.read_text("utf-8"))
    contract["frozen"] = True
    _write_canonical(contract_path, contract)
    contract_hash = canonical_sha256(contract)
    manifest["snapshot_contract_sha256"] = contract_hash
    manifest["snapshot_hash"] = canonical_sha256(
        {"files": manifest["files"], "snapshot_contract_sha256": contract_hash}
    )
    _write_canonical(snapshot / DATASET_MANIFEST_FILENAME, manifest)

    with pytest.raises(ValueError, match="supported export roots"):
        load_dataset_snapshot(snapshot)


def test_dataset_snapshot_rejects_an_unexpected_manifest_root(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = export_dataset_json_schemas(snapshot)
    filename = next(iter(manifest["files"]))
    manifest["files"]["unexpected.schema.json"] = manifest["files"][filename]
    manifest["snapshot_hash"] = canonical_sha256(
        {
            "files": manifest["files"],
            "snapshot_contract_sha256": manifest["snapshot_contract_sha256"],
        }
    )
    _write_canonical(snapshot / DATASET_MANIFEST_FILENAME, manifest)

    with pytest.raises(ValueError, match="exactly match required roots"):
        load_dataset_snapshot(snapshot)


def test_dataset_snapshot_rejects_a_manifest_path_traversal(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = export_dataset_json_schemas(snapshot)
    filename = next(iter(manifest["files"]))
    digest = manifest["files"].pop(filename)
    manifest["files"]["../outside.schema.json"] = digest
    manifest["snapshot_hash"] = canonical_sha256(
        {
            "files": manifest["files"],
            "snapshot_contract_sha256": manifest["snapshot_contract_sha256"],
        }
    )
    _write_canonical(snapshot / DATASET_MANIFEST_FILENAME, manifest)

    with pytest.raises(ValueError, match="normalized basenames"):
        load_dataset_snapshot(snapshot)


def test_dataset_snapshot_rejects_noncanonical_json(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    export_dataset_json_schemas(snapshot)
    manifest_path = snapshot / DATASET_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="not canonical JSON"):
        load_dataset_snapshot(snapshot)


def test_dataset_snapshot_rejects_a_duplicate_descriptor_key(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    export_dataset_json_schemas(snapshot)
    contract_path = snapshot / DATASET_SNAPSHOT_CONTRACT_FILENAME
    raw = contract_path.read_text("utf-8")
    contract_path.write_text(raw.replace("{", '{"frozen":false,', 1), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_dataset_snapshot(snapshot)


def test_dataset_snapshot_rejects_a_schema_symlink_escape(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = export_dataset_json_schemas(snapshot)
    filename = next(iter(manifest["files"]))
    schema_path = snapshot / filename
    outside = tmp_path / "outside.schema.json"
    outside.write_bytes(schema_path.read_bytes())
    schema_path.unlink()
    schema_path.symlink_to(outside)

    with pytest.raises(ValueError, match="escapes its directory"):
        load_dataset_snapshot(snapshot)


def test_dataset_snapshot_rejects_a_schema_checksum_mismatch(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = export_dataset_json_schemas(snapshot)
    filename = next(iter(manifest["files"]))
    schema_path = snapshot / filename
    document = json.loads(schema_path.read_text("utf-8"))
    document["description"] = "tampered"
    _write_canonical(schema_path, document)

    with pytest.raises(ValueError, match="checksum does not match"):
        load_dataset_snapshot(snapshot)
