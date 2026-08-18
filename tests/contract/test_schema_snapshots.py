from __future__ import annotations

import json
from pathlib import Path

from reactorbench.schemas import (
    SCHEMA_VERSION,
    export_json_schemas,
    load_snapshot,
    schema_documents,
    snapshot_manifest,
)
from reactorbench.schemas.export import SCHEMA_MODELS

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIRECTORY = ROOT / "schemas" / "aster" / "v0"


def test_committed_snapshot_contract_matches_export_roots() -> None:
    contract = json.loads((SNAPSHOT_DIRECTORY / "snapshot-contract.json").read_text("utf-8"))
    assert contract["schema_version"] == SCHEMA_VERSION == "0.1.0"
    assert contract["status"] == "developmental"
    assert contract["frozen"] is False
    assert contract["root_exports"] == [
        {"filename": filename, "model": model.__name__} for filename, model in SCHEMA_MODELS
    ]


def test_schema_documents_and_manifest_are_deterministic() -> None:
    first_documents = schema_documents()
    second_documents = schema_documents()
    assert first_documents == second_documents
    assert snapshot_manifest(first_documents) == snapshot_manifest(second_documents)

    for filename, document in first_documents.items():
        assert document["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert document["$id"].endswith(filename)
        assert document["additionalProperties"] is False


def test_committed_generated_snapshot_matches_the_models() -> None:
    documents, manifest = load_snapshot(SNAPSHOT_DIRECTORY)
    assert documents == schema_documents()
    assert manifest == snapshot_manifest(documents)


def test_exported_snapshots_are_byte_stable_and_self_verifying(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = export_json_schemas(first)
    second_manifest = export_json_schemas(second)

    assert first_manifest == second_manifest
    for filename in (*first_manifest["files"], "manifest.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    documents, loaded_manifest = load_snapshot(first)
    assert loaded_manifest == first_manifest
    assert documents == schema_documents()
