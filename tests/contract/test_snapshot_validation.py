from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from reactorbench.schemas import export_json_schemas, load_snapshot, schema_documents
from reactorbench.schemas.base import canonical_json_bytes, canonical_sha256
from reactorbench.schemas.export import MANIFEST_FILENAME, SCHEMA_MODELS, snapshot_manifest


def _export(directory: Path) -> dict[str, Any]:
    return export_json_schemas(directory)


def _file_hashes(manifest: dict[str, Any]) -> dict[str, Any]:
    file_hashes = manifest["files"]
    assert isinstance(file_hashes, dict)
    return cast(dict[str, Any], file_hashes)


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _rewrite_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    manifest["snapshot_hash"] = canonical_sha256(_file_hashes(manifest))
    _write_canonical(directory / MANIFEST_FILENAME, manifest)


def test_load_snapshot_rejects_an_empty_file_set(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = _export(snapshot)
    manifest["files"] = {}
    _rewrite_manifest(snapshot, manifest)

    with pytest.raises(ValueError, match="exactly match required roots"):
        load_snapshot(snapshot)


def test_load_snapshot_rejects_a_partial_file_set(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = _export(snapshot)
    file_hashes = _file_hashes(manifest)
    file_hashes.pop(next(iter(file_hashes)))
    _rewrite_manifest(snapshot, manifest)

    with pytest.raises(ValueError, match="exactly match required roots"):
        load_snapshot(snapshot)


def test_load_snapshot_rejects_an_unexpected_file(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = _export(snapshot)
    filename = next(iter(_file_hashes(manifest)))
    unexpected_filename = "unexpected.schema.json"
    (snapshot / unexpected_filename).write_bytes((snapshot / filename).read_bytes())
    _file_hashes(manifest)[unexpected_filename] = _file_hashes(manifest)[filename]
    _rewrite_manifest(snapshot, manifest)

    with pytest.raises(ValueError, match="exactly match required roots"):
        load_snapshot(snapshot)


def test_load_snapshot_rejects_traversal_before_reading_the_target(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = _export(snapshot)
    filename = next(iter(_file_hashes(manifest)))
    outside = tmp_path / "outside.schema.json"
    outside.write_bytes((snapshot / filename).read_bytes())
    digest = _file_hashes(manifest).pop(filename)
    _file_hashes(manifest)["../outside.schema.json"] = digest
    _rewrite_manifest(snapshot, manifest)

    with pytest.raises(ValueError, match="normalized basenames"):
        load_snapshot(snapshot)


def test_load_snapshot_rejects_a_schema_symlink_escape(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = _export(snapshot)
    filename = next(iter(_file_hashes(manifest)))
    schema_path = snapshot / filename
    outside = tmp_path / "outside.schema.json"
    outside.write_bytes(schema_path.read_bytes())
    schema_path.unlink()
    schema_path.symlink_to(outside)

    with pytest.raises(ValueError, match="escapes its directory"):
        load_snapshot(snapshot)


def test_load_snapshot_rejects_a_duplicate_manifest_filename(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = _export(snapshot)
    filename = next(iter(_file_hashes(manifest)))
    encoded_entry = json.dumps(filename) + ":" + json.dumps(_file_hashes(manifest)[filename])
    manifest_path = snapshot / MANIFEST_FILENAME
    raw_manifest = manifest_path.read_text("utf-8")
    manifest_path.write_text(
        raw_manifest.replace(encoded_entry, f"{encoded_entry},{encoded_entry}", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_snapshot(snapshot)


def test_load_snapshot_rejects_duplicate_keys_inside_a_schema(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = _export(snapshot)
    filename = next(iter(_file_hashes(manifest)))
    schema_path = snapshot / filename
    raw_schema = schema_path.read_text("utf-8")
    schema_id = f"urn:reactorbench:aster:0.1.0:{filename}"
    duplicate_prefix = json.dumps("$id") + ":" + json.dumps(schema_id) + ","
    schema_path.write_text(raw_schema.replace("{", "{" + duplicate_prefix, 1), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_snapshot(snapshot)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("schema_version", None, "unsupported schema_version"),
        ("schema_version", "0.2.0", "unsupported schema_version"),
        ("hash_algorithm", 256, "hash_algorithm"),
        ("hash_algorithm", "sha512", "hash_algorithm"),
        ("snapshot_hash", 0, "lowercase hexadecimal SHA-256"),
        ("snapshot_hash", "A" * 64, "lowercase hexadecimal SHA-256"),
    ],
)
def test_load_snapshot_rejects_invalid_manifest_value_types_and_literals(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
    message: str,
) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = _export(snapshot)
    manifest[field_name] = invalid_value
    _write_canonical(snapshot / MANIFEST_FILENAME, manifest)

    with pytest.raises(ValueError, match=message):
        load_snapshot(snapshot)


@pytest.mark.parametrize("extra_field", [True, False])
def test_load_snapshot_requires_exact_manifest_fields(
    tmp_path: Path,
    *,
    extra_field: bool,
) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = _export(snapshot)
    if extra_field:
        manifest["unexpected"] = "rejected"
    else:
        manifest.pop("schema_version")
    _write_canonical(snapshot / MANIFEST_FILENAME, manifest)

    with pytest.raises(ValueError, match="manifest fields do not match"):
        load_snapshot(snapshot)


@pytest.mark.parametrize("invalid_hash", [False, 7, "f" * 63, "F" * 64])
def test_load_snapshot_requires_string_sha256_file_hashes(
    tmp_path: Path,
    invalid_hash: object,
) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = _export(snapshot)
    filename = next(iter(_file_hashes(manifest)))
    _file_hashes(manifest)[filename] = invalid_hash
    _write_canonical(snapshot / MANIFEST_FILENAME, manifest)

    with pytest.raises(ValueError, match="lowercase hexadecimal SHA-256"):
        load_snapshot(snapshot)


def test_load_snapshot_rejects_a_non_object_files_field(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = _export(snapshot)
    manifest["files"] = []
    _write_canonical(snapshot / MANIFEST_FILENAME, manifest)

    with pytest.raises(ValueError, match="files must be a JSON object"):
        load_snapshot(snapshot)


def test_load_snapshot_requires_canonical_json_bytes(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = _export(snapshot)
    (snapshot / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="not canonical JSON"):
        load_snapshot(snapshot)


def test_load_snapshot_rejects_a_self_hashed_partial_schema_root(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = _export(snapshot)
    filename = next(iter(_file_hashes(manifest)))
    partial_document: dict[str, object] = {}
    _write_canonical(snapshot / filename, partial_document)
    _file_hashes(manifest)[filename] = canonical_sha256(partial_document)
    _rewrite_manifest(snapshot, manifest)

    with pytest.raises(ValueError, match="invalid root"):
        load_snapshot(snapshot)


def test_load_snapshot_rejects_a_document_checksum_mismatch(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = _export(snapshot)
    filename = next(iter(_file_hashes(manifest)))
    document = schema_documents()[filename]
    document["description"] = "tampered"
    _write_canonical(snapshot / filename, document)

    with pytest.raises(ValueError, match="checksum does not match"):
        load_snapshot(snapshot)


def test_snapshot_manifest_rejects_partial_export_roots() -> None:
    filename, _model = SCHEMA_MODELS[0]

    with pytest.raises(ValueError, match="exactly match required roots"):
        snapshot_manifest({filename: schema_documents()[filename]})
