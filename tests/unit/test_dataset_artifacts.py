from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

import reactorbench.dataset.artifacts as artifacts_module
from reactorbench.dataset.artifacts import (
    MAX_ARTIFACT_FILES,
    ArtifactError,
    ArtifactExistsError,
    ArtifactFile,
    ArtifactModelSpec,
    ArtifactVerificationError,
    ArtifactWriter,
    CandidateArtifactManifest,
    CandidateArtifactMetadata,
)
from reactorbench.schemas.base import canonical_json_bytes

_HASH = "a" * 64


class ExampleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    identifier: str
    value: int


def _metadata(**updates: object) -> CandidateArtifactMetadata:
    payload: dict[str, object] = {
        "dataset_version": "0.1.0",
        "dataset_contract_version": "0.1.0",
        "renderer_version": "0.1.0",
        "schema_version": "0.1.0",
        "generator_commit": "abcdef1",
        "candidate_bundle_sha256": _HASH,
        "structured_bundle_sha256": _HASH,
        "resolved_config_sha256": _HASH,
        "split_manifest_sha256": _HASH,
        "aster_schema_snapshot_sha256": _HASH,
        "dataset_schema_snapshot_sha256": _HASH,
        "catalog_sha256": _HASH,
        "guard_sha256": _HASH,
        "pre_render_review_packet_sha256": _HASH,
        "pre_render_review_record_sha256": _HASH,
        "postrender_review_packet_sha256": _HASH,
        "quality_report_sha256": _HASH,
    }
    payload.update(updates)
    return CandidateArtifactMetadata.model_validate(payload)


def _write(writer: ArtifactWriter) -> CandidateArtifactManifest:
    return writer.write_candidate_bundle(
        relative_directory="development/candidate-a",
        records_by_filename={"records.jsonl": (ExampleRecord(identifier="a", value=1),)},
        metadata=_metadata(),
    )


def _rewrite_manifest_for_payload(bundle: Path, filename: str, payload: bytes) -> None:
    manifest_path = bundle / "manifest.json"
    original = CandidateArtifactManifest.model_validate_json(manifest_path.read_bytes())
    files = tuple(
        ArtifactFile(
            filename=item.filename,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            record_count=len(payload.splitlines()),
        )
        if item.filename == filename
        else item
        for item in original.files
    )
    updated = original.model_copy(update={"files": files})
    manifest_path.write_bytes(canonical_json_bytes(updated.model_dump(mode="json")) + b"\n")


def test_candidate_writer_is_atomic_checksummed_typed_and_non_overwriting(
    tmp_path: Path,
) -> None:
    writer = ArtifactWriter(tmp_path)

    _write(writer)
    verified = writer.verify_typed_candidate_bundle(
        relative_directory="development/candidate-a",
        expected_files={"records.jsonl": ArtifactModelSpec(ExampleRecord, 1, 1)},
    )

    assert verified.manifest.artifact_status == "candidate_pending_postrender_review"
    assert verified.manifest.files[0].record_count == 1
    assert verified.records_for("records.jsonl", ExampleRecord) == (
        ExampleRecord(identifier="a", value=1),
    )
    assert not tuple((tmp_path / "development").glob(".candidate-a.tmp-*"))
    with pytest.raises(ArtifactExistsError, match="exists"):
        _write(writer)


def test_verifier_detects_payload_tampering_and_inventory_changes(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path)
    _write(writer)
    bundle = tmp_path / "development" / "candidate-a"

    (bundle / "records.jsonl").write_text('{"identifier":"tampered","value":1}\n')
    with pytest.raises(ArtifactVerificationError, match="checksum"):
        writer.verify_candidate_bundle(relative_directory="development/candidate-a")


def test_verifier_rejects_duplicate_keys_even_with_recomputed_outer_checksum(
    tmp_path: Path,
) -> None:
    writer = ArtifactWriter(tmp_path)
    _write(writer)
    bundle = tmp_path / "development" / "candidate-a"
    payload = b'{"identifier":"a","identifier":"b","value":1}\n'
    (bundle / "records.jsonl").write_bytes(payload)
    _rewrite_manifest_for_payload(bundle, "records.jsonl", payload)

    with pytest.raises(ArtifactVerificationError, match="invalid JSON"):
        writer.verify_candidate_bundle(relative_directory="development/candidate-a")


def test_typed_verifier_rejects_wrong_inventory_and_contract(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path)
    _write(writer)

    with pytest.raises(ArtifactVerificationError, match="inventory"):
        writer.verify_typed_candidate_bundle(
            relative_directory="development/candidate-a",
            expected_files={"other.jsonl": ArtifactModelSpec(ExampleRecord)},
        )

    class DifferentRecord(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)

        identifier: str
        different: int

    with pytest.raises(ArtifactVerificationError, match="expected contract"):
        writer.verify_typed_candidate_bundle(
            relative_directory="development/candidate-a",
            expected_files={"records.jsonl": ArtifactModelSpec(DifferentRecord, 1, 1)},
        )


def test_writer_and_verifier_enforce_record_and_manifest_byte_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = ArtifactWriter(tmp_path)
    monkeypatch.setattr(artifacts_module, "MAX_ARTIFACT_RECORD_BYTES", 8)
    with pytest.raises(ArtifactError, match="record exceeds"):
        writer.write_candidate_bundle(
            relative_directory="oversize-record",
            records_by_filename={"records.jsonl": ({"value": "too-large"},)},
            metadata=_metadata(),
        )

    monkeypatch.setattr(artifacts_module, "MAX_ARTIFACT_RECORD_BYTES", 8 * 1024 * 1024)
    _write(writer)
    manifest = tmp_path / "development" / "candidate-a" / "manifest.json"
    monkeypatch.setattr(artifacts_module, "MAX_ARTIFACT_MANIFEST_BYTES", 8)
    with pytest.raises(ArtifactVerificationError, match="manifest exceeds"):
        writer.verify_candidate_bundle(relative_directory="development/candidate-a")
    assert manifest.stat().st_size > 8


def test_writer_enforces_file_count_limit_before_creating_output(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path)
    records = {
        f"record-{index:02d}.jsonl": ({"index": index},) for index in range(MAX_ARTIFACT_FILES + 1)
    }

    with pytest.raises(ArtifactError, match="file count"):
        writer.write_candidate_bundle(
            relative_directory="too-many-files",
            records_by_filename=records,
            metadata=_metadata(),
        )
    assert not (tmp_path / "too-many-files").exists()


def test_verifier_rejects_noncanonical_manifest_encoding(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path)
    _write(writer)
    manifest = tmp_path / "development" / "candidate-a" / "manifest.json"
    manifest.write_text(f" {manifest.read_text(encoding='utf-8')}", encoding="utf-8")

    with pytest.raises(ArtifactVerificationError, match="canonical"):
        writer.verify_candidate_bundle(relative_directory="development/candidate-a")


@pytest.mark.parametrize(
    "relative",
    [
        "../escape",
        "/absolute",
        ".",
        "development/../escape",
        "bad\\path",
        "./development",
        "development//candidate",
        "development/",
    ],
)
def test_writer_rejects_traversal_and_ambiguous_relative_paths(
    tmp_path: Path, relative: str
) -> None:
    writer = ArtifactWriter(tmp_path)

    with pytest.raises(ArtifactError):
        writer.write_candidate_bundle(
            relative_directory=relative,
            records_by_filename={"records.jsonl": ({"id": "a"},)},
            metadata=_metadata(),
        )


def test_writer_rejects_symlink_roots_parents_and_verifier_payloads(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ArtifactError, match="non-symlink"):
        ArtifactWriter(linked)

    writer = ArtifactWriter(real)
    outside = tmp_path / "outside"
    outside.mkdir()
    (real / "development").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactError, match="symlink"):
        _write(writer)

    (real / "development").unlink()
    _write(writer)
    payload = real / "development" / "candidate-a" / "records.jsonl"
    payload.unlink()
    payload.symlink_to(outside / "records.jsonl")
    with pytest.raises(ArtifactVerificationError, match="symlink"):
        writer.verify_candidate_bundle(relative_directory="development/candidate-a")


def test_metadata_manifest_carries_every_required_provenance_binding(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path)
    manifest = _write(writer)
    payload = manifest.model_dump(mode="json")

    expected_hash_fields = {
        "candidate_bundle_sha256",
        "structured_bundle_sha256",
        "resolved_config_sha256",
        "split_manifest_sha256",
        "aster_schema_snapshot_sha256",
        "dataset_schema_snapshot_sha256",
        "catalog_sha256",
        "guard_sha256",
        "pre_render_review_packet_sha256",
        "pre_render_review_record_sha256",
        "postrender_review_packet_sha256",
        "quality_report_sha256",
    }
    assert {name for name in payload if name.endswith("_sha256")} == expected_hash_fields
    assert json.loads((tmp_path / "development/candidate-a/manifest.json").read_text()) == payload
