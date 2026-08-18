"""Deterministic JSON Schema export and snapshot helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Never, cast

from pydantic import BaseModel

from .base import SCHEMA_VERSION, canonical_json_bytes, canonical_sha256
from .events import CanonicalEvent
from .latent import LatentPlantState
from .observation import ObservationFrame
from .provenance import ProvenanceRecord
from .scenario import ScenarioDefinition
from .target import TaskTarget
from .trajectory import StructuredTrajectory

SCHEMA_MODELS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("latent-state.schema.json", LatentPlantState),
    ("observation.schema.json", ObservationFrame),
    ("canonical-event.schema.json", CanonicalEvent),
    ("scenario.schema.json", ScenarioDefinition),
    ("target.schema.json", TaskTarget),
    ("provenance.schema.json", ProvenanceRecord),
    ("structured-trajectory.schema.json", StructuredTrajectory),
)
MANIFEST_FILENAME = "manifest.json"
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

_EXPECTED_SCHEMA_FILENAMES = tuple(filename for filename, _model in SCHEMA_MODELS)
_EXPECTED_SCHEMA_FILENAME_SET = frozenset(_EXPECTED_SCHEMA_FILENAMES)
_MANIFEST_FIELDS = frozenset({"schema_version", "hash_algorithm", "files", "snapshot_hash"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

if not _EXPECTED_SCHEMA_FILENAMES or len(_EXPECTED_SCHEMA_FILENAMES) != len(
    _EXPECTED_SCHEMA_FILENAME_SET
):
    raise RuntimeError("schema export roots must be non-empty and unique")


class _DuplicateJsonKeyError(ValueError):
    """Raised when strict JSON decoding encounters a duplicate object key."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite_json_constant(value: str) -> Never:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _load_canonical_json_object(path: Path) -> dict[str, Any]:
    try:
        raw_document = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read schema snapshot file: {path.name}") from exc

    try:
        decoded: object = json.loads(
            raw_document.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_constant,
        )
    except _DuplicateJsonKeyError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON in schema snapshot file: {path.name}") from exc

    if not isinstance(decoded, dict):
        raise ValueError(f"schema snapshot file must contain a JSON object: {path.name}")
    document = cast(dict[str, Any], decoded)
    try:
        canonical_document = canonical_json_bytes(document) + b"\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid JSON value in schema snapshot file: {path.name}") from exc
    if raw_document != canonical_document:
        raise ValueError(f"schema snapshot file is not canonical JSON: {path.name}")
    return document


def _validate_snapshot_filename(filename: str) -> None:
    posix_path = PurePosixPath(filename)
    windows_path = PureWindowsPath(filename)
    if (
        not filename
        or filename.strip() != filename
        or "\x00" in filename
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or posix_path.name != filename
        or windows_path.name != filename
    ):
        raise ValueError("schema snapshot filenames must be normalized basenames")


def _validate_required_schema_filenames(filenames: set[str]) -> None:
    for filename in filenames:
        _validate_snapshot_filename(filename)

    if filenames != _EXPECTED_SCHEMA_FILENAME_SET:
        missing = sorted(_EXPECTED_SCHEMA_FILENAME_SET - filenames)
        unexpected = sorted(filenames - _EXPECTED_SCHEMA_FILENAME_SET)
        raise ValueError(
            "schema snapshot files must exactly match required roots "
            f"(missing={missing!r}, unexpected={unexpected!r})"
        )


def _validate_sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase hexadecimal SHA-256 digest")
    return value


def _validate_manifest(manifest: dict[str, Any]) -> dict[str, str]:
    fields = set(manifest)
    if fields != _MANIFEST_FIELDS:
        missing = sorted(_MANIFEST_FIELDS - fields)
        unexpected = sorted(fields - _MANIFEST_FIELDS)
        raise ValueError(
            "schema snapshot manifest fields do not match the contract "
            f"(missing={missing!r}, unexpected={unexpected!r})"
        )

    if not isinstance(manifest["schema_version"], str) or (
        manifest["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("schema snapshot manifest has an unsupported schema_version")
    if not isinstance(manifest["hash_algorithm"], str) or (manifest["hash_algorithm"] != "sha256"):
        raise ValueError("schema snapshot manifest hash_algorithm must be 'sha256'")

    raw_file_hashes = manifest["files"]
    if not isinstance(raw_file_hashes, dict) or not all(
        isinstance(filename, str) for filename in raw_file_hashes
    ):
        raise ValueError("schema snapshot manifest files must be a JSON object")
    file_hashes = cast(dict[str, Any], raw_file_hashes)
    _validate_required_schema_filenames(set(file_hashes))

    validated_hashes = {
        filename: _validate_sha256(file_hashes[filename], field_name=f"files[{filename!r}]")
        for filename in _EXPECTED_SCHEMA_FILENAMES
    }
    snapshot_hash = _validate_sha256(manifest["snapshot_hash"], field_name="snapshot_hash")
    if canonical_sha256(validated_hashes) != snapshot_hash:
        raise ValueError("schema snapshot manifest snapshot_hash does not match files")
    return validated_hashes


def _validate_schema_document_root(filename: str, document: dict[str, Any]) -> None:
    expected_fields: dict[str, object] = {
        "$id": f"urn:reactorbench:aster:{SCHEMA_VERSION}:{filename}",
        "$schema": JSON_SCHEMA_DIALECT,
        "additionalProperties": False,
        "type": "object",
    }
    for field_name, expected_value in expected_fields.items():
        actual_value = document.get(field_name)
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            raise ValueError(
                f"schema snapshot document {filename!r} has an invalid root {field_name!r}"
            )


def _contained_snapshot_file(root: Path, filename: str) -> Path:
    try:
        candidate = (root / filename).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"schema snapshot file is missing or inaccessible: {filename}") from exc
    if not candidate.is_relative_to(root):
        raise ValueError(f"schema snapshot file escapes its directory: {filename}")
    if not candidate.is_file():
        raise ValueError(f"schema snapshot path is not a regular file: {filename}")
    return candidate


def schema_documents() -> dict[str, dict[str, Any]]:
    """Build all versioned schema documents in deterministic file order."""

    documents: dict[str, dict[str, Any]] = {}
    for filename, model in SCHEMA_MODELS:
        document = model.model_json_schema(mode="validation")
        document["$id"] = f"urn:reactorbench:aster:{SCHEMA_VERSION}:{filename}"
        document["$schema"] = JSON_SCHEMA_DIALECT
        documents[filename] = document
    return documents


def snapshot_manifest(
    documents: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Describe and hash the complete schema snapshot."""

    selected = schema_documents() if documents is None else documents
    _validate_required_schema_filenames(set(selected))
    file_hashes = {
        filename: canonical_sha256(document) for filename, document in sorted(selected.items())
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "hash_algorithm": "sha256",
        "files": file_hashes,
        "snapshot_hash": canonical_sha256(file_hashes),
    }


def export_json_schemas(destination: Path) -> dict[str, Any]:
    """Write canonical JSON Schema snapshots and return their manifest."""

    destination.mkdir(parents=True, exist_ok=True)
    documents = schema_documents()
    for filename, document in documents.items():
        (destination / filename).write_bytes(canonical_json_bytes(document) + b"\n")

    manifest = snapshot_manifest(documents)
    (destination / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def load_snapshot(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a canonical snapshot after validating its structure, paths, and hashes."""

    try:
        root = directory.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("schema snapshot directory is missing or inaccessible") from exc
    if not root.is_dir():
        raise ValueError("schema snapshot path is not a directory")

    manifest_path = _contained_snapshot_file(root, MANIFEST_FILENAME)
    manifest = _load_canonical_json_object(manifest_path)
    file_hashes = _validate_manifest(manifest)

    documents: dict[str, dict[str, Any]] = {}
    for filename in _EXPECTED_SCHEMA_FILENAMES:
        document_path = _contained_snapshot_file(root, filename)
        document = _load_canonical_json_object(document_path)
        _validate_schema_document_root(filename, document)
        if canonical_sha256(document) != file_hashes[filename]:
            raise ValueError(f"schema snapshot checksum does not match: {filename}")
        documents[filename] = document

    if snapshot_manifest(documents) != manifest:
        raise ValueError("schema snapshot manifest does not match its documents")
    return documents, manifest
