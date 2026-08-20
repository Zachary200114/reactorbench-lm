"""Deterministic JSON Schema snapshots for public Phase 3 dataset contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Never, cast

from pydantic import BaseModel

from reactorbench.dataset.contracts import (
    DATASET_CONTRACT_VERSION,
    CounterfactualProjectionRecord,
    ProjectionRecord,
)
from reactorbench.dataset.splits import SplitManifest
from reactorbench.schemas.base import SCHEMA_VERSION, canonical_json_bytes, canonical_sha256

DATASET_SCHEMA_MODELS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("projection-record.schema.json", ProjectionRecord),
    ("counterfactual-projection-record.schema.json", CounterfactualProjectionRecord),
    ("split-manifest.schema.json", SplitManifest),
)
DATASET_MANIFEST_FILENAME = "manifest.json"
DATASET_SNAPSHOT_CONTRACT_FILENAME = "snapshot-contract.json"
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

_DATASET_SCHEMA_NAMESPACE = "urn:reactorbench:dataset"
_EXPECTED_SCHEMA_FILENAMES = tuple(filename for filename, _model in DATASET_SCHEMA_MODELS)
_EXPECTED_SCHEMA_FILENAME_SET = frozenset(_EXPECTED_SCHEMA_FILENAMES)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_contract_version",
        "hash_algorithm",
        "snapshot_contract_sha256",
        "files",
        "snapshot_hash",
    }
)
_SNAPSHOT_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_contract_version",
        "status",
        "frozen",
        "schema_dialect",
        "schema_namespace",
        "root_exports",
    }
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

if not _EXPECTED_SCHEMA_FILENAMES or len(_EXPECTED_SCHEMA_FILENAMES) != len(
    _EXPECTED_SCHEMA_FILENAME_SET
):
    raise RuntimeError("dataset schema export roots must be non-empty and unique")


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
        raise ValueError(f"cannot read dataset schema snapshot file: {path.name}") from exc

    try:
        decoded: object = json.loads(
            raw_document.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_constant,
        )
    except _DuplicateJsonKeyError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON in dataset schema snapshot file: {path.name}") from exc

    if not isinstance(decoded, dict):
        raise ValueError(f"dataset schema snapshot file must contain an object: {path.name}")
    document = cast(dict[str, Any], decoded)
    try:
        canonical_document = canonical_json_bytes(document) + b"\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid JSON value in dataset schema snapshot file: {path.name}"
        ) from exc
    if raw_document != canonical_document:
        raise ValueError(f"dataset schema snapshot file is not canonical JSON: {path.name}")
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
        raise ValueError("dataset schema snapshot filenames must be normalized basenames")


def _validate_required_schema_filenames(filenames: set[str]) -> None:
    for filename in filenames:
        _validate_snapshot_filename(filename)
    if filenames != _EXPECTED_SCHEMA_FILENAME_SET:
        missing = sorted(_EXPECTED_SCHEMA_FILENAME_SET - filenames)
        unexpected = sorted(filenames - _EXPECTED_SCHEMA_FILENAME_SET)
        raise ValueError(
            "dataset schema snapshot files must exactly match required roots "
            f"(missing={missing!r}, unexpected={unexpected!r})"
        )


def _validate_sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase hexadecimal SHA-256 digest")
    return value


def dataset_snapshot_contract() -> dict[str, Any]:
    """Describe the exact developmental dataset roots represented by this snapshot."""

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "status": "developmental",
        "frozen": False,
        "schema_dialect": JSON_SCHEMA_DIALECT,
        "schema_namespace": _DATASET_SCHEMA_NAMESPACE,
        "root_exports": [
            {"filename": filename, "model": model.__name__}
            for filename, model in DATASET_SCHEMA_MODELS
        ],
    }


def _validate_snapshot_contract(contract: dict[str, Any]) -> None:
    if set(contract) != _SNAPSHOT_CONTRACT_FIELDS:
        missing = sorted(_SNAPSHOT_CONTRACT_FIELDS - set(contract))
        unexpected = sorted(set(contract) - _SNAPSHOT_CONTRACT_FIELDS)
        raise ValueError(
            "dataset snapshot contract fields do not match the contract "
            f"(missing={missing!r}, unexpected={unexpected!r})"
        )
    if contract != dataset_snapshot_contract():
        raise ValueError("dataset snapshot contract does not match the supported export roots")


def _validate_manifest(manifest: dict[str, Any], contract: dict[str, Any]) -> dict[str, str]:
    if set(manifest) != _MANIFEST_FIELDS:
        missing = sorted(_MANIFEST_FIELDS - set(manifest))
        unexpected = sorted(set(manifest) - _MANIFEST_FIELDS)
        raise ValueError(
            "dataset schema manifest fields do not match the contract "
            f"(missing={missing!r}, unexpected={unexpected!r})"
        )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("dataset schema manifest has an unsupported schema_version")
    if manifest["dataset_contract_version"] != DATASET_CONTRACT_VERSION:
        raise ValueError("dataset schema manifest has an unsupported dataset_contract_version")
    if manifest["hash_algorithm"] != "sha256":
        raise ValueError("dataset schema manifest hash_algorithm must be 'sha256'")

    raw_file_hashes = manifest["files"]
    if not isinstance(raw_file_hashes, dict) or not all(
        isinstance(filename, str) for filename in raw_file_hashes
    ):
        raise ValueError("dataset schema manifest files must be a JSON object")
    file_hashes = cast(dict[str, Any], raw_file_hashes)
    _validate_required_schema_filenames(set(file_hashes))
    validated_hashes = {
        filename: _validate_sha256(file_hashes[filename], field_name=f"files[{filename!r}]")
        for filename in _EXPECTED_SCHEMA_FILENAMES
    }
    contract_hash = _validate_sha256(
        manifest["snapshot_contract_sha256"], field_name="snapshot_contract_sha256"
    )
    if canonical_sha256(contract) != contract_hash:
        raise ValueError("dataset schema manifest snapshot contract checksum does not match")
    snapshot_hash = _validate_sha256(manifest["snapshot_hash"], field_name="snapshot_hash")
    if (
        canonical_sha256({"files": validated_hashes, "snapshot_contract_sha256": contract_hash})
        != snapshot_hash
    ):
        raise ValueError("dataset schema manifest snapshot_hash does not match snapshot content")
    return validated_hashes


def _validate_schema_document_root(filename: str, document: dict[str, Any]) -> None:
    expected_fields: dict[str, object] = {
        "$id": f"{_DATASET_SCHEMA_NAMESPACE}:{DATASET_CONTRACT_VERSION}:{filename}",
        "$schema": JSON_SCHEMA_DIALECT,
        "additionalProperties": False,
        "type": "object",
    }
    for field_name, expected_value in expected_fields.items():
        actual_value = document.get(field_name)
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            raise ValueError(
                f"dataset schema document {filename!r} has an invalid root {field_name!r}"
            )


def _contained_snapshot_file(root: Path, filename: str) -> Path:
    try:
        candidate = (root / filename).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"dataset schema snapshot file is unavailable: {filename}") from exc
    if not candidate.is_relative_to(root):
        raise ValueError(f"dataset schema snapshot file escapes its directory: {filename}")
    if not candidate.is_file():
        raise ValueError(f"dataset schema snapshot path is not a regular file: {filename}")
    return candidate


def dataset_schema_documents() -> dict[str, dict[str, Any]]:
    """Build every public dataset schema document in deterministic file order."""

    documents: dict[str, dict[str, Any]] = {}
    for filename, model in DATASET_SCHEMA_MODELS:
        document = model.model_json_schema(mode="validation")
        document["$id"] = f"{_DATASET_SCHEMA_NAMESPACE}:{DATASET_CONTRACT_VERSION}:{filename}"
        document["$schema"] = JSON_SCHEMA_DIALECT
        documents[filename] = document
    return documents


def dataset_snapshot_manifest(
    documents: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe and bind the schema documents and exact root-export contract."""

    selected_documents = dataset_schema_documents() if documents is None else documents
    selected_contract = dataset_snapshot_contract() if contract is None else contract
    _validate_required_schema_filenames(set(selected_documents))
    file_hashes = {
        filename: canonical_sha256(document)
        for filename, document in sorted(selected_documents.items())
    }
    contract_hash = canonical_sha256(selected_contract)
    snapshot_content = {
        "files": file_hashes,
        "snapshot_contract_sha256": contract_hash,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "hash_algorithm": "sha256",
        "snapshot_contract_sha256": contract_hash,
        "files": file_hashes,
        "snapshot_hash": canonical_sha256(snapshot_content),
    }


def export_dataset_json_schemas(destination: Path) -> dict[str, Any]:
    """Write canonical dataset schema snapshots and return their manifest."""

    destination.mkdir(parents=True, exist_ok=True)
    documents = dataset_schema_documents()
    contract = dataset_snapshot_contract()
    for filename, document in documents.items():
        (destination / filename).write_bytes(canonical_json_bytes(document) + b"\n")
    (destination / DATASET_SNAPSHOT_CONTRACT_FILENAME).write_bytes(
        canonical_json_bytes(contract) + b"\n"
    )
    manifest = dataset_snapshot_manifest(documents, contract=contract)
    (destination / DATASET_MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def load_dataset_snapshot(
    directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load a canonical dataset snapshot after validating paths, roots, and hashes."""

    try:
        root = directory.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("dataset schema snapshot directory is unavailable") from exc
    if not root.is_dir():
        raise ValueError("dataset schema snapshot path is not a directory")

    contract_path = _contained_snapshot_file(root, DATASET_SNAPSHOT_CONTRACT_FILENAME)
    contract = _load_canonical_json_object(contract_path)
    _validate_snapshot_contract(contract)
    manifest_path = _contained_snapshot_file(root, DATASET_MANIFEST_FILENAME)
    manifest = _load_canonical_json_object(manifest_path)
    file_hashes = _validate_manifest(manifest, contract)

    documents: dict[str, dict[str, Any]] = {}
    for filename in _EXPECTED_SCHEMA_FILENAMES:
        document_path = _contained_snapshot_file(root, filename)
        document = _load_canonical_json_object(document_path)
        _validate_schema_document_root(filename, document)
        if canonical_sha256(document) != file_hashes[filename]:
            raise ValueError(f"dataset schema snapshot checksum does not match: {filename}")
        documents[filename] = document

    if dataset_snapshot_manifest(documents, contract=contract) != manifest:
        raise ValueError("dataset schema manifest does not match its documents")
    return documents, manifest, contract


__all__ = [
    "DATASET_MANIFEST_FILENAME",
    "DATASET_SCHEMA_MODELS",
    "DATASET_SNAPSHOT_CONTRACT_FILENAME",
    "dataset_schema_documents",
    "dataset_snapshot_contract",
    "dataset_snapshot_manifest",
    "export_dataset_json_schemas",
    "load_dataset_snapshot",
]
