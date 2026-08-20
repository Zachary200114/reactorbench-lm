"""Bounded, non-overwriting candidate artifacts with typed verification."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, model_validator

from reactorbench.schemas.base import ContractModel, SemanticVersion, canonical_json_bytes

ARTIFACT_FORMAT_VERSION: SemanticVersion = "0.1.0"
MANIFEST_FILENAME = "manifest.json"

MAX_ARTIFACT_FILES = 32
MAX_ARTIFACT_FILE_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_TOTAL_BYTES = 256 * 1024 * 1024
MAX_ARTIFACT_MANIFEST_BYTES = 1024 * 1024
MAX_ARTIFACT_RECORD_BYTES = 8 * 1024 * 1024
MAX_ARTIFACT_RECORDS_PER_FILE = 10_000
MAX_ARTIFACT_TOTAL_RECORDS = 50_000


class ArtifactError(RuntimeError):
    """Base class for safe artifact failures."""


class ArtifactExistsError(ArtifactError):
    """Raised instead of overwriting an existing bundle."""


class ArtifactVerificationError(ArtifactError):
    """Raised when an artifact does not match its manifest and expected contracts."""


class ArtifactFile(ContractModel):
    filename: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\.jsonl$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0, le=MAX_ARTIFACT_FILE_BYTES)
    record_count: int = Field(ge=0, le=MAX_ARTIFACT_RECORDS_PER_FILE)


class CandidateArtifactMetadata(ContractModel):
    """Exact provenance bindings duplicated in the outer manifest and metadata record."""

    artifact_format_version: SemanticVersion = ARTIFACT_FORMAT_VERSION
    artifact_status: Literal["candidate_pending_postrender_review"] = (
        "candidate_pending_postrender_review"
    )
    dataset_version: SemanticVersion
    dataset_contract_version: SemanticVersion
    renderer_version: SemanticVersion
    schema_version: SemanticVersion
    generator_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    candidate_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    structured_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    aster_schema_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_schema_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    guard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pre_render_review_packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pre_render_review_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    postrender_review_packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CandidateArtifactManifest(CandidateArtifactMetadata):
    files: tuple[ArtifactFile, ...] = Field(min_length=1, max_length=MAX_ARTIFACT_FILES)

    @model_validator(mode="after")
    def files_are_bounded_sorted_and_unique(self) -> CandidateArtifactManifest:
        names = tuple(item.filename for item in self.files)
        if names != tuple(sorted(names)):
            raise ValueError("artifact files must be sorted by filename")
        if len(names) != len(set(names)):
            raise ValueError("artifact filenames must be unique")
        if MANIFEST_FILENAME in names:
            raise ValueError("manifest cannot list itself as a JSONL payload")
        if sum(item.size_bytes for item in self.files) > MAX_ARTIFACT_TOTAL_BYTES:
            raise ValueError("artifact payload bytes exceed the total limit")
        if sum(item.record_count for item in self.files) > MAX_ARTIFACT_TOTAL_RECORDS:
            raise ValueError("artifact records exceed the total limit")
        return self

    def checksum(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


@dataclass(frozen=True)
class ArtifactModelSpec:
    """Expected model and cardinality for one typed JSONL payload."""

    model_type: type[BaseModel]
    minimum_records: int = 0
    maximum_records: int = MAX_ARTIFACT_RECORDS_PER_FILE

    def __post_init__(self) -> None:
        if not isinstance(self.model_type, type) or not issubclass(self.model_type, BaseModel):
            raise TypeError("artifact model spec requires a Pydantic model type")
        if type(self.minimum_records) is not int or type(self.maximum_records) is not int:
            raise TypeError("artifact model record bounds must be integers")
        if not 0 <= self.minimum_records <= self.maximum_records:
            raise ValueError("artifact model record bounds are invalid")
        if self.maximum_records > MAX_ARTIFACT_RECORDS_PER_FILE:
            raise ValueError("artifact model record bound exceeds the global limit")


@dataclass(frozen=True)
class TypedArtifactFile:
    filename: str
    records: tuple[BaseModel, ...]


@dataclass(frozen=True)
class VerifiedTypedCandidateBundle:
    manifest: CandidateArtifactManifest
    files: tuple[TypedArtifactFile, ...]

    def records_for[ModelT: BaseModel](
        self, filename: str, model_type: type[ModelT]
    ) -> tuple[ModelT, ...]:
        for item in self.files:
            if item.filename != filename:
                continue
            if any(type(record) is not model_type for record in item.records):
                raise TypeError("verified artifact records use an unexpected model type")
            return cast(tuple[ModelT, ...], item.records)
        raise KeyError(filename)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in pairs:
        if key in decoded:
            raise ValueError("artifact JSON contains a duplicate object key")
        decoded[key] = value
    return decoded


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"artifact JSON contains a non-finite constant: {value}")


def _strict_json_object(value: bytes, *, description: str) -> dict[str, Any]:
    try:
        decoded = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ArtifactVerificationError(f"{description} is invalid JSON") from error
    if type(decoded) is not dict:
        raise ArtifactVerificationError(f"{description} must contain a JSON object")
    return decoded


class ArtifactWriter:
    """Write candidate bundles below one trusted, pre-existing local root."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise TypeError("root must be a pathlib.Path")
        if root.is_symlink() or not root.is_dir():
            raise ArtifactError("artifact root must be an existing non-symlink directory")
        self._root = root.resolve(strict=True)

    @property
    def root(self) -> Path:
        return self._root

    def _relative_path(self, value: str, *, field_name: str) -> PurePosixPath:
        if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
            raise ArtifactError(f"{field_name} must be a non-empty POSIX relative path")
        candidate = PurePosixPath(value)
        if (
            candidate.is_absolute()
            or not candidate.parts
            or candidate == PurePosixPath(".")
            or candidate.as_posix() != value
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise ArtifactError(f"{field_name} must not contain traversal or empty components")
        return candidate

    def _contained(self, relative: PurePosixPath) -> Path:
        candidate = self._root.joinpath(*relative.parts)
        cursor = self._root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ArtifactError("artifact path contains a symlink")
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(self._root):
            raise ArtifactError("artifact path escapes the configured root")
        return candidate

    def _prepare_parent(self, parent: Path) -> None:
        relative = parent.relative_to(self._root)
        cursor = self._root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.exists():
                if cursor.is_symlink() or not cursor.is_dir():
                    raise ArtifactError("artifact parent contains a symlink or non-directory")
            else:
                cursor.mkdir(mode=0o750)
        if parent.resolve(strict=True) != parent:
            raise ArtifactError("artifact parent resolution changed unexpectedly")

    def preflight_candidate_bundle(self, *, relative_directory: str) -> Path:
        """Resolve one unused destination without creating directories."""

        relative = self._relative_path(relative_directory, field_name="relative_directory")
        target = self._contained(relative)
        cursor = self._root
        for index, part in enumerate(relative.parts):
            cursor /= part
            if cursor.is_symlink():
                raise ArtifactError("artifact path contains a symlink")
            if index < len(relative.parts) - 1 and cursor.exists() and not cursor.is_dir():
                raise ArtifactError("artifact parent contains a non-directory")
        if target.exists() or target.is_symlink():
            raise ArtifactExistsError("artifact bundle already exists")
        lock = target.parent / f".{target.name}.lock"
        if lock.exists() or lock.is_symlink():
            raise ArtifactExistsError("artifact bundle is locked by another writer")
        return target

    def _json_record(self, record: BaseModel | Mapping[str, object]) -> dict[str, Any]:
        if isinstance(record, BaseModel):
            payload = record.model_dump(mode="json", round_trip=True)
        elif isinstance(record, Mapping):
            payload = dict(record)
        else:
            raise ArtifactError("JSONL records must be Pydantic models or mappings")
        if not isinstance(payload, dict):
            raise ArtifactError("JSONL record must serialize to an object")
        return payload

    def _jsonl_bytes(
        self, records: Sequence[BaseModel | Mapping[str, object]]
    ) -> tuple[bytes, int]:
        if len(records) > MAX_ARTIFACT_RECORDS_PER_FILE:
            raise ArtifactError("artifact JSONL record count exceeds the per-file limit")
        lines: list[bytes] = []
        size = 0
        for record in records:
            try:
                line = canonical_json_bytes(self._json_record(record))
            except (TypeError, ValueError) as error:
                raise ArtifactError("record is not canonical JSON data") from error
            if len(line) > MAX_ARTIFACT_RECORD_BYTES:
                raise ArtifactError("artifact JSONL record exceeds the byte limit")
            size += len(line) + 1
            if size > MAX_ARTIFACT_FILE_BYTES:
                raise ArtifactError("artifact JSONL file exceeds the byte limit")
            lines.append(line)
        return (b"\n".join(lines) + (b"\n" if lines else b""), len(lines))

    def write_candidate_bundle(
        self,
        *,
        relative_directory: str,
        records_by_filename: Mapping[str, Sequence[BaseModel | Mapping[str, object]]],
        metadata: CandidateArtifactMetadata,
    ) -> CandidateArtifactManifest:
        """Write a complete candidate bundle by adjacent temp directory and rename."""

        if type(metadata) is not CandidateArtifactMetadata:
            raise TypeError("metadata must be an exact CandidateArtifactMetadata")
        target = self.preflight_candidate_bundle(relative_directory=relative_directory)
        if not records_by_filename:
            raise ArtifactError("candidate bundle requires at least one JSONL file")
        if len(records_by_filename) > MAX_ARTIFACT_FILES:
            raise ArtifactError("artifact file count exceeds the limit")

        encoded: dict[str, tuple[bytes, int]] = {}
        total_bytes = 0
        total_records = 0
        for filename, records in sorted(records_by_filename.items()):
            relative_file = self._relative_path(filename, field_name="filename")
            if len(relative_file.parts) != 1 or not filename.endswith(".jsonl"):
                raise ArtifactError("bundle JSONL filenames must be flat .jsonl names")
            payload, record_count = self._jsonl_bytes(records)
            total_bytes += len(payload)
            total_records += record_count
            if total_bytes > MAX_ARTIFACT_TOTAL_BYTES:
                raise ArtifactError("artifact payload bytes exceed the total limit")
            if total_records > MAX_ARTIFACT_TOTAL_RECORDS:
                raise ArtifactError("artifact records exceed the total limit")
            encoded[filename] = (payload, record_count)

        self._prepare_parent(target.parent)
        lock = target.parent / f".{target.name}.lock"
        lock_descriptor: int | None = None
        temp: Path | None = None
        try:
            try:
                lock_descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as error:
                raise ArtifactExistsError("artifact bundle is locked by another writer") from error
            os.write(lock_descriptor, b"candidate-write-lock\n")
            os.fsync(lock_descriptor)
            if target.exists() or target.is_symlink():
                raise ArtifactExistsError("artifact bundle already exists")

            temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
            file_records: list[ArtifactFile] = []
            for filename, (payload, record_count) in encoded.items():
                output = temp / filename
                with output.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                file_records.append(
                    ArtifactFile(
                        filename=filename,
                        sha256=hashlib.sha256(payload).hexdigest(),
                        size_bytes=len(payload),
                        record_count=record_count,
                    )
                )

            manifest = CandidateArtifactManifest(
                **metadata.model_dump(mode="python", round_trip=True),
                files=tuple(sorted(file_records, key=lambda item: item.filename)),
            )
            manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
            if len(manifest_bytes) > MAX_ARTIFACT_MANIFEST_BYTES:
                raise ArtifactError("artifact manifest exceeds the byte limit")
            with (temp / MANIFEST_FILENAME).open("xb") as stream:
                stream.write(manifest_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            directory_descriptor = os.open(temp, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            if target.exists() or target.is_symlink():
                raise ArtifactExistsError("artifact bundle appeared during write")
            os.rename(temp, target)
            temp = None
            parent_descriptor = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
            return manifest
        finally:
            if lock_descriptor is not None:
                os.close(lock_descriptor)
            if temp is not None and temp.exists():
                shutil.rmtree(temp)
            if lock_descriptor is not None:
                try:
                    lock.unlink(missing_ok=True)
                except OSError:
                    pass

    def _verified_payloads(
        self, *, relative_directory: str
    ) -> tuple[CandidateArtifactManifest, dict[str, tuple[bytes, ...]]]:
        relative = self._relative_path(relative_directory, field_name="relative_directory")
        bundle = self._contained(relative)
        if bundle.is_symlink() or not bundle.is_dir():
            raise ArtifactVerificationError("artifact bundle is missing or is a symlink")
        if bundle.resolve(strict=True) != bundle:
            raise ArtifactVerificationError("artifact bundle resolution changed")
        manifest_path = bundle / MANIFEST_FILENAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ArtifactVerificationError("artifact manifest is missing or is a symlink")
        if manifest_path.stat().st_size > MAX_ARTIFACT_MANIFEST_BYTES:
            raise ArtifactVerificationError("artifact manifest exceeds the byte limit")
        manifest_bytes = manifest_path.read_bytes()
        manifest_object = _strict_json_object(manifest_bytes, description="artifact manifest")
        if canonical_json_bytes(manifest_object) + b"\n" != manifest_bytes:
            raise ArtifactVerificationError("artifact manifest is not canonical JSON")
        try:
            manifest = CandidateArtifactManifest.model_validate_json(manifest_bytes)
        except ValueError as error:
            raise ArtifactVerificationError("artifact manifest violates its contract") from error

        paths: list[Path] = []
        for path in bundle.iterdir():
            paths.append(path)
            if len(paths) > MAX_ARTIFACT_FILES + 1:
                raise ArtifactVerificationError("artifact file count exceeds the limit")
        expected = {MANIFEST_FILENAME, *(item.filename for item in manifest.files)}
        actual = {path.name for path in paths}
        if actual != expected:
            raise ArtifactVerificationError("artifact file inventory does not match manifest")

        payloads: dict[str, tuple[bytes, ...]] = {}
        total_bytes = 0
        total_records = 0
        for item in manifest.files:
            payload_path = bundle / item.filename
            if payload_path.is_symlink() or not payload_path.is_file():
                raise ArtifactVerificationError("artifact payload is missing or is a symlink")
            if payload_path.stat().st_size > MAX_ARTIFACT_FILE_BYTES:
                raise ArtifactVerificationError("artifact payload exceeds the per-file byte limit")
            payload = payload_path.read_bytes()
            total_bytes += len(payload)
            if total_bytes > MAX_ARTIFACT_TOTAL_BYTES:
                raise ArtifactVerificationError("artifact payload bytes exceed the total limit")
            if (
                len(payload) != item.size_bytes
                or hashlib.sha256(payload).hexdigest() != item.sha256
            ):
                raise ArtifactVerificationError("artifact payload checksum mismatch")
            if payload and not payload.endswith(b"\n"):
                raise ArtifactVerificationError("artifact JSONL payload lacks its final newline")
            lines = tuple(payload.splitlines())
            if len(lines) != item.record_count:
                raise ArtifactVerificationError("artifact JSONL record count mismatch")
            total_records += len(lines)
            if total_records > MAX_ARTIFACT_TOTAL_RECORDS:
                raise ArtifactVerificationError("artifact records exceed the total limit")
            if b"\n".join(lines) + (b"\n" if lines else b"") != payload:
                raise ArtifactVerificationError("artifact JSONL line endings are not canonical")
            for line in lines:
                if len(line) > MAX_ARTIFACT_RECORD_BYTES:
                    raise ArtifactVerificationError("artifact JSONL record exceeds the byte limit")
                decoded = _strict_json_object(line, description="artifact JSONL record")
                if canonical_json_bytes(decoded) != line:
                    raise ArtifactVerificationError("artifact JSONL payload is not canonical")
            payloads[item.filename] = lines
        return manifest, payloads

    def verify_candidate_bundle(self, *, relative_directory: str) -> CandidateArtifactManifest:
        """Verify bounded paths, exact inventory, canonical JSONL, sizes, and hashes."""

        manifest, _payloads = self._verified_payloads(relative_directory=relative_directory)
        return manifest

    def verify_typed_candidate_bundle(
        self,
        *,
        relative_directory: str,
        expected_files: Mapping[str, ArtifactModelSpec],
    ) -> VerifiedTypedCandidateBundle:
        """Verify an exact file inventory and parse every record through its strict model."""

        if not expected_files or len(expected_files) > MAX_ARTIFACT_FILES:
            raise ValueError("expected typed artifact files must be nonempty and bounded")
        if any(type(spec) is not ArtifactModelSpec for spec in expected_files.values()):
            raise TypeError("expected files must use exact ArtifactModelSpec values")
        manifest, payloads = self._verified_payloads(relative_directory=relative_directory)
        if set(payloads) != set(expected_files):
            raise ArtifactVerificationError("typed artifact inventory does not match expectations")
        typed_files: list[TypedArtifactFile] = []
        for filename, spec in sorted(expected_files.items()):
            lines = payloads[filename]
            if not spec.minimum_records <= len(lines) <= spec.maximum_records:
                raise ArtifactVerificationError("typed artifact record count violates expectations")
            records: list[BaseModel] = []
            for line in lines:
                try:
                    record = spec.model_type.model_validate_json(line)
                except ValueError as error:
                    raise ArtifactVerificationError(
                        "typed artifact record violates its expected contract"
                    ) from error
                if type(record) is not spec.model_type:
                    raise ArtifactVerificationError("typed artifact returned a derived model type")
                records.append(record)
            typed_files.append(TypedArtifactFile(filename=filename, records=tuple(records)))
        return VerifiedTypedCandidateBundle(manifest=manifest, files=tuple(typed_files))


__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "MANIFEST_FILENAME",
    "MAX_ARTIFACT_FILES",
    "MAX_ARTIFACT_FILE_BYTES",
    "MAX_ARTIFACT_MANIFEST_BYTES",
    "MAX_ARTIFACT_RECORDS_PER_FILE",
    "MAX_ARTIFACT_RECORD_BYTES",
    "MAX_ARTIFACT_TOTAL_BYTES",
    "MAX_ARTIFACT_TOTAL_RECORDS",
    "ArtifactError",
    "ArtifactExistsError",
    "ArtifactFile",
    "ArtifactModelSpec",
    "ArtifactVerificationError",
    "ArtifactWriter",
    "CandidateArtifactManifest",
    "CandidateArtifactMetadata",
    "TypedArtifactFile",
    "VerifiedTypedCandidateBundle",
]
