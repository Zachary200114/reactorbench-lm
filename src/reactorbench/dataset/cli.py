"""Local-only Phase 3 audit, review, generation, and verification CLI."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from reactorbench.schemas.base import canonical_json_bytes

from .artifacts import ArtifactError, ArtifactWriter
from .catalog import RENDERER_VERSION
from .config import DevelopmentDatasetConfig, load_development_dataset_config
from .development import (
    build_review_gated_development_candidate,
    verify_development_candidate_artifact,
    write_and_verify_development_candidate,
)
from .pipeline import build_development_projection_bundle
from .quality import QualityRecord, QualityReport, TaskShortcutRecord, audit_quality
from .renderer import RenderedCandidate
from .review import (
    CatalogReviewPacket,
    HumanReviewRecord,
    prepare_catalog_review_packet,
    prepare_postrender_review_packet,
    verify_catalog_review_gate,
    verify_review_record,
)

_MAX_INPUT_BYTES = 32 * 1024 * 1024
_MAX_RECORDS = 10_000
_DEFAULT_DEVELOPMENT_CONFIG = Path("configs/dataset/development-v0.1.0.toml")
_GIT_COMMIT = re.compile(r"[0-9a-f]{7,64}\Z")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON input contains a duplicate object key")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"JSON input contains a non-finite constant: {value}")


def _strict_json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(
        value,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonfinite_json_constant,
    )
    if type(parsed) is not dict:
        raise ValueError("JSON input must contain one object")
    return parsed


def _trusted_input(path: Path) -> Path:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise ValueError("input must be an existing non-symlink local file")
    resolved = path.resolve(strict=True)
    if resolved.stat().st_size > _MAX_INPUT_BYTES:
        raise ValueError("input exceeds the local CLI byte limit")
    return resolved


def _validate_strict_json_model[ModelT: BaseModel](
    text: str,
    model_type: type[ModelT],
) -> ModelT:
    payload = _strict_json_object(text)
    return model_type.model_validate_json(canonical_json_bytes(payload))


def _read_jsonl[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> tuple[ModelT, ...]:
    source = _trusted_input(path)
    records: list[ModelT] = []
    with source.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line_number > _MAX_RECORDS:
                raise ValueError("JSONL input exceeds the record limit")
            if not line.strip():
                raise ValueError(f"JSONL line {line_number} is empty")
            try:
                records.append(_validate_strict_json_model(line, model_type))
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                raise ValueError(
                    f"JSONL line {line_number} violates its strict contract"
                ) from error
    if not records:
        raise ValueError("JSONL input must contain at least one record")
    return tuple(records)


def _read_json[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> ModelT:
    source = _trusted_input(path)
    try:
        return _validate_strict_json_model(source.read_text(encoding="utf-8"), model_type)
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        raise ValueError("JSON input violates its strict contract") from error


def _print_json(value: BaseModel | dict[str, object]) -> None:
    payload: Any = (
        value.model_dump(mode="json", round_trip=True) if isinstance(value, BaseModel) else value
    )
    sys.stdout.buffer.write(canonical_json_bytes(payload) + b"\n")


def _generator_commit(value: str) -> str:
    if not _GIT_COMMIT.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "generator commit must be 7-64 lowercase hexadecimal characters"
        )
    return value


def _load_project_config(path: Path) -> tuple[DevelopmentDatasetConfig, Path]:
    source = _trusted_input(path)
    if (
        source.suffix != ".toml"
        or source.parent.name != "dataset"
        or source.parent.parent.name != "configs"
    ):
        raise ValueError("development config must be under <project>/configs/dataset")
    project_root = source.parent.parent.parent
    required_file = project_root / "pyproject.toml"
    required_directories = (
        project_root / "src" / "reactorbench" / "dataset",
        project_root / "schemas",
    )
    if required_file.is_symlink() or not required_file.is_file():
        raise ValueError("development config is not inside a valid project checkout")
    if any(path.is_symlink() or not path.is_dir() for path in required_directories):
        raise ValueError("development config is not inside a valid project checkout")
    return load_development_dataset_config(source), project_root


def _candidate_relative_directory(config: DevelopmentDatasetConfig) -> str:
    return f"data/generated/{config.dataset.artifact_name}"


def _command_audit(arguments: argparse.Namespace) -> int:
    records = _read_jsonl(arguments.input, QualityRecord)
    task_records = _read_jsonl(arguments.task_input, TaskShortcutRecord)
    report = audit_quality(records, task_records=task_records)
    _print_json(report)
    return 0 if report.passed else 3


def _command_prepare_review(arguments: argparse.Namespace) -> int:
    config, _project_root = _load_project_config(arguments.config)
    structured = build_development_projection_bundle(
        config,
        generator_commit=arguments.generator_commit,
    )
    _print_json(prepare_catalog_review_packet(structured))
    return 0


def _command_prepare_postrender_review(arguments: argparse.Namespace) -> int:
    candidates = _read_jsonl(arguments.candidates, RenderedCandidate)
    report = _read_json(arguments.quality_report, QualityReport)
    packet = prepare_postrender_review_packet(candidates, quality_report=report)
    _print_json(packet)
    return 0


def _command_audit_development(arguments: argparse.Namespace) -> int:
    config, _project_root = _load_project_config(arguments.config)
    bundle = build_development_projection_bundle(
        config,
        generator_commit=arguments.generator_commit,
    )
    _print_json(
        {
            "artifact_status": bundle.artifact_status,
            "dataset_config_sha256": bundle.dataset_config_sha256,
            "generator_commit": bundle.generator_commit,
            "projection_bundle_sha256": bundle.checksum_sha256,
            "split_manifest_sha256": bundle.split_manifest.checksum_sha256,
            "summary": bundle.summary.model_dump(mode="json", round_trip=True),
        }
    )
    return 0


def _command_generate_full_development(arguments: argparse.Namespace) -> int:
    config, project_root = _load_project_config(arguments.config)
    packet = _read_json(arguments.review_packet, CatalogReviewPacket)
    record = _read_json(arguments.review_record, HumanReviewRecord)
    if config.dataset.renderer_version != RENDERER_VERSION:
        raise ValueError("dataset config renderer version does not match the installed renderer")
    if packet.renderer_version != config.dataset.renderer_version:
        raise ValueError("catalog review packet renderer version differs from resolved config")
    if record.reviewer_role != config.review.reviewer_role:
        raise ValueError("catalog review record does not use the configured reviewer role")

    # Reject stale/unapproved review surfaces and an occupied destination before the
    # comparatively expensive structured graph build. The exact graph binding is then
    # checked before any narrative is rendered.
    verify_review_record(packet, record, require_approved=True)
    writer = ArtifactWriter(project_root)
    relative_directory = _candidate_relative_directory(config)
    writer.preflight_candidate_bundle(relative_directory=relative_directory)
    structured = build_development_projection_bundle(
        config,
        generator_commit=arguments.generator_commit,
    )
    verify_catalog_review_gate(packet, record, structured_bundle=structured)
    candidate = build_review_gated_development_candidate(
        config,
        structured_bundle=structured,
        review_packet=packet,
        review_record=record,
    )
    if not candidate.quality_report.passed:
        raise ValueError("automated quality audit failed; candidate artifacts were not written")
    verified = write_and_verify_development_candidate(
        writer,
        relative_directory=relative_directory,
        candidate=candidate,
    )
    _print_json(
        {
            "artifact_directory": relative_directory,
            "artifact_manifest_sha256": verified.manifest.checksum(),
            "artifact_status": verified.manifest.artifact_status,
            "candidate_bundle_sha256": verified.candidate.checksum_sha256,
            "postrender_review_packet_sha256": (
                verified.candidate.postrender_review_packet.packet_sha256
            ),
            "quality_report_sha256": verified.candidate.quality_report.report_sha256,
            "verified": True,
        }
    )
    return 0


def _command_verify(arguments: argparse.Namespace) -> int:
    config, project_root = _load_project_config(arguments.config)
    relative_directory = _candidate_relative_directory(config)
    verified = verify_development_candidate_artifact(
        ArtifactWriter(project_root),
        relative_directory=relative_directory,
    )
    _print_json(
        {
            "artifact_directory": relative_directory,
            "artifact_manifest_sha256": verified.manifest.checksum(),
            "artifact_status": verified.manifest.artifact_status,
            "candidate_bundle_sha256": verified.candidate.checksum_sha256,
            "verified": True,
        }
    )
    return 0


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_DEVELOPMENT_CONFIG,
    )


def _add_generator_commit_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--generator-commit",
        required=True,
        type=_generator_commit,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reactorbench.dataset",
        description="Local candidate dataset tooling; no network services are used.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit",
        help="audit strict narrative QualityRecord and task-shortcut JSONL together",
    )
    audit.add_argument("--input", required=True, type=Path)
    audit.add_argument("--task-input", required=True, type=Path)
    audit.set_defaults(handler=_command_audit)

    review = subparsers.add_parser(
        "prepare-review",
        help="build the exact structured graph and emit its mandatory pre-render review packet",
    )
    _add_config_argument(review)
    _add_generator_commit_argument(review)
    review.set_defaults(handler=_command_prepare_review)

    postrender_review = subparsers.add_parser(
        "prepare-postrender-review",
        help="emit a separate full packet for rendered candidates and their audit",
    )
    postrender_review.add_argument("--candidates", required=True, type=Path)
    postrender_review.add_argument("--quality-report", required=True, type=Path)
    postrender_review.set_defaults(handler=_command_prepare_postrender_review)

    development_audit = subparsers.add_parser(
        "audit-development",
        help="build and summarize the deterministic structured development graph without writes",
    )
    _add_config_argument(development_audit)
    _add_generator_commit_argument(development_audit)
    development_audit.set_defaults(handler=_command_audit_development)

    full_development = subparsers.add_parser(
        "generate-full-development",
        help=(
            "build, audit, write, and typed-verify the complete candidate after exact "
            "project-owner review"
        ),
    )
    _add_config_argument(full_development)
    full_development.add_argument("--review-packet", required=True, type=Path)
    full_development.add_argument("--review-record", required=True, type=Path)
    _add_generator_commit_argument(full_development)
    full_development.set_defaults(handler=_command_generate_full_development)

    verify = subparsers.add_parser(
        "verify",
        help="typed-verify the complete config-selected candidate under this project",
    )
    _add_config_argument(verify)
    verify.set_defaults(handler=_command_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a command with bounded safe errors and no traceback disclosure."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        handler = arguments.handler
        if not callable(handler):
            raise ValueError("invalid command handler")
        return int(handler(arguments))
    except (ArtifactError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
