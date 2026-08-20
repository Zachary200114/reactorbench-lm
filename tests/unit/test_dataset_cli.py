from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel, ConfigDict

import reactorbench.dataset.cli as cli_module
from reactorbench.dataset.artifacts import ArtifactWriter
from reactorbench.dataset.cli import _read_json, build_parser, main
from reactorbench.dataset.config import DevelopmentDatasetConfig
from reactorbench.dataset.quality import QualityRecord, TaskShortcutRecord
from reactorbench.schemas.enums import SplitName, TaskName

_HASH = "a" * 64
_SOURCE_CONFIG = Path("configs/dataset/development-v0.1.0.toml")


class _AuditSummary(BaseModel):
    trajectory_count: int
    projection_count: int


@dataclass(frozen=True)
class _FakeSplitManifest:
    checksum_sha256: str = _HASH


@dataclass(frozen=True)
class _FakePacket:
    renderer_version: str = "0.1.0"


@dataclass(frozen=True)
class _FakeRecord:
    reviewer_role: str = "project-owner"


@dataclass(frozen=True)
class _FakeQualityReport:
    passed: bool = True
    report_sha256: str = _HASH


@dataclass(frozen=True)
class _FakePostrenderPacket:
    packet_sha256: str = _HASH


@dataclass(frozen=True)
class _FakeCandidate:
    quality_report: _FakeQualityReport = _FakeQualityReport()
    postrender_review_packet: _FakePostrenderPacket = _FakePostrenderPacket()
    checksum_sha256: str = _HASH


@dataclass(frozen=True)
class _FakeManifest:
    artifact_status: str = "candidate_pending_postrender_review"

    def checksum(self) -> str:
        return _HASH


@dataclass(frozen=True)
class _FakeVerified:
    manifest: _FakeManifest = _FakeManifest()
    candidate: _FakeCandidate = _FakeCandidate()


class _StrictObject(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: int


def _project_checkout(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "checkout"
    config = root / "configs" / "dataset" / _SOURCE_CONFIG.name
    config.parent.mkdir(parents=True)
    shutil.copyfile(_SOURCE_CONFIG, config)
    (root / "src" / "reactorbench" / "dataset").mkdir(parents=True)
    (root / "schemas").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test-checkout'\n")
    return root, config


def _fake_review_reader(
    path: Path,
    model_type: type[BaseModel],
) -> BaseModel:
    del path
    if model_type.__name__ == "CatalogReviewPacket":
        return cast(BaseModel, _FakePacket())
    if model_type.__name__ == "HumanReviewRecord":
        return cast(BaseModel, _FakeRecord())
    raise AssertionError("unexpected review model")


def _full_arguments(config: Path) -> list[str]:
    return [
        "generate-full-development",
        "--config",
        str(config),
        "--review-packet",
        "catalog-review.json",
        "--review-record",
        "catalog-review-record.json",
        "--generator-commit",
        "abcdef1",
    ]


def _quality_record() -> QualityRecord:
    return QualityRecord(
        example_id="render-000000000000000000000000",
        split_name=SplitName.IID_TRAIN,
        text="bounded fictional narrative",
        template_family_id="compact-log-v1",
        alias_family_id="canonical-v1",
        target_labels=(),
        context_flags=(
            "semantic:standby-context:absent",
            "semantic:dependency-map-context:absent",
            "corruption:none",
        ),
        provenance={
            "dataset_version": "0.1.0",
            "generator_commit": "abcdef1",
            "scenario_schema_version": "0.1.0",
            "renderer_version": "0.1.0",
            "seed": 1000,
            "scenario_id": "scenario-test",
            "plant_variant_id": "ASTER_A",
            "fault_family_ids": (),
            "template_family_ids": ("compact-log-v1",),
            "split_name": SplitName.IID_TRAIN.value,
            "task_name": TaskName.FAULT_FAMILY.value,
        },
    )


def _task_record() -> TaskShortcutRecord:
    return TaskShortcutRecord(
        record_id="example:000000000000000000000000",
        prompt_render_ids=("render-000000000000000000000000",),
        task_name=TaskName.FAULT_FAMILY,
        template_family_id="compact-log-v1",
        alias_family_id="canonical-v1",
        target_labels=("fault_labels.0=PUMP_TRIP",),
        context_flags=(
            "semantic:standby-context:absent",
            "semantic:dependency-map-context:absent",
            "corruption:none",
        ),
    )


def test_low_level_generate_command_is_absent_and_full_command_has_no_output_root() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["generate-development"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                *_full_arguments(_SOURCE_CONFIG),
                "--output-root",
                "arbitrary-output",
            ]
        )


def test_audit_parser_requires_task_shortcut_input() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["audit", "--input", "quality.jsonl"])


@pytest.mark.parametrize("value", ["abcdef", "ABCDEF1", "abc-def1", "g" * 64, "a" * 65])
def test_development_commands_reject_noncanonical_generator_commits(value: str) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["audit-development", "--generator-commit", value])


def test_audit_requires_and_binds_explicit_typed_task_records(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    quality_path = tmp_path / "quality.jsonl"
    task_path = tmp_path / "tasks.jsonl"
    quality_path.write_text(_quality_record().model_dump_json() + "\n")
    task_path.write_text(_task_record().model_dump_json() + "\n")

    result = main(
        [
            "audit",
            "--input",
            str(quality_path),
            "--task-input",
            str(task_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["passed"] is True
    assert payload["record_count"] == 1
    assert payload["task_record_count"] == 1
    assert payload["audited_task_records"][0]["record_id"] == _task_record().record_id


def test_strict_json_reader_rejects_duplicate_keys(tmp_path: Path) -> None:
    input_path = tmp_path / "duplicate.json"
    input_path.write_text('{"value":1,"value":2}')

    with pytest.raises(ValueError, match="strict contract"):
        _read_json(input_path, _StrictObject)


def test_audit_development_is_read_only_and_emits_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, config_path = _project_checkout(tmp_path)

    def fake_build(config: DevelopmentDatasetConfig, *, generator_commit: str) -> object:
        assert isinstance(config, DevelopmentDatasetConfig)
        assert generator_commit == "abcdef1"
        return type(
            "AuditBundle",
            (),
            {
                "artifact_status": "structured_projection_audit",
                "dataset_config_sha256": _HASH,
                "generator_commit": generator_commit,
                "checksum_sha256": "b" * 64,
                "split_manifest": _FakeSplitManifest(),
                "summary": _AuditSummary(trajectory_count=204, projection_count=1762),
            },
        )()

    monkeypatch.setattr(cli_module, "build_development_projection_bundle", fake_build)
    result = main(
        [
            "audit-development",
            "--config",
            str(config_path),
            "--generator-commit",
            "abcdef1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["summary"] == {"projection_count": 1762, "trajectory_count": 204}
    assert not (tmp_path / "checkout" / "data").exists()


def test_prepare_review_builds_and_binds_the_exact_structured_graph(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, config_path = _project_checkout(tmp_path)
    structured = object()
    calls: list[str] = []

    def fake_build(config: DevelopmentDatasetConfig, *, generator_commit: str) -> object:
        assert isinstance(config, DevelopmentDatasetConfig)
        assert generator_commit == "abcdef1"
        calls.append("build")
        return structured

    def fake_prepare(value: object) -> dict[str, object]:
        assert value is structured
        calls.append("prepare")
        return {"structured_binding": True}

    monkeypatch.setattr(cli_module, "build_development_projection_bundle", fake_build)
    monkeypatch.setattr(cli_module, "prepare_catalog_review_packet", fake_prepare)

    result = main(
        [
            "prepare-review",
            "--config",
            str(config_path),
            "--generator-commit",
            "abcdef1",
        ]
    )

    assert result == 0
    assert calls == ["build", "prepare"]
    assert json.loads(capsys.readouterr().out) == {"structured_binding": True}


def test_copied_or_external_config_is_rejected_before_graph_build(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = tmp_path / "configs" / "dataset" / _SOURCE_CONFIG.name
    copied.parent.mkdir(parents=True)
    shutil.copyfile(_SOURCE_CONFIG, copied)
    called = False

    def fail_build(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal called
        called = True
        raise AssertionError("external config must fail before build")

    monkeypatch.setattr(cli_module, "build_development_projection_bundle", fail_build)
    result = main(
        [
            "audit-development",
            "--config",
            str(copied),
            "--generator-commit",
            "abcdef1",
        ]
    )

    assert result == 2
    assert not called
    assert "valid project checkout" in capsys.readouterr().err


def test_full_generation_fails_closed_before_build_without_approval(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config_path = _project_checkout(tmp_path)
    called = False

    def reject_review(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ValueError("review packet has not been approved")

    def fail_build(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal called
        called = True
        raise AssertionError("candidate build must not start")

    monkeypatch.setattr(cli_module, "_read_json", _fake_review_reader)
    monkeypatch.setattr(cli_module, "verify_review_record", reject_review)
    monkeypatch.setattr(cli_module, "build_development_projection_bundle", fail_build)

    result = main(_full_arguments(config_path))

    assert result == 2
    assert not called
    assert "has not been approved" in capsys.readouterr().err
    assert not (root / "data").exists()


def test_full_generation_rejects_existing_fixed_project_target_before_build(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config_path = _project_checkout(tmp_path)
    target = root / "data" / "generated" / "phase3-development-v0.1.0-candidate"
    target.mkdir(parents=True)
    called = False

    def fail_build(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal called
        called = True
        raise AssertionError("candidate build must not start")

    monkeypatch.setattr(cli_module, "_read_json", _fake_review_reader)
    monkeypatch.setattr(cli_module, "verify_review_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_module, "build_development_projection_bundle", fail_build)

    result = main(_full_arguments(config_path))

    assert result == 2
    assert not called
    assert "already exists" in capsys.readouterr().err
    assert not tuple(target.iterdir())


def test_full_generation_builds_once_verifies_exact_review_and_typed_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config_path = _project_checkout(tmp_path)
    structured = object()
    candidate = _FakeCandidate()
    calls: list[str] = []

    def fake_build(config: DevelopmentDatasetConfig, *, generator_commit: str) -> object:
        assert isinstance(config, DevelopmentDatasetConfig)
        assert generator_commit == "abcdef1"
        calls.append("build")
        return structured

    def fake_gate(packet: object, record: object, *, structured_bundle: object) -> None:
        assert isinstance(packet, _FakePacket)
        assert isinstance(record, _FakeRecord)
        assert structured_bundle is structured
        calls.append("exact-review")

    def fake_candidate_build(
        config: DevelopmentDatasetConfig,
        *,
        structured_bundle: object,
        review_packet: object,
        review_record: object,
    ) -> _FakeCandidate:
        assert isinstance(config, DevelopmentDatasetConfig)
        assert structured_bundle is structured
        assert isinstance(review_packet, _FakePacket)
        assert isinstance(review_record, _FakeRecord)
        calls.append("candidate")
        return candidate

    def fake_write(
        writer: ArtifactWriter,
        *,
        relative_directory: str,
        candidate: object,
    ) -> _FakeVerified:
        assert writer.root == root
        assert relative_directory == ("data/generated/phase3-development-v0.1.0-candidate")
        assert candidate is candidate_fixture
        calls.append("write-and-typed-verify")
        return _FakeVerified()

    candidate_fixture = candidate
    monkeypatch.setattr(cli_module, "_read_json", _fake_review_reader)
    monkeypatch.setattr(cli_module, "verify_review_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_module, "build_development_projection_bundle", fake_build)
    monkeypatch.setattr(cli_module, "verify_catalog_review_gate", fake_gate)
    monkeypatch.setattr(
        cli_module,
        "build_review_gated_development_candidate",
        fake_candidate_build,
    )
    monkeypatch.setattr(cli_module, "write_and_verify_development_candidate", fake_write)

    result = main(_full_arguments(config_path))

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert calls == ["build", "exact-review", "candidate", "write-and-typed-verify"]
    assert payload == {
        "artifact_directory": "data/generated/phase3-development-v0.1.0-candidate",
        "artifact_manifest_sha256": _HASH,
        "artifact_status": "candidate_pending_postrender_review",
        "candidate_bundle_sha256": _HASH,
        "postrender_review_packet_sha256": _HASH,
        "quality_report_sha256": _HASH,
        "verified": True,
    }


def test_verify_uses_config_selected_project_relative_typed_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config_path = _project_checkout(tmp_path)

    def fake_verify(
        writer: ArtifactWriter,
        *,
        relative_directory: str,
    ) -> _FakeVerified:
        assert writer.root == root
        assert relative_directory == "data/generated/phase3-development-v0.1.0-candidate"
        return _FakeVerified()

    monkeypatch.setattr(cli_module, "verify_development_candidate_artifact", fake_verify)
    result = main(["verify", "--config", str(config_path)])

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["verified"] is True
    assert payload["artifact_directory"].startswith("data/generated/")


def test_review_input_file_must_be_local_bounded_and_not_a_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "input.json"
    target.write_text('{"value":1}')
    linked = tmp_path / "linked.json"
    linked.symlink_to(target)

    with pytest.raises(ValueError, match="non-symlink"):
        _read_json(linked, _StrictObject)
