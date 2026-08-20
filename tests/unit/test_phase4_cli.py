"""Safe, narrow Phase 4 CLI tests."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

import reactorbench.training.cli as cli_module
from reactorbench.training.cli import main

SOURCE_CONFIG = Path("configs/model/phase4-smoke-v0.1.0.toml")


@dataclass(frozen=True)
class _Report:
    checkpoint_manifest_sha256: str = "1" * 64
    final_loss: float = 0.1
    checksum_sha256: str = "2" * 64
    run_status: str = "phase4_smoke_passed"
    tokenizer_manifest_sha256: str = "3" * 64


def _checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    config = checkout / SOURCE_CONFIG
    config.parent.mkdir(parents=True)
    shutil.copyfile(SOURCE_CONFIG, config)
    return checkout


def test_run_cli_passes_the_exact_config_and_source_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkout = _checkout(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(config: object, *, project_root: Path, source_commit: str) -> _Report:
        captured.update(
            config=config,
            project_root=project_root,
            source_commit=source_commit,
        )
        return _Report()

    monkeypatch.chdir(checkout)
    monkeypatch.setattr(cli_module, "run_phase4_smoke", fake_run)

    result = main(
        [
            "run-smoke",
            "--config",
            str(SOURCE_CONFIG),
            "--source-commit",
            "abcdef0",
        ]
    )

    assert result == 0
    assert captured["project_root"] == checkout
    assert captured["source_commit"] == "abcdef0"
    assert json.loads(capsys.readouterr().out)["run_status"] == "phase4_smoke_passed"


def test_cli_returns_a_safe_error_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkout = _checkout(tmp_path)

    def fail(*args: object, **kwargs: object) -> _Report:
        del args, kwargs
        raise ValueError("bounded failure")

    monkeypatch.chdir(checkout)
    monkeypatch.setattr(cli_module, "verify_phase4_run", fail)

    result = main(["verify-smoke", "--config", str(SOURCE_CONFIG)])
    captured = capsys.readouterr()

    assert result == 2
    assert captured.out == ""
    assert captured.err == "error: bounded failure\n"
    assert "Traceback" not in captured.err


def test_cli_rejects_a_config_outside_the_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkout = _checkout(tmp_path)
    outside = tmp_path / "outside.toml"
    shutil.copyfile(SOURCE_CONFIG, outside)
    monkeypatch.chdir(checkout)

    assert main(["verify-smoke", "--config", str(outside)]) == 2
    assert "inside the current project checkout" in capsys.readouterr().err


def test_golden_review_cli_requires_explicit_owner_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    packet_path = Path("golden/golden-suite-v0.1.0.json")
    record_path = Path("artifacts/review/golden-review-record-v0.1.0.json")

    assert (
        main(
            [
                "prepare-golden-review",
                "--generator-commit",
                "4473718",
                "--output",
                str(packet_path),
            ]
        )
        == 0
    )
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["case_count"] == 15
    assert prepared["review_status"] == "pending_project_owner_review"

    arguments = [
        "record-golden-review",
        "--packet",
        str(packet_path),
        "--output",
        str(record_path),
        "--review-date",
        "2026-08-20",
        "--decision",
        "APPROVED",
    ]
    assert main(arguments) == 2
    assert "explicit --confirm-all" in capsys.readouterr().err
    assert main([*arguments, "--confirm-all", "--note", "test-only owner confirmation"]) == 0
    recorded = json.loads(capsys.readouterr().out)
    assert recorded["decision"] == "APPROVED"

    assert (
        main(
            [
                "verify-golden-review",
                "--packet",
                str(packet_path),
                "--record",
                str(record_path),
                "--expected-packet-sha256",
                prepared["packet_sha256"],
            ]
        )
        == 0
    )
    verified = json.loads(capsys.readouterr().out)
    assert verified["review_status"] == "golden_suite_approved"
