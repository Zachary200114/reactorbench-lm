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
