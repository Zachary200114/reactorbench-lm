"""Narrow local CLI for the Phase 4 smoke milestone."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reactorbench.model import load_phase4_config

from .smoke import run_phase4_smoke, verify_phase4_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reactorbench-phase4")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-smoke")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--source-commit", required=True)
    verify = subparsers.add_parser("verify-smoke")
    verify.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        project_root = Path.cwd().resolve(strict=True)
        config_path = args.config.resolve(strict=True)
        if not config_path.is_relative_to(project_root):
            raise ValueError("Phase 4 config must be inside the current project checkout")
        config = load_phase4_config(config_path)
        if args.command == "run-smoke":
            report = run_phase4_smoke(
                config, project_root=project_root, source_commit=args.source_commit
            )
        else:
            report = verify_phase4_run(config, project_root=project_root)
        print(
            json.dumps(
                {
                    "checkpoint_manifest_sha256": report.checkpoint_manifest_sha256,
                    "final_loss": report.final_loss,
                    "report_sha256": report.checksum_sha256,
                    "run_status": report.run_status,
                    "tokenizer_manifest_sha256": report.tokenizer_manifest_sha256,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


__all__ = ["main"]
