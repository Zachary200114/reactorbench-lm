"""Narrow local CLI for Phase 4 smoke and Phase 5 pilot milestones."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reactorbench.evaluation import load_phase5_config
from reactorbench.model import load_phase4_config

from .pilot import run_phase5_pilot, verify_phase5_run
from .smoke import run_phase4_smoke, verify_phase4_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reactorbench-phase4")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-smoke")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--source-commit", required=True)
    verify = subparsers.add_parser("verify-smoke")
    verify.add_argument("--config", required=True, type=Path)
    pilot = subparsers.add_parser("run-pilot")
    pilot.add_argument("--config", required=True, type=Path)
    pilot.add_argument("--source-commit", required=True)
    verify_pilot = subparsers.add_parser("verify-pilot")
    verify_pilot.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        project_root = Path.cwd().resolve(strict=True)
        config_path = args.config.resolve(strict=True)
        if not config_path.is_relative_to(project_root):
            raise ValueError("Phase 4 config must be inside the current project checkout")
        if args.command == "run-smoke":
            phase4_config = load_phase4_config(config_path)
            report = run_phase4_smoke(
                phase4_config, project_root=project_root, source_commit=args.source_commit
            )
            result = {
                "checkpoint_manifest_sha256": report.checkpoint_manifest_sha256,
                "final_loss": report.final_loss,
                "report_sha256": report.checksum_sha256,
                "run_status": report.run_status,
                "tokenizer_manifest_sha256": report.tokenizer_manifest_sha256,
            }
        elif args.command == "verify-smoke":
            phase4_config = load_phase4_config(config_path)
            report = verify_phase4_run(phase4_config, project_root=project_root)
            result = {
                "checkpoint_manifest_sha256": report.checkpoint_manifest_sha256,
                "final_loss": report.final_loss,
                "report_sha256": report.checksum_sha256,
                "run_status": report.run_status,
                "tokenizer_manifest_sha256": report.tokenizer_manifest_sha256,
            }
        elif args.command == "run-pilot":
            phase5_config = load_phase5_config(config_path)
            pilot_report = run_phase5_pilot(
                phase5_config, project_root=project_root, source_commit=args.source_commit
            )
            result = {
                "baseline_result_count": len(pilot_report.baseline_results),
                "pilot_validation_nll": (
                    pilot_report.transformer_results[1].selected_validation_nll
                ),
                "report_sha256": pilot_report.checksum_sha256,
                "run_status": pilot_report.run_status,
            }
        else:
            phase5_config = load_phase5_config(config_path)
            pilot_report = verify_phase5_run(phase5_config, project_root=project_root)
            result = {
                "baseline_result_count": len(pilot_report.baseline_results),
                "pilot_validation_nll": (
                    pilot_report.transformer_results[1].selected_validation_nll
                ),
                "report_sha256": pilot_report.checksum_sha256,
                "run_status": pilot_report.run_status,
            }
        print(json.dumps(result, separators=(",", ":"), sort_keys=True))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


__all__ = ["main"]
