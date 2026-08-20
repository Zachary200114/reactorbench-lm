"""Narrow local CLI for Phase 4 smoke and Phase 5 pilot milestones."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from reactorbench.evaluation import (
    GoldenReviewConfirmations,
    GoldenReviewDecision,
    create_golden_review_record,
    load_golden_review_packet,
    load_golden_review_record,
    load_phase5_config,
    prepare_golden_review_packet,
    verify_golden_review,
    write_golden_review_packet,
)
from reactorbench.model import load_phase4_config
from reactorbench.schemas.base import canonical_json_bytes

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
    golden = subparsers.add_parser("prepare-golden-review")
    golden.add_argument("--generator-commit", required=True)
    golden.add_argument("--output", required=True, type=Path)
    record = subparsers.add_parser("record-golden-review")
    record.add_argument("--packet", required=True, type=Path)
    record.add_argument("--output", required=True, type=Path)
    record.add_argument("--review-date", required=True)
    record.add_argument(
        "--decision",
        required=True,
        choices=tuple(item.value for item in GoldenReviewDecision),
    )
    record.add_argument("--confirm-all", action="store_true")
    record.add_argument("--note", action="append", default=[])
    verify_golden = subparsers.add_parser("verify-golden-review")
    verify_golden.add_argument("--packet", required=True, type=Path)
    verify_golden.add_argument("--record", required=True, type=Path)
    verify_golden.add_argument("--expected-packet-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        project_root = Path.cwd().resolve(strict=True)
        config_path: Path | None = None
        if hasattr(args, "config"):
            config_path = args.config.resolve(strict=True)
            if not config_path.is_relative_to(project_root):
                raise ValueError("config must be inside the current project checkout")
        if args.command == "run-smoke":
            if config_path is None:
                raise ValueError("run-smoke requires a config")
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
            if config_path is None:
                raise ValueError("verify-smoke requires a config")
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
            if config_path is None:
                raise ValueError("run-pilot requires a config")
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
        elif args.command == "verify-pilot":
            if config_path is None:
                raise ValueError("verify-pilot requires a config")
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
        elif args.command == "prepare-golden-review":
            packet = prepare_golden_review_packet(generator_commit=args.generator_commit)
            output_path = args.output.resolve(strict=False)
            if not output_path.is_relative_to(project_root):
                raise ValueError("golden packet output must stay inside the project checkout")
            write_golden_review_packet(packet, output_path)
            result = {
                "case_count": len(packet.cases),
                "packet_sha256": packet.packet_sha256,
                "review_status": "pending_project_owner_review",
            }
        elif args.command == "record-golden-review":
            packet_path = args.packet.resolve(strict=True)
            output_path = args.output.resolve(strict=False)
            if not packet_path.is_relative_to(project_root) or not output_path.is_relative_to(
                project_root
            ):
                raise ValueError("golden review paths must stay inside the project checkout")
            decision = GoldenReviewDecision(args.decision)
            if decision is GoldenReviewDecision.APPROVED and not args.confirm_all:
                raise ValueError("APPROVED requires the explicit --confirm-all flag")
            confirmations = GoldenReviewConfirmations(
                all_cases_reviewed=args.confirm_all,
                expected_structured_answers_reviewed=args.confirm_all,
                synthetic_and_fictional_only=args.confirm_all,
                no_real_setpoints_or_operating_units=args.confirm_all,
                no_real_procedures_or_facility_topology=args.confirm_all,
                no_service_derived_nonpublic_information=args.confirm_all,
                non_operational_research_use_only=args.confirm_all,
            )
            packet = load_golden_review_packet(packet_path)
            record = create_golden_review_record(
                packet,
                review_date=date.fromisoformat(args.review_date),
                decision=decision,
                confirmations=confirmations,
                notes=tuple(args.note),
            )
            if output_path.exists() or output_path.is_symlink():
                raise FileExistsError("golden review record output must not already exist")
            output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
            output_path.write_bytes(
                canonical_json_bytes(record.model_dump(mode="json", round_trip=True)) + b"\n"
            )
            result = {
                "decision": record.decision.value,
                "record_sha256": record.record_sha256,
            }
        else:
            packet_path = args.packet.resolve(strict=True)
            record_path = args.record.resolve(strict=True)
            if not packet_path.is_relative_to(project_root) or not record_path.is_relative_to(
                project_root
            ):
                raise ValueError("golden review paths must stay inside the project checkout")
            packet = load_golden_review_packet(packet_path)
            record = load_golden_review_record(record_path)
            verify_golden_review(
                packet,
                record,
                expected_packet_sha256=args.expected_packet_sha256,
            )
            result = {
                "case_count": len(packet.cases),
                "decision": record.decision.value,
                "packet_sha256": packet.packet_sha256,
                "review_status": "golden_suite_approved",
            }
        print(json.dumps(result, separators=(",", ":"), sort_keys=True))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


__all__ = ["main"]
