from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from reactorbench.evaluation.golden import (
    GoldenReviewConfirmations,
    GoldenReviewDecision,
    GoldenReviewPacket,
    create_golden_review_record,
    load_golden_review_packet,
    prepare_golden_review_packet,
    verify_golden_review,
    write_golden_review_packet,
)


def _confirmations(value: bool) -> GoldenReviewConfirmations:
    return GoldenReviewConfirmations(
        all_cases_reviewed=value,
        expected_structured_answers_reviewed=value,
        synthetic_and_fictional_only=value,
        no_real_setpoints_or_operating_units=value,
        no_real_procedures_or_facility_topology=value,
        no_service_derived_nonpublic_information=value,
        non_operational_research_use_only=value,
    )


def test_golden_packet_is_deterministic_complete_and_safe(tmp_path: Path) -> None:
    first = prepare_golden_review_packet(generator_commit="4473718")
    second = prepare_golden_review_packet(generator_commit="4473718")
    assert first == second
    assert tuple(case.case_id for case in first.cases) == tuple(
        f"G{index:02d}" for index in range(1, 16)
    )
    assert all(case.prohibited_content_finding_count == 0 for case in first.cases)
    assert first.cases[-1].expected_targets.decisions[0].immediate_action.value == (
        "INSUFFICIENT_EVIDENCE"
    )

    path = tmp_path / "golden.json"
    write_golden_review_packet(first, path)
    assert load_golden_review_packet(path) == first
    with pytest.raises(FileExistsError):
        write_golden_review_packet(first, path)


def test_golden_approval_is_explicit_and_checksum_bound() -> None:
    packet = prepare_golden_review_packet(generator_commit="4473718")
    with pytest.raises(ValidationError):
        create_golden_review_record(
            packet,
            review_date=date(2026, 8, 20),
            decision=GoldenReviewDecision.APPROVED,
            confirmations=_confirmations(False),
        )
    revise = create_golden_review_record(
        packet,
        review_date=date(2026, 8, 20),
        decision=GoldenReviewDecision.REVISE,
        confirmations=_confirmations(False),
    )
    with pytest.raises(ValueError, match="not approved"):
        verify_golden_review(
            packet,
            revise,
            expected_packet_sha256=packet.packet_sha256,
        )
    approved = create_golden_review_record(
        packet,
        review_date=date(2026, 8, 20),
        decision=GoldenReviewDecision.APPROVED,
        confirmations=_confirmations(True),
        notes=("Owner reviewed all G01-G15 structured expectations.",),
    )
    verify_golden_review(
        packet,
        approved,
        expected_packet_sha256=packet.packet_sha256,
    )
    with pytest.raises(ValueError, match="preregistered checksum"):
        verify_golden_review(packet, approved, expected_packet_sha256="0" * 64)


def test_golden_models_fail_closed_against_copy_and_json_tampering(tmp_path: Path) -> None:
    packet = prepare_golden_review_packet(generator_commit="4473718")
    forged = packet.model_copy(update={"packet_sha256": "0" * 64})
    with pytest.raises(ValidationError):
        GoldenReviewPacket.model_validate(forged.model_dump(mode="python", round_trip=True))

    path = tmp_path / "golden.json"
    path.write_text('{"packet_version":"0.1.0","packet_version":"0.1.0"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_golden_review_packet(path)

    with pytest.raises(ValueError, match="generator_commit"):
        prepare_golden_review_packet(generator_commit="not-a-commit")
