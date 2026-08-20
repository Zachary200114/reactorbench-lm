from __future__ import annotations

from dataclasses import dataclass

import pytest

from reactorbench.dataset.content_guard import (
    AUTOMATION_IS_PROOF,
    ProhibitedContentError,
    assert_no_prohibited_content,
    guard_manifest,
    normalize_text,
    scan_prohibited_content,
)


@dataclass(frozen=True)
class NestedText:
    note: str


def _rule_ids(value: object) -> set[str]:
    return {finding.rule_id for finding in scan_prohibited_content(value)}


def test_nfkc_casefold_scanner_is_deterministic_bounded_and_redacted() -> None:
    sensitive = "\uff2e\uff32\uff23 contact person@example.test"
    value = {"nested": NestedText(note=sensitive)}

    findings = scan_prohibited_content(value)

    assert findings == scan_prohibited_content(value)
    assert {"agency_acronym", "contact.email", "unicode.non_ascii"}.issubset(
        {finding.rule_id for finding in findings}
    )
    assert all(len(finding.location) <= 160 for finding in findings)
    assert all(finding.context == f"[redacted:{finding.rule_id}]" for finding in findings)
    assert all(sensitive not in finding.context for finding in findings)
    with pytest.raises(ProhibitedContentError) as error:
        assert_no_prohibited_content(value)
    assert sensitive not in str(error.value)


@pytest.mark.parametrize(
    ("value", "rule_id"),
    [
        ("naval nuclear program", "denylist.military_nuclear"),
        ("U.S. Navy reference", "denylist.military_nuclear"),
        ("three mile island", "denylist.real_facility"),
        ("physical security detail", "denylist.security_or_safeguards"),
        ("https://example.test/path", "contact.url"),
        ("555-010-0199", "contact.phone"),
        ("42 Amber Road", "contact.address"),
        ("315 MW", "real_operating_unit"),
        ("74 percent", "percent_value"),
        ("setpoint: 17.5", "setpoint_or_threshold"),
        ("LER 2026-001", "real_identifier"),
        ("<script>bad</script>", "html_markup"),
        ("1. Open the valve", "imperative_numbered_procedure"),
        ("safe\u200bhidden", "unicode.invisible_or_control"),
        ("safe\u202ereversed", "unicode.bidi_control"),
        (
            "copied reference sentinel phrase with eight distinct review tokens",
            "copied_span_fingerprint",
        ),
    ],
)
def test_scanner_covers_required_phase3_rule_classes(value: str, rule_id: str) -> None:
    assert rule_id in _rule_ids(value)


def test_guard_metadata_is_versioned_checksummed_and_explicitly_not_proof() -> None:
    manifest = guard_manifest()

    assert AUTOMATION_IS_PROOF is False
    assert manifest["automation_is_proof"] is False
    assert manifest["guard_version"] == "0.1.0"
    assert len(str(manifest["denylist_sha256"])) == 64
    assert len(str(manifest["fingerprints_sha256"])) == 64
    assert "human-approval" in str(manifest["fingerprint_review_status"])


def test_normalizer_and_safe_fictional_text() -> None:
    assert normalize_text("  A\uff33TER\tSignal  ") == "aster signal"
    assert_no_prohibited_content(
        "[T+003] observer note: fictional channel value 0.5120 is under comparison."
    )
    assert_no_prohibited_content(
        "[T+002] value 0.5000 moved to 0.5120; [T+003] value 0.5120 moved to 0.5240."
    )


def test_binary_content_fails_closed_instead_of_bypassing_text_rules() -> None:
    findings = scan_prohibited_content(b"unscanned binary")
    assert {finding.rule_id for finding in findings} == {"unsupported_binary"}
