from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pytest

from reactorbench.simulator import (
    ProhibitedContentError,
    assert_no_prohibited_content,
    scan_prohibited_content,
)


class SampleEnum(StrEnum):
    SAFE = "calm"


@dataclass(frozen=True)
class NestedSample:
    name: str
    values: tuple[object, ...]


def test_scanner_recurses_and_returns_bounded_deterministic_findings() -> None:
    sentinel = "https://unique-sentinel.test/path"
    value = NestedSample(name="calm", values=(SampleEnum.SAFE, {"note": f"visit {sentinel}"}))

    findings = scan_prohibited_content(value)

    assert len(findings) == 1
    assert findings[0].rule_id == "url"
    assert len(findings[0].location) <= 160
    assert len(findings[0].context) <= 80
    assert findings[0].context == "[redacted:url]"
    assert sentinel not in findings[0].context
    assert findings == scan_prohibited_content(value)
    with pytest.raises(ProhibitedContentError, match="url") as error:
        assert_no_prohibited_content(value)
    assert sentinel not in str(error.value)


@pytest.mark.parametrize(
    ("value", "rule_id"),
    [
        ("somebody@example.test", "email"),
        ("555-010-0199", "phone"),
        ("42 Amber Road", "address"),
        ("315 MW", "operating_unit"),
        ("4.2 kPa", "operating_unit"),
        ("9 degF", "operating_unit"),
        ("60 Hz", "operating_unit"),
        ("setpoint: 17.5", "setpoint_phrase"),
        ("trip-point of 8", "setpoint_phrase"),
        ("procedure ZX-8", "procedure_identifier"),
        ("docket AB-771", "docket_identifier"),
        ("naval reactor", "military_phrase"),
        ("NRC", "agency_phrase"),
        ("power generating station", "real_plant_phrase"),
        ("nuclear power plant", "real_plant_phrase"),
        ("security review", "security_phrase"),
    ],
)
def test_scanner_covers_required_guard_classes(value: str, rule_id: str) -> None:
    assert rule_id in {finding.rule_id for finding in scan_prohibited_content(value)}


def test_scanner_accepts_variant_like_fictional_text() -> None:
    assert_no_prohibited_content({"alias": "cirrus", "component": "aster-train-cirrus"})
