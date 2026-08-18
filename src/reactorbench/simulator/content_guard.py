"""Small, deterministic prohibited-content gate for generated simulator records."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum

from pydantic import BaseModel


@dataclass(frozen=True)
class ContentFinding:
    """A bounded description of one prohibited-content match."""

    rule_id: str
    location: str
    context: str


class ProhibitedContentError(ValueError):
    """Raised when simulator-authored material fails the Phase 2 safety gate."""


_MAX_CONTEXT = 80
_MAX_LOCATION = 160
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("url", re.compile(r"https?://|www\.", re.IGNORECASE)),
    ("email", re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)),
    ("phone", re.compile(r"(?:\+?1[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]\d{4}")),
    (
        "address",
        re.compile(r"\b\d{1,5}\s+[A-Z][A-Za-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd)\b"),
    ),
    (
        "operating_unit",
        re.compile(
            r"\b\d+(?:\.\d+)?\s*"
            r"(?:kW|MW|MWe|MWt|kPa|MPa|psi|bar|°C|degC|degF|gpm|Hz|volts?|kV|rpm)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "setpoint_phrase",
        re.compile(
            r"\b(?:setpoint|trip[- ]?point)\s*(?:of\s*)?(?:=|:)?\s*\d+(?:\.\d+)?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "procedure_identifier",
        re.compile(r"\b(?:procedure|checklist)\s+[A-Z]{1,6}[- ]?\d{1,6}\b", re.IGNORECASE),
    ),
    (
        "docket_identifier",
        re.compile(
            r"\b(?:docket|event notification|license event report)\s*"
            r"(?:no\.?\s*)?[-A-Z0-9]{3,}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "military_phrase",
        re.compile(r"\b(?:naval|military)\s+(?:nuclear|reactor)\b", re.IGNORECASE),
    ),
    (
        "agency_phrase",
        re.compile(r"\b(?:NRC|IAEA|DOE)\b|\b(?:agency|government)\s+endors", re.IGNORECASE),
    ),
    (
        "real_plant_phrase",
        re.compile(
            r"\b(?:nuclear\s+power\s+plant|nuclear\s+(?:plant|station)|"
            r"power\s+(?:plant|station|generating station))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "security_phrase",
        re.compile(
            r"\b(?:security|safeguards|attack|exploit|vulnerability|breach)\b", re.IGNORECASE
        ),
    ),
)


def _bounded(value: str, *, limit: int) -> str:
    normalized = " ".join(value.split())
    return normalized[:limit]


def _string_findings(value: str, location: str) -> tuple[ContentFinding, ...]:
    findings: list[ContentFinding] = []
    for rule_id, pattern in _RULES:
        match = pattern.search(value)
        if match is not None:
            findings.append(
                ContentFinding(
                    rule_id=rule_id,
                    location=_bounded(location, limit=_MAX_LOCATION),
                    context=f"[redacted:{rule_id}]",
                )
            )
    return tuple(findings)


def scan_prohibited_content(value: object) -> tuple[ContentFinding, ...]:
    """Return deterministic, bounded findings for recursively reachable strings.

    This guard is intentionally a narrow Phase 2 denylist gate.  It is not a
    claim that all real-facility material can be detected automatically.
    """

    findings: list[ContentFinding] = []
    visited: set[int] = set()

    def visit(item: object, location: str) -> None:
        if isinstance(item, str):
            findings.extend(_string_findings(item, location))
            return
        if item is None or isinstance(item, (bool, int, float, bytes)):
            return
        if isinstance(item, Enum):
            visit(item.value, location)
            return
        identity = id(item)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(item, BaseModel):
            visit(item.model_dump(mode="json", round_trip=True), location)
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name), f"{location}.{field.name}")
        elif isinstance(item, Mapping):
            for index, (key, nested) in enumerate(item.items()):
                visit(key, f"{location}.key[{index}]")
                visit(nested, f"{location}[{index}]")
        elif isinstance(item, Sequence):
            for index, nested in enumerate(item):
                visit(nested, f"{location}[{index}]")

    visit(value, "$")
    return tuple(findings)


def assert_no_prohibited_content(value: object) -> None:
    """Fail closed when any prohibited-content finding is present."""

    findings = scan_prohibited_content(value)
    if findings:
        rule_ids = ", ".join(finding.rule_id for finding in findings)
        raise ProhibitedContentError(f"prohibited content detected: {rule_ids}")
