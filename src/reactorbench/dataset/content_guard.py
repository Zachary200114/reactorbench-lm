"""Phase 3 prohibited-content scanner for candidate rendered records.

This deterministic gate reduces known contamination risks.  Automation is explicitly
not proof that content is safe; a hash-bound full-preview human review remains required
before any candidate dataset can be approved.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from importlib.resources import files
from typing import Any, Literal

from pydantic import BaseModel

GUARD_VERSION = "0.1.0"
NORMALIZATION_ID = "NFKC+casefold+collapsed-whitespace"
AUTOMATION_IS_PROOF: Literal[False] = False
_MAX_FINDINGS = 64
_MAX_LOCATION = 160
_MAX_RULE_ID = 64
_RESOURCE_PACKAGE = "reactorbench.dataset.resources"
_SAFE_ASCII_CONTROLS = {"\n", "\r", "\t"}
_BIDI_CONTROLS = frozenset(
    {
        "\u061c",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
)


@dataclass(frozen=True)
class ContentFinding:
    """A bounded redacted finding; matched source text is never retained."""

    rule_id: str
    location: str
    context: str


class ProhibitedContentError(ValueError):
    """Raised when candidate content fails the Phase 3 gate."""


@dataclass(frozen=True)
class _Fingerprint:
    sha256: str
    source_id: str
    token_count: int


def normalize_text(value: str) -> str:
    """Apply the single scanner normalization used by terms and fingerprints."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _load_json(name: str) -> dict[str, Any]:
    resource = files(_RESOURCE_PACKAGE).joinpath(name)
    data = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{name} must contain a JSON object")
    return data


def _load_denylist() -> tuple[tuple[str, str], ...]:
    data = _load_json("denylist-v1.json")
    if data.get("version") != GUARD_VERSION or data.get("normalization") != NORMALIZATION_ID:
        raise RuntimeError("denylist version or normalization mismatch")
    terms = data.get("terms")
    if not isinstance(terms, dict) or not terms:
        raise RuntimeError("denylist terms must be a non-empty object")
    loaded: list[tuple[str, str]] = []
    for category, values in sorted(terms.items()):
        if not isinstance(category, str) or not isinstance(values, list) or not values:
            raise RuntimeError("denylist categories must contain non-empty string lists")
        for value in values:
            if not isinstance(value, str) or normalize_text(value) != value:
                raise RuntimeError("denylist terms must already use canonical normalization")
            loaded.append((f"denylist.{category}", value))
    return tuple(loaded)


def _load_fingerprints() -> tuple[int, tuple[_Fingerprint, ...], str]:
    data = _load_json("copied-span-fingerprints-v1.json")
    if (
        data.get("version") != GUARD_VERSION
        or data.get("algorithm") != "sha256"
        or data.get("normalization") != NORMALIZATION_ID
    ):
        raise RuntimeError("copied-span fingerprint registry metadata mismatch")
    minimum_tokens = data.get("minimum_tokens")
    review_status = data.get("review_status")
    raw = data.get("fingerprints")
    if type(minimum_tokens) is not int or minimum_tokens < 4:
        raise RuntimeError("fingerprint minimum_tokens must be an integer of at least four")
    if not isinstance(review_status, str) or not review_status:
        raise RuntimeError("fingerprint registry requires review status")
    if not isinstance(raw, list):
        raise RuntimeError("fingerprints must be a list")
    loaded: list[_Fingerprint] = []
    for item in raw:
        if not isinstance(item, dict):
            raise RuntimeError("fingerprint entries must be objects")
        digest = item.get("fingerprint_sha256")
        source_id = item.get("source_id")
        token_count = item.get("token_count")
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(source_id, str)
            or not source_id
            or type(token_count) is not int
            or token_count < minimum_tokens
        ):
            raise RuntimeError("invalid copied-span fingerprint entry")
        loaded.append(_Fingerprint(digest, source_id, token_count))
    return minimum_tokens, tuple(loaded), review_status


_DENYLIST = _load_denylist()
_FINGERPRINT_MINIMUM, _FINGERPRINTS, FINGERPRINT_REVIEW_STATUS = _load_fingerprints()

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("contact.url", re.compile(r"(?:https?://|www\.)\S+")),
    ("contact.email", re.compile(r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")),
    (
        "contact.phone",
        re.compile(
            r"(?<![\d.])(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]"
            r"\d{3}[ .-]\d{4}(?![\d.])|(?<!\d)\d{10}(?!\d)"
        ),
    ),
    (
        "contact.address",
        re.compile(
            r"\b\d{1,6}\s+[a-z][a-z.'\- ]{1,48}\s+(?:street|st|avenue|ave|road|rd|"
            r"boulevard|blvd|lane|ln|drive|dr|way|court|ct)\b"
        ),
    ),
    (
        "real_operating_unit",
        re.compile(
            r"\b\d+(?:\.\d+)?\s*(?:kw|mw|mwe|mwt|kpa|mpa|psi|bar|celsius|fahrenheit|"
            r"degc|degf|°c|°f|gpm|hz|volts?|kv|rpm|kg/s|lbm?/hr)\b"
        ),
    ),
    ("percent_value", re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent)\b")),
    (
        "setpoint_or_threshold",
        re.compile(
            r"\b(?:set[ -]?point|trip[ -]?point|threshold|alarm point|limit)\s*"
            r"(?:of\s*)?(?:=|:|at)?\s*\d+(?:\.\d+)?\b"
        ),
    ),
    (
        "real_identifier",
        re.compile(
            r"\b(?:docket|event notification|licensee event report|ler|nureg|procedure|"
            r"checklist|work order)\s*(?:no\.?\s*)?[a-z0-9][a-z0-9.\-/]{2,}\b"
        ),
    ),
    ("agency_acronym", re.compile(r"\b(?:nrc|iaea|doe|nnsa|navsea)\b")),
    ("html_markup", re.compile(r"<\s*/?\s*[a-z][^>]*>|&(?:lt|gt|#x?[0-9a-f]+);")),
    (
        "imperative_numbered_procedure",
        re.compile(
            r"(?:^|\n)\s*\d{1,2}[.)]\s*(?:open|close|start|stop|isolate|restore|verify|"
            r"check|set|trip|bypass|enter|reduce|increase|decrease|operate|inspect)\b"
        ),
    ),
)


def _bounded(value: str, limit: int) -> str:
    return " ".join(value.split())[:limit]


def _finding(rule_id: str, location: str) -> ContentFinding:
    bounded_rule = _bounded(rule_id, _MAX_RULE_ID)
    return ContentFinding(
        rule_id=bounded_rule,
        location=_bounded(location, _MAX_LOCATION),
        context=f"[redacted:{bounded_rule}]",
    )


def _copied_span_findings(normalized: str, location: str) -> list[ContentFinding]:
    tokens = normalized.split()
    findings: list[ContentFinding] = []
    by_count: dict[int, set[str]] = {}
    for fingerprint in _FINGERPRINTS:
        by_count.setdefault(fingerprint.token_count, set()).add(fingerprint.sha256)
    for token_count, digests in sorted(by_count.items()):
        if token_count < _FINGERPRINT_MINIMUM or len(tokens) < token_count:
            continue
        for index in range(len(tokens) - token_count + 1):
            span = " ".join(tokens[index : index + token_count])
            digest = hashlib.sha256(span.encode("utf-8")).hexdigest()
            if digest in digests:
                findings.append(_finding("copied_span_fingerprint", location))
                break
    return findings


def _string_findings(value: str, location: str) -> tuple[ContentFinding, ...]:
    normalized = normalize_text(value)
    findings: list[ContentFinding] = []
    for character in value:
        category = unicodedata.category(character)
        if character in _BIDI_CONTROLS:
            findings.append(_finding("unicode.bidi_control", location))
            break
        if category in {"Cf", "Cs", "Co"} or (
            category == "Cc" and character not in _SAFE_ASCII_CONTROLS
        ):
            findings.append(_finding("unicode.invisible_or_control", location))
            break
    if any(ord(character) > 127 for character in value):
        findings.append(_finding("unicode.non_ascii", location))
    for rule_id, term in _DENYLIST:
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", normalized):
            findings.append(_finding(rule_id, location))
    for rule_id, pattern in _PATTERNS:
        if pattern.search(normalized):
            findings.append(_finding(rule_id, location))
    findings.extend(_copied_span_findings(normalized, location))
    unique: dict[tuple[str, str], ContentFinding] = {}
    for finding in findings:
        unique.setdefault((finding.rule_id, finding.location), finding)
    return tuple(unique.values())


def scan_prohibited_content(value: object) -> tuple[ContentFinding, ...]:
    """Recursively scan supported data and return bounded deterministic findings."""

    findings: list[ContentFinding] = []
    visited: set[int] = set()

    def visit(item: object, location: str) -> None:
        if len(findings) >= _MAX_FINDINGS:
            return
        if isinstance(item, str):
            findings.extend(_string_findings(item, location))
            return
        if item is None or isinstance(item, (bool, int, float)):
            return
        if isinstance(item, bytes):
            findings.append(_finding("unsupported_binary", location))
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
            for key in sorted(item, key=lambda candidate: str(candidate)):
                visit(str(key), f"{location}.key")
                visit(item[key], f"{location}.{key}")
        elif isinstance(item, Sequence):
            for index, nested in enumerate(item):
                visit(nested, f"{location}[{index}]")
        else:
            findings.append(_finding("unsupported_object", location))

    visit(value, "$")
    return tuple(findings[:_MAX_FINDINGS])


def assert_no_prohibited_content(value: object) -> None:
    """Fail closed without reflecting matched content into the exception."""

    findings = scan_prohibited_content(value)
    if findings:
        rule_ids = ", ".join(dict.fromkeys(finding.rule_id for finding in findings))
        raise ProhibitedContentError(f"prohibited content detected: {rule_ids}")


def guard_manifest() -> dict[str, object]:
    """Return version/checksum metadata without disclosing denylist terms."""

    denylist_resource = files(_RESOURCE_PACKAGE).joinpath("denylist-v1.json").read_bytes()
    fingerprint_resource = (
        files(_RESOURCE_PACKAGE).joinpath("copied-span-fingerprints-v1.json").read_bytes()
    )
    return {
        "guard_version": GUARD_VERSION,
        "normalization": NORMALIZATION_ID,
        "automation_is_proof": AUTOMATION_IS_PROOF,
        "fingerprint_review_status": FINGERPRINT_REVIEW_STATUS,
        "denylist_sha256": hashlib.sha256(denylist_resource).hexdigest(),
        "fingerprints_sha256": hashlib.sha256(fingerprint_resource).hexdigest(),
    }
