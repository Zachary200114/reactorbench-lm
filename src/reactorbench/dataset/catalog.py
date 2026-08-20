"""Versioned project-authored renderer and alias catalogs.

The catalog contains no generated prose and no external text.  Its identifiers and
checksums are stable inputs to dataset provenance.
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum

from pydantic import Field, model_validator

from reactorbench.schemas.base import ContractModel, SemanticVersion, canonical_json_bytes
from reactorbench.schemas.enums import EventType, StateVariable

RENDERER_VERSION: SemanticVersion = "0.1.0"
CATALOG_VERSION: SemanticVersion = "0.1.0"


class TemplateFamily(StrEnum):
    """Closed template-family vocabulary; the final family is a strict holdout."""

    COMPACT_LOG = "compact-log-v1"
    OBSERVER_NOTE = "observer-note-v1"
    SHIFT_LEDGER = "shift-ledger-v1"
    RESEARCH_EDITORIAL = "research-editorial-v1"


class AliasFamily(StrEnum):
    """Independent subject/variable alias vocabularies."""

    CANONICAL = "canonical-v1"
    SHORT = "short-v1"
    NEUTRAL = "neutral-role-v1"
    HELDOUT = "heldout-v1"


class TemplateSpec(ContractModel):
    template_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._:-]{0,95}$")
    family: TemplateFamily
    event_type: EventType
    lead: str = Field(min_length=1, max_length=96)
    separator: str = Field(min_length=1, max_length=4)
    terminal: str = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def contains_no_template_placeholders(self) -> TemplateSpec:
        authored = f"{self.lead}{self.separator}{self.terminal}"
        if re.search(r"\{[^}]*\}|\$\{[^}]*\}|<[^>]+>", authored):
            raise ValueError("template catalog cannot contain unfinished placeholders")
        return self

    def checksum(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


class AliasSpec(ContractModel):
    family: AliasFamily
    subject_prefix: str = Field(min_length=1, max_length=24, pattern=r"^[a-z][a-z-]*$")
    variable_prefix: str = Field(min_length=1, max_length=24, pattern=r"^[a-z][a-z-]*$")

    def checksum(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


class CatalogManifest(ContractModel):
    catalog_version: SemanticVersion
    renderer_version: SemanticVersion
    templates: tuple[TemplateSpec, ...]
    aliases: tuple[AliasSpec, ...]
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


_STYLE: dict[TemplateFamily, tuple[str, str, str]] = {
    TemplateFamily.COMPACT_LOG: ("event", " | ", "."),
    TemplateFamily.OBSERVER_NOTE: ("observer note", ": ", "."),
    TemplateFamily.SHIFT_LEDGER: ("sequence ledger", " - ", "."),
    TemplateFamily.RESEARCH_EDITORIAL: ("evidence record", ": ", "."),
}

_EVENT_LEADS: dict[EventType, str] = {
    EventType.OPERATING_MODE_CHANGED: "operating mode transition",
    EventType.TARGET_CHANGED: "bounded target transition",
    EventType.COMPONENT_STATE_CHANGED: "component state transition",
    EventType.OBSERVATION_CHANGED: "channel observation transition",
    EventType.CHANNEL_QUALITY_CHANGED: "channel quality transition",
    EventType.CHANNEL_DISAGREEMENT: "redundant channel disagreement",
    EventType.COMMAND_RECORDED: "bounded command record",
    EventType.COMMAND_POSITION_MISMATCH: "command and position differ",
    EventType.COMMAND_POSITION_ALIGNED: "command and position align",
    EventType.ACTION_APPLIED: "fictional label application",
    EventType.BENIGN_NOTE: "bounded context note",
}

_SHORT_VARIABLES: dict[StateVariable, str] = {
    variable: f"s{index:02d}" for index, variable in enumerate(StateVariable, start=1)
}

_ALIASES: tuple[AliasSpec, ...] = (
    AliasSpec(
        family=AliasFamily.CANONICAL,
        subject_prefix="aster",
        variable_prefix="signal",
    ),
    AliasSpec(family=AliasFamily.SHORT, subject_prefix="asset", variable_prefix="sig"),
    AliasSpec(family=AliasFamily.NEUTRAL, subject_prefix="component", variable_prefix="measure"),
    AliasSpec(family=AliasFamily.HELDOUT, subject_prefix="node", variable_prefix="indicator"),
)


def _build_templates() -> tuple[TemplateSpec, ...]:
    templates: list[TemplateSpec] = []
    for family in TemplateFamily:
        style_lead, separator, terminal = _STYLE[family]
        for event_type in EventType:
            templates.append(
                TemplateSpec(
                    template_id=f"{family.value}:{event_type.value.lower()}:v1",
                    family=family,
                    event_type=event_type,
                    lead=f"{style_lead}; {_EVENT_LEADS[event_type]}",
                    separator=separator,
                    terminal=terminal,
                )
            )
    return tuple(templates)


TEMPLATES: tuple[TemplateSpec, ...] = _build_templates()
TEMPLATE_BY_KEY: dict[tuple[TemplateFamily, EventType], TemplateSpec] = {
    (template.family, template.event_type): template for template in TEMPLATES
}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def subject_alias(subject_id: str, family: AliasFamily) -> str:
    """Return a deterministic alias without consulting a scenario or target."""

    if not isinstance(subject_id, str) or not subject_id:
        raise ValueError("subject_id must be a non-empty string")
    if type(family) is not AliasFamily:
        raise ValueError("family must be an AliasFamily")
    if family is AliasFamily.CANONICAL:
        return subject_id.replace("-", " ")
    digest = _digest(subject_id)
    if family is AliasFamily.SHORT:
        return f"asset-{digest[:6]}"
    if family is AliasFamily.NEUTRAL:
        return f"component-{digest[6:14]}"
    return f"node-{digest[14:22]}"


def variable_alias(variable: StateVariable, family: AliasFamily) -> str:
    """Return a deterministic variable alias independent of template choice."""

    if type(variable) is not StateVariable or type(family) is not AliasFamily:
        raise ValueError("variable and family must use canonical enums")
    if family is AliasFamily.CANONICAL:
        return variable.value.replace("_", " ")
    if family is AliasFamily.SHORT:
        return _SHORT_VARIABLES[variable]
    digest = _digest(variable.value)
    if family is AliasFamily.NEUTRAL:
        return f"measure-{digest[:6]}"
    return f"indicator-{digest[6:12]}"


def catalog_manifest() -> CatalogManifest:
    """Return the canonical catalog manifest and its content checksum."""

    checksum_payload = {
        "catalog_version": CATALOG_VERSION,
        "renderer_version": RENDERER_VERSION,
        "templates": tuple(template.model_dump(mode="json") for template in TEMPLATES),
        "aliases": tuple(alias.model_dump(mode="json") for alias in _ALIASES),
    }
    checksum = hashlib.sha256(canonical_json_bytes(checksum_payload)).hexdigest()
    return CatalogManifest(
        catalog_version=CATALOG_VERSION,
        renderer_version=RENDERER_VERSION,
        templates=TEMPLATES,
        aliases=_ALIASES,
        checksum_sha256=checksum,
    )


def validate_catalog() -> None:
    """Fail closed if coverage, identity, or authored-text invariants are broken."""

    expected = {(family, event_type) for family in TemplateFamily for event_type in EventType}
    if set(TEMPLATE_BY_KEY) != expected:
        raise ValueError("template catalog must cover every event type in every family")
    template_ids = tuple(template.template_id for template in TEMPLATES)
    if len(template_ids) != len(set(template_ids)):
        raise ValueError("template IDs must be unique")
    if {alias.family for alias in _ALIASES} != set(AliasFamily):
        raise ValueError("alias catalog must cover every alias family")


validate_catalog()
