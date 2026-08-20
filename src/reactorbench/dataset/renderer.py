"""Deterministic project-authored rendering from the narrow ``ModelInput`` boundary."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Literal

from pydantic import Field, model_validator

from reactorbench.schemas.base import ContractModel, SemanticVersion, canonical_json_bytes
from reactorbench.schemas.enums import (
    ChannelQuality,
    ComponentState,
    EventType,
    ObservationStatus,
    OperatingMode,
    PlantVariant,
    SplitName,
    StateVariable,
)

from .catalog import (
    RENDERER_VERSION,
    TEMPLATE_BY_KEY,
    AliasFamily,
    TemplateFamily,
    catalog_manifest,
    subject_alias,
    variable_alias,
)
from .content_guard import assert_no_prohibited_content
from .contracts import (
    DependencyLinkContextFact,
    ModelInput,
    PlantVariantContextFact,
    ProjectedEventFact,
    ProjectedObservationFact,
    StandbyRelationshipContextFact,
)

_UNFINISHED = re.compile(r"\{[^}]*\}|\$\{[^}]*\}|<[^>]+>")


class CatalogPreviewEntry(ContractModel):
    template_family_id: TemplateFamily
    alias_family_id: AliasFamily
    event_type: EventType
    template_id: str
    text: str
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CatalogPreviewPacket(ContractModel):
    """Pre-render review fixture; it is never a release-split candidate."""

    preview_status: Literal["catalog_review_only"] = "catalog_review_only"
    renderer_version: SemanticVersion = RENDERER_VERSION
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entry_count: int = Field(ge=1)
    entries: tuple[CatalogPreviewEntry, ...]
    packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def coverage_and_checksum_are_valid(self) -> CatalogPreviewPacket:
        expected_keys = {
            (template_family, alias_family, event_type)
            for template_family in TemplateFamily
            for alias_family in AliasFamily
            for event_type in EventType
        }
        actual_keys = {
            (entry.template_family_id, entry.alias_family_id, entry.event_type)
            for entry in self.entries
        }
        if actual_keys != expected_keys or self.entry_count != len(self.entries):
            raise ValueError("catalog preview must cover every template, alias, and event")
        if len(actual_keys) != len(self.entries):
            raise ValueError("catalog preview combinations must be unique")
        expected_checksum = hashlib.sha256(
            canonical_json_bytes(
                self.model_dump(mode="json", exclude={"packet_sha256"}, round_trip=True)
            )
        ).hexdigest()
        if self.packet_sha256 != expected_checksum:
            raise ValueError("catalog preview checksum mismatch")
        return self


class AuthoredSurfaceEntry(ContractModel):
    """One canonical authored phrase or line exposed for human review."""

    surface_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._:-]{0,127}$")
    text: str = Field(min_length=1, max_length=4096)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def text_checksum_matches(self) -> AuthoredSurfaceEntry:
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.text_sha256:
            raise ValueError("authored surface text checksum mismatch")
        return self


class RendererAuthoredSurfaceManifest(ContractModel):
    """Exhaustive renderer-authored language surface presented to the owner."""

    manifest_version: Literal["0.1.0"] = "0.1.0"
    observation_status_phrases: tuple[AuthoredSurfaceEntry, ...]
    channel_quality_phrases: tuple[AuthoredSurfaceEntry, ...]
    operating_mode_phrases: tuple[AuthoredSurfaceEntry, ...]
    component_state_phrases: tuple[AuthoredSurfaceEntry, ...]
    observation_lines: tuple[AuthoredSurfaceEntry, ...]
    event_clauses: tuple[AuthoredSurfaceEntry, ...]
    context_lines: tuple[AuthoredSurfaceEntry, ...]
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def inventory_and_checksum_match(self) -> RendererAuthoredSurfaceManifest:
        groups = (
            self.observation_status_phrases,
            self.channel_quality_phrases,
            self.operating_mode_phrases,
            self.component_state_phrases,
            self.observation_lines,
            self.event_clauses,
            self.context_lines,
        )
        for group in groups:
            if tuple(entry.surface_id for entry in group) != tuple(
                sorted(entry.surface_id for entry in group)
            ):
                raise ValueError("renderer authored surfaces must use canonical ID order")
        identifiers = tuple(entry.surface_id for group in groups for entry in group)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("renderer authored surface IDs must be unique")
        expected = hashlib.sha256(
            canonical_json_bytes(
                self.model_dump(mode="json", exclude={"checksum_sha256"}, round_trip=True)
            )
        ).hexdigest()
        if self.checksum_sha256 != expected:
            raise ValueError("renderer authored surface manifest checksum mismatch")
        return self


class RenderedCandidate(ContractModel):
    """Candidate prose. It is not an approved dataset record."""

    candidate_status: Literal["candidate_pending_postrender_review"] = (
        "candidate_pending_postrender_review"
    )
    renderer_version: SemanticVersion = RENDERER_VERSION
    split_name: SplitName
    template_family_id: TemplateFamily
    alias_family_id: AliasFamily
    model_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_ids: tuple[str, ...]
    lines: tuple[str, ...] = Field(min_length=1, max_length=8192)
    text: str = Field(min_length=1, max_length=65_536)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    render_id: str = Field(pattern=r"^render-[0-9a-f]{24}$")

    @model_validator(mode="after")
    def derived_fields_are_consistent(self) -> RenderedCandidate:
        if self.text != "\n".join(self.lines):
            raise ValueError("rendered text must be the newline join of rendered lines")
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.text_sha256:
            raise ValueError("rendered text checksum mismatch")
        if self.template_ids != tuple(dict.fromkeys(self.template_ids)):
            raise ValueError("template IDs must be unique in first-use order")
        payload = self.model_dump(
            mode="json", exclude={"render_id", "text_sha256"}, round_trip=True
        )
        expected_id = f"render-{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:24]}"
        if self.render_id != expected_id:
            raise ValueError("render ID does not match candidate content")
        return self


def _words(value: str) -> str:
    return value.casefold().replace("_", "-")


def _number(value: float | None) -> str:
    return "missing" if value is None else f"{value:.4f}"


def _status(value: ObservationStatus) -> str:
    return {
        ObservationStatus.NORMAL: "within the expected band",
        ObservationStatus.WATCH: "under comparison",
        ObservationStatus.ABNORMAL: "outside the expected band",
        ObservationStatus.MISSING: "not available",
        ObservationStatus.CONFLICTING: "in conflict with its pair",
    }[value]


def _quality(value: ChannelQuality) -> str:
    # Deliberately do not emit the enum word NOISY; that lexical shortcut belongs to
    # structured audit truth, not model-visible prose.
    return {
        ChannelQuality.GOOD: "available",
        ChannelQuality.SUSPECT: "flagged for comparison",
        ChannelQuality.NOISY: "variable across recent readings",
        ChannelQuality.UNAVAILABLE: "unavailable",
    }[value]


def _mode(value: OperatingMode) -> str:
    return _words(value.value)


def _component_state(value: ComponentState) -> str:
    return _words(value.value)


def _required[T](value: T | None, *, field_name: str) -> T:
    if value is None:
        raise ValueError(f"projected event is missing required field {field_name}")
    return value


def _variable(value: StateVariable, aliases: AliasFamily) -> str:
    return variable_alias(value, aliases)


def _observation_line(fact: ProjectedObservationFact, aliases: AliasFamily) -> str:
    channel = subject_alias(fact.channel_id, aliases)
    variable = _variable(fact.variable, aliases)
    return (
        f"[T+{fact.tick:03d}] [{fact.fact_ref}] observed {variable} at {channel}: "
        f"value {_number(fact.value)}; "
        f"status {_status(fact.status)}; channel quality {_quality(fact.quality)}."
    )


def _event_clauses(fact: ProjectedEventFact, aliases: AliasFamily) -> str:
    subject = subject_alias(fact.subject_id, aliases)
    event_type = fact.event_type
    if event_type is EventType.OPERATING_MODE_CHANGED:
        before_mode = _required(fact.operating_mode_before, field_name="operating_mode_before")
        after_mode = _required(fact.operating_mode_after, field_name="operating_mode_after")
        return f"{subject} moved from {_mode(before_mode)} to {_mode(after_mode)}"
    if event_type is EventType.TARGET_CHANGED:
        variable = _required(fact.variable, field_name="variable")
        return (
            f"{_variable(variable, aliases)} at {subject} changed from "
            f"{_number(fact.value_before)} to {_number(fact.value_after)}"
        )
    if event_type is EventType.COMPONENT_STATE_CHANGED:
        before_state = _required(fact.component_state_before, field_name="component_state_before")
        after_state = _required(fact.component_state_after, field_name="component_state_after")
        return (
            f"{subject} changed from {_component_state(before_state)} to "
            f"{_component_state(after_state)}"
        )
    if event_type is EventType.OBSERVATION_CHANGED:
        variable = _required(fact.variable, field_name="variable")
        status = _required(fact.observation_status, field_name="observation_status")
        before = f" from {_number(fact.value_before)}" if fact.value_before is not None else ""
        return (
            f"{_variable(variable, aliases)} at {subject} moved{before} to "
            f"{_number(fact.value_after)} and is {_status(status)}"
        )
    if event_type is EventType.CHANNEL_QUALITY_CHANGED:
        before_quality = _required(fact.channel_quality_before, field_name="channel_quality_before")
        after_quality = _required(fact.channel_quality, field_name="channel_quality")
        return (
            f"{subject} quality changed from {_quality(before_quality)} to "
            f"{_quality(after_quality)}"
        )
    if event_type is EventType.CHANNEL_DISAGREEMENT:
        variable = _required(fact.variable, field_name="variable")
        status = _required(fact.observation_status, field_name="observation_status")
        return (
            f"paired readings for {_variable(variable, aliases)} at {subject} are {_status(status)}"
        )
    if event_type is EventType.COMMAND_RECORDED:
        variable = _required(fact.variable, field_name="variable")
        return (
            f"{subject} recorded a bounded {_variable(variable, aliases)} command at "
            f"{_number(fact.commanded_value)}"
        )
    if event_type is EventType.COMMAND_POSITION_MISMATCH:
        variable = _required(fact.variable, field_name="variable")
        return (
            f"{_variable(variable, aliases)} at {subject} has command "
            f"{_number(fact.commanded_value)} and observation {_number(fact.observed_value)}"
        )
    if event_type is EventType.COMMAND_POSITION_ALIGNED:
        variable = _required(fact.variable, field_name="variable")
        return (
            f"{_variable(variable, aliases)} at {subject} has matching command and "
            f"observation {_number(fact.observed_value)}"
        )
    if event_type is EventType.BENIGN_NOTE:
        return f"{subject} retained bounded fictional context"
    if event_type is EventType.ACTION_APPLIED:
        # ProjectedEventFact rejects this type.  Keeping catalog coverage explicit makes
        # any future safe contract extension fail visibly instead of falling through.
        return f"{subject} recorded a prior fictional label application"
    raise ValueError("unsupported event type")


def _event_line(
    fact: ProjectedEventFact, templates: TemplateFamily, aliases: AliasFamily
) -> tuple[str, str]:
    spec = TEMPLATE_BY_KEY[(templates, fact.event_type)]
    line = (
        f"[T+{fact.tick:03d}] [{fact.fact_ref}] {spec.lead}{spec.separator}"
        f"{_event_clauses(fact, aliases)}{spec.terminal}"
    )
    return spec.template_id, line


def _context_lines(model_input: ModelInput, aliases: AliasFamily) -> tuple[str, ...]:
    lines: list[str] = []
    for fact in model_input.context_facts:
        if isinstance(fact, PlantVariantContextFact):
            lines.append(
                f"[CONTEXT] [{fact.fact_ref}] fictional variant "
                f"{_words(fact.plant_variant_id.value)}."
            )
        elif isinstance(fact, StandbyRelationshipContextFact):
            active = subject_alias(fact.active_component_id, aliases)
            standby = subject_alias(fact.standby_component_id, aliases)
            support = subject_alias(fact.support_component_id, aliases)
            lines.append(
                f"[CONTEXT] [{fact.fact_ref}] {standby} is the paired standby for {active}; "
                f"its state is "
                f"{_component_state(fact.standby_state)}; {support} is "
                f"{_component_state(fact.support_state)}; fictional start delay "
                f"{fact.start_delay_ticks} ticks."
            )
        elif isinstance(fact, DependencyLinkContextFact):
            support = subject_alias(fact.support_component_id, aliases)
            dependent = subject_alias(fact.dependent_component_id, aliases)
            lines.append(
                f"[CONTEXT] [{fact.fact_ref}] {dependent} has a fictional dependency on {support}."
            )
        else:  # pragma: no cover - discriminated union is closed, fail closed if extended.
            raise ValueError("unsupported projected context fact")
    return tuple(lines)


def _validate_family_split(
    template_family: TemplateFamily, alias_family: AliasFamily, split_name: SplitName
) -> None:
    if type(template_family) is not TemplateFamily:
        raise ValueError("template_family must use the canonical enum")
    if type(alias_family) is not AliasFamily:
        raise ValueError("alias_family must use the canonical enum")
    if type(split_name) is not SplitName:
        raise ValueError("split_name must use the canonical enum")
    heldout_template = template_family is TemplateFamily.RESEARCH_EDITORIAL
    if heldout_template != (split_name is SplitName.TEMPLATE_TEST):
        raise ValueError("research-editorial-v1 is reserved exclusively for template_test")
    heldout_alias = alias_family is AliasFamily.HELDOUT
    if heldout_alias != (split_name is SplitName.COMPONENT_TEST):
        raise ValueError("heldout-v1 aliases are reserved exclusively for component_test")


def _catalog_event_fixture(event_type: EventType) -> ProjectedEventFact:
    """Return the canonical review fixture for one event-clause surface.

    ACTION_APPLIED is deliberately impossible inside production ``ModelInput``. Its
    catalog clause still needs owner review, so that one entry uses an explicit
    review-only construction. All ten renderer-safe event fixtures are fully validated.
    """

    common: dict[str, object] = {
        "fact_ref": "e-0000",
        "tick": 3,
        "event_type": event_type,
        "subject_id": "aster-review-component",
    }
    payloads: dict[EventType, dict[str, object]] = {
        EventType.OPERATING_MODE_CHANGED: {
            "operating_mode_before": OperatingMode.STABLE,
            "operating_mode_after": OperatingMode.DISTURBED,
        },
        EventType.TARGET_CHANGED: {
            "variable": StateVariable.LOAD_DEMAND,
            "value_before": 0.5,
            "value_after": 0.6,
        },
        EventType.COMPONENT_STATE_CHANGED: {
            "component_state_before": ComponentState.AVAILABLE,
            "component_state_after": ComponentState.DEGRADED,
        },
        EventType.OBSERVATION_CHANGED: {
            "variable": StateVariable.PRIMARY_FLOW,
            "value_before": 0.5,
            "value_after": 0.45,
            "observation_status": ObservationStatus.WATCH,
        },
        EventType.CHANNEL_QUALITY_CHANGED: {
            "channel_quality_before": ChannelQuality.GOOD,
            "channel_quality": ChannelQuality.SUSPECT,
        },
        EventType.CHANNEL_DISAGREEMENT: {
            "variable": StateVariable.PRIMARY_FLOW,
            "observation_status": ObservationStatus.CONFLICTING,
        },
        EventType.COMMAND_RECORDED: {
            "variable": StateVariable.PRIMARY_FLOW,
            "commanded_value": 0.55,
        },
        EventType.COMMAND_POSITION_MISMATCH: {
            "variable": StateVariable.PRIMARY_FLOW,
            "commanded_value": 0.55,
            "observed_value": 0.45,
        },
        EventType.COMMAND_POSITION_ALIGNED: {
            "variable": StateVariable.PRIMARY_FLOW,
            "commanded_value": 0.55,
            "observed_value": 0.55,
        },
        EventType.ACTION_APPLIED: {},
        EventType.BENIGN_NOTE: {},
    }
    payload = {**common, **payloads[event_type]}
    if event_type is EventType.ACTION_APPLIED:
        return ProjectedEventFact.model_construct(
            fact_ref="e-0000",
            tick=3,
            event_type=EventType.ACTION_APPLIED,
            subject_id="aster-review-component",
        )
    return ProjectedEventFact.model_validate(payload)


def render_catalog_preview() -> CatalogPreviewPacket:
    """Render all catalog combinations for review without assigning release splits.

    This is the sole path allowed to combine holdout template and alias families.  Its
    output is marked review-only and cannot be converted into ``RenderedCandidate``.
    """

    entries: list[CatalogPreviewEntry] = []
    for template_family in TemplateFamily:
        for alias_family in AliasFamily:
            for event_type in EventType:
                fixture = _catalog_event_fixture(event_type)
                template_id, text = _event_line(fixture, template_family, alias_family)
                if _UNFINISHED.search(text):
                    raise ValueError("catalog preview contains an unfinished placeholder")
                assert_no_prohibited_content(text)
                entries.append(
                    CatalogPreviewEntry(
                        template_family_id=template_family,
                        alias_family_id=alias_family,
                        event_type=event_type,
                        template_id=template_id,
                        text=text,
                        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    )
                )
    catalog = catalog_manifest()
    checksum_payload = {
        "preview_status": "catalog_review_only",
        "renderer_version": RENDERER_VERSION,
        "catalog_sha256": catalog.checksum_sha256,
        "entry_count": len(entries),
        "entries": tuple(entry.model_dump(mode="json", round_trip=True) for entry in entries),
    }
    checksum = hashlib.sha256(canonical_json_bytes(checksum_payload)).hexdigest()
    return CatalogPreviewPacket(
        renderer_version=RENDERER_VERSION,
        catalog_sha256=catalog.checksum_sha256,
        entry_count=len(entries),
        entries=tuple(entries),
        packet_sha256=checksum,
    )


def _surface_entry(surface_id: str, text: str) -> AuthoredSurfaceEntry:
    assert_no_prohibited_content(text)
    return AuthoredSurfaceEntry(
        surface_id=surface_id,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _event_surface_fixtures(
    event_type: EventType,
) -> tuple[tuple[str, ProjectedEventFact], ...]:
    default = _catalog_event_fixture(event_type)
    if event_type is not EventType.OBSERVATION_CHANGED:
        return (("default", default),)
    without_before = ProjectedEventFact(
        fact_ref="e-0000",
        tick=3,
        event_type=EventType.OBSERVATION_CHANGED,
        subject_id="aster-review-component",
        variable=StateVariable.PRIMARY_FLOW,
        value_after=0.45,
        observation_status=ObservationStatus.WATCH,
    )
    newly_missing = ProjectedEventFact(
        fact_ref="e-0000",
        tick=3,
        event_type=EventType.OBSERVATION_CHANGED,
        subject_id="aster-review-component",
        variable=StateVariable.PRIMARY_FLOW,
        value_after=None,
        observation_status=ObservationStatus.MISSING,
    )
    return (
        ("with-before", default),
        ("without-before", without_before),
        ("newly-missing", newly_missing),
    )


def renderer_authored_surface_manifest() -> RendererAuthoredSurfaceManifest:
    """Return every renderer-authored phrase/shape in a strict hash-bound manifest."""

    status_phrases = tuple(
        _surface_entry(f"phrase.observation-status.{status.value.casefold()}", _status(status))
        for status in ObservationStatus
    )
    quality_phrases = tuple(
        _surface_entry(f"phrase.channel-quality.{quality.value.casefold()}", _quality(quality))
        for quality in ChannelQuality
    )
    mode_phrases = tuple(
        _surface_entry(f"phrase.operating-mode.{mode.value.casefold()}", _mode(mode))
        for mode in OperatingMode
    )
    state_phrases = tuple(
        _surface_entry(f"phrase.component-state.{state.value.casefold()}", _component_state(state))
        for state in ComponentState
    )
    observation_lines: list[AuthoredSurfaceEntry] = []
    for alias in AliasFamily:
        for quality in ChannelQuality:
            statuses = (
                (ObservationStatus.MISSING,)
                if quality is ChannelQuality.UNAVAILABLE
                else tuple(
                    status
                    for status in ObservationStatus
                    if status is not ObservationStatus.MISSING
                )
            )
            for status in statuses:
                fact = ProjectedObservationFact(
                    fact_ref="o-0000",
                    tick=3,
                    channel_id="aster-review-channel",
                    variable=StateVariable.PRIMARY_FLOW,
                    value=None if status is ObservationStatus.MISSING else 0.45,
                    quality=quality,
                    status=status,
                )
                observation_lines.append(
                    _surface_entry(
                        f"line.observation.{alias.value}.{quality.value.casefold()}."
                        f"{status.value.casefold()}",
                        _observation_line(fact, alias),
                    )
                )
    event_clauses = tuple(
        _surface_entry(
            f"clause.event.{alias.value}.{event_type.value.casefold()}.{variant_id}",
            _event_clauses(fixture, alias),
        )
        for alias in AliasFamily
        for event_type in EventType
        for variant_id, fixture in _event_surface_fixtures(event_type)
    )
    context_lines: list[AuthoredSurfaceEntry] = []
    for alias in AliasFamily:
        for variant in PlantVariant:
            model_input = ModelInput(
                cut_tick=3,
                observation_facts=(
                    ProjectedObservationFact(
                        fact_ref="o-0000",
                        tick=3,
                        channel_id="aster-review-channel",
                        variable=StateVariable.PRIMARY_FLOW,
                        value=0.45,
                        quality=ChannelQuality.GOOD,
                        status=ObservationStatus.NORMAL,
                    ),
                ),
                event_facts=(),
                context_facts=(
                    PlantVariantContextFact(fact_ref="c-0000", plant_variant_id=variant),
                ),
            )
            context_lines.append(
                _surface_entry(
                    f"line.context.general.{alias.value}.{variant.value.casefold()}",
                    _context_lines(model_input, alias)[0],
                )
            )
        for standby_state in (ComponentState.AVAILABLE, ComponentState.UNAVAILABLE):
            for support_state in (ComponentState.AVAILABLE, ComponentState.UNAVAILABLE):
                model_input = ModelInput(
                    cut_tick=3,
                    observation_facts=(
                        ProjectedObservationFact(
                            fact_ref="o-0000",
                            tick=3,
                            channel_id="aster-review-channel",
                            variable=StateVariable.PRIMARY_FLOW,
                            value=0.45,
                            quality=ChannelQuality.GOOD,
                            status=ObservationStatus.NORMAL,
                        ),
                    ),
                    event_facts=(),
                    context_facts=(
                        StandbyRelationshipContextFact(
                            fact_ref="c-0000",
                            active_component_id="aster-review-active",
                            standby_component_id="aster-review-standby",
                            standby_state=standby_state,
                            support_component_id="aster-review-support",
                            support_state=support_state,
                            start_delay_ticks=2,
                        ),
                    ),
                )
                context_lines.append(
                    _surface_entry(
                        f"line.context.g07.{alias.value}."
                        f"{standby_state.value.casefold()}.{support_state.value.casefold()}",
                        _context_lines(model_input, alias)[0],
                    )
                )
        model_input = ModelInput(
            cut_tick=3,
            observation_facts=(
                ProjectedObservationFact(
                    fact_ref="o-0000",
                    tick=3,
                    channel_id="aster-review-channel",
                    variable=StateVariable.PRIMARY_FLOW,
                    value=0.45,
                    quality=ChannelQuality.GOOD,
                    status=ObservationStatus.NORMAL,
                ),
            ),
            event_facts=(),
            context_facts=(
                DependencyLinkContextFact(
                    fact_ref="c-0000",
                    support_component_id="aster-review-support",
                    dependent_component_id="aster-review-dependent",
                ),
            ),
        )
        context_lines.append(
            _surface_entry(f"line.context.g12.{alias.value}", _context_lines(model_input, alias)[0])
        )
    sorted_status = tuple(sorted(status_phrases, key=lambda entry: entry.surface_id))
    sorted_quality = tuple(sorted(quality_phrases, key=lambda entry: entry.surface_id))
    sorted_modes = tuple(sorted(mode_phrases, key=lambda entry: entry.surface_id))
    sorted_states = tuple(sorted(state_phrases, key=lambda entry: entry.surface_id))
    sorted_observations = tuple(sorted(observation_lines, key=lambda entry: entry.surface_id))
    sorted_events = tuple(sorted(event_clauses, key=lambda entry: entry.surface_id))
    sorted_context = tuple(sorted(context_lines, key=lambda entry: entry.surface_id))
    draft = RendererAuthoredSurfaceManifest.model_construct(
        manifest_version="0.1.0",
        observation_status_phrases=sorted_status,
        channel_quality_phrases=sorted_quality,
        operating_mode_phrases=sorted_modes,
        component_state_phrases=sorted_states,
        observation_lines=sorted_observations,
        event_clauses=sorted_events,
        context_lines=sorted_context,
        checksum_sha256="0" * 64,
    )
    checksum = hashlib.sha256(
        canonical_json_bytes(
            draft.model_dump(mode="json", exclude={"checksum_sha256"}, round_trip=True)
        )
    ).hexdigest()
    return RendererAuthoredSurfaceManifest(
        observation_status_phrases=sorted_status,
        channel_quality_phrases=sorted_quality,
        operating_mode_phrases=sorted_modes,
        component_state_phrases=sorted_states,
        observation_lines=sorted_observations,
        event_clauses=sorted_events,
        context_lines=sorted_context,
        checksum_sha256=checksum,
    )


def _ordered_visible_lines(
    model_input: ModelInput, template_family: TemplateFamily, alias_family: AliasFamily
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    entries: list[tuple[int, int, str, str | None]] = []
    for observation_fact in model_input.observation_facts:
        entries.append(
            (
                observation_fact.tick,
                0,
                _observation_line(observation_fact, alias_family),
                None,
            )
        )
    for event_fact in model_input.event_facts:
        template_id, line = _event_line(event_fact, template_family, alias_family)
        entries.append((event_fact.tick, 1, line, template_id))
    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    context = _context_lines(model_input, alias_family)
    lines = (*context, *(item[2] for item in entries))
    template_ids = tuple(dict.fromkeys(item[3] for item in entries if item[3] is not None))
    return tuple(lines), template_ids


def render_model_input(
    model_input: ModelInput,
    *,
    template_family: TemplateFamily,
    alias_family: AliasFamily,
    split_name: SplitName,
) -> RenderedCandidate:
    """Render exactly one strict model input; audit trajectories are never accepted."""

    if type(model_input) is not ModelInput:
        raise TypeError("renderer accepts only an exact dataset ModelInput")
    _validate_family_split(template_family, alias_family, split_name)
    lines, template_ids = _ordered_visible_lines(model_input, template_family, alias_family)
    text = "\n".join(lines)
    if not text or _UNFINISHED.search(text):
        raise ValueError("rendered text is empty or contains an unfinished placeholder")
    assert_no_prohibited_content(text)
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    model_input_sha256 = model_input.structured_fingerprint()
    checksum_payload = {
        "candidate_status": "candidate_pending_postrender_review",
        "renderer_version": RENDERER_VERSION,
        "split_name": split_name,
        "template_family_id": template_family,
        "alias_family_id": alias_family,
        "model_input_sha256": model_input_sha256,
        "template_ids": template_ids,
        "lines": lines,
        "text": text,
    }
    render_id = f"render-{hashlib.sha256(canonical_json_bytes(checksum_payload)).hexdigest()[:24]}"
    return RenderedCandidate(
        candidate_status="candidate_pending_postrender_review",
        renderer_version=RENDERER_VERSION,
        split_name=split_name,
        template_family_id=template_family,
        alias_family_id=alias_family,
        model_input_sha256=model_input_sha256,
        template_ids=template_ids,
        lines=lines,
        text=text,
        text_sha256=text_sha256,
        render_id=render_id,
    )


def render_many(
    model_inputs: Iterable[ModelInput],
    *,
    template_family: TemplateFamily,
    alias_family: AliasFamily,
    split_name: SplitName,
) -> tuple[RenderedCandidate, ...]:
    """Render inputs in caller order while rejecting duplicate structured inputs."""

    rendered = tuple(
        render_model_input(
            model_input,
            template_family=template_family,
            alias_family=alias_family,
            split_name=split_name,
        )
        for model_input in model_inputs
    )
    fingerprints = tuple(item.model_input_sha256 for item in rendered)
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("renderer batch contains duplicate model inputs")
    return rendered
