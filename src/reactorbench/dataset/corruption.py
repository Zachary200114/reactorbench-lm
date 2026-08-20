"""Deterministic, evidence-preserving narrative corruption for ``noise_test``."""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from reactorbench.schemas.base import ContractModel, canonical_json_bytes
from reactorbench.schemas.enums import (
    ChannelQuality,
    EventType,
    ObservationStatus,
    SplitName,
    StateVariable,
)

from .catalog import AliasFamily, TemplateFamily
from .content_guard import assert_no_prohibited_content
from .contracts import ModelInput, ProjectedEventFact, ProjectedObservationFact
from .renderer import RenderedCandidate, _event_line, render_model_input

_TIMESTAMP = re.compile(r"^\[T\+(\d{3})\]")


class CorruptionPlan(StrEnum):
    OMIT_NONCRITICAL = "omit_noncritical"
    DUPLICATE_LINE = "duplicate_line"
    BENIGN_INSERT = "benign_insert"
    SAFE_REORDER = "safe_reorder"


class CorruptionReviewSurface(ContractModel):
    """One canonical operation fixture showing exact before/after narrative text."""

    corruption_plan: CorruptionPlan
    operation_policy: str = Field(min_length=1, max_length=512)
    base_text: str = Field(min_length=1, max_length=65_536)
    output_text: str = Field(min_length=1, max_length=65_536)
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def output_checksum_and_change_match(self) -> CorruptionReviewSurface:
        assert_no_prohibited_content((self.operation_policy, self.base_text, self.output_text))
        if self.base_text == self.output_text:
            raise ValueError("corruption review surface must visibly change the fixture")
        if hashlib.sha256(self.output_text.encode("utf-8")).hexdigest() != self.output_sha256:
            raise ValueError("corruption review surface checksum mismatch")
        return self


class CorruptionAuthoredSurfaceManifest(ContractModel):
    """Strict owner-review inventory for every allowed narrative corruption."""

    manifest_version: Literal["0.1.0"] = "0.1.0"
    surfaces: tuple[CorruptionReviewSurface, ...]
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def coverage_order_and_checksum_match(self) -> CorruptionAuthoredSurfaceManifest:
        plans = tuple(surface.corruption_plan for surface in self.surfaces)
        if plans != tuple(CorruptionPlan):
            raise ValueError("corruption review surfaces must cover plans in canonical order")
        expected = hashlib.sha256(
            canonical_json_bytes(
                self.model_dump(mode="json", exclude={"checksum_sha256"}, round_trip=True)
            )
        ).hexdigest()
        if self.checksum_sha256 != expected:
            raise ValueError("corruption authored surface manifest checksum mismatch")
        return self


class CorruptedCandidate(ContractModel):
    """A pending-review narrative derived without changing structured truth."""

    candidate_status: Literal["candidate_pending_postrender_review"] = (
        "candidate_pending_postrender_review"
    )
    base_render_id: str = Field(pattern=r"^render-[0-9a-f]{24}$")
    corruption_plan: CorruptionPlan
    model_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lines: tuple[str, ...] = Field(min_length=1, max_length=8192)
    text: str = Field(min_length=1, max_length=65_536)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corruption_id: str = Field(pattern=r"^corrupt-[0-9a-f]{24}$")

    @model_validator(mode="after")
    def derived_content_matches(self) -> CorruptedCandidate:
        if self.text != "\n".join(self.lines):
            raise ValueError("corrupted text must be the newline join of lines")
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.text_sha256:
            raise ValueError("corrupted text checksum mismatch")
        payload = self.model_dump(
            mode="json",
            exclude={"corruption_id", "text_sha256"},
            round_trip=True,
        )
        expected = f"corrupt-{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:24]}"
        if self.corruption_id != expected:
            raise ValueError("corruption ID does not match content")
        return self


def _single_event_line(candidate: RenderedCandidate, fact: ProjectedEventFact) -> str:
    _, line = _event_line(
        fact,
        candidate.template_family_id,
        candidate.alias_family_id,
    )
    return line


def _omit_noncritical(
    candidate: RenderedCandidate,
    model_input: ModelInput,
    protected_fact_refs: frozenset[str],
) -> tuple[str, ...]:
    eligible = tuple(
        fact
        for fact in model_input.event_facts
        if fact.event_type is EventType.BENIGN_NOTE and fact.fact_ref not in protected_fact_refs
    )
    if not eligible:
        raise ValueError("omit_noncritical requires an unprotected benign-note fact")
    omitted_line = _single_event_line(candidate, eligible[0])
    matches = tuple(index for index, line in enumerate(candidate.lines) if line == omitted_line)
    if len(matches) != 1:
        raise ValueError("the selected noncritical line must resolve exactly once")
    return tuple(line for index, line in enumerate(candidate.lines) if index != matches[0])


def _duplicate_line(candidate: RenderedCandidate) -> tuple[str, ...]:
    visible_indexes = tuple(
        index for index, line in enumerate(candidate.lines) if _TIMESTAMP.match(line)
    )
    if not visible_indexes:
        raise ValueError("duplicate_line requires a timestamped narrative line")
    index = visible_indexes[-1]
    return (*candidate.lines[: index + 1], candidate.lines[index], *candidate.lines[index + 1 :])


def _benign_insert(candidate: RenderedCandidate, model_input: ModelInput) -> tuple[str, ...]:
    line = (
        f"[T+{model_input.cut_tick:03d}] bounded distractor: an unrelated fictional "
        "indicator remained within its expected band."
    )
    return (*candidate.lines, line)


def _safe_reorder(candidate: RenderedCandidate) -> tuple[str, ...]:
    lines = list(candidate.lines)
    for index in range(len(lines) - 1):
        left = _TIMESTAMP.match(lines[index])
        right = _TIMESTAMP.match(lines[index + 1])
        if left is not None and right is not None and left.group(1) == right.group(1):
            lines[index], lines[index + 1] = lines[index + 1], lines[index]
            return tuple(lines)
    raise ValueError("safe_reorder requires two adjacent facts with the same timestamp")


def apply_narrative_corruption(
    candidate: RenderedCandidate,
    model_input: ModelInput,
    *,
    plan: CorruptionPlan,
    protected_fact_refs: tuple[str, ...] = (),
) -> CorruptedCandidate:
    """Apply one bounded corruption without changing the structured target.

    Line omission is allowed only for a projected benign note that is absent from the
    caller-supplied evidence reference set. Other transforms preserve or add facts and
    therefore do not require target recomputation.
    """

    if type(candidate) is not RenderedCandidate or type(model_input) is not ModelInput:
        raise TypeError("corruption requires canonical rendered and model-input contracts")
    if type(plan) is not CorruptionPlan:
        raise ValueError("plan must be a CorruptionPlan")
    if candidate.model_input_sha256 != model_input.structured_fingerprint():
        raise ValueError("candidate and model input fingerprints do not match")
    if len(protected_fact_refs) != len(set(protected_fact_refs)):
        raise ValueError("protected fact references must be unique")
    visible_refs = {
        *(fact.fact_ref for fact in model_input.observation_facts),
        *(fact.fact_ref for fact in model_input.event_facts),
        *(fact.fact_ref for fact in model_input.context_facts),
    }
    if not set(protected_fact_refs).issubset(visible_refs):
        raise ValueError("protected fact references must resolve inside the model input")

    if plan is CorruptionPlan.OMIT_NONCRITICAL:
        lines = _omit_noncritical(candidate, model_input, frozenset(protected_fact_refs))
    elif plan is CorruptionPlan.DUPLICATE_LINE:
        lines = _duplicate_line(candidate)
    elif plan is CorruptionPlan.BENIGN_INSERT:
        lines = _benign_insert(candidate, model_input)
    else:
        lines = _safe_reorder(candidate)
    text = "\n".join(lines)
    assert_no_prohibited_content(text)
    payload = {
        "candidate_status": "candidate_pending_postrender_review",
        "base_render_id": candidate.render_id,
        "corruption_plan": plan,
        "model_input_sha256": candidate.model_input_sha256,
        "lines": lines,
        "text": text,
    }
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    corruption_id = f"corrupt-{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:24]}"
    return CorruptedCandidate(
        base_render_id=candidate.render_id,
        corruption_plan=plan,
        model_input_sha256=candidate.model_input_sha256,
        lines=lines,
        text=text,
        text_sha256=text_sha256,
        corruption_id=corruption_id,
    )


def materialize_corrupted_candidate(
    base: RenderedCandidate,
    corruption: CorruptedCandidate,
) -> RenderedCandidate:
    """Convert an audited corruption into the common pending-review candidate shape."""

    if type(base) is not RenderedCandidate or type(corruption) is not CorruptedCandidate:
        raise TypeError("materialization requires canonical base and corruption records")
    if (
        corruption.base_render_id != base.render_id
        or corruption.model_input_sha256 != base.model_input_sha256
    ):
        raise ValueError("corruption does not reference the supplied base candidate")
    payload = {
        "candidate_status": "candidate_pending_postrender_review",
        "renderer_version": base.renderer_version,
        "split_name": base.split_name,
        "template_family_id": base.template_family_id,
        "alias_family_id": base.alias_family_id,
        "model_input_sha256": base.model_input_sha256,
        "template_ids": base.template_ids,
        "lines": corruption.lines,
        "text": corruption.text,
    }
    render_id = f"render-{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:24]}"
    return RenderedCandidate(
        renderer_version=base.renderer_version,
        split_name=base.split_name,
        template_family_id=base.template_family_id,
        alias_family_id=base.alias_family_id,
        model_input_sha256=base.model_input_sha256,
        template_ids=base.template_ids,
        lines=corruption.lines,
        text=corruption.text,
        text_sha256=corruption.text_sha256,
        render_id=render_id,
    )


def corruption_authored_surface_manifest() -> CorruptionAuthoredSurfaceManifest:
    """Render canonical fixtures that expose every authored corruption operation."""

    model_input = ModelInput(
        cut_tick=3,
        observation_facts=(
            ProjectedObservationFact(
                fact_ref="o-0000",
                tick=3,
                channel_id="aster-review-channel-a",
                variable=StateVariable.PRIMARY_FLOW,
                value=0.48,
                quality=ChannelQuality.GOOD,
                status=ObservationStatus.WATCH,
            ),
            ProjectedObservationFact(
                fact_ref="o-0001",
                tick=3,
                channel_id="aster-review-channel-b",
                variable=StateVariable.PRIMARY_FLOW,
                value=0.49,
                quality=ChannelQuality.GOOD,
                status=ObservationStatus.NORMAL,
            ),
        ),
        event_facts=(
            ProjectedEventFact(
                fact_ref="e-0000",
                tick=0,
                event_type=EventType.BENIGN_NOTE,
                subject_id="aster-review-domain",
            ),
            ProjectedEventFact(
                fact_ref="e-0001",
                tick=3,
                event_type=EventType.OBSERVATION_CHANGED,
                subject_id="aster-review-channel-a",
                variable=StateVariable.PRIMARY_FLOW,
                value_before=0.5,
                value_after=0.48,
                observation_status=ObservationStatus.WATCH,
            ),
        ),
        context_facts=(),
    )
    base = render_model_input(
        model_input,
        template_family=TemplateFamily.COMPACT_LOG,
        alias_family=AliasFamily.CANONICAL,
        split_name=SplitName.NOISE_TEST,
    )
    policies = {
        CorruptionPlan.OMIT_NONCRITICAL: (
            "Remove exactly one unprotected BENIGN_NOTE line by its complete canonical "
            "event rendering; all prompt fact references remain explicit."
        ),
        CorruptionPlan.DUPLICATE_LINE: (
            "Duplicate the final timestamped narrative line without changing structured truth."
        ),
        CorruptionPlan.BENIGN_INSERT: (
            "Append one fixed fictional bounded-distractor sentence at the prompt cut tick."
        ),
        CorruptionPlan.SAFE_REORDER: (
            "Swap the first adjacent pair of timestamped lines sharing the same tick."
        ),
    }
    review_surfaces: list[CorruptionReviewSurface] = []
    for plan in CorruptionPlan:
        transformed = apply_narrative_corruption(
            base,
            model_input,
            plan=plan,
            protected_fact_refs=("e-0001",),
        )
        review_surfaces.append(
            CorruptionReviewSurface(
                corruption_plan=plan,
                operation_policy=policies[plan],
                base_text=base.text,
                output_text=transformed.text,
                output_sha256=transformed.text_sha256,
            )
        )
    surfaces = tuple(review_surfaces)
    draft = CorruptionAuthoredSurfaceManifest.model_construct(
        manifest_version="0.1.0",
        surfaces=surfaces,
        checksum_sha256="0" * 64,
    )
    checksum = hashlib.sha256(
        canonical_json_bytes(
            draft.model_dump(mode="json", exclude={"checksum_sha256"}, round_trip=True)
        )
    ).hexdigest()
    return CorruptionAuthoredSurfaceManifest(surfaces=surfaces, checksum_sha256=checksum)


__all__ = [
    "CorruptedCandidate",
    "CorruptionAuthoredSurfaceManifest",
    "CorruptionPlan",
    "CorruptionReviewSurface",
    "apply_narrative_corruption",
    "corruption_authored_surface_manifest",
    "materialize_corrupted_candidate",
]
