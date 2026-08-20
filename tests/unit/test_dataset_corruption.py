from __future__ import annotations

import pytest

from reactorbench.dataset.catalog import AliasFamily, TemplateFamily
from reactorbench.dataset.contracts import ModelInput, ProjectedEventFact, ProjectedObservationFact
from reactorbench.dataset.corruption import (
    CorruptionPlan,
    apply_narrative_corruption,
    corruption_authored_surface_manifest,
    materialize_corrupted_candidate,
)
from reactorbench.dataset.renderer import RenderedCandidate, render_model_input
from reactorbench.schemas.enums import (
    ChannelQuality,
    EventType,
    ObservationStatus,
    SplitName,
    StateVariable,
)


def _input() -> ModelInput:
    return ModelInput(
        cut_tick=3,
        observation_facts=(
            ProjectedObservationFact(
                fact_ref="o-0000",
                tick=3,
                channel_id="aster-channel-a",
                variable=StateVariable.PRIMARY_FLOW,
                value=0.48,
                quality=ChannelQuality.GOOD,
                status=ObservationStatus.WATCH,
            ),
            ProjectedObservationFact(
                fact_ref="o-0001",
                tick=3,
                channel_id="aster-channel-b",
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
                subject_id="aster-domain-a",
            ),
            ProjectedEventFact(
                fact_ref="e-0001",
                tick=3,
                event_type=EventType.OBSERVATION_CHANGED,
                subject_id="aster-channel-a",
                variable=StateVariable.PRIMARY_FLOW,
                value_before=0.5,
                value_after=0.48,
                observation_status=ObservationStatus.WATCH,
            ),
        ),
        context_facts=(),
    )


def _candidate(model_input: ModelInput) -> RenderedCandidate:
    return render_model_input(
        model_input,
        template_family=TemplateFamily.COMPACT_LOG,
        alias_family=AliasFamily.CANONICAL,
        split_name=SplitName.NOISE_TEST,
    )


@pytest.mark.parametrize(
    "plan",
    [
        CorruptionPlan.OMIT_NONCRITICAL,
        CorruptionPlan.DUPLICATE_LINE,
        CorruptionPlan.BENIGN_INSERT,
        CorruptionPlan.SAFE_REORDER,
    ],
)
def test_corruption_is_deterministic_and_preserves_structured_identity(
    plan: CorruptionPlan,
) -> None:
    model_input = _input()
    candidate = _candidate(model_input)
    first = apply_narrative_corruption(
        candidate,
        model_input,
        plan=plan,
        protected_fact_refs=("e-0001",),
    )
    second = apply_narrative_corruption(
        candidate,
        model_input,
        plan=plan,
        protected_fact_refs=("e-0001",),
    )
    assert first == second
    assert first.model_input_sha256 == candidate.model_input_sha256
    assert first.text != candidate.text
    assert first.corruption_plan is plan
    materialized = materialize_corrupted_candidate(candidate, first)
    assert materialized.lines == first.lines
    assert materialized.text_sha256 == first.text_sha256
    assert materialized.model_input_sha256 == candidate.model_input_sha256
    assert materialized.render_id != candidate.render_id


def test_omission_cannot_remove_a_protected_or_unknown_fact() -> None:
    model_input = _input()
    candidate = _candidate(model_input)
    with pytest.raises(ValueError, match="unprotected"):
        apply_narrative_corruption(
            candidate,
            model_input,
            plan=CorruptionPlan.OMIT_NONCRITICAL,
            protected_fact_refs=("e-0000", "e-0001"),
        )
    with pytest.raises(ValueError, match="resolve"):
        apply_narrative_corruption(
            candidate,
            model_input,
            plan=CorruptionPlan.DUPLICATE_LINE,
            protected_fact_refs=("e-9999",),
        )


def test_visible_fact_refs_keep_omission_exact_and_review_surface_complete() -> None:
    model_input = _input()
    candidate = _candidate(model_input)
    omitted = apply_narrative_corruption(
        candidate,
        model_input,
        plan=CorruptionPlan.OMIT_NONCRITICAL,
        protected_fact_refs=("e-0001",),
    )

    assert "[e-0000]" in candidate.text
    assert "[e-0000]" not in omitted.text
    assert "[e-0001]" in omitted.text
    manifest = corruption_authored_surface_manifest()
    assert tuple(surface.corruption_plan for surface in manifest.surfaces) == tuple(CorruptionPlan)
    benign = next(
        surface
        for surface in manifest.surfaces
        if surface.corruption_plan is CorruptionPlan.BENIGN_INSERT
    )
    assert "bounded distractor" in benign.output_text
    assert len(manifest.checksum_sha256) == 64
