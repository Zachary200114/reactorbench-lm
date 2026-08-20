from __future__ import annotations

import pytest

from reactorbench.dataset.catalog import (
    TEMPLATE_BY_KEY,
    AliasFamily,
    TemplateFamily,
    catalog_manifest,
)
from reactorbench.dataset.contracts import (
    ModelInput,
    PlantVariantContextFact,
    ProjectedEventFact,
    ProjectedObservationFact,
)
from reactorbench.dataset.renderer import (
    render_catalog_preview,
    render_model_input,
    renderer_authored_surface_manifest,
)
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


def _event(event_type: EventType) -> ProjectedEventFact:
    common = {
        "fact_ref": "e-0000",
        "tick": 3,
        "event_type": event_type,
        "subject_id": "aster-train-cirrus",
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
        EventType.BENIGN_NOTE: {},
    }
    assert event_type is not EventType.ACTION_APPLIED
    return ProjectedEventFact.model_validate({**common, **payloads[event_type]})


def _model_input(event_type: EventType = EventType.OBSERVATION_CHANGED) -> ModelInput:
    return ModelInput(
        cut_tick=3,
        observation_facts=(
            ProjectedObservationFact(
                fact_ref="o-0000",
                tick=3,
                channel_id="aster-primary-flow-a",
                variable=StateVariable.PRIMARY_FLOW,
                value=0.45,
                quality=ChannelQuality.GOOD,
                status=ObservationStatus.WATCH,
            ),
        ),
        event_facts=(_event(event_type),),
        context_facts=(
            PlantVariantContextFact(fact_ref="c-0000", plant_variant_id=PlantVariant.ASTER_A),
        ),
    )


def test_catalog_has_four_complete_stable_template_families() -> None:
    assert len(TemplateFamily) == 4
    assert set(TEMPLATE_BY_KEY) == {
        (family, event_type) for family in TemplateFamily for event_type in EventType
    }
    manifest = catalog_manifest()
    assert len(manifest.templates) == len(TemplateFamily) * len(EventType)
    assert len({template.template_id for template in manifest.templates}) == len(manifest.templates)
    assert len(manifest.checksum_sha256) == 64

    preview = render_catalog_preview()
    assert preview.preview_status == "catalog_review_only"
    assert preview.entry_count == len(TemplateFamily) * len(AliasFamily) * len(EventType)
    assert {
        (entry.template_family_id, entry.alias_family_id, entry.event_type)
        for entry in preview.entries
    } == {
        (template, alias, event)
        for template in TemplateFamily
        for alias in AliasFamily
        for event in EventType
    }
    assert all("[e-0000]" in entry.text for entry in preview.entries)
    assert all("authored " not in entry.text for entry in preview.entries)
    assert any(
        entry.event_type is EventType.ACTION_APPLIED
        and "prior fictional label application" in entry.text
        for entry in preview.entries
    )


def test_catalog_action_fixture_is_review_only_and_production_contract_still_rejects_it() -> None:
    with pytest.raises(ValueError, match="never a renderer-safe"):
        ProjectedEventFact(
            fact_ref="e-0000",
            tick=3,
            event_type=EventType.ACTION_APPLIED,
            subject_id="aster-review-component",
        )


def test_authored_surface_manifest_is_exhaustive_and_hash_bound() -> None:
    manifest = renderer_authored_surface_manifest()

    assert len(manifest.observation_status_phrases) == len(ObservationStatus)
    assert len(manifest.channel_quality_phrases) == len(ChannelQuality)
    assert len(manifest.operating_mode_phrases) == len(OperatingMode)
    assert len(manifest.component_state_phrases) == len(ComponentState)
    assert len(manifest.event_clauses) == len(AliasFamily) * (len(EventType) + 2)
    assert len(manifest.context_lines) == len(AliasFamily) * (len(PlantVariant) + 4 + 1)
    assert all(entry.text_sha256 for entry in manifest.event_clauses)
    assert len(manifest.checksum_sha256) == 64


@pytest.mark.parametrize(
    "event_type", tuple(event for event in EventType if event is not EventType.ACTION_APPLIED)
)
def test_renderer_covers_every_renderer_safe_event_without_placeholders(
    event_type: EventType,
) -> None:
    candidate = render_model_input(
        _model_input(event_type),
        template_family=TemplateFamily.COMPACT_LOG,
        alias_family=AliasFamily.CANONICAL,
        split_name=SplitName.IID_TRAIN,
    )

    assert candidate.text
    assert "{" not in candidate.text
    assert "${" not in candidate.text
    assert "<" not in candidate.text
    assert candidate.template_ids == (
        TEMPLATE_BY_KEY[(TemplateFamily.COMPACT_LOG, event_type)].template_id,
    )
    assert "[o-0000]" in candidate.text
    assert "[e-0000]" in candidate.text
    assert "[c-0000]" in candidate.text


def test_rendering_is_deterministic_and_alias_selection_is_independent() -> None:
    model_input = _model_input()
    canonical = render_model_input(
        model_input,
        template_family=TemplateFamily.OBSERVER_NOTE,
        alias_family=AliasFamily.CANONICAL,
        split_name=SplitName.IID_TEST,
    )
    replay = render_model_input(
        model_input,
        template_family=TemplateFamily.OBSERVER_NOTE,
        alias_family=AliasFamily.CANONICAL,
        split_name=SplitName.IID_TEST,
    )
    short = render_model_input(
        model_input,
        template_family=TemplateFamily.OBSERVER_NOTE,
        alias_family=AliasFamily.SHORT,
        split_name=SplitName.IID_TEST,
    )

    assert canonical == replay
    assert canonical.render_id == replay.render_id
    assert canonical.text != short.text
    assert canonical.template_ids == short.template_ids
    assert canonical.model_input_sha256 == short.model_input_sha256


def test_holdout_template_and_alias_families_are_split_exclusive() -> None:
    model_input = _model_input()
    editorial = render_model_input(
        model_input,
        template_family=TemplateFamily.RESEARCH_EDITORIAL,
        alias_family=AliasFamily.NEUTRAL,
        split_name=SplitName.TEMPLATE_TEST,
    )
    heldout_alias = render_model_input(
        model_input,
        template_family=TemplateFamily.SHIFT_LEDGER,
        alias_family=AliasFamily.HELDOUT,
        split_name=SplitName.COMPONENT_TEST,
    )

    assert editorial.template_family_id is TemplateFamily.RESEARCH_EDITORIAL
    assert heldout_alias.alias_family_id is AliasFamily.HELDOUT
    with pytest.raises(ValueError, match="reserved"):
        render_model_input(
            model_input,
            template_family=TemplateFamily.RESEARCH_EDITORIAL,
            alias_family=AliasFamily.CANONICAL,
            split_name=SplitName.IID_TRAIN,
        )
    with pytest.raises(ValueError, match="reserved"):
        render_model_input(
            model_input,
            template_family=TemplateFamily.COMPACT_LOG,
            alias_family=AliasFamily.HELDOUT,
            split_name=SplitName.IID_TRAIN,
        )


def test_renderer_rejects_non_model_input_instead_of_accepting_audit_objects() -> None:
    with pytest.raises(TypeError, match="only an exact dataset ModelInput"):
        render_model_input(
            object(),  # type: ignore[arg-type]
            template_family=TemplateFamily.COMPACT_LOG,
            alias_family=AliasFamily.CANONICAL,
            split_name=SplitName.IID_TRAIN,
        )
