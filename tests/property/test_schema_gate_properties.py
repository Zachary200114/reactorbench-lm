from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from reactorbench.schemas import (
    AbstentionReason,
    ActionLabel,
    CausalContinuationTarget,
    ComponentState,
    CounterfactualChange,
    CounterfactualComparisonTarget,
    CounterfactualConclusion,
    DecisionTarget,
    DependencyLink,
    DependencyMapContext,
    DiagnosisStatus,
    EventType,
    EvidenceExtractionTarget,
    EvidenceSlot,
    FaultDiagnosisTarget,
    FaultFamily,
    FaultInjection,
    IncidentSummaryTarget,
    LatentPlantState,
    NextActionTarget,
    ObservationFrame,
    ObservationStatus,
    ObservedTrend,
    OperatingMode,
    PlantValues,
    PlantVariant,
    ProvenanceRecord,
    ScenarioDefinition,
    ScenarioDriver,
    ScenarioTargets,
    SeverityBand,
    SplitName,
    StandbyContext,
    StateVariable,
    StructuredTrajectory,
    TaskName,
    TaskTarget,
)
from reactorbench.schemas.events import CanonicalEvent


@st.composite
def fault_windows(draw: st.DrawFn) -> tuple[int, int, int]:
    scenario_duration = draw(st.integers(min_value=1, max_value=64))
    onset_tick = draw(st.integers(min_value=0, max_value=scenario_duration - 1))
    fault_duration = draw(st.integers(min_value=1, max_value=scenario_duration - onset_tick))
    return scenario_duration, onset_tick, fault_duration


@given(window=fault_windows())
def test_fault_duration_boundary_property(window: tuple[int, int, int]) -> None:
    scenario_duration, onset_tick, fault_duration = window
    valid = ScenarioDefinition(
        scenario_id="scenario-property",
        plant_variant_id=PlantVariant.ASTER_A,
        seed=1,
        duration_ticks=scenario_duration,
        driver=ScenarioDriver.STEADY_OPERATION,
        fault_injections=(
            FaultInjection(
                fault_family=FaultFamily.PUMP_TRIP,
                component_id="train-a",
                onset_tick=onset_tick,
                duration_ticks=fault_duration,
                severity=SeverityBand.MEDIUM,
            ),
        ),
    )
    bounded_duration = valid.fault_injections[0].duration_ticks
    assert bounded_duration is not None
    assert onset_tick + bounded_duration <= scenario_duration

    invalid_payload = valid.model_dump()
    invalid_payload["fault_injections"][0]["duration_ticks"] = scenario_duration - onset_tick + 1
    with pytest.raises(ValidationError, match="fault duration"):
        ScenarioDefinition.model_validate(invalid_payload)


@given(
    standby_state=st.sampled_from((ComponentState.AVAILABLE, ComponentState.UNAVAILABLE)),
    support_bus_state=st.sampled_from((ComponentState.AVAILABLE, ComponentState.UNAVAILABLE)),
    start_delay=st.integers(min_value=1, max_value=64),
)
def test_standby_context_round_trip_property(
    standby_state: ComponentState,
    support_bus_state: ComponentState,
    start_delay: int,
) -> None:
    context = StandbyContext(
        context_id="standby-context-property",
        active_train_id="train-cedar",
        standby_train_id="train-hemlock",
        standby_state=standby_state,
        standby_support_bus_id="support-bus-hemlock",
        support_bus_state=support_bus_state,
        standby_start_delay_ticks=start_delay,
    )

    assert StandbyContext.model_validate_json(context.model_dump_json()) == context


@given(
    field_name=st.sampled_from(("standby_state", "support_bus_state")),
    invalid_state=st.sampled_from(
        tuple(
            state
            for state in ComponentState
            if state not in {ComponentState.AVAILABLE, ComponentState.UNAVAILABLE}
        )
    ),
)
def test_standby_context_invalid_state_property(
    field_name: str, invalid_state: ComponentState
) -> None:
    context = StandbyContext(
        context_id="standby-context-property",
        active_train_id="train-cedar",
        standby_train_id="train-hemlock",
        standby_state=ComponentState.AVAILABLE,
        standby_support_bus_id="support-bus-hemlock",
        support_bus_state=ComponentState.AVAILABLE,
        standby_start_delay_ticks=2,
    )
    payload = context.model_dump()
    payload[field_name] = invalid_state

    with pytest.raises(ValidationError, match=f"{field_name} must be AVAILABLE or UNAVAILABLE"):
        StandbyContext.model_validate(payload)


@given(
    variant=st.sampled_from(tuple(PlantVariant)),
    bus_suffixes=st.lists(
        st.integers(min_value=0, max_value=999), min_size=1, max_size=12, unique=True
    ),
)
def test_dependency_map_context_round_trip_property(
    variant: PlantVariant, bus_suffixes: list[int]
) -> None:
    links = tuple(
        DependencyLink(
            support_bus_id=f"bus-{suffix:03d}",
            dependent_component_id=f"dependent-{suffix:03d}",
        )
        for suffix in bus_suffixes
    )
    context = DependencyMapContext(
        plant_variant_id=variant,
        links=tuple(
            sorted(links, key=lambda link: (link.support_bus_id, link.dependent_component_id))
        ),
    )

    assert DependencyMapContext.model_validate_json(context.model_dump_json()) == context


def _property_trajectory(duration: int) -> StructuredTrajectory:
    values = PlantValues.model_validate({variable.value: 0.5 for variable in StateVariable})
    scenario = ScenarioDefinition(
        scenario_id="scenario-property",
        plant_variant_id=PlantVariant.ASTER_B,
        seed=11,
        duration_ticks=duration,
        driver=ScenarioDriver.STEADY_OPERATION,
    )
    event = CanonicalEvent(
        event_id="evt-0",
        event_index=0,
        sim_time=0,
        event_type=EventType.BENIGN_NOTE,
        subject_id="aster-b",
        evidence_slots=(EvidenceSlot.STABLE_OPERATION,),
    )
    return StructuredTrajectory(
        trajectory_id="trajectory-property",
        scenario_id="scenario-property",
        scenario=scenario,
        provenance=ProvenanceRecord(
            dataset_version="0.1.0",
            generator_commit="abcdef1",
            renderer_version="0.1.0",
            seed=11,
            trajectory_id="trajectory-property",
            scenario_id="scenario-property",
            plant_variant_id=PlantVariant.ASTER_B,
            template_family_ids=("template-property",),
            split_name=SplitName.IID_TRAIN,
            task_name=TaskName.FAULT_FAMILY,
        ),
        latent_states=tuple(
            LatentPlantState(
                tick=tick,
                operating_mode=OperatingMode.STABLE,
                values=values,
                components=(),
            )
            for tick in range(duration)
        ),
        observations=tuple(
            ObservationFrame(
                tick=tick,
                overall_status=ObservationStatus.NORMAL,
                channels=(),
            )
            for tick in range(duration)
        ),
        events=(event,),
        targets=ScenarioTargets(
            scenario_id="scenario-property",
            decisions=(
                DecisionTarget(
                    scenario_id="scenario-property",
                    decision_tick=duration - 1,
                    diagnosis_status=DiagnosisStatus.NO_FAULT,
                    evidence_event_ids=("evt-0",),
                    evidence_slots=(EvidenceSlot.STABLE_OPERATION,),
                    immediate_action=ActionLabel.CONTINUE_MONITORING,
                ),
            ),
        ),
    )


@given(duration=st.integers(min_value=1, max_value=24))
def test_aligned_tick_coverage_property(duration: int) -> None:
    trajectory = _property_trajectory(duration)
    assert tuple(state.tick for state in trajectory.latent_states) == tuple(range(duration))
    assert tuple(frame.tick for frame in trajectory.observations) == tuple(range(duration))

    payload = trajectory.model_dump()
    payload["observations"] = payload["observations"][:-1]
    with pytest.raises(ValidationError, match="one ordered frame"):
        StructuredTrajectory.model_validate(payload)


_UNRESOLVED = CounterfactualConclusion(
    diagnosis_status=DiagnosisStatus.UNRESOLVED,
    evidence_slots=(EvidenceSlot.MISSING_DECISIVE_EVIDENCE,),
    immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE,
    abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
)
_DIAGNOSED = CounterfactualConclusion(
    diagnosis_status=DiagnosisStatus.DIAGNOSED,
    fault_labels=(FaultFamily.SENSOR_DRIFT,),
    evidence_slots=(EvidenceSlot.CHANNEL_DISAGREEMENT,),
    immediate_action=ActionLabel.VERIFY_REDUNDANT_CHANNEL,
)
_TASK_TARGETS = (
    CausalContinuationTarget(next_event_type=EventType.BENIGN_NOTE),
    FaultDiagnosisTarget(diagnosis_status=DiagnosisStatus.NO_FAULT),
    EvidenceExtractionTarget(evidence_slots=(EvidenceSlot.STABLE_OPERATION,)),
    NextActionTarget(immediate_action=ActionLabel.CONTINUE_MONITORING),
    IncidentSummaryTarget(
        affected_subsystems=(),
        observed_trend=ObservedTrend.STABLE,
        diagnosis_status=DiagnosisStatus.NO_FAULT,
        operating_mode=OperatingMode.STABLE,
        immediate_action=ActionLabel.CONTINUE_MONITORING,
    ),
    CounterfactualComparisonTarget(
        baseline=_UNRESOLVED,
        counterfactual=_DIAGNOSED,
        changed_fields=tuple(CounterfactualChange),
        decisive_evidence_slots=(EvidenceSlot.CHANNEL_DISAGREEMENT,),
    ),
)
_MISMATCHED_TASK_TARGETS = tuple(
    (task_name, target)
    for task_name in TaskName
    for target in _TASK_TARGETS
    if task_name is not target.task_name
)


@given(pair=st.sampled_from(_MISMATCHED_TASK_TARGETS))
def test_discriminated_task_mismatch_property(pair: tuple[TaskName, object]) -> None:
    task_name, target = pair
    with pytest.raises(ValidationError, match="task_name must match"):
        TaskTarget.model_validate({"task_name": task_name, "target": target})
