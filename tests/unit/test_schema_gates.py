from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from reactorbench.schemas import (
    EVENT_FIELD_MATRIX,
    AbstentionReason,
    ActionLabel,
    AsterSubsystem,
    CanonicalEvent,
    CausalContinuationTarget,
    ChannelQuality,
    ComponentLatentState,
    ComponentState,
    CounterfactualChange,
    CounterfactualComparisonTarget,
    CounterfactualConclusion,
    DecisionTarget,
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
    SensorChannelObservation,
    SeverityBand,
    SplitName,
    StateVariable,
    StructuredTrajectory,
    TaskName,
    TaskTarget,
)


def _event_payloads() -> dict[EventType, dict[str, object]]:
    return {
        EventType.OPERATING_MODE_CHANGED: {
            "operating_mode_before": OperatingMode.STABLE,
            "operating_mode_after": OperatingMode.DISTURBED,
        },
        EventType.TARGET_CHANGED: {
            "variable": StateVariable.LOAD_DEMAND,
            "value_before": 0.4,
            "value_after": 0.6,
        },
        EventType.COMPONENT_STATE_CHANGED: {
            "component_state_before": ComponentState.AVAILABLE,
            "component_state_after": ComponentState.DEGRADED,
        },
        EventType.OBSERVATION_CHANGED: {
            "variable": StateVariable.PRIMARY_FLOW,
            "value_before": 0.5,
            "value_after": 0.4,
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
            "variable": StateVariable.LOAD_DEMAND,
            "commanded_value": 0.4,
        },
        EventType.COMMAND_POSITION_MISMATCH: {
            "variable": StateVariable.PRIMARY_FLOW,
            "commanded_value": 0.6,
            "observed_value": 0.4,
        },
        EventType.ACTION_APPLIED: {"action_label": ActionLabel.CONTINUE_MONITORING},
        EventType.BENIGN_NOTE: {},
    }


def _event(event_type: EventType, *, index: int = 0, tick: int = 0) -> CanonicalEvent:
    payload: dict[str, object] = {
        "event_id": f"evt-{index}",
        "event_index": index,
        "sim_time": tick,
        "event_type": event_type,
        "subject_id": "subject-a",
    }
    payload.update(_event_payloads()[event_type])
    return CanonicalEvent.model_validate(payload)


def _values() -> PlantValues:
    return PlantValues.model_validate({variable.value: 0.5 for variable in StateVariable})


def _latent(tick: int) -> LatentPlantState:
    return LatentPlantState(
        tick=tick,
        operating_mode=OperatingMode.STABLE,
        values=_values(),
        components=(
            ComponentLatentState(
                component_id="train-a",
                state=ComponentState.AVAILABLE,
                health=1.0,
            ),
        ),
    )


def _observation(tick: int) -> ObservationFrame:
    return ObservationFrame(
        tick=tick,
        overall_status=ObservationStatus.NORMAL,
        channels=(
            SensorChannelObservation(
                channel_id="pf-a",
                variable=StateVariable.PRIMARY_FLOW,
                value=0.5,
                quality=ChannelQuality.GOOD,
                status=ObservationStatus.NORMAL,
            ),
        ),
    )


def _provenance() -> ProvenanceRecord:
    return ProvenanceRecord(
        dataset_version="0.1.0",
        generator_commit="abcdef1",
        renderer_version="0.1.0",
        seed=7,
        trajectory_id="trajectory-a",
        scenario_id="scenario-a",
        plant_variant_id=PlantVariant.ASTER_A,
        template_family_ids=("template-a",),
        split_name=SplitName.IID_TRAIN,
        task_name=TaskName.FAULT_FAMILY,
    )


def _trajectory() -> StructuredTrajectory:
    scenario = ScenarioDefinition(
        scenario_id="scenario-a",
        plant_variant_id=PlantVariant.ASTER_A,
        seed=7,
        duration_ticks=2,
        driver=ScenarioDriver.STEADY_OPERATION,
    )
    event = CanonicalEvent(
        event_id="evt-0",
        event_index=0,
        sim_time=0,
        event_type=EventType.BENIGN_NOTE,
        subject_id="aster-a",
        evidence_slots=(EvidenceSlot.STABLE_OPERATION,),
    )
    targets = ScenarioTargets(
        scenario_id="scenario-a",
        decisions=(
            DecisionTarget(
                scenario_id="scenario-a",
                decision_tick=0,
                diagnosis_status=DiagnosisStatus.NO_FAULT,
                evidence_event_ids=("evt-0",),
                evidence_slots=(EvidenceSlot.STABLE_OPERATION,),
                immediate_action=ActionLabel.CONTINUE_MONITORING,
            ),
        ),
    )
    return StructuredTrajectory(
        trajectory_id="trajectory-a",
        scenario_id="scenario-a",
        scenario=scenario,
        provenance=_provenance(),
        latent_states=(_latent(0), _latent(1)),
        observations=(_observation(0), _observation(1)),
        events=(event,),
        targets=targets,
    )


@pytest.mark.parametrize("event_type", list(EventType))
def test_every_event_type_has_one_exact_payload_contract(event_type: EventType) -> None:
    assert set(EVENT_FIELD_MATRIX) == set(EventType)
    event = _event(event_type)
    assert event.event_type is event_type

    required = EVENT_FIELD_MATRIX[event_type].required
    if required:
        payload = event.model_dump()
        payload.pop(next(iter(required)))
        with pytest.raises(ValidationError, match="requires fields"):
            CanonicalEvent.model_validate(payload)


@pytest.mark.parametrize("event_type", list(EventType))
def test_event_matrix_rejects_a_field_from_another_payload(event_type: EventType) -> None:
    candidate_values: dict[str, object] = {
        "operating_mode_after": OperatingMode.DISTURBED,
        "component_state_after": ComponentState.DEGRADED,
        "variable": StateVariable.PRIMARY_FLOW,
        "value_after": 0.4,
        "observation_status": ObservationStatus.WATCH,
        "channel_quality": ChannelQuality.SUSPECT,
        "commanded_value": 0.4,
        "action_label": ActionLabel.CONTINUE_MONITORING,
    }
    forbidden_field = next(
        field_name
        for field_name in candidate_values
        if field_name not in EVENT_FIELD_MATRIX[event_type].allowed
    )
    payload = _event(event_type).model_dump()
    payload[forbidden_field] = candidate_values[forbidden_field]
    with pytest.raises(ValidationError, match="forbids fields"):
        CanonicalEvent.model_validate(payload)


def test_event_matrix_rejects_contradictory_payload_values() -> None:
    payload = _event(EventType.CHANNEL_DISAGREEMENT).model_dump()
    payload["observation_status"] = ObservationStatus.NORMAL
    with pytest.raises(ValidationError, match="requires CONFLICTING"):
        CanonicalEvent.model_validate(payload)

    payload = _event(EventType.COMMAND_POSITION_MISMATCH).model_dump()
    payload["observed_value"] = payload["commanded_value"]
    with pytest.raises(ValidationError, match="different commanded and observed"):
        CanonicalEvent.model_validate(payload)


def test_fault_duration_must_end_inside_the_scenario_window() -> None:
    boundary = ScenarioDefinition(
        scenario_id="scenario-a",
        plant_variant_id=PlantVariant.ASTER_A,
        seed=7,
        duration_ticks=10,
        driver=ScenarioDriver.STEADY_OPERATION,
        fault_injections=(
            FaultInjection(
                fault_family=FaultFamily.PUMP_TRIP,
                component_id="train-a",
                onset_tick=7,
                duration_ticks=3,
                severity=SeverityBand.HIGH,
            ),
        ),
    )
    assert boundary.fault_injections[0].duration_ticks == 3

    payload = boundary.model_dump()
    payload["fault_injections"][0]["duration_ticks"] = 4
    with pytest.raises(ValidationError, match="fault duration"):
        ScenarioDefinition.model_validate(payload)


def test_evidence_slot_vocabulary_has_exact_reviewed_parity() -> None:
    assert tuple(slot.value for slot in EvidenceSlot) == (
        "STABLE_OPERATION",
        "COORDINATED_LOAD_RESPONSE",
        "CHANNEL_DISAGREEMENT",
        "RELATED_STATE_STABLE",
        "CHANNEL_FROZEN",
        "CORRELATED_STATE_CHANGE",
        "RAPID_INCONSISTENT_READINGS",
        "COMPONENT_HEALTH_DECLINING",
        "FLOW_DECLINING",
        "DEPENDENT_TREND_DELAY",
        "COMPONENT_UNAVAILABLE",
        "STANDBY_AVAILABLE",
        "COMMAND_POSITION_MISMATCH",
        "MISMATCH_RESOLVED",
        "MISMATCH_PERSISTED",
        "UPSTREAM_DOWNSTREAM_DIVERGENCE",
        "SECONDARY_TREND_MISMATCH",
        "SUPPORT_BUS_CHANGE",
        "MAPPED_COMPONENT_CHANGE",
        "INVENTORY_TREND_DECLINING",
        "MULTIPLE_CHANNELS_AGREE",
        "MISSING_DECISIVE_EVIDENCE",
        "CONFLICTING_OBSERVATIONS",
    )


type ConcreteTaskTarget = (
    CausalContinuationTarget
    | FaultDiagnosisTarget
    | EvidenceExtractionTarget
    | NextActionTarget
    | IncidentSummaryTarget
    | CounterfactualComparisonTarget
)


def _task_targets() -> dict[TaskName, ConcreteTaskTarget]:
    unresolved = CounterfactualConclusion(
        diagnosis_status=DiagnosisStatus.UNRESOLVED,
        evidence_slots=(EvidenceSlot.MISSING_DECISIVE_EVIDENCE,),
        immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE,
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
    )
    diagnosed = CounterfactualConclusion(
        diagnosis_status=DiagnosisStatus.DIAGNOSED,
        fault_labels=(FaultFamily.SENSOR_DRIFT,),
        evidence_slots=(EvidenceSlot.CHANNEL_DISAGREEMENT,),
        immediate_action=ActionLabel.VERIFY_REDUNDANT_CHANNEL,
    )
    return {
        TaskName.CONTINUE_LOG: CausalContinuationTarget(
            next_event_type=EventType.OBSERVATION_CHANGED
        ),
        TaskName.FAULT_FAMILY: FaultDiagnosisTarget(diagnosis_status=DiagnosisStatus.NO_FAULT),
        TaskName.EXTRACT_EVIDENCE: EvidenceExtractionTarget(
            evidence_event_ids=("evt-0",),
            evidence_slots=(EvidenceSlot.STABLE_OPERATION,),
        ),
        TaskName.NEXT_ACTION: NextActionTarget(immediate_action=ActionLabel.CONTINUE_MONITORING),
        TaskName.INCIDENT_SUMMARY: IncidentSummaryTarget(
            affected_subsystem=AsterSubsystem.PRIMARY_LOOP,
            observed_trend=ObservedTrend.STABLE,
            diagnosis_status=DiagnosisStatus.NO_FAULT,
            operating_mode=OperatingMode.STABLE,
            immediate_action=ActionLabel.CONTINUE_MONITORING,
        ),
        TaskName.COUNTERFACTUAL_COMPARE: CounterfactualComparisonTarget(
            baseline=unresolved,
            counterfactual=diagnosed,
            changed_fields=(
                CounterfactualChange.DIAGNOSIS_STATUS,
                CounterfactualChange.FAULT_LABELS,
                CounterfactualChange.EVIDENCE_SLOTS,
                CounterfactualChange.IMMEDIATE_ACTION,
            ),
            decisive_evidence_slots=(EvidenceSlot.CHANNEL_DISAGREEMENT,),
        ),
    }


def test_each_task_name_has_one_discriminated_structured_target() -> None:
    targets = _task_targets()
    assert set(targets) == set(TaskName)
    for task_name, target in targets.items():
        wrapped = TaskTarget(task_name=task_name, target=target)
        assert wrapped.target.task_name is task_name


def test_task_wrapper_rejects_task_target_mismatch() -> None:
    with pytest.raises(ValidationError, match="task_name must match"):
        TaskTarget(
            task_name=TaskName.NEXT_ACTION,
            target=CausalContinuationTarget(next_event_type=EventType.BENIGN_NOTE),
        )


def test_counterfactual_declares_exactly_the_changed_structured_fields() -> None:
    target = _task_targets()[TaskName.COUNTERFACTUAL_COMPARE]
    assert isinstance(target, CounterfactualComparisonTarget)
    payload = target.model_dump()
    payload["changed_fields"] = (CounterfactualChange.DIAGNOSIS_STATUS,)
    with pytest.raises(ValidationError, match="exactly describe"):
        CounterfactualComparisonTarget.model_validate(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.update({"scenario_id": "scenario-other"}),
            "scenario_id must match",
        ),
        (
            lambda payload: payload["provenance"].update({"trajectory_id": "trajectory-other"}),
            "provenance trajectory_id",
        ),
        (
            lambda payload: payload["provenance"].update(
                {"plant_variant_id": PlantVariant.ASTER_C}
            ),
            "provenance plant_variant_id",
        ),
        (
            lambda payload: payload["provenance"].update(
                {"fault_family_ids": (FaultFamily.PUMP_TRIP,)}
            ),
            "provenance fault families",
        ),
        (
            lambda payload: payload.update({"observations": payload["observations"][:-1]}),
            "one ordered frame",
        ),
        (
            lambda payload: payload["events"][0].update({"event_index": 1}),
            "contiguous, unique",
        ),
        (
            lambda payload: payload["targets"]["decisions"][0].update({"decision_tick": 2}),
            "decision tick",
        ),
        (
            lambda payload: payload["targets"]["decisions"][0].update(
                {"evidence_event_ids": ("evt-missing",), "evidence_slots": ()}
            ),
            "evidence_event_ids",
        ),
        (
            lambda payload: payload["events"][0].update({"sim_time": 1}),
            "future event",
        ),
    ],
)
def test_structured_trajectory_rejects_broken_cross_record_links(
    mutate: Callable[[dict[str, Any]], object], message: str
) -> None:
    payload = _trajectory().model_dump()
    mutate(payload)
    with pytest.raises(ValidationError, match=message):
        StructuredTrajectory.model_validate(payload)


def test_event_times_are_monotonic_while_simultaneous_events_are_allowed() -> None:
    payload = _trajectory().model_dump()
    second = CanonicalEvent(
        event_id="evt-1",
        event_index=1,
        sim_time=0,
        event_type=EventType.BENIGN_NOTE,
        subject_id="aster-a",
    )
    payload["events"] = (*payload["events"], second.model_dump())
    simultaneous = StructuredTrajectory.model_validate(payload)
    assert tuple(event.sim_time for event in simultaneous.events) == (0, 0)

    payload["events"][0]["sim_time"] = 1
    with pytest.raises(ValidationError, match="sim_time values must be monotonic"):
        StructuredTrajectory.model_validate(payload)


def test_structured_trajectory_rejects_hidden_truth_in_visible_events() -> None:
    payload = _trajectory().model_dump()
    payload["events"][0]["fault_family_ids"] = (FaultFamily.SENSOR_DRIFT,)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        StructuredTrajectory.model_validate(payload)
