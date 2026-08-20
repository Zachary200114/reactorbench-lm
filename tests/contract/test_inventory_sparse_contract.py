"""Public contract and fail-closed tests for G13 and G15."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from reactorbench.schemas import (
    AbstentionReason,
    ActionLabel,
    EventType,
    EvidenceSlot,
    FaultFamily,
    PlantVariant,
    ProvenanceRecord,
    ScenarioDefinition,
    SplitName,
    StateVariable,
    TaskName,
)
from reactorbench.simulator import (
    UnsupportedScenarioError,
    build_abstract_inventory_loss_scenario,
    build_sparse_primary_flow_scenario,
    generate_trace,
    get_variant_spec,
)
from reactorbench.simulator.core import SimulationTrace

_Builder = Callable[..., ScenarioDefinition]


def _g13_trace(*, variant: PlantVariant = PlantVariant.ASTER_B) -> SimulationTrace:
    return generate_trace(
        build_abstract_inventory_loss_scenario(seed=23, duration_ticks=12, plant_variant=variant)
    )


def _g15_trace(*, variant: PlantVariant = PlantVariant.ASTER_B) -> SimulationTrace:
    return generate_trace(
        build_sparse_primary_flow_scenario(seed=23, duration_ticks=8, plant_variant=variant)
    )


def test_g13_visible_payload_is_allowlisted_and_truth_free() -> None:
    trace = _g13_trace()
    payload = trace.visible_payload()
    assert set(payload) == {
        "schema_version",
        "plant_variant_id",
        "dependency_map_context",
        "standby_context",
        "observations",
        "events",
    }
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in (
        "ABSTRACT_INVENTORY_LOSS",
        "fault_family",
        "fault_injection",
        "onset_tick",
        "duration_ticks",
        "scenario_id",
        "latent_states",
        "targets",
        "provenance",
        "pending_maintenance",
    ):
        assert forbidden not in serialized


def test_g15_truth_filtered_audit_payload_contains_full_roster_without_hidden_cause() -> None:
    trace = _g15_trace()
    serialized = json.dumps(trace.visible_payload(), sort_keys=True)
    for forbidden in (
        "ABSTRACT_INVENTORY_LOSS",
        "SENSOR_DRIFT",
        "SENSOR_NOISE",
        "fault_family",
        "fault_injection",
        "onset_tick",
        "latent_states",
        "targets",
        "provenance",
    ):
        assert forbidden not in serialized
    spec = get_variant_spec(PlantVariant.ASTER_B)
    observations = trace.visible_payload()["observations"]
    assert isinstance(observations, list)
    assert all(
        isinstance(frame, dict)
        and isinstance(frame.get("channels"), list)
        and len(frame["channels"]) == len(spec.channels)
        for frame in observations
    )


def test_g13_audit_trajectory_keeps_fault_truth_only_in_provenance_and_targets() -> None:
    trace = _g13_trace()
    trajectory_id = "g13-contract-trace-23"
    trajectory = trace.to_structured_trajectory(
        trajectory_id=trajectory_id,
        provenance=ProvenanceRecord(
            dataset_version="0.1.0",
            generator_commit="abcdef1",
            renderer_version="0.1.0",
            seed=23,
            trajectory_id=trajectory_id,
            scenario_id=trace.scenario.scenario_id,
            plant_variant_id=trace.scenario.plant_variant_id,
            fault_family_ids=(FaultFamily.ABSTRACT_INVENTORY_LOSS,),
            template_family_ids=("template-g13",),
            split_name=SplitName.COMPOSITION_TEST,
            task_name=TaskName.FAULT_FAMILY,
        ),
    )
    assert trajectory.provenance.fault_family_ids == (FaultFamily.ABSTRACT_INVENTORY_LOSS,)
    assert trajectory.targets.decisions[-1].fault_labels == (FaultFamily.ABSTRACT_INVENTORY_LOSS,)
    assert "ABSTRACT_INVENTORY_LOSS" not in json.dumps(trace.visible_payload())


def test_g15_audit_trajectory_records_empty_fault_truth_and_abstention() -> None:
    trace = _g15_trace()
    trajectory_id = "g15-contract-trace-23"
    trajectory = trace.to_structured_trajectory(
        trajectory_id=trajectory_id,
        provenance=ProvenanceRecord(
            dataset_version="0.1.0",
            generator_commit="abcdef1",
            renderer_version="0.1.0",
            seed=23,
            trajectory_id=trajectory_id,
            scenario_id=trace.scenario.scenario_id,
            plant_variant_id=trace.scenario.plant_variant_id,
            fault_family_ids=(),
            template_family_ids=("template-g15",),
            split_name=SplitName.NOISE_TEST,
            task_name=TaskName.FAULT_FAMILY,
        ),
    )
    assert trajectory.provenance.fault_family_ids == ()
    decision = trajectory.targets.decisions[0]
    assert decision.diagnosis_status.value == "UNRESOLVED"
    assert decision.fault_labels == ()
    assert decision.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE
    assert decision.immediate_action is ActionLabel.INSUFFICIENT_EVIDENCE


def test_g13_events_are_causal_and_actions_apply_one_tick_after_decision() -> None:
    trace = _g13_trace()
    events_by_id = {event.event_id: event for event in trace.events}
    assert tuple(event.event_index for event in trace.events) == tuple(range(len(trace.events)))
    assert tuple(event.sim_time for event in trace.events) == tuple(
        sorted(event.sim_time for event in trace.events)
    )
    for event in trace.events:
        assert all(
            events_by_id[related].event_index < event.event_index
            for related in event.related_event_ids
        )
    action_events = {
        event.action_label: event
        for event in trace.events
        if event.event_type is EventType.ACTION_APPLIED and event.action_label is not None
    }
    assert action_events[ActionLabel.INSUFFICIENT_EVIDENCE].sim_time == 4
    assert action_events[ActionLabel.REDUCE_SIMULATED_LOAD].sim_time == 7
    assert action_events[ActionLabel.ENTER_SIMULATED_STABLE_STATE].sim_time == 8
    assert all(
        action_events[action].sim_time == decision_tick + 1
        for action, decision_tick in (
            (ActionLabel.INSUFFICIENT_EVIDENCE, 3),
            (ActionLabel.REDUCE_SIMULATED_LOAD, 6),
            (ActionLabel.ENTER_SIMULATED_STABLE_STATE, 7),
        )
    )


def test_g15_events_stop_at_the_sparse_decision_and_then_record_abstention_apply() -> None:
    trace = _g15_trace()
    spec = get_variant_spec(PlantVariant.ASTER_B)
    selected = spec.channels_for(StateVariable.PRIMARY_FLOW)[23 % 2].channel_id
    assert trace.events[0].event_type is EventType.BENIGN_NOTE
    assert EvidenceSlot.RELATED_STATE_STABLE not in trace.events[0].evidence_slots
    changed = tuple(
        event for event in trace.events if event.event_type is EventType.OBSERVATION_CHANGED
    )
    assert len(changed) == 1
    assert changed[0].sim_time == 2
    assert changed[0].subject_id == selected
    assert trace.targets.decisions[0].evidence_event_ids == (changed[0].event_id,)
    action = next(event for event in trace.events if event.event_type is EventType.ACTION_APPLIED)
    assert action.action_label is ActionLabel.INSUFFICIENT_EVIDENCE
    assert action.sim_time == 3
    assert action.related_event_ids == (changed[0].event_id,)


@pytest.mark.parametrize(
    "builder",
    [build_abstract_inventory_loss_scenario, build_sparse_primary_flow_scenario],
)
def test_g13_and_g15_do_not_accept_caller_supplied_truth_or_channel_selection(
    builder: _Builder,
) -> None:
    with pytest.raises(TypeError):
        builder(seed=23, duration_ticks=12, fault_family=FaultFamily.PUMP_TRIP)
    with pytest.raises(TypeError):
        builder(seed=23, duration_ticks=12, channel_id="caller-selected")


def test_g13_and_g15_model_copy_lookalikes_fail_closed() -> None:
    g13 = _g13_trace().scenario
    g15 = _g15_trace().scenario
    for scenario in (g13, g15):
        candidates = (
            scenario.model_copy(update={"scenario_id": "spoofed-scenario"}),
            scenario.model_copy(update={"driver": scenario.driver.value}),
            scenario.model_copy(update={"action_sequence": [*scenario.action_sequence]}),
            scenario.model_copy(update={"fault_injections": [*scenario.fault_injections]}),
            scenario.model_copy(update={"plant_variant_id": PlantVariant.ASTER_C}),
        )
        for candidate in candidates:
            with pytest.raises(UnsupportedScenarioError):
                generate_trace(candidate)
