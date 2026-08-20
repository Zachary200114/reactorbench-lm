"""Acceptance tests for developmental G14 pump degradation plus sensor drift."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pytest

from reactorbench.schemas import (
    AbstentionReason,
    ActionLabel,
    CanonicalEvent,
    ChannelQuality,
    ComponentState,
    DiagnosisStatus,
    EventType,
    EvidenceSlot,
    FaultFamily,
    ObservationStatus,
    OperatingMode,
    PlantVariant,
    ScenarioAction,
    ScenarioDriver,
    SeverityBand,
    StateVariable,
)
from reactorbench.simulator import (
    ASTER_A_SPEC,
    build_pump_degradation_scenario,
    build_pump_degradation_sensor_drift_scenario,
    generate_trace,
)
from reactorbench.simulator.core import _decision_from_compound_evidence


def _selected_channel_id(seed: int) -> str:
    channels = ASTER_A_SPEC.channels_for(StateVariable.PRIMARY_THERMAL_STATE)
    return channels[(seed // 2) % len(channels)].channel_id


def _canonical_subset(
    events: tuple[CanonicalEvent, ...], *, excluded_ids: Iterable[str]
) -> tuple[CanonicalEvent, ...]:
    """Reindex an evidence ablation without retaining hidden dangling edges."""

    excluded = set(excluded_ids)
    retained = [event for event in events if event.event_id not in excluded]
    id_map = {event.event_id: f"e-{index:04d}" for index, event in enumerate(retained)}
    return tuple(
        CanonicalEvent.model_validate(
            {
                **event.model_dump(mode="python"),
                "event_id": id_map[event.event_id],
                "event_index": index,
                "related_event_ids": tuple(
                    id_map[related_id]
                    for related_id in event.related_event_ids
                    if related_id in id_map
                ),
            }
        )
        for index, event in enumerate(retained)
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 2**32 - 1])
def test_builder_emits_the_exact_aster_a_compound_signature(seed: int) -> None:
    scenario = build_pump_degradation_sensor_drift_scenario(seed=seed, duration_ticks=12)
    expected_pump = ASTER_A_SPEC.primary_train_ids[seed % 2]
    expected_channel = _selected_channel_id(seed)
    expected_sensor_component = next(
        channel.component_id
        for channel in ASTER_A_SPEC.channels
        if channel.channel_id == expected_channel
    )

    assert scenario.plant_variant_id is PlantVariant.ASTER_A
    assert scenario.driver is ScenarioDriver.STEADY_OPERATION
    assert scenario.duration_ticks == 12
    assert scenario.standby_context is None
    assert scenario.dependency_map_context is None
    assert tuple(injection.fault_family for injection in scenario.fault_injections) == (
        FaultFamily.SENSOR_DRIFT,
        FaultFamily.PUMP_DEGRADATION,
    )
    sensor, pump = scenario.fault_injections
    assert sensor.component_id == expected_sensor_component
    assert sensor.channel_id == expected_channel
    assert pump.component_id == expected_pump
    assert pump.channel_id is None
    assert all(
        injection.onset_tick == 2
        and injection.severity is SeverityBand.LOW
        and injection.duration_ticks is None
        for injection in scenario.fault_injections
    )
    assert scenario.action_sequence == (
        ScenarioAction(decision_tick=3, action=ActionLabel.INSUFFICIENT_EVIDENCE),
        ScenarioAction(decision_tick=4, action=ActionLabel.VERIFY_REDUNDANT_CHANNEL),
        ScenarioAction(decision_tick=5, action=ActionLabel.FLAG_SENSOR_SUSPECT),
        ScenarioAction(decision_tick=6, action=ActionLabel.REQUEST_COMPONENT_INSPECTION),
        ScenarioAction(decision_tick=7, action=ActionLabel.REDUCE_SIMULATED_LOAD),
    )
    assert str(seed) in scenario.scenario_id
    assert expected_pump in scenario.scenario_id
    assert expected_channel in scenario.scenario_id
    assert not {"g14", "pump", "sensor", "drift"} & set(scenario.scenario_id.split("-"))


@pytest.mark.parametrize("seed", [20, 21])
def test_compound_case_isolates_sensor_drift_from_the_pump_latent_factor(seed: int) -> None:
    compound = generate_trace(
        build_pump_degradation_sensor_drift_scenario(seed=seed, duration_ticks=12)
    )
    pump_only = generate_trace(build_pump_degradation_scenario(seed=seed, duration_ticks=12))
    selected_channel = _selected_channel_id(seed)
    direction = 1.0 if seed % 2 == 0 else -1.0

    assert compound.latent_states == pump_only.latent_states
    for tick, (compound_frame, pump_frame) in enumerate(
        zip(compound.observations, pump_only.observations, strict=True)
    ):
        assert compound_frame.overall_status is pump_frame.overall_status
        compound_by_id = {channel.channel_id: channel for channel in compound_frame.channels}
        pump_by_id = {channel.channel_id: channel for channel in pump_frame.channels}
        assert compound_by_id.keys() == pump_by_id.keys()
        for channel_id, pump_channel in pump_by_id.items():
            actual = compound_by_id[channel_id]
            if channel_id != selected_channel or tick <= 2:
                assert actual == pump_channel
                continue
            assert actual.model_dump(exclude={"value", "status", "quality"}) == (
                pump_channel.model_dump(exclude={"value", "status", "quality"})
            )
            assert actual.value is not None
            assert pump_channel.value is not None
            expected_bias = min(0.042, 0.014 * (tick - 2))
            assert actual.value - pump_channel.value == pytest.approx(
                direction * expected_bias, abs=1e-6
            )
            assert actual.status is (
                ObservationStatus.WATCH if tick == 3 else ObservationStatus.CONFLICTING
            )
            assert actual.quality is (ChannelQuality.GOOD if tick <= 5 else ChannelQuality.SUSPECT)


def test_compound_decisions_and_actions_follow_the_preregistered_timeline() -> None:
    trace = generate_trace(build_pump_degradation_sensor_drift_scenario(seed=20, duration_ticks=12))
    decisions = trace.targets.decisions

    assert [decision.decision_tick for decision in decisions] == [3, 4, 5, 6, 7]
    assert [decision.immediate_action for decision in decisions] == [
        ActionLabel.INSUFFICIENT_EVIDENCE,
        ActionLabel.VERIFY_REDUNDANT_CHANNEL,
        ActionLabel.FLAG_SENSOR_SUSPECT,
        ActionLabel.REQUEST_COMPONENT_INSPECTION,
        ActionLabel.REDUCE_SIMULATED_LOAD,
    ]
    assert decisions[0].diagnosis_status is DiagnosisStatus.UNRESOLVED
    assert decisions[0].fault_labels == ()
    assert decisions[0].abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE
    assert decisions[1].fault_labels == decisions[2].fault_labels == (FaultFamily.SENSOR_DRIFT,)
    assert (
        decisions[3].fault_labels
        == decisions[4].fault_labels
        == (
            FaultFamily.SENSOR_DRIFT,
            FaultFamily.PUMP_DEGRADATION,
        )
    )
    assert all(decision.diagnosis_status is DiagnosisStatus.DIAGNOSED for decision in decisions[1:])
    assert all(
        EvidenceSlot.RELATED_STATE_STABLE not in decision.evidence_slots
        for decision in decisions[1:]
    )

    applied = {
        event.action_label: event.sim_time
        for event in trace.events
        if event.event_type is EventType.ACTION_APPLIED
    }
    assert applied == {
        ActionLabel.INSUFFICIENT_EVIDENCE: 4,
        ActionLabel.VERIFY_REDUNDANT_CHANNEL: 5,
        ActionLabel.FLAG_SENSOR_SUSPECT: 6,
        ActionLabel.REQUEST_COMPONENT_INSPECTION: 7,
        ActionLabel.REDUCE_SIMULATED_LOAD: 8,
    }
    selected_pump = ASTER_A_SPEC.primary_train_ids[0]
    selected_at_seven = next(
        component
        for component in trace.latent_states[7].components
        if component.component_id == selected_pump
    )
    assert selected_at_seven.state is ComponentState.DEGRADED
    assert selected_at_seven.pending_maintenance is True
    assert trace.latent_states[8].values.load_demand == pytest.approx(
        trace.latent_states[7].values.load_demand - 0.012
    )
    assert trace.latent_states[8].values.heat_source_level == pytest.approx(
        trace.latent_states[7].values.heat_source_level - 0.012
    )
    assert trace.latent_states[8].operating_mode is OperatingMode.RECOVERY


def test_compound_events_expose_each_cause_before_its_effect() -> None:
    seed = 20
    trace = generate_trace(
        build_pump_degradation_sensor_drift_scenario(seed=seed, duration_ticks=12)
    )
    selected_pump = ASTER_A_SPEC.primary_train_ids[seed % 2]
    selected_channel = _selected_channel_id(seed)

    stable = trace.events[0]
    assert stable.sim_time == 0
    assert stable.event_type is EventType.BENIGN_NOTE
    assert stable.evidence_slots == (EvidenceSlot.STABLE_OPERATION,)
    assert EvidenceSlot.RELATED_STATE_STABLE not in {
        slot for event in trace.events for slot in event.evidence_slots
    }
    assert any(
        event.sim_time == 2
        and event.event_type is EventType.COMPONENT_STATE_CHANGED
        and event.subject_id == selected_pump
        and event.component_state_after is ComponentState.DEGRADED
        for event in trace.events
    )
    assert any(
        event.sim_time == 2
        and event.event_type is EventType.OPERATING_MODE_CHANGED
        and event.operating_mode_after is OperatingMode.DISTURBED
        for event in trace.events
    )
    assert any(
        event.sim_time == 3
        and event.event_type is EventType.OBSERVATION_CHANGED
        and event.variable is StateVariable.PRIMARY_FLOW
        and EvidenceSlot.MULTIPLE_CHANNELS_AGREE in event.evidence_slots
        for event in trace.events
    )
    assert any(
        event.sim_time == 3
        and event.event_type is EventType.OBSERVATION_CHANGED
        and event.subject_id == selected_channel
        and event.observation_status is ObservationStatus.WATCH
        and EvidenceSlot.MISSING_DECISIVE_EVIDENCE in event.evidence_slots
        for event in trace.events
    )
    disagreement = next(
        event
        for event in trace.events
        if event.sim_time == 4
        and event.event_type is EventType.CHANNEL_DISAGREEMENT
        and event.subject_id == selected_channel
    )
    assert EvidenceSlot.CHANNEL_DISAGREEMENT in disagreement.evidence_slots
    assert any(
        event.sim_time == 4
        and event.event_type is EventType.OBSERVATION_CHANGED
        and event.variable is StateVariable.PRIMARY_THERMAL_STATE
        and event.subject_id != selected_channel
        and EvidenceSlot.CORRELATED_STATE_CHANGE in event.evidence_slots
        for event in trace.events
    )
    assert any(
        event.sim_time == 5
        and event.event_type is EventType.OBSERVATION_CHANGED
        and event.variable is StateVariable.STEAM_STATE
        and EvidenceSlot.DEPENDENT_TREND_DELAY in event.evidence_slots
        for event in trace.events
    )
    assert any(
        event.sim_time == 6
        and event.event_type is EventType.CHANNEL_QUALITY_CHANGED
        and event.subject_id == selected_channel
        and event.channel_quality_before is ChannelQuality.GOOD
        and event.channel_quality is ChannelQuality.SUSPECT
        for event in trace.events
    )
    assert any(
        event.sim_time == 6
        and event.event_type is EventType.OBSERVATION_CHANGED
        and event.variable is StateVariable.ELECTRICAL_OUTPUT
        and EvidenceSlot.DEPENDENT_TREND_DELAY in event.evidence_slots
        for event in trace.events
    )
    persistent = next(
        event
        for event in trace.events
        if event.sim_time == 7 and event.event_type is EventType.BENIGN_NOTE
    )
    assert {
        EvidenceSlot.CORRELATED_STATE_CHANGE,
        EvidenceSlot.DEPENDENT_TREND_DELAY,
    }.issubset(persistent.evidence_slots)
    assert any(
        event.sim_time == 8
        and event.event_type is EventType.TARGET_CHANGED
        and event.variable is StateVariable.LOAD_DEMAND
        for event in trace.events
    )
    assert any(
        event.sim_time == 8
        and event.event_type is EventType.OPERATING_MODE_CHANGED
        and event.operating_mode_after is OperatingMode.RECOVERY
        for event in trace.events
    )

    events_by_id = {event.event_id: event for event in trace.events}
    assert tuple(event.event_index for event in trace.events) == tuple(range(len(trace.events)))
    assert tuple(event.sim_time for event in trace.events) == tuple(
        sorted(event.sim_time for event in trace.events)
    )
    assert all(
        events_by_id[related_id].event_index < event.event_index
        for event in trace.events
        for related_id in event.related_event_ids
    )


@pytest.mark.parametrize("decision_tick", [6, 7])
def test_compound_evidence_helper_is_visible_prefix_only_and_factor_separable(
    decision_tick: int,
) -> None:
    trace = generate_trace(build_pump_degradation_sensor_drift_scenario(seed=20, duration_ticks=12))
    prefix = tuple(event for event in trace.events if event.sim_time <= decision_tick)
    expected = next(
        decision for decision in trace.targets.decisions if decision.decision_tick == decision_tick
    )
    neutral = _decision_from_compound_evidence(
        scenario_id="neutral-visible-prefix", decision_tick=decision_tick, events=prefix
    )
    spoofed = _decision_from_compound_evidence(
        scenario_id="sensor-drift-pump-degradation-spoof",
        decision_tick=decision_tick,
        events=prefix,
    )
    assert neutral.model_dump(exclude={"scenario_id"}) == expected.model_dump(
        exclude={"scenario_id"}
    )
    assert spoofed.model_dump(exclude={"scenario_id"}) == neutral.model_dump(
        exclude={"scenario_id"}
    )
    assert all(
        event.sim_time <= decision_tick
        for event in prefix
        if event.event_id in neutral.evidence_event_ids
    )
    if decision_tick == 6:
        assert all(
            not (
                event.sim_time == 6
                and event.event_type
                in {EventType.ACTION_APPLIED, EventType.CHANNEL_QUALITY_CHANGED}
            )
            for event in prefix
            if event.event_id in neutral.evidence_event_ids
        )

    disagreement = next(
        event
        for event in prefix
        if event.sim_time == 4 and event.event_type is EventType.CHANNEL_DISAGREEMENT
    )
    paired_thermal = next(
        event
        for event in prefix
        if event.sim_time == 4
        and event.event_type is EventType.OBSERVATION_CHANGED
        and event.variable is StateVariable.PRIMARY_THERMAL_STATE
        and event.subject_id != disagreement.subject_id
    )
    assert paired_thermal.event_id in disagreement.related_event_ids
    without_sensor = _decision_from_compound_evidence(
        scenario_id="neutral-visible-prefix",
        decision_tick=decision_tick,
        events=_canonical_subset(prefix, excluded_ids=(disagreement.event_id,)),
    )
    assert without_sensor.diagnosis_status is DiagnosisStatus.DIAGNOSED
    assert without_sensor.fault_labels == (FaultFamily.PUMP_DEGRADATION,)
    assert without_sensor.immediate_action is expected.immediate_action
    sensor_only_slots = {
        EvidenceSlot.CHANNEL_DISAGREEMENT,
        EvidenceSlot.CONFLICTING_OBSERVATIONS,
    }
    assert set(without_sensor.evidence_slots) - sensor_only_slots == (
        set(expected.evidence_slots) - sensor_only_slots
    )

    without_paired_trend = _decision_from_compound_evidence(
        scenario_id="neutral-visible-prefix",
        decision_tick=decision_tick,
        events=_canonical_subset(prefix, excluded_ids=(paired_thermal.event_id,)),
    )
    assert FaultFamily.SENSOR_DRIFT not in without_paired_trend.fault_labels

    component_event = next(
        event
        for event in prefix
        if event.sim_time == 2 and event.event_type is EventType.COMPONENT_STATE_CHANGED
    )
    without_component = _decision_from_compound_evidence(
        scenario_id="neutral-visible-prefix",
        decision_tick=decision_tick,
        events=_canonical_subset(prefix, excluded_ids=(component_event.event_id,)),
    )
    assert FaultFamily.PUMP_DEGRADATION not in without_component.fault_labels


@pytest.mark.parametrize("decision_tick", [4, 5])
def test_early_compound_prefix_abstains_when_sensor_disagreement_is_ablated(
    decision_tick: int,
) -> None:
    trace = generate_trace(build_pump_degradation_sensor_drift_scenario(seed=20, duration_ticks=12))
    prefix = tuple(event for event in trace.events if event.sim_time <= decision_tick)
    expected = next(
        decision for decision in trace.targets.decisions if decision.decision_tick == decision_tick
    )
    full = _decision_from_compound_evidence(
        scenario_id="neutral-visible-prefix", decision_tick=decision_tick, events=prefix
    )
    assert full.model_dump(exclude={"scenario_id"}) == expected.model_dump(exclude={"scenario_id"})
    disagreement = next(
        event for event in prefix if event.event_type is EventType.CHANNEL_DISAGREEMENT
    )
    ablated = _decision_from_compound_evidence(
        scenario_id="neutral-visible-prefix",
        decision_tick=decision_tick,
        events=_canonical_subset(prefix, excluded_ids=(disagreement.event_id,)),
    )
    assert ablated.diagnosis_status is DiagnosisStatus.UNRESOLVED
    assert ablated.fault_labels == ()
    assert ablated.immediate_action is ActionLabel.INSUFFICIENT_EVIDENCE
    assert ablated.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE


def test_compound_evidence_helper_abstains_on_the_tick_three_prefix() -> None:
    trace = generate_trace(build_pump_degradation_sensor_drift_scenario(seed=20))
    prefix = tuple(event for event in trace.events if event.sim_time <= 3)
    expected = trace.targets.decisions[0]
    actual = _decision_from_compound_evidence(
        scenario_id="neutral-visible-prefix", decision_tick=3, events=prefix
    )
    assert actual.model_dump(exclude={"scenario_id"}) == expected.model_dump(
        exclude={"scenario_id"}
    )
    assert actual.diagnosis_status is DiagnosisStatus.UNRESOLVED
    assert actual.fault_labels == ()


def test_compound_evidence_helper_rejects_noncanonical_or_future_event_inputs() -> None:
    trace = generate_trace(build_pump_degradation_sensor_drift_scenario(seed=20, duration_ticks=12))
    prefix = tuple(event for event in trace.events if event.sim_time <= 6)
    with pytest.raises(TypeError, match="tuple"):
        _decision_from_compound_evidence(
            scenario_id="neutral-visible-prefix",
            decision_tick=6,
            events=cast(tuple[CanonicalEvent, ...], list(prefix)),
        )
    with pytest.raises(ValueError, match="ordered visible-event prefix"):
        _decision_from_compound_evidence(
            scenario_id="neutral-visible-prefix",
            decision_tick=6,
            events=tuple(reversed(prefix)),
        )
    with pytest.raises(ValueError, match="ordered visible-event prefix"):
        _decision_from_compound_evidence(
            scenario_id="neutral-visible-prefix",
            decision_tick=6,
            events=tuple(event for event in trace.events if event.sim_time <= 7),
        )
