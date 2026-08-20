"""Acceptance tests for the G14 primary-thermal drift-only comparator."""

from __future__ import annotations

import pytest

from reactorbench.schemas import (
    AbstentionReason,
    ActionLabel,
    ChannelQuality,
    DiagnosisStatus,
    EventType,
    FaultFamily,
    ObservationStatus,
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
    build_stable_scenario,
    build_thermal_sensor_drift_scenario,
    generate_trace,
)


def _selected_channel_id(seed: int) -> str:
    channels = ASTER_A_SPEC.channels_for(StateVariable.PRIMARY_THERMAL_STATE)
    return channels[(seed // 2) % len(channels)].channel_id


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 2**32 - 1])
def test_builder_emits_the_exact_fixed_aster_a_thermal_profile(seed: int) -> None:
    scenario = build_thermal_sensor_drift_scenario(seed=seed, duration_ticks=12)
    selected_channel = _selected_channel_id(seed)
    expected_component = next(
        channel.component_id
        for channel in ASTER_A_SPEC.channels
        if channel.channel_id == selected_channel
    )

    assert scenario.scenario_id == (f"aster-a-thermal-drift-{seed}-12-2-low-{selected_channel}")
    assert scenario.plant_variant_id is PlantVariant.ASTER_A
    assert scenario.driver is ScenarioDriver.STEADY_OPERATION
    assert scenario.standby_context is None
    assert scenario.dependency_map_context is None
    assert len(scenario.fault_injections) == 1
    injection = scenario.fault_injections[0]
    assert injection.fault_family is FaultFamily.SENSOR_DRIFT
    assert injection.component_id == expected_component == ASTER_A_SPEC.instrumentation_id
    assert injection.channel_id == selected_channel
    assert injection.onset_tick == 2
    assert injection.severity is SeverityBand.LOW
    assert injection.duration_ticks is None
    assert scenario.action_sequence == (
        ScenarioAction(decision_tick=3, action=ActionLabel.INSUFFICIENT_EVIDENCE),
        ScenarioAction(decision_tick=4, action=ActionLabel.VERIFY_REDUNDANT_CHANNEL),
        ScenarioAction(decision_tick=5, action=ActionLabel.FLAG_SENSOR_SUSPECT),
    )


@pytest.mark.parametrize("seed", [20, 21, 22, 23])
def test_thermal_drift_changes_only_the_selected_observation(seed: int) -> None:
    drift = generate_trace(build_thermal_sensor_drift_scenario(seed=seed))
    stable = generate_trace(build_stable_scenario(seed=seed))
    selected_channel = _selected_channel_id(seed)
    direction = 1.0 if seed % 2 == 0 else -1.0

    assert drift.latent_states == stable.latent_states
    for tick, (drift_frame, stable_frame) in enumerate(
        zip(drift.observations, stable.observations, strict=True)
    ):
        drift_by_id = {channel.channel_id: channel for channel in drift_frame.channels}
        stable_by_id = {channel.channel_id: channel for channel in stable_frame.channels}
        assert drift_by_id.keys() == stable_by_id.keys()
        for channel_id, stable_channel in stable_by_id.items():
            actual = drift_by_id[channel_id]
            if channel_id != selected_channel or tick <= 2:
                assert actual == stable_channel
                continue
            assert actual.model_dump(exclude={"value", "status", "quality"}) == (
                stable_channel.model_dump(exclude={"value", "status", "quality"})
            )
            assert actual.value is not None
            assert stable_channel.value is not None
            expected_bias = min(0.042, 0.014 * (tick - 2))
            assert actual.value - stable_channel.value == pytest.approx(
                direction * expected_bias, abs=1e-6
            )
            assert actual.status is (
                ObservationStatus.WATCH if tick == 3 else ObservationStatus.CONFLICTING
            )
            assert actual.quality is (ChannelQuality.GOOD if tick <= 5 else ChannelQuality.SUSPECT)


@pytest.mark.parametrize("seed", [20, 21, 22, 23])
def test_thermal_drift_overlay_exactly_matches_the_g14_sensor_factor(seed: int) -> None:
    drift_only = generate_trace(build_thermal_sensor_drift_scenario(seed=seed))
    stable = generate_trace(build_stable_scenario(seed=seed))
    compound = generate_trace(build_pump_degradation_sensor_drift_scenario(seed=seed))
    pump_only = generate_trace(build_pump_degradation_scenario(seed=seed))
    selected_channel = _selected_channel_id(seed)

    assert drift_only.latent_states == stable.latent_states
    assert compound.latent_states == pump_only.latent_states
    for tick in range(12):
        drift_channel = next(
            channel
            for channel in drift_only.observations[tick].channels
            if channel.channel_id == selected_channel
        )
        stable_channel = next(
            channel
            for channel in stable.observations[tick].channels
            if channel.channel_id == selected_channel
        )
        compound_channel = next(
            channel
            for channel in compound.observations[tick].channels
            if channel.channel_id == selected_channel
        )
        pump_channel = next(
            channel
            for channel in pump_only.observations[tick].channels
            if channel.channel_id == selected_channel
        )
        drift_value = drift_channel.value
        stable_value = stable_channel.value
        compound_value = compound_channel.value
        pump_value = pump_channel.value
        assert drift_value is not None
        assert stable_value is not None
        assert compound_value is not None
        assert pump_value is not None
        drift_delta = drift_value - stable_value
        compound_delta = compound_value - pump_value
        assert drift_delta == pytest.approx(compound_delta, abs=1e-6)
        assert drift_channel.status is compound_channel.status
        assert drift_channel.quality is compound_channel.quality


def test_thermal_drift_events_targets_and_actions_use_the_selected_variable() -> None:
    trace = generate_trace(build_thermal_sensor_drift_scenario(seed=20))
    selected_channel = _selected_channel_id(20)
    selected_events = tuple(
        event
        for event in trace.events
        if event.subject_id == selected_channel
        and event.event_type in {EventType.OBSERVATION_CHANGED, EventType.CHANNEL_DISAGREEMENT}
    )

    assert tuple(event.variable for event in selected_events) == (
        StateVariable.PRIMARY_THERMAL_STATE,
        StateVariable.PRIMARY_THERMAL_STATE,
    )
    assert [decision.decision_tick for decision in trace.targets.decisions] == [3, 4, 5]
    assert [decision.immediate_action for decision in trace.targets.decisions] == [
        ActionLabel.INSUFFICIENT_EVIDENCE,
        ActionLabel.VERIFY_REDUNDANT_CHANNEL,
        ActionLabel.FLAG_SENSOR_SUSPECT,
    ]
    early, mature, flagged = trace.targets.decisions
    assert early.diagnosis_status is DiagnosisStatus.UNRESOLVED
    assert early.fault_labels == ()
    assert early.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE
    assert mature.fault_labels == flagged.fault_labels == (FaultFamily.SENSOR_DRIFT,)
    assert [
        (event.sim_time, event.action_label) for event in trace.events if event.action_label
    ] == [
        (4, ActionLabel.INSUFFICIENT_EVIDENCE),
        (5, ActionLabel.VERIFY_REDUNDANT_CHANNEL),
        (6, ActionLabel.FLAG_SENSOR_SUSPECT),
    ]
