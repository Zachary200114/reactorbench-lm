from __future__ import annotations

import random
from itertools import pairwise

import pytest

from reactorbench.schemas import (
    AbstentionReason,
    ActionLabel,
    ChannelQuality,
    ComponentState,
    DiagnosisStatus,
    EvidenceSlot,
    FaultFamily,
    ObservationStatus,
    OperatingMode,
    PlantVariant,
    ScenarioDriver,
    SeverityBand,
    StateVariable,
)
from reactorbench.simulator import (
    ASTER_A_SPEC,
    SimulationTrace,
    UnsupportedScenarioError,
    build_load_transient_scenario,
    build_sensor_drift_scenario,
    build_sensor_noise_scenario,
    build_sensor_stuck_load_scenario,
    build_stable_scenario,
    generate_trace,
)


def _channel_values(trace: SimulationTrace, channel_id: str) -> tuple[float | None, ...]:
    return tuple(
        next(channel.value for channel in frame.channels if channel.channel_id == channel_id)
        for frame in trace.observations
    )


def test_stable_trace_is_bounded_available_and_resolved() -> None:
    trace = generate_trace(build_stable_scenario(seed=19, duration_ticks=10))

    assert len(trace.latent_states) == len(trace.observations) == 10
    assert trace.scenario.fault_injections == ()
    assert trace.targets.decisions[-1].diagnosis_status is DiagnosisStatus.NO_FAULT
    assert trace.targets.decisions[-1].immediate_action is ActionLabel.CONTINUE_MONITORING
    first_latent = trace.latent_states[0]
    assert all(
        0.425 <= value <= 0.515
        for latent in trace.latent_states
        for value in latent.values.model_dump().values()
    )
    assert all(
        latent.operating_mode == first_latent.operating_mode
        and latent.values == first_latent.values
        and latent.components == first_latent.components
        for latent in trace.latent_states
    )
    assert all(
        component.state is ComponentState.AVAILABLE and component.health == 1.0
        for latent in trace.latent_states
        for component in latent.components
    )
    assert all(len(frame.channels) == 2 * len(StateVariable) for frame in trace.observations)
    assert all(frame.overall_status is ObservationStatus.NORMAL for frame in trace.observations)
    for frame in trace.observations:
        for variable in StateVariable:
            pair = [channel for channel in frame.channels if channel.variable is variable]
            assert len(pair) == 2
            assert pair[0].value == pair[1].value
        assert all(
            channel.status is ObservationStatus.NORMAL and channel.quality is ChannelQuality.GOOD
            for channel in frame.channels
        )


def test_load_transient_is_benign_coordinated_and_causally_ordered() -> None:
    scenario = build_load_transient_scenario(seed=20, duration_ticks=12)
    trace = generate_trace(scenario)
    stable = generate_trace(build_stable_scenario(seed=20, duration_ticks=12))

    assert scenario.driver is ScenarioDriver.LOAD_TRANSIENT
    assert scenario.fault_injections == ()
    decision = trace.targets.decisions[-1]
    assert decision.diagnosis_status is DiagnosisStatus.NO_FAULT
    assert decision.fault_labels == ()
    assert decision.immediate_action is ActionLabel.CONTINUE_MONITORING
    assert all(
        component.state is ComponentState.AVAILABLE
        for state in trace.latent_states
        for component in state.components
    )
    assert all(frame.overall_status is ObservationStatus.NORMAL for frame in trace.observations)
    assert all(
        channel.status is ObservationStatus.NORMAL and channel.quality is ChannelQuality.GOOD
        for frame in trace.observations
        for channel in frame.channels
    )
    assert all(latent.operating_mode is OperatingMode.STABLE for latent in trace.latent_states[:2])
    assert all(
        latent.operating_mode is OperatingMode.LOAD_CHANGE for latent in trace.latent_states[2:5]
    )
    assert all(
        latent.operating_mode is OperatingMode.LOAD_CHANGE for latent in trace.latent_states[5:7]
    )
    assert all(latent.operating_mode is OperatingMode.STABLE for latent in trace.latent_states[7:])
    assert trace.latent_states[:2] == stable.latent_states[:2]
    assert trace.observations[:2] == stable.observations[:2]
    assert trace.latent_states[-1].values.load_demand > stable.latent_states[-1].values.load_demand
    for variable in (
        StateVariable.HEAT_SOURCE_LEVEL,
        StateVariable.PRIMARY_FLOW,
        StateVariable.STEAM_STATE,
        StateVariable.TURBINE_OUTPUT,
        StateVariable.ELECTRICAL_OUTPUT,
    ):
        assert getattr(trace.latent_states[-1].values, variable.value) > getattr(
            stable.latent_states[-1].values, variable.value
        )
    assert all(
        transient.values.transfer_efficiency == baseline.values.transfer_efficiency
        for transient, baseline in zip(trace.latent_states, stable.latent_states, strict=True)
    )
    for frame in trace.observations:
        for variable in StateVariable:
            pair = [channel for channel in frame.channels if channel.variable is variable]
            assert len(pair) == 2
            assert pair[0].value == pair[1].value
    assert [(event.sim_time, event.event_type.value) for event in trace.events] == [
        (0, "BENIGN_NOTE"),
        (2, "TARGET_CHANGED"),
        (2, "OPERATING_MODE_CHANGED"),
        (6, "BENIGN_NOTE"),
        (7, "OPERATING_MODE_CHANGED"),
    ]
    assert not any(event.action_label is not None for event in trace.events)
    target_event, coordinated_event = trace.events[1], trace.events[3]
    assert target_event.sim_time < coordinated_event.sim_time
    assert coordinated_event.related_event_ids == (target_event.event_id,)


def test_load_transient_stages_latent_response_in_causal_order() -> None:
    transient = generate_trace(build_load_transient_scenario(seed=20, duration_ticks=12))
    stable = generate_trace(build_stable_scenario(seed=20, duration_ticks=12))

    def first_change(variable: StateVariable) -> int:
        return next(
            state.tick
            for state, baseline in zip(transient.latent_states, stable.latent_states, strict=True)
            if getattr(state.values, variable.value) != getattr(baseline.values, variable.value)
        )

    assert first_change(StateVariable.LOAD_DEMAND) == 2
    assert first_change(StateVariable.HEAT_SOURCE_LEVEL) == 2
    assert first_change(StateVariable.PRIMARY_FLOW) == 2
    assert first_change(StateVariable.STEAM_STATE) == 3
    assert first_change(StateVariable.TURBINE_OUTPUT) == 4
    assert first_change(StateVariable.ELECTRICAL_OUTPUT) == 4


def test_load_transient_direction_is_seed_derived_and_reversible() -> None:
    rising = generate_trace(build_load_transient_scenario(seed=20))
    falling = generate_trace(build_load_transient_scenario(seed=21))
    rising_stable = generate_trace(build_stable_scenario(seed=20))
    falling_stable = generate_trace(build_stable_scenario(seed=21))

    assert (
        rising.latent_states[-1].values.load_demand
        > rising_stable.latent_states[-1].values.load_demand
    )
    assert (
        falling.latent_states[-1].values.load_demand
        < falling_stable.latent_states[-1].values.load_demand
    )


def test_load_transient_builder_and_generator_fail_closed() -> None:
    with pytest.raises(ValueError, match="between"):
        build_load_transient_scenario(seed=1, duration_ticks=65)
    scenario = build_load_transient_scenario(seed=4)
    drift_injection = build_sensor_drift_scenario(seed=4).fault_injections[0]
    unsupported = scenario.model_copy(update={"fault_injections": (drift_injection,)})

    with pytest.raises(UnsupportedScenarioError, match="cannot include"):
        generate_trace(unsupported)


def test_sensor_stuck_load_preserves_latent_truth_and_freezes_one_channel() -> None:
    scenario = build_sensor_stuck_load_scenario(seed=20, duration_ticks=12)
    stuck = generate_trace(scenario)
    load = generate_trace(build_load_transient_scenario(seed=20, duration_ticks=12))
    injection = scenario.fault_injections[0]
    selected_channel = injection.channel_id
    assert selected_channel is not None

    assert stuck.latent_states == load.latent_states
    frozen_value = _channel_values(load, selected_channel)[1]
    for stuck_frame, load_frame in zip(stuck.observations, load.observations, strict=True):
        for stuck_channel, load_channel in zip(
            stuck_frame.channels, load_frame.channels, strict=True
        ):
            if stuck_channel.channel_id != selected_channel:
                assert stuck_channel == load_channel
            elif stuck_frame.tick < 2:
                assert stuck_channel == load_channel
            else:
                assert stuck_channel.value == frozen_value
    selected = [
        next(channel for channel in frame.channels if channel.channel_id == selected_channel)
        for frame in stuck.observations
    ]
    assert [channel.status for channel in selected[:4]] == [ObservationStatus.NORMAL] * 4
    assert selected[4].status is ObservationStatus.WATCH
    assert all(channel.status is ObservationStatus.CONFLICTING for channel in selected[5:])
    assert [frame.overall_status for frame in stuck.observations[:4]] == [
        ObservationStatus.NORMAL
    ] * 4
    assert stuck.observations[4].overall_status is ObservationStatus.WATCH
    assert all(
        frame.overall_status is ObservationStatus.CONFLICTING for frame in stuck.observations[5:]
    )
    assert all(channel.quality is ChannelQuality.GOOD for channel in selected[:7])
    assert selected[7].quality is ChannelQuality.SUSPECT
    assert all(
        decision.diagnosis_status is DiagnosisStatus.DIAGNOSED
        and decision.fault_labels == (FaultFamily.SENSOR_STUCK,)
        for decision in stuck.targets.decisions
    )


def test_sensor_stuck_load_events_are_causal_and_evidence_backed() -> None:
    trace = generate_trace(build_sensor_stuck_load_scenario(seed=21, duration_ticks=12))
    events_by_id = {event.event_id: event for event in trace.events}

    assert [
        (event.sim_time, event.event_type.value, event.action_label) for event in trace.events
    ] == [
        (0, "BENIGN_NOTE", None),
        (2, "TARGET_CHANGED", None),
        (2, "OPERATING_MODE_CHANGED", None),
        (5, "OBSERVATION_CHANGED", None),
        (5, "CHANNEL_DISAGREEMENT", None),
        (6, "ACTION_APPLIED", ActionLabel.VERIFY_REDUNDANT_CHANNEL),
        (6, "BENIGN_NOTE", None),
        (7, "ACTION_APPLIED", ActionLabel.FLAG_SENSOR_SUSPECT),
        (7, "OPERATING_MODE_CHANGED", None),
        (7, "CHANNEL_QUALITY_CHANGED", None),
    ]
    assert [decision.immediate_action for decision in trace.targets.decisions] == [
        ActionLabel.VERIFY_REDUNDANT_CHANNEL,
        ActionLabel.FLAG_SENSOR_SUSPECT,
    ]
    assert all(
        EvidenceSlot.RELATED_STATE_STABLE not in decision.evidence_slots
        and set(decision.evidence_slots)
        == {
            EvidenceSlot.CHANNEL_FROZEN,
            EvidenceSlot.CORRELATED_STATE_CHANGE,
            EvidenceSlot.CHANNEL_DISAGREEMENT,
        }
        for decision in trace.targets.decisions
    )
    assert all(
        event.sim_time <= decision.decision_tick
        for decision in trace.targets.decisions
        for evidence_id in decision.evidence_event_ids
        for event in (events_by_id[evidence_id],)
    )
    coordinated = next(
        event
        for event in trace.events
        if event.evidence_slots == (EvidenceSlot.COORDINATED_LOAD_RESPONSE,)
    )
    settlement = next(
        event
        for event in trace.events
        if event.event_type.value == "OPERATING_MODE_CHANGED" and event.sim_time == 7
    )
    assert settlement.related_event_ids == (coordinated.event_id,)


def test_sensor_stuck_load_supports_both_channels_and_load_directions() -> None:
    for seed, channel_id, direction in (
        (20, "aster-electrical-output-a", 1.0),
        (21, "aster-electrical-output-b", -1.0),
    ):
        stuck = generate_trace(build_sensor_stuck_load_scenario(seed=seed, channel_id=channel_id))
        load = generate_trace(build_load_transient_scenario(seed=seed))
        stable = generate_trace(build_stable_scenario(seed=seed))

        assert stuck.latent_states == load.latent_states
        assert (
            stuck.latent_states[-1].values.load_demand - stable.latent_states[-1].values.load_demand
        ) * direction > 0.0
        assert (
            _channel_values(stuck, channel_id)[2:] == (_channel_values(load, channel_id)[1],) * 10
        )


def test_sensor_stuck_load_builder_and_generator_fail_closed() -> None:
    scenario = build_sensor_stuck_load_scenario(seed=4, duration_ticks=8)
    injection = scenario.fault_injections[0]
    malformed = (
        scenario.model_copy(update={"driver": ScenarioDriver.STEADY_OPERATION}),
        scenario.model_copy(update={"plant_variant_id": PlantVariant.ASTER_B}),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"channel_id": "aster-electrical-output-x"}),
                )
            }
        ),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"component_id": "aster-train-cirrus"}),
                )
            }
        ),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"severity": SeverityBand.MEDIUM}),
                )
            }
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"onset_tick": 3}),)}
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"duration_ticks": 2}),)}
        ),
        scenario.model_copy(update={"action_sequence": ()}),
        scenario.model_copy(update={"fault_injections": (injection, injection)}),
        scenario.model_copy(update={"fault_injections": [injection]}),
    )

    for invalid in malformed:
        with pytest.raises(UnsupportedScenarioError):
            generate_trace(invalid)
    with pytest.raises(ValueError, match="between"):
        build_sensor_stuck_load_scenario(seed=4, duration_ticks=7)
    with pytest.raises(ValueError, match="electrical-output"):
        build_sensor_stuck_load_scenario(seed=4, channel_id="aster-primary-flow-a")


def test_sensor_noise_is_prefix_preserving_alternating_and_evidence_gated() -> None:
    scenario = build_sensor_noise_scenario(seed=20, duration_ticks=12)
    trace = generate_trace(scenario)
    stable = generate_trace(build_stable_scenario(seed=20, duration_ticks=12))
    selected_channel = scenario.fault_injections[0].channel_id
    assert selected_channel is not None
    assert scenario.scenario_id == f"aster-a-noise-20-12-2-low-{selected_channel}"

    assert trace.latent_states == stable.latent_states
    for noise_frame, stable_frame in zip(trace.observations, stable.observations, strict=True):
        for noise_channel, stable_channel in zip(
            noise_frame.channels, stable_frame.channels, strict=True
        ):
            if noise_channel.channel_id != selected_channel:
                assert noise_channel == stable_channel
            elif noise_frame.tick <= 2:
                assert noise_channel == stable_channel
    selected_values = _channel_values(trace, selected_channel)
    stable_values = _channel_values(stable, selected_channel)
    deviations = tuple(
        value - baseline
        for value, baseline in zip(selected_values[3:], stable_values[3:], strict=True)
        if value is not None and baseline is not None
    )
    assert all(0.018 - 1e-6 <= abs(value) <= 0.024 + 1e-6 for value in deviations)
    assert all(left * right < 0.0 for left, right in pairwise(deviations))
    assert all(
        abs(deviations[index]) == pytest.approx(abs(deviations[index + 1]))
        for index in range(0, len(deviations) - 1, 2)
    )
    assert [decision.immediate_action for decision in trace.targets.decisions] == [
        ActionLabel.INSUFFICIENT_EVIDENCE,
        ActionLabel.INSUFFICIENT_EVIDENCE,
        ActionLabel.COMPARE_RELATED_TRENDS,
        ActionLabel.FLAG_SENSOR_SUSPECT,
    ]
    assert [decision.diagnosis_status for decision in trace.targets.decisions[:2]] == [
        DiagnosisStatus.UNRESOLVED,
        DiagnosisStatus.UNRESOLVED,
    ]
    assert all(
        decision.fault_labels == (FaultFamily.SENSOR_NOISE,)
        for decision in trace.targets.decisions[2:]
    )
    selected = [
        next(channel for channel in frame.channels if channel.channel_id == selected_channel)
        for frame in trace.observations
    ]
    assert [channel.status for channel in selected[:3]] == [ObservationStatus.NORMAL] * 3
    assert [channel.status for channel in selected[3:5]] == [ObservationStatus.WATCH] * 2
    assert all(channel.status is ObservationStatus.CONFLICTING for channel in selected[5:])
    assert all(channel.quality is ChannelQuality.GOOD for channel in selected[:7])
    assert selected[7].quality is ChannelQuality.SUSPECT
    assert all(channel.quality is not ChannelQuality.NOISY for channel in selected)
    events_by_id = {event.event_id: event for event in trace.events}
    second_observation = next(
        event
        for event in trace.events
        if event.sim_time == 4 and event.event_type.value == "OBSERVATION_CHANGED"
    )
    second_abstention = next(
        event
        for event in trace.events
        if event.sim_time == 5
        and event.event_type.value == "ACTION_APPLIED"
        and event.action_label is ActionLabel.INSUFFICIENT_EVIDENCE
    )
    third_deviation = next(
        event
        for event in trace.events
        if event.sim_time == 5 and event.event_type.value == "OBSERVATION_CHANGED"
    )
    assert second_observation.event_id in trace.targets.decisions[1].evidence_event_ids
    assert second_abstention.related_event_ids == (second_observation.event_id,)
    assert EvidenceSlot.RAPID_INCONSISTENT_READINGS in third_deviation.evidence_slots
    assert all(
        events_by_id[evidence_id].sim_time <= decision.decision_tick
        for decision in trace.targets.decisions
        for evidence_id in decision.evidence_event_ids
    )


def test_sensor_noise_actions_and_fail_closed_constraints() -> None:
    scenario = build_sensor_noise_scenario(seed=21, duration_ticks=8)
    trace = generate_trace(scenario)
    injection = scenario.fault_injections[0]
    assert [
        (event.sim_time, event.action_label) for event in trace.events if event.action_label
    ] == [
        (4, ActionLabel.INSUFFICIENT_EVIDENCE),
        (5, ActionLabel.INSUFFICIENT_EVIDENCE),
        (6, ActionLabel.COMPARE_RELATED_TRENDS),
        (7, ActionLabel.FLAG_SENSOR_SUSPECT),
    ]
    assert [
        (event.sim_time, event.event_type.value) for event in trace.events if event.sim_time >= 3
    ] == [
        (3, "OBSERVATION_CHANGED"),
        (4, "ACTION_APPLIED"),
        (4, "OBSERVATION_CHANGED"),
        (5, "ACTION_APPLIED"),
        (5, "OBSERVATION_CHANGED"),
        (5, "CHANNEL_DISAGREEMENT"),
        (6, "ACTION_APPLIED"),
        (6, "BENIGN_NOTE"),
        (7, "ACTION_APPLIED"),
        (7, "CHANNEL_QUALITY_CHANGED"),
    ]
    invalid = (
        scenario.model_copy(update={"driver": ScenarioDriver.LOAD_TRANSIENT}),
        scenario.model_copy(update={"plant_variant_id": PlantVariant.ASTER_B}),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"channel_id": "aster-primary-flow-a"}),
                )
            }
        ),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"component_id": "aster-train-cirrus"}),
                )
            }
        ),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"severity": SeverityBand.MEDIUM}),
                )
            }
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"onset_tick": 3}),)}
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"duration_ticks": 2}),)}
        ),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"fault_family": FaultFamily.SENSOR_DRIFT}),
                )
            }
        ),
        scenario.model_copy(update={"action_sequence": ()}),
        scenario.model_copy(update={"fault_injections": (injection, injection)}),
        scenario.model_copy(update={"fault_injections": [injection]}),
        scenario.model_copy(update={"scenario_id": "spoofed-scenario"}),
        scenario.model_copy(update={"schema_version": "9.9.9"}),
    )
    for malformed in invalid:
        with pytest.raises(UnsupportedScenarioError):
            generate_trace(malformed)
    for channel_id in ("", True, 7, "aster-primary-thermal-state-x"):
        with pytest.raises(ValueError, match=r".+"):
            build_sensor_noise_scenario(seed=1, channel_id=channel_id)  # type: ignore[arg-type]


def test_model_copy_lookalikes_are_rejected_before_fault_dispatch() -> None:
    noise = build_sensor_noise_scenario(seed=31, duration_ticks=8)
    stuck = build_sensor_stuck_load_scenario(seed=31, duration_ticks=8)

    for scenario in (noise, stuck):
        injection = scenario.fault_injections[0]
        first_action = scenario.action_sequence[0]
        canonical_actions = scenario.action_sequence
        invalid = (
            scenario.model_copy(update={"driver": scenario.driver.value}),
            scenario.model_copy(
                update={"fault_injections": (injection.model_copy(update={"onset_tick": 2.0}),)}
            ),
            scenario.model_copy(
                update={"fault_injections": (injection.model_copy(update={"onset_tick": True}),)}
            ),
            scenario.model_copy(
                update={
                    "action_sequence": (
                        first_action.model_copy(
                            update={"decision_tick": float(first_action.decision_tick)}
                        ),
                        *canonical_actions[1:],
                    )
                }
            ),
            scenario.model_copy(
                update={
                    "action_sequence": (
                        first_action.model_copy(update={"decision_tick": True}),
                        *canonical_actions[1:],
                    )
                }
            ),
            scenario.model_copy(
                update={
                    "action_sequence": (
                        first_action.model_copy(update={"action": first_action.action.value}),
                        *canonical_actions[1:],
                    )
                }
            ),
            scenario.model_copy(update={"action_sequence": list(canonical_actions)}),
            scenario.model_copy(update={"action_sequence": ("malformed-action",)}),
        )
        for malformed in invalid:
            with pytest.raises(UnsupportedScenarioError):
                generate_trace(malformed)

    valid_noise = generate_trace(noise)
    assert all(
        decision.fault_labels == (FaultFamily.SENSOR_NOISE,)
        for decision in valid_noise.targets.decisions[2:]
    )


@pytest.mark.parametrize("channel_id", ["", True, 7, "aster-electrical-output-x"])
def test_sensor_stuck_builder_requires_strict_allowlisted_channel_input(channel_id: object) -> None:
    with pytest.raises(ValueError, match=r".+"):
        build_sensor_stuck_load_scenario(seed=1, channel_id=channel_id)  # type: ignore[arg-type]


def test_sensor_drift_separates_only_observation_layer_and_actions() -> None:
    drift_scenario = build_sensor_drift_scenario(seed=20, onset_tick=3, duration_ticks=12)
    drift = generate_trace(drift_scenario)
    stable = generate_trace(build_stable_scenario(seed=20, duration_ticks=12))
    injection = drift_scenario.fault_injections[0]
    selected_channel = injection.channel_id
    assert selected_channel is not None

    assert drift.latent_states == stable.latent_states
    for drift_frame, stable_frame in zip(drift.observations, stable.observations, strict=True):
        for drift_channel, stable_channel in zip(
            drift_frame.channels, stable_frame.channels, strict=True
        ):
            if drift_channel.channel_id != selected_channel:
                assert drift_channel == stable_channel
            elif drift_frame.tick <= injection.onset_tick:
                assert drift_channel == stable_channel

    selected_values = _channel_values(drift, selected_channel)
    paired_channel = next(
        channel.channel_id
        for channel in ASTER_A_SPEC.channels
        if channel.variable is StateVariable.PRIMARY_FLOW and channel.channel_id != selected_channel
    )
    paired_values = _channel_values(drift, paired_channel)
    separations = tuple(
        abs(selected - paired)
        for selected, paired in zip(selected_values, paired_values, strict=True)
        if selected is not None and paired is not None
    )
    assert separations[injection.onset_tick] == 0.0
    assert all(
        separations[index + 1] + 1e-12 >= separations[index]
        for index in range(injection.onset_tick + 1, len(separations) - 1)
    )
    assert all(
        later - earlier <= ASTER_A_SPEC.max_per_tick_step
        for earlier, later in zip(
            separations[injection.onset_tick + 1 :],
            separations[injection.onset_tick + 2 :],
            strict=False,
        )
    )
    assert [decision.immediate_action for decision in drift.targets.decisions] == [
        ActionLabel.INSUFFICIENT_EVIDENCE,
        ActionLabel.VERIFY_REDUNDANT_CHANNEL,
        ActionLabel.FLAG_SENSOR_SUSPECT,
    ]
    early, mature, flagged = drift.targets.decisions
    assert early.diagnosis_status is DiagnosisStatus.UNRESOLVED
    assert early.fault_labels == ()
    assert early.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE
    assert mature.fault_labels == flagged.fault_labels == (FaultFamily.SENSOR_DRIFT,)
    assert set(mature.evidence_slots) == {
        EvidenceSlot.CHANNEL_DISAGREEMENT,
        EvidenceSlot.RELATED_STATE_STABLE,
    }
    selected_at_flag = next(
        channel
        for channel in drift.observations[flagged.decision_tick].channels
        if channel.channel_id == selected_channel
    )
    selected_after_flag = next(
        channel
        for channel in drift.observations[flagged.decision_tick + 1].channels
        if channel.channel_id == selected_channel
    )
    assert selected_at_flag.quality is ChannelQuality.GOOD
    assert selected_after_flag.quality is ChannelQuality.SUSPECT
    assert selected_at_flag.status is ObservationStatus.CONFLICTING
    selected_statuses = tuple(
        next(channel.status for channel in frame.channels if channel.channel_id == selected_channel)
        for frame in drift.observations
    )
    assert all(
        status is ObservationStatus.NORMAL
        for status in selected_statuses[: injection.onset_tick + 1]
    )
    assert selected_statuses[injection.onset_tick + 1] is ObservationStatus.WATCH
    assert all(
        status is ObservationStatus.CONFLICTING
        for status in selected_statuses[injection.onset_tick + 2 :]
    )


def test_drift_action_events_follow_their_decisions_in_causal_order() -> None:
    scenario = build_sensor_drift_scenario(seed=20, onset_tick=3, duration_ticks=12)
    trace = generate_trace(scenario)
    early, mature, flagged = trace.targets.decisions
    action_events = [event for event in trace.events if event.action_label is not None]

    assert [(event.sim_time, event.action_label) for event in action_events] == [
        (early.decision_tick + 1, ActionLabel.INSUFFICIENT_EVIDENCE),
        (mature.decision_tick + 1, ActionLabel.VERIFY_REDUNDANT_CHANNEL),
        (flagged.decision_tick + 1, ActionLabel.FLAG_SENSOR_SUSPECT),
    ]
    mature_events = [event for event in trace.events if event.sim_time == mature.decision_tick]
    assert mature_events[0].action_label is ActionLabel.INSUFFICIENT_EVIDENCE
    assert mature_events[1].evidence_slots == (EvidenceSlot.CHANNEL_DISAGREEMENT,)
    flag_application = flagged.decision_tick + 1
    assert [
        event.event_type.value for event in trace.events if event.sim_time == flag_application
    ] == [
        "ACTION_APPLIED",
        "CHANNEL_QUALITY_CHANGED",
    ]


def test_diagnosis_references_stable_and_mature_evidence_without_future_events() -> None:
    trace = generate_trace(build_sensor_drift_scenario(seed=20))
    event_by_id = {event.event_id: event for event in trace.events}
    stable = trace.events[0]
    mature = next(
        event
        for event in trace.events
        if event.evidence_slots == (EvidenceSlot.CHANNEL_DISAGREEMENT,)
    )

    assert mature.related_event_ids == (stable.event_id, trace.events[1].event_id)
    for decision in trace.targets.decisions[1:]:
        assert decision.evidence_event_ids == (stable.event_id, mature.event_id)
        observed_slots = {
            slot
            for event_id in decision.evidence_event_ids
            for slot in event_by_id[event_id].evidence_slots
        }
        assert set(decision.evidence_slots).issubset(observed_slots)
        assert all(
            event_by_id[event_id].sim_time <= decision.decision_tick
            for event_id in decision.evidence_event_ids
        )


@pytest.mark.parametrize(
    ("builder", "kwargs"),
    [
        (build_stable_scenario, {"seed": True}),
        (build_stable_scenario, {"seed": 2**32}),
        (build_stable_scenario, {"seed": 1, "duration_ticks": True}),
        (build_stable_scenario, {"seed": 1, "duration_ticks": 7}),
        (build_sensor_drift_scenario, {"seed": 1, "onset_tick": True}),
        (build_sensor_drift_scenario, {"seed": 1, "duration_ticks": 8, "onset_tick": 4}),
        (build_sensor_noise_scenario, {"seed": True}),
        (build_sensor_noise_scenario, {"seed": 1, "duration_ticks": True}),
        (build_sensor_noise_scenario, {"seed": 1, "duration_ticks": 7}),
    ],
)
def test_builders_reject_loose_or_unsupported_inputs(builder: object, kwargs: object) -> None:
    assert callable(builder)
    assert isinstance(kwargs, dict)
    with pytest.raises(ValueError, match=r"must|between|needs"):
        builder(**kwargs)


def test_drift_builder_rejects_nonprimary_channel() -> None:
    with pytest.raises(ValueError, match="primary-flow"):
        build_sensor_drift_scenario(
            seed=1,
            channel_id="aster-secondary-flow-a",
            severity=SeverityBand.LOW,
        )


@pytest.mark.parametrize("channel_id", ["", True, 7, "aster-primary-flow-x"])
def test_drift_builder_requires_strict_allowlisted_channel_input(channel_id: object) -> None:
    with pytest.raises(ValueError, match=r".+"):
        build_sensor_drift_scenario(seed=1, channel_id=channel_id)  # type: ignore[arg-type]


def test_drift_identity_includes_selected_channel_and_severity() -> None:
    first = build_sensor_drift_scenario(
        seed=1, channel_id="aster-primary-flow-a", severity=SeverityBand.LOW
    )
    alternate_channel = build_sensor_drift_scenario(
        seed=1, channel_id="aster-primary-flow-b", severity=SeverityBand.LOW
    )
    alternate_severity = build_sensor_drift_scenario(
        seed=1, channel_id="aster-primary-flow-a", severity=SeverityBand.MEDIUM
    )

    assert first.scenario_id == "aster-a-drift-1-12-3-low-aster-primary-flow-a"
    assert (
        len({first.scenario_id, alternate_channel.scenario_id, alternate_severity.scenario_id}) == 3
    )


def test_direct_noncanonical_scenario_is_rejected() -> None:
    scenario = build_sensor_drift_scenario(seed=4)
    unsupported = scenario.model_copy(update={"action_sequence": ()})
    with pytest.raises(UnsupportedScenarioError, match="noncanonical"):
        generate_trace(unsupported)


def test_direct_wrong_component_channel_mapping_is_rejected() -> None:
    scenario = build_sensor_drift_scenario(seed=4)
    wrong_injection = scenario.fault_injections[0].model_copy(
        update={"component_id": "aster-train-kestrel"}
    )
    unsupported = scenario.model_copy(update={"fault_injections": (wrong_injection,)})
    with pytest.raises(UnsupportedScenarioError, match="mapping"):
        generate_trace(unsupported)


def test_generate_trace_rejects_unsupported_direct_schema_variants() -> None:
    stable = build_stable_scenario(seed=4)
    drift = build_sensor_drift_scenario(seed=4)
    injection = drift.fault_injections[0]
    scenarios = (
        stable.model_copy(update={"plant_variant_id": PlantVariant.ASTER_B}),
        drift.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"fault_family": FaultFamily.SENSOR_NOISE}),
                )
            }
        ),
        drift.model_copy(update={"fault_injections": (injection, injection)}),
        drift.model_copy(
            update={"fault_injections": (injection.model_copy(update={"duration_ticks": 2}),)}
        ),
    )

    for scenario in scenarios:
        with pytest.raises(UnsupportedScenarioError):
            generate_trace(scenario)


def test_generate_trace_rejects_spoofed_scenario_ids_and_schema_versions() -> None:
    scenarios = (
        build_stable_scenario(seed=4),
        build_load_transient_scenario(seed=4),
        build_sensor_drift_scenario(seed=4),
        build_sensor_stuck_load_scenario(seed=4),
    )

    for scenario in scenarios:
        with pytest.raises(UnsupportedScenarioError, match="id"):
            generate_trace(scenario.model_copy(update={"scenario_id": "spoofed-scenario"}))
        with pytest.raises(UnsupportedScenarioError, match="schema"):
            generate_trace(scenario.model_copy(update={"schema_version": "9.9.9"}))


def test_global_random_state_is_unchanged_and_distinct_seeds_vary() -> None:
    original_state = random.getstate()
    try:
        random.seed(817)
        before = random.getstate()
        first = generate_trace(build_stable_scenario(seed=10))
        assert random.getstate() == before
        drift = generate_trace(build_sensor_drift_scenario(seed=10))
        assert random.getstate() == before
        load = generate_trace(build_load_transient_scenario(seed=10))
        assert random.getstate() == before
        stuck = generate_trace(build_sensor_stuck_load_scenario(seed=10))
        assert random.getstate() == before
        noise = generate_trace(build_sensor_noise_scenario(seed=10))
        assert random.getstate() == before
        second = generate_trace(build_stable_scenario(seed=11))

        assert first.latent_states != second.latent_states
        assert first.visible_payload() != second.visible_payload()
        assert drift.latent_states == first.latent_states
        assert load.latent_states != first.latent_states
        assert stuck.latent_states == load.latent_states
        assert noise.latent_states == first.latent_states
    finally:
        random.setstate(original_state)


def test_maximum_duration_and_aster_spec_channel_cardinality() -> None:
    trace = generate_trace(build_stable_scenario(seed=4, duration_ticks=64))
    drift = generate_trace(build_sensor_drift_scenario(seed=4, duration_ticks=64))
    load = generate_trace(build_load_transient_scenario(seed=4, duration_ticks=64))
    noise = generate_trace(build_sensor_noise_scenario(seed=4, duration_ticks=64))

    assert len(trace.observations) == 64
    assert len(drift.observations) == 64
    assert len(load.observations) == 64
    assert len(noise.observations) == 64
    assert len(set(ASTER_A_SPEC.primary_train_ids)) == 2
    assert len(set(ASTER_A_SPEC.support_bus_ids)) == 2
    for variable in StateVariable:
        channel_ids = [
            channel.channel_id for channel in ASTER_A_SPEC.channels if channel.variable is variable
        ]
        assert len(channel_ids) == len(set(channel_ids)) == 2
