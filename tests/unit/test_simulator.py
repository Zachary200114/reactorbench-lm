from __future__ import annotations

import random

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
        second = generate_trace(build_stable_scenario(seed=11))

        assert first.latent_states != second.latent_states
        assert first.visible_payload() != second.visible_payload()
        assert drift.latent_states == first.latent_states
        assert load.latent_states != first.latent_states
    finally:
        random.setstate(original_state)


def test_maximum_duration_and_aster_spec_channel_cardinality() -> None:
    trace = generate_trace(build_stable_scenario(seed=4, duration_ticks=64))
    drift = generate_trace(build_sensor_drift_scenario(seed=4, duration_ticks=64))
    load = generate_trace(build_load_transient_scenario(seed=4, duration_ticks=64))

    assert len(trace.observations) == 64
    assert len(drift.observations) == 64
    assert len(load.observations) == 64
    assert len(set(ASTER_A_SPEC.primary_train_ids)) == 2
    assert len(set(ASTER_A_SPEC.support_bus_ids)) == 2
    for variable in StateVariable:
        channel_ids = [
            channel.channel_id for channel in ASTER_A_SPEC.channels if channel.variable is variable
        ]
        assert len(channel_ids) == len(set(channel_ids)) == 2
