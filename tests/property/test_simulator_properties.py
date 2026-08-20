from __future__ import annotations

from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reactorbench.schemas import (
    ActionLabel,
    ComponentState,
    DiagnosisStatus,
    SeverityBand,
    StateVariable,
)
from reactorbench.simulator import (
    ASTER_A_SPEC,
    build_load_transient_scenario,
    build_pump_degradation_scenario,
    build_pump_trip_scenario,
    build_sensor_drift_scenario,
    build_sensor_noise_scenario,
    build_sensor_stuck_load_scenario,
    build_stable_scenario,
    build_valve_lag_scenario,
    build_valve_stuck_scenario,
    generate_trace,
)


@st.composite
def _supported_drift_inputs(draw: st.DrawFn) -> tuple[int, int, SeverityBand, str]:
    duration = draw(st.integers(min_value=8, max_value=24))
    onset = draw(st.integers(min_value=0, max_value=duration - 5))
    channels = tuple(
        channel.channel_id
        for channel in ASTER_A_SPEC.channels
        if channel.variable is StateVariable.PRIMARY_FLOW
    )
    return (
        duration,
        onset,
        draw(st.sampled_from(tuple(SeverityBand))),
        draw(st.sampled_from(channels)),
    )


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1), duration=st.integers(8, 24))
def test_stable_generation_is_replayable_and_seed_controlled(seed: int, duration: int) -> None:
    first = generate_trace(build_stable_scenario(seed=seed, duration_ticks=duration))
    second = generate_trace(build_stable_scenario(seed=seed, duration_ticks=duration))

    assert first == second
    assert len(first.observations) == duration
    assert all(len(frame.channels) == 2 * len(StateVariable) for frame in first.observations)


@settings(max_examples=30, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    lag_ticks=st.sampled_from((3, 4)),
)
def test_valve_counterfactual_is_seeded_prefix_safe_and_bounded(seed: int, lag_ticks: int) -> None:
    lag = generate_trace(
        build_valve_lag_scenario(seed=seed, duration_ticks=12, lag_ticks=lag_ticks)
    )
    stuck = generate_trace(build_valve_stuck_scenario(seed=seed, duration_ticks=12))
    stable = generate_trace(build_stable_scenario(seed=seed, duration_ticks=12))

    assert lag == generate_trace(lag.scenario)
    assert stuck == generate_trace(stuck.scenario)
    assert lag.scenario.fault_injections[0].duration_ticks == lag_ticks
    assert lag.targets.decisions[-1].decision_tick == 2 + lag_ticks
    assert (
        lag.latent_states[2 + lag_ticks].values.primary_flow
        != stable.latent_states[2 + lag_ticks].values.primary_flow
    )
    assert stuck.latent_states[6].values.primary_flow == stable.latent_states[6].values.primary_flow
    assert all(
        0.0 <= value <= 1.0
        for trace in (lag, stuck)
        for state in trace.latent_states
        for value in state.values.model_dump().values()
    )


@settings(max_examples=35, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    parameters=_supported_drift_inputs(),
)
def test_drift_preserves_latent_and_bounds_observations(
    seed: int, parameters: tuple[int, int, SeverityBand, str]
) -> None:
    duration, onset, severity, channel_id = parameters
    scenario = build_sensor_drift_scenario(
        seed=seed,
        duration_ticks=duration,
        onset_tick=onset,
        severity=severity,
        channel_id=channel_id,
    )
    drift = generate_trace(scenario)
    replay = generate_trace(scenario)
    stable = generate_trace(build_stable_scenario(seed=seed, duration_ticks=duration))

    assert drift == replay
    assert drift.latent_states == stable.latent_states
    assert scenario.fault_injections[0].channel_id == channel_id
    for drift_frame, stable_frame in zip(drift.observations, stable.observations, strict=True):
        for drift_channel, stable_channel in zip(
            drift_frame.channels, stable_frame.channels, strict=True
        ):
            if drift_channel.channel_id != channel_id:
                assert drift_channel == stable_channel
            elif drift_frame.tick <= onset:
                assert drift_channel == stable_channel
    assert all(
        channel.value is None or 0.0 <= channel.value <= 1.0
        for frame in drift.observations
        for channel in frame.channels
    )
    assert [event.event_index for event in drift.events] == list(range(len(drift.events)))
    assert [event.sim_time for event in drift.events] == sorted(
        event.sim_time for event in drift.events
    )
    event_by_id = {event.event_id: event for event in drift.events}
    assert all(
        event_by_id[evidence_id].sim_time <= decision.decision_tick
        for decision in drift.targets.decisions
        for evidence_id in decision.evidence_event_ids
    )


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1), duration=st.integers(8, 64))
def test_load_transient_is_replayable_bounded_and_no_fault(seed: int, duration: int) -> None:
    scenario = build_load_transient_scenario(seed=seed, duration_ticks=duration)
    first = generate_trace(scenario)
    second = generate_trace(scenario)
    stable = generate_trace(build_stable_scenario(seed=seed, duration_ticks=duration))

    assert first == second
    assert scenario.fault_injections == ()
    assert first.latent_states[:2] == stable.latent_states[:2]
    assert first.observations[:2] == stable.observations[:2]
    assert all(
        decision.diagnosis_status is DiagnosisStatus.NO_FAULT
        and decision.fault_labels == ()
        and decision.abstention_reason is None
        and decision.immediate_action is ActionLabel.CONTINUE_MONITORING
        for decision in first.targets.decisions
    )
    assert all(
        transient.values.transfer_efficiency == baseline.values.transfer_efficiency
        for transient, baseline in zip(first.latent_states, stable.latent_states, strict=True)
    )
    for frame in first.observations:
        for variable in StateVariable:
            pair = [channel for channel in frame.channels if channel.variable is variable]
            assert len(pair) == 2
            assert pair[0].value == pair[1].value
    first_changes = {
        variable: next(
            state.tick
            for state, baseline in zip(first.latent_states, stable.latent_states, strict=True)
            if getattr(state.values, variable.value) != getattr(baseline.values, variable.value)
        )
        for variable in (
            StateVariable.LOAD_DEMAND,
            StateVariable.HEAT_SOURCE_LEVEL,
            StateVariable.PRIMARY_FLOW,
            StateVariable.STEAM_STATE,
            StateVariable.TURBINE_OUTPUT,
            StateVariable.ELECTRICAL_OUTPUT,
        )
    }
    assert first_changes[StateVariable.LOAD_DEMAND] == 2
    assert first_changes[StateVariable.HEAT_SOURCE_LEVEL] == 2
    assert first_changes[StateVariable.PRIMARY_FLOW] == 2
    assert first_changes[StateVariable.STEAM_STATE] == 3
    assert first_changes[StateVariable.TURBINE_OUTPUT] >= first_changes[StateVariable.STEAM_STATE]
    assert (
        first_changes[StateVariable.ELECTRICAL_OUTPUT] >= first_changes[StateVariable.STEAM_STATE]
    )
    direction = 1.0 if seed % 2 == 0 else -1.0
    assert (
        first.latent_states[-1].values.load_demand - stable.latent_states[-1].values.load_demand
    ) * direction > 0.0
    assert all(
        0.0 <= value <= 1.0
        for state in first.latent_states
        for value in state.values.model_dump().values()
    )
    assert all(
        abs(later - earlier) <= ASTER_A_SPEC.max_per_tick_step
        for before, after in zip(first.latent_states, first.latent_states[1:], strict=False)
        for earlier, later in zip(
            before.values.model_dump().values(), after.values.model_dump().values(), strict=True
        )
    )
    assert [event.event_index for event in first.events] == list(range(len(first.events)))
    assert [event.sim_time for event in first.events] == sorted(
        event.sim_time for event in first.events
    )
    event_by_id = {event.event_id: event for event in first.events}
    assert all(
        event_by_id[evidence_id].sim_time <= decision.decision_tick
        for decision in first.targets.decisions
        for evidence_id in decision.evidence_event_ids
    )


@settings(max_examples=30, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    duration=st.integers(min_value=8, max_value=24),
    channel_id=st.sampled_from(
        tuple(
            channel.channel_id
            for channel in ASTER_A_SPEC.channels
            if channel.variable is StateVariable.ELECTRICAL_OUTPUT
        )
    ),
)
def test_sensor_stuck_load_is_replayable_and_prefix_preserving(
    seed: int, duration: int, channel_id: str
) -> None:
    scenario = build_sensor_stuck_load_scenario(
        seed=seed, duration_ticks=duration, channel_id=channel_id
    )
    first = generate_trace(scenario)
    replay = generate_trace(scenario)
    load = generate_trace(build_load_transient_scenario(seed=seed, duration_ticks=duration))
    stable = generate_trace(build_stable_scenario(seed=seed, duration_ticks=duration))

    assert first == replay
    assert first.latent_states == load.latent_states
    direction = 1.0 if seed % 2 == 0 else -1.0
    assert (
        first.latent_states[-1].values.load_demand - stable.latent_states[-1].values.load_demand
    ) * direction > 0.0
    frozen_value = next(
        channel.value
        for channel in load.observations[1].channels
        if channel.channel_id == channel_id
    )
    for stuck_frame, load_frame in zip(first.observations, load.observations, strict=True):
        for stuck_channel, load_channel in zip(
            stuck_frame.channels, load_frame.channels, strict=True
        ):
            if stuck_channel.channel_id != channel_id:
                assert stuck_channel == load_channel
            elif stuck_frame.tick < 2:
                assert stuck_channel == load_channel
            else:
                assert stuck_channel.value == frozen_value
    assert all(
        channel.value is None or 0.0 <= channel.value <= 1.0
        for frame in first.observations
        for channel in frame.channels
    )
    events_by_id = {event.event_id: event for event in first.events}
    correlated = next(
        event
        for event in first.events
        if event.event_type.value == "OBSERVATION_CHANGED"
        and event.variable is StateVariable.ELECTRICAL_OUTPUT
    )
    assert correlated.value_before is not None
    assert correlated.value_after is not None
    direction = 1.0 if seed % 2 == 0 else -1.0
    assert (correlated.value_after - correlated.value_before) * direction > 0.0
    assert all(
        events_by_id[evidence_id].sim_time <= decision.decision_tick
        for decision in first.targets.decisions
        for evidence_id in decision.evidence_event_ids
    )


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    channel_id=st.sampled_from(
        tuple(
            channel.channel_id
            for channel in ASTER_A_SPEC.channels
            if channel.variable is StateVariable.ELECTRICAL_OUTPUT
        )
    ),
)
def test_sensor_stuck_minimum_trace_is_prefix_of_a_lengthened_trace(
    seed: int, channel_id: str
) -> None:
    short = generate_trace(
        build_sensor_stuck_load_scenario(seed=seed, duration_ticks=8, channel_id=channel_id)
    )
    long = generate_trace(
        build_sensor_stuck_load_scenario(seed=seed, duration_ticks=12, channel_id=channel_id)
    )

    assert short.latent_states == long.latent_states[:8]
    assert short.observations == long.observations[:8]
    assert short.events == long.events


@settings(max_examples=35, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    duration=st.integers(min_value=8, max_value=24),
    channel_id=st.sampled_from(
        tuple(
            channel.channel_id
            for channel in ASTER_A_SPEC.channels
            if channel.variable is StateVariable.PRIMARY_THERMAL_STATE
        )
    ),
)
def test_sensor_noise_is_replayable_alternating_and_isolated(
    seed: int, duration: int, channel_id: str
) -> None:
    scenario = build_sensor_noise_scenario(
        seed=seed, duration_ticks=duration, channel_id=channel_id
    )
    noise = generate_trace(scenario)
    replay = generate_trace(scenario)
    stable = generate_trace(build_stable_scenario(seed=seed, duration_ticks=duration))

    assert noise == replay
    assert noise.latent_states == stable.latent_states
    selected_offsets: list[float] = []
    for noise_frame, stable_frame in zip(noise.observations, stable.observations, strict=True):
        for noise_channel, stable_channel in zip(
            noise_frame.channels, stable_frame.channels, strict=True
        ):
            if noise_channel.channel_id != channel_id:
                assert noise_channel == stable_channel
            elif noise_frame.tick <= 2:
                assert noise_channel == stable_channel
            else:
                assert noise_channel.value is not None
                assert stable_channel.value is not None
                selected_offsets.append(noise_channel.value - stable_channel.value)
    assert all(0.018 - 1e-6 <= abs(offset) <= 0.024 + 1e-6 for offset in selected_offsets)
    assert selected_offsets[0] * (1.0 if seed % 2 == 0 else -1.0) > 0.0
    assert all(earlier * later < 0.0 for earlier, later in pairwise(selected_offsets))
    assert all(
        abs(selected_offsets[index]) == pytest.approx(abs(selected_offsets[index + 1]))
        for index in range(0, len(selected_offsets) - 1, 2)
    )
    assert all(
        channel.value is None or 0.0 <= channel.value <= 1.0
        for frame in noise.observations
        for channel in frame.channels
    )
    assert [event.event_index for event in noise.events] == list(range(len(noise.events)))
    assert [event.sim_time for event in noise.events] == sorted(
        event.sim_time for event in noise.events
    )
    events_by_id = {event.event_id: event for event in noise.events}
    assert all(
        events_by_id[evidence_id].sim_time <= decision.decision_tick
        for decision in noise.targets.decisions
        for evidence_id in decision.evidence_event_ids
    )


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    channel_id=st.sampled_from(
        tuple(
            channel.channel_id
            for channel in ASTER_A_SPEC.channels
            if channel.variable is StateVariable.PRIMARY_THERMAL_STATE
        )
    ),
)
def test_sensor_noise_minimum_trace_is_prefix_of_a_lengthened_trace(
    seed: int, channel_id: str
) -> None:
    short = generate_trace(
        build_sensor_noise_scenario(seed=seed, duration_ticks=8, channel_id=channel_id)
    )
    long = generate_trace(
        build_sensor_noise_scenario(seed=seed, duration_ticks=12, channel_id=channel_id)
    )

    assert short.latent_states == long.latent_states[:8]
    assert short.observations == long.observations[:8]
    assert short.events == long.events
    assert tuple(
        decision.model_dump(exclude={"scenario_id"}) for decision in short.targets.decisions
    ) == tuple(decision.model_dump(exclude={"scenario_id"}) for decision in long.targets.decisions)


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    duration=st.integers(min_value=9, max_value=24),
    component_id=st.sampled_from(ASTER_A_SPEC.primary_train_ids),
)
def test_pump_degradation_is_replayable_bounded_and_prefix_preserving(
    seed: int, duration: int, component_id: str
) -> None:
    scenario = build_pump_degradation_scenario(
        seed=seed, duration_ticks=duration, component_id=component_id
    )
    first = generate_trace(scenario)
    second = generate_trace(scenario)
    stable = generate_trace(build_stable_scenario(seed=seed, duration_ticks=duration))

    assert first == second
    assert scenario.fault_injections[0].component_id == component_id
    assert first.latent_states[:2] == stable.latent_states[:2]
    assert first.observations[:2] == stable.observations[:2]
    assert [state.operating_mode for state in first.latent_states[:2]] == [
        stable.latent_states[0].operating_mode
    ] * 2
    assert all(
        0.0 <= value <= 1.0
        for state in first.latent_states
        for value in state.values.model_dump().values()
    )
    assert all(
        abs(later - earlier) <= ASTER_A_SPEC.max_per_tick_step
        for before, after in zip(first.latent_states, first.latent_states[1:], strict=False)
        for earlier, later in zip(
            before.values.model_dump().values(), after.values.model_dump().values(), strict=True
        )
    )
    assert [decision.decision_tick for decision in first.targets.decisions] == [4, 6, 7]


@settings(max_examples=30, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    duration=st.integers(min_value=8, max_value=24),
    component_id=st.sampled_from(ASTER_A_SPEC.primary_train_ids),
    standby_state=st.sampled_from((ComponentState.AVAILABLE, ComponentState.UNAVAILABLE)),
)
def test_pump_trip_is_replayable_prefix_stable_and_bounded_except_for_trip(
    seed: int,
    duration: int,
    component_id: str,
    standby_state: ComponentState,
) -> None:
    scenario = build_pump_trip_scenario(
        seed=seed,
        duration_ticks=duration,
        component_id=component_id,
        standby_state=standby_state,
    )
    first = generate_trace(scenario)
    replay = generate_trace(scenario)
    short = generate_trace(
        build_pump_trip_scenario(
            seed=seed,
            duration_ticks=8,
            component_id=component_id,
            standby_state=standby_state,
        )
    )

    assert first == replay
    assert short.latent_states == first.latent_states[:8]
    assert short.observations == first.observations[:8]
    assert short.events == first.events
    assert tuple(
        decision.model_dump(exclude={"scenario_id"}) for decision in short.targets.decisions
    ) == tuple(decision.model_dump(exclude={"scenario_id"}) for decision in first.targets.decisions)
    assert all(
        0.0 <= value <= 1.0
        for latent in first.latent_states
        for value in latent.values.model_dump().values()
    )
    for before, after in pairwise(first.latent_states):
        for variable in StateVariable:
            step = abs(
                getattr(after.values, variable.value) - getattr(before.values, variable.value)
            )
            if before.tick == 2 and variable is StateVariable.PRIMARY_FLOW:
                assert step > ASTER_A_SPEC.max_per_tick_step
            else:
                assert step <= ASTER_A_SPEC.max_per_tick_step
    event_by_id = {event.event_id: event for event in first.events}
    assert [event.event_index for event in first.events] == list(range(len(first.events)))
    assert [event.sim_time for event in first.events] == sorted(
        event.sim_time for event in first.events
    )
    assert all(
        event_by_id[related_id].event_index < event.event_index
        for event in first.events
        for related_id in event.related_event_ids
    )
    assert all(
        event_by_id[evidence_id].sim_time <= decision.decision_tick
        for decision in first.targets.decisions
        for evidence_id in decision.evidence_event_ids
    )
