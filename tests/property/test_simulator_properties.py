from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from reactorbench.schemas import SeverityBand, StateVariable
from reactorbench.simulator import (
    ASTER_A_SPEC,
    build_sensor_drift_scenario,
    build_stable_scenario,
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
