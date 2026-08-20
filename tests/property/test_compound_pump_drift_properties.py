"""Property checks for the developmental G14 compound fixture."""

from __future__ import annotations

import random
from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reactorbench.schemas import ChannelQuality, FaultFamily, StateVariable
from reactorbench.simulator import (
    ASTER_A_SPEC,
    build_pump_degradation_scenario,
    build_pump_degradation_sensor_drift_scenario,
    generate_trace,
    scan_prohibited_content,
)
from reactorbench.simulator.core import SimulationTrace


def _selected_channel_id(seed: int) -> str:
    channels = ASTER_A_SPEC.channels_for(StateVariable.PRIMARY_THERMAL_STATE)
    return channels[(seed // 2) % len(channels)].channel_id


def _event_shapes(trace: SimulationTrace, *, cutoff: int) -> tuple[dict[str, object], ...]:
    return tuple(
        event.model_dump(mode="json", exclude={"event_id", "related_event_ids"})
        for event in trace.events
        if event.sim_time <= cutoff
    )


def _decision_shapes(trace: SimulationTrace) -> tuple[dict[str, object], ...]:
    return tuple(
        decision.model_dump(mode="json", exclude={"scenario_id", "evidence_event_ids"})
        for decision in trace.targets.decisions
    )


@settings(max_examples=24, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    duration=st.sampled_from((9, 12, 64)),
)
def test_compound_generation_is_replayable_bounded_and_content_safe(
    seed: int, duration: int
) -> None:
    scenario = build_pump_degradation_sensor_drift_scenario(seed=seed, duration_ticks=duration)
    first = generate_trace(scenario)
    second = generate_trace(scenario)

    assert first == second
    assert len(first.latent_states) == len(first.observations) == duration
    assert tuple(injection.fault_family for injection in scenario.fault_injections) == (
        FaultFamily.SENSOR_DRIFT,
        FaultFamily.PUMP_DEGRADATION,
    )
    assert scan_prohibited_content(scenario) == ()
    assert scan_prohibited_content(first) == ()
    assert all(
        0.0 <= value <= 1.0
        for state in first.latent_states
        for value in state.values.model_dump().values()
    )
    assert all(
        channel.value is None or 0.0 <= channel.value <= 1.0
        for frame in first.observations
        for channel in frame.channels
    )
    for previous, current in pairwise(first.latent_states):
        for variable in StateVariable:
            before = getattr(previous.values, variable.value)
            after = getattr(current.values, variable.value)
            assert abs(after - before) <= ASTER_A_SPEC.max_per_tick_step + 1e-9


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_compound_short_duration_is_a_semantic_prefix(seed: int) -> None:
    traces = {
        duration: generate_trace(
            build_pump_degradation_sensor_drift_scenario(seed=seed, duration_ticks=duration)
        )
        for duration in (9, 12, 64)
    }
    short = traces[9]
    for duration in (12, 64):
        longer = traces[duration]
        assert short.latent_states == longer.latent_states[:9]
        assert short.observations == longer.observations[:9]
        assert _event_shapes(short, cutoff=8) == _event_shapes(longer, cutoff=8)
        assert _decision_shapes(short) == _decision_shapes(longer)


@settings(max_examples=24, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_compound_latent_and_nondrift_observations_match_pump_only(seed: int) -> None:
    compound = generate_trace(
        build_pump_degradation_sensor_drift_scenario(seed=seed, duration_ticks=12)
    )
    pump_only = generate_trace(build_pump_degradation_scenario(seed=seed, duration_ticks=12))
    selected_channel = _selected_channel_id(seed)

    assert compound.latent_states == pump_only.latent_states
    for compound_frame, pump_frame in zip(
        compound.observations, pump_only.observations, strict=True
    ):
        assert compound_frame.overall_status is pump_frame.overall_status
        compound_by_id = {channel.channel_id: channel for channel in compound_frame.channels}
        for pump_channel in pump_frame.channels:
            actual = compound_by_id[pump_channel.channel_id]
            if pump_channel.channel_id != selected_channel:
                assert actual == pump_channel
            else:
                assert actual.model_dump(exclude={"value", "status", "quality"}) == (
                    pump_channel.model_dump(exclude={"value", "status", "quality"})
                )


@settings(max_examples=24, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_compound_bias_has_seeded_direction_fixed_ramp_and_plateau(seed: int) -> None:
    compound = generate_trace(
        build_pump_degradation_sensor_drift_scenario(seed=seed, duration_ticks=12)
    )
    pump_only = generate_trace(build_pump_degradation_scenario(seed=seed, duration_ticks=12))
    selected_channel = _selected_channel_id(seed)
    direction = 1.0 if seed % 2 == 0 else -1.0

    for tick in range(12):
        actual = next(
            channel.value
            for channel in compound.observations[tick].channels
            if channel.channel_id == selected_channel
        )
        baseline = next(
            channel.value
            for channel in pump_only.observations[tick].channels
            if channel.channel_id == selected_channel
        )
        assert actual is not None
        assert baseline is not None
        expected = 0.0 if tick <= 2 else min(0.042, 0.014 * (tick - 2))
        assert actual - baseline == pytest.approx(direction * expected, abs=1e-6)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_compound_does_not_read_or_mutate_process_global_rng(seed: int) -> None:
    original_state = random.getstate()
    try:
        random.seed(817)
        before = random.getstate()
        first = generate_trace(
            build_pump_degradation_sensor_drift_scenario(seed=seed, duration_ticks=12)
        )
        assert random.getstate() == before
        random.seed(91)
        alternate_state = random.getstate()
        second = generate_trace(
            build_pump_degradation_sensor_drift_scenario(seed=seed, duration_ticks=12)
        )
        assert random.getstate() == alternate_state
    finally:
        random.setstate(original_state)
    assert first == second


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_compound_full_channel_roster_and_quality_transition_are_total(seed: int) -> None:
    trace = generate_trace(
        build_pump_degradation_sensor_drift_scenario(seed=seed, duration_ticks=12)
    )
    expected_ids = tuple(sorted(channel.channel_id for channel in ASTER_A_SPEC.channels))
    selected_channel = _selected_channel_id(seed)

    for frame in trace.observations:
        assert tuple(channel.channel_id for channel in frame.channels) == expected_ids
        selected = next(
            channel for channel in frame.channels if channel.channel_id == selected_channel
        )
        assert selected.quality is (
            ChannelQuality.GOOD if frame.tick <= 5 else ChannelQuality.SUSPECT
        )
