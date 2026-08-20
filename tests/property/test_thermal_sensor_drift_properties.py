"""Property checks for the narrow G14 thermal drift-only comparator."""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reactorbench.schemas import ChannelQuality, StateVariable
from reactorbench.simulator import (
    ASTER_A_SPEC,
    build_pump_degradation_scenario,
    build_pump_degradation_sensor_drift_scenario,
    build_stable_scenario,
    build_thermal_sensor_drift_scenario,
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
def test_thermal_drift_is_replayable_bounded_and_content_safe(seed: int, duration: int) -> None:
    scenario = build_thermal_sensor_drift_scenario(seed=seed, duration_ticks=duration)
    first = generate_trace(scenario)
    second = generate_trace(scenario)

    assert first == second
    assert len(first.latent_states) == len(first.observations) == duration
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


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_thermal_drift_short_duration_is_a_semantic_prefix(seed: int) -> None:
    traces = {
        duration: generate_trace(
            build_thermal_sensor_drift_scenario(seed=seed, duration_ticks=duration)
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
def test_thermal_drift_preserves_every_nondrift_channel(seed: int) -> None:
    drift = generate_trace(build_thermal_sensor_drift_scenario(seed=seed))
    stable = generate_trace(build_stable_scenario(seed=seed))
    selected_channel = _selected_channel_id(seed)

    assert drift.latent_states == stable.latent_states
    for drift_frame, stable_frame in zip(drift.observations, stable.observations, strict=True):
        drift_by_id = {channel.channel_id: channel for channel in drift_frame.channels}
        for stable_channel in stable_frame.channels:
            if stable_channel.channel_id != selected_channel:
                assert drift_by_id[stable_channel.channel_id] == stable_channel


@settings(max_examples=24, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_thermal_drift_and_compound_have_the_same_sensor_overlay(seed: int) -> None:
    drift = generate_trace(build_thermal_sensor_drift_scenario(seed=seed))
    stable = generate_trace(build_stable_scenario(seed=seed))
    compound = generate_trace(build_pump_degradation_sensor_drift_scenario(seed=seed))
    pump = generate_trace(build_pump_degradation_scenario(seed=seed))
    selected_channel = _selected_channel_id(seed)

    for tick in range(12):
        observed = {
            name: next(
                channel
                for channel in trace.observations[tick].channels
                if channel.channel_id == selected_channel
            )
            for name, trace in {
                "drift": drift,
                "stable": stable,
                "compound": compound,
                "pump": pump,
            }.items()
        }
        drift_value = observed["drift"].value
        stable_value = observed["stable"].value
        compound_value = observed["compound"].value
        pump_value = observed["pump"].value
        assert drift_value is not None
        assert stable_value is not None
        assert compound_value is not None
        assert pump_value is not None
        drift_delta = drift_value - stable_value
        compound_delta = compound_value - pump_value
        assert drift_delta == pytest.approx(compound_delta, abs=1e-6)
        assert observed["drift"].status is observed["compound"].status
        assert observed["drift"].quality is observed["compound"].quality


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_thermal_drift_does_not_read_or_mutate_process_global_rng(seed: int) -> None:
    original_state = random.getstate()
    try:
        random.seed(817)
        before = random.getstate()
        first = generate_trace(build_thermal_sensor_drift_scenario(seed=seed))
        assert random.getstate() == before
        random.seed(91)
        alternate_state = random.getstate()
        second = generate_trace(build_thermal_sensor_drift_scenario(seed=seed))
        assert random.getstate() == alternate_state
    finally:
        random.setstate(original_state)
    assert first == second


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_thermal_drift_channel_roster_and_quality_transition_are_total(seed: int) -> None:
    trace = generate_trace(build_thermal_sensor_drift_scenario(seed=seed))
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
