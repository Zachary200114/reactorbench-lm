"""Property tests for the developmental G13/G15 fixtures."""

from __future__ import annotations

import random
from itertools import pairwise

from hypothesis import given, settings
from hypothesis import strategies as st

from reactorbench.schemas import ChannelQuality, ObservationStatus, PlantVariant, StateVariable
from reactorbench.simulator import (
    ChannelRole,
    build_abstract_inventory_loss_scenario,
    build_sparse_primary_flow_scenario,
    build_stable_scenario,
    generate_trace,
    get_variant_spec,
)
from reactorbench.simulator.core import SimulationTrace

_VARIANTS = (PlantVariant.ASTER_A, PlantVariant.ASTER_B, PlantVariant.ASTER_C)


@st.composite
def _g13_inputs(draw: st.DrawFn) -> tuple[int, int, PlantVariant]:
    return (
        draw(st.integers(min_value=0, max_value=2**32 - 1)),
        draw(st.sampled_from((10, 12, 64))),
        draw(st.sampled_from(_VARIANTS)),
    )


@st.composite
def _g15_inputs(draw: st.DrawFn) -> tuple[int, int, PlantVariant]:
    return (
        draw(st.integers(min_value=0, max_value=2**32 - 1)),
        draw(st.sampled_from((8, 12, 64))),
        draw(st.sampled_from(_VARIANTS)),
    )


def _event_prefix_shape(trace: SimulationTrace, cutoff: int) -> tuple[dict[str, object], ...]:
    return tuple(
        event.model_dump(mode="json", exclude={"event_id"})
        for event in trace.events
        if event.sim_time <= cutoff
    )


def _decision_shape(trace: SimulationTrace) -> tuple[dict[str, object], ...]:
    return tuple(
        decision.model_dump(mode="json", exclude={"scenario_id", "evidence_event_ids"})
        for decision in trace.targets.decisions
    )


def _channel_value(trace: SimulationTrace, tick: int, channel_id: str) -> float | None:
    return next(
        channel.value
        for channel in trace.observations[tick].channels
        if channel.channel_id == channel_id
    )


@settings(max_examples=24, deadline=None)
@given(parameters=_g13_inputs())
def test_g13_replay_is_seeded_and_duration_prefix_preserving(
    parameters: tuple[int, int, PlantVariant],
) -> None:
    seed, duration, variant = parameters
    first = generate_trace(
        build_abstract_inventory_loss_scenario(
            seed=seed, duration_ticks=duration, plant_variant=variant
        )
    )
    second = generate_trace(
        build_abstract_inventory_loss_scenario(
            seed=seed, duration_ticks=duration, plant_variant=variant
        )
    )
    assert first == second

    traces = {
        length: generate_trace(
            build_abstract_inventory_loss_scenario(
                seed=seed, duration_ticks=length, plant_variant=variant
            )
        )
        for length in (10, 12, 64)
    }
    short = traces[10]
    for length in (12, 64):
        longer = traces[length]
        assert short.latent_states == longer.latent_states[:10]
        assert short.observations == longer.observations[:10]
        assert _event_prefix_shape(short, 9) == _event_prefix_shape(longer, 9)


@settings(max_examples=24, deadline=None)
@given(parameters=_g13_inputs())
def test_g13_process_effects_are_normalized_and_step_bounded(
    parameters: tuple[int, int, PlantVariant],
) -> None:
    seed, duration, variant = parameters
    trace = generate_trace(
        build_abstract_inventory_loss_scenario(
            seed=seed, duration_ticks=duration, plant_variant=variant
        )
    )
    spec = get_variant_spec(variant)
    for state in trace.latent_states:
        assert all(0.0 <= value <= 1.0 for value in state.values.model_dump().values())
    for previous, current in pairwise(trace.latent_states):
        for variable in StateVariable:
            before = getattr(previous.values, variable.value)
            after = getattr(current.values, variable.value)
            assert abs(after - before) <= spec.max_per_tick_step + 1e-9

    assert (
        trace.latent_states[:3]
        == generate_trace(
            build_stable_scenario(seed=seed, duration_ticks=duration, plant_variant=variant)
        ).latent_states[:3]
    )
    assert all(
        trace.latent_states[tick].values.primary_inventory
        <= trace.latent_states[tick - 1].values.primary_inventory
        for tick in range(3, duration)
    )


@settings(max_examples=24, deadline=None)
@given(parameters=_g13_inputs())
def test_g13_independent_channel_evidence_matures_after_the_first_inventory_signal(
    parameters: tuple[int, int, PlantVariant],
) -> None:
    seed, duration, variant = parameters
    trace = generate_trace(
        build_abstract_inventory_loss_scenario(
            seed=seed, duration_ticks=duration, plant_variant=variant
        )
    )
    spec = get_variant_spec(variant)
    primary = spec.channel_for(StateVariable.PRIMARY_INVENTORY, ChannelRole.PRIMARY).channel_id
    redundant = spec.channel_for(StateVariable.PRIMARY_INVENTORY, ChannelRole.REDUNDANT).channel_id
    assert _channel_value(trace, 3, primary) != _channel_value(
        generate_trace(
            build_stable_scenario(seed=seed, duration_ticks=duration, plant_variant=variant)
        ),
        3,
        primary,
    )
    stable = generate_trace(
        build_stable_scenario(seed=seed, duration_ticks=duration, plant_variant=variant)
    )
    assert _channel_value(trace, 3, redundant) != _channel_value(stable, 3, redundant)
    assert _channel_value(trace, 3, primary) == _channel_value(trace, 3, redundant)
    assert _channel_value(trace, 4, primary) == _channel_value(trace, 4, redundant)
    for tick in (3, 4):
        cells = {
            channel.channel_id: channel
            for channel in trace.observations[tick].channels
            if channel.channel_id in {primary, redundant}
        }
        assert cells[primary].quality is ChannelQuality.GOOD
        assert cells[redundant].quality is ChannelQuality.GOOD
    assert _channel_value(trace, 4, redundant) != _channel_value(stable, 4, redundant)
    assert trace.targets.decisions[0].decision_tick == 3
    assert trace.targets.decisions[0].fault_labels == ()
    assert trace.targets.decisions[-1].fault_labels


@settings(max_examples=24, deadline=None)
@given(parameters=_g15_inputs())
def test_g15_latent_truth_is_stable_and_only_selected_primary_flow_cell_changes(
    parameters: tuple[int, int, PlantVariant],
) -> None:
    seed, duration, variant = parameters
    sparse = generate_trace(
        build_sparse_primary_flow_scenario(
            seed=seed, duration_ticks=duration, plant_variant=variant
        )
    )
    stable = generate_trace(
        build_stable_scenario(seed=seed, duration_ticks=duration, plant_variant=variant)
    )
    spec = get_variant_spec(variant)
    selected = spec.channels_for(StateVariable.PRIMARY_FLOW)[seed % 2].channel_id
    assert sparse.latent_states == stable.latent_states
    assert sparse.scenario.fault_injections == ()
    expected_channels = tuple(sorted(channel.channel_id for channel in spec.channels))
    assert all(
        tuple(channel.channel_id for channel in frame.channels) == expected_channels
        for frame in sparse.observations
    )
    for tick in range(duration):
        differences = tuple(
            channel.channel_id
            for channel, baseline in zip(
                sparse.observations[tick].channels,
                stable.observations[tick].channels,
                strict=True,
            )
            if channel != baseline
        )
        assert differences == ((selected,) if tick == 2 else ())
    selected_cell = next(
        channel for channel in sparse.observations[2].channels if channel.channel_id == selected
    )
    stable_cell = next(
        channel for channel in stable.observations[2].channels if channel.channel_id == selected
    )
    assert selected_cell.status is ObservationStatus.WATCH
    assert selected_cell.quality is ChannelQuality.GOOD
    assert selected_cell.value is not None
    assert stable_cell.value is not None
    fictional_offset = stable_cell.value - selected_cell.value
    assert 0.018 - 1e-6 <= fictional_offset <= 0.024 + 1e-6


@settings(max_examples=24, deadline=None)
@given(parameters=_g15_inputs())
def test_g15_is_rng_independent_and_has_only_a_sparse_decision_prefix(
    parameters: tuple[int, int, PlantVariant],
) -> None:
    seed, duration, variant = parameters
    random.seed(7)
    first = generate_trace(
        build_sparse_primary_flow_scenario(
            seed=seed, duration_ticks=duration, plant_variant=variant
        )
    )
    random.seed(91)
    second = generate_trace(
        build_sparse_primary_flow_scenario(
            seed=seed, duration_ticks=duration, plant_variant=variant
        )
    )
    assert first == second
    decision = first.targets.decisions[0]
    assert decision.decision_tick == 2
    assert decision.fault_labels == ()
    assert decision.evidence_event_ids
    assert _decision_shape(first) == (
        {
            "schema_version": "0.1.0",
            "decision_tick": 2,
            "diagnosis_status": "UNRESOLVED",
            "fault_labels": [],
            "evidence_slots": ["MISSING_DECISIVE_EVIDENCE"],
            "immediate_action": "INSUFFICIENT_EVIDENCE",
            "abstention_reason": "INSUFFICIENT_EVIDENCE",
        },
    )
