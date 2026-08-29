"""Property checks for the deterministic developmental G12 fixture."""

from __future__ import annotations

import random
from itertools import pairwise

from hypothesis import example, given, settings
from hypothesis import strategies as st

from reactorbench.schemas import ComponentState, PlantVariant, StateVariable
from reactorbench.simulator import (
    ASTER_A_SPEC,
    ASTER_B_SPEC,
    AsterVariantSpec,
    ComponentRole,
    build_support_power_interruption_scenario,
    generate_trace,
)
from reactorbench.simulator.core import SimulationTrace


@st.composite
def _g12_inputs(draw: st.DrawFn) -> tuple[int, int, PlantVariant, bool]:
    return (
        draw(st.integers(min_value=0, max_value=2**32 - 1)),
        draw(st.sampled_from((8, 12, 64))),
        draw(st.sampled_from((PlantVariant.ASTER_A, PlantVariant.ASTER_B))),
        draw(st.booleans()),
    )


def _spec_for(variant: PlantVariant) -> AsterVariantSpec:
    return ASTER_A_SPEC if variant is PlantVariant.ASTER_A else ASTER_B_SPEC


def _bus_id(variant: PlantVariant) -> str:
    return _spec_for(variant).component_for_role(ComponentRole.SUPPORT_BUS_TWO).component_id


def _event_prefix_shape(trace: SimulationTrace, cutoff: int) -> tuple[dict[str, object], ...]:
    """Compare visible event semantics while ignoring only scenario-bearing IDs."""

    return tuple(
        event.model_dump(mode="json", exclude={"event_id", "related_event_ids"})
        for event in trace.events
        if event.sim_time <= cutoff
    )


def _decision_shapes(trace: SimulationTrace) -> tuple[dict[str, object], ...]:
    """Compare target semantics while ignoring scenario/event IDs derived from duration."""

    return tuple(
        decision.model_dump(mode="json", exclude={"scenario_id", "evidence_event_ids"})
        for decision in trace.targets.decisions
    )


@settings(max_examples=24, deadline=None)
@given(parameters=_g12_inputs())
def test_g12_replay_is_seeded(parameters: tuple[int, int, PlantVariant, bool]) -> None:
    seed, duration, variant, include_dependency_map = parameters
    scenario = build_support_power_interruption_scenario(
        seed=seed,
        duration_ticks=duration,
        plant_variant=variant,
        include_dependency_map=include_dependency_map,
    )
    first = generate_trace(scenario)
    second = generate_trace(scenario)
    assert first == second
    assert len(first.latent_states) == len(first.observations) == duration


@settings(max_examples=20, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    variant=st.sampled_from((PlantVariant.ASTER_A, PlantVariant.ASTER_B)),
    include_dependency_map=st.booleans(),
)
def test_g12_shorter_duration_is_a_prefix_of_longer_durations(
    seed: int, variant: PlantVariant, include_dependency_map: bool
) -> None:
    traces = {
        duration: generate_trace(
            build_support_power_interruption_scenario(
                seed=seed,
                duration_ticks=duration,
                plant_variant=variant,
                include_dependency_map=include_dependency_map,
            )
        )
        for duration in (8, 12, 64)
    }
    short = traces[8]
    for duration in (12, 64):
        longer = traces[duration]
        assert short.latent_states == longer.latent_states[:8]
        assert short.observations == longer.observations[:8]
        assert _event_prefix_shape(short, 7) == _event_prefix_shape(longer, 7)
        assert _decision_shapes(short) == _decision_shapes(longer)


@settings(max_examples=24, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    variant=st.sampled_from((PlantVariant.ASTER_A, PlantVariant.ASTER_B)),
    include_dependency_map=st.booleans(),
)
def test_g12_values_are_normalized_and_per_tick_steps_are_bounded(
    seed: int, variant: PlantVariant, include_dependency_map: bool
) -> None:
    trace = generate_trace(
        build_support_power_interruption_scenario(
            seed=seed,
            duration_ticks=12,
            plant_variant=variant,
            include_dependency_map=include_dependency_map,
        )
    )
    spec = _spec_for(variant)
    for state in trace.latent_states:
        assert all(0.0 <= value <= 1.0 for value in state.values.model_dump().values())
    for previous, current in pairwise(trace.latent_states):
        for variable in StateVariable:
            before = getattr(previous.values, variable.value)
            after = getattr(current.values, variable.value)
            assert abs(after - before) <= spec.max_per_tick_step + 1e-9


@settings(max_examples=20, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    variant=st.sampled_from((PlantVariant.ASTER_A, PlantVariant.ASTER_B)),
)
def test_g12_map_context_does_not_change_pre_decision_generation(
    seed: int, variant: PlantVariant
) -> None:
    included = generate_trace(
        build_support_power_interruption_scenario(
            seed=seed,
            duration_ticks=12,
            plant_variant=variant,
            include_dependency_map=True,
        )
    )
    withheld = generate_trace(
        build_support_power_interruption_scenario(
            seed=seed,
            duration_ticks=12,
            plant_variant=variant,
            include_dependency_map=False,
        )
    )
    assert included.latent_states[:6] == withheld.latent_states[:6]
    assert included.observations[:6] == withheld.observations[:6]
    assert tuple(event for event in included.events if event.sim_time <= 5) == tuple(
        event for event in withheld.events if event.sim_time <= 5
    )


@settings(max_examples=20, deadline=None)
@example(seed=532786336, variant=PlantVariant.ASTER_A)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    variant=st.sampled_from((PlantVariant.ASTER_A, PlantVariant.ASTER_B)),
)
def test_g12_affected_set_is_exactly_the_registry_bus_two_dependents(
    seed: int, variant: PlantVariant
) -> None:
    trace = generate_trace(
        build_support_power_interruption_scenario(
            seed=seed, duration_ticks=12, plant_variant=variant
        )
    )
    spec = _spec_for(variant)
    bus_id = _bus_id(variant)
    expected_unavailable = {bus_id, *spec.dependents_for(bus_id)}
    unavailable_at_t3 = {
        component.component_id
        for component in trace.latent_states[3].components
        if component.state is ComponentState.UNAVAILABLE
    }
    assert unavailable_at_t3 == expected_unavailable

    all_component_ids = {component.component_id for component in spec.components}
    assert all(
        component.state is ComponentState.AVAILABLE
        for component in trace.latent_states[3].components
        if component.component_id in all_component_ids - expected_unavailable
    )


@settings(max_examples=20, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    variant=st.sampled_from((PlantVariant.ASTER_A, PlantVariant.ASTER_B)),
)
def test_g12_does_not_use_process_global_randomness(seed: int, variant: PlantVariant) -> None:
    random.seed(7)
    first = generate_trace(
        build_support_power_interruption_scenario(seed=seed, plant_variant=variant)
    )
    random.seed(91)
    second = generate_trace(
        build_support_power_interruption_scenario(seed=seed, plant_variant=variant)
    )
    assert first == second
