"""Exact family-role tests for target-independent counterfactual grouping."""

from __future__ import annotations

from reactorbench.dataset.grouping import (
    CounterfactualFamily,
    CounterfactualVariant,
    derive_group_assignment,
    group_scenarios,
)
from reactorbench.schemas import ComponentState
from reactorbench.simulator import (
    build_pump_degradation_scenario,
    build_pump_degradation_sensor_drift_scenario,
    build_pump_trip_scenario,
    build_sparse_primary_flow_scenario,
    build_support_power_interruption_scenario,
    build_thermal_sensor_drift_scenario,
    build_valve_lag_scenario,
    build_valve_stuck_scenario,
)


def test_g07_group_hash_excludes_standby_availability() -> None:
    available = build_pump_trip_scenario(seed=1100, standby_state=ComponentState.AVAILABLE)
    unavailable = build_pump_trip_scenario(seed=1100, standby_state=ComponentState.UNAVAILABLE)
    first = derive_group_assignment(available)
    second = derive_group_assignment(unavailable)
    assert first is not None
    assert second is not None
    assert first.counterfactual_group_id == second.counterfactual_group_id
    assert {first.variant_id, second.variant_id} == {
        CounterfactualVariant.STANDBY_AVAILABLE,
        CounterfactualVariant.STANDBY_UNAVAILABLE,
    }
    assert group_scenarios((available, unavailable))[0].is_complete


def test_g08_g09_requires_lag_three_lag_four_and_stuck_roles() -> None:
    scenarios = (
        build_valve_lag_scenario(seed=1101, lag_ticks=3),
        build_valve_lag_scenario(seed=1101, lag_ticks=4),
        build_valve_stuck_scenario(seed=1101),
    )
    assignments = tuple(derive_group_assignment(scenario) for scenario in scenarios)
    assert all(assignment is not None for assignment in assignments)
    assert (
        len({assignment.counterfactual_group_id for assignment in assignments if assignment}) == 1
    )
    group = group_scenarios(scenarios)[0]
    assert group.is_complete
    assert tuple(member.variant_id for member in group.members) == (
        CounterfactualVariant.VALVE_LAG_3,
        CounterfactualVariant.VALVE_LAG_4,
        CounterfactualVariant.VALVE_STUCK,
    )


def test_g12_and_g14_exact_siblings_form_complete_family_groups() -> None:
    g12 = (
        build_support_power_interruption_scenario(seed=1102, include_dependency_map=True),
        build_support_power_interruption_scenario(seed=1102, include_dependency_map=False),
    )
    g14 = (
        build_pump_degradation_scenario(seed=1103),
        build_thermal_sensor_drift_scenario(seed=1103),
        build_pump_degradation_sensor_drift_scenario(seed=1103),
    )
    groups = group_scenarios((*g12, *g14))
    assert {group.family for group in groups} == {
        CounterfactualFamily.G12_DEPENDENCY_MAP,
        CounterfactualFamily.G14_COMPOSITION,
    }
    assert all(group.is_complete for group in groups)


def test_g15_is_explicitly_incomplete_without_fabricated_relatives() -> None:
    sparse = build_sparse_primary_flow_scenario(seed=1104)
    assignment = derive_group_assignment(sparse)
    assert assignment is not None
    assert assignment.family is CounterfactualFamily.G15_EVIDENCE_SUFFICIENCY
    assert not assignment.expanded_siblings_supported
    group = group_scenarios((sparse,))[0]
    assert not group.is_complete
    assert group.exclusion_reason is not None
