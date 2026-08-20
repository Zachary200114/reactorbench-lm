"""Cross-cutting completion gate for the developmental Phase 2 generator."""

from __future__ import annotations

import json
import random

import pytest

from reactorbench.schemas import (
    ComponentState,
    DiagnosisStatus,
    EventType,
    FaultFamily,
    PlantVariant,
    ProvenanceRecord,
    ScenarioDefinition,
    SplitName,
    TaskName,
)
from reactorbench.simulator import (
    build_abstract_inventory_loss_scenario,
    build_flow_imbalance_scenario,
    build_load_transient_scenario,
    build_pump_degradation_scenario,
    build_pump_degradation_sensor_drift_scenario,
    build_pump_trip_scenario,
    build_sensor_drift_scenario,
    build_sensor_noise_scenario,
    build_sensor_stuck_load_scenario,
    build_sparse_primary_flow_scenario,
    build_stable_scenario,
    build_support_power_interruption_scenario,
    build_transfer_efficiency_loss_scenario,
    build_valve_lag_scenario,
    build_valve_stuck_scenario,
    generate_trace,
    get_variant_spec,
    scan_prohibited_content,
)

_PROCESS_FAULTS = {
    FaultFamily.PUMP_DEGRADATION,
    FaultFamily.PUMP_TRIP,
    FaultFamily.VALVE_LAG,
    FaultFamily.VALVE_STUCK,
    FaultFamily.TRANSFER_EFFICIENCY_LOSS,
    FaultFamily.FLOW_IMBALANCE,
    FaultFamily.SUPPORT_POWER_INTERRUPTION,
    FaultFamily.ABSTRACT_INVENTORY_LOSS,
}
_HIDDEN_AUDIT_KEYS = {
    "scenario_id",
    "seed",
    "driver",
    "fault_injections",
    "severity",
    "onset_tick",
    "duration_ticks",
    "action_sequence",
    "latent_states",
    "targets",
    "provenance",
    "health",
    "pending_maintenance",
}


def _phase2_cases() -> tuple[tuple[str, ScenarioDefinition], ...]:
    cases: list[tuple[str, ScenarioDefinition]] = []
    for variant in PlantVariant:
        cases.extend(
            (
                (
                    f"g01-{variant.value}",
                    build_stable_scenario(seed=23, plant_variant=variant),
                ),
                (
                    f"g10-{variant.value}",
                    build_transfer_efficiency_loss_scenario(seed=23, plant_variant=variant),
                ),
                (
                    f"g11-{variant.value}",
                    build_flow_imbalance_scenario(seed=23, plant_variant=variant),
                ),
                (
                    f"g13-{variant.value}",
                    build_abstract_inventory_loss_scenario(seed=23, plant_variant=variant),
                ),
                (
                    f"g15-{variant.value}",
                    build_sparse_primary_flow_scenario(seed=23, plant_variant=variant),
                ),
            )
        )
    cases.extend((f"g02-{seed}", build_load_transient_scenario(seed=seed)) for seed in (20, 21))
    cases.extend((f"g03-{seed}", build_sensor_drift_scenario(seed=seed)) for seed in (20, 21))
    cases.extend((f"g04-{seed}", build_sensor_stuck_load_scenario(seed=seed)) for seed in (20, 21))
    cases.extend((f"g05-{seed}", build_sensor_noise_scenario(seed=seed)) for seed in (20, 21))
    cases.extend((f"g06-{seed}", build_pump_degradation_scenario(seed=seed)) for seed in (20, 21))
    cases.extend(
        (
            f"g07-{seed}-{standby_state.value}",
            build_pump_trip_scenario(seed=seed, standby_state=standby_state),
        )
        for seed in (20, 21)
        for standby_state in (ComponentState.AVAILABLE, ComponentState.UNAVAILABLE)
    )
    cases.extend(
        (
            f"g08-{lag_ticks}",
            build_valve_lag_scenario(seed=23, lag_ticks=lag_ticks),
        )
        for lag_ticks in (3, 4)
    )
    cases.append(("g09", build_valve_stuck_scenario(seed=23)))
    cases.extend(
        (
            f"g12-{variant.value}-map-{include_map}",
            build_support_power_interruption_scenario(
                seed=23,
                plant_variant=variant,
                include_dependency_map=include_map,
            ),
        )
        for variant in (PlantVariant.ASTER_A, PlantVariant.ASTER_B)
        for include_map in (True, False)
    )
    cases.extend(
        (
            f"g14-{seed}",
            build_pump_degradation_sensor_drift_scenario(seed=seed),
        )
        for seed in range(4)
    )
    return tuple(cases)


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(
            *(_nested_keys(item) for item in value.values()),
            set(),
        )
    if isinstance(value, list):
        return set().union(*(_nested_keys(item) for item in value), set())
    return set()


@pytest.mark.parametrize(("case_id", "scenario"), _phase2_cases())
def test_phase2_scenarios_share_the_global_generator_contract(
    case_id: str, scenario: ScenarioDefinition
) -> None:
    original_rng_state = random.getstate()
    try:
        random.seed(918_273)
        seeded_rng_state = random.getstate()
        trace = generate_trace(scenario)
        assert random.getstate() == seeded_rng_state
    finally:
        random.setstate(original_rng_state)

    assert trace == generate_trace(scenario)
    assert len(trace.latent_states) == len(trace.observations) == scenario.duration_ticks
    assert tuple(state.tick for state in trace.latent_states) == tuple(
        range(scenario.duration_ticks)
    )
    assert tuple(frame.tick for frame in trace.observations) == tuple(
        range(scenario.duration_ticks)
    )

    spec = get_variant_spec(scenario.plant_variant_id)
    expected_components = {component.component_id for component in spec.components}
    expected_channels = {channel.channel_id for channel in spec.channels}
    assert all(
        {component.component_id for component in state.components} == expected_components
        for state in trace.latent_states
    )
    assert all(
        {channel.channel_id for channel in frame.channels} == expected_channels
        for frame in trace.observations
    )
    assert all(
        0.0 <= value <= 1.0
        for state in trace.latent_states
        for value in state.values.model_dump().values()
    )
    assert all(
        channel.value is None or 0.0 <= channel.value <= 1.0
        for frame in trace.observations
        for channel in frame.channels
    )

    expected_decisions = tuple(
        (action.decision_tick, action.action) for action in scenario.action_sequence
    )
    actual_decisions = tuple(
        (decision.decision_tick, decision.immediate_action) for decision in trace.targets.decisions
    )
    assert actual_decisions == expected_decisions
    for decision in trace.targets.decisions:
        if decision.decision_tick + 1 < scenario.duration_ticks:
            assert any(
                event.event_type is EventType.ACTION_APPLIED
                and event.sim_time == decision.decision_tick + 1
                and event.action_label is decision.immediate_action
                for event in trace.events
            )

    assert scan_prohibited_content(scenario) == ()
    assert scan_prohibited_content(trace) == ()
    audit_payload = trace.visible_payload()
    assert set(audit_payload) == {
        "schema_version",
        "plant_variant_id",
        "dependency_map_context",
        "standby_context",
        "observations",
        "events",
    }
    assert not (_nested_keys(audit_payload) & _HIDDEN_AUDIT_KEYS)
    serialized_payload = json.dumps(audit_payload, sort_keys=True)
    assert all(fault.value not in serialized_payload for fault in FaultFamily)

    trajectory_id = f"phase2-{case_id.lower()}"
    provenance = ProvenanceRecord(
        dataset_version="0.1.0",
        generator_commit="abcdef1",
        renderer_version="0.1.0",
        seed=scenario.seed,
        trajectory_id=trajectory_id,
        scenario_id=scenario.scenario_id,
        plant_variant_id=scenario.plant_variant_id,
        fault_family_ids=tuple(injection.fault_family for injection in scenario.fault_injections),
        template_family_ids=("phase2-matrix",),
        split_name=SplitName.COUNTERFACTUAL_TEST,
        task_name=TaskName.FAULT_FAMILY,
    )
    trajectory = trace.to_structured_trajectory(
        trajectory_id=trajectory_id,
        provenance=provenance,
    )
    assert trajectory.scenario_id == scenario.scenario_id
    assert trajectory.provenance.fault_family_ids == provenance.fault_family_ids


def test_phase2_single_fault_matrix_covers_the_closed_fault_vocabulary() -> None:
    observed = {
        scenario.fault_injections[0].fault_family
        for _, scenario in _phase2_cases()
        if len(scenario.fault_injections) == 1
    }
    assert observed == set(FaultFamily)


def test_process_faults_have_a_latent_effect_before_each_diagnosis() -> None:
    for case_id, scenario in _phase2_cases():
        trace = generate_trace(scenario)
        diagnosed_process_faults = {
            fault
            for decision in trace.targets.decisions
            if decision.diagnosis_status is DiagnosisStatus.DIAGNOSED
            for fault in decision.fault_labels
            if fault in _PROCESS_FAULTS
        }
        if not diagnosed_process_faults:
            continue
        stable = generate_trace(
            build_stable_scenario(
                seed=scenario.seed,
                duration_ticks=scenario.duration_ticks,
                plant_variant=scenario.plant_variant_id,
            )
        )
        first_effect_tick = next(
            state.tick
            for state, baseline in zip(trace.latent_states, stable.latent_states, strict=True)
            if state != baseline
        )
        for fault in diagnosed_process_faults:
            first_diagnosis_tick = min(
                decision.decision_tick
                for decision in trace.targets.decisions
                if fault in decision.fault_labels
            )
            assert first_effect_tick < first_diagnosis_tick, case_id
