"""Public and fail-closed contracts for developmental G14."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from reactorbench.schemas import (
    ActionLabel,
    FaultFamily,
    PlantVariant,
    ProvenanceRecord,
    ScenarioAction,
    ScenarioDefinition,
    ScenarioDriver,
    SeverityBand,
    SplitName,
    StateVariable,
    TaskName,
)
from reactorbench.simulator import (
    ASTER_A_SPEC,
    UnsupportedScenarioError,
    build_pump_degradation_sensor_drift_scenario,
    build_sensor_noise_scenario,
    generate_trace,
    scan_prohibited_content,
)
from reactorbench.simulator.core import SimulationTrace


def _valid_trace(*, seed: int = 20, duration_ticks: int = 12) -> SimulationTrace:
    return generate_trace(
        build_pump_degradation_sensor_drift_scenario(seed=seed, duration_ticks=duration_ticks)
    )


def _selected_pt_channel(seed: int) -> str:
    channels = ASTER_A_SPEC.channels_for(StateVariable.PRIMARY_THERMAL_STATE)
    return channels[(seed // 2) % len(channels)].channel_id


def test_visible_payload_is_allowlisted_and_does_not_leak_compound_truth() -> None:
    trace = _valid_trace()
    payload = trace.visible_payload()
    serialized = json.dumps(payload, sort_keys=True)

    assert set(payload) == {
        "schema_version",
        "plant_variant_id",
        "dependency_map_context",
        "standby_context",
        "observations",
        "events",
    }
    assert payload["plant_variant_id"] == PlantVariant.ASTER_A.value
    assert payload["dependency_map_context"] is None
    assert payload["standby_context"] is None
    assert trace.scenario.scenario_id not in serialized
    for forbidden in (
        "SENSOR_DRIFT",
        "PUMP_DEGRADATION",
        "fault_family",
        "fault_injection",
        "STEADY_OPERATION",
        "driver",
        "severity",
        "onset_tick",
        "duration_ticks",
        "scenario_id",
        "action_sequence",
        "latent_states",
        "health",
        "pending_maintenance",
        "targets",
        "provenance",
    ):
        assert forbidden not in serialized
    assert scan_prohibited_content(trace) == ()


def test_structured_trajectory_preserves_canonical_compound_audit_truth() -> None:
    trace = _valid_trace(seed=23)
    trajectory_id = "g14-compound-trace-23"
    provenance = ProvenanceRecord(
        dataset_version="0.1.0",
        generator_commit="abcdef1",
        renderer_version="0.1.0",
        seed=23,
        trajectory_id=trajectory_id,
        scenario_id=trace.scenario.scenario_id,
        plant_variant_id=PlantVariant.ASTER_A,
        fault_family_ids=(FaultFamily.PUMP_DEGRADATION, FaultFamily.SENSOR_DRIFT),
        template_family_ids=("template-g14",),
        split_name=SplitName.COMPOSITION_TEST,
        task_name=TaskName.FAULT_FAMILY,
    )
    trajectory = trace.to_structured_trajectory(trajectory_id=trajectory_id, provenance=provenance)

    expected_faults = (FaultFamily.SENSOR_DRIFT, FaultFamily.PUMP_DEGRADATION)
    assert provenance.fault_family_ids == expected_faults
    assert trajectory.provenance.fault_family_ids == expected_faults
    assert (
        tuple(injection.fault_family for injection in trajectory.scenario.fault_injections)
        == expected_faults
    )
    assert trajectory.targets.decisions[-1].fault_labels == expected_faults
    assert trajectory.events == trace.events
    assert (
        provenance.stable_hash()
        == ProvenanceRecord.model_validate(provenance.model_dump(mode="python")).stable_hash()
    )


def test_builder_accepts_only_exact_aster_a_component_and_thermal_channel_overrides() -> None:
    component_id = ASTER_A_SPEC.primary_train_ids[1]
    channel_id = ASTER_A_SPEC.channels_for(StateVariable.PRIMARY_THERMAL_STATE)[0].channel_id
    scenario = build_pump_degradation_sensor_drift_scenario(
        seed=20,
        duration_ticks=12,
        component_id=component_id,
        channel_id=channel_id,
    )
    sensor, pump = scenario.fault_injections

    assert sensor.channel_id == channel_id
    assert sensor.component_id == next(
        channel.component_id
        for channel in ASTER_A_SPEC.channels
        if channel.channel_id == channel_id
    )
    assert pump.component_id == component_id
    assert component_id in scenario.scenario_id
    assert channel_id in scenario.scenario_id


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"seed": 20.0},
        {"seed": "20"},
        {"seed": -1},
        {"seed": 2**32},
        {"duration_ticks": True},
        {"duration_ticks": 12.0},
        {"duration_ticks": 8},
        {"duration_ticks": 65},
        {"component_id": True},
        {"component_id": 7},
        {"component_id": "cirrus"},
        {"component_id": "aster-train-cirrus "},
        {"component_id": "aster-b-train-nomad"},
        {"channel_id": True},
        {"channel_id": 7},
        {"channel_id": "aster-primary-thermal-state-a "},
        {"channel_id": "aster-primary-flow-a"},
        {"channel_id": "aster-b-primary-thermal-state-a"},
        {"plant_variant": PlantVariant.ASTER_A},
        {"plant_variant": PlantVariant.ASTER_B},
    ],
)
def test_builder_rejects_loose_out_of_scope_or_noncanonical_inputs(
    kwargs: dict[str, object],
) -> None:
    arguments: dict[str, object] = {"seed": 20, "duration_ticks": 12}
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError, UnsupportedScenarioError)):
        cast(Any, build_pump_degradation_sensor_drift_scenario)(**arguments)


def test_raw_schema_input_canonicalizes_fault_order_but_model_copy_must_not() -> None:
    scenario = build_pump_degradation_sensor_drift_scenario(seed=20, duration_ticks=12)
    raw = scenario.model_dump(mode="python")
    # Strict contracts require the canonical tuple container. The normal
    # validation path still canonicalizes a reversed tuple of raw members.
    raw["fault_injections"] = tuple(
        injection.model_dump(mode="python") for injection in reversed(scenario.fault_injections)
    )
    canonical = ScenarioDefinition.model_validate(raw)

    assert canonical.fault_injections == scenario.fault_injections
    assert generate_trace(canonical) == generate_trace(scenario)
    reversed_copy = scenario.model_copy(
        update={"fault_injections": tuple(reversed(scenario.fault_injections))}
    )
    with pytest.raises(UnsupportedScenarioError):
        generate_trace(reversed_copy)


def test_compound_model_copy_injection_tampering_fails_closed() -> None:
    scenario = build_pump_degradation_sensor_drift_scenario(seed=20, duration_ticks=12)
    sensor, pump = scenario.fault_injections
    selected_pt = _selected_pt_channel(20)
    pf_channel = ASTER_A_SPEC.channels_for(StateVariable.PRIMARY_FLOW)[0].channel_id

    def rejected(candidate: ScenarioDefinition) -> None:
        with pytest.raises(UnsupportedScenarioError):
            generate_trace(candidate)

    sensor_updates: tuple[dict[str, object], ...] = (
        {"fault_family": FaultFamily.SENSOR_NOISE},
        {"component_id": pump.component_id},
        {"component_id": "aster-instrument-vireo "},
        {"channel_id": pf_channel},
        {"channel_id": "aster-primary-thermal-state-x"},
        {"channel_id": selected_pt + " "},
        {"onset_tick": 3},
        {"onset_tick": 2.0},
        {"onset_tick": True},
        {"severity": SeverityBand.MEDIUM},
        {"severity": "LOW"},
        {"duration_ticks": 1},
    )
    for update in sensor_updates:
        rejected(
            scenario.model_copy(
                update={"fault_injections": (sensor.model_copy(update=update), pump)}
            )
        )

    pump_updates: tuple[dict[str, object], ...] = (
        {"fault_family": FaultFamily.PUMP_TRIP},
        {"component_id": "aster-train-unknown"},
        {"component_id": "cirrus"},
        {"channel_id": selected_pt},
        {"onset_tick": 3},
        {"onset_tick": 2.0},
        {"onset_tick": True},
        {"severity": SeverityBand.MEDIUM},
        {"severity": "LOW"},
        {"duration_ticks": 1},
    )
    for update in pump_updates:
        rejected(
            scenario.model_copy(
                update={"fault_injections": (sensor, pump.model_copy(update=update))}
            )
        )

    noise = build_sensor_noise_scenario(seed=20).fault_injections[0]
    rejected(scenario.model_copy(update={"fault_injections": (sensor, pump, noise)}))
    rejected(scenario.model_copy(update={"fault_injections": (sensor, sensor)}))
    rejected(scenario.model_copy(update={"fault_injections": (sensor,)}))
    rejected(scenario.model_copy(update={"fault_injections": (pump,)}))
    rejected(scenario.model_copy(update={"fault_injections": [sensor, pump]}))
    rejected(
        scenario.model_copy(
            update={
                "fault_injections": (
                    sensor.model_dump(mode="python"),
                    pump,
                )
            }
        )
    )
    rejected(scenario.model_copy(update={"fault_injections": (sensor, object())}))


def test_compound_model_copy_scenario_and_action_tampering_fails_closed() -> None:
    scenario = build_pump_degradation_sensor_drift_scenario(seed=20, duration_ticks=12)
    actions = scenario.action_sequence
    invalid_scenarios = (
        scenario.model_copy(update={"scenario_id": "aster-a-case-spoofed"}),
        scenario.model_copy(update={"plant_variant_id": PlantVariant.ASTER_B}),
        scenario.model_copy(update={"plant_variant_id": PlantVariant.ASTER_C}),
        scenario.model_copy(update={"plant_variant_id": "ASTER-A"}),
        scenario.model_copy(update={"driver": ScenarioDriver.LOAD_TRANSIENT}),
        scenario.model_copy(update={"driver": "STEADY_OPERATION"}),
        scenario.model_copy(update={"seed": True}),
        scenario.model_copy(update={"seed": 20.0}),
        scenario.model_copy(update={"duration_ticks": 8}),
        scenario.model_copy(update={"duration_ticks": 12.0}),
        scenario.model_copy(update={"standby_context": {}}),
        scenario.model_copy(update={"dependency_map_context": {}}),
        scenario.model_copy(update={"action_sequence": ()}),
        scenario.model_copy(update={"action_sequence": [*actions]}),
        scenario.model_copy(update={"action_sequence": tuple(reversed(actions))}),
        scenario.model_copy(
            update={
                "action_sequence": (
                    actions[0].model_copy(update={"decision_tick": 4}),
                    *actions[1:],
                )
            }
        ),
        scenario.model_copy(
            update={
                "action_sequence": (
                    actions[0].model_copy(update={"action": "INSUFFICIENT_EVIDENCE"}),
                    *actions[1:],
                )
            }
        ),
        scenario.model_copy(
            update={
                "action_sequence": (
                    *actions,
                    ScenarioAction(decision_tick=8, action=ActionLabel.CONTINUE_MONITORING),
                )
            }
        ),
    )
    for candidate in invalid_scenarios:
        with pytest.raises(UnsupportedScenarioError):
            generate_trace(candidate)


def test_compound_events_and_targets_respect_visible_causal_time() -> None:
    trace = _valid_trace()
    events_by_id = {event.event_id: event for event in trace.events}

    assert tuple(event.event_index for event in trace.events) == tuple(range(len(trace.events)))
    assert tuple(event.sim_time for event in trace.events) == tuple(
        sorted(event.sim_time for event in trace.events)
    )
    for event in trace.events:
        assert all(
            events_by_id[related_id].event_index < event.event_index
            for related_id in event.related_event_ids
        )
    for decision in trace.targets.decisions:
        assert all(
            events_by_id[evidence_id].sim_time <= decision.decision_tick
            for evidence_id in decision.evidence_event_ids
        )
        applied = next(
            event
            for event in trace.events
            if event.action_label is decision.immediate_action
            and event.sim_time == decision.decision_tick + 1
        )
        assert applied.event_type.value == "ACTION_APPLIED"
