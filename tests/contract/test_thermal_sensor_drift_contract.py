"""Public, leakage, provenance, and fail-closed contracts for G14's comparator."""

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
    build_thermal_sensor_drift_scenario,
    generate_trace,
    scan_prohibited_content,
)
from reactorbench.simulator.core import SimulationTrace


def _valid_trace(*, seed: int = 20, duration_ticks: int = 12) -> SimulationTrace:
    return generate_trace(
        build_thermal_sensor_drift_scenario(seed=seed, duration_ticks=duration_ticks)
    )


def _selected_channel(seed: int) -> str:
    channels = ASTER_A_SPEC.channels_for(StateVariable.PRIMARY_THERMAL_STATE)
    return channels[(seed // 2) % len(channels)].channel_id


def test_public_visible_payload_is_allowlisted_and_does_not_leak_truth() -> None:
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


def test_structured_trajectory_preserves_canonical_audit_truth_and_provenance() -> None:
    trace = _valid_trace(seed=23)
    trajectory_id = "g14-thermal-comparator-23"
    provenance = ProvenanceRecord(
        dataset_version="0.1.0",
        generator_commit="abcdef1",
        renderer_version="0.1.0",
        seed=23,
        trajectory_id=trajectory_id,
        scenario_id=trace.scenario.scenario_id,
        plant_variant_id=PlantVariant.ASTER_A,
        fault_family_ids=(FaultFamily.SENSOR_DRIFT,),
        template_family_ids=("template-g14",),
        split_name=SplitName.COMPOSITION_TEST,
        task_name=TaskName.COUNTERFACTUAL_COMPARE,
    )
    trajectory = trace.to_structured_trajectory(trajectory_id=trajectory_id, provenance=provenance)

    assert trajectory.provenance == provenance
    assert trajectory.scenario == trace.scenario
    assert trajectory.latent_states == trace.latent_states
    assert trajectory.observations == trace.observations
    assert trajectory.events == trace.events
    assert trajectory.targets == trace.targets
    assert (
        provenance.stable_hash()
        == ProvenanceRecord.model_validate(provenance.model_dump(mode="python")).stable_hash()
    )


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
        {"channel_id": True},
        {"channel_id": 7},
        {"channel_id": "aster-primary-thermal-state-a "},
        {"channel_id": "aster-primary-flow-a"},
        {"channel_id": "aster-b-primary-thermal-state-a"},
        {"plant_variant": PlantVariant.ASTER_A},
        {"onset_tick": 2},
        {"severity": SeverityBand.LOW},
    ],
)
def test_builder_rejects_loose_out_of_scope_or_profile_broadening(
    kwargs: dict[str, object],
) -> None:
    arguments: dict[str, object] = {"seed": 20, "duration_ticks": 12}
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError, UnsupportedScenarioError)):
        cast(Any, build_thermal_sensor_drift_scenario)(**arguments)


def test_model_copy_tampering_fails_closed() -> None:
    scenario = build_thermal_sensor_drift_scenario(seed=20)
    injection = scenario.fault_injections[0]
    selected_channel = _selected_channel(20)
    alternate_channel = next(
        channel.channel_id
        for channel in ASTER_A_SPEC.channels_for(StateVariable.PRIMARY_THERMAL_STATE)
        if channel.channel_id != selected_channel
    )
    actions = scenario.action_sequence
    invalid_injection_updates: tuple[dict[str, object], ...] = (
        {"fault_family": FaultFamily.SENSOR_NOISE},
        {"component_id": "aster-train-cirrus"},
        {"component_id": "aster-instrument-vireo "},
        {"channel_id": alternate_channel},
        {"channel_id": "aster-primary-flow-a"},
        {"channel_id": selected_channel + " "},
        {"onset_tick": 3},
        {"onset_tick": 2.0},
        {"onset_tick": True},
        {"severity": SeverityBand.MEDIUM},
        {"severity": "LOW"},
        {"duration_ticks": 1},
    )
    invalid_scenarios: list[ScenarioDefinition] = [
        scenario.model_copy(update={"fault_injections": (injection.model_copy(update=update),)})
        for update in invalid_injection_updates
    ]
    invalid_scenarios.extend(
        (
            scenario.model_copy(update={"scenario_id": "aster-a-thermal-drift-spoofed"}),
            scenario.model_copy(update={"plant_variant_id": PlantVariant.ASTER_B}),
            scenario.model_copy(update={"driver": ScenarioDriver.LOAD_TRANSIENT}),
            scenario.model_copy(update={"duration_ticks": 8}),
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
                        *actions,
                        ScenarioAction(
                            decision_tick=6,
                            action=ActionLabel.CONTINUE_MONITORING,
                        ),
                    )
                }
            ),
            scenario.model_copy(update={"fault_injections": [injection]}),
            scenario.model_copy(update={"fault_injections": (injection, injection)}),
        )
    )

    for candidate in invalid_scenarios:
        with pytest.raises(UnsupportedScenarioError):
            generate_trace(candidate)


def test_events_and_targets_respect_visible_causal_time() -> None:
    trace = _valid_trace()
    events_by_id = {event.event_id: event for event in trace.events}

    assert tuple(event.event_index for event in trace.events) == tuple(range(len(trace.events)))
    assert tuple(event.sim_time for event in trace.events) == tuple(
        sorted(event.sim_time for event in trace.events)
    )
    assert all(
        events_by_id[related_id].event_index < event.event_index
        for event in trace.events
        for related_id in event.related_event_ids
    )
    for decision in trace.targets.decisions:
        assert all(
            events_by_id[event_id].sim_time <= decision.decision_tick
            for event_id in decision.evidence_event_ids
        )
        future_actions = {
            event.event_id
            for event in trace.events
            if event.action_label is not None and event.sim_time > decision.decision_tick
        }
        assert future_actions.isdisjoint(decision.evidence_event_ids)
