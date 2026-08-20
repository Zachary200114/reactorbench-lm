from __future__ import annotations

import json

from reactorbench.schemas import (
    ComponentState,
    FaultFamily,
    ProvenanceRecord,
    SplitName,
    TaskName,
)
from reactorbench.simulator import (
    build_load_transient_scenario,
    build_pump_degradation_scenario,
    build_pump_trip_scenario,
    build_sensor_drift_scenario,
    build_sensor_noise_scenario,
    build_sensor_stuck_load_scenario,
    generate_trace,
)


def test_visible_payload_hides_truth_and_structured_trajectory_validates() -> None:
    trace = generate_trace(build_sensor_drift_scenario(seed=13))
    payload = trace.visible_payload()
    serialized = json.dumps(payload, sort_keys=True)

    assert set(payload) == {"schema_version", "standby_context", "observations", "events"}
    assert payload["standby_context"] is None
    for forbidden in ("SENSOR_DRIFT", "fault_family", "fault_injection", "latent", "targets"):
        assert forbidden not in serialized

    trajectory = trace.to_structured_trajectory(
        trajectory_id="trace-13",
        provenance=ProvenanceRecord(
            dataset_version="0.1.0",
            generator_commit="abcdef1",
            renderer_version="0.1.0",
            seed=13,
            trajectory_id="trace-13",
            scenario_id=trace.scenario.scenario_id,
            plant_variant_id=trace.scenario.plant_variant_id,
            fault_family_ids=tuple(
                injection.fault_family for injection in trace.scenario.fault_injections
            ),
            template_family_ids=("template-a",),
            split_name=SplitName.IID_TEST,
            task_name=TaskName.FAULT_FAMILY,
        ),
    )

    assert trajectory.events == trace.events
    assert all(
        related_id in {earlier.event_id for earlier in trace.events[:index]}
        for index, event in enumerate(trace.events)
        for related_id in event.related_event_ids
    )


def test_load_transient_visible_payload_hides_driver_and_fault_truth() -> None:
    trace = generate_trace(build_load_transient_scenario(seed=14))
    serialized = json.dumps(trace.visible_payload(), sort_keys=True)

    for forbidden in ("LOAD_TRANSIENT", "fault_family", "fault_injection", "latent", "targets"):
        assert forbidden not in serialized


def test_sensor_stuck_load_visible_payload_and_provenance_keep_driver_hidden() -> None:
    trace = generate_trace(build_sensor_stuck_load_scenario(seed=14))
    payload = json.dumps(trace.visible_payload(), sort_keys=True)
    provenance = ProvenanceRecord(
        dataset_version="0.1.0",
        generator_commit="abcdef1",
        renderer_version="0.1.0",
        seed=14,
        trajectory_id="stuck-trace-14",
        scenario_id=trace.scenario.scenario_id,
        plant_variant_id=trace.scenario.plant_variant_id,
        fault_family_ids=tuple(
            injection.fault_family for injection in trace.scenario.fault_injections
        ),
        template_family_ids=("template-a",),
        split_name=SplitName.IID_TEST,
        task_name=TaskName.FAULT_FAMILY,
    )

    assert "SENSOR_STUCK" not in payload
    assert "LOAD_TRANSIENT" not in payload
    assert trace.scenario.scenario_id not in payload
    for forbidden in ("driver", "severity", "onset", "provenance"):
        assert forbidden not in payload
    assert provenance.fault_family_ids == (FaultFamily.SENSOR_STUCK,)
    trajectory = trace.to_structured_trajectory(
        trajectory_id="stuck-trace-14", provenance=provenance
    )
    assert trajectory.events == trace.events
    assert trajectory.targets == trace.targets
    assert trajectory.provenance.fault_family_ids == (FaultFamily.SENSOR_STUCK,)
    assert tuple(injection.fault_family for injection in trajectory.scenario.fault_injections) == (
        FaultFamily.SENSOR_STUCK,
    )


def test_sensor_noise_visible_payload_hides_truth_and_trajectory_validates() -> None:
    trace = generate_trace(build_sensor_noise_scenario(seed=17, duration_ticks=8))
    payload = json.dumps(trace.visible_payload(), sort_keys=True)
    provenance = ProvenanceRecord(
        dataset_version="0.1.0",
        generator_commit="abcdef1",
        renderer_version="0.1.0",
        seed=17,
        trajectory_id="noise-trace-17",
        scenario_id=trace.scenario.scenario_id,
        plant_variant_id=trace.scenario.plant_variant_id,
        fault_family_ids=(FaultFamily.SENSOR_NOISE,),
        template_family_ids=("template-a",),
        split_name=SplitName.IID_TEST,
        task_name=TaskName.FAULT_FAMILY,
    )

    assert set(trace.visible_payload()) == {
        "schema_version",
        "standby_context",
        "observations",
        "events",
    }
    assert trace.visible_payload()["standby_context"] is None
    assert trace.scenario.scenario_id not in payload
    for forbidden in (
        "SENSOR_NOISE",
        "STEADY_OPERATION",
        "fault_family",
        "fault_injection",
        "driver",
        "severity",
        "onset",
        "provenance",
        "latent",
        "targets",
        "NOISY",
    ):
        assert forbidden not in payload
    trajectory = trace.to_structured_trajectory(
        trajectory_id="noise-trace-17", provenance=provenance
    )
    assert trajectory.events == trace.events
    assert trajectory.targets == trace.targets
    assert trajectory.provenance.fault_family_ids == (FaultFamily.SENSOR_NOISE,)
    assert tuple(injection.fault_family for injection in trajectory.scenario.fault_injections) == (
        FaultFamily.SENSOR_NOISE,
    )


def test_pump_degradation_visible_payload_hides_fault_and_health_truth() -> None:
    trace = generate_trace(build_pump_degradation_scenario(seed=20, duration_ticks=9))
    payload = json.dumps(trace.visible_payload(), sort_keys=True)
    provenance = ProvenanceRecord(
        dataset_version="0.1.0",
        generator_commit="abcdef1",
        renderer_version="0.1.0",
        seed=20,
        trajectory_id="pump-trace-20",
        scenario_id=trace.scenario.scenario_id,
        plant_variant_id=trace.scenario.plant_variant_id,
        fault_family_ids=(FaultFamily.PUMP_DEGRADATION,),
        template_family_ids=("template-a",),
        split_name=SplitName.IID_TEST,
        task_name=TaskName.FAULT_FAMILY,
    )

    assert set(trace.visible_payload()) == {
        "schema_version",
        "standby_context",
        "observations",
        "events",
    }
    assert trace.visible_payload()["standby_context"] is None
    assert trace.scenario.scenario_id not in payload
    for forbidden in (
        "PUMP_DEGRADATION",
        "fault_family",
        "fault_injection",
        "STEADY_OPERATION",
        "severity",
        "onset",
        "health",
        "pending_maintenance",
        "provenance",
        "latent",
        "targets",
    ):
        assert forbidden not in payload
    trajectory = trace.to_structured_trajectory(
        trajectory_id="pump-trace-20", provenance=provenance
    )
    assert trajectory.events == trace.events
    assert trajectory.targets == trace.targets


def test_pump_trip_visible_context_is_safe_and_trajectory_retains_audit_truth() -> None:
    trace = generate_trace(
        build_pump_trip_scenario(
            seed=23,
            duration_ticks=8,
            standby_state=ComponentState.AVAILABLE,
        )
    )
    payload = trace.visible_payload()
    serialized = json.dumps(payload, sort_keys=True)
    context = payload["standby_context"]
    assert isinstance(context, dict)
    assert set(context) == {
        "context_id",
        "active_train_id",
        "standby_train_id",
        "standby_state",
        "standby_support_bus_id",
        "support_bus_state",
        "standby_start_delay_ticks",
    }
    assert context["standby_state"] == ComponentState.AVAILABLE.value
    assert context["support_bus_state"] == ComponentState.AVAILABLE.value
    assert trace.scenario.scenario_id not in serialized
    for forbidden in (
        "PUMP_TRIP",
        "fault_family",
        "fault_injection",
        "STEADY_OPERATION",
        "severity",
        "onset",
        "action_sequence",
        "provenance",
        "latent",
        "targets",
    ):
        assert forbidden not in serialized
    serialized_context = json.dumps(context, sort_keys=True)
    for forbidden in (
        "fault",
        "severity",
        "action",
        "decision_tick",
        "injection",
        "SELECT_SYNTHETIC_STANDBY_TRAIN",
    ):
        assert forbidden not in serialized_context

    provenance = ProvenanceRecord(
        dataset_version="0.1.0",
        generator_commit="abcdef1",
        renderer_version="0.1.0",
        seed=23,
        trajectory_id="trip-trace-23",
        scenario_id=trace.scenario.scenario_id,
        plant_variant_id=trace.scenario.plant_variant_id,
        fault_family_ids=(FaultFamily.PUMP_TRIP,),
        template_family_ids=("template-a",),
        split_name=SplitName.IID_TEST,
        task_name=TaskName.FAULT_FAMILY,
    )
    trajectory = trace.to_structured_trajectory(
        trajectory_id="trip-trace-23", provenance=provenance
    )
    assert trajectory.events == trace.events
    assert trajectory.targets == trace.targets
    assert trajectory.scenario.standby_context == trace.scenario.standby_context
    assert trajectory.provenance.fault_family_ids == (FaultFamily.PUMP_TRIP,)
    assert all(
        related_id in {earlier.event_id for earlier in trace.events[:index]}
        for index, event in enumerate(trace.events)
        for related_id in event.related_event_ids
    )

    unavailable = generate_trace(
        build_pump_trip_scenario(
            seed=23,
            duration_ticks=8,
            standby_state=ComponentState.UNAVAILABLE,
        )
    )
    unavailable_provenance = provenance.model_copy(
        update={
            "trajectory_id": "trip-unavailable-23",
            "scenario_id": unavailable.scenario.scenario_id,
        }
    )
    unavailable_trajectory = unavailable.to_structured_trajectory(
        trajectory_id="trip-unavailable-23",
        provenance=unavailable_provenance,
    )
    assert [
        (decision.decision_tick, decision.immediate_action.value)
        for decision in unavailable_trajectory.targets.decisions
    ] == [
        (5, "REDUCE_SIMULATED_LOAD"),
        (6, "ENTER_SIMULATED_STABLE_STATE"),
    ]
