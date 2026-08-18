from __future__ import annotations

import json

from reactorbench.schemas import ProvenanceRecord, SplitName, TaskName
from reactorbench.simulator import build_sensor_drift_scenario, generate_trace


def test_visible_payload_hides_truth_and_structured_trajectory_validates() -> None:
    trace = generate_trace(build_sensor_drift_scenario(seed=13))
    payload = trace.visible_payload()
    serialized = json.dumps(payload, sort_keys=True)

    assert set(payload) == {"schema_version", "observations", "events"}
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
