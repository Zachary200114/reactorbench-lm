"""Small deterministic properties for projection and family grouping."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from reactorbench.dataset.contracts import ProjectionRecord
from reactorbench.dataset.grouping import derive_group_assignment
from reactorbench.dataset.projection import infer_projection_view, project_trajectory
from reactorbench.schemas import ProvenanceRecord, SplitName, TaskName
from reactorbench.simulator import (
    build_pump_trip_scenario,
    build_sensor_noise_scenario,
    generate_trace,
)


def _project_noise(seed: int) -> ProjectionRecord:
    scenario = build_sensor_noise_scenario(seed=seed)
    trace = generate_trace(scenario)
    trajectory_id = f"property-noise-{seed}"
    trajectory = trace.to_structured_trajectory(
        trajectory_id=trajectory_id,
        provenance=ProvenanceRecord(
            dataset_version="0.1.0",
            generator_commit="abcdef1",
            renderer_version="0.1.0",
            seed=seed,
            trajectory_id=trajectory_id,
            scenario_id=scenario.scenario_id,
            plant_variant_id=scenario.plant_variant_id,
            fault_family_ids=tuple(
                injection.fault_family for injection in scenario.fault_injections
            ),
            template_family_ids=("property",),
            split_name=SplitName.IID_TEST,
            task_name=TaskName.FAULT_FAMILY,
        ),
    )
    decision_tick = trajectory.targets.decisions[-1].decision_tick
    return project_trajectory(
        trajectory,
        decision_tick=decision_tick,
        task_name=TaskName.FAULT_FAMILY,
        view=infer_projection_view(trajectory),
    )


@settings(max_examples=8, deadline=None)
@given(seed=st.integers(min_value=1300, max_value=1400))
def test_projection_replay_is_byte_stable(seed: int) -> None:
    first = _project_noise(seed)
    second = _project_noise(seed)
    assert first == second
    assert first.checksum_sha256 == second.checksum_sha256


@settings(max_examples=8, deadline=None)
@given(seed=st.integers(min_value=1401, max_value=1500))
def test_group_identity_changes_with_the_atomic_seed(seed: int) -> None:
    first = derive_group_assignment(build_pump_trip_scenario(seed=seed))
    second = derive_group_assignment(build_pump_trip_scenario(seed=seed + 1))
    assert first is not None
    assert second is not None
    assert first.counterfactual_group_id != second.counterfactual_group_id
