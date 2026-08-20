"""Global split atomicity and checksummed manifest contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from reactorbench.dataset.grouping import derive_group_assignment
from reactorbench.dataset.projection import infer_projection_view, project_trajectory
from reactorbench.dataset.splits import (
    SplitManifestEntry,
    build_split_manifest,
    make_split_manifest_entry,
    make_template_alias_plan,
)
from reactorbench.schemas import (
    ComponentState,
    PlantVariant,
    ProvenanceRecord,
    SplitName,
    TaskName,
)
from reactorbench.schemas.scenario import ScenarioDefinition
from reactorbench.schemas.trajectory import StructuredTrajectory
from reactorbench.simulator import (
    build_pump_trip_scenario,
    build_stable_scenario,
    generate_trace,
)


def _trajectory(scenario: ScenarioDefinition, *, suffix: str) -> StructuredTrajectory:
    trace = generate_trace(scenario)
    trajectory_id = f"manifest-{suffix}"
    provenance = ProvenanceRecord(
        dataset_version="0.1.0",
        generator_commit="abcdef1",
        renderer_version="0.1.0",
        seed=scenario.seed,
        trajectory_id=trajectory_id,
        scenario_id=scenario.scenario_id,
        plant_variant_id=scenario.plant_variant_id,
        fault_family_ids=tuple(injection.fault_family for injection in scenario.fault_injections),
        template_family_ids=("compact-log-v1",),
        split_name=SplitName.COUNTERFACTUAL_TEST,
        task_name=TaskName.NEXT_ACTION,
    )
    return trace.to_structured_trajectory(
        trajectory_id=trajectory_id,
        provenance=provenance,
    )


def _entry(
    trajectory: StructuredTrajectory,
    *,
    split_name: SplitName,
    grouped: bool,
) -> SplitManifestEntry:
    decision_tick = trajectory.targets.decisions[-1].decision_tick
    record = project_trajectory(
        trajectory,
        decision_tick=decision_tick,
        task_name=TaskName.NEXT_ACTION,
        view=infer_projection_view(trajectory),
    )
    return make_split_manifest_entry(
        record,
        trajectory,
        split_name=split_name,
        template_family_ids=("compact-log-v1",),
        alias_family_ids=("canonical-v1",),
        group_assignment=(derive_group_assignment(trajectory.scenario) if grouped else None),
    )


def test_complete_g07_group_builds_deterministically_in_counterfactual_test() -> None:
    available = _trajectory(
        build_pump_trip_scenario(
            seed=1200,
            standby_state=ComponentState.AVAILABLE,
        ),
        suffix="g07-available",
    )
    unavailable = _trajectory(
        build_pump_trip_scenario(
            seed=1200,
            standby_state=ComponentState.UNAVAILABLE,
        ),
        suffix="g07-unavailable",
    )
    entries = tuple(
        _entry(trajectory, split_name=SplitName.COUNTERFACTUAL_TEST, grouped=True)
        for trajectory in (available, unavailable)
    )
    plan = make_template_alias_plan(
        {
            SplitName.COUNTERFACTUAL_TEST: (
                ("compact-log-v1",),
                ("canonical-v1",),
            )
        }
    )
    manifest = build_split_manifest(entries, template_alias_plan=plan)
    reversed_manifest = build_split_manifest(reversed(entries), template_alias_plan=plan)
    assert manifest == reversed_manifest
    assert len(manifest.entries) == 2
    assert {entry.counterfactual_group_id for entry in manifest.entries} == {
        entries[0].counterfactual_group_id
    }


def test_complete_group_fails_closed_outside_required_split() -> None:
    trajectories = (
        _trajectory(
            build_pump_trip_scenario(
                seed=1201,
                standby_state=state,
            ),
            suffix=f"wrong-{state.value.lower()}",
        )
        for state in (ComponentState.AVAILABLE, ComponentState.UNAVAILABLE)
    )
    entries = tuple(
        _entry(trajectory, split_name=SplitName.IID_TEST, grouped=True)
        for trajectory in trajectories
    )
    plan = make_template_alias_plan({SplitName.IID_TEST: (("compact-log-v1",), ("canonical-v1",))})
    with pytest.raises(ValidationError, match="counterfactual_test"):
        build_split_manifest(entries, template_alias_plan=plan)


def test_seed_atomicity_and_golden_reservation_are_hard_gates() -> None:
    a = _trajectory(
        build_stable_scenario(seed=1202, plant_variant=PlantVariant.ASTER_A),
        suffix="stable-a",
    )
    b = _trajectory(
        build_stable_scenario(seed=1202, plant_variant=PlantVariant.ASTER_B),
        suffix="stable-b",
    )
    entries = (
        _entry(a, split_name=SplitName.IID_TRAIN, grouped=False),
        _entry(b, split_name=SplitName.IID_TEST, grouped=False),
    )
    plan = make_template_alias_plan(
        {
            SplitName.IID_TRAIN: (("compact-log-v1",), ("canonical-v1",)),
            SplitName.IID_TEST: (("compact-log-v1",), ("canonical-v1",)),
        }
    )
    with pytest.raises(ValidationError, match="seed crosses"):
        build_split_manifest(entries, template_alias_plan=plan)

    golden = _trajectory(build_stable_scenario(seed=50), suffix="golden")
    golden_entry = _entry(golden, split_name=SplitName.IID_TEST, grouped=False)
    with pytest.raises(ValidationError, match="reserved seeds"):
        build_split_manifest((golden_entry,), template_alias_plan=plan)
