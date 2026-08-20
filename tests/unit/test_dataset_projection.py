"""Focused acceptance tests for the Phase 3 renderer-input projection boundary."""

from __future__ import annotations

import pytest

from reactorbench.dataset.contracts import PromptEvidenceTarget
from reactorbench.dataset.grouping import group_scenarios
from reactorbench.dataset.projection import (
    ProjectionError,
    infer_projection_view,
    project_continuation,
    project_counterfactual_pair,
    project_trajectory,
)
from reactorbench.schemas import (
    ActionLabel,
    ComponentState,
    EventType,
    EvidenceSlot,
    ProvenanceRecord,
    SplitName,
    StateVariable,
    TaskName,
)
from reactorbench.schemas.trajectory import StructuredTrajectory
from reactorbench.simulator import (
    build_pump_trip_scenario,
    build_sensor_noise_scenario,
    build_sparse_primary_flow_scenario,
    build_support_power_interruption_scenario,
    build_valve_lag_scenario,
    generate_trace,
)


def _trajectory(scenario: object, *, suffix: str) -> StructuredTrajectory:
    trace = generate_trace(scenario)  # type: ignore[arg-type]
    trajectory_id = f"phase3-{suffix}"
    provenance = ProvenanceRecord(
        dataset_version="0.1.0",
        generator_commit="abcdef1",
        renderer_version="0.1.0",
        seed=trace.scenario.seed,
        trajectory_id=trajectory_id,
        scenario_id=trace.scenario.scenario_id,
        plant_variant_id=trace.scenario.plant_variant_id,
        fault_family_ids=tuple(
            injection.fault_family for injection in trace.scenario.fault_injections
        ),
        template_family_ids=("phase3-projection",),
        split_name=SplitName.IID_TEST,
        task_name=TaskName.FAULT_FAMILY,
    )
    return trace.to_structured_trajectory(
        trajectory_id=trajectory_id,
        provenance=provenance,
    )


def _decision_tick(trajectory: StructuredTrajectory, *, final: bool = True) -> int:
    return trajectory.targets.decisions[-1 if final else 0].decision_tick


def test_standard_recipe_is_family_bounded_not_a_full_channel_roster() -> None:
    trajectory = _trajectory(build_sensor_noise_scenario(seed=1000), suffix="noise")
    view = infer_projection_view(trajectory)
    record = project_trajectory(
        trajectory,
        decision_tick=_decision_tick(trajectory),
        task_name=TaskName.FAULT_FAMILY,
        view=view,
    )
    selected_variables = {fact.variable for fact in record.model_input.observation_facts}
    assert selected_variables == {
        StateVariable.PRIMARY_FLOW,
        StateVariable.PRIMARY_THERMAL_STATE,
    }
    full_roster = len(trajectory.observations[0].channels)
    selected_roster = len({fact.channel_id for fact in record.model_input.observation_facts})
    assert selected_roster < full_roster
    assert record.lineage.projection_recipe_id.startswith("g03-g05-observation-fault-d")
    assert record.model_input.context_facts == ()
    assert all(
        event.event_type is not EventType.ACTION_APPLIED for event in record.model_input.event_facts
    )


def test_g15_exposes_exactly_one_sparse_fact_and_no_context() -> None:
    trajectory = _trajectory(build_sparse_primary_flow_scenario(seed=1001), suffix="g15")
    record = project_trajectory(
        trajectory,
        decision_tick=2,
        task_name=TaskName.FAULT_FAMILY,
        view=infer_projection_view(trajectory),
    )
    assert len(record.model_input.observation_facts) == 1
    assert record.model_input.observation_facts[0].tick == 2
    assert record.model_input.observation_facts[0].variable is StateVariable.PRIMARY_FLOW
    assert len(record.model_input.event_facts) == 1
    assert record.model_input.context_facts == ()


def test_g12_context_contains_only_links_when_included_and_nothing_when_withheld() -> None:
    included = _trajectory(
        build_support_power_interruption_scenario(seed=1002, include_dependency_map=True),
        suffix="g12-included",
    )
    withheld = _trajectory(
        build_support_power_interruption_scenario(seed=1002, include_dependency_map=False),
        suffix="g12-withheld",
    )
    included_record = project_trajectory(
        included,
        decision_tick=5,
        task_name=TaskName.NEXT_ACTION,
        view=infer_projection_view(included),
    )
    withheld_record = project_trajectory(
        withheld,
        decision_tick=5,
        task_name=TaskName.NEXT_ACTION,
        view=infer_projection_view(withheld),
    )
    assert included_record.model_input.context_facts
    assert {fact.fact_kind.value for fact in included_record.model_input.context_facts} == {
        "dependency_link"
    }
    assert withheld_record.model_input.context_facts == ()


def test_g07_context_is_task_aware_and_evidence_refs_are_semantically_visible() -> None:
    trajectory = _trajectory(
        build_pump_trip_scenario(
            seed=1003,
            standby_state=ComponentState.AVAILABLE,
        ),
        suffix="g07",
    )
    view = infer_projection_view(trajectory)
    diagnosis = project_trajectory(
        trajectory,
        decision_tick=5,
        task_name=TaskName.FAULT_FAMILY,
        view=view,
    )
    evidence = project_trajectory(
        trajectory,
        decision_tick=5,
        task_name=TaskName.EXTRACT_EVIDENCE,
        view=view,
    )
    action = project_trajectory(
        trajectory,
        decision_tick=5,
        task_name=TaskName.NEXT_ACTION,
        view=view,
    )
    assert diagnosis.model_input.context_facts == ()
    assert len(evidence.model_input.context_facts) == 1
    assert len(action.model_input.context_facts) == 1
    target = evidence.task_target.target
    assert isinstance(target, PromptEvidenceTarget)
    assert EvidenceSlot.STANDBY_AVAILABLE in target.evidence_slots
    visible_refs = {
        *(fact.fact_ref for fact in evidence.model_input.observation_facts),
        *(fact.fact_ref for fact in evidence.model_input.event_facts),
        *(fact.fact_ref for fact in evidence.model_input.context_facts),
    }
    assert set(target.fact_refs).issubset(visible_refs)
    assert evidence.model_input.context_facts[0].fact_ref in target.fact_refs
    assert all(
        event.event_type is not EventType.BENIGN_NOTE for event in evidence.model_input.event_facts
    )


def test_model_input_is_invariant_to_a_target_only_mutation() -> None:
    trajectory = _trajectory(build_sensor_noise_scenario(seed=1004), suffix="mutation")
    tick = _decision_tick(trajectory)
    original = project_trajectory(
        trajectory,
        decision_tick=tick,
        task_name=TaskName.NEXT_ACTION,
        view=infer_projection_view(trajectory),
    )
    decision = trajectory.targets.decisions[-1].model_copy(
        update={"immediate_action": ActionLabel.CONTINUE_MONITORING}
    )
    mutated_targets = trajectory.targets.model_copy(
        update={"decisions": (*trajectory.targets.decisions[:-1], decision)}
    )
    mutated = trajectory.model_copy(update={"targets": mutated_targets})
    changed = project_trajectory(
        mutated,
        decision_tick=tick,
        task_name=TaskName.NEXT_ACTION,
        view=infer_projection_view(mutated),
    )
    assert (
        original.model_input.structured_fingerprint()
        == changed.model_input.structured_fingerprint()
    )


def test_evidence_grounding_ignores_audit_event_ids_and_source_annotations() -> None:
    trajectory = _trajectory(build_sensor_noise_scenario(seed=1010), suffix="audit-neutral")
    tick = _decision_tick(trajectory)
    view = infer_projection_view(trajectory)
    original = project_trajectory(
        trajectory,
        decision_tick=tick,
        task_name=TaskName.EXTRACT_EVIDENCE,
        view=view,
    )

    decisions = tuple(
        decision.model_copy(update={"evidence_event_ids": ()})
        if decision.decision_tick == tick
        else decision
        for decision in trajectory.targets.decisions
    )
    audit_neutral = trajectory.model_copy(
        update={
            "events": tuple(
                event.model_copy(update={"evidence_slots": (), "related_event_ids": ()})
                for event in trajectory.events
            ),
            "targets": trajectory.targets.model_copy(update={"decisions": decisions}),
        }
    )
    changed = project_trajectory(
        audit_neutral,
        decision_tick=tick,
        task_name=TaskName.EXTRACT_EVIDENCE,
        view=view,
    )

    assert original.model_input == changed.model_input
    assert original.task_target == changed.task_target


def test_continuation_uses_an_event_index_exclusive_cut_and_rejects_action_target() -> None:
    trajectory = _trajectory(build_valve_lag_scenario(seed=1005), suffix="continue")
    target_index = next(
        event.event_index
        for event in trajectory.events
        if event.event_index > 0 and event.event_type is not EventType.ACTION_APPLIED
    )
    record = project_continuation(
        trajectory,
        next_event_index=target_index,
        view=infer_projection_view(trajectory),
    )
    assert record.model_input.source_event_index_exclusive == target_index
    assert record.lineage.source_event_index_exclusive == target_index
    assert record.model_input.observation_facts == ()
    action_index = next(
        event.event_index
        for event in trajectory.events
        if event.event_type is EventType.ACTION_APPLIED
    )
    with pytest.raises(ProjectionError, match="ACTION_APPLIED"):
        project_continuation(
            trajectory,
            next_event_index=action_index,
            view=infer_projection_view(trajectory),
        )


def test_projection_rejects_a_declared_view_that_bypasses_special_policy() -> None:
    trajectory = _trajectory(build_sparse_primary_flow_scenario(seed=1006), suffix="view")
    with pytest.raises(ProjectionError, match="does not match required policy"):
        project_trajectory(
            trajectory,
            decision_tick=2,
            task_name=TaskName.FAULT_FAMILY,
            view=next(
                view
                for view in type(infer_projection_view(trajectory))
                if view.value == "standard_decision"
            ),
        )


def test_counterfactual_compare_requires_and_uses_a_complete_group() -> None:
    available = _trajectory(
        build_pump_trip_scenario(
            seed=1007,
            standby_state=ComponentState.AVAILABLE,
        ),
        suffix="pair-available",
    )
    unavailable = _trajectory(
        build_pump_trip_scenario(
            seed=1007,
            standby_state=ComponentState.UNAVAILABLE,
        ),
        suffix="pair-unavailable",
    )
    group = group_scenarios((available.scenario, unavailable.scenario))[0]
    pair = project_counterfactual_pair(
        available,
        unavailable,
        baseline_decision_tick=5,
        counterfactual_decision_tick=5,
        group=group,
        baseline_view=infer_projection_view(available),
        counterfactual_view=infer_projection_view(unavailable),
    )
    assert pair.task_target.task_name is TaskName.COUNTERFACTUAL_COMPARE
    assert pair.model_input.baseline.context_facts
    assert pair.model_input.counterfactual.context_facts
