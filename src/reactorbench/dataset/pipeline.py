"""Deterministic Phase 3 structured build graph before narrative rendering.

This module deliberately stops at the renderer boundary.  It produces validated
audit trajectories, task projections, counterfactual pairs, and a split-first
manifest.  Human catalog approval is enforced separately before any prose is
rendered.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Literal

from pydantic import Field, field_validator, model_validator

from reactorbench.dataset.config import DevelopmentDatasetConfig, dataset_config_sha256
from reactorbench.dataset.contracts import (
    DATASET_CONTRACT_VERSION,
    CounterfactualProjectionRecord,
    ProjectionRecord,
)
from reactorbench.dataset.grouping import (
    CounterfactualFamily,
    CounterfactualGroup,
    CounterfactualVariant,
    GroupAssignment,
    derive_group_assignment,
    group_scenarios,
    require_complete_group,
)
from reactorbench.dataset.projection import (
    infer_projection_view,
    project_continuation,
    project_counterfactual_pair,
    project_trajectory,
)
from reactorbench.dataset.scenarios import PlannedScenario, build_development_scenario_plan
from reactorbench.dataset.splits import (
    SplitManifest,
    build_split_manifest,
    make_split_manifest_entry,
    make_template_alias_plan,
)
from reactorbench.schemas.base import (
    SCHEMA_VERSION,
    ContractId,
    ContractModel,
    SchemaVersion,
    canonical_sha256,
    require_unique,
)
from reactorbench.schemas.enums import EventType, SplitName, TaskName
from reactorbench.schemas.provenance import GitCommit, ProvenanceRecord
from reactorbench.schemas.trajectory import StructuredTrajectory
from reactorbench.simulator import generate_trace

_DECISION_TASKS: tuple[TaskName, ...] = (
    TaskName.FAULT_FAMILY,
    TaskName.EXTRACT_EVIDENCE,
    TaskName.NEXT_ACTION,
    TaskName.INCIDENT_SUMMARY,
)
_SPLIT_ORDER = {split: index for index, split in enumerate(SplitName)}
_PAIR_ROLES: dict[
    CounterfactualFamily,
    tuple[tuple[CounterfactualVariant, CounterfactualVariant], ...],
] = {
    CounterfactualFamily.G07_STANDBY: (
        (
            CounterfactualVariant.STANDBY_AVAILABLE,
            CounterfactualVariant.STANDBY_UNAVAILABLE,
        ),
    ),
    CounterfactualFamily.G08_G09_VALVE: (
        (CounterfactualVariant.VALVE_LAG_3, CounterfactualVariant.VALVE_STUCK),
        (CounterfactualVariant.VALVE_LAG_4, CounterfactualVariant.VALVE_STUCK),
    ),
    CounterfactualFamily.G12_DEPENDENCY_MAP: (
        (CounterfactualVariant.MAP_INCLUDED, CounterfactualVariant.MAP_WITHHELD),
    ),
    CounterfactualFamily.G14_COMPOSITION: (
        (CounterfactualVariant.PUMP_ONLY, CounterfactualVariant.COMPOUND),
        (CounterfactualVariant.SENSOR_ONLY, CounterfactualVariant.COMPOUND),
    ),
    # Generator-supported G15 evidence-expanded siblings do not exist yet.
    CounterfactualFamily.G15_EVIDENCE_SUFFICIENCY: (),
}


class DevelopmentTrajectoryRecord(ContractModel):
    """One split-assigned audit trajectory and its pre-render plan."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    dataset_contract_version: Literal["0.1.0"] = DATASET_CONTRACT_VERSION
    split_name: SplitName
    case_family: ContractId
    template_family_id: ContractId
    alias_family_id: ContractId
    corruption_plan: ContractId
    trajectory: StructuredTrajectory
    group_assignment: GroupAssignment | None = None
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def plan_and_trajectory_are_consistent(self) -> DevelopmentTrajectoryRecord:
        provenance = self.trajectory.provenance
        if provenance.split_name is not self.split_name:
            raise ValueError("trajectory provenance must match the planned split")
        if provenance.template_family_ids != (self.template_family_id,):
            raise ValueError("trajectory provenance must match the planned template family")
        if (
            self.group_assignment is not None
            and self.group_assignment.scenario_id != self.trajectory.scenario_id
        ):
            raise ValueError("group assignment must reference this trajectory scenario")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("development trajectory record checksum mismatch")
        return self


class DevelopmentProjectionSummary(ContractModel):
    """Measured inventory for the deterministic pre-render build."""

    trajectory_count: int = Field(ge=1)
    projection_count: int = Field(ge=1)
    counterfactual_pair_count: int = Field(ge=0)
    complete_group_count: int = Field(ge=0)
    incomplete_g15_group_count: int = Field(ge=0)
    single_input_structured_duplicate_count: int = Field(ge=0)
    counterfactual_input_structured_duplicate_count: int = Field(ge=0)
    trajectory_counts_by_split: tuple[tuple[SplitName, int], ...]
    projection_counts_by_task: tuple[tuple[TaskName, int], ...]

    @field_validator("trajectory_counts_by_split", "projection_counts_by_task", mode="after")
    @classmethod
    def counts_are_positive_and_unique(
        cls, values: tuple[tuple[object, int], ...]
    ) -> tuple[tuple[object, int], ...]:
        if not values or any(count <= 0 for _, count in values):
            raise ValueError("summary counts must be nonempty and positive")
        require_unique(tuple(key for key, _ in values), field_name="summary count keys")
        return values


class DevelopmentProjectionBundle(ContractModel):
    """Complete structured Phase 3 development candidate before rendering."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    dataset_contract_version: Literal["0.1.0"] = DATASET_CONTRACT_VERSION
    artifact_status: Literal["structured_projection_audit"] = "structured_projection_audit"
    dataset_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_commit: GitCommit
    trajectories: tuple[DevelopmentTrajectoryRecord, ...]
    groups: tuple[CounterfactualGroup, ...]
    projections: tuple[ProjectionRecord, ...]
    counterfactual_projections: tuple[CounterfactualProjectionRecord, ...]
    split_manifest: SplitManifest
    summary: DevelopmentProjectionSummary
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def inventory_and_checksum_are_consistent(self) -> DevelopmentProjectionBundle:
        trajectory_ids = tuple(item.trajectory.trajectory_id for item in self.trajectories)
        projection_ids = tuple(item.projection_id for item in self.projections)
        pair_ids = tuple(item.pair_id for item in self.counterfactual_projections)
        require_unique(trajectory_ids, field_name="development trajectory IDs")
        require_unique(projection_ids, field_name="development projection IDs")
        require_unique(pair_ids, field_name="counterfactual pair IDs")
        if trajectory_ids != tuple(sorted(trajectory_ids)):
            raise ValueError("development trajectories must use canonical ID order")
        if projection_ids != tuple(sorted(projection_ids)):
            raise ValueError("development projections must use canonical ID order")
        if pair_ids != tuple(sorted(pair_ids)):
            raise ValueError("counterfactual projections must use canonical ID order")
        if tuple(group.counterfactual_group_id for group in self.groups) != tuple(
            sorted(group.counterfactual_group_id for group in self.groups)
        ):
            raise ValueError("counterfactual groups must use canonical ID order")
        manifest_ids = {entry.projection_id for entry in self.split_manifest.entries}
        if manifest_ids != set(projection_ids):
            raise ValueError("split manifest must cover every single-input projection exactly")
        _validate_structured_relationships(self)
        if self.summary.trajectory_count != len(self.trajectories):
            raise ValueError("summary trajectory count mismatch")
        if self.summary.projection_count != len(self.projections):
            raise ValueError("summary projection count mismatch")
        if self.summary.counterfactual_pair_count != len(self.counterfactual_projections):
            raise ValueError("summary counterfactual count mismatch")
        expected_summary = _summary(
            self.trajectories,
            self.groups,
            self.projections,
            self.counterfactual_projections,
        )
        if self.summary != expected_summary:
            raise ValueError("summary does not match the canonical structured inventory")
        single_duplicate_count = _single_input_structured_duplicate_count(self.projections)
        if self.summary.single_input_structured_duplicate_count != single_duplicate_count:
            raise ValueError("summary single-input structured duplicate count mismatch")
        pair_duplicate_count = _counterfactual_input_structured_duplicate_count(
            self.counterfactual_projections
        )
        if self.summary.counterfactual_input_structured_duplicate_count != pair_duplicate_count:
            raise ValueError("summary counterfactual structured duplicate count mismatch")
        if single_duplicate_count or pair_duplicate_count:
            raise ValueError("task-scoped model inputs must not contain structured duplicates")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("development projection bundle checksum mismatch")
        return self


def _validate_structured_relationships(bundle: DevelopmentProjectionBundle) -> None:
    """Prove joins between separately serialized structured Phase 3 records.

    Individual contracts checksum their own contents.  This additional pass binds
    every projection, manifest entry, group, and pair back to the exact source
    trajectory so typed artifact reconstruction cannot accept substituted rows.
    """

    trajectories = {record.trajectory.trajectory_id: record for record in bundle.trajectories}
    projections = {record.projection_id: record for record in bundle.projections}
    entries = {entry.projection_id: entry for entry in bundle.split_manifest.entries}
    groups = {group.counterfactual_group_id: group for group in bundle.groups}

    assignments = {
        record.trajectory.scenario_id: record.group_assignment
        for record in bundle.trajectories
        if record.group_assignment is not None
    }
    if len(assignments) != sum(
        record.group_assignment is not None for record in bundle.trajectories
    ):
        raise ValueError("counterfactual assignments must have unique source scenarios")
    expected_group_ids = {assignment.counterfactual_group_id for assignment in assignments.values()}
    if set(groups) != expected_group_ids:
        raise ValueError("counterfactual groups must cover every and only assigned scenario")
    for group_id, counterfactual_group in groups.items():
        expected_members = tuple(
            assignment
            for assignment in assignments.values()
            if assignment.counterfactual_group_id == group_id
        )
        if tuple(counterfactual_group.members) != tuple(
            sorted(
                expected_members,
                key=lambda item: (
                    item.expected_variants.index(item.variant_id),
                    item.scenario_id,
                ),
            )
        ):
            raise ValueError("counterfactual group members do not match trajectory assignments")

    for projection_id, projection in projections.items():
        trajectory_record = trajectories.get(projection.lineage.trajectory_id)
        if trajectory_record is None:
            raise ValueError("projection references a nonexistent source trajectory")
        trajectory = trajectory_record.trajectory
        if projection.lineage.scenario_id != trajectory.scenario_id:
            raise ValueError("projection scenario does not match its source trajectory")
        if projection.lineage.seed != trajectory.scenario.seed:
            raise ValueError("projection seed does not match its source trajectory")
        if projection.lineage.source_trajectory_sha256 != canonical_sha256(
            trajectory.model_dump(mode="json", round_trip=True)
        ):
            raise ValueError("projection source trajectory checksum is stale")
        if projection.lineage.provenance_sha256 != trajectory.provenance.stable_hash():
            raise ValueError("projection provenance checksum is stale")
        if trajectory.provenance.generator_commit != bundle.generator_commit:
            raise ValueError("trajectory generator commit differs from the bundle")

        entry = entries[projection_id]
        expected_entry_fields = {
            "projection_checksum_sha256": projection.checksum_sha256,
            "trajectory_id": trajectory.trajectory_id,
            "scenario_id": trajectory.scenario_id,
            "source_trajectory_sha256": projection.lineage.source_trajectory_sha256,
            "structured_fingerprint_sha256": projection.lineage.structured_fingerprint_sha256,
            "seed": trajectory.scenario.seed,
            "plant_variant_id": trajectory.scenario.plant_variant_id,
            "driver": trajectory.scenario.driver,
            "fault_family_ids": tuple(
                injection.fault_family for injection in trajectory.scenario.fault_injections
            ),
            "task_name": projection.task_target.task_name,
            "projection_view": projection.projection_view,
            "cut_tick": projection.model_input.cut_tick,
            "source_event_index_exclusive": projection.model_input.source_event_index_exclusive,
            "split_name": trajectory_record.split_name,
            "template_family_ids": (trajectory_record.template_family_id,),
            "alias_family_ids": (trajectory_record.alias_family_id,),
            "corruption_plan": _projection_corruption_plan(trajectory_record, projection),
        }
        if any(getattr(entry, field) != value for field, value in expected_entry_fields.items()):
            raise ValueError("split manifest entry is stale relative to its projection source")
        assignment = trajectory_record.group_assignment
        expected_group_fields = (
            assignment.counterfactual_group_id if assignment else None,
            assignment.family if assignment else None,
            assignment.variant_id if assignment else None,
            assignment.expected_variants if assignment else (),
            assignment.expanded_siblings_supported if assignment else None,
        )
        actual_group_fields = (
            entry.counterfactual_group_id,
            entry.counterfactual_family,
            entry.counterfactual_variant_id,
            entry.counterfactual_expected_variants,
            entry.expanded_siblings_supported,
        )
        if actual_group_fields != expected_group_fields:
            raise ValueError("split manifest group metadata differs from the source trajectory")

    for pair in bundle.counterfactual_projections:
        baseline = projections.get(pair.lineage.baseline_projection_id)
        counterfactual = projections.get(pair.lineage.counterfactual_projection_id)
        group = groups.get(pair.lineage.counterfactual_group_id)
        if baseline is None or counterfactual is None:
            raise ValueError("counterfactual pair references a nonexistent source projection")
        if group is None or not group.is_complete:
            raise ValueError("counterfactual pair references an absent or incomplete group")
        if baseline.task_target.task_name is not TaskName.INCIDENT_SUMMARY or (
            counterfactual.task_target.task_name is not TaskName.INCIDENT_SUMMARY
        ):
            raise ValueError("counterfactual pairs must use final incident-summary projections")
        if pair.model_input.baseline != baseline.model_input or (
            pair.model_input.counterfactual != counterfactual.model_input
        ):
            raise ValueError("counterfactual pair inputs differ from their source projections")
        baseline_assignment = trajectories[baseline.lineage.trajectory_id].group_assignment
        counterfactual_assignment = trajectories[
            counterfactual.lineage.trajectory_id
        ].group_assignment
        if (
            baseline_assignment is None
            or counterfactual_assignment is None
            or baseline_assignment.counterfactual_group_id != group.counterfactual_group_id
            or counterfactual_assignment.counterfactual_group_id != group.counterfactual_group_id
        ):
            raise ValueError("counterfactual pair members are outside their declared group")
        pair_identity = canonical_sha256(
            {
                "group_id": group.counterfactual_group_id,
                "baseline_projection_id": baseline.projection_id,
                "counterfactual_projection_id": counterfactual.projection_id,
            }
        )
        expected_pair_id = f"counterfactual:{pair_identity[:24]}"
        if pair.pair_id != expected_pair_id:
            raise ValueError("counterfactual pair ID does not match its source projection lineage")
        expected_pair = _pair_record(
            group=group,
            baseline_record=trajectories[baseline.lineage.trajectory_id],
            counterfactual_record=trajectories[counterfactual.lineage.trajectory_id],
            baseline_projection=baseline,
            counterfactual_projection=counterfactual,
        )
        if pair != expected_pair:
            raise ValueError(
                "counterfactual pair does not match its deterministic source projection pair"
            )


def _trajectory_id(plan: PlannedScenario, *, generator_commit: str) -> str:
    identity = canonical_sha256(
        {
            "scenario": plan.scenario.model_dump(mode="json", round_trip=True),
            "generator_commit": generator_commit,
            "dataset_contract_version": DATASET_CONTRACT_VERSION,
        }
    )
    return f"trajectory:{identity[:24]}"


def _is_group_candidate(plan: PlannedScenario) -> bool:
    return plan.counterfactual_family is not None or plan.case_family.startswith("g15-sparse-")


def _trajectory_record(
    plan: PlannedScenario,
    *,
    config: DevelopmentDatasetConfig,
    generator_commit: str,
) -> DevelopmentTrajectoryRecord:
    trajectory_id = _trajectory_id(plan, generator_commit=generator_commit)
    scenario = plan.scenario
    provenance = ProvenanceRecord(
        dataset_version=config.dataset.dataset_version,
        generator_commit=generator_commit,
        renderer_version=config.dataset.renderer_version,
        seed=scenario.seed,
        trajectory_id=trajectory_id,
        scenario_id=scenario.scenario_id,
        plant_variant_id=scenario.plant_variant_id,
        fault_family_ids=tuple(injection.fault_family for injection in scenario.fault_injections),
        template_family_ids=(plan.template_family_id,),
        split_name=plan.split_name,
        task_name=TaskName.FAULT_FAMILY,
    )
    trajectory = generate_trace(scenario).to_structured_trajectory(
        trajectory_id=trajectory_id,
        provenance=provenance,
    )
    assignment = derive_group_assignment(scenario) if _is_group_candidate(plan) else None
    if plan.counterfactual_family is not None and assignment is None:
        raise ValueError("planned counterfactual scenario has no structural group assignment")
    if plan.counterfactual_variant is not None and (
        assignment is None or assignment.variant_id.value != plan.counterfactual_variant
    ):
        raise ValueError("planned counterfactual role disagrees with structural grouping")
    draft = DevelopmentTrajectoryRecord.model_construct(
        schema_version=SCHEMA_VERSION,
        dataset_contract_version=DATASET_CONTRACT_VERSION,
        split_name=plan.split_name,
        case_family=plan.case_family,
        template_family_id=plan.template_family_id,
        alias_family_id=plan.alias_family_id,
        corruption_plan=plan.corruption_plan,
        trajectory=trajectory,
        group_assignment=assignment,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return DevelopmentTrajectoryRecord(
        split_name=plan.split_name,
        case_family=plan.case_family,
        template_family_id=plan.template_family_id,
        alias_family_id=plan.alias_family_id,
        corruption_plan=plan.corruption_plan,
        trajectory=trajectory,
        group_assignment=assignment,
        checksum_sha256=checksum,
    )


def _continuation_index(trajectory: StructuredTrajectory) -> int | None:
    eligible = tuple(
        event.event_index
        for event in trajectory.events
        if event.event_index > 0 and event.event_type is not EventType.ACTION_APPLIED
    )
    return eligible[-1] if eligible else None


def _projections(record: DevelopmentTrajectoryRecord) -> tuple[ProjectionRecord, ...]:
    trajectory = record.trajectory
    view = infer_projection_view(trajectory)
    assignment = record.group_assignment
    decisions = trajectory.targets.decisions
    tasks = _DECISION_TASKS
    # Shared unresolved valve prefixes are useful audit evidence but would create three
    # exact duplicate task records.  The development candidate retains the final,
    # decisive decision for every matched role instead.
    if assignment is not None and assignment.family is CounterfactualFamily.G08_G09_VALVE:
        decisions = (decisions[-1],)
    # The controlled-noise split applies one declared narrative mutation to the final
    # decision prefix. Earlier decision prefixes remain generator audit evidence rather
    # than multiplying near-identical corrupted task records.
    if record.split_name is SplitName.NOISE_TEST:
        decisions = (decisions[-1],)
    # G07 availability is intentionally hidden from fault-family prompts.  Both roles
    # therefore have the same visible prefix and the same PUMP_TRIP target; omit that
    # duplicated task symmetrically while retaining all context-sensitive tasks.
    if assignment is not None and assignment.family is CounterfactualFamily.G07_STANDBY:
        tasks = tuple(task for task in tasks if task is not TaskName.FAULT_FAMILY)
    projections = [
        project_trajectory(
            trajectory,
            decision_tick=decision.decision_tick,
            task_name=task_name,
            view=view,
        )
        for decision in decisions
        for task_name in tasks
    ]
    continuation_index = _continuation_index(trajectory)
    # A G15 continuation prefix would contain only the identical stable note and would
    # duplicate across seeds and strict splits.  Its decision tasks retain the sparse
    # tick-2 observation that defines the abstention case.
    if (
        assignment is not None
        and assignment.family is CounterfactualFamily.G15_EVIDENCE_SUFFICIENCY
    ):
        continuation_index = None
    # All six current component-holdout continuations end in the same
    # OPERATING_MODE_CHANGED target. Keeping them would make the held-out alias family a
    # task-specific target shortcut. The split still exercises every decision task; a
    # continuation component holdout remains deferred until the scenario catalog can
    # support at least two balanced continuation outcomes.
    if record.split_name is SplitName.COMPONENT_TEST:
        continuation_index = None
    if continuation_index is not None:
        projections.append(
            project_continuation(
                trajectory,
                next_event_index=continuation_index,
                view=view,
            )
        )
    return tuple(projections)


def _pair_record(
    *,
    group: CounterfactualGroup,
    baseline_record: DevelopmentTrajectoryRecord,
    counterfactual_record: DevelopmentTrajectoryRecord,
    baseline_projection: ProjectionRecord,
    counterfactual_projection: ProjectionRecord,
) -> CounterfactualProjectionRecord:
    baseline_decision = baseline_record.trajectory.targets.decisions[-1]
    counterfactual_decision = counterfactual_record.trajectory.targets.decisions[-1]
    pair = project_counterfactual_pair(
        baseline_record.trajectory,
        counterfactual_record.trajectory,
        baseline_decision_tick=baseline_decision.decision_tick,
        counterfactual_decision_tick=counterfactual_decision.decision_tick,
        group=group,
        baseline_view=infer_projection_view(baseline_record.trajectory),
        counterfactual_view=infer_projection_view(counterfactual_record.trajectory),
    )
    if pair.lineage.baseline_projection_id != baseline_projection.projection_id:
        raise ValueError("counterfactual helper baseline disagrees with the projection inventory")
    if pair.lineage.counterfactual_projection_id != counterfactual_projection.projection_id:
        raise ValueError("counterfactual helper comparison disagrees with the projection inventory")
    return pair


def _counterfactual_pairs(
    records: tuple[DevelopmentTrajectoryRecord, ...],
    groups: tuple[CounterfactualGroup, ...],
    projections: tuple[ProjectionRecord, ...],
) -> tuple[CounterfactualProjectionRecord, ...]:
    records_by_scenario = {record.trajectory.scenario_id: record for record in records}
    summary_by_trajectory_and_tick = {
        (projection.lineage.trajectory_id, projection.lineage.decision_tick): projection
        for projection in projections
        if projection.task_target.task_name is TaskName.INCIDENT_SUMMARY
    }
    pairs: list[CounterfactualProjectionRecord] = []
    for group in groups:
        if not group.is_complete:
            continue
        require_complete_group(group)
        members = {member.variant_id: member for member in group.members}
        for baseline_role, counterfactual_role in _PAIR_ROLES[group.family]:
            baseline_record = records_by_scenario[members[baseline_role].scenario_id]
            counterfactual_record = records_by_scenario[members[counterfactual_role].scenario_id]
            baseline_tick = baseline_record.trajectory.targets.decisions[-1].decision_tick
            counterfactual_tick = counterfactual_record.trajectory.targets.decisions[
                -1
            ].decision_tick
            baseline_projection = summary_by_trajectory_and_tick[
                (baseline_record.trajectory.trajectory_id, baseline_tick)
            ]
            counterfactual_projection = summary_by_trajectory_and_tick[
                (counterfactual_record.trajectory.trajectory_id, counterfactual_tick)
            ]
            pairs.append(
                _pair_record(
                    group=group,
                    baseline_record=baseline_record,
                    counterfactual_record=counterfactual_record,
                    baseline_projection=baseline_projection,
                    counterfactual_projection=counterfactual_projection,
                )
            )
    return tuple(sorted(pairs, key=lambda pair: pair.pair_id))


def _summary(
    records: tuple[DevelopmentTrajectoryRecord, ...],
    groups: tuple[CounterfactualGroup, ...],
    projections: tuple[ProjectionRecord, ...],
    pairs: tuple[CounterfactualProjectionRecord, ...],
) -> DevelopmentProjectionSummary:
    split_counts = Counter(record.split_name for record in records)
    task_counts = Counter(projection.task_target.task_name for projection in projections)
    return DevelopmentProjectionSummary(
        trajectory_count=len(records),
        projection_count=len(projections),
        counterfactual_pair_count=len(pairs),
        complete_group_count=sum(group.is_complete for group in groups),
        incomplete_g15_group_count=sum(
            not group.is_complete and group.family is CounterfactualFamily.G15_EVIDENCE_SUFFICIENCY
            for group in groups
        ),
        single_input_structured_duplicate_count=_single_input_structured_duplicate_count(
            projections
        ),
        counterfactual_input_structured_duplicate_count=(
            _counterfactual_input_structured_duplicate_count(pairs)
        ),
        trajectory_counts_by_split=tuple(
            (split, split_counts[split]) for split in SplitName if split_counts[split]
        ),
        projection_counts_by_task=tuple(
            (task, task_counts[task]) for task in TaskName if task_counts[task]
        ),
    )


def _duplicate_member_count(values: Iterable[object]) -> int:
    """Count repeated members beyond the first occurrence of each value."""

    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)


def _single_input_structured_duplicate_count(
    projections: tuple[ProjectionRecord, ...],
) -> int:
    return _duplicate_member_count(
        (
            projection.model_input.structured_fingerprint(),
            projection.task_target.task_name,
        )
        for projection in projections
    )


def _counterfactual_input_structured_duplicate_count(
    projections: tuple[CounterfactualProjectionRecord, ...],
) -> int:
    return _duplicate_member_count(
        canonical_sha256(projection.model_input.model_dump(mode="json", round_trip=True))
        for projection in projections
    )


def _group_source_plans(
    plans: Iterable[PlannedScenario],
) -> tuple[PlannedScenario, ...]:
    return tuple(plan for plan in plans if _is_group_candidate(plan))


def _projection_corruption_plan(
    record: DevelopmentTrajectoryRecord, projection: ProjectionRecord
) -> str:
    if record.split_name is not SplitName.NOISE_TEST:
        return "none"
    if projection.task_target.task_name is TaskName.CONTINUE_LOG:
        # Every non-stable continuation contains an unprotected benign note. Alternating
        # omission with duplication across seeds prevents one corruption cue from
        # identifying a single structured target.
        return "omit_noncritical" if record.trajectory.scenario.seed % 2 == 0 else "duplicate_line"
    even_seed_plan, odd_seed_plan = {
        "g01-stable-a": ("duplicate_line", "benign_insert"),
        "g02-load": ("benign_insert", "safe_reorder"),
        "g03-drift": ("safe_reorder", "duplicate_line"),
        "g05-noise": ("duplicate_line", "safe_reorder"),
        "g10-transfer-b": ("benign_insert", "duplicate_line"),
    }[record.case_family]
    return even_seed_plan if record.trajectory.scenario.seed % 2 == 0 else odd_seed_plan


def build_development_projection_bundle(
    config: DevelopmentDatasetConfig,
    *,
    generator_commit: str,
) -> DevelopmentProjectionBundle:
    """Build and validate the full split-first, pre-render Phase 3 graph."""

    if type(config) is not DevelopmentDatasetConfig:
        raise TypeError("config must be a DevelopmentDatasetConfig")
    # Let the strict provenance contract validate exact commit shape before any broad work.
    if type(generator_commit) is not str:
        raise TypeError("generator_commit must be a string")
    plans = build_development_scenario_plan(config)
    records = tuple(
        sorted(
            (
                _trajectory_record(
                    plan,
                    config=config,
                    generator_commit=generator_commit,
                )
                for plan in plans
            ),
            key=lambda item: item.trajectory.trajectory_id,
        )
    )
    groups = group_scenarios(plan.scenario for plan in _group_source_plans(plans))
    assignments = {
        record.trajectory.scenario_id: record.group_assignment
        for record in records
        if record.group_assignment is not None
    }
    projections = tuple(
        sorted(
            (projection for record in records for projection in _projections(record)),
            key=lambda projection: projection.projection_id,
        )
    )
    records_by_trajectory = {record.trajectory.trajectory_id: record for record in records}
    split_styles = {
        split: (
            getattr(config.splits, split.value).template_families,
            getattr(config.splits, split.value).alias_families,
        )
        for split in SplitName
    }
    template_alias_plan = make_template_alias_plan(split_styles)
    manifest_entries = tuple(
        make_split_manifest_entry(
            projection,
            record.trajectory,
            split_name=record.split_name,
            template_family_ids=(record.template_family_id,),
            alias_family_ids=(record.alias_family_id,),
            group_assignment=assignments.get(record.trajectory.scenario_id),
            corruption_plan=_projection_corruption_plan(record, projection),
        )
        for projection in projections
        for record in (records_by_trajectory[projection.lineage.trajectory_id],)
    )
    manifest = build_split_manifest(
        manifest_entries,
        template_alias_plan=template_alias_plan,
        golden_reserved_seed_max=config.dataset.golden_reserved_seed_max,
    )
    pairs = _counterfactual_pairs(records, groups, projections)
    task_record_count = len(projections) + len(pairs)
    if task_record_count > config.dataset.maximum_task_records:
        raise ValueError("development task count exceeds the reviewed configuration bound")
    summary = _summary(records, groups, projections, pairs)
    config_hash = dataset_config_sha256(config)
    draft = DevelopmentProjectionBundle.model_construct(
        schema_version=SCHEMA_VERSION,
        dataset_contract_version=DATASET_CONTRACT_VERSION,
        artifact_status="structured_projection_audit",
        dataset_config_sha256=config_hash,
        generator_commit=generator_commit,
        trajectories=records,
        groups=groups,
        projections=projections,
        counterfactual_projections=pairs,
        split_manifest=manifest,
        summary=summary,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return DevelopmentProjectionBundle(
        dataset_config_sha256=config_hash,
        generator_commit=generator_commit,
        trajectories=records,
        groups=groups,
        projections=projections,
        counterfactual_projections=pairs,
        split_manifest=manifest,
        summary=summary,
        checksum_sha256=checksum,
    )


__all__ = [
    "DevelopmentProjectionBundle",
    "DevelopmentProjectionSummary",
    "DevelopmentTrajectoryRecord",
    "build_development_projection_bundle",
]
