"""Review-gated rendering and task assembly for the Phase 3 development candidate."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from importlib.resources import as_file
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from reactorbench.dataset.artifacts import (
    ArtifactModelSpec,
    ArtifactVerificationError,
    ArtifactWriter,
    CandidateArtifactManifest,
    CandidateArtifactMetadata,
)
from reactorbench.dataset.catalog import AliasFamily, TemplateFamily
from reactorbench.dataset.config import DevelopmentDatasetConfig, dataset_config_sha256
from reactorbench.dataset.contracts import (
    DATASET_CONTRACT_VERSION,
    CounterfactualProjectionRecord,
    DependencyLinkContextFact,
    ProjectionRecord,
    ProjectionTaskTarget,
    PromptEvidenceTarget,
    StandbyRelationshipContextFact,
)
from reactorbench.dataset.corruption import (
    CorruptedCandidate,
    CorruptionPlan,
    apply_narrative_corruption,
    materialize_corrupted_candidate,
)
from reactorbench.dataset.pipeline import (
    DevelopmentProjectionBundle,
    DevelopmentProjectionSummary,
    DevelopmentTrajectoryRecord,
)
from reactorbench.dataset.quality import (
    QualityRecord,
    QualityReport,
    TaskShortcutRecord,
    audit_quality,
)
from reactorbench.dataset.renderer import RenderedCandidate, render_model_input
from reactorbench.dataset.review import (
    CatalogReviewPacket,
    HumanReviewRecord,
    PostrenderReviewPacket,
    prepare_postrender_review_packet,
    verify_catalog_review_gate,
)
from reactorbench.dataset.schema_export import load_dataset_snapshot
from reactorbench.dataset.splits import SplitManifest, SplitManifestEntry
from reactorbench.resources import (
    canonical_dataset_schema_snapshot_resource,
    canonical_schema_snapshot_resource,
)
from reactorbench.schemas.base import (
    SCHEMA_VERSION,
    ContractId,
    ContractModel,
    SchemaVersion,
    canonical_json_bytes,
    canonical_sha256,
    require_unique,
)
from reactorbench.schemas.enums import SplitName, TaskName
from reactorbench.schemas.export import load_snapshot
from reactorbench.schemas.provenance import ProvenanceRecord

from .grouping import CounterfactualGroup


def _schema_snapshot_hashes() -> tuple[str, str]:
    """Validate and return the exact packaged Aster and dataset snapshot hashes."""

    with as_file(canonical_schema_snapshot_resource()) as aster_directory:
        _aster_documents, aster_manifest = load_snapshot(aster_directory)
    with as_file(canonical_dataset_schema_snapshot_resource()) as dataset_directory:
        _dataset_documents, dataset_manifest, _dataset_contract = load_dataset_snapshot(
            dataset_directory
        )
    aster_hash = aster_manifest.get("snapshot_hash")
    dataset_hash = dataset_manifest.get("snapshot_hash")
    if not isinstance(aster_hash, str) or not isinstance(dataset_hash, str):
        raise ValueError("schema snapshot manifests do not expose valid snapshot hashes")
    return aster_hash, dataset_hash


class TaskExampleCandidate(ContractModel):
    """One pending-review supervised task record referencing rendered prompt text."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    dataset_contract_version: Literal["0.1.0"] = DATASET_CONTRACT_VERSION
    candidate_status: Literal["candidate_pending_postrender_review"] = (
        "candidate_pending_postrender_review"
    )
    example_id: ContractId
    split_name: SplitName
    task_name: TaskName
    source_record_ids: tuple[ContractId, ...]
    prompt_render_ids: tuple[ContractId, ...]
    corruption_ids: tuple[ContractId, ...] = ()
    task_target: ProjectionTaskTarget
    provenance_records: tuple[ProvenanceRecord, ...]
    counterfactual_group_id: ContractId | None = None
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator(
        "source_record_ids",
        "prompt_render_ids",
        "corruption_ids",
        mode="after",
    )
    @classmethod
    def identifiers_are_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return require_unique(values, field_name="task-example identifiers")

    @model_validator(mode="after")
    def record_is_bound_and_checksummed(self) -> TaskExampleCandidate:
        if self.task_target.task_name is not self.task_name:
            raise ValueError("task example name must match its structured target")
        is_pair = self.task_name is TaskName.COUNTERFACTUAL_COMPARE
        expected_prompt_count = 2 if is_pair else 1
        if len(self.prompt_render_ids) != expected_prompt_count:
            raise ValueError("task example has the wrong number of rendered prompts")
        if len(self.provenance_records) != expected_prompt_count:
            raise ValueError("task example has the wrong number of provenance records")
        if len(self.source_record_ids) != 1:
            raise ValueError("task example must reference one projection or pair record")
        if is_pair != (self.counterfactual_group_id is not None):
            raise ValueError("only counterfactual tasks may declare a group ID")
        if self.corruption_ids and len(self.corruption_ids) != expected_prompt_count:
            raise ValueError("corruption IDs must align with every rendered prompt or be absent")
        if any(record.task_name is not self.task_name for record in self.provenance_records):
            raise ValueError("task provenance must carry the task example name")
        if any(record.split_name is not self.split_name for record in self.provenance_records):
            raise ValueError("task provenance must carry the task example split")
        identity = canonical_sha256(
            self.model_dump(
                mode="json",
                round_trip=True,
                exclude={"example_id", "checksum_sha256"},
            )
        )
        if self.example_id != f"example:{identity[:24]}":
            raise ValueError("task example ID does not match its content")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("task example checksum mismatch")
        return self


class DevelopmentCandidateBundle(ContractModel):
    """Complete local candidate, still blocked on post-render human approval."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    dataset_contract_version: Literal["0.1.0"] = DATASET_CONTRACT_VERSION
    artifact_status: Literal["candidate_pending_postrender_review"] = (
        "candidate_pending_postrender_review"
    )
    resolved_config: DevelopmentDatasetConfig
    aster_schema_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_schema_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    guard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    structured_bundle: DevelopmentProjectionBundle
    pre_render_review_packet: CatalogReviewPacket
    pre_render_review_record: HumanReviewRecord
    rendered_candidates: tuple[RenderedCandidate, ...]
    corruption_records: tuple[CorruptedCandidate, ...]
    task_examples: tuple[TaskExampleCandidate, ...]
    task_shortcut_records: tuple[TaskShortcutRecord, ...]
    quality_report: QualityReport
    postrender_review_packet: PostrenderReviewPacket
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def inventories_and_checksums_are_exact(self) -> DevelopmentCandidateBundle:
        config_sha256 = dataset_config_sha256(self.resolved_config)
        if self.structured_bundle.dataset_config_sha256 != config_sha256:
            raise ValueError("structured bundle is bound to a different resolved config")
        if (
            self.resolved_config.dataset.renderer_version
            != self.pre_render_review_packet.renderer_version
        ):
            raise ValueError("resolved config and review packet renderer versions differ")
        if self.resolved_config.review.reviewer_role != self.pre_render_review_record.reviewer_role:
            raise ValueError("resolved config and review record roles differ")
        if self.catalog_sha256 != self.pre_render_review_packet.catalog_sha256:
            raise ValueError("candidate catalog checksum differs from the reviewed catalog")
        if self.guard_sha256 != self.pre_render_review_packet.guard_sha256:
            raise ValueError("candidate guard checksum differs from the reviewed guard")
        if self.postrender_review_packet.catalog_sha256 != self.catalog_sha256:
            raise ValueError("post-render packet catalog checksum differs from the candidate")
        if self.postrender_review_packet.guard_sha256 != self.guard_sha256:
            raise ValueError("post-render packet guard checksum differs from the candidate")
        current_aster, current_dataset = _schema_snapshot_hashes()
        if self.aster_schema_snapshot_sha256 != current_aster:
            raise ValueError("candidate Aster schema snapshot checksum is stale")
        if self.dataset_schema_snapshot_sha256 != current_dataset:
            raise ValueError("candidate dataset schema snapshot checksum is stale")
        verify_catalog_review_gate(
            self.pre_render_review_packet,
            self.pre_render_review_record,
            structured_bundle=self.structured_bundle,
        )
        render_ids = tuple(candidate.render_id for candidate in self.rendered_candidates)
        example_ids = tuple(example.example_id for example in self.task_examples)
        corruption_ids = tuple(item.corruption_id for item in self.corruption_records)
        require_unique(render_ids, field_name="rendered candidate IDs")
        require_unique(example_ids, field_name="task example IDs")
        require_unique(corruption_ids, field_name="corruption IDs")
        if render_ids != tuple(sorted(render_ids)):
            raise ValueError("rendered candidates must use canonical ID order")
        if example_ids != tuple(sorted(example_ids)):
            raise ValueError("task examples must use canonical ID order")
        shortcut_ids = tuple(record.record_id for record in self.task_shortcut_records)
        if shortcut_ids != example_ids:
            raise ValueError("task shortcut records must bind every task example exactly")
        if corruption_ids != tuple(sorted(corruption_ids)):
            raise ValueError("corruption records must use canonical ID order")
        available_renders = set(render_ids)
        if any(
            not set(example.prompt_render_ids).issubset(available_renders)
            for example in self.task_examples
        ):
            raise ValueError("task examples must reference rendered candidates in this bundle")
        if self.quality_report.record_count != len(self.rendered_candidates):
            raise ValueError("quality report must cover every rendered candidate")
        audited_tasks = tuple(
            (record.record_id, record.record_sha256)
            for record in self.quality_report.audited_task_records
        )
        expected_audited_tasks = tuple(
            (
                record.record_id,
                canonical_sha256(record.model_dump(mode="json", round_trip=True)),
            )
            for record in self.task_shortcut_records
        )
        if (
            self.quality_report.task_record_count != len(self.task_examples)
            or audited_tasks != expected_audited_tasks
        ):
            raise ValueError("quality report must bind every task shortcut record exactly")
        if not self.quality_report.passed:
            raise ValueError("development candidate requires a passing quality report")
        if self.postrender_review_packet.candidate_count != len(self.rendered_candidates):
            raise ValueError("post-render review must cover every rendered candidate")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("development candidate bundle checksum mismatch")
        return self


def _validate_rendered_relationships(candidate: DevelopmentCandidateBundle) -> None:
    """Rebuild the deterministic post-render graph to validate every serialized join.

    Render IDs and record-local hashes are necessary but insufficient: they do not
    prove that a prompt, corruption, task example, or audit record came from the
    structured projection graph in this artifact.  Re-deriving the small candidate
    graph makes those bindings explicit and rejects cross-file substitution.
    """

    (
        expected_candidates,
        expected_corruptions,
        projection_to_render,
        projection_to_corruption,
        quality_records,
    ) = _render_candidates(candidate.structured_bundle)
    if candidate.rendered_candidates != expected_candidates:
        raise ValueError("rendered candidates do not match the structured projection graph")
    if candidate.corruption_records != expected_corruptions:
        raise ValueError("corruption records do not match the rendered projection graph")
    expected_examples = _task_examples(
        candidate.structured_bundle,
        projection_to_render=projection_to_render,
        projection_to_corruption=projection_to_corruption,
    )
    if candidate.task_examples != expected_examples:
        raise ValueError("task examples do not match projection, render, and provenance links")
    expected_shortcuts = _task_shortcut_records(expected_examples, quality_records)
    if candidate.task_shortcut_records != expected_shortcuts:
        raise ValueError("task shortcut records do not match the exact task examples")
    expected_quality = audit_quality(
        quality_records,
        task_records=expected_shortcuts,
        n_values=candidate.resolved_config.quality.ngram_sizes,
    )
    if candidate.quality_report != expected_quality:
        raise ValueError("quality report does not audit the exact rendered and task records")
    expected_postrender = prepare_postrender_review_packet(
        expected_candidates,
        quality_report=expected_quality,
    )
    if candidate.postrender_review_packet != expected_postrender:
        raise ValueError("post-render review packet does not match exact rendered candidates")


def _provenance_for(
    record: DevelopmentTrajectoryRecord,
    entry: SplitManifestEntry,
    *,
    task_name: TaskName,
) -> ProvenanceRecord:
    source = record.trajectory.provenance
    return ProvenanceRecord(
        dataset_version=source.dataset_version,
        generator_commit=source.generator_commit,
        renderer_version=source.renderer_version,
        seed=source.seed,
        trajectory_id=source.trajectory_id,
        scenario_id=source.scenario_id,
        plant_variant_id=source.plant_variant_id,
        fault_family_ids=source.fault_family_ids,
        template_family_ids=entry.template_family_ids,
        split_name=entry.split_name,
        task_name=task_name,
    )


def _target_labels(target: ProjectionTaskTarget) -> tuple[str, ...]:
    """Collect output labels that must not leak verbatim into a rendered prompt."""

    labels: set[str] = set()
    leak_sensitive_fields = {
        "diagnosis_status",
        "fault_labels",
        "immediate_action",
        "abstention_reason",
        "next_event_type",
    }

    def visit(value: object, field_name: str | None = None) -> None:
        if isinstance(value, Enum):
            # Evidence extraction deliberately asks the model to identify facts present
            # in the prompt; evidence-slot phrases are therefore not target leakage.
            if field_name in leak_sensitive_fields:
                labels.add(str(value.value))
            return
        if isinstance(value, BaseModel):
            for name in type(value).model_fields:
                visit(getattr(value, name), name)
            return
        if isinstance(value, tuple):
            for item in value:
                visit(item, field_name)

    visit(target.target)
    return tuple(sorted(labels))


def _shortcut_target_labels(target: ProjectionTaskTarget) -> tuple[str, ...]:
    """Collect every categorical structured answer label for contingency audits."""

    labels: set[str] = set()

    def visit(value: object, path: str) -> None:
        if isinstance(value, Enum):
            labels.add(f"{path}={value.value}")
            return
        if isinstance(value, BaseModel):
            for name in type(value).model_fields:
                if name != "task_name":
                    visit(getattr(value, name), f"{path}.{name}" if path else name)
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}")
            return
        if isinstance(value, dict):
            for name, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                visit(item, f"{path}.{name}" if path else str(name))

    # The wrapper's TaskName is already the audit stratum. Only categorical values in
    # the answer itself belong in the within-task target contingency.
    visit(target.target, "")
    return tuple(sorted(labels))


def _candidate_context_flags(
    projection: ProjectionRecord,
    *,
    corruption_plan: str,
) -> tuple[str, ...]:
    flags: list[str] = []
    standby_present = False
    dependency_map_present = False
    for fact in projection.model_input.context_facts:
        if isinstance(fact, StandbyRelationshipContextFact):
            standby_present = True
            flags.extend(
                (
                    f"semantic:standby-state:{fact.standby_state.value.casefold()}",
                    f"semantic:support-state:{fact.support_state.value.casefold()}",
                    "semantic:context-note:standby-v1",
                )
            )
        elif isinstance(fact, DependencyLinkContextFact):
            dependency_map_present = True
    flags.extend(
        (
            f"semantic:standby-context:{'present' if standby_present else 'absent'}",
            f"semantic:dependency-map-context:{'present' if dependency_map_present else 'absent'}",
        )
    )
    if dependency_map_present:
        flags.append("semantic:context-note:dependency-v1")
    flags.append(f"corruption:{corruption_plan}")
    return tuple(sorted(set(flags)))


def _candidate_groups(
    bundle: DevelopmentProjectionBundle,
) -> tuple[tuple[tuple[str, str, str, SplitName, str], tuple[ProjectionRecord, ...]], ...]:
    entries = {entry.projection_id: entry for entry in bundle.split_manifest.entries}
    grouped: dict[tuple[str, str, str, SplitName, str], list[ProjectionRecord]] = defaultdict(list)
    for projection in bundle.projections:
        entry = entries[projection.projection_id]
        key = (
            projection.model_input.structured_fingerprint(),
            entry.template_family_ids[0],
            entry.alias_family_ids[0],
            entry.split_name,
            entry.corruption_plan,
        )
        grouped[key].append(projection)
    return tuple(
        (key, tuple(sorted(values, key=lambda item: item.projection_id)))
        for key, values in sorted(grouped.items(), key=lambda item: str(item[0]))
    )


def _render_candidates(
    bundle: DevelopmentProjectionBundle,
) -> tuple[
    tuple[RenderedCandidate, ...],
    tuple[CorruptedCandidate, ...],
    dict[str, str],
    dict[str, str],
    tuple[QualityRecord, ...],
]:
    entries = {entry.projection_id: entry for entry in bundle.split_manifest.entries}
    records = {record.trajectory.trajectory_id: record for record in bundle.trajectories}
    candidates: list[RenderedCandidate] = []
    corruptions: list[CorruptedCandidate] = []
    projection_to_render: dict[str, str] = {}
    projection_to_corruption: dict[str, str] = {}
    quality_records: list[QualityRecord] = []
    for (_, template_id, alias_id, split_name, corruption_plan), projections in _candidate_groups(
        bundle
    ):
        representative = projections[0]
        candidate = render_model_input(
            representative.model_input,
            template_family=TemplateFamily(template_id),
            alias_family=AliasFamily(alias_id),
            split_name=split_name,
        )
        corruption: CorruptedCandidate | None = None
        if corruption_plan != "none":
            protected_refs = tuple(
                dict.fromkeys(
                    ref
                    for projection in projections
                    if isinstance(projection.task_target.target, PromptEvidenceTarget)
                    for ref in projection.task_target.target.fact_refs
                )
            )
            corruption = apply_narrative_corruption(
                candidate,
                representative.model_input,
                plan=CorruptionPlan(corruption_plan),
                protected_fact_refs=protected_refs,
            )
            candidate = materialize_corrupted_candidate(candidate, corruption)
            corruptions.append(corruption)
        candidates.append(candidate)
        for projection in projections:
            projection_to_render[projection.projection_id] = candidate.render_id
            if corruption is not None:
                projection_to_corruption[projection.projection_id] = corruption.corruption_id

        labels = tuple(
            sorted(
                {
                    label
                    for projection in projections
                    for label in _target_labels(projection.task_target)
                }
            )
        )
        representative_entry = entries[representative.projection_id]
        source = records[representative.lineage.trajectory_id]
        quality_records.append(
            QualityRecord(
                example_id=candidate.render_id,
                split_name=split_name,
                text=candidate.text,
                template_family_id=template_id,
                alias_family_id=alias_id,
                target_labels=labels,
                context_flags=_candidate_context_flags(
                    representative,
                    corruption_plan=corruption_plan,
                ),
                provenance={
                    "dataset_version": source.trajectory.provenance.dataset_version,
                    "generator_commit": source.trajectory.provenance.generator_commit,
                    "scenario_schema_version": SCHEMA_VERSION,
                    "renderer_version": source.trajectory.provenance.renderer_version,
                    "seed": representative_entry.seed,
                    "scenario_id": representative_entry.scenario_id,
                    "plant_variant_id": representative_entry.plant_variant_id.value,
                    "fault_family_ids": tuple(
                        fault.value for fault in representative_entry.fault_family_ids
                    ),
                    "template_family_ids": representative_entry.template_family_ids,
                    "split_name": split_name.value,
                    "task_name": tuple(
                        sorted(
                            {projection.task_target.task_name.value for projection in projections}
                        )
                    ),
                    "projection_ids": tuple(projection.projection_id for projection in projections),
                },
            )
        )
    return (
        tuple(sorted(candidates, key=lambda item: item.render_id)),
        tuple(sorted(corruptions, key=lambda item: item.corruption_id)),
        projection_to_render,
        projection_to_corruption,
        tuple(quality_records),
    )


def _example(
    *,
    split_name: SplitName,
    source_record_ids: tuple[str, ...],
    prompt_render_ids: tuple[str, ...],
    corruption_ids: tuple[str, ...],
    task_target: ProjectionTaskTarget,
    provenance_records: tuple[ProvenanceRecord, ...],
    counterfactual_group_id: str | None = None,
) -> TaskExampleCandidate:
    identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "candidate_status": "candidate_pending_postrender_review",
        "split_name": split_name,
        "task_name": task_target.task_name,
        "source_record_ids": source_record_ids,
        "prompt_render_ids": prompt_render_ids,
        "corruption_ids": corruption_ids,
        "task_target": task_target.model_dump(mode="json", round_trip=True),
        "provenance_records": tuple(
            record.model_dump(mode="json", round_trip=True) for record in provenance_records
        ),
        "counterfactual_group_id": counterfactual_group_id,
    }
    identity = canonical_sha256(identity_payload)
    example_id = f"example:{identity[:24]}"
    draft = TaskExampleCandidate.model_construct(
        schema_version=SCHEMA_VERSION,
        dataset_contract_version=DATASET_CONTRACT_VERSION,
        candidate_status="candidate_pending_postrender_review",
        example_id=example_id,
        split_name=split_name,
        task_name=task_target.task_name,
        source_record_ids=source_record_ids,
        prompt_render_ids=prompt_render_ids,
        corruption_ids=corruption_ids,
        task_target=task_target,
        provenance_records=provenance_records,
        counterfactual_group_id=counterfactual_group_id,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return TaskExampleCandidate(
        example_id=example_id,
        split_name=split_name,
        task_name=task_target.task_name,
        source_record_ids=source_record_ids,
        prompt_render_ids=prompt_render_ids,
        corruption_ids=corruption_ids,
        task_target=task_target,
        provenance_records=provenance_records,
        counterfactual_group_id=counterfactual_group_id,
        checksum_sha256=checksum,
    )


def _task_examples(
    bundle: DevelopmentProjectionBundle,
    *,
    projection_to_render: dict[str, str],
    projection_to_corruption: dict[str, str],
) -> tuple[TaskExampleCandidate, ...]:
    entries = {entry.projection_id: entry for entry in bundle.split_manifest.entries}
    trajectories = {record.trajectory.trajectory_id: record for record in bundle.trajectories}
    examples: list[TaskExampleCandidate] = []
    for projection in bundle.projections:
        entry = entries[projection.projection_id]
        record = trajectories[projection.lineage.trajectory_id]
        corruption_id = projection_to_corruption.get(projection.projection_id)
        examples.append(
            _example(
                split_name=entry.split_name,
                source_record_ids=(projection.projection_id,),
                prompt_render_ids=(projection_to_render[projection.projection_id],),
                corruption_ids=(corruption_id,) if corruption_id is not None else (),
                task_target=projection.task_target,
                provenance_records=(
                    _provenance_for(
                        record,
                        entry,
                        task_name=projection.task_target.task_name,
                    ),
                ),
            )
        )

    projections = {projection.projection_id: projection for projection in bundle.projections}
    for pair in bundle.counterfactual_projections:
        baseline = projections[pair.lineage.baseline_projection_id]
        counterfactual = projections[pair.lineage.counterfactual_projection_id]
        baseline_entry = entries[baseline.projection_id]
        counterfactual_entry = entries[counterfactual.projection_id]
        if baseline_entry.split_name is not counterfactual_entry.split_name:
            raise ValueError("counterfactual pair crosses split boundaries")
        baseline_record = trajectories[baseline.lineage.trajectory_id]
        counterfactual_record = trajectories[counterfactual.lineage.trajectory_id]
        examples.append(
            _example(
                split_name=baseline_entry.split_name,
                source_record_ids=(pair.pair_id,),
                prompt_render_ids=(
                    projection_to_render[baseline.projection_id],
                    projection_to_render[counterfactual.projection_id],
                ),
                corruption_ids=(),
                task_target=pair.task_target,
                provenance_records=(
                    _provenance_for(
                        baseline_record,
                        baseline_entry,
                        task_name=TaskName.COUNTERFACTUAL_COMPARE,
                    ),
                    _provenance_for(
                        counterfactual_record,
                        counterfactual_entry,
                        task_name=TaskName.COUNTERFACTUAL_COMPARE,
                    ),
                ),
                counterfactual_group_id=pair.lineage.counterfactual_group_id,
            )
        )
    return tuple(sorted(examples, key=lambda item: item.example_id))


def _task_shortcut_records(
    examples: tuple[TaskExampleCandidate, ...],
    quality_records: tuple[QualityRecord, ...],
) -> tuple[TaskShortcutRecord, ...]:
    """Build one task-scoped contingency input for every supervised example."""

    quality_by_render = {record.example_id: record for record in quality_records}
    records: list[TaskShortcutRecord] = []
    for example in examples:
        prompts = tuple(quality_by_render[render_id] for render_id in example.prompt_render_ids)
        template_families = {record.template_family_id for record in prompts}
        alias_families = {record.alias_family_id for record in prompts}
        if len(template_families) != 1 or len(alias_families) != 1:
            raise ValueError("paired task prompts must share one renderer plan")
        context_flags = tuple(sorted({flag for record in prompts for flag in record.context_flags}))
        records.append(
            TaskShortcutRecord(
                record_id=example.example_id,
                prompt_render_ids=example.prompt_render_ids,
                task_name=example.task_name,
                template_family_id=next(iter(template_families)),
                alias_family_id=next(iter(alias_families)),
                target_labels=_shortcut_target_labels(example.task_target),
                context_flags=context_flags,
            )
        )
    return tuple(sorted(records, key=lambda record: record.record_id))


def _require_candidate_bounds(
    candidates: tuple[RenderedCandidate, ...],
    examples: tuple[TaskExampleCandidate, ...],
    *,
    maximum_task_records: int,
    maximum_rendered_bytes: int,
) -> None:
    if len(examples) > maximum_task_records:
        raise ValueError("rendered task count exceeds the reviewed configuration bound")
    rendered_bytes = sum(len(candidate.text.encode("utf-8")) for candidate in candidates)
    if rendered_bytes > maximum_rendered_bytes:
        raise ValueError("rendered text exceeds the reviewed configuration byte bound")


def render_development_candidate(
    structured_bundle: DevelopmentProjectionBundle,
    *,
    config: DevelopmentDatasetConfig,
    review_packet: CatalogReviewPacket,
    review_record: HumanReviewRecord,
) -> DevelopmentCandidateBundle:
    """Render the exact config-bound graph only after combined owner review verifies."""

    if type(structured_bundle) is not DevelopmentProjectionBundle:
        raise TypeError("structured_bundle must be a DevelopmentProjectionBundle")
    if type(config) is not DevelopmentDatasetConfig:
        raise TypeError("config must be an exact DevelopmentDatasetConfig")
    if structured_bundle.dataset_config_sha256 != dataset_config_sha256(config):
        raise ValueError("structured bundle is bound to a different resolved config")
    if config.dataset.renderer_version != review_packet.renderer_version:
        raise ValueError("resolved config and review packet renderer versions differ")
    if review_record.reviewer_role != config.review.reviewer_role:
        raise ValueError("review record does not use the configured reviewer role")
    verify_catalog_review_gate(
        review_packet,
        review_record,
        structured_bundle=structured_bundle,
    )
    aster_schema_sha256, dataset_schema_sha256 = _schema_snapshot_hashes()
    (
        candidates,
        corruptions,
        projection_to_render,
        projection_to_corruption,
        quality_records,
    ) = _render_candidates(structured_bundle)
    examples = _task_examples(
        structured_bundle,
        projection_to_render=projection_to_render,
        projection_to_corruption=projection_to_corruption,
    )
    _require_candidate_bounds(
        candidates,
        examples,
        maximum_task_records=config.dataset.maximum_task_records,
        maximum_rendered_bytes=config.dataset.maximum_rendered_bytes,
    )
    task_shortcut_records = _task_shortcut_records(examples, quality_records)
    quality_report = audit_quality(
        quality_records,
        task_records=task_shortcut_records,
        n_values=config.quality.ngram_sizes,
    )
    if not quality_report.passed:
        categories = []
        if quality_report.exact_duplicates:
            categories.append("exact duplicates")
        if quality_report.forbidden_skeleton_duplicates:
            categories.append("forbidden skeleton overlap")
        if quality_report.shortcut_findings:
            categories.append("task-scoped shortcut findings")
        if quality_report.target_text_findings:
            categories.append("target text leakage")
        if quality_report.provenance_issues:
            categories.append("provenance issues")
        raise ValueError(f"automated quality audit failed: {', '.join(categories)}")
    postrender = prepare_postrender_review_packet(
        candidates,
        quality_report=quality_report,
    )
    draft = DevelopmentCandidateBundle.model_construct(
        schema_version=SCHEMA_VERSION,
        dataset_contract_version=DATASET_CONTRACT_VERSION,
        artifact_status="candidate_pending_postrender_review",
        resolved_config=config,
        aster_schema_snapshot_sha256=aster_schema_sha256,
        dataset_schema_snapshot_sha256=dataset_schema_sha256,
        catalog_sha256=review_packet.catalog_sha256,
        guard_sha256=review_packet.guard_sha256,
        structured_bundle=structured_bundle,
        pre_render_review_packet=review_packet,
        pre_render_review_record=review_record,
        rendered_candidates=candidates,
        corruption_records=corruptions,
        task_examples=examples,
        task_shortcut_records=task_shortcut_records,
        quality_report=quality_report,
        postrender_review_packet=postrender,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return DevelopmentCandidateBundle(
        resolved_config=config,
        aster_schema_snapshot_sha256=aster_schema_sha256,
        dataset_schema_snapshot_sha256=dataset_schema_sha256,
        catalog_sha256=review_packet.catalog_sha256,
        guard_sha256=review_packet.guard_sha256,
        structured_bundle=structured_bundle,
        pre_render_review_packet=review_packet,
        pre_render_review_record=review_record,
        rendered_candidates=candidates,
        corruption_records=corruptions,
        task_examples=examples,
        task_shortcut_records=task_shortcut_records,
        quality_report=quality_report,
        postrender_review_packet=postrender,
        checksum_sha256=checksum,
    )


def build_review_gated_development_candidate(
    config: DevelopmentDatasetConfig,
    *,
    structured_bundle: DevelopmentProjectionBundle,
    review_packet: CatalogReviewPacket,
    review_record: HumanReviewRecord,
) -> DevelopmentCandidateBundle:
    """Build a pending candidate from one already-built, exactly reviewed graph."""

    if type(config) is not DevelopmentDatasetConfig:
        raise TypeError("config must be a DevelopmentDatasetConfig")
    if type(structured_bundle) is not DevelopmentProjectionBundle:
        raise TypeError("structured_bundle must be a DevelopmentProjectionBundle")
    if type(review_packet) is not CatalogReviewPacket:
        raise TypeError("review_packet must be a CatalogReviewPacket")
    if type(review_record) is not HumanReviewRecord:
        raise TypeError("review_record must be a HumanReviewRecord")
    return render_development_candidate(
        structured_bundle,
        config=config,
        review_packet=review_packet,
        review_record=review_record,
    )


def candidate_artifact_metadata(bundle: DevelopmentCandidateBundle) -> CandidateArtifactMetadata:
    """Return the strict outer provenance record for one full candidate."""

    if type(bundle) is not DevelopmentCandidateBundle:
        raise TypeError("bundle must be a DevelopmentCandidateBundle")
    return CandidateArtifactMetadata(
        dataset_version=bundle.resolved_config.dataset.dataset_version,
        dataset_contract_version=DATASET_CONTRACT_VERSION,
        renderer_version=bundle.resolved_config.dataset.renderer_version,
        schema_version=SCHEMA_VERSION,
        generator_commit=bundle.structured_bundle.generator_commit,
        candidate_bundle_sha256=bundle.checksum_sha256,
        structured_bundle_sha256=bundle.structured_bundle.checksum_sha256,
        resolved_config_sha256=dataset_config_sha256(bundle.resolved_config),
        split_manifest_sha256=bundle.structured_bundle.split_manifest.checksum_sha256,
        aster_schema_snapshot_sha256=bundle.aster_schema_snapshot_sha256,
        dataset_schema_snapshot_sha256=bundle.dataset_schema_snapshot_sha256,
        catalog_sha256=bundle.catalog_sha256,
        guard_sha256=bundle.guard_sha256,
        pre_render_review_packet_sha256=bundle.pre_render_review_packet.packet_sha256,
        pre_render_review_record_sha256=bundle.pre_render_review_record.review_record_sha256,
        postrender_review_packet_sha256=bundle.postrender_review_packet.packet_sha256,
        quality_report_sha256=bundle.quality_report.report_sha256,
    )


def candidate_artifact_records(
    bundle: DevelopmentCandidateBundle,
) -> dict[str, tuple[BaseModel, ...]]:
    """Return the complete flat canonical JSONL inventory for ``ArtifactWriter``."""

    if type(bundle) is not DevelopmentCandidateBundle:
        raise TypeError("bundle must be a DevelopmentCandidateBundle")
    return {
        "candidate-metadata.jsonl": (candidate_artifact_metadata(bundle),),
        "candidate-summary.jsonl": (bundle.structured_bundle.summary,),
        "corruptions.jsonl": bundle.corruption_records,
        "counterfactual-projections.jsonl": bundle.structured_bundle.counterfactual_projections,
        "groups.jsonl": bundle.structured_bundle.groups,
        "postrender-review.jsonl": (bundle.postrender_review_packet,),
        "pre-render-review-packet.jsonl": (bundle.pre_render_review_packet,),
        "pre-render-review-record.jsonl": (bundle.pre_render_review_record,),
        "projections.jsonl": bundle.structured_bundle.projections,
        "quality-report.jsonl": (bundle.quality_report,),
        "rendered-candidates.jsonl": bundle.rendered_candidates,
        "resolved-config.jsonl": (bundle.resolved_config,),
        "split-manifest.jsonl": (bundle.structured_bundle.split_manifest,),
        "task-examples.jsonl": bundle.task_examples,
        "task-shortcut-records.jsonl": bundle.task_shortcut_records,
        "trajectories.jsonl": bundle.structured_bundle.trajectories,
    }


def development_artifact_model_specs() -> dict[str, ArtifactModelSpec]:
    """Return the exact typed file contract for a full development candidate."""

    singleton = 1
    return {
        "candidate-metadata.jsonl": ArtifactModelSpec(
            CandidateArtifactMetadata, singleton, singleton
        ),
        "candidate-summary.jsonl": ArtifactModelSpec(
            DevelopmentProjectionSummary, singleton, singleton
        ),
        "corruptions.jsonl": ArtifactModelSpec(CorruptedCandidate),
        "counterfactual-projections.jsonl": ArtifactModelSpec(CounterfactualProjectionRecord),
        "groups.jsonl": ArtifactModelSpec(CounterfactualGroup),
        "postrender-review.jsonl": ArtifactModelSpec(PostrenderReviewPacket, singleton, singleton),
        "pre-render-review-packet.jsonl": ArtifactModelSpec(
            CatalogReviewPacket, singleton, singleton
        ),
        "pre-render-review-record.jsonl": ArtifactModelSpec(
            HumanReviewRecord, singleton, singleton
        ),
        "projections.jsonl": ArtifactModelSpec(ProjectionRecord, 1),
        "quality-report.jsonl": ArtifactModelSpec(QualityReport, singleton, singleton),
        "rendered-candidates.jsonl": ArtifactModelSpec(RenderedCandidate, 1),
        "resolved-config.jsonl": ArtifactModelSpec(DevelopmentDatasetConfig, singleton, singleton),
        "split-manifest.jsonl": ArtifactModelSpec(SplitManifest, singleton, singleton),
        "task-examples.jsonl": ArtifactModelSpec(TaskExampleCandidate, 1),
        "task-shortcut-records.jsonl": ArtifactModelSpec(TaskShortcutRecord, 1),
        "trajectories.jsonl": ArtifactModelSpec(DevelopmentTrajectoryRecord, 1),
    }


@dataclass(frozen=True)
class VerifiedDevelopmentCandidateArtifact:
    manifest: CandidateArtifactManifest
    metadata: CandidateArtifactMetadata
    candidate: DevelopmentCandidateBundle


def _single_record[ModelT: BaseModel](records: tuple[ModelT, ...], *, description: str) -> ModelT:
    if len(records) != 1:
        raise ArtifactVerificationError(f"{description} must contain exactly one record")
    return records[0]


def _verify_development_candidate_artifact(
    writer: ArtifactWriter,
    *,
    relative_directory: str,
) -> VerifiedDevelopmentCandidateArtifact:
    """Typed-load a full artifact and reconstruct every cross-file contract."""

    if type(writer) is not ArtifactWriter:
        raise TypeError("writer must be an exact ArtifactWriter")
    verified = writer.verify_typed_candidate_bundle(
        relative_directory=relative_directory,
        expected_files=development_artifact_model_specs(),
    )
    metadata = _single_record(
        verified.records_for("candidate-metadata.jsonl", CandidateArtifactMetadata),
        description="candidate metadata",
    )
    config = _single_record(
        verified.records_for("resolved-config.jsonl", DevelopmentDatasetConfig),
        description="resolved config",
    )
    split_manifest = _single_record(
        verified.records_for("split-manifest.jsonl", SplitManifest),
        description="split manifest",
    )
    summary = _single_record(
        verified.records_for("candidate-summary.jsonl", DevelopmentProjectionSummary),
        description="candidate summary",
    )
    structured = DevelopmentProjectionBundle(
        dataset_config_sha256=metadata.resolved_config_sha256,
        generator_commit=metadata.generator_commit,
        trajectories=verified.records_for("trajectories.jsonl", DevelopmentTrajectoryRecord),
        groups=verified.records_for("groups.jsonl", CounterfactualGroup),
        projections=verified.records_for("projections.jsonl", ProjectionRecord),
        counterfactual_projections=verified.records_for(
            "counterfactual-projections.jsonl", CounterfactualProjectionRecord
        ),
        split_manifest=split_manifest,
        summary=summary,
        checksum_sha256=metadata.structured_bundle_sha256,
    )
    candidate = DevelopmentCandidateBundle(
        resolved_config=config,
        aster_schema_snapshot_sha256=metadata.aster_schema_snapshot_sha256,
        dataset_schema_snapshot_sha256=metadata.dataset_schema_snapshot_sha256,
        catalog_sha256=metadata.catalog_sha256,
        guard_sha256=metadata.guard_sha256,
        structured_bundle=structured,
        pre_render_review_packet=_single_record(
            verified.records_for("pre-render-review-packet.jsonl", CatalogReviewPacket),
            description="pre-render review packet",
        ),
        pre_render_review_record=_single_record(
            verified.records_for("pre-render-review-record.jsonl", HumanReviewRecord),
            description="pre-render review record",
        ),
        rendered_candidates=verified.records_for("rendered-candidates.jsonl", RenderedCandidate),
        corruption_records=verified.records_for("corruptions.jsonl", CorruptedCandidate),
        task_examples=verified.records_for("task-examples.jsonl", TaskExampleCandidate),
        task_shortcut_records=verified.records_for(
            "task-shortcut-records.jsonl", TaskShortcutRecord
        ),
        quality_report=_single_record(
            verified.records_for("quality-report.jsonl", QualityReport),
            description="quality report",
        ),
        postrender_review_packet=_single_record(
            verified.records_for("postrender-review.jsonl", PostrenderReviewPacket),
            description="post-render review packet",
        ),
        checksum_sha256=metadata.candidate_bundle_sha256,
    )
    _validate_rendered_relationships(candidate)
    expected_metadata = candidate_artifact_metadata(candidate)
    manifest_metadata = CandidateArtifactMetadata.model_validate_json(
        canonical_json_bytes(
            verified.manifest.model_dump(mode="json", exclude={"files"}, round_trip=True)
        )
    )
    if metadata != expected_metadata or manifest_metadata != expected_metadata:
        raise ArtifactVerificationError("artifact metadata bindings do not match candidate files")
    expected_records = candidate_artifact_records(candidate)
    for filename, records in expected_records.items():
        actual = next(item.records for item in verified.files if item.filename == filename)
        if actual != records:
            raise ArtifactVerificationError("typed artifact records fail cross-file reconstruction")
    return VerifiedDevelopmentCandidateArtifact(
        manifest=verified.manifest,
        metadata=metadata,
        candidate=candidate,
    )


def verify_development_candidate_artifact(
    writer: ArtifactWriter,
    *,
    relative_directory: str,
) -> VerifiedDevelopmentCandidateArtifact:
    """Typed-load a full candidate and normalize all contract failures safely."""

    if type(writer) is not ArtifactWriter:
        raise TypeError("writer must be an exact ArtifactWriter")
    try:
        return _verify_development_candidate_artifact(
            writer,
            relative_directory=relative_directory,
        )
    except ArtifactVerificationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactVerificationError(
            "development artifact fails typed cross-file verification"
        ) from error


def write_and_verify_development_candidate(
    writer: ArtifactWriter,
    *,
    relative_directory: str,
    candidate: DevelopmentCandidateBundle,
) -> VerifiedDevelopmentCandidateArtifact:
    """Atomically write and immediately typed-verify one complete pending candidate."""

    if type(candidate) is not DevelopmentCandidateBundle:
        raise TypeError("candidate must be an exact DevelopmentCandidateBundle")
    metadata = candidate_artifact_metadata(candidate)
    written = writer.write_candidate_bundle(
        relative_directory=relative_directory,
        records_by_filename=candidate_artifact_records(candidate),
        metadata=metadata,
    )
    verified = verify_development_candidate_artifact(
        writer,
        relative_directory=relative_directory,
    )
    if verified.manifest != written or verified.candidate != candidate:
        raise ArtifactVerificationError("written candidate differs from typed verification")
    return verified


__all__ = [
    "DevelopmentCandidateBundle",
    "TaskExampleCandidate",
    "VerifiedDevelopmentCandidateArtifact",
    "build_review_gated_development_candidate",
    "candidate_artifact_metadata",
    "candidate_artifact_records",
    "development_artifact_model_specs",
    "render_development_candidate",
    "verify_development_candidate_artifact",
    "write_and_verify_development_candidate",
]
