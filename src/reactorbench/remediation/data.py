"""Physically isolated development views for Phase 6 remediation.

The builders in this module generate only an explicit split allowlist.  They never
load the historical mixed Phase 3 candidate and never import final/golden evaluation
code.  Narrative text is rendered from the narrow ``ModelInput`` boundary, while
compact decoder context is derived directly from structured prompt-local references.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, StrictStr, field_validator, model_validator

from reactorbench.dataset.catalog import AliasFamily, TemplateFamily
from reactorbench.dataset.config import DevelopmentDatasetConfig, dataset_config_sha256
from reactorbench.dataset.contracts import (
    ModelInput,
    ProjectionRecord,
    ProjectionTaskTargetValue,
    PromptCounterfactualComparisonTarget,
    PromptEvidenceTarget,
)
from reactorbench.dataset.pipeline import (
    ScopedProjectionInventory,
    build_scoped_projection_inventory,
)
from reactorbench.dataset.renderer import render_model_input
from reactorbench.evaluation.compact import (
    COMPACT_TARGET_VERSION,
    CompactTargetContext,
    compact_target_json,
    parse_compact_target,
    serialize_compact_target,
)
from reactorbench.schemas.base import (
    ContractId,
    ContractModel,
    canonical_json_bytes,
    canonical_sha256,
)
from reactorbench.schemas.enums import (
    AbstentionReason,
    ActionLabel,
    DiagnosisStatus,
    SplitName,
    TaskName,
)
from reactorbench.schemas.target import (
    FaultDiagnosisTarget,
    IncidentSummaryTarget,
    NextActionTarget,
)

from .config import VIEW_SOURCE_SPLIT, RemediationView

MAX_EXAMPLE_BYTES = 2 * 1024 * 1024
MAX_EXAMPLES = 100_000
SAFE_ARTIFACT_VERSION: Literal["0.3.0"] = "0.3.0"
FROZEN_V03_DATASET_CONFIG_SHA256 = (
    "d178f44dbaf4740c5025d5525e230f66f263b5dfe3b15b5fc2e2ed6ee8499e73"
)
FROZEN_V03_SOURCE_COMMIT = "992d86823a32813b226b73bc495d2ae6723d47ab"
FROZEN_V03_RAW_MANIFEST_SHA256 = "87420933f3e9549f8ef6785994a55e845e957c2942e0c5db0c4df5931075790a"
FROZEN_V03_DEDUPLICATED_MANIFEST_SHA256 = (
    "02ab7b7e29de7c74df5d308683b8c3d9f5d6204db0649a02d8288489d3be0af3"
)
FROZEN_V03_RAW_EXAMPLE_COUNT = 5_859
FROZEN_V03_DEDUPLICATED_EXAMPLE_COUNT = 5_835
FROZEN_V03_REMOVED_DUPLICATE_COUNT = 24
FROZEN_V03_COUNTERFACTUAL_COUNTS = (
    (RemediationView.IID_TRAIN, 40),
    (RemediationView.IID_VALIDATION, 15),
)
FROZEN_V03_TRAIN_TEMPLATE_FAMILIES = (
    "compact-log-v1",
    "observer-note-v1",
    "shift-ledger-v1",
)
FROZEN_V03_TRAIN_ALIAS_FAMILIES = (
    "canonical-v1",
    "short-v1",
    "neutral-role-v1",
)
_DEVELOPMENT_VIEWS = frozenset(RemediationView)
_VIEW_BY_SPLIT = {split: view for view, split in VIEW_SOURCE_SPLIT.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_group_id(value: str) -> str:
    """Preserve readable atomic-group IDs and hash only oversized source names."""

    if len(value) <= 96:
        return value
    prefix = value.split(":", maxsplit=1)[0]
    return f"{prefix}:sha256:{canonical_sha256(value)[:48]}"


def _strict_json(payload: bytes) -> object:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("remediation artifact contains a duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"remediation artifact contains non-finite JSON: {value}")
        ),
    )


def _canonical_examples_jsonl_bytes(examples: tuple[RemediationExample, ...]) -> bytes:
    """Return the exact canonical byte stream bound by a safe dataset manifest."""

    return b"".join(
        canonical_json_bytes(item.model_dump(mode="json", round_trip=True)) + b"\n"
        for item in examples
    )


def _visible_refs(model_input: ModelInput) -> tuple[str, ...]:
    return tuple(
        fact.fact_ref
        for fact in (
            *model_input.observation_facts,
            *model_input.event_facts,
            *model_input.context_facts,
        )
    )


def renderer_visible_structured_fingerprint(model_input: ModelInput) -> str:
    """Hash structured facts while excluding metadata the renderer never emits.

    ``ModelInput.structured_fingerprint()`` also binds ``cut_tick`` and
    ``source_event_index_exclusive``. Normalize those fields from the visible facts
    before using the canonical method, so metadata-only differences cannot evade a
    cross-view separation gate.
    """

    if type(model_input) is not ModelInput:
        raise TypeError("renderer-visible fingerprint requires an exact ModelInput")
    visible_ticks = tuple(fact.tick for fact in model_input.observation_facts) + tuple(
        fact.tick for fact in model_input.event_facts
    )
    visible_cut_tick = max(visible_ticks)
    normalized = ModelInput(
        schema_version=model_input.schema_version,
        cut_tick=visible_cut_tick,
        source_event_index_exclusive=None,
        observation_facts=model_input.observation_facts,
        event_facts=model_input.event_facts,
        context_facts=model_input.context_facts,
    )
    return normalized.structured_fingerprint()


def compact_context_for_projection(record: ProjectionRecord) -> CompactTargetContext:
    if type(record) is not ProjectionRecord:
        raise TypeError("compact context requires an exact ProjectionRecord")
    return CompactTargetContext(
        task_name=record.task_target.task_name,
        visible_fact_refs=_visible_refs(record.model_input),
    )


def _classification_label(target: ProjectionTaskTargetValue) -> str | None:
    if type(target) is FaultDiagnosisTarget:
        if target.diagnosis_status is not DiagnosisStatus.DIAGNOSED:
            return target.diagnosis_status.value
        return "DIAGNOSED:" + "+".join(label.value for label in target.fault_labels)
    if type(target) is NextActionTarget:
        return target.immediate_action.value
    task_name = target.task_name
    if task_name is TaskName.CONTINUE_LOG:
        return str(target.next_event_type.value)  # type: ignore[union-attr]
    return None


def _refs_in_prompt_order(values: tuple[str, ...], visible: tuple[str, ...]) -> tuple[str, ...]:
    positions = {value: index for index, value in enumerate(visible)}
    if not set(values).issubset(positions):
        raise ValueError("compact target references a fact outside its visible prompt")
    return tuple(sorted(values, key=positions.__getitem__))


def _canonical_compact_target(
    target: ProjectionTaskTargetValue, *, context: CompactTargetContext
) -> ProjectionTaskTargetValue:
    """Normalize set-like prompt references for the v0.2 learned contract.

    Historical v0.1 evidence targets preserved semantic slot discovery order, which
    can interleave prompt-reference namespaces.  The compact language freezes prompt
    order for those set-like references while preserving the separately meaningful
    evidence-slot order.  Historical records remain untouched.
    """

    if type(target) is PromptEvidenceTarget:
        return PromptEvidenceTarget(
            fact_refs=_refs_in_prompt_order(target.fact_refs, context.visible_fact_refs),
            evidence_slots=target.evidence_slots,
        )
    if type(target) is PromptCounterfactualComparisonTarget:
        return PromptCounterfactualComparisonTarget(
            baseline=target.baseline,
            counterfactual=target.counterfactual,
            changed_fields=target.changed_fields,
            baseline_decisive_fact_refs=_refs_in_prompt_order(
                target.baseline_decisive_fact_refs, context.visible_fact_refs
            ),
            counterfactual_decisive_fact_refs=_refs_in_prompt_order(
                target.counterfactual_decisive_fact_refs,
                context.counterfactual_visible_fact_refs,
            ),
            decisive_evidence_slots=target.decisive_evidence_slots,
        )
    return target


class RemediationExample(ContractModel):
    artifact_version: Literal["0.3.0"] = SAFE_ARTIFACT_VERSION
    example_id: ContractId
    view: RemediationView
    source_split: SplitName
    task_name: TaskName
    group_id: ContractId
    source_record_ids: tuple[ContractId, ...]
    parent_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_text: str = Field(min_length=1, max_length=1024 * 1024)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_family_id: ContractId
    alias_family_id: ContractId
    compact_context: CompactTargetContext
    compact_target: str = Field(min_length=1, max_length=16 * 1024)
    canonical_target_json: str = Field(min_length=2, max_length=65_536)
    classification_label: StrictStr | None = Field(default=None, max_length=512)
    augmentation: Literal["none", "renderer_variant", "remove_decisive_evidence"]
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("source_record_ids", mode="after")
    @classmethod
    def sources_are_nonempty_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("source record IDs must be non-empty and unique")
        return value

    @model_validator(mode="after")
    def boundaries_and_checksum_match(self) -> RemediationExample:
        if self.source_split is not VIEW_SOURCE_SPLIT[self.view]:
            raise ValueError("remediation view does not match its source split recipe")
        if self.task_name is not self.compact_context.task_name:
            raise ValueError("example task and compact context differ")
        if hashlib.sha256(self.prompt_text.encode("utf-8")).hexdigest() != self.prompt_sha256:
            raise ValueError("prompt checksum mismatch")
        target = parse_compact_target(self.compact_target, context=self.compact_context)
        if target.task_name is not self.task_name:
            raise ValueError("compact target task mismatch")
        if self.classification_label != _classification_label(target):
            raise ValueError("classification label differs from the compact target")
        if compact_target_json(self.compact_target, context=self.compact_context) != (
            self.canonical_target_json
        ):
            raise ValueError("compact and canonical target representations differ")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("remediation example checksum mismatch")
        return self


class SafeDevelopmentManifest(ContractModel):
    artifact_version: Literal["0.3.0"] = SAFE_ARTIFACT_VERSION
    boundary: Literal["development_only_no_final_or_golden_payloads"] = (
        "development_only_no_final_or_golden_payloads"
    )
    source_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    dataset_version: str
    dataset_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compact_contract_version: Literal["0.2.0"] = COMPACT_TARGET_VERSION
    views: tuple[RemediationView, ...]
    example_count: int = Field(ge=1, le=MAX_EXAMPLES)
    counts_by_view: tuple[tuple[RemediationView, int], ...]
    counts_by_task: tuple[tuple[TaskName, int], ...]
    examples_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    examples_size_bytes: int = Field(gt=0, le=MAX_EXAMPLE_BYTES * MAX_EXAMPLES)
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def inventory_is_canonical(self) -> SafeDevelopmentManifest:
        if not self.views or any(view not in _DEVELOPMENT_VIEWS for view in self.views):
            raise ValueError("safe artifact contains an unsupported view")
        canonical_views = tuple(view for view in RemediationView if view in set(self.views))
        if self.views != canonical_views:
            raise ValueError("safe artifact views must be unique and canonical")
        view_keys = tuple(view for view, _count in self.counts_by_view)
        if view_keys != self.views or any(count <= 0 for _view, count in self.counts_by_view):
            raise ValueError("view counts must be positive and keyed in canonical view order")
        task_keys = tuple(task for task, _count in self.counts_by_task)
        canonical_tasks = tuple(task for task in TaskName if task in set(task_keys))
        if (
            not task_keys
            or task_keys != canonical_tasks
            or any(count <= 0 for _task, count in self.counts_by_task)
        ):
            raise ValueError("task counts must be positive, unique, and canonical")
        if sum(count for _, count in self.counts_by_view) != self.example_count:
            raise ValueError("view counts do not cover the safe artifact")
        if sum(count for _, count in self.counts_by_task) != self.example_count:
            raise ValueError("task counts do not cover the safe artifact")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("safe development manifest checksum mismatch")
        return self


class SafeDevelopmentDataset(ContractModel):
    manifest: SafeDevelopmentManifest
    examples: tuple[RemediationExample, ...]

    @model_validator(mode="after")
    def examples_match_manifest(self) -> SafeDevelopmentDataset:
        if len(self.examples) != self.manifest.example_count:
            raise ValueError("safe dataset count differs from its manifest")
        ids = tuple(item.example_id for item in self.examples)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("safe examples must have unique canonical IDs")
        inventory = canonical_sha256(
            tuple((item.example_id, item.checksum_sha256) for item in self.examples)
        )
        if inventory != self.manifest.inventory_sha256:
            raise ValueError("safe dataset inventory checksum mismatch")
        view_counts = Counter(item.view for item in self.examples)
        expected_views = tuple(view for view in RemediationView if view_counts[view])
        expected_view_counts = tuple((view, view_counts[view]) for view in expected_views)
        if self.manifest.views != expected_views:
            raise ValueError("safe dataset view inventory differs from its manifest")
        if self.manifest.counts_by_view != expected_view_counts:
            raise ValueError("safe dataset view counts differ from its examples")
        task_counts = Counter(item.task_name for item in self.examples)
        expected_task_counts = tuple(
            (task, task_counts[task]) for task in TaskName if task_counts[task]
        )
        if self.manifest.counts_by_task != expected_task_counts:
            raise ValueError("safe dataset task counts differ from its examples")
        payload = _canonical_examples_jsonl_bytes(self.examples)
        if (
            len(payload) != self.manifest.examples_size_bytes
            or hashlib.sha256(payload).hexdigest() != self.manifest.examples_sha256
        ):
            raise ValueError("safe dataset canonical payload differs from its manifest")
        return self


@dataclass(frozen=True, slots=True)
class TaskScopedStructuredFingerprint:
    """In-memory proof that one example derives from renderer-visible structure.

    This deliberately remains outside the serialized dataset contract.  The v0.4
    pipeline checksum-binds canonical inventories of these records without changing
    the already frozen ``RemediationExample`` or ``SafeDevelopmentDataset`` schemas.
    """

    example_id: str
    view: RemediationView
    task_name: TaskName
    structured_fingerprint_sha256: str

    def __post_init__(self) -> None:
        fingerprint = self.structured_fingerprint_sha256
        if (
            type(self.example_id) is not str
            or not self.example_id
            or type(self.view) is not RemediationView
            or type(self.task_name) is not TaskName
            or type(fingerprint) is not str
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError("task-scoped structured fingerprint record is invalid")

    @property
    def separation_key(self) -> tuple[TaskName, str]:
        """Return the truth-independent key that must not cross development views."""

        return self.task_name, self.structured_fingerprint_sha256


@dataclass(frozen=True, slots=True)
class FrozenV03IIDMaterial:
    """One-pass raw reproduction plus the exact deduplicated training material."""

    raw_dataset: SafeDevelopmentDataset
    dataset: SafeDevelopmentDataset
    structured_fingerprints: tuple[TaskScopedStructuredFingerprint, ...]

    def __post_init__(self) -> None:
        if (
            type(self.raw_dataset) is not SafeDevelopmentDataset
            or type(self.dataset) is not SafeDevelopmentDataset
            or type(self.structured_fingerprints) is not tuple
            or any(
                type(item) is not TaskScopedStructuredFingerprint
                for item in self.structured_fingerprints
            )
        ):
            raise TypeError("frozen v0.3 material requires exact in-memory contracts")
        if (
            self.raw_dataset.manifest.checksum_sha256 != FROZEN_V03_RAW_MANIFEST_SHA256
            or self.dataset.manifest.checksum_sha256 != FROZEN_V03_DEDUPLICATED_MANIFEST_SHA256
            or len(self.raw_dataset.examples) != FROZEN_V03_RAW_EXAMPLE_COUNT
            or len(self.dataset.examples) != FROZEN_V03_DEDUPLICATED_EXAMPLE_COUNT
        ):
            raise ValueError("frozen v0.3 raw or deduplicated manifest differs")
        expected_views = (RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION)
        if (
            self.raw_dataset.manifest.views != expected_views
            or self.dataset.manifest.views != expected_views
            or self.raw_dataset.manifest.source_commit != FROZEN_V03_SOURCE_COMMIT
            or self.dataset.manifest.source_commit != FROZEN_V03_SOURCE_COMMIT
            or self.raw_dataset.manifest.dataset_config_sha256 != FROZEN_V03_DATASET_CONFIG_SHA256
            or self.dataset.manifest.dataset_config_sha256 != FROZEN_V03_DATASET_CONFIG_SHA256
        ):
            raise ValueError("frozen v0.3 material crossed its IID recipe binding")

        raw_inventory = {
            (item.example_id, item.checksum_sha256): item for item in self.raw_dataset.examples
        }
        deduplicated_inventory = {
            (item.example_id, item.checksum_sha256): item for item in self.dataset.examples
        }
        if (
            len(raw_inventory) != len(self.raw_dataset.examples)
            or len(deduplicated_inventory) != len(self.dataset.examples)
            or not set(deduplicated_inventory).issubset(raw_inventory)
        ):
            raise ValueError("deduplicated v0.3 inventory is not a bit-exact raw subset")
        removed_keys = set(raw_inventory) - set(deduplicated_inventory)
        if len(removed_keys) != FROZEN_V03_REMOVED_DUPLICATE_COUNT:
            raise ValueError("frozen v0.3 deduplication removed the wrong row count")
        removed = tuple(raw_inventory[key] for key in sorted(removed_keys))
        retained_signatures = Counter(
            (item.task_name, item.prompt_sha256, item.canonical_target_json)
            for item in self.dataset.examples
        )
        if any(
            item.augmentation != "remove_decisive_evidence"
            or item.task_name is TaskName.COUNTERFACTUAL_COMPARE
            or retained_signatures[(item.task_name, item.prompt_sha256, item.canonical_target_json)]
            != 1
            for item in removed
        ):
            raise ValueError("v0.3 removed rows are not exact evidence-removal duplicates")
        raw_prompt_counts = Counter(
            (item.task_name, item.prompt_sha256) for item in self.raw_dataset.examples
        )
        deduplicated_prompt_counts = Counter(
            (item.task_name, item.prompt_sha256) for item in self.dataset.examples
        )
        if sum(
            count - 1 for count in raw_prompt_counts.values() if count > 1
        ) != FROZEN_V03_REMOVED_DUPLICATE_COUNT or any(
            count != 1 for count in deduplicated_prompt_counts.values()
        ):
            raise ValueError("v0.3 task-scoped prompt deduplication is not exact")

        counterfactual_raw = tuple(
            (item.example_id, item.checksum_sha256, item.view)
            for item in self.raw_dataset.examples
            if item.task_name is TaskName.COUNTERFACTUAL_COMPARE
        )
        counterfactual_deduplicated = tuple(
            (item.example_id, item.checksum_sha256, item.view)
            for item in self.dataset.examples
            if item.task_name is TaskName.COUNTERFACTUAL_COMPARE
        )
        counterfactual_counts = Counter(item[2] for item in counterfactual_raw)
        if (
            counterfactual_raw != counterfactual_deduplicated
            or tuple(
                (view, counterfactual_counts[view])
                for view, _expected in FROZEN_V03_COUNTERFACTUAL_COUNTS
            )
            != FROZEN_V03_COUNTERFACTUAL_COUNTS
        ):
            raise ValueError("v0.3 counterfactual inventory changed during deduplication")
        if tuple(item.example_id for item in self.structured_fingerprints) != tuple(
            item.example_id for item in self.dataset.examples
        ):
            raise ValueError("deduplicated fingerprints differ from v0.3 examples")

    @property
    def removed_examples(self) -> tuple[RemediationExample, ...]:
        """Return the exact raw rows excluded from training and evaluation."""

        retained = {(item.example_id, item.checksum_sha256) for item in self.dataset.examples}
        return tuple(
            item
            for item in self.raw_dataset.examples
            if (item.example_id, item.checksum_sha256) not in retained
        )


def _make_example(
    *,
    view: RemediationView,
    source_split: SplitName,
    source_record_ids: tuple[str, ...],
    parent_record_sha256: str,
    group_id: str,
    prompt_text: str,
    template_family: TemplateFamily,
    alias_family: AliasFamily,
    context: CompactTargetContext,
    target: ProjectionTaskTargetValue,
    augmentation: Literal["none", "renderer_variant", "remove_decisive_evidence"],
) -> RemediationExample:
    target = _canonical_compact_target(target, context=context)
    compact = serialize_compact_target(target, context=context)
    canonical = compact_target_json(compact, context=context)
    identity = canonical_sha256(
        {
            "view": view,
            "source_record_ids": source_record_ids,
            "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            "task_name": target.task_name,
            "compact_target": compact,
            "augmentation": augmentation,
        }
    )
    draft = RemediationExample.model_construct(
        artifact_version=SAFE_ARTIFACT_VERSION,
        example_id=f"rbexample:{identity[:24]}",
        view=view,
        source_split=source_split,
        task_name=target.task_name,
        group_id=group_id,
        source_record_ids=source_record_ids,
        parent_record_sha256=parent_record_sha256,
        prompt_text=prompt_text,
        prompt_sha256=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        template_family_id=template_family.value,
        alias_family_id=alias_family.value,
        compact_context=context,
        compact_target=compact,
        canonical_target_json=canonical,
        classification_label=_classification_label(target),
        augmentation=augmentation,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return RemediationExample(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
        checksum_sha256=checksum,
    )


def _style_variants(
    record: ProjectionRecord,
    *,
    planned_template: str,
    planned_alias: str,
    view: RemediationView,
    template_families: tuple[str, ...],
    alias_families: tuple[str, ...],
    variant_count: int,
) -> tuple[tuple[TemplateFamily, AliasFamily, Literal["none", "renderer_variant"]], ...]:
    original: tuple[TemplateFamily, AliasFamily, Literal["none", "renderer_variant"]] = (
        TemplateFamily(planned_template),
        AliasFamily(planned_alias),
        "none",
    )
    if view is not RemediationView.IID_TRAIN or variant_count == 1:
        return (original,)
    combinations: list[tuple[TemplateFamily, AliasFamily, Literal["none", "renderer_variant"]]] = [
        (TemplateFamily(template), AliasFamily(alias), "renderer_variant")
        for template in template_families
        for alias in alias_families
        if (template, alias) != (planned_template, planned_alias)
    ]
    combinations.sort(
        key=lambda item: canonical_sha256((record.projection_id, item[0].value, item[1].value))
    )
    candidates = (original, *combinations)
    if len(candidates) < variant_count:
        raise ValueError("requested renderer variant count exceeds the declared family product")
    return candidates


def _renumber_visible_facts(
    model_input: ModelInput, *, remove_refs: frozenset[str]
) -> ModelInput | None:
    observations = tuple(
        fact for fact in model_input.observation_facts if fact.fact_ref not in remove_refs
    )
    events = tuple(fact for fact in model_input.event_facts if fact.fact_ref not in remove_refs)
    contexts = tuple(fact for fact in model_input.context_facts if fact.fact_ref not in remove_refs)
    if not observations and not events:
        return None
    observations = tuple(
        fact.model_copy(update={"fact_ref": f"o-{index:04d}"})
        for index, fact in enumerate(observations)
    )
    events = tuple(
        fact.model_copy(update={"fact_ref": f"e-{index:04d}"}) for index, fact in enumerate(events)
    )
    contexts = tuple(
        fact.model_copy(update={"fact_ref": f"c-{index:04d}"})
        for index, fact in enumerate(contexts)
    )
    return ModelInput(
        cut_tick=model_input.cut_tick,
        source_event_index_exclusive=model_input.source_event_index_exclusive,
        observation_facts=observations,
        event_facts=events,
        context_facts=contexts,
    )


def _insufficient_target(target: ProjectionTaskTargetValue) -> ProjectionTaskTargetValue | None:
    if type(target) is FaultDiagnosisTarget:
        return FaultDiagnosisTarget(
            diagnosis_status=DiagnosisStatus.UNRESOLVED,
            fault_labels=(),
            abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
        )
    if type(target) is NextActionTarget:
        return NextActionTarget(immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE)
    if type(target) is PromptEvidenceTarget:
        return PromptEvidenceTarget(fact_refs=(), evidence_slots=())
    if type(target) is IncidentSummaryTarget:
        return IncidentSummaryTarget(
            affected_subsystems=(),
            observed_trend=target.observed_trend,
            diagnosis_status=DiagnosisStatus.UNRESOLVED,
            fault_labels=(),
            operating_mode=target.operating_mode,
            immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE,
            abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
        )
    return None


def _evidence_removal_examples(
    inventory: ScopedProjectionInventory,
    *,
    trajectories_by_id: dict[str, Any],
) -> tuple[tuple[RemediationExample, str], ...]:
    by_fingerprint: dict[str, list[ProjectionRecord]] = defaultdict(list)
    for record in inventory.projections:
        by_fingerprint[renderer_visible_structured_fingerprint(record.model_input)].append(record)
    results: list[tuple[RemediationExample, str]] = []
    for records in by_fingerprint.values():
        evidence = next(
            (
                item.task_target.target
                for item in records
                if type(item.task_target.target) is PromptEvidenceTarget
            ),
            None,
        )
        if not isinstance(evidence, PromptEvidenceTarget) or not evidence.fact_refs:
            continue
        remove_refs = frozenset(evidence.fact_refs)
        for record in records:
            target = _insufficient_target(record.task_target.target)
            if target is None:
                continue
            reduced = _renumber_visible_facts(record.model_input, remove_refs=remove_refs)
            if reduced is None:
                continue
            trajectory = trajectories_by_id[record.lineage.trajectory_id]
            rendered = render_model_input(
                reduced,
                template_family=TemplateFamily(trajectory.template_family_id),
                alias_family=AliasFamily(trajectory.alias_family_id),
                split_name=SplitName.IID_TRAIN,
            )
            context = CompactTargetContext(
                task_name=target.task_name,
                visible_fact_refs=_visible_refs(reduced),
            )
            example = _make_example(
                view=RemediationView.IID_TRAIN,
                source_split=SplitName.IID_TRAIN,
                source_record_ids=(record.projection_id,),
                parent_record_sha256=record.checksum_sha256,
                group_id=_bounded_group_id(
                    f"evidence-removal:{record.lineage.scenario_id}:{target.task_name.value}"
                ),
                prompt_text=rendered.text,
                template_family=TemplateFamily(trajectory.template_family_id),
                alias_family=AliasFamily(trajectory.alias_family_id),
                context=context,
                target=target,
                augmentation="remove_decisive_evidence",
            )
            results.append((example, renderer_visible_structured_fingerprint(reduced)))
    return tuple(results)


def _deduplicate_evidence_removal_examples(
    candidates: tuple[tuple[RemediationExample, str], ...],
) -> tuple[tuple[RemediationExample, str], ...]:
    """Keep one canonical row per task/prompt without hiding target conflicts."""

    selected: dict[tuple[TaskName, str], tuple[RemediationExample, str]] = {}
    for candidate in sorted(candidates, key=lambda item: item[0].example_id):
        example, _fingerprint = candidate
        key = example.task_name, example.prompt_sha256
        existing = selected.get(key)
        if existing is not None:
            if existing[0].canonical_target_json != example.canonical_target_json:
                raise ValueError(
                    "evidence-removal prompt collision has conflicting compact targets"
                )
            continue
        selected[key] = candidate
    return tuple(selected.values())


def _finalize_safe_development_dataset(
    examples: tuple[RemediationExample, ...],
    *,
    config: DevelopmentDatasetConfig,
    source_commit: str,
    views: tuple[RemediationView, ...],
) -> SafeDevelopmentDataset:
    if not examples or len(examples) > MAX_EXAMPLES:
        raise ValueError("safe development example count is outside its bound")
    if tuple(item.example_id for item in examples) != tuple(
        sorted(item.example_id for item in examples)
    ) or len({item.example_id for item in examples}) != len(examples):
        raise ValueError("safe development example IDs are not unique and canonical")
    view_counts = Counter(item.view for item in examples)
    task_counts = Counter(item.task_name for item in examples)
    payload = _canonical_examples_jsonl_bytes(examples)
    draft = SafeDevelopmentManifest.model_construct(
        artifact_version=SAFE_ARTIFACT_VERSION,
        boundary="development_only_no_final_or_golden_payloads",
        source_commit=source_commit,
        dataset_version=config.dataset.dataset_version,
        dataset_config_sha256=dataset_config_sha256(config),
        compact_contract_version=COMPACT_TARGET_VERSION,
        views=views,
        example_count=len(examples),
        counts_by_view=tuple((view, view_counts[view]) for view in views),
        counts_by_task=tuple((task, task_counts[task]) for task in TaskName if task_counts[task]),
        examples_sha256=hashlib.sha256(payload).hexdigest(),
        examples_size_bytes=len(payload),
        inventory_sha256=canonical_sha256(
            tuple((item.example_id, item.checksum_sha256) for item in examples)
        ),
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    manifest = SafeDevelopmentManifest(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
        checksum_sha256=checksum,
    )
    return SafeDevelopmentDataset(manifest=manifest, examples=examples)


def _build_safe_development_material(
    config: DevelopmentDatasetConfig,
    *,
    source_commit: str,
    views: tuple[RemediationView, ...],
    train_template_families: tuple[str, ...] = (),
    train_alias_families: tuple[str, ...] = (),
    renderer_variants_per_projection: int = 1,
    include_insufficient_evidence_views: bool = False,
) -> tuple[
    SafeDevelopmentDataset,
    SafeDevelopmentDataset,
    tuple[TaskScopedStructuredFingerprint, ...],
    tuple[TaskScopedStructuredFingerprint, ...],
]:
    """Build a checksum-bound dataset without constructing any unrequested split."""

    if type(config) is not DevelopmentDatasetConfig:
        raise TypeError("config must be an exact DevelopmentDatasetConfig")
    if (
        type(source_commit) is not str
        or not 7 <= len(source_commit) <= 64
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("source_commit must be a lowercase hexadecimal Git revision")
    if (
        type(views) is not tuple
        or not views
        or any(type(view) is not RemediationView for view in views)
    ):
        raise TypeError("views must be a non-empty exact RemediationView tuple")
    if len(views) != len(set(views)):
        raise ValueError("development views must be unique")
    if any(view in SHADOW_VIEW_SET for view in views) and config.dataset.dataset_version != "0.3.0":
        raise ValueError("shadow development views require the dedicated v0.3 dataset policy")
    if (
        type(renderer_variants_per_projection) is not int
        or not 1 <= renderer_variants_per_projection <= 9
    ):
        raise ValueError("renderer variant count must be in [1,9]")
    if renderer_variants_per_projection > 1 and (
        not train_template_families or not train_alias_families
    ):
        raise ValueError("renderer augmentation requires explicit template and alias families")
    source_splits = tuple(VIEW_SOURCE_SPLIT[view] for view in views)
    if SplitName.IID_TEST in source_splits:
        raise ValueError("development builder cannot request a final IID test split")
    inventory = build_scoped_projection_inventory(
        config,
        generator_commit=source_commit,
        splits=source_splits,
    )
    trajectories_by_id = {
        record.trajectory.trajectory_id: record for record in inventory.trajectories
    }
    examples: list[RemediationExample] = []
    structured_fingerprints: list[TaskScopedStructuredFingerprint] = []
    for record in inventory.projections:
        trajectory = trajectories_by_id[record.lineage.trajectory_id]
        view = _VIEW_BY_SPLIT[trajectory.split_name]
        rendered_prompt_hashes: set[str] = set()
        rendered_variant_count = 0
        for template, alias, augmentation in _style_variants(
            record,
            planned_template=trajectory.template_family_id,
            planned_alias=trajectory.alias_family_id,
            view=view,
            template_families=train_template_families,
            alias_families=train_alias_families,
            variant_count=renderer_variants_per_projection,
        ):
            rendered = render_model_input(
                record.model_input,
                template_family=template,
                alias_family=alias,
                split_name=trajectory.split_name,
            )
            prompt_hash = hashlib.sha256(rendered.text.encode("utf-8")).hexdigest()
            if prompt_hash in rendered_prompt_hashes:
                continue
            rendered_prompt_hashes.add(prompt_hash)
            context = compact_context_for_projection(record)
            example = _make_example(
                view=view,
                source_split=trajectory.split_name,
                source_record_ids=(record.projection_id,),
                parent_record_sha256=record.checksum_sha256,
                group_id=_bounded_group_id(
                    f"source:{record.lineage.scenario_id}:{record.task_target.task_name.value}"
                ),
                prompt_text=rendered.text,
                template_family=template,
                alias_family=alias,
                context=context,
                target=record.task_target.target,
                augmentation=augmentation,
            )
            examples.append(example)
            structured_fingerprints.append(
                TaskScopedStructuredFingerprint(
                    example_id=example.example_id,
                    view=view,
                    task_name=record.task_target.task_name,
                    structured_fingerprint_sha256=renderer_visible_structured_fingerprint(
                        record.model_input
                    ),
                )
            )
            rendered_variant_count += 1
            if rendered_variant_count == renderer_variants_per_projection:
                break
        expected_variants = (
            renderer_variants_per_projection if view is RemediationView.IID_TRAIN else 1
        )
        if rendered_variant_count != expected_variants:
            raise ValueError(
                "renderer families did not produce the requested number of distinct prompts"
            )
    records_by_projection = {record.projection_id: record for record in inventory.projections}
    for pair in inventory.counterfactual_projections:
        baseline = records_by_projection[pair.lineage.baseline_projection_id]
        counterfactual = records_by_projection[pair.lineage.counterfactual_projection_id]
        baseline_source = trajectories_by_id[baseline.lineage.trajectory_id]
        counterfactual_source = trajectories_by_id[counterfactual.lineage.trajectory_id]
        if baseline_source.split_name is not counterfactual_source.split_name:
            raise ValueError("counterfactual pair crossed source splits")
        view = _VIEW_BY_SPLIT[baseline_source.split_name]
        baseline_render = render_model_input(
            pair.model_input.baseline,
            template_family=TemplateFamily(baseline_source.template_family_id),
            alias_family=AliasFamily(baseline_source.alias_family_id),
            split_name=baseline_source.split_name,
        )
        counterfactual_render = render_model_input(
            pair.model_input.counterfactual,
            template_family=TemplateFamily(counterfactual_source.template_family_id),
            alias_family=AliasFamily(counterfactual_source.alias_family_id),
            split_name=counterfactual_source.split_name,
        )
        context = CompactTargetContext(
            task_name=TaskName.COUNTERFACTUAL_COMPARE,
            visible_fact_refs=_visible_refs(pair.model_input.baseline),
            counterfactual_visible_fact_refs=_visible_refs(pair.model_input.counterfactual),
        )
        example = _make_example(
            view=view,
            source_split=baseline_source.split_name,
            source_record_ids=(pair.pair_id,),
            parent_record_sha256=pair.checksum_sha256,
            group_id=_bounded_group_id(f"counterfactual:{pair.lineage.counterfactual_group_id}"),
            prompt_text=(
                "[BASELINE]\n"
                + baseline_render.text
                + "\n[COUNTERFACTUAL]\n"
                + counterfactual_render.text
            ),
            template_family=TemplateFamily(baseline_source.template_family_id),
            alias_family=AliasFamily(baseline_source.alias_family_id),
            context=context,
            target=pair.task_target.target,
            augmentation="none",
        )
        examples.append(example)
        paired_fingerprint = canonical_sha256(
            {
                "fingerprint_version": "counterfactual-pair-1.0.0",
                "baseline": renderer_visible_structured_fingerprint(pair.model_input.baseline),
                "counterfactual": renderer_visible_structured_fingerprint(
                    pair.model_input.counterfactual
                ),
            }
        )
        structured_fingerprints.append(
            TaskScopedStructuredFingerprint(
                example_id=example.example_id,
                view=view,
                task_name=TaskName.COUNTERFACTUAL_COMPARE,
                structured_fingerprint_sha256=paired_fingerprint,
            )
        )
    raw_examples = list(examples)
    raw_structured_fingerprints = list(structured_fingerprints)
    if include_insufficient_evidence_views and RemediationView.IID_TRAIN in views:
        train_inventory = ScopedProjectionInventory(
            requested_splits=(SplitName.IID_TRAIN,),
            trajectories=tuple(
                item for item in inventory.trajectories if item.split_name is SplitName.IID_TRAIN
            ),
            groups=tuple(
                group
                for group in inventory.groups
                if all(
                    member.scenario_id
                    in {
                        item.trajectory.scenario_id
                        for item in inventory.trajectories
                        if item.split_name is SplitName.IID_TRAIN
                    }
                    for member in group.members
                )
            ),
            projections=tuple(
                item
                for item in inventory.projections
                if trajectories_by_id[item.lineage.trajectory_id].split_name is SplitName.IID_TRAIN
            ),
            counterfactual_projections=(),
        )
        raw_evidence_removals = _evidence_removal_examples(
            train_inventory, trajectories_by_id=trajectories_by_id
        )
        evidence_removals = _deduplicate_evidence_removal_examples(raw_evidence_removals)
        raw_examples.extend(example for example, _fingerprint in raw_evidence_removals)
        raw_structured_fingerprints.extend(
            TaskScopedStructuredFingerprint(
                example_id=example.example_id,
                view=example.view,
                task_name=example.task_name,
                structured_fingerprint_sha256=fingerprint,
            )
            for example, fingerprint in raw_evidence_removals
        )
        examples.extend(example for example, _fingerprint in evidence_removals)
        structured_fingerprints.extend(
            TaskScopedStructuredFingerprint(
                example_id=example.example_id,
                view=example.view,
                task_name=example.task_name,
                structured_fingerprint_sha256=fingerprint,
            )
            for example, fingerprint in evidence_removals
        )
    raw_examples.sort(key=lambda item: item.example_id)
    raw_structured_fingerprints.sort(key=lambda item: item.example_id)
    examples.sort(key=lambda item: item.example_id)
    structured_fingerprints.sort(key=lambda item: item.example_id)
    if tuple(item.example_id for item in raw_structured_fingerprints) != tuple(
        item.example_id for item in raw_examples
    ) or tuple(item.example_id for item in structured_fingerprints) != tuple(
        item.example_id for item in examples
    ):
        raise ValueError("structured fingerprint inventory differs from safe examples")
    raw_dataset = _finalize_safe_development_dataset(
        tuple(raw_examples),
        config=config,
        source_commit=source_commit,
        views=views,
    )
    deduplicated_dataset = _finalize_safe_development_dataset(
        tuple(examples),
        config=config,
        source_commit=source_commit,
        views=views,
    )
    return (
        raw_dataset,
        deduplicated_dataset,
        tuple(raw_structured_fingerprints),
        tuple(structured_fingerprints),
    )


def build_safe_development_dataset(
    config: DevelopmentDatasetConfig,
    *,
    source_commit: str,
    views: tuple[RemediationView, ...],
    train_template_families: tuple[str, ...] = (),
    train_alias_families: tuple[str, ...] = (),
    renderer_variants_per_projection: int = 1,
    include_insufficient_evidence_views: bool = False,
) -> SafeDevelopmentDataset:
    """Build a checksum-bound dataset without constructing any unrequested split."""

    _raw_dataset, dataset, _raw_inventory, _inventory = _build_safe_development_material(
        config,
        source_commit=source_commit,
        views=views,
        train_template_families=train_template_families,
        train_alias_families=train_alias_families,
        renderer_variants_per_projection=renderer_variants_per_projection,
        include_insufficient_evidence_views=include_insufficient_evidence_views,
    )
    return dataset


def build_safe_development_dataset_with_structured_fingerprints(
    config: DevelopmentDatasetConfig,
    *,
    source_commit: str,
    views: tuple[RemediationView, ...],
    train_template_families: tuple[str, ...] = (),
    train_alias_families: tuple[str, ...] = (),
    renderer_variants_per_projection: int = 1,
    include_insufficient_evidence_views: bool = False,
) -> tuple[SafeDevelopmentDataset, tuple[TaskScopedStructuredFingerprint, ...]]:
    """Build a safe dataset and its parallel renderer-visible fingerprint proof."""

    _raw_dataset, dataset, _raw_inventory, inventory = _build_safe_development_material(
        config,
        source_commit=source_commit,
        views=views,
        train_template_families=train_template_families,
        train_alias_families=train_alias_families,
        renderer_variants_per_projection=renderer_variants_per_projection,
        include_insufficient_evidence_views=include_insufficient_evidence_views,
    )
    return dataset, inventory


def build_frozen_v03_iid_material(
    config: DevelopmentDatasetConfig,
    *,
    source_commit: str,
    train_template_families: tuple[str, ...],
    train_alias_families: tuple[str, ...],
    renderer_variants_per_projection: int,
    include_insufficient_evidence_views: bool,
) -> FrozenV03IIDMaterial:
    """Reproduce raw cap evidence and deduplicated IID material in one generator pass.

    This API has no view parameter by design: it can construct only the frozen v0.3
    IID training and IID validation recipe, never final, golden, or shadow inputs.
    """

    if (
        type(config) is not DevelopmentDatasetConfig
        or dataset_config_sha256(config) != FROZEN_V03_DATASET_CONFIG_SHA256
        or source_commit != FROZEN_V03_SOURCE_COMMIT
        or train_template_families != FROZEN_V03_TRAIN_TEMPLATE_FAMILIES
        or train_alias_families != FROZEN_V03_TRAIN_ALIAS_FAMILIES
        or type(renderer_variants_per_projection) is not int
        or renderer_variants_per_projection != 3
        or type(include_insufficient_evidence_views) is not bool
        or not include_insufficient_evidence_views
    ):
        raise ValueError("frozen v0.3 IID material request differs from its immutable recipe")
    raw_dataset, dataset, _raw_fingerprints, fingerprints = _build_safe_development_material(
        config,
        source_commit=source_commit,
        views=(RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION),
        train_template_families=train_template_families,
        train_alias_families=train_alias_families,
        renderer_variants_per_projection=renderer_variants_per_projection,
        include_insufficient_evidence_views=include_insufficient_evidence_views,
    )
    return FrozenV03IIDMaterial(
        raw_dataset=raw_dataset,
        dataset=dataset,
        structured_fingerprints=fingerprints,
    )


SHADOW_VIEW_SET = frozenset(view for view in RemediationView if view.value.startswith("shadow_"))


def write_safe_development_artifact(
    dataset: SafeDevelopmentDataset, output_directory: Path
) -> SafeDevelopmentManifest:
    if type(dataset) is not SafeDevelopmentDataset or not isinstance(output_directory, Path):
        raise TypeError("safe artifact write requires an exact dataset and Path")
    SafeDevelopmentDataset.model_validate(dataset.model_dump(mode="python", round_trip=True))
    parent = output_directory.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    if (
        parent.is_symlink()
        or not parent.is_dir()
        or output_directory.exists()
        or output_directory.is_symlink()
    ):
        raise FileExistsError("safe development output must be a new regular directory")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.tmp-", dir=parent))
    try:
        examples_path = temporary / "examples.jsonl"
        with examples_path.open("xb") as stream:
            for item in dataset.examples:
                payload = (
                    canonical_json_bytes(item.model_dump(mode="json", round_trip=True)) + b"\n"
                )
                if len(payload) > MAX_EXAMPLE_BYTES:
                    raise ValueError("one safe development example exceeds its byte bound")
                stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if (
            _sha256(examples_path) != dataset.manifest.examples_sha256
            or examples_path.stat().st_size != dataset.manifest.examples_size_bytes
        ):
            raise RuntimeError("safe development serialization changed its manifest")
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(
                canonical_json_bytes(dataset.manifest.model_dump(mode="json", round_trip=True))
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, output_directory)
        return dataset.manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_safe_development_artifact(output_directory: Path) -> SafeDevelopmentDataset:
    if (
        not isinstance(output_directory, Path)
        or output_directory.is_symlink()
        or not output_directory.is_dir()
        or {item.name for item in output_directory.iterdir()} != {"examples.jsonl", "manifest.json"}
    ):
        raise ValueError("safe development artifact inventory is missing or unsafe")
    manifest_path = output_directory / "manifest.json"
    examples_path = output_directory / "examples.jsonl"
    if any(path.is_symlink() or not path.is_file() for path in (manifest_path, examples_path)):
        raise ValueError("safe development artifact contains a symlink or non-file")
    manifest_payload = manifest_path.read_bytes()
    if not 0 < len(manifest_payload) <= 256 * 1024:
        raise ValueError("safe development manifest exceeds its size bound")
    _strict_json(manifest_payload)
    manifest = SafeDevelopmentManifest.model_validate_json(manifest_payload)
    if (
        examples_path.stat().st_size != manifest.examples_size_bytes
        or _sha256(examples_path) != manifest.examples_sha256
    ):
        raise ValueError("safe development example payload checksum mismatch")
    examples: list[RemediationExample] = []
    with examples_path.open("rb") as stream:
        for line in stream:
            if not line or len(line) > MAX_EXAMPLE_BYTES:
                raise ValueError("safe development JSONL row exceeds its bound")
            _strict_json(line)
            examples.append(RemediationExample.model_validate_json(line))
            if len(examples) > MAX_EXAMPLES:
                raise ValueError("safe development example count exceeds its bound")
    return SafeDevelopmentDataset(manifest=manifest, examples=tuple(examples))


__all__ = [
    "FROZEN_V03_DEDUPLICATED_MANIFEST_SHA256",
    "FROZEN_V03_RAW_MANIFEST_SHA256",
    "FrozenV03IIDMaterial",
    "RemediationExample",
    "SafeDevelopmentDataset",
    "SafeDevelopmentManifest",
    "TaskScopedStructuredFingerprint",
    "build_frozen_v03_iid_material",
    "build_safe_development_dataset",
    "build_safe_development_dataset_with_structured_fingerprints",
    "compact_context_for_projection",
    "load_safe_development_artifact",
    "renderer_visible_structured_fingerprint",
    "write_safe_development_artifact",
]
