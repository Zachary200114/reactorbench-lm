"""Development-only leakage, provenance, augmentation, and shortcut audits."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from enum import Enum
from itertools import combinations

from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr, model_validator

from reactorbench.dataset.content_guard import normalize_text
from reactorbench.dataset.contracts import PromptEvidenceTarget
from reactorbench.evaluation.compact import parse_compact_target
from reactorbench.schemas.base import ContractModel, canonical_sha256
from reactorbench.schemas.enums import (
    ActionLabel,
    DiagnosisStatus,
    TaskName,
)
from reactorbench.schemas.target import (
    FaultDiagnosisTarget,
    IncidentSummaryTarget,
    NextActionTarget,
)

from .config import RemediationView
from .data import RemediationExample, SafeDevelopmentDataset

_LEAK_SENSITIVE_TARGET_FIELDS = frozenset(
    {
        "diagnosis_status",
        "fault_labels",
        "immediate_action",
        "abstention_reason",
        "next_event_type",
    }
)
_LABEL_SEPARATOR = re.compile(r"[_-]+")
_CLASSIFICATION_TASKS = (
    TaskName.CONTINUE_LOG,
    TaskName.FAULT_FAMILY,
    TaskName.NEXT_ACTION,
)


class ClassInventory(ContractModel):
    """Canonical, report-only class distribution for one classification task."""

    task_name: TaskName
    counts: tuple[tuple[StrictStr, StrictInt], ...] = Field(min_length=1)
    total: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def inventory_is_canonical(self) -> ClassInventory:
        if self.task_name not in _CLASSIFICATION_TASKS:
            raise ValueError("class inventory is limited to classification tasks")
        labels = tuple(label for label, _count in self.counts)
        if labels != tuple(sorted(labels)) or len(labels) != len(set(labels)):
            raise ValueError("class inventory labels must be unique and sorted")
        if any(not label or type(count) is not int or count < 1 for label, count in self.counts):
            raise ValueError("class inventory entries must have positive exact counts")
        if sum(count for _label, count in self.counts) != self.total:
            raise ValueError("class inventory counts do not sum to their total")
        return self


class ViewAudit(ContractModel):
    view: RemediationView
    example_count: int = Field(ge=1)
    group_count: int = Field(ge=1)
    prompt_checksum_count: int = Field(ge=1)
    task_counts: tuple[tuple[TaskName, int], ...]
    class_inventories: tuple[ClassInventory, ...]
    renderer_variant_count: int = Field(ge=0)
    evidence_removal_count: int = Field(ge=0)

    @model_validator(mode="after")
    def counts_are_self_consistent(self) -> ViewAudit:
        task_names = tuple(task for task, _count in self.task_counts)
        expected_task_names = tuple(task for task in TaskName if task in task_names)
        if task_names != expected_task_names or len(task_names) != len(set(task_names)):
            raise ValueError("view task counts must be unique and canonically ordered")
        if any(type(count) is not int or count < 1 for _task, count in self.task_counts):
            raise ValueError("view task counts must be positive exact integers")
        task_counts = dict(self.task_counts)
        expected_class_tasks = tuple(task for task in _CLASSIFICATION_TASKS if task in task_counts)
        observed_class_tasks = tuple(item.task_name for item in self.class_inventories)
        if observed_class_tasks != expected_class_tasks:
            raise ValueError("view class inventories must match canonical task support")
        if any(item.total != task_counts[item.task_name] for item in self.class_inventories):
            raise ValueError("view class inventories do not cover their task counts")
        return self


class DevelopmentAuditReport(ContractModel):
    report_version: str = "0.3.0"
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    views: tuple[ViewAudit, ...]
    total_examples: int = Field(ge=1)
    exact_prompt_duplicate_count: int = Field(ge=0)
    cross_view_group_overlap_count: int = Field(ge=0)
    cross_view_prompt_overlap_count: int = Field(ge=0)
    target_text_leakage_count: int = Field(ge=0)
    invalid_evidence_removal_count: int = Field(ge=0)
    parent_checksum_overlap_count: int = Field(ge=0)
    template_label_shortcut_count: int = Field(ge=0)
    alias_label_shortcut_count: int = Field(ge=0)
    passed: StrictBool
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def result_and_checksum_match(self) -> DevelopmentAuditReport:
        findings = (
            self.exact_prompt_duplicate_count,
            self.cross_view_group_overlap_count,
            self.cross_view_prompt_overlap_count,
            self.target_text_leakage_count,
            self.invalid_evidence_removal_count,
            self.parent_checksum_overlap_count,
            self.template_label_shortcut_count,
            self.alias_label_shortcut_count,
        )
        if self.passed is not all(value == 0 for value in findings):
            raise ValueError("development audit pass state differs from its findings")
        if sum(view.example_count for view in self.views) != self.total_examples:
            raise ValueError("development audit view counts do not cover the artifact")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("development audit checksum mismatch")
        return self


def _intersection_count(values: dict[RemediationView, set[str]]) -> int:
    overlaps: set[str] = set()
    for first, second in combinations(values, 2):
        overlaps.update(values[first] & values[second])
    return len(overlaps)


def _leak_sensitive_target_labels(example: RemediationExample) -> tuple[str, ...]:
    """Return answer labels whose surface form must not appear in the prompt.

    This intentionally mirrors the field-sensitive Phase 3 audit. Evidence-slot,
    subsystem, trend, and operating-mode labels describe prompt-visible evidence or
    context and may legitimately occur in the input. Diagnosis, fault, action,
    abstention, and predicted-next-event labels are answer-bearing across the task
    contracts, including nested counterfactual conclusions, and are therefore scanned.
    """

    target = parse_compact_target(example.compact_target, context=example.compact_context)
    labels: set[str] = set()

    def visit(value: object, field_name: str | None = None) -> None:
        if isinstance(value, Enum):
            if field_name in _LEAK_SENSITIVE_TARGET_FIELDS:
                labels.add(str(value.value))
            return
        if isinstance(value, BaseModel):
            for name in type(value).model_fields:
                visit(getattr(value, name), name)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                visit(item, field_name)

    visit(target)
    return tuple(sorted(labels))


def _normalized_label_surface(value: str) -> str:
    """Canonicalize case and interchangeable label separators for matching."""

    return " ".join(_LABEL_SEPARATOR.sub(" ", normalize_text(value)).split())


def _label_surface_in_normalized_prompt(normalized_prompt: str, label: str) -> bool:
    normalized_label = _normalized_label_surface(label)
    if not normalized_label:
        return False
    return (
        re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_label)}(?![a-z0-9])",
            normalized_prompt,
        )
        is not None
    )


def _has_target_text_leakage(example: RemediationExample) -> bool:
    if (
        example.compact_target in example.prompt_text
        or example.canonical_target_json in example.prompt_text
    ):
        return True
    normalized_prompt = _normalized_label_surface(example.prompt_text)
    return any(
        _label_surface_in_normalized_prompt(normalized_prompt, label)
        for label in _leak_sensitive_target_labels(example)
    )


def _evidence_removal_is_valid(example: RemediationExample) -> bool:
    if example.augmentation != "remove_decisive_evidence":
        return True
    target = parse_compact_target(example.compact_target, context=example.compact_context)
    if type(target) is FaultDiagnosisTarget:
        return (
            target.diagnosis_status is DiagnosisStatus.UNRESOLVED
            and not target.fault_labels
            and target.abstention_reason is not None
        )
    if type(target) is NextActionTarget:
        return target.immediate_action is ActionLabel.INSUFFICIENT_EVIDENCE
    if type(target) is PromptEvidenceTarget:
        return not target.fact_refs and not target.evidence_slots
    if type(target) is IncidentSummaryTarget:
        return (
            target.diagnosis_status is DiagnosisStatus.UNRESOLVED
            and not target.fault_labels
            and target.immediate_action is ActionLabel.INSUFFICIENT_EVIDENCE
            and target.abstention_reason is not None
        )
    return False


def _shortcut_count(examples: tuple[RemediationExample, ...], *, attribute: str) -> int:
    """Count classification tasks where one surface family determines every label.

    A family is a concerning deterministic shortcut only when each family value maps
    to exactly one label *and* at least two labels and two family values are present.
    Renderer augmentation should make this count zero on the training view.
    """

    count = 0
    for task in (TaskName.FAULT_FAMILY, TaskName.NEXT_ACTION, TaskName.CONTINUE_LOG):
        rows = tuple(
            item
            for item in examples
            if item.view is RemediationView.IID_TRAIN
            and item.task_name is task
            and item.classification_label is not None
        )
        if not rows:
            continue
        by_family: dict[str, set[str]] = defaultdict(set)
        for item in rows:
            by_family[str(getattr(item, attribute))].add(str(item.classification_label))
        labels = {str(item.classification_label) for item in rows}
        if (
            len(by_family) > 1
            and len(labels) > 1
            and all(len(family_labels) == 1 for family_labels in by_family.values())
        ):
            count += 1
    return count


def audit_safe_development_dataset(dataset: SafeDevelopmentDataset) -> DevelopmentAuditReport:
    """Fail closed on cross-view overlap, leakage, invalid augmentation, or shortcuts."""

    if type(dataset) is not SafeDevelopmentDataset:
        raise TypeError("development audit requires an exact safe dataset")
    examples = dataset.examples
    by_view: dict[RemediationView, tuple[RemediationExample, ...]] = {
        view: tuple(item for item in examples if item.view is view)
        for view in dataset.manifest.views
    }
    group_ids = {view: {item.group_id for item in rows} for view, rows in by_view.items()}
    prompt_hashes = {view: {item.prompt_sha256 for item in rows} for view, rows in by_view.items()}
    parent_hashes = {
        view: {item.parent_record_sha256 for item in rows} for view, rows in by_view.items()
    }
    # The same model input intentionally feeds several different task heads. Only a
    # duplicate inside the same task is a shortcut/leakage finding.
    prompt_counts = Counter((item.task_name, item.prompt_sha256) for item in examples)
    exact_duplicates = sum(count - 1 for count in prompt_counts.values() if count > 1)
    target_leakage = sum(_has_target_text_leakage(item) for item in examples)
    invalid_evidence = sum(not _evidence_removal_is_valid(item) for item in examples)
    view_audits: list[ViewAudit] = []
    for view in dataset.manifest.views:
        rows = by_view[view]
        counts = Counter(item.task_name for item in rows)
        class_inventories: list[ClassInventory] = []
        for task in _CLASSIFICATION_TASKS:
            task_rows = tuple(item for item in rows if item.task_name is task)
            if not task_rows:
                continue
            if any(item.classification_label is None for item in task_rows):
                raise ValueError("classification task is missing a derived class label")
            class_counts = Counter(
                item.classification_label
                for item in task_rows
                if item.classification_label is not None
            )
            class_inventories.append(
                ClassInventory(
                    task_name=task,
                    counts=tuple(sorted(class_counts.items())),
                    total=len(task_rows),
                )
            )
        view_audits.append(
            ViewAudit(
                view=view,
                example_count=len(rows),
                group_count=len(group_ids[view]),
                prompt_checksum_count=len(prompt_hashes[view]),
                task_counts=tuple((task, counts[task]) for task in TaskName if counts[task]),
                class_inventories=tuple(class_inventories),
                renderer_variant_count=sum(
                    item.augmentation == "renderer_variant" for item in rows
                ),
                evidence_removal_count=sum(
                    item.augmentation == "remove_decisive_evidence" for item in rows
                ),
            )
        )
    findings = {
        "exact_prompt_duplicate_count": exact_duplicates,
        "cross_view_group_overlap_count": _intersection_count(group_ids),
        "cross_view_prompt_overlap_count": _intersection_count(prompt_hashes),
        "target_text_leakage_count": target_leakage,
        "invalid_evidence_removal_count": invalid_evidence,
        "parent_checksum_overlap_count": _intersection_count(parent_hashes),
        "template_label_shortcut_count": _shortcut_count(examples, attribute="template_family_id"),
        "alias_label_shortcut_count": _shortcut_count(examples, attribute="alias_family_id"),
    }
    passed = all(value == 0 for value in findings.values())
    draft = DevelopmentAuditReport.model_construct(
        dataset_manifest_sha256=dataset.manifest.checksum_sha256,
        views=tuple(view_audits),
        total_examples=len(examples),
        exact_prompt_duplicate_count=findings["exact_prompt_duplicate_count"],
        cross_view_group_overlap_count=findings["cross_view_group_overlap_count"],
        cross_view_prompt_overlap_count=findings["cross_view_prompt_overlap_count"],
        target_text_leakage_count=findings["target_text_leakage_count"],
        invalid_evidence_removal_count=findings["invalid_evidence_removal_count"],
        parent_checksum_overlap_count=findings["parent_checksum_overlap_count"],
        template_label_shortcut_count=findings["template_label_shortcut_count"],
        alias_label_shortcut_count=findings["alias_label_shortcut_count"],
        passed=passed,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return DevelopmentAuditReport(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
        checksum_sha256=checksum,
    )


__all__ = ["DevelopmentAuditReport", "ViewAudit", "audit_safe_development_dataset"]
