"""Development-only leakage, provenance, augmentation, and shortcut audits."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

from pydantic import Field, StrictBool, model_validator

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


class ViewAudit(ContractModel):
    view: RemediationView
    example_count: int = Field(ge=1)
    group_count: int = Field(ge=1)
    prompt_checksum_count: int = Field(ge=1)
    task_counts: tuple[tuple[TaskName, int], ...]
    renderer_variant_count: int = Field(ge=0)
    evidence_removal_count: int = Field(ge=0)


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
    target_leakage = sum(
        item.compact_target in item.prompt_text or item.canonical_target_json in item.prompt_text
        for item in examples
    )
    invalid_evidence = sum(not _evidence_removal_is_valid(item) for item in examples)
    view_audits: list[ViewAudit] = []
    for view in dataset.manifest.views:
        rows = by_view[view]
        counts = Counter(item.task_name for item in rows)
        view_audits.append(
            ViewAudit(
                view=view,
                example_count=len(rows),
                group_count=len(group_ids[view]),
                prompt_checksum_count=len(prompt_hashes[view]),
                task_counts=tuple((task, counts[task]) for task in TaskName if counts[task]),
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
