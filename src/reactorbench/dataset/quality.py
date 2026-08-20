"""Deterministic duplicate, overlap, shortcut, and provenance audits."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from reactorbench.schemas.base import ContractModel, canonical_json_bytes
from reactorbench.schemas.enums import SplitName, TaskName

from .content_guard import normalize_text

_REQUIRED_PROVENANCE = frozenset(
    {
        "dataset_version",
        "generator_commit",
        "scenario_schema_version",
        "renderer_version",
        "seed",
        "scenario_id",
        "plant_variant_id",
        "fault_family_ids",
        "template_family_ids",
        "split_name",
        "task_name",
    }
)
_TOKEN = re.compile(r"[a-z0-9_]+")
_TIME = re.compile(r"\b(?:t\+|tick\s*)\d+\b")
_NUMBER = re.compile(r"(?<![a-z])[-+]?\d+(?:\.\d+)?(?![a-z])")
_ALIAS = re.compile(r"\b(?:asset|component|node|measure|indicator)-[0-9a-f]{6,12}\b")
_CANONICAL_ID = re.compile(r"\baster[- ][a-z0-9._:-]+\b")


class QualityRecord(ContractModel):
    """Renderer output plus audit metadata; targets never come from text."""

    example_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
    split_name: SplitName
    text: str = Field(min_length=1, max_length=65_536)
    template_family_id: str = Field(min_length=1, max_length=96)
    alias_family_id: str = Field(min_length=1, max_length=96)
    target_labels: tuple[str, ...] = ()
    context_flags: tuple[str, ...] = ()
    provenance: dict[str, object]

    @field_validator("target_labels", "context_flags", mode="after")
    @classmethod
    def tuples_are_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("set-like audit fields must not contain duplicates")
        return tuple(sorted(values))


class TaskShortcutRecord(ContractModel):
    """One task target and its classified categorical audit features.

    Semantic context is reported separately from renderer-plan nuisance features, so
    intended evidence remains visible without being mislabeled as an accidental cue.
    """

    record_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    prompt_render_ids: tuple[
        Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")], ...
    ] = Field(min_length=1, max_length=2)
    task_name: TaskName
    template_family_id: str = Field(min_length=1, max_length=96)
    alias_family_id: str = Field(min_length=1, max_length=96)
    target_labels: tuple[str, ...] = ()
    context_flags: tuple[str, ...] = ()

    @field_validator("prompt_render_ids", "target_labels", mode="after")
    @classmethod
    def set_fields_are_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ValueError("task shortcut fields must contain unique non-empty strings")
        return tuple(sorted(values))

    @field_validator("context_flags", mode="after")
    @classmethod
    def context_flags_are_classified(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ValueError("task shortcut flags must contain unique non-empty strings")
        if any(not value.startswith(("semantic:", "corruption:")) for value in values):
            raise ValueError("task shortcut flags must declare semantic or corruption class")
        if sum(value.startswith("corruption:") for value in values) != 1:
            raise ValueError("task shortcut flags require exactly one corruption plan")
        return tuple(sorted(values))

    @model_validator(mode="after")
    def prompt_arity_matches_task(self) -> TaskShortcutRecord:
        expected = 2 if self.task_name is TaskName.COUNTERFACTUAL_COMPARE else 1
        if len(self.prompt_render_ids) != expected:
            raise ValueError("task shortcut prompt count does not match task shape")
        return self


class DuplicateGroup(ContractModel):
    fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    example_ids: tuple[str, ...]
    split_names: tuple[SplitName, ...]


class NgramOverlap(ContractModel):
    left_split: SplitName
    right_split: SplitName
    n: int = Field(ge=2, le=8)
    left_unique: int = Field(ge=0)
    right_unique: int = Field(ge=0)
    shared_unique: int = Field(ge=0)
    jaccard: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class ShortcutFinding(ContractModel):
    task_name: TaskName
    feature_class: Literal["renderer_nuisance"] = "renderer_nuisance"
    feature_name: str
    feature_value: str
    sole_target: str
    support: int = Field(ge=1)


class ShortcutContingency(ContractModel):
    """Full task-scoped feature-to-target counts, including non-failures."""

    task_name: TaskName
    feature_class: Literal["renderer_nuisance", "semantic_context"]
    feature_name: str
    feature_value: str
    target_counts: tuple[tuple[str, int], ...]
    support: int = Field(ge=1)

    @model_validator(mode="after")
    def counts_are_complete_and_canonical(self) -> ShortcutContingency:
        if not self.target_counts or any(count <= 0 for _, count in self.target_counts):
            raise ValueError("shortcut contingency counts must be positive and nonempty")
        if self.target_counts != tuple(sorted(self.target_counts)):
            raise ValueError("shortcut contingency targets must use canonical order")
        if len({target for target, _ in self.target_counts}) != len(self.target_counts):
            raise ValueError("shortcut contingency targets must be unique")
        if self.support != sum(count for _, count in self.target_counts):
            raise ValueError("shortcut contingency support must equal its target counts")
        return self


class TargetTextFinding(ContractModel):
    example_id: str
    target_label: str


class ProvenanceIssue(ContractModel):
    example_id: str
    missing_fields: tuple[str, ...]


class AuditedRecordIdentity(ContractModel):
    example_id: str
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AuditedTaskRecordIdentity(ContractModel):
    record_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class QualityReport(ContractModel):
    report_version: str = "0.1.0"
    record_count: int = Field(ge=0)
    audited_records: tuple[AuditedRecordIdentity, ...]
    task_record_count: int = Field(ge=0)
    audited_task_records: tuple[AuditedTaskRecordIdentity, ...]
    exact_duplicates: tuple[DuplicateGroup, ...]
    skeleton_duplicates: tuple[DuplicateGroup, ...]
    forbidden_skeleton_duplicates: tuple[DuplicateGroup, ...]
    ngram_overlaps: tuple[NgramOverlap, ...]
    shortcut_contingencies: tuple[ShortcutContingency, ...]
    shortcut_findings: tuple[ShortcutFinding, ...]
    target_text_findings: tuple[TargetTextFinding, ...]
    provenance_issues: tuple[ProvenanceIssue, ...]
    passed: bool
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def checksum_matches_report(self) -> QualityReport:
        identifiers = tuple(record.example_id for record in self.audited_records)
        if self.record_count != len(self.audited_records):
            raise ValueError("quality report record count mismatch")
        if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(set(identifiers)):
            raise ValueError("quality report audited records must be unique and sorted")
        task_identifiers = tuple(record.record_id for record in self.audited_task_records)
        if self.task_record_count != len(self.audited_task_records):
            raise ValueError("quality report task-record count mismatch")
        if task_identifiers != tuple(sorted(task_identifiers)) or len(task_identifiers) != len(
            set(task_identifiers)
        ):
            raise ValueError("quality report audited task records must be unique and sorted")
        expected = hashlib.sha256(
            canonical_json_bytes(
                self.model_dump(mode="json", exclude={"report_sha256"}, round_trip=True)
            )
        ).hexdigest()
        if self.report_sha256 != expected:
            raise ValueError("quality report checksum mismatch")
        return self


def text_skeleton(value: str) -> str:
    """Normalize lexical surface details while preserving event structure."""

    normalized = normalize_text(value)
    normalized = _TIME.sub("<time>", normalized)
    normalized = _ALIAS.sub("<alias>", normalized)
    normalized = _CANONICAL_ID.sub("<alias>", normalized)
    normalized = _NUMBER.sub("<number>", normalized)
    return " ".join(normalized.split())


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _duplicate_groups(
    records: tuple[QualityRecord, ...], *, skeleton: bool
) -> tuple[DuplicateGroup, ...]:
    grouped: dict[str, list[QualityRecord]] = defaultdict(list)
    for record in records:
        material = text_skeleton(record.text) if skeleton else record.text
        grouped[_sha(material)].append(record)
    findings: list[DuplicateGroup] = []
    for digest, siblings in sorted(grouped.items()):
        if len(siblings) < 2:
            continue
        findings.append(
            DuplicateGroup(
                fingerprint_sha256=digest,
                example_ids=tuple(sorted(record.example_id for record in siblings)),
                split_names=tuple(sorted({record.split_name for record in siblings}, key=str)),
            )
        )
    return tuple(findings)


def _forbidden_skeleton_groups(
    records: tuple[QualityRecord, ...],
    groups: tuple[DuplicateGroup, ...],
) -> tuple[DuplicateGroup, ...]:
    """Return skeleton overlap that defeats an explicit renderer-family holdout.

    Repeated causal shapes with different values are expected in a synthetic corpus and
    remain fully reported. They are a release blocker only when the same template family
    leaks into ``template_test`` or the same alias family leaks into ``component_test``.
    Exact text duplicates remain an unconditional failure.
    """

    by_id = {record.example_id: record for record in records}
    forbidden: list[DuplicateGroup] = []
    for group in groups:
        siblings = tuple(by_id[example_id] for example_id in group.example_ids)
        template_holdout = tuple(
            record for record in siblings if record.split_name is SplitName.TEMPLATE_TEST
        )
        component_holdout = tuple(
            record for record in siblings if record.split_name is SplitName.COMPONENT_TEST
        )
        template_leak = any(
            heldout.template_family_id == other.template_family_id
            for heldout in template_holdout
            for other in siblings
            if other.split_name is not SplitName.TEMPLATE_TEST
        )
        alias_leak = any(
            heldout.alias_family_id == other.alias_family_id
            for heldout in component_holdout
            for other in siblings
            if other.split_name is not SplitName.COMPONENT_TEST
        )
        if template_leak or alias_leak:
            forbidden.append(group)
    return tuple(forbidden)


def _ngrams(text: str, n: int) -> set[str]:
    tokens = _TOKEN.findall(normalize_text(text))
    return {_sha(" ".join(tokens[index : index + n])) for index in range(len(tokens) - n + 1)}


def _ngram_report(
    records: tuple[QualityRecord, ...], n_values: tuple[int, ...]
) -> tuple[NgramOverlap, ...]:
    splits = tuple(sorted({record.split_name for record in records}, key=str))
    by_split: dict[SplitName, tuple[QualityRecord, ...]] = {
        split: tuple(record for record in records if record.split_name is split) for split in splits
    }
    findings: list[NgramOverlap] = []
    for n in n_values:
        grams = {
            split: set().union(*(_ngrams(record.text, n) for record in split_records))
            if split_records
            else set()
            for split, split_records in by_split.items()
        }
        for left_index, left in enumerate(splits):
            for right in splits[left_index + 1 :]:
                shared = grams[left] & grams[right]
                union = grams[left] | grams[right]
                findings.append(
                    NgramOverlap(
                        left_split=left,
                        right_split=right,
                        n=n,
                        left_unique=len(grams[left]),
                        right_unique=len(grams[right]),
                        shared_unique=len(shared),
                        jaccard=(len(shared) / len(union)) if union else 0.0,
                    )
                )
    return tuple(findings)


def _target_key(record: QualityRecord | TaskShortcutRecord) -> str:
    return "+".join(record.target_labels) if record.target_labels else "<none>"


def _fallback_task_records(records: tuple[QualityRecord, ...]) -> tuple[TaskShortcutRecord, ...]:
    fallback: list[TaskShortcutRecord] = []
    for record in records:
        raw_task = record.provenance.get("task_name")
        if not isinstance(raw_task, str):
            raise ValueError(
                "implicit task audit requires one scalar provenance task_name per render"
            )
        task_name = TaskName(raw_task)
        corruption_flags = tuple(
            flag for flag in record.context_flags if flag.startswith("corruption:")
        )
        context_flags = (
            record.context_flags
            if corruption_flags
            else tuple(sorted((*record.context_flags, "corruption:none")))
        )
        fallback.append(
            TaskShortcutRecord(
                record_id=record.example_id,
                prompt_render_ids=(record.example_id,),
                task_name=task_name,
                template_family_id=record.template_family_id,
                alias_family_id=record.alias_family_id,
                target_labels=record.target_labels,
                context_flags=context_flags,
            )
        )
    return tuple(fallback)


def _shortcut_report(
    records: tuple[TaskShortcutRecord, ...],
) -> tuple[tuple[ShortcutContingency, ...], tuple[ShortcutFinding, ...]]:
    values: dict[
        tuple[TaskName, Literal["renderer_nuisance", "semantic_context"], str, str],
        list[TaskShortcutRecord],
    ] = defaultdict(list)
    global_targets: dict[TaskName, set[str]] = defaultdict(set)
    for record in records:
        global_targets[record.task_name].add(_target_key(record))
        values[
            (
                record.task_name,
                "renderer_nuisance",
                "template_family_id",
                record.template_family_id,
            )
        ].append(record)
        corruption_plan = next(
            flag for flag in record.context_flags if flag.startswith("corruption:")
        )
        nuisance_interactions = (
            (
                "template_alias_plan",
                f"{record.template_family_id}|{record.alias_family_id}",
            ),
            (
                "template_corruption_plan",
                f"{record.template_family_id}|{corruption_plan}",
            ),
            (
                "alias_corruption_plan",
                f"{record.alias_family_id}|{corruption_plan}",
            ),
            (
                "renderer_plan",
                f"{record.template_family_id}|{record.alias_family_id}|{corruption_plan}",
            ),
        )
        for feature_name, feature_value in nuisance_interactions:
            values[(record.task_name, "renderer_nuisance", feature_name, feature_value)].append(
                record
            )
        values[
            (
                record.task_name,
                "renderer_nuisance",
                "alias_family_id",
                record.alias_family_id,
            )
        ].append(record)
        for flag in record.context_flags:
            feature_class: Literal["renderer_nuisance", "semantic_context"] = (
                "semantic_context" if flag.startswith("semantic:") else "renderer_nuisance"
            )
            values[(record.task_name, feature_class, "context_flag", flag)].append(record)
    contingencies: list[ShortcutContingency] = []
    findings: list[ShortcutFinding] = []
    for (task_name, feature_class, feature_name, feature_value), supporting in sorted(
        values.items(),
        key=lambda item: (item[0][0].value, item[0][1], item[0][2], item[0][3]),
    ):
        counts: defaultdict[str, int] = defaultdict(int)
        for record in supporting:
            counts[_target_key(record)] += 1
        target_counts = tuple(sorted(counts.items()))
        contingencies.append(
            ShortcutContingency(
                task_name=task_name,
                feature_class=feature_class,
                feature_name=feature_name,
                feature_value=feature_value,
                target_counts=target_counts,
                support=len(supporting),
            )
        )
        if (
            feature_class == "renderer_nuisance"
            and len(supporting) >= 2
            and len(target_counts) == 1
            and len(global_targets[task_name]) >= 2
        ):
            findings.append(
                ShortcutFinding(
                    task_name=task_name,
                    feature_class="renderer_nuisance",
                    feature_name=feature_name,
                    feature_value=feature_value,
                    sole_target=target_counts[0][0],
                    support=len(supporting),
                )
            )
    return tuple(contingencies), tuple(findings)


def _target_text_report(records: tuple[QualityRecord, ...]) -> tuple[TargetTextFinding, ...]:
    findings: list[TargetTextFinding] = []
    for record in records:
        normalized_text = normalize_text(record.text)
        for label in record.target_labels:
            candidates = {normalize_text(label), normalize_text(label.replace("_", " "))}
            if any(
                re.search(rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])", normalized_text)
                for candidate in candidates
                if candidate
            ):
                findings.append(TargetTextFinding(example_id=record.example_id, target_label=label))
    return tuple(findings)


def _provenance_report(records: tuple[QualityRecord, ...]) -> tuple[ProvenanceIssue, ...]:
    findings: list[ProvenanceIssue] = []
    for record in records:
        missing = tuple(sorted(_REQUIRED_PROVENANCE - record.provenance.keys()))
        blank = tuple(
            sorted(
                key
                for key in _REQUIRED_PROVENANCE & record.provenance.keys()
                if record.provenance[key] is None or record.provenance[key] == ""
            )
        )
        if missing or blank:
            findings.append(
                ProvenanceIssue(
                    example_id=record.example_id,
                    missing_fields=tuple(sorted({*missing, *blank})),
                )
            )
    return tuple(findings)


def audit_quality(
    records: Iterable[QualityRecord],
    *,
    task_records: Iterable[TaskShortcutRecord] | None = None,
    n_values: tuple[int, ...] = (3, 4, 5),
) -> QualityReport:
    """Return the exact deterministic candidate-corpus audit report."""

    canonical = tuple(sorted(records, key=lambda record: record.example_id))
    if len({record.example_id for record in canonical}) != len(canonical):
        raise ValueError("quality records require unique example IDs")
    if not n_values or any(type(n) is not int or not 2 <= n <= 8 for n in n_values):
        raise ValueError("n_values must contain unique integers in [2, 8]")
    if len(set(n_values)) != len(n_values):
        raise ValueError("n_values must not contain duplicates")
    exact = _duplicate_groups(canonical, skeleton=False)
    skeleton = _duplicate_groups(canonical, skeleton=True)
    forbidden_skeleton = _forbidden_skeleton_groups(canonical, skeleton)
    canonical_tasks = tuple(
        sorted(
            _fallback_task_records(canonical) if task_records is None else task_records,
            key=lambda record: record.record_id,
        )
    )
    if len({record.record_id for record in canonical_tasks}) != len(canonical_tasks):
        raise ValueError("task shortcut records require unique record IDs")
    render_ids = {record.example_id for record in canonical}
    if any(
        render_id not in render_ids
        for record in canonical_tasks
        for render_id in record.prompt_render_ids
    ):
        raise ValueError("task shortcut records must reference audited rendered candidates")
    contingencies, shortcuts = _shortcut_report(canonical_tasks)
    target_text = _target_text_report(canonical)
    provenance = _provenance_report(canonical)
    audited = tuple(
        AuditedRecordIdentity(
            example_id=record.example_id,
            text_sha256=hashlib.sha256(record.text.encode("utf-8")).hexdigest(),
        )
        for record in canonical
    )
    audited_tasks = tuple(
        AuditedTaskRecordIdentity(
            record_id=record.record_id,
            record_sha256=hashlib.sha256(
                canonical_json_bytes(record.model_dump(mode="json", round_trip=True))
            ).hexdigest(),
        )
        for record in canonical_tasks
    )
    ngrams = _ngram_report(canonical, n_values)
    passed = not any((exact, forbidden_skeleton, shortcuts, target_text, provenance))
    payload = {
        "report_version": "0.1.0",
        "record_count": len(canonical),
        "audited_records": tuple(item.model_dump(mode="json") for item in audited),
        "task_record_count": len(canonical_tasks),
        "audited_task_records": tuple(item.model_dump(mode="json") for item in audited_tasks),
        "exact_duplicates": tuple(item.model_dump(mode="json") for item in exact),
        "skeleton_duplicates": tuple(item.model_dump(mode="json") for item in skeleton),
        "forbidden_skeleton_duplicates": tuple(
            item.model_dump(mode="json") for item in forbidden_skeleton
        ),
        "ngram_overlaps": tuple(item.model_dump(mode="json") for item in ngrams),
        "shortcut_contingencies": tuple(item.model_dump(mode="json") for item in contingencies),
        "shortcut_findings": tuple(item.model_dump(mode="json") for item in shortcuts),
        "target_text_findings": tuple(item.model_dump(mode="json") for item in target_text),
        "provenance_issues": tuple(item.model_dump(mode="json") for item in provenance),
        "passed": passed,
    }
    checksum = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return QualityReport(
        report_version="0.1.0",
        record_count=len(canonical),
        audited_records=audited,
        task_record_count=len(canonical_tasks),
        audited_task_records=audited_tasks,
        exact_duplicates=exact,
        skeleton_duplicates=skeleton,
        forbidden_skeleton_duplicates=forbidden_skeleton,
        ngram_overlaps=ngrams,
        shortcut_contingencies=contingencies,
        shortcut_findings=shortcuts,
        target_text_findings=target_text,
        provenance_issues=provenance,
        passed=passed,
        report_sha256=checksum,
    )
