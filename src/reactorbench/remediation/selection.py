"""Target-independent semantic checkpoint-selection subset freeze.

The v0.3 semantic composite is intentionally evaluated on a fixed 48-example IID
validation subset at every checkpoint.  Selection uses only model-visible or
rendering metadata: task, prompt checksum/size, prompt-local reference counts, surface
families, and augmentation kind.  Target text, labels, example IDs, record lineage,
and example checksums never participate in ranking or tie-breaking.  Identities and
checksums are recorded only after selection to bind the frozen result to its source.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from reactorbench.schemas.base import (
    ContractId,
    ContractModel,
    canonical_json_bytes,
    canonical_sha256,
)
from reactorbench.schemas.enums import SplitName, TaskName

from .config import RemediationView, V03Config, config_sha256
from .data import RemediationExample, SafeDevelopmentDataset

SELECTION_EXAMPLE_COUNT: Literal[48] = 48
CALIBRATION_SELECTION_EXAMPLE_COUNT: Literal[56] = 56
EXAMPLES_PER_TASK: Literal[8] = 8
EXAMPLES_PER_TASK_STRATUM: Literal[4] = 4
MAX_SELECTION_MANIFEST_BYTES = 2 * 1024 * 1024

Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[StrictInt, Field(ge=1)]


class ContextSizeStratum(StrEnum):
    LOWER = "lower_context_complexity"
    UPPER = "upper_context_complexity"


CONTEXT_STRATA: tuple[ContextSizeStratum, ...] = (
    ContextSizeStratum.LOWER,
    ContextSizeStratum.UPPER,
)


def _policy_descriptor() -> dict[str, object]:
    return {
        "policy_version": "0.3.0",
        "boundary": "iid_validation_only_no_test_or_golden",
        "selection_algorithm": "task_context_strata_surface_round_robin_v1",
        "example_count": SELECTION_EXAMPLE_COUNT,
        "examples_per_task": EXAMPLES_PER_TASK,
        "examples_per_task_stratum": EXAMPLES_PER_TASK_STRATUM,
        "tasks": [task.value for task in TaskName],
        "context_strata": [stratum.value for stratum in CONTEXT_STRATA],
        "ranking_fields": [
            "visible_fact_count",
            "counterfactual_visible_fact_count",
            "prompt_utf8_bytes",
            "prompt_sha256",
        ],
        "surface_round_robin_fields": [
            "template_family_id",
            "alias_family_id",
            "augmentation",
        ],
        "identity_fields_are_binding_only": True,
        "prohibited_ranking_fields": [
            "compact_target",
            "canonical_target_json",
            "classification_label",
            "example_id",
            "checksum_sha256",
            "source_record_ids",
            "parent_record_sha256",
            "group_id",
        ],
    }


SELECTION_POLICY_SHA256 = canonical_sha256(_policy_descriptor())


@dataclass(frozen=True, slots=True)
class _ObservableCandidate:
    example: RemediationExample
    task_name: TaskName
    prompt_sha256: str
    prompt_utf8_bytes: int
    visible_fact_count: int
    counterfactual_visible_fact_count: int
    template_family_id: str
    alias_family_id: str
    augmentation: Literal["none", "renderer_variant", "remove_decisive_evidence"]

    @property
    def complexity_key(self) -> tuple[int, int, int, str]:
        return (
            self.visible_fact_count + self.counterfactual_visible_fact_count,
            self.counterfactual_visible_fact_count,
            self.prompt_utf8_bytes,
            self.prompt_sha256,
        )

    @property
    def surface_key(self) -> tuple[str, str, str]:
        return (
            self.template_family_id,
            self.alias_family_id,
            self.augmentation,
        )

    def observable_payload(self) -> dict[str, object]:
        return {
            "task_name": self.task_name.value,
            "prompt_sha256": self.prompt_sha256,
            "prompt_utf8_bytes": self.prompt_utf8_bytes,
            "visible_fact_count": self.visible_fact_count,
            "counterfactual_visible_fact_count": self.counterfactual_visible_fact_count,
            "template_family_id": self.template_family_id,
            "alias_family_id": self.alias_family_id,
            "augmentation": self.augmentation,
        }


def _observable_candidate(example: RemediationExample) -> _ObservableCandidate:
    """Copy the complete target-independent selection boundary from one example."""

    if type(example) is not RemediationExample:
        raise TypeError("semantic selection requires exact RemediationExample objects")
    if (
        example.view is not RemediationView.IID_VALIDATION
        or example.source_split is not SplitName.IID_VALIDATION
    ):
        raise ValueError("semantic selection candidates must be IID validation only")
    if example.task_name is not example.compact_context.task_name:
        raise ValueError("selection candidate task differs from its compact context")
    prompt_bytes = example.prompt_text.encode("utf-8")
    if hashlib.sha256(prompt_bytes).hexdigest() != example.prompt_sha256:
        raise ValueError("selection candidate prompt checksum mismatch")
    return _ObservableCandidate(
        example=example,
        task_name=example.task_name,
        prompt_sha256=example.prompt_sha256,
        prompt_utf8_bytes=len(prompt_bytes),
        visible_fact_count=len(example.compact_context.visible_fact_refs),
        counterfactual_visible_fact_count=len(
            example.compact_context.counterfactual_visible_fact_refs
        ),
        template_family_id=example.template_family_id,
        alias_family_id=example.alias_family_id,
        augmentation=example.augmentation,
    )


def _observable_inventory_sha256(candidates: tuple[_ObservableCandidate, ...]) -> str:
    ordered = tuple(
        sorted(
            (candidate.observable_payload() for candidate in candidates),
            key=lambda payload: (
                list(TaskName).index(TaskName(str(payload["task_name"]))),
                str(payload["prompt_sha256"]),
            ),
        )
    )
    return canonical_sha256(ordered)


def _identity_inventory_sha256(examples: tuple[RemediationExample, ...]) -> str:
    return canonical_sha256(
        tuple(
            sorted(
                ((example.example_id, example.checksum_sha256) for example in examples),
                key=lambda item: item[0],
            )
        )
    )


def _round_robin_surface_pick(
    candidates: tuple[_ObservableCandidate, ...], *, count: int
) -> tuple[_ObservableCandidate, ...]:
    if type(count) is not int or count < 1 or len(candidates) < count:
        raise ValueError("surface-balanced selection has insufficient candidates")
    groups: dict[tuple[str, str, str], list[_ObservableCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.surface_key].append(candidate)
    for rows in groups.values():
        rows.sort(key=lambda candidate: candidate.prompt_sha256)
    selected: list[_ObservableCandidate] = []
    group_keys = tuple(sorted(groups))
    depth = 0
    while len(selected) < count:
        progressed = False
        for key in group_keys:
            rows = groups[key]
            if depth < len(rows):
                selected.append(rows[depth])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise RuntimeError("surface-balanced selection could not fill its fixed quota")
        depth += 1
    return tuple(selected)


def _select_task_candidates(
    candidates: tuple[_ObservableCandidate, ...],
) -> tuple[tuple[ContextSizeStratum, _ObservableCandidate], ...]:
    if len(candidates) < EXAMPLES_PER_TASK:
        raise ValueError("every semantic-selection task requires at least eight candidates")
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.complexity_key))
    midpoint = len(ordered) // 2
    lower = ordered[:midpoint]
    upper = ordered[midpoint:]
    if min(len(lower), len(upper)) < EXAMPLES_PER_TASK_STRATUM:
        raise ValueError("task context strata cannot each supply four candidates")
    selected = (
        *(
            (ContextSizeStratum.LOWER, candidate)
            for candidate in _round_robin_surface_pick(
                lower,
                count=EXAMPLES_PER_TASK_STRATUM,
            )
        ),
        *(
            (ContextSizeStratum.UPPER, candidate)
            for candidate in _round_robin_surface_pick(
                upper,
                count=EXAMPLES_PER_TASK_STRATUM,
            )
        ),
    )
    return tuple(
        sorted(
            selected,
            key=lambda item: (CONTEXT_STRATA.index(item[0]), item[1].prompt_sha256),
        )
    )


class SemanticSelectionEntry(ContractModel):
    selection_index: int = Field(strict=True, ge=0, lt=SELECTION_EXAMPLE_COUNT)
    task_name: TaskName
    context_stratum: ContextSizeStratum
    example_id: ContractId
    example_checksum_sha256: Sha256
    prompt_sha256: Sha256
    prompt_utf8_bytes: PositiveInt
    visible_fact_count: PositiveInt
    counterfactual_visible_fact_count: int = Field(strict=True, ge=0)
    template_family_id: ContractId
    alias_family_id: ContractId
    augmentation: Literal["none", "renderer_variant", "remove_decisive_evidence"]
    observable_selection_key_sha256: Sha256

    @model_validator(mode="after")
    def observable_key_matches(self) -> SemanticSelectionEntry:
        expected = canonical_sha256(
            {
                "task_name": self.task_name.value,
                "prompt_sha256": self.prompt_sha256,
                "prompt_utf8_bytes": self.prompt_utf8_bytes,
                "visible_fact_count": self.visible_fact_count,
                "counterfactual_visible_fact_count": self.counterfactual_visible_fact_count,
                "template_family_id": self.template_family_id,
                "alias_family_id": self.alias_family_id,
                "augmentation": self.augmentation,
            }
        )
        if self.observable_selection_key_sha256 != expected:
            raise ValueError("semantic-selection observable key checksum mismatch")
        return self


class SemanticSelectionManifest(ContractModel):
    manifest_version: Literal["0.3.0"] = "0.3.0"
    boundary: Literal["iid_validation_only_no_test_or_golden"] = (
        "iid_validation_only_no_test_or_golden"
    )
    selection_algorithm: Literal["task_context_strata_surface_round_robin_v1"] = (
        "task_context_strata_surface_round_robin_v1"
    )
    policy_sha256: Sha256
    v03_config_sha256: Sha256
    source_commit: StrictStr = Field(pattern=r"^[0-9a-f]{7,64}$")
    source_dataset_manifest_sha256: Sha256
    source_dataset_example_count: PositiveInt
    source_dataset_inventory_sha256: Sha256
    iid_validation_candidate_count: int = Field(strict=True, ge=SELECTION_EXAMPLE_COUNT)
    iid_validation_inventory_sha256: Sha256
    iid_validation_observable_inventory_sha256: Sha256
    selected_example_count: Literal[48] = SELECTION_EXAMPLE_COUNT
    examples_per_task: Literal[8] = EXAMPLES_PER_TASK
    examples_per_task_stratum: Literal[4] = EXAMPLES_PER_TASK_STRATUM
    counts_by_task: tuple[tuple[TaskName, Literal[8]], ...]
    counts_by_context_stratum: tuple[tuple[ContextSizeStratum, Literal[24]], ...]
    counts_by_task_and_stratum: tuple[tuple[TaskName, ContextSizeStratum, Literal[4]], ...]
    entries: tuple[SemanticSelectionEntry, ...] = Field(
        min_length=SELECTION_EXAMPLE_COUNT,
        max_length=SELECTION_EXAMPLE_COUNT,
    )
    selected_inventory_sha256: Sha256
    selected_observable_inventory_sha256: Sha256
    checksum_sha256: Sha256

    @field_validator("entries", mode="before")
    @classmethod
    def json_entry_array_becomes_tuple(cls, value: object) -> object:
        if type(value) is list:
            return tuple(value)
        return value

    @field_validator(
        "counts_by_task",
        "counts_by_context_stratum",
        "counts_by_task_and_stratum",
        mode="before",
    )
    @classmethod
    def json_count_arrays_become_tuples(cls, value: object) -> object:
        if type(value) is list:
            return tuple(tuple(row) if type(row) is list else row for row in value)
        return value

    @model_validator(mode="after")
    def freeze_and_checksum_match(self) -> SemanticSelectionManifest:
        if self.policy_sha256 != SELECTION_POLICY_SHA256:
            raise ValueError("semantic-selection policy checksum mismatch")
        if self.source_dataset_example_count < self.iid_validation_candidate_count:
            raise ValueError("IID candidate count exceeds the source dataset")
        expected_task_counts = tuple((task, EXAMPLES_PER_TASK) for task in TaskName)
        if self.counts_by_task != expected_task_counts:
            raise ValueError("semantic selection must contain exactly eight examples per task")
        expected_stratum_counts = tuple(
            (stratum, SELECTION_EXAMPLE_COUNT // len(CONTEXT_STRATA)) for stratum in CONTEXT_STRATA
        )
        if self.counts_by_context_stratum != expected_stratum_counts:
            raise ValueError("semantic selection must balance both context strata")
        expected_joint_counts = tuple(
            (task, stratum, EXAMPLES_PER_TASK_STRATUM)
            for task in TaskName
            for stratum in CONTEXT_STRATA
        )
        if self.counts_by_task_and_stratum != expected_joint_counts:
            raise ValueError("semantic selection task/context quotas differ from the freeze")

        task_order = {task: index for index, task in enumerate(TaskName)}
        stratum_order = {stratum: index for index, stratum in enumerate(CONTEXT_STRATA)}
        expected_entries = tuple(
            sorted(
                self.entries,
                key=lambda entry: (
                    task_order[entry.task_name],
                    stratum_order[entry.context_stratum],
                    entry.prompt_sha256,
                ),
            )
        )
        if self.entries != expected_entries or tuple(
            entry.selection_index for entry in self.entries
        ) != tuple(range(SELECTION_EXAMPLE_COUNT)):
            raise ValueError("semantic-selection entries must use canonical indexed order")
        if len({entry.example_id for entry in self.entries}) != SELECTION_EXAMPLE_COUNT:
            raise ValueError("semantic-selection example IDs must be unique")
        if len({entry.example_checksum_sha256 for entry in self.entries}) != (
            SELECTION_EXAMPLE_COUNT
        ):
            raise ValueError("semantic-selection example checksums must be unique")
        if len({(entry.task_name, entry.prompt_sha256) for entry in self.entries}) != (
            SELECTION_EXAMPLE_COUNT
        ):
            raise ValueError("semantic-selection task/prompt identities must be unique")

        selected_inventory = canonical_sha256(
            tuple((entry.example_id, entry.example_checksum_sha256) for entry in self.entries)
        )
        if self.selected_inventory_sha256 != selected_inventory:
            raise ValueError("semantic-selection selected inventory checksum mismatch")
        observable_inventory = canonical_sha256(
            tuple(entry.observable_selection_key_sha256 for entry in self.entries)
        )
        if self.selected_observable_inventory_sha256 != observable_inventory:
            raise ValueError("semantic-selection observable inventory checksum mismatch")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("semantic-selection manifest checksum mismatch")
        return self


class CalibrationSelectionEntry(ContractModel):
    """Identity-bound member of the disjoint post-selection calibration subset."""

    selection_index: int = Field(strict=True, ge=0, lt=CALIBRATION_SELECTION_EXAMPLE_COUNT)
    task_name: TaskName
    context_stratum: ContextSizeStratum
    example_id: ContractId
    example_checksum_sha256: Sha256
    prompt_sha256: Sha256
    observable_selection_key_sha256: Sha256


class CalibrationSelectionManifest(ContractModel):
    """Target-independent 56-row calibration freeze, disjoint from semantic selection."""

    manifest_version: Literal["0.3.1-targeted"] = "0.3.1-targeted"
    semantic_selection_manifest_sha256: Sha256
    iid_validation_inventory_sha256: Sha256
    selected_example_count: Literal[56] = CALIBRATION_SELECTION_EXAMPLE_COUNT
    entries: tuple[CalibrationSelectionEntry, ...] = Field(min_length=56, max_length=56)
    selected_inventory_sha256: Sha256
    checksum_sha256: Sha256

    @field_validator("entries", mode="before")
    @classmethod
    def calibration_entries_become_tuple(cls, value: object) -> object:
        return tuple(value) if type(value) is list else value

    @model_validator(mode="after")
    def disjoint_shape_and_checksum_match(self) -> CalibrationSelectionManifest:
        if tuple(item.selection_index for item in self.entries) != tuple(range(56)):
            raise ValueError("calibration entries must use canonical indexed order")
        if len({item.example_id for item in self.entries}) != 56:
            raise ValueError("calibration example IDs must be unique")
        expected_inventory = canonical_sha256(
            tuple((item.example_id, item.example_checksum_sha256) for item in self.entries)
        )
        if self.selected_inventory_sha256 != expected_inventory:
            raise ValueError("calibration selected inventory checksum mismatch")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("calibration selection manifest checksum mismatch")
        return self


def _entry(
    *,
    selection_index: int,
    context_stratum: ContextSizeStratum,
    candidate: _ObservableCandidate,
) -> SemanticSelectionEntry:
    observable = candidate.observable_payload()
    return SemanticSelectionEntry(
        selection_index=selection_index,
        task_name=candidate.task_name,
        context_stratum=context_stratum,
        example_id=candidate.example.example_id,
        example_checksum_sha256=candidate.example.checksum_sha256,
        prompt_sha256=candidate.prompt_sha256,
        prompt_utf8_bytes=candidate.prompt_utf8_bytes,
        visible_fact_count=candidate.visible_fact_count,
        counterfactual_visible_fact_count=candidate.counterfactual_visible_fact_count,
        template_family_id=candidate.template_family_id,
        alias_family_id=candidate.alias_family_id,
        augmentation=candidate.augmentation,
        observable_selection_key_sha256=canonical_sha256(observable),
    )


def _source_examples(dataset: SafeDevelopmentDataset) -> tuple[RemediationExample, ...]:
    if type(dataset) is not SafeDevelopmentDataset:
        raise TypeError("semantic selection requires an exact SafeDevelopmentDataset")
    examples = dataset.examples
    if not examples or any(type(example) is not RemediationExample for example in examples):
        raise TypeError("source dataset contains an invalid example inventory")
    if dataset.manifest.example_count != len(examples):
        raise ValueError("source dataset count differs from its manifest")
    identifiers = tuple(example.example_id for example in examples)
    checksums = tuple(example.checksum_sha256 for example in examples)
    if len(identifiers) != len(set(identifiers)) or len(checksums) != len(set(checksums)):
        raise ValueError("source example identities and checksums must be unique")
    inventory_sha256 = _identity_inventory_sha256(examples)
    if dataset.manifest.inventory_sha256 != inventory_sha256:
        raise ValueError("source dataset inventory checksum mismatch")
    return examples


def build_semantic_selection_manifest(
    dataset: SafeDevelopmentDataset,
    config: V03Config,
) -> SemanticSelectionManifest:
    """Select and freeze the exact target-independent v0.3 checkpoint subset."""

    if type(config) is not V03Config:
        raise TypeError("semantic selection requires an exact V03Config")
    if config.semantic_selection_example_limit != SELECTION_EXAMPLE_COUNT:
        raise ValueError("v0.3 semantic selection limit differs from 48")
    source_examples = _source_examples(dataset)
    iid_examples = tuple(
        example for example in source_examples if example.view is RemediationView.IID_VALIDATION
    )
    candidates = tuple(_observable_candidate(example) for example in iid_examples)
    prompt_identities = tuple(
        (candidate.task_name, candidate.prompt_sha256) for candidate in candidates
    )
    if len(prompt_identities) != len(set(prompt_identities)):
        raise ValueError("IID validation contains a duplicate task/prompt identity")

    by_task: dict[TaskName, tuple[_ObservableCandidate, ...]] = {
        task: tuple(candidate for candidate in candidates if candidate.task_name is task)
        for task in TaskName
    }
    if any(len(rows) < EXAMPLES_PER_TASK for rows in by_task.values()):
        raise ValueError("every required semantic-selection task needs eight IID candidates")
    selected: list[tuple[TaskName, ContextSizeStratum, _ObservableCandidate]] = []
    for task in TaskName:
        selected.extend(
            (task, stratum, candidate)
            for stratum, candidate in _select_task_candidates(by_task[task])
        )
    selected.sort(
        key=lambda item: (
            list(TaskName).index(item[0]),
            CONTEXT_STRATA.index(item[1]),
            item[2].prompt_sha256,
        )
    )
    entries = tuple(
        _entry(
            selection_index=index,
            context_stratum=stratum,
            candidate=candidate,
        )
        for index, (_task, stratum, candidate) in enumerate(selected)
    )
    task_counts = Counter(entry.task_name for entry in entries)
    stratum_counts = Counter(entry.context_stratum for entry in entries)
    joint_counts = Counter((entry.task_name, entry.context_stratum) for entry in entries)
    selected_inventory_sha256 = canonical_sha256(
        tuple((entry.example_id, entry.example_checksum_sha256) for entry in entries)
    )
    selected_observable_sha256 = canonical_sha256(
        tuple(entry.observable_selection_key_sha256 for entry in entries)
    )
    draft = SemanticSelectionManifest.model_construct(
        manifest_version="0.3.0",
        boundary="iid_validation_only_no_test_or_golden",
        selection_algorithm="task_context_strata_surface_round_robin_v1",
        policy_sha256=SELECTION_POLICY_SHA256,
        v03_config_sha256=config_sha256(config),
        source_commit=dataset.manifest.source_commit,
        source_dataset_manifest_sha256=dataset.manifest.checksum_sha256,
        source_dataset_example_count=len(source_examples),
        source_dataset_inventory_sha256=_identity_inventory_sha256(source_examples),
        iid_validation_candidate_count=len(iid_examples),
        iid_validation_inventory_sha256=_identity_inventory_sha256(iid_examples),
        iid_validation_observable_inventory_sha256=_observable_inventory_sha256(candidates),
        selected_example_count=SELECTION_EXAMPLE_COUNT,
        examples_per_task=EXAMPLES_PER_TASK,
        examples_per_task_stratum=EXAMPLES_PER_TASK_STRATUM,
        counts_by_task=tuple((task, task_counts[task]) for task in TaskName),
        counts_by_context_stratum=tuple(
            (stratum, stratum_counts[stratum]) for stratum in CONTEXT_STRATA
        ),
        counts_by_task_and_stratum=tuple(
            (task, stratum, joint_counts[(task, stratum)])
            for task in TaskName
            for stratum in CONTEXT_STRATA
        ),
        entries=entries,
        selected_inventory_sha256=selected_inventory_sha256,
        selected_observable_inventory_sha256=selected_observable_sha256,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return SemanticSelectionManifest(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
        checksum_sha256=checksum,
    )


def resolve_semantic_selection_examples(
    dataset: SafeDevelopmentDataset,
    manifest: SemanticSelectionManifest,
    config: V03Config,
) -> tuple[RemediationExample, ...]:
    """Verify a freeze against its complete source, then resolve canonical examples."""

    if type(manifest) is not SemanticSelectionManifest or type(config) is not V03Config:
        raise TypeError("semantic selection resolution requires exact manifest/config contracts")
    source_examples = _source_examples(dataset)
    iid_examples = tuple(
        example for example in source_examples if example.view is RemediationView.IID_VALIDATION
    )
    candidates = tuple(_observable_candidate(example) for example in iid_examples)
    if (
        manifest.v03_config_sha256 != config_sha256(config)
        or manifest.source_commit != dataset.manifest.source_commit
        or manifest.source_dataset_manifest_sha256 != dataset.manifest.checksum_sha256
        or manifest.source_dataset_example_count != len(source_examples)
        or manifest.source_dataset_inventory_sha256 != _identity_inventory_sha256(source_examples)
        or manifest.iid_validation_candidate_count != len(iid_examples)
        or manifest.iid_validation_inventory_sha256 != _identity_inventory_sha256(iid_examples)
        or manifest.iid_validation_observable_inventory_sha256
        != _observable_inventory_sha256(candidates)
    ):
        raise ValueError("semantic-selection manifest does not match its complete source")

    by_id = {example.example_id: example for example in iid_examples}
    resolved: list[RemediationExample] = []
    for entry in manifest.entries:
        example = by_id.get(entry.example_id)
        if example is None or example.checksum_sha256 != entry.example_checksum_sha256:
            raise ValueError("semantic-selection entry identity does not match the source")
        candidate = _observable_candidate(example)
        if (
            canonical_sha256(candidate.observable_payload())
            != entry.observable_selection_key_sha256
        ):
            raise ValueError("semantic-selection entry observable key does not match the source")
        resolved.append(example)
    expected_manifest = build_semantic_selection_manifest(dataset, config)
    if manifest != expected_manifest:
        raise ValueError(
            "semantic-selection manifest differs from the deterministic source selection"
        )
    return tuple(resolved)


def build_calibration_selection_manifest(
    dataset: SafeDevelopmentDataset,
    config: V03Config,
    semantic_manifest: SemanticSelectionManifest,
) -> CalibrationSelectionManifest:
    """Freeze the disjoint target-independent 56-row temperature-calibration subset."""

    if config.targeted_policy is None:
        raise ValueError("calibration selection is reserved for the targeted v0.3 policy")
    semantic_rows = resolve_semantic_selection_examples(dataset, semantic_manifest, config)
    semantic_ids = {item.example_id for item in semantic_rows}
    iid = tuple(
        item for item in _source_examples(dataset) if item.view is RemediationView.IID_VALIDATION
    )
    candidates = tuple(
        _observable_candidate(item) for item in iid if item.example_id not in semantic_ids
    )
    by_task = {
        task: tuple(item for item in candidates if item.task_name is task) for task in TaskName
    }
    selected: list[tuple[TaskName, ContextSizeStratum, _ObservableCandidate]] = []
    for task in TaskName:
        total = 6 if task is TaskName.COUNTERFACTUAL_COMPARE else 10
        per_stratum = total // 2
        rows = tuple(sorted(by_task[task], key=lambda item: item.complexity_key))
        midpoint = len(rows) // 2
        lower, upper = rows[:midpoint], rows[midpoint:]
        if min(len(lower), len(upper)) < per_stratum:
            raise ValueError("calibration selection cannot satisfy its disjoint context quota")
        selected.extend(
            (task, ContextSizeStratum.LOWER, item)
            for item in _round_robin_surface_pick(lower, count=per_stratum)
        )
        selected.extend(
            (task, ContextSizeStratum.UPPER, item)
            for item in _round_robin_surface_pick(upper, count=per_stratum)
        )
    selected.sort(
        key=lambda item: (
            list(TaskName).index(item[0]),
            CONTEXT_STRATA.index(item[1]),
            item[2].prompt_sha256,
        )
    )
    entries = tuple(
        CalibrationSelectionEntry(
            selection_index=index,
            task_name=task,
            context_stratum=stratum,
            example_id=candidate.example.example_id,
            example_checksum_sha256=candidate.example.checksum_sha256,
            prompt_sha256=candidate.prompt_sha256,
            observable_selection_key_sha256=canonical_sha256(candidate.observable_payload()),
        )
        for index, (task, stratum, candidate) in enumerate(selected)
    )
    if len(entries) != CALIBRATION_SELECTION_EXAMPLE_COUNT:
        raise RuntimeError("calibration selection filled the wrong number of rows")
    draft = CalibrationSelectionManifest.model_construct(
        semantic_selection_manifest_sha256=semantic_manifest.checksum_sha256,
        iid_validation_inventory_sha256=_identity_inventory_sha256(iid),
        entries=entries,
        selected_inventory_sha256=canonical_sha256(
            tuple((item.example_id, item.example_checksum_sha256) for item in entries)
        ),
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return CalibrationSelectionManifest(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}), checksum_sha256=checksum
    )


def resolve_calibration_selection_examples(
    dataset: SafeDevelopmentDataset,
    config: V03Config,
    semantic_manifest: SemanticSelectionManifest,
    calibration_manifest: CalibrationSelectionManifest,
) -> tuple[RemediationExample, ...]:
    """Rebuild and verify the disjoint calibration freeze before it is used."""

    expected = build_calibration_selection_manifest(dataset, config, semantic_manifest)
    if calibration_manifest != expected:
        raise ValueError(
            "calibration selection manifest differs from deterministic source selection"
        )
    semantic_ids = {
        item.example_id
        for item in resolve_semantic_selection_examples(dataset, semantic_manifest, config)
    }
    iid = {
        item.example_id: item
        for item in _source_examples(dataset)
        if item.view is RemediationView.IID_VALIDATION
    }
    resolved = tuple(
        iid[item.example_id] for item in calibration_manifest.entries if item.example_id in iid
    )
    if len(resolved) != 56 or semantic_ids & {item.example_id for item in resolved}:
        raise ValueError("calibration selection is missing or overlaps semantic selection")
    return resolved


def _strict_json(payload: bytes) -> object:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("semantic-selection manifest contains a duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"semantic-selection manifest contains non-finite JSON: {value}")

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )


def write_semantic_selection_manifest(
    manifest: SemanticSelectionManifest,
    path: Path,
) -> None:
    """Atomically create one non-overwriting canonical selection manifest."""

    if type(manifest) is not SemanticSelectionManifest or not isinstance(path, Path):
        raise TypeError("selection-manifest write requires exact manifest and Path")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("selection-manifest parent must be a regular directory")
    if path.exists() or path.is_symlink():
        raise FileExistsError("selection manifest must not overwrite an existing path")
    payload = canonical_json_bytes(manifest.model_dump(mode="json", round_trip=True)) + b"\n"
    if len(payload) > MAX_SELECTION_MANIFEST_BYTES:
        raise ValueError("selection manifest exceeds its byte bound")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("selection-manifest temporary path already exists")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise


def load_semantic_selection_manifest(
    path: Path,
    *,
    expected_checksum: str | None = None,
) -> SemanticSelectionManifest:
    """Load one bounded canonical manifest and optionally pin its checksum."""

    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise ValueError("selection manifest must be a regular non-symlink file")
    size = path.stat().st_size
    if not 0 < size <= MAX_SELECTION_MANIFEST_BYTES:
        raise ValueError("selection manifest size is outside its bound")
    payload = path.read_bytes()
    # The first parse enforces duplicate-key and non-finite rejection.  Pydantic's
    # JSON-mode validator then performs strict JSON-to-enum/tuple conversion without
    # enabling Python-mode scalar coercion.
    _strict_json(payload)
    manifest = SemanticSelectionManifest.model_validate_json(payload)
    canonical = canonical_json_bytes(manifest.model_dump(mode="json", round_trip=True)) + b"\n"
    if payload != canonical:
        raise ValueError("selection manifest is not canonical JSON")
    if expected_checksum is not None:
        if (
            type(expected_checksum) is not str
            or len(expected_checksum) != 64
            or any(character not in "0123456789abcdef" for character in expected_checksum)
        ):
            raise TypeError("expected selection-manifest checksum is invalid")
        if manifest.checksum_sha256 != expected_checksum:
            raise ValueError("selection manifest does not match the expected checksum")
    return manifest


__all__ = [
    "CALIBRATION_SELECTION_EXAMPLE_COUNT",
    "CONTEXT_STRATA",
    "EXAMPLES_PER_TASK",
    "EXAMPLES_PER_TASK_STRATUM",
    "SELECTION_EXAMPLE_COUNT",
    "SELECTION_POLICY_SHA256",
    "CalibrationSelectionEntry",
    "CalibrationSelectionManifest",
    "ContextSizeStratum",
    "SemanticSelectionEntry",
    "SemanticSelectionManifest",
    "build_calibration_selection_manifest",
    "build_semantic_selection_manifest",
    "load_semantic_selection_manifest",
    "resolve_calibration_selection_examples",
    "resolve_semantic_selection_examples",
    "write_semantic_selection_manifest",
]
