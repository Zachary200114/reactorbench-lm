"""Tests for the frozen target-independent v0.3 semantic subset."""

from __future__ import annotations

import hashlib
import stat
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from reactorbench.evaluation.compact import CompactTargetContext
from reactorbench.remediation.config import RemediationView, V03Config, load_v03_config
from reactorbench.remediation.data import (
    RemediationExample,
    SafeDevelopmentDataset,
    SafeDevelopmentManifest,
)
from reactorbench.remediation.selection import (
    CONTEXT_STRATA,
    EXAMPLES_PER_TASK,
    EXAMPLES_PER_TASK_STRATUM,
    MAX_SELECTION_MANIFEST_BYTES,
    SELECTION_EXAMPLE_COUNT,
    ContextSizeStratum,
    SemanticSelectionManifest,
    build_semantic_selection_manifest,
    load_semantic_selection_manifest,
    resolve_semantic_selection_examples,
    write_semantic_selection_manifest,
)
from reactorbench.schemas.base import canonical_sha256
from reactorbench.schemas.enums import SplitName, TaskName

ROOT = Path(__file__).resolve().parents[2]
V03_CONFIG_PATH = ROOT / "configs/experiments/phase6-remediation-v0.3.0.toml"


def _config() -> V03Config:
    return load_v03_config(V03_CONFIG_PATH)


def _fact_refs(count: int) -> tuple[str, ...]:
    return tuple(f"o-{index:04d}" for index in range(count))


def _example(
    *,
    task_name: TaskName,
    index: int,
    view: RemediationView = RemediationView.IID_VALIDATION,
    hidden_variant: str = "target-a",
) -> RemediationExample:
    visible_count = 1 + index % 5
    counterfactual_count = 1 + index % 3 if task_name is TaskName.COUNTERFACTUAL_COMPARE else 0
    context = CompactTargetContext(
        task_name=task_name,
        visible_fact_refs=_fact_refs(visible_count),
        counterfactual_visible_fact_refs=_fact_refs(counterfactual_count),
    )
    prompt = (
        f"Fictional Aster observation for {task_name.value} sample {index}."
        + " visible-context" * index
    )
    view_tag = "train" if view is RemediationView.IID_TRAIN else "validation"
    values: dict[str, Any] = {
        "artifact_version": "0.3.0",
        "example_id": f"selection:{view_tag}:{task_name.value}:{index:02d}",
        "view": view,
        "source_split": SplitName(view.value),
        "task_name": task_name,
        "group_id": f"group:{view_tag}:{task_name.value}:{index:02d}",
        "source_record_ids": (f"projection:{view_tag}:{task_name.value}:{index:02d}",),
        "parent_record_sha256": canonical_sha256(("parent", view_tag, task_name.value, index)),
        "prompt_text": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "template_family_id": f"template-{index % 4}",
        "alias_family_id": f"alias-{(index // 2) % 3}",
        "compact_context": context,
        # These three fields deliberately remain opaque to selection.  They are
        # syntactically valid field values but need not be valid task targets here.
        "compact_target": f"hidden compact {hidden_variant}",
        "canonical_target_json": f'{{"hidden":"{hidden_variant}"}}',
        "classification_label": hidden_variant,
        "augmentation": "none",
    }
    draft = RemediationExample.model_construct(**values, checksum_sha256="0" * 64)
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return draft.model_copy(update={"checksum_sha256": checksum})


def _dataset_from_examples(
    examples: tuple[RemediationExample, ...],
) -> SafeDevelopmentDataset:
    inventory = canonical_sha256(
        tuple(
            sorted(
                ((example.example_id, example.checksum_sha256) for example in examples),
                key=lambda item: item[0],
            )
        )
    )
    view_counts = Counter(example.view for example in examples)
    task_counts = Counter(example.task_name for example in examples)
    views = tuple(view for view in RemediationView if view_counts[view])
    values: dict[str, Any] = {
        "artifact_version": "0.3.0",
        "boundary": "development_only_no_final_or_golden_payloads",
        "source_commit": "a" * 40,
        "dataset_version": "selection-test-v1",
        "dataset_config_sha256": "b" * 64,
        "compact_contract_version": "0.2.0",
        "views": views,
        "example_count": len(examples),
        "counts_by_view": tuple((view, view_counts[view]) for view in views),
        "counts_by_task": tuple((task, task_counts[task]) for task in TaskName),
        "examples_sha256": canonical_sha256(tuple(example.checksum_sha256 for example in examples)),
        "examples_size_bytes": sum(
            len(example.prompt_text.encode("utf-8")) for example in examples
        ),
        "inventory_sha256": inventory,
    }
    draft = SafeDevelopmentManifest.model_construct(**values, checksum_sha256="0" * 64)
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    manifest = SafeDevelopmentManifest(**values, checksum_sha256=checksum)
    # model_construct is intentional: production artifacts are canonical by contract,
    # while ordering invariance is a property of the selection function itself.
    return SafeDevelopmentDataset.model_construct(manifest=manifest, examples=examples)


def _dataset(
    *,
    per_task: int = 10,
    hidden_variant: str = "target-a",
    within_task_order: tuple[int, ...] | None = None,
    include_train: bool = True,
) -> SafeDevelopmentDataset:
    order = within_task_order or tuple(range(per_task))
    if tuple(sorted(order)) != tuple(range(per_task)):
        raise ValueError("test order must be a permutation of all per-task indices")
    examples = tuple(
        _example(
            task_name=task,
            index=index,
            hidden_variant=hidden_variant,
        )
        for task in TaskName
        for index in order
    )
    if include_train:
        examples += tuple(
            _example(
                task_name=task,
                index=99,
                view=RemediationView.IID_TRAIN,
                hidden_variant=hidden_variant,
            )
            for task in TaskName
        )
    return _dataset_from_examples(examples)


def _observable_freeze(
    manifest: SemanticSelectionManifest,
) -> tuple[tuple[TaskName, ContextSizeStratum, str, str], ...]:
    return tuple(
        (
            entry.task_name,
            entry.context_stratum,
            entry.prompt_sha256,
            entry.observable_selection_key_sha256,
        )
        for entry in manifest.entries
    )


def _validated_manifest_payload(
    payload: dict[str, Any],
) -> SemanticSelectionManifest:
    entries = cast(tuple[dict[str, Any], ...], payload["entries"])
    payload["selected_inventory_sha256"] = canonical_sha256(
        tuple((entry["example_id"], entry["example_checksum_sha256"]) for entry in entries)
    )
    payload["selected_observable_inventory_sha256"] = canonical_sha256(
        tuple(entry["observable_selection_key_sha256"] for entry in entries)
    )
    body = {key: value for key, value in payload.items() if key != "checksum_sha256"}
    payload["checksum_sha256"] = canonical_sha256(body)
    return SemanticSelectionManifest.model_validate(payload)


def test_build_freezes_exact_task_and_context_balanced_subset() -> None:
    dataset = _dataset()
    first = build_semantic_selection_manifest(dataset, _config())
    replay = build_semantic_selection_manifest(dataset, _config())

    assert replay == first
    assert first.selected_example_count == SELECTION_EXAMPLE_COUNT == 48
    assert first.examples_per_task == EXAMPLES_PER_TASK == 8
    assert first.examples_per_task_stratum == EXAMPLES_PER_TASK_STRATUM == 4
    assert first.iid_validation_candidate_count == 60
    assert first.source_dataset_example_count == 66
    assert Counter(entry.task_name for entry in first.entries) == dict.fromkeys(
        TaskName, EXAMPLES_PER_TASK
    )
    assert Counter(entry.context_stratum for entry in first.entries) == dict.fromkeys(
        CONTEXT_STRATA, 24
    )
    assert Counter((entry.task_name, entry.context_stratum) for entry in first.entries) == {
        (task, stratum): EXAMPLES_PER_TASK_STRATUM
        for task in TaskName
        for stratum in CONTEXT_STRATA
    }
    assert tuple(entry.selection_index for entry in first.entries) == tuple(range(48))
    assert len({entry.example_id for entry in first.entries}) == 48
    for task in TaskName:
        for stratum in CONTEXT_STRATA:
            entries = tuple(
                entry
                for entry in first.entries
                if entry.task_name is task and entry.context_stratum is stratum
            )
            assert len({entry.template_family_id for entry in entries}) >= 2
            assert len({entry.alias_family_id for entry in entries}) >= 2

    resolved = resolve_semantic_selection_examples(dataset, first, _config())
    assert tuple(example.example_id for example in resolved) == tuple(
        entry.example_id for entry in first.entries
    )
    assert all(example.view is RemediationView.IID_VALIDATION for example in resolved)


@settings(max_examples=12, deadline=None)
@given(order=st.permutations(tuple(range(10))))
def test_selection_is_stable_under_every_generated_input_permutation(
    order: list[int],
) -> None:
    canonical_dataset = _dataset()
    canonical = build_semantic_selection_manifest(canonical_dataset, _config())
    reordered_examples = _dataset(within_task_order=tuple(order)).examples
    same_bound_source = SafeDevelopmentDataset.model_construct(
        manifest=canonical_dataset.manifest,
        examples=reordered_examples,
    )
    permuted = build_semantic_selection_manifest(
        same_bound_source,
        _config(),
    )

    assert permuted == canonical


def test_selection_is_target_independent_but_target_inventory_remains_bound() -> None:
    first = build_semantic_selection_manifest(
        _dataset(hidden_variant="entirely-hidden-target-a"),
        _config(),
    )
    changed_targets = build_semantic_selection_manifest(
        _dataset(hidden_variant="entirely-hidden-target-b"),
        _config(),
    )

    assert _observable_freeze(changed_targets) == _observable_freeze(first)
    assert changed_targets.selected_observable_inventory_sha256 == (
        first.selected_observable_inventory_sha256
    )
    assert changed_targets.iid_validation_observable_inventory_sha256 == (
        first.iid_validation_observable_inventory_sha256
    )
    assert changed_targets.selected_inventory_sha256 != first.selected_inventory_sha256
    assert changed_targets.iid_validation_inventory_sha256 != (
        first.iid_validation_inventory_sha256
    )
    assert changed_targets.source_dataset_inventory_sha256 != (
        first.source_dataset_inventory_sha256
    )
    assert changed_targets.checksum_sha256 != first.checksum_sha256


def test_resolution_binds_noncandidate_source_inventory_as_well_as_iid_inventory() -> None:
    dataset = _dataset()
    manifest = build_semantic_selection_manifest(dataset, _config())
    train_index = next(
        index
        for index, example in enumerate(dataset.examples)
        if example.view is RemediationView.IID_TRAIN
    )
    changed = list(dataset.examples)
    changed[train_index] = changed[train_index].model_copy(
        update={"checksum_sha256": canonical_sha256(("changed-train-only", train_index))}
    )
    changed_dataset = _dataset_from_examples(tuple(changed))
    changed_manifest = build_semantic_selection_manifest(changed_dataset, _config())

    assert changed_manifest.iid_validation_inventory_sha256 == (
        manifest.iid_validation_inventory_sha256
    )
    assert changed_manifest.iid_validation_observable_inventory_sha256 == (
        manifest.iid_validation_observable_inventory_sha256
    )
    assert changed_manifest.source_dataset_inventory_sha256 != (
        manifest.source_dataset_inventory_sha256
    )
    with pytest.raises(ValueError, match="complete source"):
        resolve_semantic_selection_examples(changed_dataset, manifest, _config())


def test_build_fails_closed_when_any_task_has_fewer_than_eight_candidates() -> None:
    examples = tuple(
        _example(task_name=task, index=index)
        for task in TaskName
        for index in range(7 if task is TaskName.INCIDENT_SUMMARY else 8)
    )

    with pytest.raises(ValueError, match="needs eight"):
        build_semantic_selection_manifest(_dataset_from_examples(examples), _config())


@pytest.mark.parametrize("duplicate_field", ["example_id", "checksum_sha256"])
def test_build_rejects_duplicate_source_identities(duplicate_field: str) -> None:
    dataset = _dataset()
    changed = list(dataset.examples)
    changed[1] = changed[1].model_copy(
        update={duplicate_field: getattr(changed[0], duplicate_field)}
    )

    with pytest.raises(ValueError, match="identities and checksums must be unique"):
        build_semantic_selection_manifest(_dataset_from_examples(tuple(changed)), _config())


def test_build_rejects_duplicate_prompt_identity_and_manifest_inventory_mismatch() -> None:
    dataset = _dataset()
    changed = list(dataset.examples)
    changed[1] = changed[1].model_copy(
        update={
            "prompt_text": changed[0].prompt_text,
            "prompt_sha256": changed[0].prompt_sha256,
        }
    )
    duplicate_prompt = _dataset_from_examples(tuple(changed))
    with pytest.raises(ValueError, match="duplicate task/prompt identity"):
        build_semantic_selection_manifest(duplicate_prompt, _config())

    wrong_manifest = dataset.manifest.model_copy(update={"inventory_sha256": "f" * 64})
    mismatched = SafeDevelopmentDataset.model_construct(
        manifest=wrong_manifest,
        examples=dataset.examples,
    )
    with pytest.raises(ValueError, match="inventory checksum mismatch"):
        build_semantic_selection_manifest(mismatched, _config())


def test_build_rejects_empty_invalid_count_and_changed_selection_limit() -> None:
    dataset = _dataset()
    empty = SafeDevelopmentDataset.model_construct(
        manifest=dataset.manifest,
        examples=(),
    )
    with pytest.raises(TypeError, match="invalid example inventory"):
        build_semantic_selection_manifest(empty, _config())

    wrong_count_manifest = dataset.manifest.model_copy(update={"example_count": 1})
    wrong_count = SafeDevelopmentDataset.model_construct(
        manifest=wrong_count_manifest,
        examples=dataset.examples,
    )
    with pytest.raises(ValueError, match="count differs"):
        build_semantic_selection_manifest(wrong_count, _config())

    changed_limit = _config().model_copy(update={"semantic_selection_example_limit": 47})
    with pytest.raises(ValueError, match="limit differs from 48"):
        build_semantic_selection_manifest(dataset, changed_limit)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"source_split": SplitName.IID_TRAIN}, "IID validation only"),
        ({"prompt_sha256": "0" * 64}, "prompt checksum mismatch"),
        (
            {
                "compact_context": CompactTargetContext(
                    task_name=TaskName.NEXT_ACTION,
                    visible_fact_refs=("o-0000",),
                )
            },
            "task differs",
        ),
    ],
)
def test_candidate_boundary_rejects_inconsistent_observable_context(
    update: dict[str, object],
    message: str,
) -> None:
    dataset = _dataset()
    changed = list(dataset.examples)
    changed[0] = changed[0].model_copy(update=update)

    with pytest.raises(ValueError, match=message):
        build_semantic_selection_manifest(_dataset_from_examples(tuple(changed)), _config())


def test_manifest_contract_rejects_unknown_coerced_tampered_and_reordered_data() -> None:
    manifest = build_semantic_selection_manifest(_dataset(), _config())

    unknown = manifest.model_dump(mode="python", round_trip=True)
    unknown["unexpected"] = "field"
    with pytest.raises(ValidationError):
        SemanticSelectionManifest.model_validate(unknown)

    coerced = manifest.model_dump(mode="python", round_trip=True)
    coerced["selected_example_count"] = "48"
    with pytest.raises(ValidationError):
        SemanticSelectionManifest.model_validate(coerced)

    observable_tamper = manifest.model_dump(mode="python", round_trip=True)
    observable_entries = list(cast(tuple[dict[str, Any], ...], observable_tamper["entries"]))
    observable_entries[0]["visible_fact_count"] += 1
    observable_tamper["entries"] = tuple(observable_entries)
    with pytest.raises(ValidationError, match="observable key checksum mismatch"):
        SemanticSelectionManifest.model_validate(observable_tamper)

    reordered = manifest.model_dump(mode="python", round_trip=True)
    reordered_entries = list(cast(tuple[dict[str, Any], ...], reordered["entries"]))
    reordered_entries[0], reordered_entries[1] = reordered_entries[1], reordered_entries[0]
    reordered["entries"] = tuple(reordered_entries)
    with pytest.raises(ValidationError, match="canonical indexed order"):
        SemanticSelectionManifest.model_validate(reordered)

    checksum_tamper = manifest.model_dump(mode="python", round_trip=True)
    checksum_tamper["checksum_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="manifest checksum mismatch"):
        SemanticSelectionManifest.model_validate(checksum_tamper)


def test_manifest_contract_rejects_quota_and_identity_tampering() -> None:
    manifest = build_semantic_selection_manifest(_dataset(), _config())

    task_quota = manifest.model_dump(mode="python", round_trip=True)
    task_counts = list(cast(tuple[tuple[TaskName, int], ...], task_quota["counts_by_task"]))
    task_counts[0] = (task_counts[0][0], 7)
    task_quota["counts_by_task"] = tuple(task_counts)
    with pytest.raises(ValidationError):
        SemanticSelectionManifest.model_validate(task_quota)

    duplicate_id = manifest.model_dump(mode="python", round_trip=True)
    entries = list(cast(tuple[dict[str, Any], ...], duplicate_id["entries"]))
    entries[1]["example_id"] = entries[0]["example_id"]
    duplicate_id["entries"] = tuple(entries)
    with pytest.raises(ValidationError, match="example IDs must be unique"):
        SemanticSelectionManifest.model_validate(duplicate_id)


def test_manifest_contract_rejects_every_frozen_binding_and_quota_order_drift() -> None:
    manifest = build_semantic_selection_manifest(_dataset(), _config())

    policy = manifest.model_dump(mode="python", round_trip=True)
    policy["policy_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="policy checksum mismatch"):
        SemanticSelectionManifest.model_validate(policy)

    candidate_count = manifest.model_dump(mode="python", round_trip=True)
    candidate_count["source_dataset_example_count"] = 48
    with pytest.raises(ValidationError, match="candidate count exceeds"):
        SemanticSelectionManifest.model_validate(candidate_count)

    task_order = manifest.model_dump(mode="python", round_trip=True)
    tasks = list(cast(tuple[tuple[TaskName, int], ...], task_order["counts_by_task"]))
    tasks[0], tasks[1] = tasks[1], tasks[0]
    task_order["counts_by_task"] = tuple(tasks)
    with pytest.raises(ValidationError, match="exactly eight examples per task"):
        SemanticSelectionManifest.model_validate(task_order)

    stratum_order = manifest.model_dump(mode="python", round_trip=True)
    strata = cast(
        tuple[tuple[ContextSizeStratum, int], ...],
        stratum_order["counts_by_context_stratum"],
    )
    stratum_order["counts_by_context_stratum"] = tuple(reversed(strata))
    with pytest.raises(ValidationError, match="balance both context strata"):
        SemanticSelectionManifest.model_validate(stratum_order)

    joint_order = manifest.model_dump(mode="python", round_trip=True)
    joints = list(
        cast(
            tuple[tuple[TaskName, ContextSizeStratum, int], ...],
            joint_order["counts_by_task_and_stratum"],
        )
    )
    joints[0], joints[1] = joints[1], joints[0]
    joint_order["counts_by_task_and_stratum"] = tuple(joints)
    with pytest.raises(ValidationError, match="task/context quotas"):
        SemanticSelectionManifest.model_validate(joint_order)


def test_manifest_contract_rejects_duplicate_checksums_prompts_and_inventory_hashes() -> None:
    manifest = build_semantic_selection_manifest(_dataset(), _config())

    duplicate_checksum = manifest.model_dump(mode="python", round_trip=True)
    checksum_entries = list(cast(tuple[dict[str, Any], ...], duplicate_checksum["entries"]))
    checksum_entries[1]["example_checksum_sha256"] = checksum_entries[0]["example_checksum_sha256"]
    duplicate_checksum["entries"] = tuple(checksum_entries)
    with pytest.raises(ValidationError, match="example checksums must be unique"):
        SemanticSelectionManifest.model_validate(duplicate_checksum)

    duplicate_prompt = manifest.model_dump(mode="python", round_trip=True)
    prompt_entries = list(cast(tuple[dict[str, Any], ...], duplicate_prompt["entries"]))
    observable_fields = (
        "prompt_sha256",
        "prompt_utf8_bytes",
        "visible_fact_count",
        "counterfactual_visible_fact_count",
        "template_family_id",
        "alias_family_id",
        "augmentation",
        "observable_selection_key_sha256",
    )
    for field in observable_fields:
        prompt_entries[1][field] = prompt_entries[0][field]
    duplicate_prompt["entries"] = tuple(prompt_entries)
    with pytest.raises(ValidationError, match="task/prompt identities must be unique"):
        SemanticSelectionManifest.model_validate(duplicate_prompt)

    selected_inventory = manifest.model_dump(mode="python", round_trip=True)
    selected_inventory["selected_inventory_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="selected inventory checksum mismatch"):
        SemanticSelectionManifest.model_validate(selected_inventory)

    observable_inventory = manifest.model_dump(mode="python", round_trip=True)
    observable_inventory["selected_observable_inventory_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="observable inventory checksum mismatch"):
        SemanticSelectionManifest.model_validate(observable_inventory)


def test_resolution_rejects_entry_identity_and_observable_binding_drift() -> None:
    dataset = _dataset()
    manifest = build_semantic_selection_manifest(dataset, _config())
    selected_ids = {entry.example_id for entry in manifest.entries}
    unselected = next(
        example
        for example in dataset.examples
        if example.view is RemediationView.IID_VALIDATION and example.example_id not in selected_ids
    )

    identity_payload = manifest.model_dump(mode="python", round_trip=True)
    identity_entries = list(cast(tuple[dict[str, Any], ...], identity_payload["entries"]))
    identity_entries[0]["example_id"] = unselected.example_id
    identity_payload["entries"] = tuple(identity_entries)
    identity_drift = _validated_manifest_payload(identity_payload)
    with pytest.raises(ValueError, match="entry identity"):
        resolve_semantic_selection_examples(dataset, identity_drift, _config())

    observable_payload = manifest.model_dump(mode="python", round_trip=True)
    observable_entries = list(cast(tuple[dict[str, Any], ...], observable_payload["entries"]))
    observable_entries[0]["template_family_id"] = "tampered-template"
    observable_entries[0]["observable_selection_key_sha256"] = canonical_sha256(
        {
            "task_name": observable_entries[0]["task_name"].value,
            "prompt_sha256": observable_entries[0]["prompt_sha256"],
            "prompt_utf8_bytes": observable_entries[0]["prompt_utf8_bytes"],
            "visible_fact_count": observable_entries[0]["visible_fact_count"],
            "counterfactual_visible_fact_count": observable_entries[0][
                "counterfactual_visible_fact_count"
            ],
            "template_family_id": observable_entries[0]["template_family_id"],
            "alias_family_id": observable_entries[0]["alias_family_id"],
            "augmentation": observable_entries[0]["augmentation"],
        }
    )
    observable_payload["entries"] = tuple(observable_entries)
    observable_drift = _validated_manifest_payload(observable_payload)
    with pytest.raises(ValueError, match="observable key"):
        resolve_semantic_selection_examples(dataset, observable_drift, _config())


def test_resolution_rejects_fully_rebound_unselected_row_substitution() -> None:
    """A valid source row cannot replace the deterministic algorithm's row."""

    dataset = _dataset()
    config = _config()
    manifest = build_semantic_selection_manifest(dataset, config)
    selected_ids = {entry.example_id for entry in manifest.entries}
    unselected = next(
        example
        for example in dataset.examples
        if example.view is RemediationView.IID_VALIDATION and example.example_id not in selected_ids
    )
    task_candidates = tuple(
        sorted(
            (
                example
                for example in dataset.examples
                if example.view is RemediationView.IID_VALIDATION
                and example.task_name is unselected.task_name
            ),
            key=lambda example: (
                len(example.compact_context.visible_fact_refs)
                + len(example.compact_context.counterfactual_visible_fact_refs),
                len(example.compact_context.counterfactual_visible_fact_refs),
                len(example.prompt_text.encode("utf-8")),
                example.prompt_sha256,
            ),
        )
    )
    midpoint = len(task_candidates) // 2
    stratum = (
        ContextSizeStratum.LOWER
        if unselected in task_candidates[:midpoint]
        else ContextSizeStratum.UPPER
    )

    payload = manifest.model_dump(mode="python", round_trip=True)
    entries = [dict(entry) for entry in cast(tuple[dict[str, Any], ...], payload["entries"])]
    replacement_index = next(
        index
        for index, entry in enumerate(entries)
        if entry["task_name"] is unselected.task_name and entry["context_stratum"] is stratum
    )
    observable = {
        "task_name": unselected.task_name.value,
        "prompt_sha256": unselected.prompt_sha256,
        "prompt_utf8_bytes": len(unselected.prompt_text.encode("utf-8")),
        "visible_fact_count": len(unselected.compact_context.visible_fact_refs),
        "counterfactual_visible_fact_count": len(
            unselected.compact_context.counterfactual_visible_fact_refs
        ),
        "template_family_id": unselected.template_family_id,
        "alias_family_id": unselected.alias_family_id,
        "augmentation": unselected.augmentation,
    }
    entries[replacement_index] = {
        "selection_index": replacement_index,
        **observable,
        "task_name": unselected.task_name,
        "context_stratum": stratum,
        "example_id": unselected.example_id,
        "example_checksum_sha256": unselected.checksum_sha256,
        "observable_selection_key_sha256": canonical_sha256(observable),
    }
    task_order = {task: index for index, task in enumerate(TaskName)}
    stratum_order = {item: index for index, item in enumerate(CONTEXT_STRATA)}
    entries.sort(
        key=lambda entry: (
            task_order[cast(TaskName, entry["task_name"])],
            stratum_order[cast(ContextSizeStratum, entry["context_stratum"])],
            cast(str, entry["prompt_sha256"]),
        )
    )
    for index, entry in enumerate(entries):
        entry["selection_index"] = index
    payload["entries"] = tuple(entries)
    rebound = _validated_manifest_payload(payload)

    assert rebound.source_dataset_manifest_sha256 == manifest.source_dataset_manifest_sha256
    assert rebound.iid_validation_inventory_sha256 == manifest.iid_validation_inventory_sha256
    assert rebound.checksum_sha256 != manifest.checksum_sha256
    with pytest.raises(ValueError, match="deterministic source selection"):
        resolve_semantic_selection_examples(dataset, rebound, config)


def test_manifest_atomic_write_load_pin_and_nonoverwrite(tmp_path: Path) -> None:
    manifest = build_semantic_selection_manifest(_dataset(), _config())
    path = tmp_path / "nested" / "semantic-selection.json"

    write_semantic_selection_manifest(manifest, path)

    assert (
        load_semantic_selection_manifest(
            path,
            expected_checksum=manifest.checksum_sha256,
        )
        == manifest
    )
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode & 0o600 == 0o600
    assert mode & 0o137 == 0
    assert not path.with_name(f".{path.name}.tmp").exists()
    with pytest.raises(FileExistsError, match="must not overwrite"):
        write_semantic_selection_manifest(manifest, path)
    with pytest.raises(ValueError, match="expected checksum"):
        load_semantic_selection_manifest(path, expected_checksum="f" * 64)
    with pytest.raises(TypeError, match="checksum is invalid"):
        load_semantic_selection_manifest(path, expected_checksum="not-a-checksum")


def test_manifest_load_rejects_noncanonical_duplicate_nonfinite_and_symlink(
    tmp_path: Path,
) -> None:
    manifest = build_semantic_selection_manifest(_dataset(), _config())
    canonical_path = tmp_path / "canonical.json"
    write_semantic_selection_manifest(manifest, canonical_path)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(b" " + canonical_path.read_bytes())
    with pytest.raises(ValueError, match="not canonical JSON"):
        load_semantic_selection_manifest(noncanonical)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"manifest_version":"0.3.0","manifest_version":"0.3.0"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_semantic_selection_manifest(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_bytes(b'{"value":NaN}')
    with pytest.raises(ValueError, match="non-finite JSON"):
        load_semantic_selection_manifest(nonfinite)

    symlink = tmp_path / "manifest-link.json"
    symlink.symlink_to(canonical_path)
    with pytest.raises(ValueError, match="non-symlink"):
        load_semantic_selection_manifest(symlink)


def test_manifest_io_bounds_and_temporary_path_are_fail_closed(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.touch()
    with pytest.raises(ValueError, match="size is outside"):
        load_semantic_selection_manifest(empty)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (MAX_SELECTION_MANIFEST_BYTES + 1))
    with pytest.raises(ValueError, match="size is outside"):
        load_semantic_selection_manifest(oversized)

    manifest = build_semantic_selection_manifest(_dataset(), _config())
    destination = tmp_path / "blocked.json"
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.touch()
    with pytest.raises(FileExistsError, match="temporary path"):
        write_semantic_selection_manifest(manifest, destination)
    assert not destination.exists()
    assert temporary.exists()


def test_atomic_write_cleans_up_after_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_semantic_selection_manifest(_dataset(), _config())
    destination = tmp_path / "replace-failure.json"
    temporary = destination.with_name(f".{destination.name}.tmp")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr("reactorbench.remediation.selection.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated atomic replace failure"):
        write_semantic_selection_manifest(manifest, destination)

    assert not destination.exists()
    assert not temporary.exists()


def test_public_boundaries_require_exact_contract_types(tmp_path: Path) -> None:
    dataset = _dataset()
    config = _config()
    manifest = build_semantic_selection_manifest(dataset, config)

    with pytest.raises(TypeError, match="exact SafeDevelopmentDataset"):
        build_semantic_selection_manifest(cast(Any, object()), config)
    with pytest.raises(TypeError, match="exact V03Config"):
        build_semantic_selection_manifest(dataset, cast(Any, object()))
    with pytest.raises(TypeError, match="exact manifest/config"):
        resolve_semantic_selection_examples(dataset, cast(Any, object()), config)
    with pytest.raises(TypeError, match="exact manifest and Path"):
        write_semantic_selection_manifest(manifest, cast(Any, str(tmp_path / "bad")))
    with pytest.raises(ValueError, match="regular non-symlink"):
        load_semantic_selection_manifest(cast(Any, str(tmp_path / "bad")))
