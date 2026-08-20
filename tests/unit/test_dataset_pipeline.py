from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from reactorbench.dataset.config import load_development_dataset_config
from reactorbench.dataset.contracts import PromptCounterfactualComparisonTarget
from reactorbench.dataset.grouping import CounterfactualFamily
from reactorbench.dataset.pipeline import (
    DevelopmentProjectionBundle,
    _counterfactual_input_structured_duplicate_count,
    build_development_projection_bundle,
)
from reactorbench.schemas.base import canonical_sha256
from reactorbench.schemas.enums import SplitName, TaskName

CONFIG = Path("configs/dataset/development-v0.1.0.toml")


@pytest.fixture(scope="module")
def bundle() -> DevelopmentProjectionBundle:
    config = load_development_dataset_config(CONFIG)
    return build_development_projection_bundle(config, generator_commit="abcdef0")


def _with_recomputed_checksum[ModelT: BaseModel](model: ModelT, **updates: object) -> ModelT:
    """Return a fully self-consistent record after a deliberate relation mutation."""

    draft = model.model_copy(update=updates)
    checksum_field = (
        "entry_checksum_sha256"
        if "entry_checksum_sha256" in type(draft).model_fields
        else "checksum_sha256"
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={checksum_field})
    )
    return draft.model_copy(update={checksum_field: checksum})


def test_full_development_build_is_deterministic_and_complete(
    bundle: DevelopmentProjectionBundle,
) -> None:
    assert bundle.summary.trajectory_count == 204
    assert bundle.summary.projection_count == 1_762
    assert bundle.summary.counterfactual_pair_count == 14
    assert bundle.summary.complete_group_count == 10
    assert bundle.summary.incomplete_g15_group_count == 24
    assert bundle.summary.single_input_structured_duplicate_count == 0
    assert bundle.summary.counterfactual_input_structured_duplicate_count == 0
    assert len(bundle.split_manifest.entries) == len(bundle.projections)

    split_counts = dict(bundle.summary.trajectory_counts_by_split)
    assert split_counts == {
        SplitName.IID_TRAIN: 70,
        SplitName.IID_VALIDATION: 28,
        SplitName.IID_TEST: 28,
        SplitName.TEMPLATE_TEST: 28,
        SplitName.COMPONENT_TEST: 10,
        SplitName.SEVERITY_TEST: 4,
        SplitName.COMPOSITION_TEST: 8,
        SplitName.COUNTERFACTUAL_TEST: 18,
        SplitName.NOISE_TEST: 10,
    }
    task_counts = dict(bundle.summary.projection_counts_by_task)
    assert task_counts == {
        TaskName.CONTINUE_LOG: 148,
        TaskName.FAULT_FAMILY: 399,
        TaskName.EXTRACT_EVIDENCE: 405,
        TaskName.NEXT_ACTION: 405,
        TaskName.INCIDENT_SUMMARY: 405,
    }

    rebuilt = build_development_projection_bundle(
        load_development_dataset_config(CONFIG), generator_commit="abcdef0"
    )
    assert rebuilt.checksum_sha256 == bundle.checksum_sha256
    assert rebuilt == bundle


def test_groups_are_atomic_and_only_g15_is_intentionally_incomplete(
    bundle: DevelopmentProjectionBundle,
) -> None:
    incomplete = tuple(group for group in bundle.groups if not group.is_complete)
    assert len(incomplete) == 24
    assert {group.family for group in incomplete} == {CounterfactualFamily.G15_EVIDENCE_SUFFICIENCY}

    split_by_projection = {
        entry.projection_id: entry.split_name for entry in bundle.split_manifest.entries
    }
    for pair in bundle.counterfactual_projections:
        assert (
            split_by_projection[pair.lineage.baseline_projection_id]
            is split_by_projection[pair.lineage.counterfactual_projection_id]
        )
        target = pair.task_target.target
        assert isinstance(target, PromptCounterfactualComparisonTarget)
        assert target.changed_fields
        assert target.baseline_decisive_fact_refs or target.counterfactual_decisive_fact_refs


def _mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_mapping_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_mapping_keys(item) for item in value))
    return set()


def test_projection_inputs_never_contain_audit_truth_or_applied_actions(
    bundle: DevelopmentProjectionBundle,
) -> None:
    forbidden_keys = {
        "scenario_id",
        "trajectory_id",
        "seed",
        "fault_injections",
        "fault_labels",
        "targets",
        "provenance",
        "latent_states",
        "related_event_ids",
        "evidence_slots",
        "context_id",
    }
    fingerprints: Counter[tuple[str, TaskName]] = Counter()
    for projection in bundle.projections:
        payload = projection.model_input.model_dump(mode="json", round_trip=True)
        assert not forbidden_keys.intersection(_mapping_keys(payload))
        assert all(
            event.event_type.value != "ACTION_APPLIED"
            for event in projection.model_input.event_facts
        )
        fingerprints[
            (
                projection.model_input.structured_fingerprint(),
                projection.task_target.task_name,
            )
        ] += 1
    assert all(count == 1 for count in fingerprints.values())


def test_noise_manifest_uses_only_applicable_preregistered_corruptions(
    bundle: DevelopmentProjectionBundle,
) -> None:
    noise_entries = tuple(
        entry for entry in bundle.split_manifest.entries if entry.split_name is SplitName.NOISE_TEST
    )
    assert len(noise_entries) == 48
    assert {entry.corruption_plan for entry in noise_entries} == {
        "benign_insert",
        "duplicate_line",
        "omit_noncritical",
        "safe_reorder",
    }
    omitted = tuple(entry for entry in noise_entries if entry.corruption_plan == "omit_noncritical")
    assert len(omitted) == 4
    assert all(entry.task_name is TaskName.CONTINUE_LOG for entry in omitted)


@pytest.mark.parametrize("commit", ["", "ABCDEF0", "123456", "g123456"])
def test_build_rejects_invalid_generator_commit(commit: str) -> None:
    config = load_development_dataset_config(CONFIG)
    with pytest.raises(ValidationError):
        build_development_projection_bundle(config, generator_commit=commit)


def test_build_rejects_noncanonical_input_types() -> None:
    config = load_development_dataset_config(CONFIG)
    with pytest.raises(TypeError):
        build_development_projection_bundle(config.model_dump(), generator_commit="abcdef0")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_development_projection_bundle(config, generator_commit=7)  # type: ignore[arg-type]


def test_build_enforces_reviewed_task_record_bound() -> None:
    config = load_development_dataset_config(CONFIG)
    bounded = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(update={"maximum_task_records": 1}),
        }
    )
    with pytest.raises(ValueError, match="task count"):
        build_development_projection_bundle(bounded, generator_commit="abcdef0")


def test_bundle_rejects_misreported_structured_duplicate_counts(
    bundle: DevelopmentProjectionBundle,
) -> None:
    payload = bundle.model_dump(mode="json", round_trip=True)
    summary = payload["summary"]
    assert isinstance(summary, dict)
    summary["single_input_structured_duplicate_count"] = 1
    with pytest.raises(
        ValidationError,
        match="summary does not match the canonical structured inventory",
    ):
        DevelopmentProjectionBundle.model_validate_json(json.dumps(payload))


def test_bundle_rejects_duplicate_counterfactual_pair_inputs(
    bundle: DevelopmentProjectionBundle,
) -> None:
    original = bundle.counterfactual_projections[0]
    duplicate = original.model_copy(update={"pair_id": "pair-duplicate-audit"})
    duplicate = duplicate.model_copy(
        update={
            "checksum_sha256": canonical_sha256(
                duplicate.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
            )
        }
    )
    pairs = tuple(
        sorted((*bundle.counterfactual_projections, duplicate), key=lambda pair: pair.pair_id)
    )
    assert _counterfactual_input_structured_duplicate_count(pairs) == 1
    summary = bundle.summary.model_copy(
        update={
            "counterfactual_pair_count": len(pairs),
            "counterfactual_input_structured_duplicate_count": 1,
        }
    )
    draft = bundle.model_dump(mode="python", round_trip=True)
    draft["counterfactual_projections"] = pairs
    draft["summary"] = summary

    with pytest.raises(
        ValueError,
        match=r"counterfactual pair ID|task-scoped model inputs",
    ):
        DevelopmentProjectionBundle.model_validate(draft)


def test_bundle_rejects_projection_with_nonexistent_trajectory_even_when_rehashed(
    bundle: DevelopmentProjectionBundle,
) -> None:
    original = bundle.projections[0]
    tampered_lineage = original.lineage.model_copy(
        update={"trajectory_id": "trajectory:nonexistent-source"}
    )
    tampered_projection = _with_recomputed_checksum(original, lineage=tampered_lineage)
    projections = tuple(
        sorted(
            (
                tampered_projection if item.projection_id == original.projection_id else item
                for item in bundle.projections
            ),
            key=lambda item: item.projection_id,
        )
    )
    tampered = _with_recomputed_checksum(bundle, projections=projections)

    with pytest.raises(ValueError, match="nonexistent source trajectory"):
        DevelopmentProjectionBundle.model_validate(tampered.__dict__)


def test_bundle_rejects_stale_manifest_entry_even_when_all_local_hashes_are_recomputed(
    bundle: DevelopmentProjectionBundle,
) -> None:
    original = bundle.split_manifest.entries[0]
    stale_entry = _with_recomputed_checksum(
        original,
        projection_checksum_sha256="f" * 64,
    )
    entries = tuple(
        stale_entry if item.projection_id == original.projection_id else item
        for item in bundle.split_manifest.entries
    )
    stale_manifest = _with_recomputed_checksum(bundle.split_manifest, entries=entries)
    tampered = _with_recomputed_checksum(bundle, split_manifest=stale_manifest)

    with pytest.raises(ValueError, match="split manifest entry is stale"):
        DevelopmentProjectionBundle.model_validate(tampered.__dict__)


def test_bundle_rejects_rehashed_substituted_counterfactual_target(
    bundle: DevelopmentProjectionBundle,
) -> None:
    original = next(
        pair
        for pair in bundle.counterfactual_projections
        if isinstance(pair.task_target.target, PromptCounterfactualComparisonTarget)
        and pair.task_target.target.decisive_evidence_slots
    )
    source_target = original.task_target.target
    assert isinstance(source_target, PromptCounterfactualComparisonTarget)
    target = source_target.model_copy(update={"decisive_evidence_slots": ()})
    task_target = original.task_target.model_copy(update={"target": target})
    tampered_pair = _with_recomputed_checksum(original, task_target=task_target)
    pairs = tuple(
        sorted(
            (
                tampered_pair if item.pair_id == original.pair_id else item
                for item in bundle.counterfactual_projections
            ),
            key=lambda item: item.pair_id,
        )
    )
    tampered = _with_recomputed_checksum(bundle, counterfactual_projections=pairs)

    with pytest.raises(ValueError, match="deterministic source projection pair"):
        DevelopmentProjectionBundle.model_validate(tampered.__dict__)


def test_bundle_rejects_rehashed_summary_that_lies_about_derived_inventory(
    bundle: DevelopmentProjectionBundle,
) -> None:
    summary = bundle.summary.model_copy(
        update={
            "complete_group_count": bundle.summary.complete_group_count + 1,
            "incomplete_g15_group_count": bundle.summary.incomplete_g15_group_count + 1,
            "trajectory_counts_by_split": (
                (SplitName.IID_TRAIN, bundle.summary.trajectory_counts_by_split[0][1] + 1),
                *bundle.summary.trajectory_counts_by_split[1:],
            ),
            "projection_counts_by_task": (
                (TaskName.CONTINUE_LOG, bundle.summary.projection_counts_by_task[0][1] + 1),
                *bundle.summary.projection_counts_by_task[1:],
            ),
        }
    )
    tampered = _with_recomputed_checksum(bundle, summary=summary)

    with pytest.raises(ValueError, match="canonical structured inventory"):
        DevelopmentProjectionBundle.model_validate(tampered.__dict__)
