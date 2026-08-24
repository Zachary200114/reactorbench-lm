"""Isolation, augmentation, and safe-artifact tests for remediation data."""

from __future__ import annotations

import ast
import hashlib
import inspect
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from pydantic import ValidationError

import reactorbench.dataset as dataset_package
import reactorbench.dataset.development as historical_development
import reactorbench.dataset.pipeline as dataset_pipeline
import reactorbench.evaluation.golden as golden_evaluation
import reactorbench.remediation.data as remediation_data
from reactorbench.dataset.catalog import AliasFamily, TemplateFamily
from reactorbench.dataset.config import DevelopmentDatasetConfig, load_development_dataset_config
from reactorbench.dataset.contracts import ModelInput, PromptEvidenceTarget
from reactorbench.dataset.pipeline import (
    ScopedProjectionInventory,
    build_scoped_projection_inventory,
)
from reactorbench.evaluation.compact import compact_target_json, parse_compact_target
from reactorbench.remediation.audit import audit_safe_development_dataset
from reactorbench.remediation.config import VIEW_SOURCE_SPLIT, RemediationView
from reactorbench.remediation.data import (
    FROZEN_V03_DEDUPLICATED_MANIFEST_SHA256,
    FROZEN_V03_RAW_MANIFEST_SHA256,
    FROZEN_V03_SOURCE_COMMIT,
    FROZEN_V03_TRAIN_ALIAS_FAMILIES,
    FROZEN_V03_TRAIN_TEMPLATE_FAMILIES,
    MAX_EXAMPLE_BYTES,
    FrozenV03IIDMaterial,
    RemediationExample,
    SafeDevelopmentDataset,
    SafeDevelopmentManifest,
    build_frozen_v03_iid_material,
    build_safe_development_dataset,
    build_safe_development_dataset_with_structured_fingerprints,
    load_safe_development_artifact,
    write_safe_development_artifact,
)
from reactorbench.schemas.base import canonical_json_bytes, canonical_sha256
from reactorbench.schemas.enums import (
    AbstentionReason,
    ActionLabel,
    DiagnosisStatus,
    OperatingMode,
    SplitName,
    TaskName,
)
from reactorbench.schemas.target import (
    FaultDiagnosisTarget,
    IncidentSummaryTarget,
    NextActionTarget,
)

ROOT = Path(__file__).resolve().parents[2]
V01_DATASET_CONFIG = ROOT / "configs/dataset/development-v0.1.0.toml"
V03_DATASET_CONFIG = ROOT / "configs/dataset/remediation-development-v0.3.0.toml"
SOURCE_COMMIT = "abcdef1"


@pytest.fixture(scope="module")
def v01_config() -> DevelopmentDatasetConfig:
    return load_development_dataset_config(V01_DATASET_CONFIG)


@pytest.fixture(scope="module")
def scoped_train_validation(v01_config: DevelopmentDatasetConfig) -> ScopedProjectionInventory:
    """The suite's one real generator call; no serialized artifact is opened."""

    return build_scoped_projection_inventory(
        v01_config,
        generator_commit=SOURCE_COMMIT,
        splits=(SplitName.IID_TRAIN, SplitName.IID_VALIDATION),
    )


@pytest.fixture(scope="module")
def base_dataset(
    v01_config: DevelopmentDatasetConfig,
    scoped_train_validation: ScopedProjectionInventory,
) -> SafeDevelopmentDataset:
    with patch.object(
        remediation_data,
        "build_scoped_projection_inventory",
        return_value=scoped_train_validation,
    ):
        return build_safe_development_dataset(
            v01_config,
            source_commit=SOURCE_COMMIT,
            views=(RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION),
        )


@pytest.fixture(scope="module")
def frozen_v03_material() -> tuple[FrozenV03IIDMaterial, int]:
    config = load_development_dataset_config(V03_DATASET_CONFIG)
    generator_calls = 0

    def counted_generator(
        selected_config: DevelopmentDatasetConfig,
        *,
        generator_commit: str,
        splits: tuple[SplitName, ...],
    ) -> ScopedProjectionInventory:
        nonlocal generator_calls
        generator_calls += 1
        return build_scoped_projection_inventory(
            selected_config,
            generator_commit=generator_commit,
            splits=splits,
        )

    with patch.object(
        remediation_data,
        "build_scoped_projection_inventory",
        side_effect=counted_generator,
    ):
        material = build_frozen_v03_iid_material(
            config,
            source_commit=FROZEN_V03_SOURCE_COMMIT,
            train_template_families=FROZEN_V03_TRAIN_TEMPLATE_FAMILIES,
            train_alias_families=FROZEN_V03_TRAIN_ALIAS_FAMILIES,
            renderer_variants_per_projection=3,
            include_insufficient_evidence_views=True,
        )
    return material, generator_calls


def _inventory_for_split(
    inventory: ScopedProjectionInventory, split_name: SplitName
) -> ScopedProjectionInventory:
    trajectories = tuple(item for item in inventory.trajectories if item.split_name is split_name)
    trajectory_ids = {item.trajectory.trajectory_id for item in trajectories}
    scenario_ids = {item.trajectory.scenario_id for item in trajectories}
    projections = tuple(
        item for item in inventory.projections if item.lineage.trajectory_id in trajectory_ids
    )
    projection_ids = {item.projection_id for item in projections}
    return ScopedProjectionInventory(
        requested_splits=(split_name,),
        trajectories=trajectories,
        groups=tuple(
            group
            for group in inventory.groups
            if all(member.scenario_id in scenario_ids for member in group.members)
        ),
        projections=projections,
        counterfactual_projections=tuple(
            pair
            for pair in inventory.counterfactual_projections
            if pair.lineage.baseline_projection_id in projection_ids
            and pair.lineage.counterfactual_projection_id in projection_ids
        ),
    )


def _rebuilt_manifest(
    original: SafeDevelopmentManifest,
    *,
    examples_sha256: str,
    examples_size_bytes: int,
) -> SafeDevelopmentManifest:
    values = original.model_dump(mode="python", exclude={"checksum_sha256"})
    values["examples_sha256"] = examples_sha256
    values["examples_size_bytes"] = examples_size_bytes
    draft = SafeDevelopmentManifest.model_construct(**values, checksum_sha256="0" * 64)
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return SafeDevelopmentManifest(**values, checksum_sha256=checksum)


def _write_manifest(path: Path, manifest: SafeDevelopmentManifest) -> None:
    path.write_bytes(
        canonical_json_bytes(manifest.model_dump(mode="json", round_trip=True)) + b"\n"
    )


def _rehash_contract_payload(payload: dict[str, object]) -> dict[str, object]:
    rebound = dict(payload)
    rebound.pop("checksum_sha256", None)
    rebound["checksum_sha256"] = canonical_sha256(rebound)
    return rebound


def _with_prompt_text(
    dataset: SafeDevelopmentDataset,
    example: RemediationExample,
    prompt_text: str,
) -> SafeDevelopmentDataset:
    """Return a valid self-rehashed dataset carrying one prompt-text mutation."""

    example_payload = example.model_dump(mode="python", round_trip=True)
    example_payload["prompt_text"] = prompt_text
    example_payload["prompt_sha256"] = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    mutated_example = RemediationExample.model_validate(_rehash_contract_payload(example_payload))
    examples = tuple(
        mutated_example if item.example_id == example.example_id else item
        for item in dataset.examples
    )
    serialized = b"".join(
        canonical_json_bytes(item.model_dump(mode="json", round_trip=True)) + b"\n"
        for item in examples
    )
    manifest_payload = dataset.manifest.model_dump(mode="python", round_trip=True)
    manifest_payload["examples_sha256"] = hashlib.sha256(serialized).hexdigest()
    manifest_payload["examples_size_bytes"] = len(serialized)
    manifest_payload["inventory_sha256"] = canonical_sha256(
        tuple((item.example_id, item.checksum_sha256) for item in examples)
    )
    manifest = SafeDevelopmentManifest.model_validate(_rehash_contract_payload(manifest_payload))
    return SafeDevelopmentDataset(manifest=manifest, examples=examples)


def test_v02_train_validation_inventory_has_exact_counts_and_round_trips(
    base_dataset: SafeDevelopmentDataset,
) -> None:
    assert base_dataset.manifest.example_count == 882
    assert base_dataset.manifest.counts_by_view == (
        (RemediationView.IID_TRAIN, 630),
        (RemediationView.IID_VALIDATION, 252),
    )
    assert base_dataset.manifest.counts_by_task == (
        (TaskName.CONTINUE_LOG, 70),
        (TaskName.FAULT_FAMILY, 203),
        (TaskName.EXTRACT_EVIDENCE, 203),
        (TaskName.NEXT_ACTION, 203),
        (TaskName.INCIDENT_SUMMARY, 203),
    )
    assert base_dataset.manifest.checksum_sha256 == (
        "8280c10ffa7c0694d43fbb72e49649f35546c9b490b0ef3be73d26b24a5b1eef"
    )
    assert {item.source_split for item in base_dataset.examples} == {
        SplitName.IID_TRAIN,
        SplitName.IID_VALIDATION,
    }

    for example in base_dataset.examples:
        parsed = parse_compact_target(example.compact_target, context=example.compact_context)
        assert parsed.task_name is example.task_name
        assert (
            compact_target_json(example.compact_target, context=example.compact_context)
            == example.canonical_target_json
        )
        if type(parsed) is PromptEvidenceTarget:
            positions = {
                fact_ref: index
                for index, fact_ref in enumerate(example.compact_context.visible_fact_refs)
            }
            assert parsed.fact_refs == tuple(sorted(parsed.fact_refs, key=positions.__getitem__))


def test_frozen_v03_bridge_reproduces_raw_and_deduplicated_manifests_once(
    frozen_v03_material: tuple[FrozenV03IIDMaterial, int],
) -> None:
    material, generator_calls = frozen_v03_material

    assert generator_calls == 1
    assert material.raw_dataset.manifest.example_count == 5_859
    assert material.raw_dataset.manifest.checksum_sha256 == (FROZEN_V03_RAW_MANIFEST_SHA256)
    assert material.dataset.manifest.example_count == 5_835
    assert material.dataset.manifest.checksum_sha256 == (FROZEN_V03_DEDUPLICATED_MANIFEST_SHA256)
    # The earlier 9fa4bf... observation used the test-only ``abcdef1`` source.
    # The scientifically correct frozen-source deduplicated manifest is 02ab7b....
    assert material.dataset.manifest.source_commit == FROZEN_V03_SOURCE_COMMIT
    assert len(material.structured_fingerprints) == 5_835


def test_renderer_visible_fingerprint_excludes_unrendered_model_input_metadata(
    scoped_train_validation: ScopedProjectionInventory,
) -> None:
    original = next(
        item.model_input
        for item in scoped_train_validation.projections
        if len(item.model_input.observation_facts) > 1
    )
    metadata_only_change = ModelInput(
        schema_version=original.schema_version,
        cut_tick=original.cut_tick + 1,
        source_event_index_exclusive=(
            None if original.source_event_index_exclusive is not None else 1
        ),
        observation_facts=original.observation_facts,
        event_facts=original.event_facts,
        context_facts=original.context_facts,
    )
    visible_fact_change = ModelInput(
        schema_version=original.schema_version,
        cut_tick=original.cut_tick,
        source_event_index_exclusive=original.source_event_index_exclusive,
        observation_facts=original.observation_facts[:-1],
        event_facts=original.event_facts,
        context_facts=original.context_facts,
    )

    assert original.structured_fingerprint() != metadata_only_change.structured_fingerprint()
    assert remediation_data.renderer_visible_structured_fingerprint(original) == (
        remediation_data.renderer_visible_structured_fingerprint(metadata_only_change)
    )
    assert remediation_data.renderer_visible_structured_fingerprint(original) != (
        remediation_data.renderer_visible_structured_fingerprint(visible_fact_change)
    )


def test_frozen_v03_dedup_is_exact_subset_with_only_evidence_duplicates_removed(
    frozen_v03_material: tuple[FrozenV03IIDMaterial, int],
) -> None:
    material, _generator_calls = frozen_v03_material
    raw_inventory = {
        (item.example_id, item.checksum_sha256): item for item in material.raw_dataset.examples
    }
    deduplicated_inventory = {
        (item.example_id, item.checksum_sha256): item for item in material.dataset.examples
    }
    removed_keys = set(raw_inventory) - set(deduplicated_inventory)

    assert set(deduplicated_inventory) < set(raw_inventory)
    assert len(removed_keys) == 24
    assert {
        (item.example_id, item.checksum_sha256) for item in material.removed_examples
    } == removed_keys
    retained_signatures = Counter(
        (item.task_name, item.prompt_sha256, item.canonical_target_json)
        for item in material.dataset.examples
    )
    assert all(
        item.augmentation == "remove_decisive_evidence"
        and item.task_name is not TaskName.COUNTERFACTUAL_COMPARE
        and retained_signatures[(item.task_name, item.prompt_sha256, item.canonical_target_json)]
        == 1
        for item in material.removed_examples
    )
    raw_prompt_counts = Counter(
        (item.task_name, item.prompt_sha256) for item in material.raw_dataset.examples
    )
    deduplicated_prompt_counts = Counter(
        (item.task_name, item.prompt_sha256) for item in material.dataset.examples
    )
    assert sum(count - 1 for count in raw_prompt_counts.values() if count > 1) == 24
    assert all(count == 1 for count in deduplicated_prompt_counts.values())


def test_frozen_v03_dedup_preserves_all_counterfactual_rows_and_view_counts(
    frozen_v03_material: tuple[FrozenV03IIDMaterial, int],
) -> None:
    material, _generator_calls = frozen_v03_material

    def counterfactual_inventory(
        dataset: SafeDevelopmentDataset,
    ) -> tuple[tuple[str, str, RemediationView], ...]:
        return tuple(
            (item.example_id, item.checksum_sha256, item.view)
            for item in dataset.examples
            if item.task_name is TaskName.COUNTERFACTUAL_COMPARE
        )

    raw = counterfactual_inventory(material.raw_dataset)
    deduplicated = counterfactual_inventory(material.dataset)
    counts = Counter(item[2] for item in raw)

    assert raw == deduplicated
    assert len(raw) == 55
    assert counts == {
        RemediationView.IID_TRAIN: 40,
        RemediationView.IID_VALIDATION: 15,
    }


def test_frozen_v03_bridge_is_iid_only_and_public_builders_remain_deduplicated(
    frozen_v03_material: tuple[FrozenV03IIDMaterial, int],
) -> None:
    material, _generator_calls = frozen_v03_material
    config = load_development_dataset_config(V03_DATASET_CONFIG)
    assert "views" not in inspect.signature(build_frozen_v03_iid_material).parameters

    def forbidden_generator(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid frozen request must fail before generation")

    with patch.object(
        remediation_data,
        "build_scoped_projection_inventory",
        side_effect=forbidden_generator,
    ):
        with pytest.raises(ValueError, match="immutable recipe"):
            build_frozen_v03_iid_material(
                config,
                source_commit=SOURCE_COMMIT,
                train_template_families=FROZEN_V03_TRAIN_TEMPLATE_FAMILIES,
                train_alias_families=FROZEN_V03_TRAIN_ALIAS_FAMILIES,
                renderer_variants_per_projection=3,
                include_insufficient_evidence_views=True,
            )

    mocked_material = (
        material.raw_dataset,
        material.dataset,
        material.structured_fingerprints,
        material.structured_fingerprints,
    )
    with patch.object(
        remediation_data,
        "_build_safe_development_material",
        return_value=mocked_material,
    ):
        public_dataset = build_safe_development_dataset(
            config,
            source_commit=FROZEN_V03_SOURCE_COMMIT,
            views=(RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION),
        )
        public_with_fingerprints = build_safe_development_dataset_with_structured_fingerprints(
            config,
            source_commit=FROZEN_V03_SOURCE_COMMIT,
            views=(RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION),
        )
    assert public_dataset is material.dataset
    assert public_with_fingerprints == (
        material.dataset,
        material.structured_fingerprints,
    )


def test_frozen_v03_material_and_deduplication_fail_closed_on_tamper(
    frozen_v03_material: tuple[FrozenV03IIDMaterial, int],
) -> None:
    material, _generator_calls = frozen_v03_material
    with pytest.raises(ValueError, match="raw or deduplicated manifest differs"):
        replace(material, dataset=material.raw_dataset)

    removed = material.removed_examples[0]
    retained = next(
        item
        for item in material.dataset.examples
        if (
            item.task_name,
            item.prompt_sha256,
            item.canonical_target_json,
        )
        == (
            removed.task_name,
            removed.prompt_sha256,
            removed.canonical_target_json,
        )
    )
    conflicting = removed.model_copy(update={"canonical_target_json": "{}"})
    with pytest.raises(ValueError, match="conflicting compact targets"):
        remediation_data._deduplicate_evidence_removal_examples(
            ((retained, "a" * 64), (conflicting, "a" * 64))
        )


def test_builder_requests_only_the_explicit_train_validation_splits(
    v01_config: DevelopmentDatasetConfig,
    scoped_train_validation: ScopedProjectionInventory,
) -> None:
    observed: list[tuple[SplitName, ...]] = []

    def scoped_only(
        config: DevelopmentDatasetConfig,
        *,
        generator_commit: str,
        splits: tuple[SplitName, ...],
    ) -> ScopedProjectionInventory:
        assert config is v01_config
        assert generator_commit == SOURCE_COMMIT
        observed.append(splits)
        return scoped_train_validation

    with patch.object(
        remediation_data,
        "build_scoped_projection_inventory",
        side_effect=scoped_only,
    ):
        dataset = build_safe_development_dataset(
            v01_config,
            source_commit=SOURCE_COMMIT,
            views=(RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION),
        )

    assert observed == [(SplitName.IID_TRAIN, SplitName.IID_VALIDATION)]
    assert dataset.manifest.views == (
        RemediationView.IID_TRAIN,
        RemediationView.IID_VALIDATION,
    )


def test_builder_refuses_any_mapping_to_iid_test(
    monkeypatch: pytest.MonkeyPatch,
    v01_config: DevelopmentDatasetConfig,
) -> None:
    monkeypatch.setitem(
        VIEW_SOURCE_SPLIT,
        RemediationView.IID_TRAIN,
        SplitName.IID_TEST,
    )

    def forbidden(*_args: object, **_kwargs: object) -> ScopedProjectionInventory:
        raise AssertionError("scoped builder must not run after an IID_TEST request")

    monkeypatch.setattr(remediation_data, "build_scoped_projection_inventory", forbidden)
    with pytest.raises(ValueError, match="cannot request a final IID test"):
        build_safe_development_dataset(
            v01_config,
            source_commit=SOURCE_COMMIT,
            views=(RemediationView.IID_TRAIN,),
        )


def test_safe_builder_never_calls_mixed_artifact_or_golden_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    v01_config: DevelopmentDatasetConfig,
    scoped_train_validation: ScopedProjectionInventory,
) -> None:
    def prohibited(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("prohibited mixed/final/golden surface was called")

    prohibited_surfaces: tuple[tuple[object, str], ...] = (
        (dataset_pipeline, "build_development_projection_bundle"),
        (dataset_package, "build_development_projection_bundle"),
        (historical_development, "build_review_gated_development_candidate"),
        (historical_development, "verify_development_candidate_artifact"),
        (historical_development, "write_and_verify_development_candidate"),
        (golden_evaluation, "load_golden_review_packet"),
        (golden_evaluation, "load_golden_review_record"),
        (golden_evaluation, "prepare_golden_review_packet"),
    )
    for module, name in prohibited_surfaces:
        monkeypatch.setattr(module, name, prohibited)
    monkeypatch.setattr(
        remediation_data,
        "build_scoped_projection_inventory",
        lambda *_args, **_kwargs: scoped_train_validation,
    )

    dataset = build_safe_development_dataset(
        v01_config,
        source_commit=SOURCE_COMMIT,
        views=(RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION),
    )
    assert dataset.manifest.example_count == 882

    source = inspect.getsource(remediation_data)
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        name.endswith((".golden", ".development", ".artifacts"))
        or name.startswith("reactorbench.training")
        for name in imported_modules
    )


def test_augmentation_is_deterministic_group_atomic_and_target_text_free(
    v01_config: DevelopmentDatasetConfig,
    scoped_train_validation: ScopedProjectionInventory,
) -> None:
    train = _inventory_for_split(scoped_train_validation, SplitName.IID_TRAIN)

    def build() -> SafeDevelopmentDataset:
        with patch.object(
            remediation_data,
            "build_scoped_projection_inventory",
            return_value=train,
        ):
            return build_safe_development_dataset(
                v01_config,
                source_commit=SOURCE_COMMIT,
                views=(RemediationView.IID_TRAIN,),
                train_template_families=tuple(item.value for item in TemplateFamily)[:3],
                train_alias_families=tuple(item.value for item in AliasFamily)[:3],
                renderer_variants_per_projection=3,
                include_insufficient_evidence_views=True,
            )

    first = build()
    second = build()
    assert first == second
    assert first.manifest.checksum_sha256 == second.manifest.checksum_sha256

    style_rows = tuple(
        item
        for item in first.examples
        if item.augmentation in {"none", "renderer_variant"}
        and item.task_name is not TaskName.COUNTERFACTUAL_COMPARE
    )
    by_source: dict[tuple[str, ...], list[RemediationExample]] = defaultdict(list)
    for item in style_rows:
        by_source[item.source_record_ids].append(item)
    assert by_source
    assert {len(rows) for rows in by_source.values()} == {3}
    for rows in by_source.values():
        assert len({item.group_id for item in rows}) == 1
        assert len({item.task_name for item in rows}) == 1
        assert {item.augmentation for item in rows} == {"none", "renderer_variant"}

    evidence_removals = tuple(
        item for item in first.examples if item.augmentation == "remove_decisive_evidence"
    )
    assert evidence_removals
    assert all(item.group_id.startswith("evidence-removal:") for item in evidence_removals)
    for item in evidence_removals:
        target = parse_compact_target(item.compact_target, context=item.compact_context)
        if type(target) is FaultDiagnosisTarget:
            assert target.diagnosis_status is DiagnosisStatus.UNRESOLVED
            assert target.fault_labels == ()
            assert target.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE
        elif type(target) is NextActionTarget:
            assert target.immediate_action is ActionLabel.INSUFFICIENT_EVIDENCE
        elif type(target) is PromptEvidenceTarget:
            assert target.fact_refs == ()
            assert target.evidence_slots == ()
        else:
            assert type(target) is IncidentSummaryTarget
            assert target.diagnosis_status is DiagnosisStatus.UNRESOLVED
            assert target.immediate_action is ActionLabel.INSUFFICIENT_EVIDENCE
            assert target.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE

    for item in first.examples:
        assert item.compact_target not in item.prompt_text
        assert item.canonical_target_json not in item.prompt_text
        if item.classification_label is not None:
            assert item.classification_label not in item.prompt_text


def test_remediation_audit_accepts_the_untampered_development_dataset(
    base_dataset: SafeDevelopmentDataset,
) -> None:
    report = audit_safe_development_dataset(base_dataset)
    replay = audit_safe_development_dataset(base_dataset)

    assert report.passed
    assert report.target_text_leakage_count == 0
    assert replay == report
    for view_audit in report.views:
        view_examples = tuple(
            item for item in base_dataset.examples if item.view is view_audit.view
        )
        task_counts = dict(view_audit.task_counts)
        assert sum(task_counts.values()) == view_audit.example_count
        for inventory in view_audit.class_inventories:
            expected = Counter(
                item.classification_label
                for item in view_examples
                if item.task_name is inventory.task_name and item.classification_label is not None
            )
            assert inventory.counts == tuple(sorted(expected.items()))
            assert inventory.total == task_counts[inventory.task_name]


def test_remediation_audit_rejects_inconsistent_class_inventory(
    base_dataset: SafeDevelopmentDataset,
) -> None:
    report = audit_safe_development_dataset(base_dataset)
    payload = report.model_dump(mode="python", round_trip=True)
    first_view = cast(dict[str, object], cast(list[object], payload["views"])[0])
    inventories = cast(list[object], first_view["class_inventories"])
    first_inventory = cast(dict[str, object], inventories[0])
    counts = cast(tuple[tuple[str, int], ...], first_inventory["counts"])
    first_inventory["counts"] = (
        (counts[0][0], counts[0][1] + 1),
        *counts[1:],
    )
    first_inventory["total"] = cast(int, first_inventory["total"]) + 1

    with pytest.raises(ValidationError, match="do not cover their task counts"):
        type(report).model_validate(payload)


@pytest.mark.parametrize(
    "surface_style",
    ["underscore_lower", "hyphen_mixed", "space_lower"],
)
def test_remediation_audit_detects_partial_classification_label_variants(
    base_dataset: SafeDevelopmentDataset,
    surface_style: str,
) -> None:
    example = next(
        item
        for item in base_dataset.examples
        if item.task_name is TaskName.FAULT_FAMILY
        and item.classification_label is not None
        and item.classification_label.startswith("DIAGNOSED:")
    )
    target = parse_compact_target(example.compact_target, context=example.compact_context)
    assert type(target) is FaultDiagnosisTarget
    label = target.fault_labels[0].value
    surfaces = {
        "underscore_lower": label.lower(),
        "hyphen_mixed": label.replace("_", "-").title(),
        "space_lower": label.replace("_", " ").lower(),
    }
    partial_label = surfaces[surface_style]
    assert example.classification_label != partial_label
    tampered = _with_prompt_text(
        base_dataset,
        example,
        f"{example.prompt_text}\nLeaked answer fragment: [{partial_label}]",
    )

    report = audit_safe_development_dataset(tampered)

    assert not report.passed
    assert report.target_text_leakage_count == 1


def test_remediation_audit_detects_partial_nonclassification_target_label(
    base_dataset: SafeDevelopmentDataset,
) -> None:
    example = next(
        item
        for item in base_dataset.examples
        if item.task_name is TaskName.INCIDENT_SUMMARY
        and item.classification_label is None
        and type(parse_compact_target(item.compact_target, context=item.compact_context))
        is IncidentSummaryTarget
    )
    target = parse_compact_target(example.compact_target, context=example.compact_context)
    assert type(target) is IncidentSummaryTarget
    partial_label = target.immediate_action.value.replace("_", "-").lower()
    tampered = _with_prompt_text(
        base_dataset,
        example,
        f"{example.prompt_text}\nLeaked structured field: {partial_label}.",
    )

    report = audit_safe_development_dataset(tampered)

    assert not report.passed
    assert report.target_text_leakage_count == 1


@pytest.mark.parametrize(
    ("task_name", "field_name"),
    [
        (TaskName.CONTINUE_LOG, "next_event_type"),
        (TaskName.FAULT_FAMILY, "diagnosis_status"),
        (TaskName.FAULT_FAMILY, "abstention_reason"),
        (TaskName.NEXT_ACTION, "immediate_action"),
    ],
)
def test_remediation_audit_covers_each_leak_sensitive_answer_field(
    base_dataset: SafeDevelopmentDataset,
    task_name: TaskName,
    field_name: str,
) -> None:
    candidates = tuple(item for item in base_dataset.examples if item.task_name is task_name)
    example = next(
        item
        for item in candidates
        if getattr(
            parse_compact_target(item.compact_target, context=item.compact_context),
            field_name,
            None,
        )
        is not None
    )
    target = parse_compact_target(example.compact_target, context=example.compact_context)
    value = getattr(target, field_name)
    label = value.value.replace("_", " ").lower()
    tampered = _with_prompt_text(
        base_dataset,
        example,
        f"{example.prompt_text}\nLeaked field value: {label}.",
    )

    assert audit_safe_development_dataset(tampered).target_text_leakage_count == 1


def test_remediation_audit_uses_word_boundaries_and_ignores_prompt_evidence_labels(
    base_dataset: SafeDevelopmentDataset,
) -> None:
    classification = next(
        item
        for item in base_dataset.examples
        if item.task_name is TaskName.FAULT_FAMILY
        and item.classification_label is not None
        and item.classification_label.startswith("DIAGNOSED:")
    )
    diagnosis = parse_compact_target(
        classification.compact_target,
        context=classification.compact_context,
    )
    assert type(diagnosis) is FaultDiagnosisTarget
    embedded = diagnosis.fault_labels[0].value.replace("_", "-").lower()
    boundary_safe = _with_prompt_text(
        base_dataset,
        classification,
        f"{classification.prompt_text}\nnot{embedded}suffix",
    )
    assert audit_safe_development_dataset(boundary_safe).target_text_leakage_count == 0

    evidence = next(
        item for item in base_dataset.examples if item.task_name is TaskName.EXTRACT_EVIDENCE
    )
    evidence_target = parse_compact_target(
        evidence.compact_target,
        context=evidence.compact_context,
    )
    assert type(evidence_target) is PromptEvidenceTarget
    assert evidence_target.evidence_slots
    prompt_visible_label = evidence_target.evidence_slots[0].value.replace("_", " ").lower()
    intended_evidence = _with_prompt_text(
        base_dataset,
        evidence,
        f"{evidence.prompt_text}\nVisible evidence phrase: {prompt_visible_label}.",
    )
    assert audit_safe_development_dataset(intended_evidence).target_text_leakage_count == 0

    summary = next(
        item
        for item in base_dataset.examples
        if item.task_name is TaskName.INCIDENT_SUMMARY
        and getattr(
            parse_compact_target(
                item.compact_target,
                context=item.compact_context,
            ),
            "operating_mode",
            None,
        )
        is OperatingMode.STABLE
    )
    common_context_word = _with_prompt_text(
        base_dataset,
        summary,
        f"{summary.prompt_text}\nThe fictional station context remains stable.",
    )
    assert audit_safe_development_dataset(common_context_word).target_text_leakage_count == 0


@pytest.mark.parametrize(
    "views",
    [
        [RemediationView.IID_TRAIN],
        (),
        ("iid_train",),
    ],
)
def test_builder_rejects_nonexact_or_empty_view_tuples(
    views: object,
    v01_config: DevelopmentDatasetConfig,
) -> None:
    with pytest.raises(TypeError, match="non-empty exact"):
        build_safe_development_dataset(
            v01_config,
            source_commit=SOURCE_COMMIT,
            views=cast(tuple[RemediationView, ...], views),
        )


def test_builder_rejects_duplicate_views_bad_commit_and_unsafe_augmentation(
    v01_config: DevelopmentDatasetConfig,
) -> None:
    with pytest.raises(ValueError, match="must be unique"):
        build_safe_development_dataset(
            v01_config,
            source_commit=SOURCE_COMMIT,
            views=(RemediationView.IID_TRAIN, RemediationView.IID_TRAIN),
        )
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        build_safe_development_dataset(
            v01_config,
            source_commit="ABCDEF1",
            views=(RemediationView.IID_TRAIN,),
        )
    with pytest.raises(ValueError, match=r"in \[1,9\]"):
        build_safe_development_dataset(
            v01_config,
            source_commit=SOURCE_COMMIT,
            views=(RemediationView.IID_TRAIN,),
            renderer_variants_per_projection=True,
        )
    with pytest.raises(ValueError, match="requires explicit"):
        build_safe_development_dataset(
            v01_config,
            source_commit=SOURCE_COMMIT,
            views=(RemediationView.IID_TRAIN,),
            renderer_variants_per_projection=2,
        )


def test_data_contracts_reject_unknown_fields_and_tampering(
    base_dataset: SafeDevelopmentDataset,
) -> None:
    example = base_dataset.examples[0]
    raw_example = example.model_dump(mode="python", round_trip=True)
    raw_example["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RemediationExample.model_validate(raw_example)

    bad_prompt = example.model_copy(update={"prompt_text": example.prompt_text + " tampered"})
    with pytest.raises(ValidationError, match="prompt checksum mismatch"):
        RemediationExample.model_validate(bad_prompt.model_dump(mode="python", round_trip=True))

    manifest = base_dataset.manifest.model_dump(mode="python", round_trip=True)
    manifest["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SafeDevelopmentManifest.model_validate(manifest)


def test_manifest_rejects_self_rehashed_noncanonical_count_keys(
    base_dataset: SafeDevelopmentDataset,
) -> None:
    payload = base_dataset.manifest.model_dump(mode="python", round_trip=True)
    payload["counts_by_task"] = tuple(reversed(cast(tuple[object, ...], payload["counts_by_task"])))
    with pytest.raises(ValidationError, match="canonical"):
        SafeDevelopmentManifest.model_validate(_rehash_contract_payload(payload))

    payload = base_dataset.manifest.model_dump(mode="python", round_trip=True)
    counts = list(cast(tuple[tuple[RemediationView, int], ...], payload["counts_by_view"]))
    counts[1] = (counts[0][0], counts[1][1])
    payload["counts_by_view"] = tuple(counts)
    with pytest.raises(ValidationError, match="canonical view order"):
        SafeDevelopmentManifest.model_validate(_rehash_contract_payload(payload))


def test_dataset_recomputes_self_rehashed_view_and_task_count_values(
    base_dataset: SafeDevelopmentDataset,
) -> None:
    view_payload = base_dataset.manifest.model_dump(mode="python", round_trip=True)
    view_counts = list(
        cast(tuple[tuple[RemediationView, int], ...], view_payload["counts_by_view"])
    )
    view_counts[0] = (view_counts[0][0], view_counts[0][1] + 1)
    view_counts[1] = (view_counts[1][0], view_counts[1][1] - 1)
    view_payload["counts_by_view"] = tuple(view_counts)
    forged_view_manifest = SafeDevelopmentManifest.model_validate(
        _rehash_contract_payload(view_payload)
    )
    with pytest.raises(ValidationError, match="view counts differ"):
        SafeDevelopmentDataset(
            manifest=forged_view_manifest,
            examples=base_dataset.examples,
        )

    task_payload = base_dataset.manifest.model_dump(mode="python", round_trip=True)
    task_counts = list(cast(tuple[tuple[TaskName, int], ...], task_payload["counts_by_task"]))
    task_counts[0] = (task_counts[0][0], task_counts[0][1] + 1)
    task_counts[1] = (task_counts[1][0], task_counts[1][1] - 1)
    task_payload["counts_by_task"] = tuple(task_counts)
    forged_task_manifest = SafeDevelopmentManifest.model_validate(
        _rehash_contract_payload(task_payload)
    )
    with pytest.raises(ValidationError, match="task counts differ"):
        SafeDevelopmentDataset(
            manifest=forged_task_manifest,
            examples=base_dataset.examples,
        )


def test_dataset_recomputes_self_rehashed_jsonl_byte_binding(
    base_dataset: SafeDevelopmentDataset,
) -> None:
    payload = base_dataset.manifest.model_dump(mode="python", round_trip=True)
    payload["examples_sha256"] = "0" * 64
    forged_manifest = SafeDevelopmentManifest.model_validate(_rehash_contract_payload(payload))
    with pytest.raises(ValidationError, match="canonical payload differs"):
        SafeDevelopmentDataset(
            manifest=forged_manifest,
            examples=base_dataset.examples,
        )


def test_safe_artifact_roundtrip_nonoverwrite_tamper_extra_and_symlink_failures(
    tmp_path: Path,
    base_dataset: SafeDevelopmentDataset,
) -> None:
    artifact = tmp_path / "safe"
    written = write_safe_development_artifact(base_dataset, artifact)
    assert written == base_dataset.manifest
    assert load_safe_development_artifact(artifact) == base_dataset
    with pytest.raises(FileExistsError, match="new regular directory"):
        write_safe_development_artifact(base_dataset, artifact)

    tampered = tmp_path / "tampered"
    write_safe_development_artifact(base_dataset, tampered)
    examples_path = tampered / "examples.jsonl"
    payload = examples_path.read_bytes()
    examples_path.write_bytes(payload[:-2] + b"X\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_safe_development_artifact(tampered)

    extra = tmp_path / "extra"
    write_safe_development_artifact(base_dataset, extra)
    (extra / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory is missing or unsafe"):
        load_safe_development_artifact(extra)

    linked_directory = tmp_path / "linked-directory"
    linked_directory.symlink_to(artifact, target_is_directory=True)
    with pytest.raises(ValueError, match="inventory is missing or unsafe"):
        load_safe_development_artifact(linked_directory)

    linked_member = tmp_path / "linked-member"
    write_safe_development_artifact(base_dataset, linked_member)
    external_manifest = tmp_path / "external-manifest.json"
    external_manifest.write_bytes((linked_member / "manifest.json").read_bytes())
    (linked_member / "manifest.json").unlink()
    (linked_member / "manifest.json").symlink_to(external_manifest)
    with pytest.raises(ValueError, match="symlink or non-file"):
        load_safe_development_artifact(linked_member)

    dangling = tmp_path / "dangling-output"
    dangling.symlink_to(tmp_path / "does-not-exist", target_is_directory=True)
    with pytest.raises(FileExistsError, match="new regular directory"):
        write_safe_development_artifact(base_dataset, dangling)


def test_safe_artifact_size_bounds_fail_before_parsing_untrusted_payloads(
    tmp_path: Path,
    base_dataset: SafeDevelopmentDataset,
) -> None:
    manifest_too_large = tmp_path / "manifest-too-large"
    write_safe_development_artifact(base_dataset, manifest_too_large)
    (manifest_too_large / "manifest.json").write_bytes(b"{" + b" " * (256 * 1024) + b"}")
    with pytest.raises(ValueError, match="manifest exceeds its size bound"):
        load_safe_development_artifact(manifest_too_large)

    row_too_large = tmp_path / "row-too-large"
    write_safe_development_artifact(base_dataset, row_too_large)
    oversized_payload = b"{" + b" " * MAX_EXAMPLE_BYTES + b"}\n"
    (row_too_large / "examples.jsonl").write_bytes(oversized_payload)
    manifest = _rebuilt_manifest(
        base_dataset.manifest,
        examples_sha256=hashlib.sha256(oversized_payload).hexdigest(),
        examples_size_bytes=len(oversized_payload),
    )
    _write_manifest(row_too_large / "manifest.json", manifest)
    with pytest.raises(ValueError, match="JSONL row exceeds its bound"):
        load_safe_development_artifact(row_too_large)


def test_safe_artifact_api_requires_exact_contracts_and_paths(
    tmp_path: Path,
    base_dataset: SafeDevelopmentDataset,
) -> None:
    with pytest.raises(TypeError, match="exact dataset and Path"):
        write_safe_development_artifact(
            cast(SafeDevelopmentDataset, object()),
            tmp_path / "unsafe-dataset",
        )
    with pytest.raises(TypeError, match="exact dataset and Path"):
        write_safe_development_artifact(
            base_dataset,
            cast(Path, str(tmp_path / "unsafe-path")),
        )
    with pytest.raises(ValueError, match="inventory is missing or unsafe"):
        load_safe_development_artifact(tmp_path / "missing")
    with pytest.raises(ValueError, match="inventory is missing or unsafe"):
        load_safe_development_artifact(cast(Path, "not-a-path"))
