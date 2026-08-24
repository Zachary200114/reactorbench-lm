"""Isolation, augmentation, and safe-artifact tests for remediation data."""

from __future__ import annotations

import ast
import hashlib
import inspect
from collections import defaultdict
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
from reactorbench.dataset.contracts import PromptEvidenceTarget
from reactorbench.dataset.pipeline import (
    ScopedProjectionInventory,
    build_scoped_projection_inventory,
)
from reactorbench.evaluation.compact import compact_target_json, parse_compact_target
from reactorbench.remediation.config import VIEW_SOURCE_SPLIT, RemediationView
from reactorbench.remediation.data import (
    MAX_EXAMPLE_BYTES,
    RemediationExample,
    SafeDevelopmentDataset,
    SafeDevelopmentManifest,
    build_safe_development_dataset,
    load_safe_development_artifact,
    write_safe_development_artifact,
)
from reactorbench.schemas.base import canonical_json_bytes, canonical_sha256
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

ROOT = Path(__file__).resolve().parents[2]
V01_DATASET_CONFIG = ROOT / "configs/dataset/development-v0.1.0.toml"
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
