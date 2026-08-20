from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from reactorbench.dataset.artifacts import (
    ArtifactFile,
    ArtifactVerificationError,
    ArtifactWriter,
    CandidateArtifactManifest,
)
from reactorbench.dataset.config import load_development_dataset_config
from reactorbench.dataset.contracts import ProjectionRecord, PromptEvidenceTarget
from reactorbench.dataset.development import (
    DevelopmentCandidateBundle,
    _require_candidate_bounds,
    _shortcut_target_labels,
    _task_shortcut_records,
    build_review_gated_development_candidate,
    candidate_artifact_metadata,
    candidate_artifact_records,
    verify_development_candidate_artifact,
    write_and_verify_development_candidate,
)
from reactorbench.dataset.pipeline import build_development_projection_bundle
from reactorbench.dataset.quality import QualityRecord
from reactorbench.dataset.review import (
    ReviewConfirmations,
    ReviewDecision,
    create_review_record,
    prepare_catalog_review_packet,
)
from reactorbench.schemas.base import canonical_json_bytes
from reactorbench.schemas.enums import SplitName, TaskName


def _confirmations() -> ReviewConfirmations:
    return ReviewConfirmations(
        all_preview_entries_reviewed=True,
        structured_answers_reviewed_separately=True,
        no_real_facility_or_procedure_content=True,
        no_navy_or_service_derived_content=True,
        no_operational_or_security_instructions=True,
        no_unfinished_templates_or_shortcuts=True,
        fingerprint_registry_reviewed=True,
    )


@pytest.fixture(scope="module")
def candidate() -> DevelopmentCandidateBundle:
    config = load_development_dataset_config(Path("configs/dataset/development-v0.1.0.toml"))
    structured = build_development_projection_bundle(config, generator_commit="abcdef0")
    packet = prepare_catalog_review_packet(structured)
    # This is a test-only synthetic approval record. Production requires an explicit
    # project-owner review record bound to the generated packet.
    record = create_review_record(
        packet,
        reviewer_role="project-owner",
        review_date=date(2026, 8, 20),
        decision=ReviewDecision.APPROVED,
        confirmations=_confirmations(),
        notes=("test-only gate fixture",),
    )
    return build_review_gated_development_candidate(
        config,
        structured_bundle=structured,
        review_packet=packet,
        review_record=record,
    )


def test_full_review_gated_candidate_inventory_and_quality(
    candidate: DevelopmentCandidateBundle,
) -> None:
    assert candidate.artifact_status == "candidate_pending_postrender_review"
    assert len(candidate.structured_bundle.trajectories) == 204
    assert len(candidate.structured_bundle.projections) == 1_762
    assert len(candidate.structured_bundle.counterfactual_projections) == 14
    assert len(candidate.rendered_candidates) == 553
    assert len(candidate.task_examples) == 1_776
    assert len(candidate.task_shortcut_records) == 1_776
    assert tuple(record.record_id for record in candidate.task_shortcut_records) == tuple(
        example.example_id for example in candidate.task_examples
    )
    assert candidate.quality_report.record_count == len(candidate.rendered_candidates)
    assert candidate.quality_report.task_record_count == len(candidate.task_examples)
    assert tuple(
        record.record_id for record in candidate.quality_report.audited_task_records
    ) == tuple(example.example_id for example in candidate.task_examples)
    assert not candidate.quality_report.exact_duplicates
    assert not candidate.quality_report.forbidden_skeleton_duplicates
    assert not candidate.quality_report.shortcut_findings
    assert candidate.quality_report.shortcut_contingencies
    assert any(
        item.feature_class == "semantic_context"
        for item in candidate.quality_report.shortcut_contingencies
    )
    corruption_contingencies = tuple(
        item
        for item in candidate.quality_report.shortcut_contingencies
        if item.feature_value.startswith("corruption:")
    )
    assert any(item.feature_value == "corruption:none" for item in corruption_contingencies)
    for task_name in TaskName:
        assert sum(
            item.support for item in corruption_contingencies if item.task_name is task_name
        ) == sum(example.task_name is task_name for example in candidate.task_examples)
    assert not candidate.quality_report.target_text_findings
    assert not candidate.quality_report.provenance_issues
    assert candidate.quality_report.skeleton_duplicates
    assert candidate.quality_report.passed
    assert candidate.postrender_review_packet.human_review_required
    assert candidate.postrender_review_packet.candidate_count == 553


def test_task_examples_have_exact_prompt_and_provenance_arity(
    candidate: DevelopmentCandidateBundle,
) -> None:
    by_task: dict[TaskName, int] = dict.fromkeys(TaskName, 0)
    for example in candidate.task_examples:
        by_task[example.task_name] += 1
        expected = 2 if example.task_name is TaskName.COUNTERFACTUAL_COMPARE else 1
        assert len(example.prompt_render_ids) == expected
        assert len(example.provenance_records) == expected
        assert all(record.task_name is example.task_name for record in example.provenance_records)
    assert by_task[TaskName.COUNTERFACTUAL_COMPARE] == 14
    assert sum(by_task.values()) == 1_776


def test_noise_candidate_contains_all_four_bounded_corruption_types(
    candidate: DevelopmentCandidateBundle,
) -> None:
    plans = {record.corruption_plan.value for record in candidate.corruption_records}
    assert plans == {
        "benign_insert",
        "duplicate_line",
        "omit_noncritical",
        "safe_reorder",
    }
    noise_examples = tuple(
        example for example in candidate.task_examples if example.split_name is SplitName.NOISE_TEST
    )
    assert noise_examples
    assert all(example.corruption_ids for example in noise_examples)


def test_candidate_artifact_inventory_is_complete_and_flat(
    candidate: DevelopmentCandidateBundle,
) -> None:
    records = candidate_artifact_records(candidate)
    assert set(records) == {
        "candidate-summary.jsonl",
        "candidate-metadata.jsonl",
        "corruptions.jsonl",
        "counterfactual-projections.jsonl",
        "groups.jsonl",
        "postrender-review.jsonl",
        "pre-render-review-packet.jsonl",
        "pre-render-review-record.jsonl",
        "projections.jsonl",
        "quality-report.jsonl",
        "rendered-candidates.jsonl",
        "resolved-config.jsonl",
        "split-manifest.jsonl",
        "task-examples.jsonl",
        "task-shortcut-records.jsonl",
        "trajectories.jsonl",
    }
    assert all(name.endswith(".jsonl") and values for name, values in records.items())
    metadata = candidate_artifact_metadata(candidate)
    assert metadata.candidate_bundle_sha256 == candidate.checksum_sha256
    assert metadata.structured_bundle_sha256 == candidate.structured_bundle.checksum_sha256


def test_unapproved_catalog_record_stops_before_rendering() -> None:
    config = load_development_dataset_config(Path("configs/dataset/development-v0.1.0.toml"))
    structured = build_development_projection_bundle(config, generator_commit="abcdef0")
    packet = prepare_catalog_review_packet(structured)
    record = create_review_record(
        packet,
        reviewer_role="project-owner",
        review_date=date(2026, 8, 20),
        decision=ReviewDecision.REVISE,
        confirmations=_confirmations(),
    )
    with pytest.raises(ValueError, match="approved"):
        build_review_gated_development_candidate(
            config,
            structured_bundle=structured,
            review_packet=packet,
            review_record=record,
        )


def test_configured_project_owner_role_is_enforced_before_build() -> None:
    config = load_development_dataset_config(Path("configs/dataset/development-v0.1.0.toml"))
    structured = build_development_projection_bundle(config, generator_commit="abcdef0")
    packet = prepare_catalog_review_packet(structured)
    with pytest.raises(ValueError, match="project-owner"):
        create_review_record(
            packet,
            reviewer_role="test fixture reviewer",  # type: ignore[arg-type]
            review_date=date(2026, 8, 20),
            decision=ReviewDecision.APPROVED,
            confirmations=_confirmations(),
        )


def test_task_shortcut_audit_classifies_semantic_context_and_requires_common_pair_plan(
    candidate: DevelopmentCandidateBundle,
) -> None:
    pair = next(
        example
        for example in candidate.task_examples
        if example.task_name is TaskName.COUNTERFACTUAL_COMPARE
    )
    quality = tuple(
        QualityRecord(
            example_id=render_id,
            split_name=pair.split_name,
            text="bounded synthetic prompt",
            template_family_id="compact-log-v1",
            alias_family_id="canonical-v1",
            context_flags=(
                "semantic:standby-context:present",
                "corruption:safe_reorder",
            ),
            provenance={},
        )
        for render_id in pair.prompt_render_ids
    )

    (shortcut,) = _task_shortcut_records((pair,), quality)

    assert shortcut.prompt_render_ids == tuple(sorted(pair.prompt_render_ids))
    assert shortcut.template_family_id == "compact-log-v1"
    assert shortcut.alias_family_id == "canonical-v1"
    assert shortcut.context_flags == (
        "corruption:safe_reorder",
        "semantic:standby-context:present",
    )

    incompatible = (quality[0], quality[1].model_copy(update={"alias_family_id": "neutral-v1"}))
    with pytest.raises(ValueError, match="share one renderer plan"):
        _task_shortcut_records((pair,), incompatible)


def test_shortcut_targets_include_structured_evidence_categories(
    candidate: DevelopmentCandidateBundle,
) -> None:
    example = next(
        item for item in candidate.task_examples if item.task_name is TaskName.EXTRACT_EVIDENCE
    )
    assert isinstance(example.task_target.target, PromptEvidenceTarget)

    labels = set(_shortcut_target_labels(example.task_target))

    assert {
        f"evidence_slots.{index}={slot.value}"
        for index, slot in enumerate(example.task_target.target.evidence_slots)
    } <= labels


def test_reviewed_task_and_rendered_byte_bounds_are_enforced(
    candidate: DevelopmentCandidateBundle,
) -> None:
    rendered_bytes = sum(
        len(rendered.text.encode("utf-8")) for rendered in candidate.rendered_candidates
    )
    with pytest.raises(ValueError, match="task count"):
        _require_candidate_bounds(
            candidate.rendered_candidates,
            candidate.task_examples,
            maximum_task_records=len(candidate.task_examples) - 1,
            maximum_rendered_bytes=rendered_bytes,
        )
    with pytest.raises(ValueError, match="rendered text"):
        _require_candidate_bounds(
            candidate.rendered_candidates,
            candidate.task_examples,
            maximum_task_records=len(candidate.task_examples),
            maximum_rendered_bytes=rendered_bytes - 1,
        )


def test_full_artifact_round_trip_and_cross_file_config_tamper_detection(
    candidate: DevelopmentCandidateBundle,
    tmp_path: Path,
) -> None:
    writer = ArtifactWriter(tmp_path)
    verified = write_and_verify_development_candidate(
        writer,
        relative_directory="development-candidate",
        candidate=candidate,
    )

    assert verified.candidate == candidate
    assert len(verified.manifest.files) == 16
    assert verified.metadata.candidate_bundle_sha256 == candidate.checksum_sha256

    bundle = tmp_path / "development-candidate"
    config_path = bundle / "resolved-config.jsonl"
    changed_dataset = candidate.resolved_config.dataset.model_copy(
        update={
            "maximum_rendered_bytes": (candidate.resolved_config.dataset.maximum_rendered_bytes + 1)
        }
    )
    changed_config = candidate.resolved_config.model_copy(update={"dataset": changed_dataset})
    payload = canonical_json_bytes(changed_config.model_dump(mode="json")) + b"\n"
    config_path.write_bytes(payload)

    manifest_path = bundle / "manifest.json"
    manifest = CandidateArtifactManifest.model_validate_json(manifest_path.read_bytes())
    files = tuple(
        ArtifactFile(
            filename=item.filename,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            record_count=1,
        )
        if item.filename == "resolved-config.jsonl"
        else item
        for item in manifest.files
    )
    changed_manifest = manifest.model_copy(update={"files": files})
    manifest_path.write_bytes(
        canonical_json_bytes(changed_manifest.model_dump(mode="json")) + b"\n"
    )

    with pytest.raises(ArtifactVerificationError, match="cross-file"):
        verify_development_candidate_artifact(
            writer,
            relative_directory="development-candidate",
        )


def test_typed_artifact_reconstruction_rejects_a_rehashed_nonexistent_projection_source(
    candidate: DevelopmentCandidateBundle,
    tmp_path: Path,
) -> None:
    """A checksummed JSONL replacement must still prove its trajectory relationship."""

    writer = ArtifactWriter(tmp_path)
    relative_directory = "development-candidate"
    write_and_verify_development_candidate(
        writer,
        relative_directory=relative_directory,
        candidate=candidate,
    )
    original = candidate.structured_bundle.projections[0]
    lineage = original.lineage.model_copy(
        update={"trajectory_id": "trajectory:nonexistent-artifact-source"}
    )
    draft = original.model_copy(update={"lineage": lineage})
    # Use the actual canonical checksum, not a malformed local record, so this is a
    # relational—not syntax or checksum—reconstruction probe.
    tampered = draft.model_copy(
        update={
            "checksum_sha256": hashlib.sha256(
                canonical_json_bytes(
                    draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
                )
            ).hexdigest()
        }
    )
    assert isinstance(tampered, ProjectionRecord)

    artifact_directory = tmp_path / relative_directory
    projection_path = artifact_directory / "projections.jsonl"
    rows = [json.loads(line) for line in projection_path.read_bytes().splitlines()]
    for index, row in enumerate(rows):
        if row["projection_id"] == original.projection_id:
            rows[index] = tampered.model_dump(mode="json", round_trip=True)
            break
    else:  # pragma: no cover - the writer inventory is independently tested.
        raise AssertionError("test projection was not present in its artifact JSONL")
    payload = b"\n".join(canonical_json_bytes(row) for row in rows) + b"\n"
    projection_path.write_bytes(payload)

    manifest_path = artifact_directory / "manifest.json"
    manifest = CandidateArtifactManifest.model_validate_json(manifest_path.read_bytes())
    files = tuple(
        ArtifactFile(
            filename=item.filename,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            record_count=len(rows),
        )
        if item.filename == "projections.jsonl"
        else item
        for item in manifest.files
    )
    manifest_path.write_bytes(
        canonical_json_bytes(manifest.model_copy(update={"files": files}).model_dump(mode="json"))
        + b"\n"
    )

    with pytest.raises(ArtifactVerificationError, match="cross-file"):
        verify_development_candidate_artifact(
            writer,
            relative_directory=relative_directory,
        )
