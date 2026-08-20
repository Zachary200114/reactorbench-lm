from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

import reactorbench.dataset.renderer as renderer_module
from reactorbench.dataset.catalog import AliasFamily, TemplateFamily
from reactorbench.dataset.config import load_development_dataset_config
from reactorbench.dataset.contracts import ModelInput, ProjectedObservationFact
from reactorbench.dataset.pipeline import (
    DevelopmentProjectionBundle,
    build_development_projection_bundle,
)
from reactorbench.dataset.quality import QualityRecord, TaskShortcutRecord, audit_quality
from reactorbench.dataset.renderer import RenderedCandidate, render_model_input
from reactorbench.dataset.review import (
    HumanReviewRecord,
    ReviewConfirmations,
    ReviewDecision,
    create_review_record,
    prepare_catalog_review_packet,
    prepare_postrender_review_packet,
    verify_catalog_review_gate,
    verify_review_record,
)
from reactorbench.schemas.base import canonical_sha256
from reactorbench.schemas.enums import (
    ChannelQuality,
    ObservationStatus,
    SplitName,
    StateVariable,
    TaskName,
)


def _candidate(*, value: float, status: ObservationStatus, split: SplitName) -> RenderedCandidate:
    model_input = ModelInput(
        cut_tick=3,
        observation_facts=(
            ProjectedObservationFact(
                fact_ref="o-0000",
                tick=3,
                channel_id="aster-primary-flow-a",
                variable=StateVariable.PRIMARY_FLOW,
                value=value,
                quality=ChannelQuality.GOOD,
                status=status,
            ),
        ),
        event_facts=(),
        context_facts=(),
    )
    return render_model_input(
        model_input,
        template_family=TemplateFamily.COMPACT_LOG,
        alias_family=AliasFamily.CANONICAL,
        split_name=split,
    )


def _provenance(*, split: SplitName, scenario: str) -> dict[str, object]:
    return {
        "dataset_version": "0.1.0",
        "generator_commit": "abcdef1",
        "scenario_schema_version": "0.1.0",
        "renderer_version": "0.1.0",
        "seed": 1,
        "scenario_id": scenario,
        "plant_variant_id": "ASTER-A",
        "fault_family_ids": [],
        "template_family_ids": ["compact-log-v1"],
        "split_name": split.value,
        "task_name": "fault_family",
    }


def _quality_record(candidate: RenderedCandidate, *, target: str, scenario: str) -> QualityRecord:
    return QualityRecord(
        example_id=candidate.render_id,
        split_name=candidate.split_name,
        text=candidate.text,
        template_family_id=candidate.template_family_id.value,
        alias_family_id=candidate.alias_family_id.value,
        target_labels=(target,),
        provenance=_provenance(split=candidate.split_name, scenario=scenario),
    )


def _task_shortcut_record(record: QualityRecord, *, record_id: str) -> TaskShortcutRecord:
    return TaskShortcutRecord(
        record_id=record_id,
        prompt_render_ids=(record.example_id,),
        task_name=TaskName.FAULT_FAMILY,
        template_family_id=record.template_family_id,
        alias_family_id=record.alias_family_id,
        target_labels=record.target_labels,
        context_flags=("corruption:none",),
    )


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


def _incomplete_confirmations() -> ReviewConfirmations:
    return ReviewConfirmations(
        all_preview_entries_reviewed=False,
        structured_answers_reviewed_separately=True,
        no_real_facility_or_procedure_content=True,
        no_navy_or_service_derived_content=False,
        no_operational_or_security_instructions=True,
        no_unfinished_templates_or_shortcuts=False,
        fingerprint_registry_reviewed=True,
    )


@pytest.fixture(scope="module")
def structured_bundle() -> DevelopmentProjectionBundle:
    config = load_development_dataset_config(Path("configs/dataset/development-v0.1.0.toml"))
    return build_development_projection_bundle(config, generator_commit="abcdef1")


def test_quality_report_covers_duplicates_ngrams_shortcuts_and_provenance() -> None:
    first = _candidate(value=0.45, status=ObservationStatus.NORMAL, split=SplitName.IID_TRAIN)
    second = _candidate(value=0.55, status=ObservationStatus.WATCH, split=SplitName.IID_TEST)
    passed = audit_quality(
        (
            _quality_record(first, target="ALPHA", scenario="scenario-a"),
            _quality_record(second, target="BETA", scenario="scenario-b"),
        )
    )

    assert passed.passed
    assert not passed.exact_duplicates
    assert not passed.skeleton_duplicates
    assert len(passed.ngram_overlaps) == 3
    assert tuple(record.example_id for record in passed.audited_records) == tuple(
        sorted((first.render_id, second.render_id))
    )

    duplicate = QualityRecord(
        example_id="duplicate-copy",
        split_name=SplitName.IID_TEST,
        text=first.text,
        template_family_id=first.template_family_id.value,
        alias_family_id=first.alias_family_id.value,
        target_labels=("BETA",),
        provenance={},
    )
    original = _quality_record(first, target="ALPHA", scenario="a")
    failed = audit_quality(
        (original, duplicate),
        task_records=(
            _task_shortcut_record(original, record_id="original-fault-family"),
            _task_shortcut_record(duplicate, record_id="duplicate-fault-family"),
        ),
    )
    assert not failed.passed
    assert failed.exact_duplicates
    assert failed.skeleton_duplicates
    assert failed.provenance_issues


def test_repeated_skeletons_are_reported_but_only_holdout_family_leaks_fail() -> None:
    first = _candidate(value=0.45, status=ObservationStatus.NORMAL, split=SplitName.IID_TRAIN)
    second = _candidate(value=0.55, status=ObservationStatus.NORMAL, split=SplitName.IID_TEST)
    reported = audit_quality(
        (
            _quality_record(first, target="ALPHA", scenario="scenario-a"),
            _quality_record(second, target="BETA", scenario="scenario-b"),
        )
    )
    assert reported.passed
    assert reported.skeleton_duplicates
    assert not reported.forbidden_skeleton_duplicates

    leaked = QualityRecord(
        example_id="template-leak",
        split_name=SplitName.TEMPLATE_TEST,
        text=second.text,
        template_family_id=first.template_family_id.value,
        alias_family_id=first.alias_family_id.value,
        target_labels=("BETA",),
        provenance=_provenance(split=SplitName.TEMPLATE_TEST, scenario="scenario-c"),
    )
    failed = audit_quality((_quality_record(first, target="ALPHA", scenario="a"), leaked))
    assert not failed.passed
    assert failed.forbidden_skeleton_duplicates


def test_pre_render_catalog_gate_is_hash_bound_and_required_before_generation(
    structured_bundle: DevelopmentProjectionBundle,
) -> None:
    packet = prepare_catalog_review_packet(structured_bundle)
    replay = prepare_catalog_review_packet(structured_bundle)
    record = create_review_record(
        packet,
        reviewer_role="project-owner",
        review_date=date(2026, 8, 20),
        decision=ReviewDecision.APPROVED,
        confirmations=_confirmations(),
    )

    assert packet == replay
    assert packet.catalog_preview.entry_count == 176
    assert packet.guard_manifest.normalization
    assert packet.guard_manifest.denylist_sha256
    assert packet.authored_language_surfaces.renderer.event_clauses
    assert packet.structured_binding.target_inventory_count == (
        len(structured_bundle.projections) + len(structured_bundle.counterfactual_projections)
    )
    assert packet.structured_binding.single_projection_count == len(structured_bundle.projections)
    assert packet.structured_binding.counterfactual_projection_count == len(
        structured_bundle.counterfactual_projections
    )
    verify_catalog_review_gate(packet, record, structured_bundle=structured_bundle)

    tampered_packet = packet.model_copy(update={"packet_sha256": "0" * 64})
    with pytest.raises(ValueError, match="checksum"):
        verify_catalog_review_gate(tampered_packet, record, structured_bundle=structured_bundle)
    tampered_record = record.model_copy(update={"packet_sha256": "1" * 64})
    with pytest.raises(ValueError, match=r"checksum|different"):
        verify_catalog_review_gate(packet, tampered_record, structured_bundle=structured_bundle)


def test_catalog_review_stales_when_an_actual_authored_surface_changes(
    structured_bundle: DevelopmentProjectionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = prepare_catalog_review_packet(structured_bundle)
    record = create_review_record(
        packet,
        reviewer_role="project-owner",
        review_date=date(2026, 8, 20),
        decision=ReviewDecision.APPROVED,
        confirmations=_confirmations(),
    )
    original_status = renderer_module._status

    def changed_status(status: ObservationStatus) -> str:
        if status is ObservationStatus.NORMAL:
            return "inside the revised fictional band"
        return original_status(status)

    monkeypatch.setattr(renderer_module, "_status", changed_status)
    with pytest.raises(ValueError, match=r"wording|language surfaces"):
        verify_catalog_review_gate(packet, record, structured_bundle=structured_bundle)


def test_catalog_review_is_bound_to_the_exact_structured_graph(
    structured_bundle: DevelopmentProjectionBundle,
) -> None:
    packet = prepare_catalog_review_packet(structured_bundle)
    record = create_review_record(
        packet,
        reviewer_role="project-owner",
        review_date=date(2026, 8, 20),
        decision=ReviewDecision.APPROVED,
        confirmations=_confirmations(),
    )
    other = structured_bundle.model_copy(update={"generator_commit": "1234567"})

    with pytest.raises(
        ValueError,
        match=r"checksum|different structured development graph|differs from the bundle",
    ):
        verify_catalog_review_gate(packet, record, structured_bundle=other)


def test_only_project_owner_can_create_a_canonical_review_record(
    structured_bundle: DevelopmentProjectionBundle,
) -> None:
    packet = prepare_catalog_review_packet(structured_bundle)
    with pytest.raises(ValueError, match="project-owner"):
        create_review_record(
            packet,
            reviewer_role="dataset reviewer",  # type: ignore[arg-type]
            review_date=date(2026, 8, 20),
            decision=ReviewDecision.APPROVED,
            confirmations=_confirmations(),
        )


def test_approved_review_requires_every_confirmation_true(
    structured_bundle: DevelopmentProjectionBundle,
) -> None:
    packet = prepare_catalog_review_packet(structured_bundle)
    confirmations = _incomplete_confirmations()

    with pytest.raises(ValueError, match="all confirmations to be true"):
        create_review_record(
            packet,
            reviewer_role="project-owner",
            review_date=date(2026, 8, 20),
            decision=ReviewDecision.APPROVED,
            confirmations=confirmations,
        )

    unapproved = create_review_record(
        packet,
        reviewer_role="project-owner",
        review_date=date(2026, 8, 20),
        decision=ReviewDecision.REVISE,
        confirmations=confirmations,
    )
    bypass_draft = unapproved.model_copy(update={"decision": ReviewDecision.APPROVED})
    bypass = bypass_draft.model_copy(
        update={
            "review_record_sha256": canonical_sha256(
                bypass_draft.model_dump(
                    mode="json", exclude={"review_record_sha256"}, round_trip=True
                )
            )
        }
    )
    with pytest.raises(ValueError, match="all confirmations to be true"):
        verify_review_record(packet, bypass, require_approved=False)


def test_negative_review_decisions_preserve_false_confirmations_as_valid_evidence(
    structured_bundle: DevelopmentProjectionBundle,
) -> None:
    packet = prepare_catalog_review_packet(structured_bundle)
    for decision in (ReviewDecision.REVISE, ReviewDecision.REJECTED):
        record = create_review_record(
            packet,
            reviewer_role="project-owner",
            review_date=date(2026, 8, 20),
            decision=decision,
            confirmations=_incomplete_confirmations(),
            notes=("The review is incomplete.",),
        )
        replay = HumanReviewRecord.model_validate_json(record.model_dump_json())

        assert replay == record
        assert replay.decision is decision
        assert not replay.confirmations.all_preview_entries_reviewed
        verify_review_record(packet, replay, require_approved=False)
        with pytest.raises(ValueError, match="has not been approved"):
            verify_review_record(packet, replay)


@pytest.mark.parametrize("invalid", ["true", "false", 1, 0])
def test_review_confirmations_reject_string_and_integer_coercion(invalid: object) -> None:
    payload = _confirmations().model_dump(mode="python")
    payload["all_preview_entries_reviewed"] = invalid

    with pytest.raises(ValidationError):
        ReviewConfirmations.model_validate(payload)


def test_postrender_review_is_separate_full_and_quality_inventory_bound() -> None:
    first = _candidate(value=0.45, status=ObservationStatus.NORMAL, split=SplitName.IID_TRAIN)
    second = _candidate(value=0.55, status=ObservationStatus.WATCH, split=SplitName.IID_TEST)
    report = audit_quality(
        (
            _quality_record(first, target="ALPHA", scenario="scenario-a"),
            _quality_record(second, target="BETA", scenario="scenario-b"),
        )
    )
    packet = prepare_postrender_review_packet((second, first), quality_report=report)
    record = create_review_record(
        packet,
        reviewer_role="project-owner",
        review_date=date(2026, 8, 20),
        decision=ReviewDecision.APPROVED,
        confirmations=_confirmations(),
    )

    assert packet.candidate_count == 2
    assert tuple(candidate.render_id for candidate in packet.candidates) == tuple(
        sorted((first.render_id, second.render_id))
    )
    verify_review_record(packet, record)
    wrong_role_draft = record.model_copy(update={"reviewer_role": "postrender reviewer"})
    wrong_role = wrong_role_draft.model_copy(
        update={
            "review_record_sha256": canonical_sha256(
                wrong_role_draft.model_dump(
                    mode="json", exclude={"review_record_sha256"}, round_trip=True
                )
            )
        }
    )
    with pytest.raises(ValueError, match="project-owner"):
        verify_review_record(packet, wrong_role)

    wrong_report = audit_quality((_quality_record(first, target="ALPHA", scenario="scenario-a"),))
    with pytest.raises(ValueError, match=r"candidate count|inventory"):
        prepare_postrender_review_packet((first, second), quality_report=wrong_report)


def test_postrender_verification_rejects_a_quality_failure() -> None:
    first = _candidate(value=0.45, status=ObservationStatus.NORMAL, split=SplitName.IID_TRAIN)
    second = _candidate(value=0.55, status=ObservationStatus.WATCH, split=SplitName.IID_TEST)
    first_quality = _quality_record(first, target="ALPHA", scenario="scenario-a")
    second_quality = QualityRecord(
        example_id=second.render_id,
        split_name=second.split_name,
        text=second.text,
        template_family_id=second.template_family_id.value,
        alias_family_id=second.alias_family_id.value,
        target_labels=("BETA",),
        provenance={},
    )
    failed_report = audit_quality(
        (first_quality, second_quality),
        task_records=(
            _task_shortcut_record(first_quality, record_id="first-fault-family"),
            _task_shortcut_record(second_quality, record_id="second-fault-family"),
        ),
    )
    assert not failed_report.passed
    inconsistent_draft = failed_report.model_copy(update={"passed": True})
    inconsistent = inconsistent_draft.model_copy(
        update={
            "report_sha256": canonical_sha256(
                inconsistent_draft.model_dump(
                    mode="json", exclude={"report_sha256"}, round_trip=True
                )
            )
        }
    )
    with pytest.raises(ValueError, match="pass flag"):
        prepare_postrender_review_packet((first, second), quality_report=inconsistent)
    packet = prepare_postrender_review_packet((first, second), quality_report=failed_report)
    record = create_review_record(
        packet,
        reviewer_role="project-owner",
        review_date=date(2026, 8, 20),
        decision=ReviewDecision.APPROVED,
        confirmations=_confirmations(),
    )

    with pytest.raises(ValueError, match="failing quality report"):
        verify_review_record(packet, record)
