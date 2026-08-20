"""Separate hash-bound pre-render catalog and post-render candidate reviews."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import Field, StrictBool, model_validator

from reactorbench.schemas.base import (
    ContractId,
    ContractModel,
    SemanticVersion,
    canonical_json_bytes,
    canonical_sha256,
)
from reactorbench.schemas.enums import TaskName
from reactorbench.schemas.provenance import GitCommit

from .catalog import RENDERER_VERSION, catalog_manifest
from .content_guard import AUTOMATION_IS_PROOF, assert_no_prohibited_content, guard_manifest
from .corruption import (
    CorruptionAuthoredSurfaceManifest,
    corruption_authored_surface_manifest,
)
from .pipeline import DevelopmentProjectionBundle
from .quality import QualityReport
from .renderer import (
    CatalogPreviewPacket,
    RenderedCandidate,
    RendererAuthoredSurfaceManifest,
    render_catalog_preview,
    renderer_authored_surface_manifest,
)

REVIEW_FORMAT_VERSION: SemanticVersion = "0.1.0"
PROJECT_OWNER_REVIEWER_ROLE: Literal["project-owner"] = "project-owner"


class ReviewStage(StrEnum):
    PRE_RENDER_CATALOG = "pre_render_catalog"
    POSTRENDER_CANDIDATES = "postrender_candidates"


class ReviewDecision(StrEnum):
    APPROVED = "approved"
    REVISE = "revise"
    REJECTED = "rejected"


class CandidatePreview(ContractModel):
    render_id: str
    split_name: str
    template_family_id: str
    alias_family_id: str
    model_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_ids: tuple[str, ...]
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str = Field(min_length=1, max_length=65_536)


class GuardReviewManifest(ContractModel):
    """Exact public metadata for the automated content guard under review."""

    guard_version: SemanticVersion
    normalization: str = Field(min_length=1, max_length=96)
    automation_is_proof: Literal[False]
    fingerprint_review_status: str = Field(min_length=1, max_length=96)
    denylist_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fingerprints_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AuthoredLanguageSurfaceManifest(ContractModel):
    """All actual renderer and corruption wording presented for owner review."""

    manifest_version: Literal["0.1.0"] = "0.1.0"
    renderer: RendererAuthoredSurfaceManifest
    corruption: CorruptionAuthoredSurfaceManifest
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def checksum_matches_surfaces(self) -> AuthoredLanguageSurfaceManifest:
        expected = canonical_sha256(
            self.model_dump(mode="json", exclude={"checksum_sha256"}, round_trip=True)
        )
        if self.checksum_sha256 != expected:
            raise ValueError("authored language surface checksum mismatch")
        return self


class StructuredTargetReviewPreview(ContractModel):
    """One complete structured target, separate from all renderer prose."""

    record_id: ContractId
    task_name: TaskName
    target_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_target_json: str = Field(min_length=2, max_length=65_536)

    @model_validator(mode="after")
    def target_is_canonical_and_hash_bound(self) -> StructuredTargetReviewPreview:
        try:
            parsed = json.loads(self.canonical_target_json)
        except json.JSONDecodeError as error:
            raise ValueError("structured target preview must contain valid JSON") from error
        canonical = canonical_json_bytes(parsed)
        if canonical.decode("utf-8") != self.canonical_target_json:
            raise ValueError("structured target preview JSON must be canonical")
        if hashlib.sha256(canonical).hexdigest() != self.target_sha256:
            raise ValueError("structured target preview checksum mismatch")
        return self


class StructuredReviewBinding(ContractModel):
    """Exact structured development graph whose targets the owner reviewed."""

    dataset_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_commit: GitCommit
    projection_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trajectory_count: int = Field(ge=1)
    single_projection_count: int = Field(ge=1)
    counterfactual_projection_count: int = Field(ge=0)
    target_inventory_count: int = Field(ge=1)
    target_inventory: tuple[StructuredTargetReviewPreview, ...]
    target_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def inventory_is_complete_sorted_and_hash_bound(self) -> StructuredReviewBinding:
        if self.target_inventory_count != len(self.target_inventory):
            raise ValueError("structured target inventory count mismatch")
        expected_count = self.single_projection_count + self.counterfactual_projection_count
        if self.target_inventory_count != expected_count:
            raise ValueError("structured target inventory does not cover every projection")
        identifiers = tuple(item.record_id for item in self.target_inventory)
        if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(set(identifiers)):
            raise ValueError("structured target inventory must have unique canonical ID order")
        expected = canonical_sha256(
            tuple(item.model_dump(mode="json", round_trip=True) for item in self.target_inventory)
        )
        if self.target_inventory_sha256 != expected:
            raise ValueError("structured target inventory checksum mismatch")
        return self


class CatalogReviewPacket(ContractModel):
    """Mandatory human gate before any development rendering can start."""

    review_format_version: SemanticVersion = REVIEW_FORMAT_VERSION
    review_stage: Literal[ReviewStage.PRE_RENDER_CATALOG] = ReviewStage.PRE_RENDER_CATALOG
    renderer_version: SemanticVersion = RENDERER_VERSION
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    guard_manifest: GuardReviewManifest
    guard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authored_language_surfaces: AuthoredLanguageSurfaceManifest
    structured_binding: StructuredReviewBinding
    automation_is_proof: Literal[False] = False
    human_review_required: Literal[True] = True
    catalog_preview: CatalogPreviewPacket
    packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def checksum_and_catalog_match(self) -> CatalogReviewPacket:
        if self.catalog_preview.catalog_sha256 != self.catalog_sha256:
            raise ValueError("catalog preview and review packet catalog checksums differ")
        if self.guard_sha256 != canonical_sha256(
            self.guard_manifest.model_dump(mode="json", round_trip=True)
        ):
            raise ValueError("catalog review guard manifest checksum mismatch")
        if self.packet_sha256 != _packet_checksum(self):
            raise ValueError("catalog review packet checksum mismatch")
        return self


class PostrenderReviewPacket(ContractModel):
    """Full candidate review required after generation and quality audit."""

    review_format_version: SemanticVersion = REVIEW_FORMAT_VERSION
    review_stage: Literal[ReviewStage.POSTRENDER_CANDIDATES] = ReviewStage.POSTRENDER_CANDIDATES
    artifact_status: Literal["candidate_pending_postrender_review"] = (
        "candidate_pending_postrender_review"
    )
    renderer_version: SemanticVersion = RENDERER_VERSION
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    guard_manifest: GuardReviewManifest
    guard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authored_language_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    automation_is_proof: Literal[False] = False
    human_review_required: Literal[True] = True
    candidate_count: int = Field(ge=1)
    candidates: tuple[CandidatePreview, ...]
    quality_report: QualityReport
    packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def packet_is_complete_sorted_and_hash_bound(self) -> PostrenderReviewPacket:
        if self.guard_sha256 != canonical_sha256(
            self.guard_manifest.model_dump(mode="json", round_trip=True)
        ):
            raise ValueError("post-render guard manifest checksum mismatch")
        if self.candidate_count != len(self.candidates):
            raise ValueError("post-render review candidate count mismatch")
        identifiers = tuple(candidate.render_id for candidate in self.candidates)
        if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(set(identifiers)):
            raise ValueError("post-render candidates must be unique and sorted")
        if self.quality_report.record_count != self.candidate_count:
            raise ValueError("quality report candidate count must cover every preview candidate")
        expected_quality_pass = not any(
            (
                self.quality_report.exact_duplicates,
                self.quality_report.forbidden_skeleton_duplicates,
                self.quality_report.shortcut_findings,
                self.quality_report.target_text_findings,
                self.quality_report.provenance_issues,
            )
        )
        if self.quality_report.passed != expected_quality_pass:
            raise ValueError("quality report pass flag is inconsistent with its findings")
        audited = tuple(
            (record.example_id, record.text_sha256)
            for record in self.quality_report.audited_records
        )
        candidate_inventory = tuple(
            sorted((candidate.render_id, candidate.text_sha256) for candidate in self.candidates)
        )
        if audited != candidate_inventory:
            raise ValueError("quality report is not bound to this exact candidate inventory")
        if self.packet_sha256 != _packet_checksum(self):
            raise ValueError("post-render review packet checksum mismatch")
        return self


type ReviewPacket = CatalogReviewPacket | PostrenderReviewPacket


class ReviewConfirmations(ContractModel):
    all_preview_entries_reviewed: StrictBool
    structured_answers_reviewed_separately: StrictBool
    no_real_facility_or_procedure_content: StrictBool
    no_navy_or_service_derived_content: StrictBool
    no_operational_or_security_instructions: StrictBool
    no_unfinished_templates_or_shortcuts: StrictBool
    fingerprint_registry_reviewed: StrictBool


class HumanReviewRecord(ContractModel):
    review_format_version: SemanticVersion = REVIEW_FORMAT_VERSION
    review_stage: ReviewStage
    packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_role: Literal["project-owner"]
    review_date: date
    decision: ReviewDecision
    confirmations: ReviewConfirmations
    notes: tuple[str, ...] = Field(max_length=32)
    review_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def record_checksum_matches(self) -> HumanReviewRecord:
        if any(not note or len(note) > 500 for note in self.notes):
            raise ValueError("review notes must be non-empty and at most 500 characters")
        if self.decision is ReviewDecision.APPROVED and not all(
            self.confirmations.model_dump(mode="python").values()
        ):
            raise ValueError("approved human review requires all confirmations to be true")
        if self.review_record_sha256 != _record_checksum(self):
            raise ValueError("human review record checksum mismatch")
        return self


def _packet_checksum(packet: CatalogReviewPacket | PostrenderReviewPacket) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            packet.model_dump(mode="json", exclude={"packet_sha256"}, round_trip=True)
        )
    ).hexdigest()


def _record_checksum(record: HumanReviewRecord) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            record.model_dump(mode="json", exclude={"review_record_sha256"}, round_trip=True)
        )
    ).hexdigest()


def _current_guard_manifest() -> GuardReviewManifest:
    return GuardReviewManifest.model_validate(guard_manifest())


def _guard_checksum(manifest: GuardReviewManifest) -> str:
    return canonical_sha256(manifest.model_dump(mode="json", round_trip=True))


def authored_language_surface_manifest() -> AuthoredLanguageSurfaceManifest:
    """Assemble the exact renderer and corruption surfaces under human review."""

    renderer = renderer_authored_surface_manifest()
    corruption = corruption_authored_surface_manifest()
    draft = AuthoredLanguageSurfaceManifest.model_construct(
        manifest_version="0.1.0",
        renderer=renderer,
        corruption=corruption,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", exclude={"checksum_sha256"}, round_trip=True)
    )
    return AuthoredLanguageSurfaceManifest(
        renderer=renderer,
        corruption=corruption,
        checksum_sha256=checksum,
    )


def _target_preview(
    *, record_id: str, task_name: TaskName, target: ContractModel
) -> StructuredTargetReviewPreview:
    target_payload = target.model_dump(mode="json", round_trip=True)
    target_bytes = canonical_json_bytes(target_payload)
    return StructuredTargetReviewPreview(
        record_id=record_id,
        task_name=task_name,
        target_sha256=hashlib.sha256(target_bytes).hexdigest(),
        canonical_target_json=target_bytes.decode("utf-8"),
    )


def build_structured_review_binding(
    structured_bundle: DevelopmentProjectionBundle,
) -> StructuredReviewBinding:
    """Expose and hash every target in one exact validated development graph."""

    if type(structured_bundle) is not DevelopmentProjectionBundle:
        raise TypeError("structured review binding requires a DevelopmentProjectionBundle")
    # Revalidate a serialized copy so callers cannot bypass nested bundle and checksum
    # validators with Pydantic's intentionally unvalidated ``model_copy(update=...)``.
    structured_bundle = DevelopmentProjectionBundle.model_validate(
        structured_bundle.model_dump(mode="python", round_trip=True)
    )
    inventory = tuple(
        sorted(
            (
                *(
                    _target_preview(
                        record_id=projection.projection_id,
                        task_name=projection.task_target.task_name,
                        target=projection.task_target,
                    )
                    for projection in structured_bundle.projections
                ),
                *(
                    _target_preview(
                        record_id=projection.pair_id,
                        task_name=projection.task_target.task_name,
                        target=projection.task_target,
                    )
                    for projection in structured_bundle.counterfactual_projections
                ),
            ),
            key=lambda item: item.record_id,
        )
    )
    inventory_sha256 = canonical_sha256(
        tuple(item.model_dump(mode="json", round_trip=True) for item in inventory)
    )
    return StructuredReviewBinding(
        dataset_config_sha256=structured_bundle.dataset_config_sha256,
        generator_commit=structured_bundle.generator_commit,
        projection_bundle_sha256=structured_bundle.checksum_sha256,
        split_manifest_sha256=structured_bundle.split_manifest.checksum_sha256,
        trajectory_count=len(structured_bundle.trajectories),
        single_projection_count=len(structured_bundle.projections),
        counterfactual_projection_count=len(structured_bundle.counterfactual_projections),
        target_inventory_count=len(inventory),
        target_inventory=inventory,
        target_inventory_sha256=inventory_sha256,
    )


def prepare_catalog_review_packet(
    structured_bundle: DevelopmentProjectionBundle,
) -> CatalogReviewPacket:
    """Prepare complete authored-language and structured-target owner review."""

    if type(structured_bundle) is not DevelopmentProjectionBundle:
        raise TypeError("catalog review requires an exact structured development bundle")

    catalog = catalog_manifest()
    preview = render_catalog_preview()
    guard = _current_guard_manifest()
    guard_sha256 = _guard_checksum(guard)
    surfaces = authored_language_surface_manifest()
    structured_binding = build_structured_review_binding(structured_bundle)
    checksum_payload = {
        "review_format_version": REVIEW_FORMAT_VERSION,
        "review_stage": ReviewStage.PRE_RENDER_CATALOG.value,
        "renderer_version": RENDERER_VERSION,
        "catalog_sha256": catalog.checksum_sha256,
        "guard_manifest": guard.model_dump(mode="json", round_trip=True),
        "guard_sha256": guard_sha256,
        "authored_language_surfaces": surfaces.model_dump(mode="json", round_trip=True),
        "structured_binding": structured_binding.model_dump(mode="json", round_trip=True),
        "automation_is_proof": AUTOMATION_IS_PROOF,
        "human_review_required": True,
        "catalog_preview": preview.model_dump(mode="json", round_trip=True),
    }
    checksum = hashlib.sha256(canonical_json_bytes(checksum_payload)).hexdigest()
    return CatalogReviewPacket(
        review_format_version=REVIEW_FORMAT_VERSION,
        review_stage=ReviewStage.PRE_RENDER_CATALOG,
        renderer_version=RENDERER_VERSION,
        catalog_sha256=catalog.checksum_sha256,
        guard_manifest=guard,
        guard_sha256=guard_sha256,
        authored_language_surfaces=surfaces,
        structured_binding=structured_binding,
        automation_is_proof=AUTOMATION_IS_PROOF,
        human_review_required=True,
        catalog_preview=preview,
        packet_sha256=checksum,
    )


def _preview(candidate: RenderedCandidate) -> CandidatePreview:
    return CandidatePreview(
        render_id=candidate.render_id,
        split_name=candidate.split_name.value,
        template_family_id=candidate.template_family_id.value,
        alias_family_id=candidate.alias_family_id.value,
        model_input_sha256=candidate.model_input_sha256,
        template_ids=candidate.template_ids,
        text_sha256=candidate.text_sha256,
        text=candidate.text,
    )


def prepare_postrender_review_packet(
    candidates: tuple[RenderedCandidate, ...], *, quality_report: QualityReport
) -> PostrenderReviewPacket:
    """Prepare a deterministic full candidate packet after rendering and audit."""

    if not candidates:
        raise ValueError("post-render review packet requires at least one candidate")
    canonical = tuple(sorted(candidates, key=lambda candidate: candidate.render_id))
    if len({candidate.render_id for candidate in canonical}) != len(canonical):
        raise ValueError("post-render packet cannot contain duplicate candidate IDs")
    for candidate in canonical:
        assert_no_prohibited_content(candidate.text)
    previews = tuple(_preview(candidate) for candidate in canonical)
    catalog_sha256 = catalog_manifest().checksum_sha256
    guard = _current_guard_manifest()
    guard_sha256 = _guard_checksum(guard)
    authored_language_sha256 = authored_language_surface_manifest().checksum_sha256
    checksum_payload = {
        "review_format_version": REVIEW_FORMAT_VERSION,
        "review_stage": ReviewStage.POSTRENDER_CANDIDATES.value,
        "artifact_status": "candidate_pending_postrender_review",
        "renderer_version": RENDERER_VERSION,
        "catalog_sha256": catalog_sha256,
        "guard_manifest": guard.model_dump(mode="json", round_trip=True),
        "guard_sha256": guard_sha256,
        "authored_language_sha256": authored_language_sha256,
        "automation_is_proof": AUTOMATION_IS_PROOF,
        "human_review_required": True,
        "candidate_count": len(canonical),
        "candidates": tuple(
            preview.model_dump(mode="json", round_trip=True) for preview in previews
        ),
        "quality_report": quality_report.model_dump(mode="json", round_trip=True),
    }
    checksum = hashlib.sha256(canonical_json_bytes(checksum_payload)).hexdigest()
    return PostrenderReviewPacket(
        review_format_version=REVIEW_FORMAT_VERSION,
        review_stage=ReviewStage.POSTRENDER_CANDIDATES,
        artifact_status="candidate_pending_postrender_review",
        renderer_version=RENDERER_VERSION,
        catalog_sha256=catalog_sha256,
        guard_manifest=guard,
        guard_sha256=guard_sha256,
        authored_language_sha256=authored_language_sha256,
        automation_is_proof=AUTOMATION_IS_PROOF,
        human_review_required=True,
        candidate_count=len(canonical),
        candidates=previews,
        quality_report=quality_report,
        packet_sha256=checksum,
    )


def create_review_record(
    packet: ReviewPacket,
    *,
    reviewer_role: Literal["project-owner"],
    review_date: date,
    decision: ReviewDecision,
    confirmations: ReviewConfirmations,
    notes: tuple[str, ...] = (),
) -> HumanReviewRecord:
    """Create a record only from explicit human-supplied review fields."""

    if type(packet) not in {CatalogReviewPacket, PostrenderReviewPacket}:
        raise TypeError("packet must be an exact review packet")
    checksum_payload = {
        "review_format_version": REVIEW_FORMAT_VERSION,
        "review_stage": packet.review_stage.value,
        "packet_sha256": packet.packet_sha256,
        "reviewer_role": reviewer_role,
        "review_date": review_date.isoformat(),
        "decision": decision.value,
        "confirmations": confirmations.model_dump(mode="json", round_trip=True),
        "notes": notes,
    }
    checksum = hashlib.sha256(canonical_json_bytes(checksum_payload)).hexdigest()
    return HumanReviewRecord(
        review_format_version=REVIEW_FORMAT_VERSION,
        review_stage=packet.review_stage,
        packet_sha256=packet.packet_sha256,
        reviewer_role=reviewer_role,
        review_date=review_date,
        decision=decision,
        confirmations=confirmations,
        notes=notes,
        review_record_sha256=checksum,
    )


def verify_review_record(
    packet: ReviewPacket,
    record: HumanReviewRecord,
    *,
    require_approved: bool = True,
) -> None:
    """Verify exact binding, current hashes, stage, and candidate quality."""

    if type(packet) not in {CatalogReviewPacket, PostrenderReviewPacket}:
        raise TypeError("review verification requires a canonical review packet")
    if type(record) is not HumanReviewRecord:
        raise TypeError("review verification requires a canonical human record")
    # Revalidate a serialized copy so an unvalidated ``model_copy(update=...)``
    # cannot bypass the decision-specific confirmation contract.
    record = HumanReviewRecord.model_validate(record.model_dump(mode="python", round_trip=True))
    if packet.packet_sha256 != _packet_checksum(packet):
        raise ValueError("review packet checksum mismatch")
    if record.review_record_sha256 != _record_checksum(record):
        raise ValueError("human review record checksum mismatch")
    if record.reviewer_role != PROJECT_OWNER_REVIEWER_ROLE:
        raise ValueError("human review requires the canonical project-owner role")
    if (
        record.packet_sha256 != packet.packet_sha256
        or record.review_stage is not packet.review_stage
    ):
        raise ValueError("human review record is bound to a different packet or stage")
    if packet.catalog_sha256 != catalog_manifest().checksum_sha256:
        raise ValueError("review packet catalog checksum is stale")
    current_guard = _current_guard_manifest()
    if packet.guard_manifest != current_guard:
        raise ValueError("review packet guard manifest metadata is stale")
    if packet.guard_sha256 != _guard_checksum(current_guard):
        raise ValueError("review packet guard checksum is stale")
    current_surfaces = authored_language_surface_manifest()
    if isinstance(packet, CatalogReviewPacket):
        if packet.catalog_preview != render_catalog_preview():
            raise ValueError("catalog preview wording is stale")
        if packet.authored_language_surfaces != current_surfaces:
            raise ValueError("catalog authored language surfaces are stale")
    elif packet.authored_language_sha256 != current_surfaces.checksum_sha256:
        raise ValueError("post-render authored language checksum is stale")
    if require_approved and record.decision is not ReviewDecision.APPROVED:
        raise ValueError("review packet has not been approved")
    if isinstance(packet, PostrenderReviewPacket) and not packet.quality_report.passed:
        raise ValueError("candidate packet cannot be approved with a failing quality report")


def verify_catalog_review_gate(
    packet: CatalogReviewPacket,
    record: HumanReviewRecord,
    *,
    structured_bundle: DevelopmentProjectionBundle,
) -> None:
    """Fail closed unless the current complete catalog has explicit human approval."""

    if type(packet) is not CatalogReviewPacket:
        raise TypeError("development rendering requires a CatalogReviewPacket")
    if type(structured_bundle) is not DevelopmentProjectionBundle:
        raise TypeError("development rendering requires a DevelopmentProjectionBundle")
    verify_review_record(packet, record, require_approved=True)
    if packet.structured_binding != build_structured_review_binding(structured_bundle):
        raise ValueError("catalog review is bound to a different structured development graph")
