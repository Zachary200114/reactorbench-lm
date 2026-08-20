"""Checksum-bound owner review gate for the 15 developmental golden scenarios."""

from __future__ import annotations

import json
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictBool, model_validator

from reactorbench.schemas.base import ContractModel, canonical_json_bytes, canonical_sha256
from reactorbench.schemas.enums import ComponentState, PlantVariant
from reactorbench.schemas.scenario import ScenarioDefinition
from reactorbench.schemas.target import ScenarioTargets
from reactorbench.simulator import (
    GENERATOR_VERSION,
    build_abstract_inventory_loss_scenario,
    build_flow_imbalance_scenario,
    build_load_transient_scenario,
    build_pump_degradation_scenario,
    build_pump_degradation_sensor_drift_scenario,
    build_pump_trip_scenario,
    build_sensor_drift_scenario,
    build_sensor_noise_scenario,
    build_sensor_stuck_load_scenario,
    build_sparse_primary_flow_scenario,
    build_stable_scenario,
    build_support_power_interruption_scenario,
    build_transfer_efficiency_loss_scenario,
    build_valve_lag_scenario,
    build_valve_stuck_scenario,
    generate_trace,
    scan_prohibited_content,
)

GOLDEN_PACKET_VERSION: Literal["0.1.0"] = "0.1.0"
MAX_GOLDEN_PACKET_BYTES = 4 * 1024 * 1024


class GoldenReviewDecision(StrEnum):
    APPROVED = "APPROVED"
    REVISE = "REVISE"
    REJECTED = "REJECTED"


class GoldenReviewConfirmations(ContractModel):
    all_cases_reviewed: StrictBool
    expected_structured_answers_reviewed: StrictBool
    synthetic_and_fictional_only: StrictBool
    no_real_setpoints_or_operating_units: StrictBool
    no_real_procedures_or_facility_topology: StrictBool
    no_service_derived_nonpublic_information: StrictBool
    non_operational_research_use_only: StrictBool


class GoldenCaseReview(ContractModel):
    case_id: str = Field(pattern=r"^G(?:0[1-9]|1[0-5])$")
    title: str = Field(min_length=1, max_length=120)
    review_intent: str = Field(min_length=1, max_length=500)
    scenario: ScenarioDefinition
    expected_targets: ScenarioTargets
    scenario_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    targets_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prohibited_content_finding_count: Literal[0]

    @model_validator(mode="after")
    def checksums_match(self) -> GoldenCaseReview:
        if self.scenario_sha256 != canonical_sha256(
            self.scenario.model_dump(mode="json", round_trip=True)
        ):
            raise ValueError("golden scenario checksum mismatch")
        if self.targets_sha256 != canonical_sha256(
            self.expected_targets.model_dump(mode="json", round_trip=True)
        ):
            raise ValueError("golden target checksum mismatch")
        if self.expected_targets.scenario_id != self.scenario.scenario_id:
            raise ValueError("golden target belongs to a different scenario")
        return self


class GoldenReviewPacket(ContractModel):
    packet_version: Literal["0.1.0"] = GOLDEN_PACKET_VERSION
    review_stage: Literal["golden_suite_pretest"] = "golden_suite_pretest"
    generator_version: Literal["0.1.0"]
    generator_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    human_review_required: Literal[True] = True
    automation_is_not_human_approval: Literal[True] = True
    cases: tuple[GoldenCaseReview, ...]
    packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exact_suite_and_checksum_match(self) -> GoldenReviewPacket:
        if tuple(case.case_id for case in self.cases) != tuple(
            f"G{index:02d}" for index in range(1, 16)
        ):
            raise ValueError("golden packet must contain G01-G15 in exact order")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"packet_sha256"})
        )
        if self.packet_sha256 != expected:
            raise ValueError("golden packet checksum mismatch")
        return self


class GoldenReviewRecord(ContractModel):
    record_version: Literal["0.1.0"] = GOLDEN_PACKET_VERSION
    review_stage: Literal["golden_suite_pretest"] = "golden_suite_pretest"
    packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_role: Literal["project-owner"]
    review_date: date
    decision: GoldenReviewDecision
    confirmations: GoldenReviewConfirmations
    notes: tuple[str, ...] = Field(max_length=32)
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def decision_and_checksum_match(self) -> GoldenReviewRecord:
        if any(not note or len(note) > 500 for note in self.notes):
            raise ValueError("golden review notes must be non-empty and at most 500 characters")
        confirmations = self.confirmations.model_dump(mode="python")
        if self.decision is GoldenReviewDecision.APPROVED and not all(confirmations.values()):
            raise ValueError("approved golden review requires every confirmation")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"record_sha256"})
        )
        if self.record_sha256 != expected:
            raise ValueError("golden review record checksum mismatch")
        return self


def _scenario_catalog() -> tuple[tuple[str, str, str, ScenarioDefinition], ...]:
    return (
        (
            "G01",
            "Stable operation",
            "Confirm the no-fault stable reference.",
            build_stable_scenario(seed=101),
        ),
        (
            "G02",
            "Benign load transition",
            "Confirm a coordinated non-fault load response.",
            build_load_transient_scenario(seed=102),
        ),
        (
            "G03",
            "Single-channel drift",
            "Confirm persistent disagreement supports a sensor-drift conclusion.",
            build_sensor_drift_scenario(seed=103),
        ),
        (
            "G04",
            "Stuck channel under load",
            "Confirm a frozen channel is separated from the coherent process response.",
            build_sensor_stuck_load_scenario(seed=104),
        ),
        (
            "G05",
            "Alternating sensor noise",
            "Confirm inconsistent readings mature before a noise conclusion.",
            build_sensor_noise_scenario(seed=105),
        ),
        (
            "G06",
            "Gradual pump degradation",
            "Confirm process evidence precedes inspection and load-reduction actions.",
            build_pump_degradation_scenario(seed=106),
        ),
        (
            "G07",
            "Pump trip with standby available",
            "Confirm dependency context supports the synthetic standby action.",
            build_pump_trip_scenario(seed=107, standby_state=ComponentState.AVAILABLE),
        ),
        (
            "G08",
            "Bounded valve lag",
            "Confirm the command mismatch resolves inside the declared lag band.",
            build_valve_lag_scenario(seed=108, lag_ticks=4),
        ),
        (
            "G09",
            "Persistent valve mismatch",
            "Confirm persistence distinguishes the stuck case from bounded lag.",
            build_valve_stuck_scenario(seed=109),
        ),
        (
            "G10",
            "Transfer efficiency loss",
            "Confirm upstream/downstream divergence supports the process label.",
            build_transfer_efficiency_loss_scenario(seed=110, plant_variant=PlantVariant.ASTER_A),
        ),
        (
            "G11",
            "Secondary flow imbalance",
            "Confirm secondary mismatch and persistence precede stabilization.",
            build_flow_imbalance_scenario(seed=111, plant_variant=PlantVariant.ASTER_A),
        ),
        (
            "G12",
            "Support power interruption",
            "Confirm the fictional dependency map grounds affected components.",
            build_support_power_interruption_scenario(
                seed=112, plant_variant=PlantVariant.ASTER_B, include_dependency_map=True
            ),
        ),
        (
            "G13",
            "Abstract inventory loss",
            "Confirm agreeing inventory trends and delayed effects support diagnosis.",
            build_abstract_inventory_loss_scenario(seed=113, plant_variant=PlantVariant.ASTER_A),
        ),
        (
            "G14",
            "Compound pump degradation and drift",
            "Confirm both independent evidence chains are required for two labels.",
            build_pump_degradation_sensor_drift_scenario(seed=114),
        ),
        (
            "G15",
            "Sparse ambiguous primary flow",
            "Confirm sparse evidence requires explicit abstention.",
            build_sparse_primary_flow_scenario(
                seed=115, duration_ticks=8, plant_variant=PlantVariant.ASTER_A
            ),
        ),
    )


def prepare_golden_review_packet(*, generator_commit: str) -> GoldenReviewPacket:
    if GENERATOR_VERSION != "0.1.0":
        raise ValueError("golden review packet requires generator version 0.1.0")
    if (
        type(generator_commit) is not str
        or not 7 <= len(generator_commit) <= 64
        or any(character not in "0123456789abcdef" for character in generator_commit)
    ):
        raise ValueError("generator_commit must be a lowercase hexadecimal Git revision")
    cases: list[GoldenCaseReview] = []
    for case_id, title, intent, scenario in _scenario_catalog():
        trace = generate_trace(scenario)
        if scan_prohibited_content(trace):
            raise ValueError(f"{case_id} contains prohibited content")
        trace_payload = {
            "scenario": trace.scenario.model_dump(mode="json", round_trip=True),
            "latent_states": tuple(
                state.model_dump(mode="json", round_trip=True) for state in trace.latent_states
            ),
            "observations": tuple(
                frame.model_dump(mode="json", round_trip=True) for frame in trace.observations
            ),
            "events": tuple(
                event.model_dump(mode="json", round_trip=True) for event in trace.events
            ),
            "targets": trace.targets.model_dump(mode="json", round_trip=True),
        }
        cases.append(
            GoldenCaseReview(
                case_id=case_id,
                title=title,
                review_intent=intent,
                scenario=trace.scenario,
                expected_targets=trace.targets,
                scenario_sha256=canonical_sha256(
                    trace.scenario.model_dump(mode="json", round_trip=True)
                ),
                targets_sha256=canonical_sha256(
                    trace.targets.model_dump(mode="json", round_trip=True)
                ),
                trace_sha256=canonical_sha256(trace_payload),
                prohibited_content_finding_count=0,
            )
        )
    draft = GoldenReviewPacket.model_construct(
        generator_version="0.1.0",
        generator_commit=generator_commit,
        cases=tuple(cases),
        packet_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"packet_sha256"})
    )
    return GoldenReviewPacket(
        generator_version="0.1.0",
        generator_commit=generator_commit,
        cases=tuple(cases),
        packet_sha256=checksum,
    )


def create_golden_review_record(
    packet: GoldenReviewPacket,
    *,
    review_date: date,
    decision: GoldenReviewDecision,
    confirmations: GoldenReviewConfirmations,
    notes: tuple[str, ...] = (),
) -> GoldenReviewRecord:
    if type(packet) is not GoldenReviewPacket:
        raise TypeError("packet must be an exact GoldenReviewPacket")
    draft = GoldenReviewRecord.model_construct(
        packet_sha256=packet.packet_sha256,
        reviewer_role="project-owner",
        review_date=review_date,
        decision=decision,
        confirmations=confirmations,
        notes=notes,
        record_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"record_sha256"})
    )
    return GoldenReviewRecord(
        packet_sha256=packet.packet_sha256,
        reviewer_role="project-owner",
        review_date=review_date,
        decision=decision,
        confirmations=confirmations,
        notes=notes,
        record_sha256=checksum,
    )


def verify_golden_review(
    packet: GoldenReviewPacket,
    record: GoldenReviewRecord,
    *,
    expected_packet_sha256: str,
) -> None:
    if type(packet) is not GoldenReviewPacket or type(record) is not GoldenReviewRecord:
        raise TypeError("golden verification requires exact packet and record objects")
    packet = GoldenReviewPacket.model_validate(packet.model_dump(mode="python", round_trip=True))
    record = GoldenReviewRecord.model_validate(record.model_dump(mode="python", round_trip=True))
    if packet.packet_sha256 != expected_packet_sha256:
        raise ValueError("golden packet differs from the preregistered checksum")
    if record.packet_sha256 != packet.packet_sha256:
        raise ValueError("golden review record is bound to another packet")
    if record.reviewer_role != "project-owner":
        raise ValueError("golden review requires the project-owner role")
    if record.decision is not GoldenReviewDecision.APPROVED:
        raise ValueError("golden suite is not approved")
    regenerated = prepare_golden_review_packet(generator_commit=packet.generator_commit)
    if regenerated != packet:
        raise ValueError("golden packet is stale relative to the generator")


def write_golden_review_packet(packet: GoldenReviewPacket, path: Path) -> None:
    if type(packet) is not GoldenReviewPacket or not isinstance(path, Path):
        raise TypeError("golden packet write requires exact packet and pathlib.Path")
    if path.exists() or path.is_symlink():
        raise FileExistsError("golden packet output must not already exist")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    payload = canonical_json_bytes(packet.model_dump(mode="json", round_trip=True)) + b"\n"
    if len(payload) > MAX_GOLDEN_PACKET_BYTES:
        raise ValueError("golden packet exceeds its size limit")
    path.write_bytes(payload)


def load_golden_review_packet(path: Path) -> GoldenReviewPacket:
    return _load_json_model(path, GoldenReviewPacket, maximum_bytes=MAX_GOLDEN_PACKET_BYTES)


def load_golden_review_record(path: Path) -> GoldenReviewRecord:
    return _load_json_model(path, GoldenReviewRecord, maximum_bytes=1024 * 1024)


def _load_json_model[T: ContractModel](path: Path, model: type[T], *, maximum_bytes: int) -> T:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise ValueError("review artifact must be a regular non-symlink file")
    payload = path.read_bytes()
    if not payload or len(payload) > maximum_bytes:
        raise ValueError("review artifact is empty or oversized")

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError("review artifact contains a duplicate JSON key")
            result[key] = value
        return result

    json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {value}")),
    )
    return model.model_validate_json(payload)


__all__ = [
    "GoldenCaseReview",
    "GoldenReviewConfirmations",
    "GoldenReviewDecision",
    "GoldenReviewPacket",
    "GoldenReviewRecord",
    "create_golden_review_record",
    "load_golden_review_packet",
    "load_golden_review_record",
    "prepare_golden_review_packet",
    "verify_golden_review",
    "write_golden_review_packet",
]
