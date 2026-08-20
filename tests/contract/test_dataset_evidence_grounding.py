"""Exhaustive evidence-grounding gates for the 204-trajectory development plan."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from reactorbench.dataset.config import load_development_dataset_config
from reactorbench.dataset.contracts import (
    ModelInput,
    PromptCounterfactualComparisonTarget,
    PromptEvidenceTarget,
)
from reactorbench.dataset.pipeline import (
    DevelopmentProjectionBundle,
    build_development_projection_bundle,
)
from reactorbench.dataset.projection import visible_evidence_fact_refs
from reactorbench.schemas.enums import (
    DiagnosisStatus,
    EvidenceSlot,
    TaskName,
)

CONFIG = Path("configs/dataset/development-v0.1.0.toml")


@pytest.fixture(scope="module")
def development_bundle() -> DevelopmentProjectionBundle:
    return build_development_projection_bundle(
        load_development_dataset_config(CONFIG), generator_commit="abcdef0"
    )


def _visible_refs(model_input: ModelInput) -> set[str]:
    # Keeping this small helper independent from matcher internals verifies that
    # every emitted reference actually resolves against the strict input contract.
    return {
        *(fact.fact_ref for fact in model_input.observation_facts),
        *(fact.fact_ref for fact in model_input.event_facts),
        *(fact.fact_ref for fact in model_input.context_facts),
    }


def test_every_emitted_evidence_slot_is_visibly_grounded_across_full_plan(
    development_bundle: DevelopmentProjectionBundle,
) -> None:
    evidence = tuple(
        projection
        for projection in development_bundle.projections
        if projection.task_target.task_name is TaskName.EXTRACT_EVIDENCE
    )
    assert len(evidence) == 405

    for projection in evidence:
        target = projection.task_target.target
        assert isinstance(target, PromptEvidenceTarget)
        assert target.fact_refs
        visible = _visible_refs(projection.model_input)
        assert set(target.fact_refs).issubset(visible)
        matched_union: set[str] = set()
        for slot in target.evidence_slots:
            matched = visible_evidence_fact_refs(projection.model_input, slot)
            assert matched
            assert set(matched).issubset(visible)
            assert set(matched).issubset(target.fact_refs)
            matched_union.update(matched)
        assert matched_union == set(target.fact_refs)


def test_only_map_withheld_semantics_are_intentionally_omitted(
    development_bundle: DevelopmentProjectionBundle,
) -> None:
    records = {
        record.trajectory.trajectory_id: record for record in development_bundle.trajectories
    }
    omissions: Counter[tuple[str, int, EvidenceSlot]] = Counter()
    for projection in development_bundle.projections:
        if projection.task_target.task_name is not TaskName.EXTRACT_EVIDENCE:
            continue
        target = projection.task_target.target
        assert isinstance(target, PromptEvidenceTarget)
        record = records[projection.lineage.trajectory_id]
        decision = next(
            decision
            for decision in record.trajectory.targets.decisions
            if decision.decision_tick == projection.lineage.decision_tick
        )
        for slot in set(decision.evidence_slots) - set(target.evidence_slots):
            omissions[(record.case_family, decision.decision_tick, slot)] += 1

    # With the dependency map withheld, the visible component changes remain
    # observable but their mapped relationship cannot be inferred. All other
    # declared development-plan evidence is grounded without audit annotations.
    assert omissions == Counter(
        {
            (
                "g12-map-aster-a-map_withheld",
                5,
                EvidenceSlot.MAPPED_COMPONENT_CHANGE,
            ): 2,
            (
                "g12-map-aster-b-map_withheld",
                5,
                EvidenceSlot.MAPPED_COMPONENT_CHANGE,
            ): 2,
        }
    )


def test_no_fault_and_diagnosed_related_state_slots_survive_projection(
    development_bundle: DevelopmentProjectionBundle,
) -> None:
    records = {
        record.trajectory.trajectory_id: record for record in development_bundle.trajectories
    }
    stable_count = 0
    load_count = 0
    diagnosed_related_count = 0
    for projection in development_bundle.projections:
        if projection.task_target.task_name is not TaskName.EXTRACT_EVIDENCE:
            continue
        target = projection.task_target.target
        assert isinstance(target, PromptEvidenceTarget)
        record = records[projection.lineage.trajectory_id]
        decision = next(
            decision
            for decision in record.trajectory.targets.decisions
            if decision.decision_tick == projection.lineage.decision_tick
        )
        if not decision.fault_labels and record.case_family.startswith("g01-stable"):
            stable_count += 1
            assert target.evidence_slots == (EvidenceSlot.STABLE_OPERATION,)
        if not decision.fault_labels and record.case_family.startswith("g02-load"):
            load_count += 1
            assert target.evidence_slots == (
                EvidenceSlot.STABLE_OPERATION,
                EvidenceSlot.COORDINATED_LOAD_RESPONSE,
            )
        if (
            decision.diagnosis_status is DiagnosisStatus.DIAGNOSED
            and EvidenceSlot.RELATED_STATE_STABLE in decision.evidence_slots
        ):
            diagnosed_related_count += 1
            assert EvidenceSlot.RELATED_STATE_STABLE in target.evidence_slots

    assert stable_count == 26
    assert load_count == 13
    assert diagnosed_related_count > 0


def test_counterfactual_references_resolve_and_slot_deltas_are_semantic(
    development_bundle: DevelopmentProjectionBundle,
) -> None:
    unequal_reference_counts = 0
    for pair in development_bundle.counterfactual_projections:
        target = pair.task_target.target
        assert isinstance(target, PromptCounterfactualComparisonTarget)
        baseline_refs = _visible_refs(pair.model_input.baseline)
        counterfactual_refs = _visible_refs(pair.model_input.counterfactual)
        assert set(target.baseline_decisive_fact_refs).issubset(baseline_refs)
        assert set(target.counterfactual_decisive_fact_refs).issubset(counterfactual_refs)
        for slot in target.baseline.evidence_slots:
            assert visible_evidence_fact_refs(pair.model_input.baseline, slot)
        for slot in target.counterfactual.evidence_slots:
            assert visible_evidence_fact_refs(pair.model_input.counterfactual, slot)
        if len(target.baseline_decisive_fact_refs) != len(target.counterfactual_decisive_fact_refs):
            unequal_reference_counts += 1
        for slot in target.decisive_evidence_slots:
            baseline_match = visible_evidence_fact_refs(pair.model_input.baseline, slot)
            counterfactual_match = visible_evidence_fact_refs(pair.model_input.counterfactual, slot)
            assert bool(baseline_match) != bool(counterfactual_match)
            assert set(baseline_match).issubset(baseline_refs)
            assert set(counterfactual_match).issubset(counterfactual_refs)

    assert len(development_bundle.counterfactual_projections) == 14
    assert unequal_reference_counts > 0


def test_matcher_is_closed_over_the_declared_evidence_vocabulary(
    development_bundle: DevelopmentProjectionBundle,
) -> None:
    # Every reviewed slot has an implemented branch. This synthetic call uses a
    # real input so a newly added enum value cannot silently fall through.
    scenario_projection = next(
        projection
        for projection in development_bundle.projections
        if projection.task_target.task_name is TaskName.EXTRACT_EVIDENCE
    )
    for slot in EvidenceSlot:
        visible_evidence_fact_refs(scenario_projection.model_input, slot)
