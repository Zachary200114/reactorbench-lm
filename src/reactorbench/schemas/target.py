"""Structured task targets and abstention semantics."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .base import (
    SCHEMA_VERSION,
    ContractId,
    ContractModel,
    NonNegativeInt,
    SchemaVersion,
    canonical_enum_tuple,
    require_unique,
)
from .enums import (
    AbstentionReason,
    ActionLabel,
    AsterSubsystem,
    CounterfactualChange,
    DiagnosisStatus,
    EventType,
    EvidenceSlot,
    FaultFamily,
    ObservedTrend,
    OperatingMode,
    TaskName,
)


def _validate_diagnosis(
    *,
    diagnosis_status: DiagnosisStatus,
    fault_labels: tuple[FaultFamily, ...],
    abstention_reason: AbstentionReason | None,
    immediate_action: ActionLabel | None = None,
) -> None:
    """Apply the single documented diagnosis/abstention truth table."""

    if diagnosis_status is DiagnosisStatus.DIAGNOSED:
        if not fault_labels:
            raise ValueError("DIAGNOSED requires at least one fault label")
        if abstention_reason is not None:
            raise ValueError("DIAGNOSED cannot include an abstention reason")
        if immediate_action is ActionLabel.INSUFFICIENT_EVIDENCE:
            raise ValueError("DIAGNOSED cannot select INSUFFICIENT_EVIDENCE")
    elif diagnosis_status is DiagnosisStatus.NO_FAULT:
        if fault_labels:
            raise ValueError("NO_FAULT requires an empty fault-label set")
        if abstention_reason is not None:
            raise ValueError("NO_FAULT cannot include an abstention reason")
        if immediate_action is ActionLabel.INSUFFICIENT_EVIDENCE:
            raise ValueError("NO_FAULT is a resolved conclusion, not an abstention")
    else:
        if fault_labels:
            raise ValueError("UNRESOLVED requires an empty fault-label set")
        if abstention_reason is not AbstentionReason.INSUFFICIENT_EVIDENCE:
            raise ValueError("UNRESOLVED requires abstention reason INSUFFICIENT_EVIDENCE")
        if (
            immediate_action is not None
            and immediate_action is not ActionLabel.INSUFFICIENT_EVIDENCE
        ):
            raise ValueError("UNRESOLVED requires immediate action INSUFFICIENT_EVIDENCE")


def _canonical_fault_labels(values: tuple[FaultFamily, ...]) -> tuple[FaultFamily, ...]:
    return canonical_enum_tuple(values, enum_type=FaultFamily, field_name="fault_labels")


class DecisionTarget(ContractModel):
    """Per-tick audit truth joining diagnosis, evidence, and selected action.

    Supervised task records use the task-specific contracts below instead of
    serializing this aggregate decision shape as every task's target.
    """

    schema_version: SchemaVersion = SCHEMA_VERSION
    scenario_id: ContractId
    decision_tick: NonNegativeInt
    diagnosis_status: DiagnosisStatus
    fault_labels: tuple[FaultFamily, ...] = ()
    evidence_event_ids: tuple[ContractId, ...] = ()
    evidence_slots: tuple[EvidenceSlot, ...] = ()
    immediate_action: ActionLabel
    abstention_reason: AbstentionReason | None = None

    @field_validator("fault_labels", mode="after")
    @classmethod
    def faults_are_a_canonical_set(cls, values: tuple[FaultFamily, ...]) -> tuple[FaultFamily, ...]:
        return _canonical_fault_labels(values)

    @field_validator("evidence_event_ids", "evidence_slots", mode="after")
    @classmethod
    def evidence_is_unique(cls, values: tuple[object, ...]) -> tuple[object, ...]:
        return require_unique(values, field_name="evidence")

    @model_validator(mode="after")
    def diagnosis_and_abstention_are_consistent(self) -> DecisionTarget:
        _validate_diagnosis(
            diagnosis_status=self.diagnosis_status,
            fault_labels=self.fault_labels,
            abstention_reason=self.abstention_reason,
            immediate_action=self.immediate_action,
        )
        if self.diagnosis_status is DiagnosisStatus.DIAGNOSED and not self.evidence_event_ids:
            raise ValueError("DIAGNOSED requires model-visible evidence events")
        return self


class ScenarioTargets(ContractModel):
    """Ordered trajectory-level decision truth used by ``StructuredTrajectory``."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    scenario_id: ContractId
    decisions: tuple[DecisionTarget, ...]

    @model_validator(mode="after")
    def decisions_are_ordered_and_scenario_scoped(self) -> ScenarioTargets:
        if not self.decisions:
            raise ValueError("decisions must contain at least one target")
        if any(item.scenario_id != self.scenario_id for item in self.decisions):
            raise ValueError("all decisions must reference the enclosing scenario_id")
        ticks = tuple(item.decision_tick for item in self.decisions)
        if ticks != tuple(sorted(ticks)):
            raise ValueError("decisions must be ordered by decision_tick")
        if len(ticks) != len(set(ticks)):
            raise ValueError("only one immediate action is allowed per decision tick")
        return self


class CausalContinuationTarget(ContractModel):
    """Structured half of ``continue_log``; generated prose is deliberately absent."""

    task_name: Literal[TaskName.CONTINUE_LOG] = TaskName.CONTINUE_LOG
    next_event_type: EventType


class FaultDiagnosisTarget(ContractModel):
    task_name: Literal[TaskName.FAULT_FAMILY] = TaskName.FAULT_FAMILY
    diagnosis_status: DiagnosisStatus
    fault_labels: tuple[FaultFamily, ...] = ()
    abstention_reason: AbstentionReason | None = None

    @field_validator("fault_labels", mode="after")
    @classmethod
    def faults_are_a_canonical_set(cls, values: tuple[FaultFamily, ...]) -> tuple[FaultFamily, ...]:
        return _canonical_fault_labels(values)

    @model_validator(mode="after")
    def diagnosis_is_consistent(self) -> FaultDiagnosisTarget:
        _validate_diagnosis(
            diagnosis_status=self.diagnosis_status,
            fault_labels=self.fault_labels,
            abstention_reason=self.abstention_reason,
        )
        return self


class EvidenceExtractionTarget(ContractModel):
    task_name: Literal[TaskName.EXTRACT_EVIDENCE] = TaskName.EXTRACT_EVIDENCE
    evidence_event_ids: tuple[ContractId, ...] = ()
    evidence_slots: tuple[EvidenceSlot, ...] = ()

    @field_validator("evidence_event_ids", "evidence_slots", mode="after")
    @classmethod
    def evidence_is_unique(cls, values: tuple[object, ...]) -> tuple[object, ...]:
        return require_unique(values, field_name="evidence")


class NextActionTarget(ContractModel):
    task_name: Literal[TaskName.NEXT_ACTION] = TaskName.NEXT_ACTION
    immediate_action: ActionLabel


class IncidentSummaryTarget(ContractModel):
    task_name: Literal[TaskName.INCIDENT_SUMMARY] = TaskName.INCIDENT_SUMMARY
    affected_subsystems: tuple[AsterSubsystem, ...] = ()
    observed_trend: ObservedTrend
    diagnosis_status: DiagnosisStatus
    fault_labels: tuple[FaultFamily, ...] = ()
    operating_mode: OperatingMode
    immediate_action: ActionLabel
    abstention_reason: AbstentionReason | None = None

    @field_validator("fault_labels", mode="after")
    @classmethod
    def faults_are_a_canonical_set(cls, values: tuple[FaultFamily, ...]) -> tuple[FaultFamily, ...]:
        return _canonical_fault_labels(values)

    @field_validator("affected_subsystems", mode="after")
    @classmethod
    def subsystems_are_a_canonical_set(
        cls, values: tuple[AsterSubsystem, ...]
    ) -> tuple[AsterSubsystem, ...]:
        return canonical_enum_tuple(
            values,
            enum_type=AsterSubsystem,
            field_name="affected_subsystems",
        )

    @model_validator(mode="after")
    def summary_is_consistent(self) -> IncidentSummaryTarget:
        _validate_diagnosis(
            diagnosis_status=self.diagnosis_status,
            fault_labels=self.fault_labels,
            abstention_reason=self.abstention_reason,
            immediate_action=self.immediate_action,
        )
        if self.diagnosis_status is DiagnosisStatus.DIAGNOSED and not self.affected_subsystems:
            raise ValueError("DIAGNOSED incident summaries require affected_subsystems")
        if self.diagnosis_status is DiagnosisStatus.NO_FAULT and self.affected_subsystems:
            raise ValueError("NO_FAULT incident summaries cannot claim affected_subsystems")
        return self


class CounterfactualConclusion(ContractModel):
    diagnosis_status: DiagnosisStatus
    fault_labels: tuple[FaultFamily, ...] = ()
    evidence_slots: tuple[EvidenceSlot, ...] = ()
    immediate_action: ActionLabel
    abstention_reason: AbstentionReason | None = None

    @field_validator("fault_labels", mode="after")
    @classmethod
    def faults_are_a_canonical_set(cls, values: tuple[FaultFamily, ...]) -> tuple[FaultFamily, ...]:
        return _canonical_fault_labels(values)

    @field_validator("evidence_slots", mode="after")
    @classmethod
    def evidence_is_unique(cls, values: tuple[EvidenceSlot, ...]) -> tuple[EvidenceSlot, ...]:
        return require_unique(values, field_name="evidence_slots")

    @model_validator(mode="after")
    def conclusion_is_consistent(self) -> CounterfactualConclusion:
        _validate_diagnosis(
            diagnosis_status=self.diagnosis_status,
            fault_labels=self.fault_labels,
            abstention_reason=self.abstention_reason,
            immediate_action=self.immediate_action,
        )
        return self


class CounterfactualComparisonTarget(ContractModel):
    task_name: Literal[TaskName.COUNTERFACTUAL_COMPARE] = TaskName.COUNTERFACTUAL_COMPARE
    baseline: CounterfactualConclusion
    counterfactual: CounterfactualConclusion
    changed_fields: tuple[CounterfactualChange, ...]
    decisive_evidence_slots: tuple[EvidenceSlot, ...]

    @field_validator("changed_fields", mode="after")
    @classmethod
    def changes_are_canonical(
        cls, values: tuple[CounterfactualChange, ...]
    ) -> tuple[CounterfactualChange, ...]:
        return canonical_enum_tuple(
            values,
            enum_type=CounterfactualChange,
            field_name="changed_fields",
        )

    @field_validator("decisive_evidence_slots", mode="after")
    @classmethod
    def decisive_evidence_is_unique(
        cls, values: tuple[EvidenceSlot, ...]
    ) -> tuple[EvidenceSlot, ...]:
        return require_unique(values, field_name="decisive_evidence_slots")

    @model_validator(mode="after")
    def declared_changes_match_conclusions(self) -> CounterfactualComparisonTarget:
        comparisons = {
            CounterfactualChange.DIAGNOSIS_STATUS: (
                self.baseline.diagnosis_status,
                self.counterfactual.diagnosis_status,
            ),
            CounterfactualChange.FAULT_LABELS: (
                self.baseline.fault_labels,
                self.counterfactual.fault_labels,
            ),
            CounterfactualChange.EVIDENCE_SLOTS: (
                self.baseline.evidence_slots,
                self.counterfactual.evidence_slots,
            ),
            CounterfactualChange.IMMEDIATE_ACTION: (
                self.baseline.immediate_action,
                self.counterfactual.immediate_action,
            ),
        }
        actual_changes = tuple(
            field_name
            for field_name in CounterfactualChange
            if comparisons[field_name][0] != comparisons[field_name][1]
        )
        if not actual_changes:
            raise ValueError("counterfactual conclusions must differ")
        if self.changed_fields != actual_changes:
            raise ValueError("changed_fields must exactly describe the changed conclusions")
        if not self.decisive_evidence_slots:
            raise ValueError("counterfactual comparison requires decisive evidence slots")
        changed_evidence = set(self.baseline.evidence_slots) ^ set(
            self.counterfactual.evidence_slots
        )
        if not changed_evidence.intersection(self.decisive_evidence_slots):
            raise ValueError("decisive evidence must identify a changed evidence fact")
        return self


type TaskTargetValue = Annotated[
    CausalContinuationTarget
    | FaultDiagnosisTarget
    | EvidenceExtractionTarget
    | NextActionTarget
    | IncidentSummaryTarget
    | CounterfactualComparisonTarget,
    Field(discriminator="task_name"),
]


class TaskTarget(ContractModel):
    """Versioned discriminated envelope preventing task/target mismatches."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    task_name: TaskName
    target: TaskTargetValue

    @model_validator(mode="after")
    def task_matches_target(self) -> TaskTarget:
        if self.task_name is not self.target.task_name:
            raise ValueError("task_name must match the discriminated target shape")
        return self
