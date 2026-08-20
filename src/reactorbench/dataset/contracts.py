"""Strict Phase 3 contracts separating audit lineage from renderer input.

``StructuredTrajectory`` is intentionally an audit-source object.  The models in
this module are the narrower boundary a renderer may consume: prompt-local facts,
bounded semantic context, and a task target stored outside the model input.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StrictStr, field_validator, model_validator

from reactorbench.schemas.base import (
    SCHEMA_VERSION,
    ContractId,
    ContractModel,
    NonNegativeInt,
    NormalizedFloat,
    SchemaVersion,
    SeedInt,
    canonical_enum_tuple,
    canonical_sha256,
    require_unique,
)
from reactorbench.schemas.enums import (
    AbstentionReason,
    ActionLabel,
    AsterSubsystem,
    ChannelQuality,
    ComponentState,
    CounterfactualChange,
    DiagnosisStatus,
    EventType,
    EvidenceSlot,
    FaultFamily,
    ObservationStatus,
    ObservedTrend,
    OperatingMode,
    PlantVariant,
    StateVariable,
    TaskName,
)
from reactorbench.schemas.events import EVENT_FIELD_MATRIX
from reactorbench.schemas.target import (
    CounterfactualConclusion,
    FaultDiagnosisTarget,
    IncidentSummaryTarget,
    NextActionTarget,
)

DATASET_CONTRACT_VERSION: Literal["0.1.0"] = "0.1.0"

Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
PromptFactRef = Annotated[StrictStr, Field(pattern=r"^[oec]-[0-9]{4}$")]


class ProjectionView(StrEnum):
    """Independently selected, allowlisted fact-selection policies.

    A view is audit metadata on ``ProjectionRecord`` and is deliberately absent
    from ``ModelInput``.  Selection code must choose a view from scenario shape,
    never from a target label or evidence identifier.
    """

    STANDARD_DECISION = "standard_decision"
    G07_STANDBY_DECISION = "g07_standby_decision"
    G12_MAP_INCLUDED_DECISION = "g12_map_included_decision"
    G12_MAP_WITHHELD_DECISION = "g12_map_withheld_decision"
    G13_INVENTORY_DECISION = "g13_inventory_decision"
    G14_FACTORED_DECISION = "g14_factored_decision"
    G15_SPARSE_DECISION = "g15_sparse_decision"


class ContextFactKind(StrEnum):
    PLANT_VARIANT = "plant_variant"
    STANDBY_RELATIONSHIP = "standby_relationship"
    DEPENDENCY_LINK = "dependency_link"


class ProjectedObservationFact(ContractModel):
    """One renderer-safe observed channel cell with a prompt-local reference."""

    fact_ref: PromptFactRef
    tick: NonNegativeInt
    channel_id: ContractId
    variable: StateVariable
    value: NormalizedFloat | None
    quality: ChannelQuality
    status: ObservationStatus

    @model_validator(mode="after")
    def missingness_is_consistent(self) -> ProjectedObservationFact:
        unavailable = self.quality is ChannelQuality.UNAVAILABLE
        missing = self.status is ObservationStatus.MISSING
        if unavailable != missing:
            raise ValueError("UNAVAILABLE quality and MISSING status must occur together")
        if missing != (self.value is None):
            raise ValueError("missing observation facts must have a null value")
        return self


_PROJECTED_EVENT_PAYLOAD_FIELDS = frozenset(
    {
        "operating_mode_before",
        "operating_mode_after",
        "component_state_before",
        "component_state_after",
        "variable",
        "value_before",
        "value_after",
        "observation_status",
        "channel_quality_before",
        "channel_quality",
        "commanded_value",
        "observed_value",
    }
)


class ProjectedEventFact(ContractModel):
    """One visible event without audit IDs, relations, evidence slots, or actions."""

    fact_ref: PromptFactRef
    tick: NonNegativeInt
    event_type: EventType
    subject_id: ContractId
    operating_mode_before: OperatingMode | None = None
    operating_mode_after: OperatingMode | None = None
    component_state_before: ComponentState | None = None
    component_state_after: ComponentState | None = None
    variable: StateVariable | None = None
    value_before: NormalizedFloat | None = None
    value_after: NormalizedFloat | None = None
    observation_status: ObservationStatus | None = None
    channel_quality_before: ChannelQuality | None = None
    channel_quality: ChannelQuality | None = None
    commanded_value: NormalizedFloat | None = None
    observed_value: NormalizedFloat | None = None

    @model_validator(mode="after")
    def payload_matches_visible_event_type(self) -> ProjectedEventFact:
        if self.event_type is EventType.ACTION_APPLIED:
            raise ValueError("ACTION_APPLIED is never a renderer-safe event fact")
        contract = EVENT_FIELD_MATRIX[self.event_type]
        # ACTION_APPLIED is rejected above, so its action_label field is never needed.
        required = contract.required.intersection(_PROJECTED_EVENT_PAYLOAD_FIELDS)
        missing = required - self.model_fields_set
        if missing:
            raise ValueError(
                f"{self.event_type.value} requires fields: {', '.join(sorted(missing))}"
            )
        null_required = {name for name in required if getattr(self, name) is None}
        if self.event_type is EventType.OBSERVATION_CHANGED:
            null_required.discard("value_after")
        if null_required:
            raise ValueError(
                f"{self.event_type.value} requires non-null fields: "
                f"{', '.join(sorted(null_required))}"
            )
        populated = {
            name for name in _PROJECTED_EVENT_PAYLOAD_FIELDS if getattr(self, name) is not None
        }
        allowed = contract.allowed.intersection(_PROJECTED_EVENT_PAYLOAD_FIELDS)
        forbidden = populated - allowed
        if forbidden:
            raise ValueError(
                f"{self.event_type.value} forbids fields: {', '.join(sorted(forbidden))}"
            )
        if self.event_type is EventType.OBSERVATION_CHANGED:
            missing_observation = self.observation_status is ObservationStatus.MISSING
            if missing_observation != (self.value_after is None):
                raise ValueError("MISSING observation status and null value_after must agree")
        return self


class PlantVariantContextFact(ContractModel):
    fact_ref: PromptFactRef
    fact_kind: Literal[ContextFactKind.PLANT_VARIANT] = ContextFactKind.PLANT_VARIANT
    plant_variant_id: PlantVariant


class StandbyRelationshipContextFact(ContractModel):
    """Semantic G07 context with the availability-bearing audit context ID removed."""

    fact_ref: PromptFactRef
    fact_kind: Literal[ContextFactKind.STANDBY_RELATIONSHIP] = ContextFactKind.STANDBY_RELATIONSHIP
    active_component_id: ContractId
    standby_component_id: ContractId
    standby_state: ComponentState
    support_component_id: ContractId
    support_state: ComponentState
    start_delay_ticks: NonNegativeInt

    @model_validator(mode="after")
    def states_and_roles_are_bounded(self) -> StandbyRelationshipContextFact:
        if self.active_component_id == self.standby_component_id:
            raise ValueError("active and standby component IDs must differ")
        allowed_states = {ComponentState.AVAILABLE, ComponentState.UNAVAILABLE}
        if self.standby_state not in allowed_states or self.support_state not in allowed_states:
            raise ValueError("standby context states must be AVAILABLE or UNAVAILABLE")
        if self.start_delay_ticks == 0:
            raise ValueError("standby start delay must be positive")
        return self


class DependencyLinkContextFact(ContractModel):
    """One G12 semantic dependency edge without an audit context identifier."""

    fact_ref: PromptFactRef
    fact_kind: Literal[ContextFactKind.DEPENDENCY_LINK] = ContextFactKind.DEPENDENCY_LINK
    support_component_id: ContractId
    dependent_component_id: ContractId

    @model_validator(mode="after")
    def endpoints_are_distinct(self) -> DependencyLinkContextFact:
        if self.support_component_id == self.dependent_component_id:
            raise ValueError("dependency endpoints must differ")
        return self


type ProjectedContextFact = Annotated[
    PlantVariantContextFact | StandbyRelationshipContextFact | DependencyLinkContextFact,
    Field(discriminator="fact_kind"),
]


class ModelInput(ContractModel):
    """The only structured contract a Phase 3 renderer may consume."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    cut_tick: NonNegativeInt
    source_event_index_exclusive: NonNegativeInt | None = None
    observation_facts: tuple[ProjectedObservationFact, ...] = Field(max_length=4096)
    event_facts: tuple[ProjectedEventFact, ...] = Field(max_length=1024)
    context_facts: tuple[ProjectedContextFact, ...] = Field(max_length=128)

    @model_validator(mode="after")
    def facts_are_local_bounded_and_canonical(self) -> ModelInput:
        if not self.observation_facts and not self.event_facts:
            raise ValueError("model input must contain at least one visible fact")
        if any(fact.tick > self.cut_tick for fact in self.observation_facts):
            raise ValueError("observation facts cannot occur after cut_tick")
        if any(fact.tick > self.cut_tick for fact in self.event_facts):
            raise ValueError("event facts cannot occur after cut_tick")

        observation_refs = tuple(fact.fact_ref for fact in self.observation_facts)
        event_refs = tuple(fact.fact_ref for fact in self.event_facts)
        if observation_refs != tuple(f"o-{index:04d}" for index in range(len(observation_refs))):
            raise ValueError("observation fact references must be canonical and contiguous")
        if event_refs != tuple(f"e-{index:04d}" for index in range(len(event_refs))):
            raise ValueError("event fact references must be canonical and contiguous")
        require_unique(observation_refs + event_refs, field_name="prompt fact references")

        observation_order = tuple(
            (fact.tick, fact.channel_id, fact.fact_ref) for fact in self.observation_facts
        )
        if observation_order != tuple(sorted(observation_order)):
            raise ValueError("observation facts must be ordered by tick and channel")
        event_order = tuple((fact.tick, fact.fact_ref) for fact in self.event_facts)
        if event_order != tuple(sorted(event_order)):
            raise ValueError("event facts must be ordered by tick and local reference")

        context_refs = tuple(fact.fact_ref for fact in self.context_facts)
        if context_refs != tuple(f"c-{index:04d}" for index in range(len(context_refs))):
            raise ValueError("context fact references must be canonical and contiguous")
        require_unique(
            observation_refs + event_refs + context_refs,
            field_name="prompt fact references",
        )
        context_payloads = tuple(
            canonical_sha256(fact.model_dump(mode="json", exclude={"fact_ref"}))
            for fact in self.context_facts
        )
        require_unique(context_payloads, field_name="context facts")
        return self

    def structured_fingerprint(self) -> str:
        """Hash only renderer-visible structured input, never lineage or target truth."""

        return canonical_sha256(self.model_dump(mode="json", round_trip=True))


class PromptEvidenceTarget(ContractModel):
    """Evidence target whose references resolve only inside the visible prompt."""

    task_name: Literal[TaskName.EXTRACT_EVIDENCE] = TaskName.EXTRACT_EVIDENCE
    fact_refs: tuple[PromptFactRef, ...] = ()
    evidence_slots: tuple[EvidenceSlot, ...] = ()

    @field_validator("fact_refs", "evidence_slots", mode="after")
    @classmethod
    def evidence_is_unique(cls, values: tuple[object, ...]) -> tuple[object, ...]:
        return require_unique(values, field_name="prompt evidence")


class PromptContinuationTarget(ContractModel):
    """Safe next-event target; applied action records are never prediction targets."""

    task_name: Literal[TaskName.CONTINUE_LOG] = TaskName.CONTINUE_LOG
    next_event_type: EventType

    @model_validator(mode="after")
    def next_event_is_renderer_safe(self) -> PromptContinuationTarget:
        if self.next_event_type is EventType.ACTION_APPLIED:
            raise ValueError("ACTION_APPLIED cannot be a continuation target")
        return self


class PromptCounterfactualComparisonTarget(ContractModel):
    """Pair target using prompt-local fact references instead of audit event IDs."""

    task_name: Literal[TaskName.COUNTERFACTUAL_COMPARE] = TaskName.COUNTERFACTUAL_COMPARE
    baseline: CounterfactualConclusion
    counterfactual: CounterfactualConclusion
    changed_fields: tuple[CounterfactualChange, ...]
    baseline_decisive_fact_refs: tuple[PromptFactRef, ...] = ()
    counterfactual_decisive_fact_refs: tuple[PromptFactRef, ...] = ()
    decisive_evidence_slots: tuple[EvidenceSlot, ...] = ()

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

    @field_validator(
        "baseline_decisive_fact_refs",
        "counterfactual_decisive_fact_refs",
        "decisive_evidence_slots",
        mode="after",
    )
    @classmethod
    def decisive_facts_are_unique(cls, values: tuple[object, ...]) -> tuple[object, ...]:
        return require_unique(values, field_name="counterfactual decisive facts")

    @model_validator(mode="after")
    def changes_match_conclusions(self) -> PromptCounterfactualComparisonTarget:
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
        actual = tuple(
            field
            for field in CounterfactualChange
            if comparisons[field][0] != comparisons[field][1]
        )
        if not actual:
            raise ValueError("counterfactual conclusions must differ")
        if self.changed_fields != actual:
            raise ValueError("changed_fields must exactly describe changed conclusions")
        if not self.baseline_decisive_fact_refs and not self.counterfactual_decisive_fact_refs:
            raise ValueError("counterfactual target requires at least one visible decisive fact")
        return self


type ProjectionTaskTargetValue = Annotated[
    PromptContinuationTarget
    | FaultDiagnosisTarget
    | PromptEvidenceTarget
    | NextActionTarget
    | IncidentSummaryTarget
    | PromptCounterfactualComparisonTarget,
    Field(discriminator="task_name"),
]


class ProjectionTaskTarget(ContractModel):
    """Decision-task envelope kept strictly separate from ``ModelInput``."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    task_name: TaskName
    target: ProjectionTaskTargetValue

    @model_validator(mode="after")
    def task_matches_target(self) -> ProjectionTaskTarget:
        if self.task_name is not self.target.task_name:
            raise ValueError("task_name must match the projection target shape")
        return self


class ProjectionLineage(ContractModel):
    """Audit-only lineage; this object must never be passed to a renderer."""

    trajectory_id: ContractId
    scenario_id: ContractId
    seed: SeedInt
    decision_tick: NonNegativeInt | None = None
    source_event_index_exclusive: NonNegativeInt | None = None
    task_name: TaskName
    projection_recipe_id: ContractId
    source_trajectory_sha256: Sha256
    provenance_sha256: Sha256
    structured_fingerprint_sha256: Sha256


class ProjectionRecord(ContractModel):
    """Auditable join of lineage, renderer input, and a separate supervised target."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    dataset_contract_version: Literal["0.1.0"] = DATASET_CONTRACT_VERSION
    projection_id: ContractId
    projection_view: ProjectionView
    lineage: ProjectionLineage
    model_input: ModelInput
    task_target: ProjectionTaskTarget
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def boundary_fields_are_consistent(self) -> ProjectionRecord:
        is_continuation = self.task_target.task_name is TaskName.CONTINUE_LOG
        if is_continuation:
            if self.lineage.decision_tick is not None:
                raise ValueError("continuation lineage cannot declare a decision tick")
            if self.lineage.source_event_index_exclusive is None:
                raise ValueError("continuation lineage requires an event-index cut")
            if (
                self.model_input.source_event_index_exclusive
                != self.lineage.source_event_index_exclusive
            ):
                raise ValueError("continuation model input and lineage event-index cuts must match")
        else:
            if self.lineage.decision_tick != self.model_input.cut_tick:
                raise ValueError("decision lineage tick must match model input cut_tick")
            if self.lineage.source_event_index_exclusive is not None:
                raise ValueError("decision lineage cannot declare an event-index cut")
            if self.model_input.source_event_index_exclusive is not None:
                raise ValueError("decision model input cannot declare an event-index cut")
        if self.task_target.task_name is TaskName.COUNTERFACTUAL_COMPARE:
            raise ValueError("counterfactual comparison requires CounterfactualProjectionRecord")
        if self.lineage.task_name is not self.task_target.task_name:
            raise ValueError("lineage and target task names must match")
        if self.lineage.structured_fingerprint_sha256 != self.model_input.structured_fingerprint():
            raise ValueError("lineage structured fingerprint must match model input")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("projection checksum does not match record content")
        visible_refs = {
            *(fact.fact_ref for fact in self.model_input.observation_facts),
            *(fact.fact_ref for fact in self.model_input.event_facts),
            *(fact.fact_ref for fact in self.model_input.context_facts),
        }
        if isinstance(self.task_target.target, PromptEvidenceTarget) and not set(
            self.task_target.target.fact_refs
        ).issubset(visible_refs):
            raise ValueError("evidence target references must resolve to visible prompt facts")
        return self


class CounterfactualPairInput(ContractModel):
    """Two independently safe renderer inputs for one matched comparison."""

    baseline: ModelInput
    counterfactual: ModelInput

    @model_validator(mode="after")
    def members_are_distinct(self) -> CounterfactualPairInput:
        if self.baseline.structured_fingerprint() == self.counterfactual.structured_fingerprint():
            raise ValueError("counterfactual inputs must differ")
        return self


class CounterfactualProjectionLineage(ContractModel):
    """Audit-only references to an already validated, complete matched group."""

    counterfactual_group_id: ContractId
    baseline_projection_id: ContractId
    counterfactual_projection_id: ContractId


class CounterfactualProjectionRecord(ContractModel):
    """Complete paired renderer input with target and audit lineage separated."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    dataset_contract_version: Literal["0.1.0"] = DATASET_CONTRACT_VERSION
    pair_id: ContractId
    lineage: CounterfactualProjectionLineage
    model_input: CounterfactualPairInput
    task_target: ProjectionTaskTarget
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def pair_target_and_refs_are_consistent(self) -> CounterfactualProjectionRecord:
        target = self.task_target.target
        if not isinstance(target, PromptCounterfactualComparisonTarget):
            raise ValueError("paired projection requires a counterfactual comparison target")
        baseline_refs = {
            *(fact.fact_ref for fact in self.model_input.baseline.observation_facts),
            *(fact.fact_ref for fact in self.model_input.baseline.event_facts),
            *(fact.fact_ref for fact in self.model_input.baseline.context_facts),
        }
        counterfactual_refs = {
            *(fact.fact_ref for fact in self.model_input.counterfactual.observation_facts),
            *(fact.fact_ref for fact in self.model_input.counterfactual.event_facts),
            *(fact.fact_ref for fact in self.model_input.counterfactual.context_facts),
        }
        if not set(target.baseline_decisive_fact_refs).issubset(baseline_refs):
            raise ValueError("baseline decisive references must be visible")
        if not set(target.counterfactual_decisive_fact_refs).issubset(counterfactual_refs):
            raise ValueError("counterfactual decisive references must be visible")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("counterfactual projection checksum does not match record")
        return self


def make_fault_target(
    *,
    diagnosis_status: DiagnosisStatus,
    fault_labels: tuple[FaultFamily, ...],
    abstention_reason: AbstentionReason | None,
) -> ProjectionTaskTarget:
    """Small strict constructor used by the projector and development pipeline."""

    canonical_faults = canonical_enum_tuple(
        fault_labels, enum_type=FaultFamily, field_name="fault_labels"
    )
    return ProjectionTaskTarget(
        task_name=TaskName.FAULT_FAMILY,
        target=FaultDiagnosisTarget(
            diagnosis_status=diagnosis_status,
            fault_labels=canonical_faults,
            abstention_reason=abstention_reason,
        ),
    )


def make_action_target(*, action: ActionLabel) -> ProjectionTaskTarget:
    return ProjectionTaskTarget(
        task_name=TaskName.NEXT_ACTION,
        target=NextActionTarget(immediate_action=action),
    )


def make_evidence_target(
    *, fact_refs: tuple[str, ...], evidence_slots: tuple[EvidenceSlot, ...]
) -> ProjectionTaskTarget:
    return ProjectionTaskTarget(
        task_name=TaskName.EXTRACT_EVIDENCE,
        target=PromptEvidenceTarget(fact_refs=fact_refs, evidence_slots=evidence_slots),
    )


def make_continuation_target(*, next_event_type: EventType) -> ProjectionTaskTarget:
    return ProjectionTaskTarget(
        task_name=TaskName.CONTINUE_LOG,
        target=PromptContinuationTarget(next_event_type=next_event_type),
    )


def make_incident_summary_target(
    *,
    affected_subsystems: tuple[AsterSubsystem, ...],
    observed_trend: ObservedTrend,
    diagnosis_status: DiagnosisStatus,
    fault_labels: tuple[FaultFamily, ...],
    operating_mode: OperatingMode,
    immediate_action: ActionLabel,
    abstention_reason: AbstentionReason | None,
) -> ProjectionTaskTarget:
    return ProjectionTaskTarget(
        task_name=TaskName.INCIDENT_SUMMARY,
        target=IncidentSummaryTarget(
            affected_subsystems=affected_subsystems,
            observed_trend=observed_trend,
            diagnosis_status=diagnosis_status,
            fault_labels=fault_labels,
            operating_mode=operating_mode,
            immediate_action=immediate_action,
            abstention_reason=abstention_reason,
        ),
    )
