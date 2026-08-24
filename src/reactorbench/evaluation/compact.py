"""Versioned compact targets and truth-independent constrained token selection.

The compact wire language is deliberately smaller than the canonical JSON audit
representation.  It carries only the task discriminator and target fields already
present in :class:`ProjectionTaskTargetValue`.  Prompt-local reference allowlists are
provided separately through :class:`CompactTargetContext`; no target truth, lineage,
scenario identifier, provenance, or latent simulator state is accepted by the decoder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, cast

from pydantic import Field, field_validator, model_validator

from reactorbench.dataset.contracts import (
    ProjectionTaskTargetValue,
    PromptContinuationTarget,
    PromptCounterfactualComparisonTarget,
    PromptEvidenceTarget,
    PromptFactRef,
)
from reactorbench.schemas.base import ContractModel, canonical_json_bytes, require_unique
from reactorbench.schemas.enums import (
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
from reactorbench.schemas.target import (
    CounterfactualConclusion,
    FaultDiagnosisTarget,
    IncidentSummaryTarget,
    NextActionTarget,
)

if TYPE_CHECKING:
    import torch

    from reactorbench.tokenizer import ProjectTokenizer

COMPACT_TARGET_VERSION: Literal["0.2.0"] = "0.2.0"
COMPACT_WIRE_PREFIX: Literal["RB2"] = "RB2"
FIELD_SEPARATOR = "|"
LIST_SEPARATOR = ","
CONCLUSION_SEPARATOR = "~"
EMPTY_VALUE = "-"
_CODE_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_COMPACT_TARGET_BYTES = 16 * 1024
MAX_VISIBLE_FACT_REFS = 4096 + 1024 + 128
MAX_CONSTRAINED_GENERATED_TOKENS = 4096
MAX_CONSTRAINT_CACHE_ENTRIES = 4096

_SAFE_WIRE = re.compile(r"^[A-Za-z0-9_|,~\-]*$")
_FACT_REF = re.compile(r"^([oec])-([0-9]{4})$")
_TARGET_TYPES = (
    PromptContinuationTarget,
    FaultDiagnosisTarget,
    PromptEvidenceTarget,
    NextActionTarget,
    IncidentSummaryTarget,
    PromptCounterfactualComparisonTarget,
)


def _tokenizer_runtime_contract() -> tuple[type[ProjectTokenizer], frozenset[int], int]:
    """Load the tokenizer contract lazily to avoid package-initialization cycles."""

    from reactorbench.tokenizer import (
        BOS_ID,
        EOS_ID,
        PAD_ID,
        UNK_ID,
        ProjectTokenizer,
    )

    return ProjectTokenizer, frozenset({UNK_ID, BOS_ID, EOS_ID, PAD_ID}), EOS_ID


class CompactTargetError(ValueError):
    """Raised when compact target text violates the closed v0.2 contract."""


class CompactDecodingError(RuntimeError):
    """Raised when a constrained prefix has no legal next token."""


def _fact_ref_parts(value: str) -> tuple[str, int]:
    match = _FACT_REF.fullmatch(value)
    if match is None:
        raise CompactTargetError("fact reference does not match the prompt-local contract")
    return match.group(1), int(match.group(2))


def _refs_are_locally_ordered(values: tuple[str, ...]) -> bool:
    last_by_namespace: dict[str, int] = {}
    for value in values:
        namespace, index = _fact_ref_parts(value)
        if index <= last_by_namespace.get(namespace, -1):
            return False
        last_by_namespace[namespace] = index
    return True


class CompactTargetContext(ContractModel):
    """Truth-independent inputs from which one decoder constraint is constructed."""

    contract_version: Literal["0.2.0"] = COMPACT_TARGET_VERSION
    task_name: TaskName
    visible_fact_refs: tuple[PromptFactRef, ...] = Field(
        min_length=1, max_length=MAX_VISIBLE_FACT_REFS
    )
    counterfactual_visible_fact_refs: tuple[PromptFactRef, ...] = Field(
        default=(), max_length=MAX_VISIBLE_FACT_REFS
    )

    @field_validator("visible_fact_refs", "counterfactual_visible_fact_refs", mode="after")
    @classmethod
    def reference_inventory_is_bounded_and_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        require_unique(values, field_name="compact decoder visible fact references")
        by_namespace: dict[str, list[int]] = {"o": [], "e": [], "c": []}
        for value in values:
            namespace, index = _fact_ref_parts(value)
            by_namespace[namespace].append(index)
        if any(indices != list(range(len(indices))) for indices in by_namespace.values()):
            raise ValueError("visible fact-reference namespaces must be canonical and contiguous")
        return values

    @model_validator(mode="after")
    def secondary_inventory_is_task_scoped(self) -> CompactTargetContext:
        is_counterfactual = self.task_name is TaskName.COUNTERFACTUAL_COMPARE
        if is_counterfactual != bool(self.counterfactual_visible_fact_refs):
            raise ValueError(
                "only counterfactual comparison requires a second visible reference inventory"
            )
        return self


def _enum_codes(enum_type: type[StrEnum]) -> tuple[str, ...]:
    members = tuple(enum_type)
    if len(members) > len(_CODE_ALPHABET):  # pragma: no cover - closed-enum defense
        raise CompactTargetError("compact enum exceeds the single-atom code table")
    return tuple(_CODE_ALPHABET[index] for index in range(len(members)))


def _encode_enum_code(value: StrEnum, enum_type: type[StrEnum]) -> str:
    members = tuple(enum_type)
    try:
        return _CODE_ALPHABET[members.index(value)]
    except ValueError as error:  # pragma: no cover - exact schema values prevent this
        raise CompactTargetError("value is outside its compact enum code table") from error


def _decode_enum_code(value: str, enum_type: type[StrEnum], *, field_name: str) -> StrEnum:
    codes = _enum_codes(enum_type)
    if value not in codes:
        raise CompactTargetError(f"{field_name} contains an invalid compact enum code")
    return tuple(enum_type)[codes.index(value)]


def _encode_enum_code_list(values: tuple[StrEnum, ...], enum_type: type[StrEnum]) -> str:
    if not values:
        return EMPTY_VALUE
    return LIST_SEPARATOR.join(_encode_enum_code(value, enum_type) for value in values)


def _parse_enum_code_list(
    value: str,
    enum_type: type[StrEnum],
    *,
    field_name: str,
    canonical: bool,
) -> tuple[StrEnum, ...]:
    if value == EMPTY_VALUE:
        return ()
    if not value or EMPTY_VALUE in value.split(LIST_SEPARATOR):
        raise CompactTargetError(f"{field_name} has an invalid empty-list encoding")
    parsed = tuple(
        _decode_enum_code(item, enum_type, field_name=field_name)
        for item in value.split(LIST_SEPARATOR)
    )
    if len(parsed) != len(set(parsed)):
        raise CompactTargetError(f"{field_name} must not contain duplicates")
    if canonical:
        order = {member: index for index, member in enumerate(enum_type)}
        if parsed != tuple(sorted(parsed, key=order.__getitem__)):
            raise CompactTargetError(f"{field_name} is not in canonical enum order")
    return parsed


def _validate_refs(
    values: tuple[str, ...],
    *,
    visible: tuple[str, ...],
    field_name: str,
) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise CompactTargetError(f"{field_name} must not contain duplicates")
    visible_set = frozenset(visible)
    if not set(values).issubset(visible_set):
        raise CompactTargetError(f"{field_name} must resolve to prompt-visible facts")
    if not _refs_are_locally_ordered(values):
        raise CompactTargetError(f"{field_name} must increase within each reference namespace")
    return values


def _encode_refs(
    values: tuple[str, ...],
    *,
    visible: tuple[str, ...],
    field_name: str,
) -> str:
    validated = _validate_refs(values, visible=visible, field_name=field_name)
    return LIST_SEPARATOR.join(validated) if validated else EMPTY_VALUE


def _encode_conclusion(value: CounterfactualConclusion) -> str:
    return CONCLUSION_SEPARATOR.join(
        (
            _encode_enum_code(value.diagnosis_status, DiagnosisStatus),
            _encode_enum_code_list(cast(tuple[StrEnum, ...], value.fault_labels), FaultFamily),
            _encode_enum_code_list(cast(tuple[StrEnum, ...], value.evidence_slots), EvidenceSlot),
            _encode_enum_code(value.immediate_action, ActionLabel),
            (
                EMPTY_VALUE
                if value.abstention_reason is None
                else _encode_enum_code(value.abstention_reason, AbstentionReason)
            ),
        )
    )


def serialize_compact_target(
    target: ProjectionTaskTargetValue,
    *,
    context: CompactTargetContext,
) -> str:
    """Compile one strict project target into its unique compact representation."""

    if type(target) not in _TARGET_TYPES:
        raise TypeError("target must be an exact projection task target")
    if type(context) is not CompactTargetContext:
        raise TypeError("context must be an exact CompactTargetContext")
    if target.task_name is not context.task_name:
        raise CompactTargetError("target task does not match compact decoder context")

    fields: tuple[str, ...]
    if type(target) is PromptContinuationTarget:
        fields = (_encode_enum_code(target.next_event_type, EventType),)
    elif type(target) is FaultDiagnosisTarget:
        fields = (
            _encode_enum_code(target.diagnosis_status, DiagnosisStatus),
            _encode_enum_code_list(cast(tuple[StrEnum, ...], target.fault_labels), FaultFamily),
            (
                EMPTY_VALUE
                if target.abstention_reason is None
                else _encode_enum_code(target.abstention_reason, AbstentionReason)
            ),
        )
    elif type(target) is PromptEvidenceTarget:
        fields = (
            _encode_refs(
                target.fact_refs,
                visible=context.visible_fact_refs,
                field_name="evidence fact references",
            ),
            _encode_enum_code_list(cast(tuple[StrEnum, ...], target.evidence_slots), EvidenceSlot),
        )
    elif type(target) is NextActionTarget:
        fields = (_encode_enum_code(target.immediate_action, ActionLabel),)
    elif type(target) is IncidentSummaryTarget:
        fields = (
            _encode_enum_code_list(
                cast(tuple[StrEnum, ...], target.affected_subsystems), AsterSubsystem
            ),
            _encode_enum_code(target.observed_trend, ObservedTrend),
            _encode_enum_code(target.diagnosis_status, DiagnosisStatus),
            _encode_enum_code_list(cast(tuple[StrEnum, ...], target.fault_labels), FaultFamily),
            _encode_enum_code(target.operating_mode, OperatingMode),
            _encode_enum_code(target.immediate_action, ActionLabel),
            (
                EMPTY_VALUE
                if target.abstention_reason is None
                else _encode_enum_code(target.abstention_reason, AbstentionReason)
            ),
        )
    else:
        counterfactual = cast(PromptCounterfactualComparisonTarget, target)
        fields = (
            _encode_conclusion(counterfactual.baseline),
            _encode_conclusion(counterfactual.counterfactual),
            _encode_enum_code_list(
                cast(tuple[StrEnum, ...], counterfactual.changed_fields), CounterfactualChange
            ),
            _encode_refs(
                counterfactual.baseline_decisive_fact_refs,
                visible=context.visible_fact_refs,
                field_name="baseline decisive fact references",
            ),
            _encode_refs(
                counterfactual.counterfactual_decisive_fact_refs,
                visible=context.counterfactual_visible_fact_refs,
                field_name="counterfactual decisive fact references",
            ),
            _encode_enum_code_list(
                cast(tuple[StrEnum, ...], counterfactual.decisive_evidence_slots), EvidenceSlot
            ),
        )
    text = FIELD_SEPARATOR.join((COMPACT_WIRE_PREFIX, target.task_name.value, *fields))
    if len(text.encode("utf-8")) > MAX_COMPACT_TARGET_BYTES:
        raise CompactTargetError("compact target exceeds its UTF-8 byte limit")
    return text


def _parse_refs(
    value: str,
    *,
    visible: tuple[str, ...],
    field_name: str,
) -> tuple[str, ...]:
    if value == EMPTY_VALUE:
        return ()
    if not value or EMPTY_VALUE in value.split(LIST_SEPARATOR):
        raise CompactTargetError(f"{field_name} has an invalid empty-list encoding")
    refs = tuple(value.split(LIST_SEPARATOR))
    for ref in refs:
        _fact_ref_parts(ref)
    return _validate_refs(refs, visible=visible, field_name=field_name)


def _parse_conclusion(value: str) -> CounterfactualConclusion:
    fields = value.split(CONCLUSION_SEPARATOR)
    if len(fields) != 5:
        raise CompactTargetError("counterfactual conclusion must contain exactly five fields")
    status = cast(
        DiagnosisStatus,
        _decode_enum_code(fields[0], DiagnosisStatus, field_name="diagnosis status"),
    )
    faults = cast(
        tuple[FaultFamily, ...],
        _parse_enum_code_list(fields[1], FaultFamily, field_name="fault labels", canonical=True),
    )
    slots = cast(
        tuple[EvidenceSlot, ...],
        _parse_enum_code_list(
            fields[2], EvidenceSlot, field_name="evidence slots", canonical=False
        ),
    )
    action = cast(
        ActionLabel,
        _decode_enum_code(fields[3], ActionLabel, field_name="immediate action"),
    )
    abstention = (
        None
        if fields[4] == EMPTY_VALUE
        else cast(
            AbstentionReason,
            _decode_enum_code(fields[4], AbstentionReason, field_name="abstention reason"),
        )
    )
    return CounterfactualConclusion(
        diagnosis_status=status,
        fault_labels=faults,
        evidence_slots=slots,
        immediate_action=action,
        abstention_reason=abstention,
    )


def parse_compact_target(
    text: str,
    *,
    context: CompactTargetContext,
) -> ProjectionTaskTargetValue:
    """Compile strict compact text back to the canonical project target model."""

    if type(text) is not str:
        raise TypeError("compact target text must be an exact string")
    if type(context) is not CompactTargetContext:
        raise TypeError("context must be an exact CompactTargetContext")
    if not text or len(text.encode("utf-8")) > MAX_COMPACT_TARGET_BYTES:
        raise CompactTargetError("compact target text is empty or exceeds its byte limit")
    if _SAFE_WIRE.fullmatch(text) is None:
        raise CompactTargetError("compact target contains a character outside the wire allowlist")
    pieces = text.split(FIELD_SEPARATOR)
    if len(pieces) < 3 or pieces[0] != COMPACT_WIRE_PREFIX:
        raise CompactTargetError("compact target has an invalid wire prefix")
    try:
        task_name = TaskName(pieces[1])
    except ValueError as error:
        raise CompactTargetError("compact target contains an invalid task name") from error
    if task_name is not context.task_name:
        raise CompactTargetError("compact target task does not match decoder context")
    fields = pieces[2:]

    target: ProjectionTaskTargetValue
    try:
        if task_name is TaskName.CONTINUE_LOG:
            if len(fields) != 1:
                raise CompactTargetError("continue_log requires exactly one field")
            target = PromptContinuationTarget(
                next_event_type=cast(
                    EventType,
                    _decode_enum_code(fields[0], EventType, field_name="next event type"),
                )
            )
        elif task_name is TaskName.FAULT_FAMILY:
            if len(fields) != 3:
                raise CompactTargetError("fault_family requires exactly three fields")
            target = FaultDiagnosisTarget(
                diagnosis_status=cast(
                    DiagnosisStatus,
                    _decode_enum_code(fields[0], DiagnosisStatus, field_name="diagnosis status"),
                ),
                fault_labels=cast(
                    tuple[FaultFamily, ...],
                    _parse_enum_code_list(
                        fields[1], FaultFamily, field_name="fault labels", canonical=True
                    ),
                ),
                abstention_reason=(
                    None
                    if fields[2] == EMPTY_VALUE
                    else cast(
                        AbstentionReason,
                        _decode_enum_code(
                            fields[2],
                            AbstentionReason,
                            field_name="abstention reason",
                        ),
                    )
                ),
            )
        elif task_name is TaskName.EXTRACT_EVIDENCE:
            if len(fields) != 2:
                raise CompactTargetError("extract_evidence requires exactly two fields")
            target = PromptEvidenceTarget(
                fact_refs=_parse_refs(
                    fields[0],
                    visible=context.visible_fact_refs,
                    field_name="evidence fact references",
                ),
                evidence_slots=cast(
                    tuple[EvidenceSlot, ...],
                    _parse_enum_code_list(
                        fields[1], EvidenceSlot, field_name="evidence slots", canonical=False
                    ),
                ),
            )
        elif task_name is TaskName.NEXT_ACTION:
            if len(fields) != 1:
                raise CompactTargetError("next_action requires exactly one field")
            target = NextActionTarget(
                immediate_action=cast(
                    ActionLabel,
                    _decode_enum_code(fields[0], ActionLabel, field_name="immediate action"),
                )
            )
        elif task_name is TaskName.INCIDENT_SUMMARY:
            if len(fields) != 7:
                raise CompactTargetError("incident_summary requires exactly seven fields")
            target = IncidentSummaryTarget(
                affected_subsystems=cast(
                    tuple[AsterSubsystem, ...],
                    _parse_enum_code_list(
                        fields[0],
                        AsterSubsystem,
                        field_name="affected subsystems",
                        canonical=True,
                    ),
                ),
                observed_trend=cast(
                    ObservedTrend,
                    _decode_enum_code(fields[1], ObservedTrend, field_name="observed trend"),
                ),
                diagnosis_status=cast(
                    DiagnosisStatus,
                    _decode_enum_code(fields[2], DiagnosisStatus, field_name="diagnosis status"),
                ),
                fault_labels=cast(
                    tuple[FaultFamily, ...],
                    _parse_enum_code_list(
                        fields[3], FaultFamily, field_name="fault labels", canonical=True
                    ),
                ),
                operating_mode=cast(
                    OperatingMode,
                    _decode_enum_code(fields[4], OperatingMode, field_name="operating mode"),
                ),
                immediate_action=cast(
                    ActionLabel,
                    _decode_enum_code(fields[5], ActionLabel, field_name="immediate action"),
                ),
                abstention_reason=(
                    None
                    if fields[6] == EMPTY_VALUE
                    else cast(
                        AbstentionReason,
                        _decode_enum_code(
                            fields[6],
                            AbstentionReason,
                            field_name="abstention reason",
                        ),
                    )
                ),
            )
        else:
            if len(fields) != 6:
                raise CompactTargetError("counterfactual_compare requires exactly six fields")
            target = PromptCounterfactualComparisonTarget(
                baseline=_parse_conclusion(fields[0]),
                counterfactual=_parse_conclusion(fields[1]),
                changed_fields=cast(
                    tuple[CounterfactualChange, ...],
                    _parse_enum_code_list(
                        fields[2],
                        CounterfactualChange,
                        field_name="changed fields",
                        canonical=True,
                    ),
                ),
                baseline_decisive_fact_refs=_parse_refs(
                    fields[3],
                    visible=context.visible_fact_refs,
                    field_name="baseline decisive fact references",
                ),
                counterfactual_decisive_fact_refs=_parse_refs(
                    fields[4],
                    visible=context.counterfactual_visible_fact_refs,
                    field_name="counterfactual decisive fact references",
                ),
                decisive_evidence_slots=cast(
                    tuple[EvidenceSlot, ...],
                    _parse_enum_code_list(
                        fields[5],
                        EvidenceSlot,
                        field_name="decisive evidence slots",
                        canonical=False,
                    ),
                ),
            )
    except CompactTargetError:
        raise
    except ValueError as error:
        raise CompactTargetError("compact target violates the strict task schema") from error

    if serialize_compact_target(target, context=context) != text:
        raise CompactTargetError("compact target is not in its unique canonical form")
    return target


def compact_target_json(text: str, *, context: CompactTargetContext) -> str:
    """Return the canonical JSON audit/API representation for compact text."""

    target = parse_compact_target(text, context=context)
    return canonical_json_bytes(target.model_dump(mode="json", round_trip=True)).decode("utf-8")


@dataclass(frozen=True)
class _ScalarSpec:
    atoms: tuple[str, ...]

    def complete(self, value: str) -> bool:
        return value in self.atoms

    def prefix(self, value: str) -> bool:
        return any(atom.startswith(value) for atom in self.atoms)


@dataclass(frozen=True)
class _ListSpec:
    atoms: tuple[str, ...]
    allow_empty: bool = True
    require_canonical_enum_order: bool = False
    fact_references: bool = False

    def _parts_are_valid(self, value: str, *, partial: bool) -> bool:
        if value == EMPTY_VALUE:
            return self.allow_empty
        if value.startswith(EMPTY_VALUE):
            return False
        parts = value.split(LIST_SEPARATOR)
        completed = parts[:-1] if partial else parts
        current = parts[-1] if partial else None
        if any(part not in self.atoms for part in completed):
            return False
        if len(completed) != len(set(completed)):
            return False
        if self.require_canonical_enum_order:
            indices = tuple(self.atoms.index(part) for part in completed)
            if indices != tuple(sorted(indices)):
                return False
        if self.fact_references:
            try:
                if not _refs_are_locally_ordered(tuple(completed)):
                    return False
            except CompactTargetError:
                return False
        if not partial:
            return bool(completed)
        available = tuple(atom for atom in self.atoms if atom not in completed)
        if self.require_canonical_enum_order and completed:
            lower_bound = self.atoms.index(completed[-1])
            available = tuple(atom for atom in available if self.atoms.index(atom) > lower_bound)
        if self.fact_references and completed:
            last_by_namespace: dict[str, int] = {}
            for ref in completed:
                namespace, index = _fact_ref_parts(ref)
                last_by_namespace[namespace] = index
            available = tuple(
                ref
                for ref in available
                if _fact_ref_parts(ref)[1] > last_by_namespace.get(_fact_ref_parts(ref)[0], -1)
            )
        if current is None:
            return True
        return any(atom.startswith(current) for atom in available)

    def complete(self, value: str) -> bool:
        if value == EMPTY_VALUE:
            return self.allow_empty
        return self._parts_are_valid(value, partial=False)

    def prefix(self, value: str) -> bool:
        if value == "":
            return self.allow_empty or bool(self.atoms)
        return self._parts_are_valid(value, partial=True)


@dataclass(frozen=True)
class _ConclusionSpec:
    def complete(self, value: str) -> bool:
        try:
            conclusion = _parse_conclusion(value)
        except (CompactTargetError, ValueError):
            return False
        return _encode_conclusion(conclusion) == value

    def prefix(self, value: str) -> bool:
        fields = value.split(CONCLUSION_SEPARATOR)
        if len(fields) > 5:
            return False
        completed: list[str] = []
        for field in fields[:-1]:
            spec = _conclusion_field_spec(len(completed), tuple(completed))
            if not spec.complete(field):
                return False
            completed.append(field)
        spec = _conclusion_field_spec(len(completed), tuple(completed))
        return spec.prefix(fields[-1])


type _FieldSpec = _ScalarSpec | _ListSpec | _ConclusionSpec


def _diagnosis_fault_spec(status: DiagnosisStatus) -> _ListSpec:
    atoms = _enum_codes(FaultFamily) if status is DiagnosisStatus.DIAGNOSED else ()
    return _ListSpec(
        atoms,
        allow_empty=status is not DiagnosisStatus.DIAGNOSED,
        require_canonical_enum_order=True,
    )


def _diagnosis_action_spec(status: DiagnosisStatus) -> _ScalarSpec:
    if status is DiagnosisStatus.UNRESOLVED:
        return _ScalarSpec((_encode_enum_code(ActionLabel.INSUFFICIENT_EVIDENCE, ActionLabel),))
    return _ScalarSpec(
        tuple(
            _encode_enum_code(action, ActionLabel)
            for action in ActionLabel
            if action is not ActionLabel.INSUFFICIENT_EVIDENCE
        )
    )


def _diagnosis_abstention_spec(status: DiagnosisStatus) -> _ScalarSpec:
    if status is DiagnosisStatus.UNRESOLVED:
        return _ScalarSpec(
            (_encode_enum_code(AbstentionReason.INSUFFICIENT_EVIDENCE, AbstentionReason),)
        )
    return _ScalarSpec((EMPTY_VALUE,))


def _conclusion_field_spec(index: int, completed: tuple[str, ...]) -> _FieldSpec:
    if index == 0:
        return _ScalarSpec(_enum_codes(DiagnosisStatus))
    status = cast(
        DiagnosisStatus,
        _decode_enum_code(completed[0], DiagnosisStatus, field_name="diagnosis status"),
    )
    if index == 1:
        return _ListSpec(
            _enum_codes(FaultFamily) if status is DiagnosisStatus.DIAGNOSED else (),
            allow_empty=status is not DiagnosisStatus.DIAGNOSED,
            require_canonical_enum_order=True,
        )
    if index == 2:
        return _ListSpec(_enum_codes(EvidenceSlot))
    if index == 3:
        actions = (
            (ActionLabel.INSUFFICIENT_EVIDENCE,)
            if status is DiagnosisStatus.UNRESOLVED
            else tuple(
                action for action in ActionLabel if action is not ActionLabel.INSUFFICIENT_EVIDENCE
            )
        )
        return _ScalarSpec(tuple(_encode_enum_code(action, ActionLabel) for action in actions))
    if index == 4:
        return _ScalarSpec(
            (_encode_enum_code(AbstentionReason.INSUFFICIENT_EVIDENCE, AbstentionReason),)
            if status is DiagnosisStatus.UNRESOLVED
            else (EMPTY_VALUE,)
        )
    raise CompactTargetError("counterfactual conclusion has too many fields")


def _changed_fields(
    baseline: CounterfactualConclusion,
    counterfactual: CounterfactualConclusion,
) -> tuple[CounterfactualChange, ...]:
    comparisons = {
        CounterfactualChange.DIAGNOSIS_STATUS: (
            baseline.diagnosis_status,
            counterfactual.diagnosis_status,
        ),
        CounterfactualChange.FAULT_LABELS: (
            baseline.fault_labels,
            counterfactual.fault_labels,
        ),
        CounterfactualChange.EVIDENCE_SLOTS: (
            baseline.evidence_slots,
            counterfactual.evidence_slots,
        ),
        CounterfactualChange.IMMEDIATE_ACTION: (
            baseline.immediate_action,
            counterfactual.immediate_action,
        ),
    }
    return tuple(
        field for field in CounterfactualChange if comparisons[field][0] != comparisons[field][1]
    )


def _task_field_spec(
    task_name: TaskName,
    index: int,
    completed: tuple[str, ...],
    context: CompactTargetContext,
) -> _FieldSpec:
    if task_name is TaskName.CONTINUE_LOG:
        if index == 0:
            return _ScalarSpec(
                tuple(
                    _encode_enum_code(event, EventType)
                    for event in EventType
                    if event is not EventType.ACTION_APPLIED
                )
            )
    elif task_name is TaskName.FAULT_FAMILY:
        if index == 0:
            return _ScalarSpec(_enum_codes(DiagnosisStatus))
        status = cast(
            DiagnosisStatus,
            _decode_enum_code(completed[0], DiagnosisStatus, field_name="diagnosis status"),
        )
        if index == 1:
            return _diagnosis_fault_spec(status)
        if index == 2:
            return _diagnosis_abstention_spec(status)
    elif task_name is TaskName.EXTRACT_EVIDENCE:
        if index == 0:
            return _ListSpec(
                context.visible_fact_refs,
                fact_references=True,
            )
        if index == 1:
            return _ListSpec(_enum_codes(EvidenceSlot))
    elif task_name is TaskName.NEXT_ACTION:
        if index == 0:
            return _ScalarSpec(_enum_codes(ActionLabel))
    elif task_name is TaskName.INCIDENT_SUMMARY:
        if index == 0:
            return _ListSpec(_enum_codes(AsterSubsystem), require_canonical_enum_order=True)
        if index == 1:
            return _ScalarSpec(_enum_codes(ObservedTrend))
        if index == 2:
            subsystems_present = completed[0] != EMPTY_VALUE
            statuses = tuple(
                _encode_enum_code(status, DiagnosisStatus)
                for status in DiagnosisStatus
                if not (
                    (status is DiagnosisStatus.DIAGNOSED and not subsystems_present)
                    or (status is DiagnosisStatus.NO_FAULT and subsystems_present)
                )
            )
            return _ScalarSpec(statuses)
        status = cast(
            DiagnosisStatus,
            _decode_enum_code(completed[2], DiagnosisStatus, field_name="diagnosis status"),
        )
        if index == 3:
            return _diagnosis_fault_spec(status)
        if index == 4:
            return _ScalarSpec(_enum_codes(OperatingMode))
        if index == 5:
            return _diagnosis_action_spec(status)
        if index == 6:
            return _diagnosis_abstention_spec(status)
    else:
        if index in {0, 1}:
            return _ConclusionSpec()
        if index == 2:
            baseline = _parse_conclusion(completed[0])
            counterfactual = _parse_conclusion(completed[1])
            changes = _changed_fields(baseline, counterfactual)
            if not changes:
                return _ScalarSpec(())
            return _ScalarSpec(
                (_encode_enum_code_list(cast(tuple[StrEnum, ...], changes), CounterfactualChange),)
            )
        if index == 3:
            return _ListSpec(context.visible_fact_refs, fact_references=True)
        if index == 4:
            baseline_has_refs = completed[3] != EMPTY_VALUE
            return _ListSpec(
                context.counterfactual_visible_fact_refs,
                allow_empty=baseline_has_refs,
                fact_references=True,
            )
        if index == 5:
            return _ListSpec(_enum_codes(EvidenceSlot))
    raise CompactTargetError("compact target contains too many task fields")


def _payload_prefix_is_viable(payload: str, context: CompactTargetContext) -> bool:
    fields = payload.split(FIELD_SEPARATOR)
    completed: list[str] = []
    try:
        for field in fields[:-1]:
            spec = _task_field_spec(context.task_name, len(completed), tuple(completed), context)
            if not spec.complete(field):
                return False
            completed.append(field)
        spec = _task_field_spec(context.task_name, len(completed), tuple(completed), context)
        return spec.prefix(fields[-1])
    except (CompactTargetError, ValueError):
        return False


def compact_output_contract() -> dict[str, object]:
    """Return the deterministic public descriptor committed under ``schemas/``."""

    task_fields = {
        TaskName.CONTINUE_LOG.value: ["next_event_type"],
        TaskName.FAULT_FAMILY.value: [
            "diagnosis_status",
            "fault_labels",
            "abstention_reason",
        ],
        TaskName.EXTRACT_EVIDENCE.value: ["fact_refs", "evidence_slots"],
        TaskName.NEXT_ACTION.value: ["immediate_action"],
        TaskName.INCIDENT_SUMMARY.value: [
            "affected_subsystems",
            "observed_trend",
            "diagnosis_status",
            "fault_labels",
            "operating_mode",
            "immediate_action",
            "abstention_reason",
        ],
        TaskName.COUNTERFACTUAL_COMPARE.value: [
            "baseline_conclusion",
            "counterfactual_conclusion",
            "changed_fields",
            "baseline_decisive_fact_refs",
            "counterfactual_decisive_fact_refs",
            "decisive_evidence_slots",
        ],
    }
    return {
        "contract_version": COMPACT_TARGET_VERSION,
        "status": "developmental",
        "frozen": True,
        "wire_prefix": COMPACT_WIRE_PREFIX,
        "source_target_contract": "ProjectionTaskTargetValue@0.1.0",
        "max_utf8_bytes": MAX_COMPACT_TARGET_BYTES,
        "delimiters": {
            "field": FIELD_SEPARATOR,
            "list": LIST_SEPARATOR,
            "conclusion": CONCLUSION_SEPARATOR,
            "empty_or_absent": EMPTY_VALUE,
        },
        "enum_code_tables": {
            enum_type.__name__: {
                code: member.value
                for code, member in zip(_enum_codes(enum_type), enum_type, strict=True)
            }
            for enum_type in (
                DiagnosisStatus,
                FaultFamily,
                EvidenceSlot,
                ActionLabel,
                AbstentionReason,
                CounterfactualChange,
                EventType,
                AsterSubsystem,
                ObservedTrend,
                OperatingMode,
            )
        },
        "task_fields": task_fields,
    }


class CompactTargetConstraint:
    """Grammar and greedy-token constraint built without target or latent truth."""

    __slots__ = ("_allowed_cache", "context", "maximum_generated_tokens")

    def __init__(
        self,
        context: CompactTargetContext,
        *,
        maximum_generated_tokens: int,
    ) -> None:
        if type(context) is not CompactTargetContext:
            raise TypeError("context must be an exact CompactTargetContext")
        if (
            type(maximum_generated_tokens) is not int
            or not 1 <= maximum_generated_tokens <= MAX_CONSTRAINED_GENERATED_TOKENS
        ):
            raise ValueError("maximum generated-token bound is invalid")
        self.context = context
        self.maximum_generated_tokens = maximum_generated_tokens
        self._allowed_cache: dict[tuple[int, tuple[int, ...]], tuple[int, ...]] = {}

    def accepts_prefix(self, text: str) -> bool:
        """Return whether text is a prefix of at least one contract-shaped output."""

        if type(text) is not str:
            raise TypeError("compact prefix must be an exact string")
        if len(text.encode("utf-8")) > MAX_COMPACT_TARGET_BYTES:
            return False
        if _SAFE_WIRE.fullmatch(text) is None:
            return False
        header = FIELD_SEPARATOR.join((COMPACT_WIRE_PREFIX, self.context.task_name.value, ""))
        if len(text) <= len(header):
            return header.startswith(text)
        if not text.startswith(header):
            return False
        return _payload_prefix_is_viable(text[len(header) :], self.context)

    def accepts_complete(self, text: str) -> bool:
        """Return whether text compiles exactly; no repair or fallback is attempted."""

        if type(text) is not str:
            raise TypeError("compact target must be an exact string")
        try:
            parse_compact_target(text, context=self.context)
        except (CompactTargetError, TypeError, ValueError):
            return False
        return True

    def allowed_next_token_ids(
        self,
        tokenizer: ProjectTokenizer,
        generated_token_ids: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Return sorted legal next IDs, including EOS only for a complete target."""

        tokenizer_type, special_token_ids, eos_id = _tokenizer_runtime_contract()
        if type(tokenizer) is not tokenizer_type:
            raise TypeError("tokenizer must be an exact ProjectTokenizer")
        if type(generated_token_ids) is not tuple or any(
            type(token_id) is not int for token_id in generated_token_ids
        ):
            raise TypeError("generated token IDs must be an exact integer tuple")
        if any(
            token_id in special_token_ids or not 0 <= token_id < tokenizer.vocab_size
            for token_id in generated_token_ids
        ):
            raise ValueError("generated prefix contains an invalid or special token ID")
        cache_key = (id(tokenizer), generated_token_ids)
        cached = self._allowed_cache.get(cache_key)
        if cached is not None:
            return cached
        current_text = tokenizer.decode(generated_token_ids) if generated_token_ids else ""
        if not self.accepts_prefix(current_text):
            raise ValueError("generated token prefix violates the compact target contract")

        allowed: list[int] = []
        if self.accepts_complete(current_text):
            allowed.append(eos_id)
        if len(generated_token_ids) < self.maximum_generated_tokens:
            for token_id in range(tokenizer.vocab_size):
                if token_id in special_token_ids:
                    continue
                candidate_text = tokenizer.decode((*generated_token_ids, token_id))
                leading_boundary = not generated_token_ids and candidate_text == current_text == ""
                if (candidate_text != current_text or leading_boundary) and self.accepts_prefix(
                    candidate_text
                ):
                    allowed.append(token_id)
        result = tuple(sorted(allowed))
        if len(self._allowed_cache) >= MAX_CONSTRAINT_CACHE_ENTRIES:
            self._allowed_cache.clear()
        self._allowed_cache[cache_key] = result
        return result

    def select_next_token_id(
        self,
        logits: torch.Tensor,
        tokenizer: ProjectTokenizer,
        generated_token_ids: tuple[int, ...],
    ) -> int:
        """Choose the highest-logit legal token deterministically, without sampling."""

        import torch

        tokenizer_type, _special_token_ids, _eos_id = _tokenizer_runtime_contract()
        if type(tokenizer) is not tokenizer_type:
            raise TypeError("tokenizer must be an exact ProjectTokenizer")
        if type(logits) is not torch.Tensor or logits.ndim != 1:
            raise TypeError("constrained selection requires one exact rank-one tensor")
        if logits.numel() != tokenizer.vocab_size or not logits.is_floating_point():
            raise ValueError("decoder logits do not match the tokenizer vocabulary")
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError("decoder logits must all be finite")
        allowed = self.allowed_next_token_ids(tokenizer, generated_token_ids)
        if not allowed:
            raise CompactDecodingError("compact decoder reached a bounded dead end")
        allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=logits.device)
        selected_offset = int(torch.argmax(logits.index_select(0, allowed_tensor)).item())
        return allowed[selected_offset]


__all__ = [
    "COMPACT_TARGET_VERSION",
    "COMPACT_WIRE_PREFIX",
    "CompactDecodingError",
    "CompactTargetConstraint",
    "CompactTargetContext",
    "CompactTargetError",
    "compact_output_contract",
    "compact_target_json",
    "parse_compact_target",
    "serialize_compact_target",
]
