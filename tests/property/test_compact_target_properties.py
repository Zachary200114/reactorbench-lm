from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from reactorbench.dataset.contracts import (
    ProjectionTaskTargetValue,
    PromptContinuationTarget,
    PromptCounterfactualComparisonTarget,
    PromptEvidenceTarget,
)
from reactorbench.evaluation.compact import (
    CompactTargetConstraint,
    CompactTargetContext,
    CompactTargetError,
    parse_compact_target,
    serialize_compact_target,
)
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

_VISIBLE = ("o-0000", "e-0000", "o-0001", "e-0001", "c-0000")
_SECONDARY_VISIBLE = ("e-0000", "o-0000", "e-0001", "o-0001", "c-0000")
_RESOLVED_ACTIONS = tuple(
    action for action in ActionLabel if action is not ActionLabel.INSUFFICIENT_EVIDENCE
)


def _context(task_name: TaskName) -> CompactTargetContext:
    return CompactTargetContext(
        task_name=task_name,
        visible_fact_refs=_VISIBLE,
        counterfactual_visible_fact_refs=(
            _SECONDARY_VISIBLE if task_name is TaskName.COUNTERFACTUAL_COMPARE else ()
        ),
    )


def _canonical_members[T](members: list[T], enum_members: tuple[T, ...]) -> tuple[T, ...]:
    order = {member: index for index, member in enumerate(enum_members)}
    return tuple(sorted(members, key=order.__getitem__))


@st.composite
def _locally_ordered_refs(
    draw: st.DrawFn,
    *,
    visible: tuple[str, ...],
    minimum: int = 0,
) -> tuple[str, ...]:
    by_namespace = {
        namespace: tuple(ref for ref in visible if ref.startswith(f"{namespace}-"))
        for namespace in ("o", "e", "c")
    }
    namespace_order = draw(st.permutations(("o", "e", "c")))
    selected: list[str] = []
    for namespace in namespace_order:
        members = draw(
            st.lists(
                st.sampled_from(by_namespace[namespace]),
                unique=True,
                max_size=len(by_namespace[namespace]),
            )
        )
        selected.extend(sorted(members))
    if len(selected) < minimum:
        return (visible[0],)
    return tuple(selected)


@st.composite
def _fault_target(draw: st.DrawFn) -> FaultDiagnosisTarget:
    status = draw(st.sampled_from(tuple(DiagnosisStatus)))
    if status is DiagnosisStatus.DIAGNOSED:
        faults = _canonical_members(
            draw(
                st.lists(
                    st.sampled_from(tuple(FaultFamily)),
                    min_size=1,
                    max_size=5,
                    unique=True,
                )
            ),
            tuple(FaultFamily),
        )
        return FaultDiagnosisTarget(diagnosis_status=status, fault_labels=faults)
    if status is DiagnosisStatus.NO_FAULT:
        return FaultDiagnosisTarget(diagnosis_status=status)
    return FaultDiagnosisTarget(
        diagnosis_status=status,
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
    )


@st.composite
def _summary_target(draw: st.DrawFn) -> IncidentSummaryTarget:
    status = draw(st.sampled_from(tuple(DiagnosisStatus)))
    affected = _canonical_members(
        draw(
            st.lists(
                st.sampled_from(tuple(AsterSubsystem)),
                min_size=1 if status is DiagnosisStatus.DIAGNOSED else 0,
                max_size=4,
                unique=True,
            )
        ),
        tuple(AsterSubsystem),
    )
    faults: tuple[FaultFamily, ...] = ()
    action: ActionLabel
    abstention: AbstentionReason | None = None
    if status is DiagnosisStatus.DIAGNOSED:
        faults = _canonical_members(
            draw(
                st.lists(
                    st.sampled_from(tuple(FaultFamily)),
                    min_size=1,
                    max_size=4,
                    unique=True,
                )
            ),
            tuple(FaultFamily),
        )
        action = draw(st.sampled_from(_RESOLVED_ACTIONS))
    elif status is DiagnosisStatus.NO_FAULT:
        affected = ()
        action = draw(st.sampled_from(_RESOLVED_ACTIONS))
    else:
        action = ActionLabel.INSUFFICIENT_EVIDENCE
        abstention = AbstentionReason.INSUFFICIENT_EVIDENCE
    return IncidentSummaryTarget(
        affected_subsystems=affected,
        observed_trend=draw(st.sampled_from(tuple(ObservedTrend))),
        diagnosis_status=status,
        fault_labels=faults,
        operating_mode=draw(st.sampled_from(tuple(OperatingMode))),
        immediate_action=action,
        abstention_reason=abstention,
    )


def _actual_changes(
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


@st.composite
def _counterfactual_target(draw: st.DrawFn) -> PromptCounterfactualComparisonTarget:
    baseline_slots = tuple(
        draw(
            st.lists(
                st.sampled_from(tuple(EvidenceSlot)),
                max_size=4,
                unique=True,
            )
        )
    )
    counterfactual_slots = tuple(
        draw(
            st.lists(
                st.sampled_from(tuple(EvidenceSlot)),
                max_size=4,
                unique=True,
            )
        )
    )
    baseline = CounterfactualConclusion(
        diagnosis_status=DiagnosisStatus.DIAGNOSED,
        fault_labels=(draw(st.sampled_from(tuple(FaultFamily))),),
        evidence_slots=baseline_slots,
        immediate_action=draw(st.sampled_from(_RESOLVED_ACTIONS)),
    )
    counter_status = draw(st.sampled_from((DiagnosisStatus.NO_FAULT, DiagnosisStatus.UNRESOLVED)))
    if counter_status is DiagnosisStatus.NO_FAULT:
        counterfactual = CounterfactualConclusion(
            diagnosis_status=counter_status,
            evidence_slots=counterfactual_slots,
            immediate_action=draw(st.sampled_from(_RESOLVED_ACTIONS)),
        )
    else:
        counterfactual = CounterfactualConclusion(
            diagnosis_status=counter_status,
            evidence_slots=counterfactual_slots,
            immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE,
            abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
        )
    baseline_refs = draw(_locally_ordered_refs(visible=_VISIBLE, minimum=1))
    counterfactual_refs = draw(_locally_ordered_refs(visible=_SECONDARY_VISIBLE))
    decisive_slots = tuple(
        draw(
            st.lists(
                st.sampled_from(tuple(EvidenceSlot)),
                max_size=4,
                unique=True,
            )
        )
    )
    return PromptCounterfactualComparisonTarget(
        baseline=baseline,
        counterfactual=counterfactual,
        changed_fields=_actual_changes(baseline, counterfactual),
        baseline_decisive_fact_refs=baseline_refs,
        counterfactual_decisive_fact_refs=counterfactual_refs,
        decisive_evidence_slots=decisive_slots,
    )


@st.composite
def _target_case(
    draw: st.DrawFn,
) -> tuple[ProjectionTaskTargetValue, CompactTargetContext]:
    task_name = draw(st.sampled_from(tuple(TaskName)))
    target: ProjectionTaskTargetValue
    if task_name is TaskName.CONTINUE_LOG:
        target = PromptContinuationTarget(
            next_event_type=draw(
                st.sampled_from(
                    tuple(event for event in EventType if event is not EventType.ACTION_APPLIED)
                )
            )
        )
    elif task_name is TaskName.FAULT_FAMILY:
        target = draw(_fault_target())
    elif task_name is TaskName.EXTRACT_EVIDENCE:
        target = PromptEvidenceTarget(
            fact_refs=draw(_locally_ordered_refs(visible=_VISIBLE)),
            evidence_slots=tuple(
                draw(
                    st.lists(
                        st.sampled_from(tuple(EvidenceSlot)),
                        max_size=6,
                        unique=True,
                    )
                )
            ),
        )
    elif task_name is TaskName.NEXT_ACTION:
        target = NextActionTarget(immediate_action=draw(st.sampled_from(tuple(ActionLabel))))
    elif task_name is TaskName.INCIDENT_SUMMARY:
        target = draw(_summary_target())
    else:
        target = draw(_counterfactual_target())
    return target, _context(task_name)


@settings(max_examples=120, deadline=None)
@given(case=_target_case())
def test_compact_targets_round_trip_and_every_character_prefix_is_reachable(
    case: tuple[ProjectionTaskTargetValue, CompactTargetContext],
) -> None:
    target, context = case
    compact = serialize_compact_target(target, context=context)
    constraint = CompactTargetConstraint(
        context,
        maximum_generated_tokens=4096,
    )
    assert parse_compact_target(compact, context=context) == target
    assert constraint.accepts_complete(compact)
    assert all(constraint.accepts_prefix(compact[:index]) for index in range(len(compact) + 1))
    assert not constraint.accepts_prefix(f"{compact}X")
    assert not constraint.accepts_complete(f"{compact}X")


@settings(max_examples=60, deadline=None)
@given(
    task_name=st.sampled_from(tuple(TaskName)),
    text=st.text(
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_|,~-",
        max_size=96,
    ),
)
def test_every_string_accepted_as_complete_compiles_without_repair(
    task_name: TaskName,
    text: str,
) -> None:
    context = _context(task_name)
    constraint = CompactTargetConstraint(context, maximum_generated_tokens=256)
    if constraint.accepts_complete(text):
        assert (
            serialize_compact_target(parse_compact_target(text, context=context), context=context)
            == text
        )


@settings(max_examples=40, deadline=None)
@given(case=_target_case())
def test_compiler_and_constraint_do_not_change_torch_global_rng(
    case: tuple[ProjectionTaskTargetValue, CompactTargetContext],
) -> None:
    target, context = case
    before = torch.random.get_rng_state().clone()
    compact = serialize_compact_target(target, context=context)
    constraint = CompactTargetConstraint(context, maximum_generated_tokens=4096)
    assert constraint.accepts_prefix(compact)
    assert constraint.accepts_complete(compact)
    assert parse_compact_target(compact, context=context) == target
    assert torch.equal(torch.random.get_rng_state(), before)


@given(first=st.integers(min_value=1, max_value=9), second=st.integers(min_value=0, max_value=0))
def test_misordered_prompt_local_references_always_fail_closed(first: int, second: int) -> None:
    context = CompactTargetContext(
        task_name=TaskName.EXTRACT_EVIDENCE,
        visible_fact_refs=tuple(f"e-{index:04d}" for index in range(first + 1)),
    )
    text = f"RB2|extract_evidence|e-{first:04d},e-{second:04d}|2"
    try:
        parse_compact_target(text, context=context)
    except CompactTargetError:
        pass
    else:
        raise AssertionError("misordered prompt-local references were accepted")
