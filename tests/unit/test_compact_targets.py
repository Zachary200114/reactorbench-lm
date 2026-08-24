from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from pydantic import ValidationError

from reactorbench.dataset.contracts import (
    ProjectionTaskTargetValue,
    PromptContinuationTarget,
    PromptCounterfactualComparisonTarget,
    PromptEvidenceTarget,
)
from reactorbench.evaluation.compact import (
    COMPACT_TARGET_VERSION,
    MAX_COMPACT_TARGET_BYTES,
    CompactDecodingError,
    CompactTargetConstraint,
    CompactTargetContext,
    CompactTargetError,
    compact_target_json,
    parse_compact_target,
    serialize_compact_target,
)
from reactorbench.schemas.base import canonical_json_bytes
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
from reactorbench.tokenizer import EOS_ID, ProjectTokenizer

_VISIBLE = ("o-0000", "e-0000", "o-0001", "e-0001", "c-0000")
_COUNTERFACTUAL_VISIBLE = ("e-0000", "o-0000", "e-0001", "c-0000")


def _context(task_name: TaskName) -> CompactTargetContext:
    return CompactTargetContext(
        task_name=task_name,
        visible_fact_refs=_VISIBLE,
        counterfactual_visible_fact_refs=(
            _COUNTERFACTUAL_VISIBLE if task_name is TaskName.COUNTERFACTUAL_COMPARE else ()
        ),
    )


def _counterfactual_target() -> PromptCounterfactualComparisonTarget:
    baseline = CounterfactualConclusion(
        diagnosis_status=DiagnosisStatus.DIAGNOSED,
        fault_labels=(FaultFamily.SENSOR_DRIFT,),
        evidence_slots=(EvidenceSlot.CHANNEL_DISAGREEMENT,),
        immediate_action=ActionLabel.VERIFY_REDUNDANT_CHANNEL,
    )
    counterfactual = CounterfactualConclusion(
        diagnosis_status=DiagnosisStatus.NO_FAULT,
        evidence_slots=(EvidenceSlot.STABLE_OPERATION,),
        immediate_action=ActionLabel.CONTINUE_MONITORING,
    )
    return PromptCounterfactualComparisonTarget(
        baseline=baseline,
        counterfactual=counterfactual,
        changed_fields=tuple(CounterfactualChange),
        baseline_decisive_fact_refs=("e-0000",),
        counterfactual_decisive_fact_refs=("o-0000",),
        decisive_evidence_slots=(
            EvidenceSlot.CHANNEL_DISAGREEMENT,
            EvidenceSlot.STABLE_OPERATION,
        ),
    )


def _representative_targets() -> tuple[ProjectionTaskTargetValue, ...]:
    return (
        PromptContinuationTarget(next_event_type=EventType.OBSERVATION_CHANGED),
        FaultDiagnosisTarget(
            diagnosis_status=DiagnosisStatus.DIAGNOSED,
            fault_labels=(FaultFamily.SENSOR_DRIFT, FaultFamily.PUMP_TRIP),
        ),
        PromptEvidenceTarget(
            fact_refs=("e-0000", "o-0001", "e-0001"),
            evidence_slots=(
                EvidenceSlot.CHANNEL_DISAGREEMENT,
                EvidenceSlot.RELATED_STATE_STABLE,
            ),
        ),
        NextActionTarget(immediate_action=ActionLabel.COMPARE_RELATED_TRENDS),
        IncidentSummaryTarget(
            affected_subsystems=(
                AsterSubsystem.PRIMARY_LOOP,
                AsterSubsystem.INSTRUMENTATION,
            ),
            observed_trend=ObservedTrend.FALLING,
            diagnosis_status=DiagnosisStatus.DIAGNOSED,
            fault_labels=(FaultFamily.PUMP_TRIP,),
            operating_mode=OperatingMode.DISTURBED,
            immediate_action=ActionLabel.REQUEST_COMPONENT_INSPECTION,
        ),
        _counterfactual_target(),
    )


def test_representative_compact_targets_have_exact_canonical_round_trips() -> None:
    expected = (
        "RB2|continue_log|3",
        "RB2|fault_family|0|0,4|-",
        "RB2|extract_evidence|e-0000,o-0001,e-0001|2,3",
        "RB2|next_action|1",
        "RB2|incident_summary|1,7|2|0|4|2|3|-",
        ("RB2|counterfactual_compare|0~0~2~0~-|1~-~0~7~-|0,1,2,3|e-0000|o-0000|2,0"),
    )
    for target, compact in zip(_representative_targets(), expected, strict=True):
        context = _context(target.task_name)
        assert serialize_compact_target(target, context=context) == compact
        assert parse_compact_target(compact, context=context) == target
        assert (
            serialize_compact_target(
                parse_compact_target(compact, context=context), context=context
            )
            == compact
        )
        assert compact_target_json(compact, context=context) == canonical_json_bytes(
            target.model_dump(mode="json", round_trip=True)
        ).decode("utf-8")


def test_every_closed_enum_member_is_reachable_through_a_valid_target() -> None:
    for event in EventType:
        if event is EventType.ACTION_APPLIED:
            continue
        continuation = PromptContinuationTarget(next_event_type=event)
        context = _context(continuation.task_name)
        assert (
            parse_compact_target(
                serialize_compact_target(continuation, context=context), context=context
            )
            == continuation
        )

    for action in ActionLabel:
        action_target = NextActionTarget(immediate_action=action)
        context = _context(action_target.task_name)
        assert (
            parse_compact_target(
                serialize_compact_target(action_target, context=context), context=context
            )
            == action_target
        )

    for fault in FaultFamily:
        fault_target = FaultDiagnosisTarget(
            diagnosis_status=DiagnosisStatus.DIAGNOSED,
            fault_labels=(fault,),
        )
        context = _context(fault_target.task_name)
        assert (
            parse_compact_target(
                serialize_compact_target(fault_target, context=context), context=context
            )
            == fault_target
        )

    diagnosis_targets = (
        FaultDiagnosisTarget(
            diagnosis_status=DiagnosisStatus.DIAGNOSED,
            fault_labels=(FaultFamily.SENSOR_STUCK,),
        ),
        FaultDiagnosisTarget(diagnosis_status=DiagnosisStatus.NO_FAULT),
        FaultDiagnosisTarget(
            diagnosis_status=DiagnosisStatus.UNRESOLVED,
            abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
        ),
    )
    assert {target.diagnosis_status for target in diagnosis_targets} == set(DiagnosisStatus)
    for diagnosis_target in diagnosis_targets:
        context = _context(diagnosis_target.task_name)
        assert (
            parse_compact_target(
                serialize_compact_target(diagnosis_target, context=context), context=context
            )
            == diagnosis_target
        )

    for slot in EvidenceSlot:
        evidence_target = PromptEvidenceTarget(
            fact_refs=("o-0000",),
            evidence_slots=(slot,),
        )
        context = _context(evidence_target.task_name)
        assert (
            parse_compact_target(
                serialize_compact_target(evidence_target, context=context), context=context
            )
            == evidence_target
        )

    for subsystem in AsterSubsystem:
        for trend in ObservedTrend:
            for mode in OperatingMode:
                summary_target = IncidentSummaryTarget(
                    affected_subsystems=(subsystem,),
                    observed_trend=trend,
                    diagnosis_status=DiagnosisStatus.DIAGNOSED,
                    fault_labels=(FaultFamily.SENSOR_NOISE,),
                    operating_mode=mode,
                    immediate_action=ActionLabel.FLAG_SENSOR_SUSPECT,
                )
                context = _context(summary_target.task_name)
                assert (
                    parse_compact_target(
                        serialize_compact_target(summary_target, context=context), context=context
                    )
                    == summary_target
                )

    counterfactual = _counterfactual_target()
    assert set(counterfactual.changed_fields) == set(CounterfactualChange)
    context = _context(counterfactual.task_name)
    assert (
        parse_compact_target(
            serialize_compact_target(counterfactual, context=context), context=context
        )
        == counterfactual
    )


@pytest.mark.parametrize(
    ("text", "task_name", "error"),
    [
        ("RB2|next_action|X", TaskName.NEXT_ACTION, "invalid compact enum code"),
        ("RB2|fault_family|1|0|-", TaskName.FAULT_FAMILY, "strict"),
        (
            "RB2|fault_family|0|4,0|-",
            TaskName.FAULT_FAMILY,
            "canonical enum order",
        ),
        (
            "RB2|extract_evidence|e-0000,e-0000|2",
            TaskName.EXTRACT_EVIDENCE,
            "duplicates",
        ),
        (
            "RB2|extract_evidence|e-0001,e-0000|2",
            TaskName.EXTRACT_EVIDENCE,
            "increase",
        ),
        (
            "RB2|extract_evidence|e-9999|2",
            TaskName.EXTRACT_EVIDENCE,
            "prompt-visible",
        ),
        ("RB2|next_action|7|", TaskName.NEXT_ACTION, "exactly one"),
        ("RB2|next_action|XX", TaskName.NEXT_ACTION, "invalid compact enum code"),
        ("RB2|next_action|CONTINUE MONITORING", TaskName.NEXT_ACTION, "allowlist"),
        ("RB9|next_action|7", TaskName.NEXT_ACTION, "wire prefix"),
        ("RB2|unknown|7", TaskName.NEXT_ACTION, "task name"),
        (
            "RB2|counterfactual_compare|X~0~2~0~-|1~-~0~7~-|0,1,2,3|e-0000|o-0000|2,0",
            TaskName.COUNTERFACTUAL_COMPARE,
            "invalid compact enum code",
        ),
    ],
)
def test_parser_rejects_unknown_duplicate_misordered_and_trailing_text(
    text: str,
    task_name: TaskName,
    error: str,
) -> None:
    with pytest.raises(CompactTargetError, match=error):
        parse_compact_target(text, context=_context(task_name))


def test_parser_and_serializer_reject_wrong_types_tasks_and_maximum_length() -> None:
    context = _context(TaskName.NEXT_ACTION)
    with pytest.raises(TypeError, match="exact string"):
        parse_compact_target(cast(Any, b"RB2"), context=context)
    with pytest.raises(TypeError, match="exact CompactTargetContext"):
        parse_compact_target("RB2", context=cast(Any, SimpleNamespace()))
    with pytest.raises(TypeError, match="exact projection"):
        serialize_compact_target(cast(Any, SimpleNamespace()), context=context)
    with pytest.raises(CompactTargetError, match="does not match"):
        serialize_compact_target(
            PromptEvidenceTarget(),
            context=context,
        )
    with pytest.raises(CompactTargetError, match="does not match"):
        parse_compact_target(
            "RB2|fault_family|1|-|-",
            context=context,
        )
    oversized = "A" * (MAX_COMPACT_TARGET_BYTES + 1)
    with pytest.raises(CompactTargetError, match="exceeds"):
        parse_compact_target(oversized, context=context)
    with pytest.raises(CompactTargetError, match="prompt-visible"):
        serialize_compact_target(
            PromptEvidenceTarget(fact_refs=("c-0001",)),
            context=_context(TaskName.EXTRACT_EVIDENCE),
        )


def test_context_is_strict_bounded_and_contains_no_truth_fields() -> None:
    context = _context(TaskName.EXTRACT_EVIDENCE)
    assert context.contract_version == COMPACT_TARGET_VERSION
    assert tuple(context.__class__.model_fields) == (
        "contract_version",
        "task_name",
        "visible_fact_refs",
        "counterfactual_visible_fact_refs",
    )
    with pytest.raises(ValidationError, match="Extra inputs"):
        CompactTargetContext.model_validate(
            {
                **context.model_dump(mode="python"),
                "fault_labels": ["SENSOR_DRIFT"],
            }
        )
    with pytest.raises(ValidationError, match="canonical and contiguous"):
        CompactTargetContext(
            task_name=TaskName.EXTRACT_EVIDENCE,
            visible_fact_refs=("o-0001", "o-0000"),
        )
    with pytest.raises(ValidationError, match="second visible"):
        CompactTargetContext(
            task_name=TaskName.NEXT_ACTION,
            visible_fact_refs=("o-0000",),
            counterfactual_visible_fact_refs=("o-0000",),
        )
    with pytest.raises(ValidationError, match="second visible"):
        CompactTargetContext(
            task_name=TaskName.COUNTERFACTUAL_COMPARE,
            visible_fact_refs=("o-0000",),
        )


def _character_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
) -> tuple[ProjectTokenizer, dict[str, int], int]:
    characters = tuple(sorted({*text, "X"}))
    by_character = {character: index + 4 for index, character in enumerate(characters)}
    by_token = {token_id: character for character, token_id in by_character.items()}
    vocabulary_size = len(characters) + 4
    tokenizer = object.__new__(ProjectTokenizer)
    monkeypatch.setattr(
        ProjectTokenizer,
        "vocab_size",
        property(lambda _self: vocabulary_size),
    )
    monkeypatch.setattr(
        ProjectTokenizer,
        "decode",
        lambda _self, token_ids: "".join(by_token[token_id] for token_id in token_ids),
    )
    return tokenizer, by_character, vocabulary_size


def test_constraint_reaches_a_canonical_target_and_delays_eos_until_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _counterfactual_target()
    context = _context(target.task_name)
    text = serialize_compact_target(target, context=context)
    tokenizer, by_character, _vocabulary_size = _character_tokenizer(monkeypatch, text)
    constraint = CompactTargetConstraint(
        context,
        maximum_generated_tokens=len(text),
    )
    generated: tuple[int, ...] = ()
    assert constraint.accepts_prefix("")
    for character in text:
        allowed = constraint.allowed_next_token_ids(tokenizer, generated)
        assert by_character[character] in allowed
        if not constraint.accepts_complete(tokenizer.decode(generated) if generated else ""):
            assert EOS_ID not in allowed
        generated = (*generated, by_character[character])
    assert tokenizer.decode(generated) == text
    allowed = constraint.allowed_next_token_ids(tokenizer, generated)
    assert EOS_ID in allowed
    assert constraint.allowed_next_token_ids(tokenizer, generated) is allowed
    assert constraint.accepts_complete(text)
    assert not constraint.accepts_prefix(f"{text}X")


def test_constraint_selection_is_deterministic_bounded_and_rejects_bad_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = NextActionTarget(immediate_action=ActionLabel.CONTINUE_MONITORING)
    context = _context(target.task_name)
    text = serialize_compact_target(target, context=context)
    tokenizer, by_character, vocabulary_size = _character_tokenizer(monkeypatch, text)
    constraint = CompactTargetConstraint(context, maximum_generated_tokens=len(text))
    generated: tuple[int, ...] = ()
    for character in text:
        logits = torch.zeros(vocabulary_size)
        logits[by_character["X"]] = 100.0
        logits[by_character[character]] = 1.0
        selected = constraint.select_next_token_id(logits, tokenizer, generated)
        assert selected == by_character[character]
        generated = (*generated, selected)
    final_logits = torch.zeros(vocabulary_size)
    final_logits[EOS_ID] = 1.0
    assert constraint.select_next_token_id(final_logits, tokenizer, generated) == EOS_ID

    with pytest.raises(ValueError, match="generated-token bound"):
        CompactTargetConstraint(context, maximum_generated_tokens=0)
    bounded = CompactTargetConstraint(context, maximum_generated_tokens=1)
    one_token = (by_character[text[0]],)
    assert bounded.allowed_next_token_ids(tokenizer, one_token) == ()
    with pytest.raises(CompactDecodingError, match="bounded dead end"):
        bounded.select_next_token_id(torch.zeros(vocabulary_size), tokenizer, one_token)
    with pytest.raises(ValueError, match="special token"):
        constraint.allowed_next_token_ids(tokenizer, (EOS_ID,))
    with pytest.raises(TypeError, match="integer tuple"):
        constraint.allowed_next_token_ids(tokenizer, cast(Any, [4]))
    with pytest.raises(ValueError, match="violates"):
        constraint.allowed_next_token_ids(tokenizer, (by_character["X"],))
    with pytest.raises(TypeError, match="rank-one"):
        constraint.select_next_token_id(cast(Any, [0.0] * vocabulary_size), tokenizer, ())
    with pytest.raises(ValueError, match="vocabulary"):
        constraint.select_next_token_id(torch.zeros(vocabulary_size - 1), tokenizer, ())
    nonfinite = torch.zeros(vocabulary_size)
    nonfinite[4] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        constraint.select_next_token_id(nonfinite, tokenizer, ())


def test_constraint_allows_one_tokenizer_boundary_marker_without_zero_width_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(TaskName.NEXT_ACTION)
    text = "RB2|next_action|7"
    tokenizer = object.__new__(ProjectTokenizer)
    monkeypatch.setattr(ProjectTokenizer, "vocab_size", property(lambda _self: 6))
    monkeypatch.setattr(
        ProjectTokenizer,
        "decode",
        lambda _self, token_ids: "".join("" if token_id == 4 else text for token_id in token_ids),
    )
    constraint = CompactTargetConstraint(context, maximum_generated_tokens=3)
    assert 4 in constraint.allowed_next_token_ids(tokenizer, ())
    assert 4 not in constraint.allowed_next_token_ids(tokenizer, (4,))
    assert 5 in constraint.allowed_next_token_ids(tokenizer, (4,))
    assert constraint.accepts_complete(tokenizer.decode((4, 5)))
    assert EOS_ID in constraint.allowed_next_token_ids(tokenizer, (4, 5))


def test_constraint_rejects_premature_completion_wrong_task_and_trailing_content() -> None:
    context = _context(TaskName.FAULT_FAMILY)
    constraint = CompactTargetConstraint(context, maximum_generated_tokens=128)
    canonical = "RB2|fault_family|1|-|-"
    for prefix in ("", "RB2", "RB2|fault_family|", canonical[:-1]):
        assert constraint.accepts_prefix(prefix)
        assert not constraint.accepts_complete(prefix)
    assert constraint.accepts_complete(canonical)
    assert not constraint.accepts_prefix("RB2|next_action|")
    assert not constraint.accepts_complete(f"{canonical}|extra")
    assert not constraint.accepts_prefix(f"{canonical}\n")
