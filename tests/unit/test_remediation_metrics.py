"""Focused behavioral tests for dual-path compact remediation evaluation."""

from __future__ import annotations

import hashlib
from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError

from reactorbench.dataset.contracts import (
    ProjectionTaskTargetValue,
    PromptContinuationTarget,
    PromptEvidenceTarget,
)
from reactorbench.evaluation.baselines import _result
from reactorbench.evaluation.compact import (
    CompactTargetContext,
    compact_target_json,
    serialize_compact_target,
)
from reactorbench.evaluation.metrics import classification_metrics
from reactorbench.remediation.acceptance import DevelopmentArtifactBinding
from reactorbench.remediation.baselines import RemediationBaselineReport
from reactorbench.remediation.config import RemediationView
from reactorbench.remediation.data import RemediationExample
from reactorbench.remediation.decoding import (
    CompactPathPrediction,
    DecodePath,
    DualPathCompactPrediction,
)
from reactorbench.remediation.metrics import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    SemanticEvaluationReport,
    canonical_prediction_jsonl_bytes,
    compact_output_contract_byte_sha256,
    evaluate_semantic_predictions,
    prediction_artifact_byte_sha256,
    semantic_composite_score,
)
from reactorbench.schemas.base import canonical_sha256
from reactorbench.schemas.enums import (
    AbstentionReason,
    ActionLabel,
    DiagnosisStatus,
    EventType,
    EvidenceSlot,
    FaultFamily,
    ObservedTrend,
    OperatingMode,
    SplitName,
    TaskName,
)
from reactorbench.schemas.target import (
    FaultDiagnosisTarget,
    IncidentSummaryTarget,
    NextActionTarget,
)


def _truth_targets() -> tuple[ProjectionTaskTargetValue, ...]:
    return (
        FaultDiagnosisTarget(diagnosis_status=DiagnosisStatus.NO_FAULT),
        FaultDiagnosisTarget(
            diagnosis_status=DiagnosisStatus.DIAGNOSED,
            fault_labels=(FaultFamily.SENSOR_DRIFT,),
        ),
        NextActionTarget(immediate_action=ActionLabel.CONTINUE_MONITORING),
        NextActionTarget(immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE),
        PromptContinuationTarget(next_event_type=EventType.BENIGN_NOTE),
        PromptEvidenceTarget(
            fact_refs=("o-0000",),
            evidence_slots=(EvidenceSlot.STABLE_OPERATION,),
        ),
        IncidentSummaryTarget(
            affected_subsystems=(),
            observed_trend=ObservedTrend.UNKNOWN,
            diagnosis_status=DiagnosisStatus.UNRESOLVED,
            fault_labels=(),
            operating_mode=OperatingMode.UNKNOWN,
            immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE,
            abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
        ),
    )


def _classification_label(target: ProjectionTaskTargetValue) -> str | None:
    if type(target) is FaultDiagnosisTarget:
        if target.diagnosis_status is DiagnosisStatus.DIAGNOSED:
            return "DIAGNOSED:" + "+".join(item.value for item in target.fault_labels)
        return target.diagnosis_status.value
    if type(target) is NextActionTarget:
        return target.immediate_action.value
    if type(target) is PromptContinuationTarget:
        return target.next_event_type.value
    return None


def _example(index: int, target: ProjectionTaskTargetValue) -> RemediationExample:
    context = CompactTargetContext(
        task_name=target.task_name,
        visible_fact_refs=("o-0000",),
    )
    compact = serialize_compact_target(target, context=context)
    prompt = f"[o-0000] fictional metric prompt {index}"
    values: dict[str, Any] = {
        "artifact_version": "0.3.0",
        "example_id": f"metric:{index:04d}",
        "view": RemediationView.IID_VALIDATION,
        "source_split": SplitName.IID_VALIDATION,
        "task_name": target.task_name,
        "group_id": f"metric-group:{index:04d}",
        "source_record_ids": (f"projection:{index:04d}",),
        "parent_record_sha256": f"{index + 1:064x}",
        "prompt_text": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "template_family_id": "compact-log-v1",
        "alias_family_id": "canonical-v1",
        "compact_context": context,
        "compact_target": compact,
        "canonical_target_json": compact_target_json(compact, context=context),
        "classification_label": _classification_label(target),
        "augmentation": "none",
    }
    draft = RemediationExample.model_construct(**values, checksum_sha256="0" * 64)
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return RemediationExample(**values, checksum_sha256=checksum)


def _examples() -> tuple[RemediationExample, ...]:
    return tuple(_example(index, target) for index, target in enumerate(_truth_targets()))


def _path(
    *,
    path: DecodePath,
    task: TaskName,
    context: CompactTargetContext,
    target: ProjectionTaskTargetValue | None,
    confidence: float,
    malformed_text: str = "not-a-compact-target",
    lexical_parse: bool = False,
    exhausted: bool = False,
) -> CompactPathPrediction:
    if target is None:
        text = malformed_text
        canonical = None
        schema_valid = False
    else:
        text = serialize_compact_target(target, context=context)
        canonical = compact_target_json(text, context=context)
        schema_valid = True
    generation_cap = 16
    token_ids = (4,) * generation_cap if exhausted else (4,)
    values: dict[str, Any] = {
        "result_version": "0.3.0",
        "path": path,
        "task_name": task,
        "generation_cap": generation_cap,
        "prompt_token_count": 10,
        "prompt_tokens_retained": 10,
        "prompt_truncated": False,
        "generated_token_ids": token_ids,
        "generated_token_count": len(token_ids),
        "selected_token_count": len(token_ids) if exhausted else len(token_ids) + 1,
        "generated_text": text,
        "eos_emitted": not exhausted,
        "generation_cap_exhausted": exhausted,
        "compact_parse_success": schema_valid or lexical_parse,
        "schema_valid": schema_valid,
        "canonical_target_json": canonical,
        "selected_token_geometric_mean_probability": confidence,
        "elapsed_seconds": 0.01,
        "used_cache": True,
    }
    draft = CompactPathPrediction.model_construct(**values, checksum_sha256="0" * 64)
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return CompactPathPrediction(**values, checksum_sha256=checksum)


def _dual(
    example: RemediationExample,
    constrained_target: ProjectionTaskTargetValue | None,
    *,
    confidence: float,
    malformed_text: str = "not-a-compact-target",
    lexical_parse: bool = False,
    exhausted: bool = False,
) -> DualPathCompactPrediction:
    unconstrained = _path(
        path=DecodePath.UNCONSTRAINED,
        task=example.task_name,
        context=example.compact_context,
        target=None,
        confidence=0.0,
    )
    constrained = _path(
        path=DecodePath.CONSTRAINED,
        task=example.task_name,
        context=example.compact_context,
        target=constrained_target,
        confidence=confidence,
        malformed_text=malformed_text,
        lexical_parse=lexical_parse,
        exhausted=exhausted,
    )
    values: dict[str, Any] = {
        "result_version": "0.3.0",
        "example_id": example.example_id,
        "example_checksum_sha256": example.checksum_sha256,
        "task_name": example.task_name,
        "model_config_sha256": "a" * 64,
        "tokenizer_manifest_sha256": "e" * 64,
        "generation_caps_sha256": "c" * 64,
        "unconstrained": unconstrained,
        "constrained": constrained,
    }
    draft = DualPathCompactPrediction.model_construct(**values, checksum_sha256="0" * 64)
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return DualPathCompactPrediction(**values, checksum_sha256=checksum)


def _baseline_report(
    view: RemediationView = RemediationView.IID_VALIDATION,
    *,
    fault_support: int = 2,
) -> RemediationBaselineReport:
    fault_truth = tuple(("NO_FAULT", "DIAGNOSED:SENSOR_DRIFT")[:fault_support])
    fault_metrics = classification_metrics(TaskName.FAULT_FAMILY, fault_truth, fault_truth)
    action_truth = ("CONTINUE_MONITORING", "INSUFFICIENT_EVIDENCE")
    action_metrics = classification_metrics(TaskName.NEXT_ACTION, action_truth, action_truth)
    results = (
        _result(
            name="fault-test-comparator",
            task_name=TaskName.FAULT_FAMILY.value,
            parameter_count=0,
            elapsed_seconds=0.0,
            classification=fault_metrics,
        ),
        _result(
            name="action-test-comparator",
            task_name=TaskName.NEXT_ACTION.value,
            parameter_count=0,
            elapsed_seconds=0.0,
            classification=action_metrics,
        ),
    )
    values: dict[str, Any] = {
        "report_version": "0.3.0",
        "dataset_manifest_sha256": "d" * 64,
        "evaluation_view": view,
        "tokenizer_manifest_sha256": "e" * 64,
        "baseline_config_sha256": "f" * 64,
        "result_count": len(results),
        "results": results,
        "strongest_fault_comparator_macro_f1": 1.0,
        "strongest_action_comparator_macro_f1": 1.0,
    }
    draft = RemediationBaselineReport.model_construct(**values, checksum_sha256="0" * 64)
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return RemediationBaselineReport(**values, checksum_sha256=checksum)


def _artifacts(
    predictions: tuple[DualPathCompactPrediction, ...],
    baseline: RemediationBaselineReport,
    **updates: str,
) -> DevelopmentArtifactBinding:
    values = {
        "source_commit": "0123456789abcdef",
        "config_sha256": "1" * 64,
        "dataset_manifest_sha256": baseline.dataset_manifest_sha256,
        "tokenizer_manifest_sha256": baseline.tokenizer_manifest_sha256,
        "output_contract_sha256": compact_output_contract_byte_sha256(),
        "checkpoint_sha256": "4" * 64,
        "prediction_artifact_sha256": prediction_artifact_byte_sha256(predictions),
        "comparator_artifact_sha256": baseline.checksum_sha256,
    }
    values.update(updates)
    return DevelopmentArtifactBinding(**values)


def _perfect_predictions(
    examples: tuple[RemediationExample, ...],
) -> tuple[DualPathCompactPrediction, ...]:
    return tuple(
        _dual(example, target, confidence=1.0)
        for example, target in zip(examples, _truth_targets(), strict=True)
    )


def _prediction_with_tokenizer(
    prediction: DualPathCompactPrediction,
    tokenizer_manifest_sha256: str,
) -> DualPathCompactPrediction:
    draft = prediction.model_copy(
        update={
            "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
            "checksum_sha256": "0" * 64,
        }
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    payload = draft.model_dump(mode="python", round_trip=True)
    payload["checksum_sha256"] = checksum
    return DualPathCompactPrediction.model_validate(payload)


def _wrong_target(target: ProjectionTaskTargetValue) -> ProjectionTaskTargetValue:
    if type(target) is FaultDiagnosisTarget:
        if target.diagnosis_status is DiagnosisStatus.NO_FAULT:
            return FaultDiagnosisTarget(
                diagnosis_status=DiagnosisStatus.DIAGNOSED,
                fault_labels=(FaultFamily.SENSOR_DRIFT,),
            )
        return FaultDiagnosisTarget(diagnosis_status=DiagnosisStatus.NO_FAULT)
    if type(target) is NextActionTarget:
        alternate = (
            ActionLabel.INSUFFICIENT_EVIDENCE
            if target.immediate_action is ActionLabel.CONTINUE_MONITORING
            else ActionLabel.CONTINUE_MONITORING
        )
        return NextActionTarget(immediate_action=alternate)
    if type(target) is PromptContinuationTarget:
        return PromptContinuationTarget(next_event_type=EventType.CHANNEL_QUALITY_CHANGED)
    if type(target) is PromptEvidenceTarget:
        return PromptEvidenceTarget(fact_refs=(), evidence_slots=())
    if type(target) is IncidentSummaryTarget:
        return IncidentSummaryTarget(
            affected_subsystems=(),
            observed_trend=ObservedTrend.STABLE,
            diagnosis_status=DiagnosisStatus.NO_FAULT,
            fault_labels=(),
            operating_mode=OperatingMode.STABLE,
            immediate_action=ActionLabel.CONTINUE_MONITORING,
        )
    raise AssertionError(f"unexpected target type: {type(target)}")


def _evaluate(
    examples: tuple[RemediationExample, ...],
    predictions: tuple[DualPathCompactPrediction, ...],
    *,
    baseline: RemediationBaselineReport | None = None,
) -> SemanticEvaluationReport:
    selected_baseline = baseline or _baseline_report()
    return evaluate_semantic_predictions(
        view=RemediationView.IID_VALIDATION,
        examples=examples,
        predictions=predictions,
        baseline_report=selected_baseline,
        artifacts=_artifacts(predictions, selected_baseline),
    )


def test_perfect_dual_predictions_score_every_supported_metric_and_bootstrap() -> None:
    examples = _examples()
    predictions = _perfect_predictions(examples)
    state_before = np.random.get_state()

    first = _evaluate(tuple(reversed(examples)), tuple(reversed(predictions)))
    second = _evaluate(examples, predictions)
    state_after = np.random.get_state()

    assert first == second
    assert first.checksum_sha256 == second.checksum_sha256
    assert first.constrained.parse_rate == 1.0
    assert first.constrained.schema_validity_rate == 1.0
    assert first.constrained.exact_match_rate == 1.0
    assert first.unconstrained.parse_rate == 0.0
    assert first.unconstrained.schema_validity_rate == 0.0
    assert tuple(item.task_name for item in first.task_metrics) == (
        TaskName.CONTINUE_LOG,
        TaskName.FAULT_FAMILY,
        TaskName.NEXT_ACTION,
    )
    assert all(item.exact_match_rate == 1.0 and item.macro_f1 == 1.0 for item in first.task_metrics)

    metrics = first.view_metrics.metrics
    assert metrics.fault_family_macro_f1.support == 2
    assert metrics.next_action_macro_f1.support == 2
    assert metrics.continuation_macro_f1.support == 1
    assert metrics.evidence_f1.support == 1
    assert metrics.required_abstention_accuracy.support == 2
    assert metrics.no_fault_false_positive_rate.support == 1
    for estimate in (
        metrics.fault_family_macro_f1,
        metrics.next_action_macro_f1,
        metrics.continuation_macro_f1,
        metrics.evidence_f1,
        metrics.required_abstention_accuracy,
    ):
        assert estimate.estimate == estimate.interval_lower == estimate.interval_upper == 1.0
        assert estimate.bootstrap_resamples == BOOTSTRAP_RESAMPLES
        assert estimate.bootstrap_seed == BOOTSTRAP_SEED
    assert metrics.no_fault_false_positive_rate.estimate == 0.0
    assert metrics.expected_calibration_error.estimate == 0.0
    assert metrics.selective_risk_at_80_percent_coverage.estimate == 0.0
    assert {item.category: item.count for item in first.failures} == {
        "unconstrained_invalid_schema": len(examples)
    }
    assert semantic_composite_score(examples, predictions) == 1.0
    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]

    payload = first.model_dump(mode="python", round_trip=True)
    constrained = cast(dict[str, Any], payload["constrained"])
    constrained["exact_match_rate"] = 0.0
    with pytest.raises(ValidationError, match="checksum"):
        SemanticEvaluationReport.model_validate(payload)


def test_wrong_valid_predictions_expose_errors_abstention_and_calibration() -> None:
    examples = _examples()
    predictions = tuple(
        _dual(example, _wrong_target(target), confidence=0.8)
        for example, target in zip(examples, _truth_targets(), strict=True)
    )

    report = _evaluate(examples, predictions)
    metrics = report.view_metrics.metrics

    assert report.constrained.parse_rate == 1.0
    assert report.constrained.schema_validity_rate == 1.0
    assert report.constrained.exact_match_rate == 0.0
    assert metrics.fault_family_macro_f1.estimate == 0.0
    assert metrics.next_action_macro_f1.estimate == 0.0
    assert metrics.continuation_macro_f1.estimate == 0.0
    assert metrics.evidence_f1.estimate == 0.0
    assert metrics.required_abstention_accuracy.estimate == 0.0
    assert metrics.no_fault_false_positive_rate.estimate == 1.0
    assert metrics.expected_calibration_error.estimate == pytest.approx(0.8)
    assert metrics.selective_risk_at_80_percent_coverage.estimate == 1.0
    assert {item.category: item.count for item in report.failures} == {
        "evidence_mismatch": 1,
        "required_abstention_missed": 2,
        "semantic_mismatch": len(examples),
        "unconstrained_invalid_schema": len(examples),
    }
    assert 0.0 < semantic_composite_score(examples, predictions) < 0.1


def test_malformed_constrained_paths_fail_closed_and_report_both_failure_tiers() -> None:
    examples = _examples()
    predictions = list(_perfect_predictions(examples))
    predictions[0] = _dual(
        examples[0],
        None,
        confidence=0.0,
        malformed_text="RB2|fault_family|BOGUS|~|~",
        lexical_parse=True,
        exhausted=True,
    )
    predictions[1] = _dual(examples[1], None, confidence=0.0)

    report = _evaluate(examples, tuple(predictions))

    assert report.constrained.parse_rate == pytest.approx(6 / 7)
    assert report.constrained.schema_validity_rate == pytest.approx(5 / 7)
    assert report.constrained.exact_match_rate == pytest.approx(5 / 7)
    assert report.constrained.generation_cap_exhaustion_rate == pytest.approx(1 / 7)
    failures = {item.category: item.count for item in report.failures}
    assert failures["generation_cap_exhausted"] == 1
    assert failures["invalid_schema"] == 2
    assert failures["unconstrained_invalid_schema"] == len(examples)
    assert semantic_composite_score(examples, tuple(predictions)) == 0.0


def test_semantic_evaluation_rejects_alignment_provenance_and_comparator_drift() -> None:
    examples = _examples()
    predictions = _perfect_predictions(examples)

    with pytest.raises(ValueError, match="aligned non-empty"):
        semantic_composite_score((), ())
    with pytest.raises(ValueError, match="prediction provenance"):
        semantic_composite_score(examples, (*predictions[:-1], predictions[0]))
    with pytest.raises(ValueError, match="another evaluation view"):
        _evaluate(
            examples,
            predictions,
            baseline=_baseline_report(RemediationView.SHADOW_RENDERER),
        )
    with pytest.raises(ValueError, match="supports differ"):
        _evaluate(examples, predictions, baseline=_baseline_report(fault_support=1))


def test_semantic_evaluation_binds_exact_comparator_contract_checksum() -> None:
    examples = _examples()
    predictions = _perfect_predictions(examples)
    baseline = _baseline_report()
    with pytest.raises(ValueError, match="comparator artifact hash"):
        evaluate_semantic_predictions(
            view=RemediationView.IID_VALIDATION,
            examples=examples,
            predictions=predictions,
            baseline_report=baseline,
            artifacts=_artifacts(
                predictions,
                baseline,
                comparator_artifact_sha256="0" * 64,
            ),
        )


def test_semantic_evaluation_binds_dataset_manifest_to_comparator() -> None:
    examples = _examples()
    predictions = _perfect_predictions(examples)
    baseline = _baseline_report()
    with pytest.raises(ValueError, match="dataset artifact hash"):
        evaluate_semantic_predictions(
            view=RemediationView.IID_VALIDATION,
            examples=examples,
            predictions=predictions,
            baseline_report=baseline,
            artifacts=_artifacts(
                predictions,
                baseline,
                dataset_manifest_sha256="0" * 64,
            ),
        )


def test_semantic_evaluation_binds_tokenizer_manifest_to_comparator() -> None:
    examples = _examples()
    predictions = _perfect_predictions(examples)
    baseline = _baseline_report()
    with pytest.raises(ValueError, match="tokenizer artifact hash"):
        evaluate_semantic_predictions(
            view=RemediationView.IID_VALIDATION,
            examples=examples,
            predictions=predictions,
            baseline_report=baseline,
            artifacts=_artifacts(
                predictions,
                baseline,
                tokenizer_manifest_sha256="0" * 64,
            ),
        )


def test_semantic_evaluation_rejects_prediction_tokenizer_drift() -> None:
    examples = _examples()
    predictions = list(_perfect_predictions(examples))
    predictions[0] = _prediction_with_tokenizer(predictions[0], "0" * 64)
    prediction_tuple = tuple(predictions)
    baseline = _baseline_report()
    with pytest.raises(ValueError, match="prediction tokenizer hash"):
        evaluate_semantic_predictions(
            view=RemediationView.IID_VALIDATION,
            examples=examples,
            predictions=prediction_tuple,
            baseline_report=baseline,
            artifacts=_artifacts(prediction_tuple, baseline),
        )


def test_semantic_evaluation_binds_frozen_output_contract_artifact_bytes() -> None:
    examples = _examples()
    predictions = _perfect_predictions(examples)
    baseline = _baseline_report()
    with pytest.raises(ValueError, match="output-contract artifact hash"):
        evaluate_semantic_predictions(
            view=RemediationView.IID_VALIDATION,
            examples=examples,
            predictions=predictions,
            baseline_report=baseline,
            artifacts=_artifacts(
                predictions,
                baseline,
                output_contract_sha256="0" * 64,
            ),
        )


def test_semantic_evaluation_rejects_prediction_jsonl_byte_tampering() -> None:
    examples = _examples()
    predictions = _perfect_predictions(examples)
    baseline = _baseline_report()
    canonical_bytes = canonical_prediction_jsonl_bytes(tuple(reversed(predictions)))
    assert canonical_bytes == canonical_prediction_jsonl_bytes(predictions)
    assert (
        prediction_artifact_byte_sha256(predictions) == hashlib.sha256(canonical_bytes).hexdigest()
    )
    with pytest.raises(ValueError, match="prediction artifact byte hash"):
        evaluate_semantic_predictions(
            view=RemediationView.IID_VALIDATION,
            examples=examples,
            predictions=predictions,
            baseline_report=baseline,
            artifacts=_artifacts(
                predictions,
                baseline,
                prediction_artifact_sha256="0" * 64,
            ),
        )
