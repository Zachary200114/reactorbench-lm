"""Behavioral, calibration, abstention, and error-analysis metrics for remediation."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Callable

import numpy as np
from pydantic import Field, StrictFloat, model_validator

from reactorbench.dataset.contracts import (
    ProjectionTaskTargetValue,
    PromptContinuationTarget,
    PromptEvidenceTarget,
)
from reactorbench.evaluation.compact import compact_output_contract, parse_compact_target
from reactorbench.schemas.base import ContractModel, canonical_json_bytes, canonical_sha256
from reactorbench.schemas.enums import ActionLabel, DiagnosisStatus, TaskName
from reactorbench.schemas.target import (
    FaultDiagnosisTarget,
    IncidentSummaryTarget,
    NextActionTarget,
)

from .acceptance import (
    DevelopmentArtifactBinding,
    DevelopmentView,
    DevelopmentViewMetrics,
    MetricEstimate,
    RemediationVersion,
    SemanticMetricSet,
    bind_development_view_metrics,
)
from .baselines import RemediationBaselineReport
from .config import RemediationView
from .data import RemediationExample
from .decoding import CompactPathPrediction, DualPathCompactPrediction

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 6602
CALIBRATION_BINS = 10
SELECTIVE_COVERAGE = 0.8

_VIEW_MAP: dict[RemediationView, DevelopmentView] = {
    RemediationView.IID_VALIDATION: DevelopmentView.IID_VALIDATION,
    RemediationView.SHADOW_RENDERER: DevelopmentView.RENDERER_SHADOW,
    RemediationView.SHADOW_COMPONENT: DevelopmentView.COMPONENT_ROLE_SHADOW,
    RemediationView.SHADOW_SEVERITY: DevelopmentView.SEVERITY_SHADOW,
    RemediationView.SHADOW_COMPOSITION: DevelopmentView.COMPOSITION_SHADOW,
    RemediationView.SHADOW_COUNTERFACTUAL: DevelopmentView.COUNTERFACTUAL_SHADOW,
    RemediationView.SHADOW_NOISE: DevelopmentView.NOISE_SHADOW,
}


def canonical_prediction_jsonl_bytes(
    predictions: tuple[DualPathCompactPrediction, ...],
) -> bytes:
    """Serialize predictions exactly as the immutable pipeline JSONL artifact.

    This is an artifact-byte contract, not a Pydantic contract checksum.  The
    deterministic example-ID ordering mirrors the pipeline writer so evaluation
    cannot accidentally bind metrics to a different serialization.
    """

    if type(predictions) is not tuple or not predictions:
        raise ValueError("prediction artifact requires a non-empty exact tuple")
    if any(type(item) is not DualPathCompactPrediction for item in predictions):
        raise TypeError("prediction artifact contains a non-contract record")
    ordered = tuple(sorted(predictions, key=lambda item: item.example_id))
    if len({item.example_id for item in ordered}) != len(ordered):
        raise ValueError("prediction artifact example IDs must be unique")
    return b"".join(
        canonical_json_bytes(item.model_dump(mode="json", round_trip=True)) + b"\n"
        for item in ordered
    )


def prediction_artifact_byte_sha256(
    predictions: tuple[DualPathCompactPrediction, ...],
) -> str:
    """Hash the exact canonical JSONL bytes written for a prediction artifact."""

    return hashlib.sha256(canonical_prediction_jsonl_bytes(predictions)).hexdigest()


def compact_output_contract_byte_sha256() -> str:
    """Hash the exact newline-terminated compact-contract snapshot bytes."""

    payload = canonical_json_bytes(compact_output_contract()) + b"\n"
    return hashlib.sha256(payload).hexdigest()


class PathBehaviorMetrics(ContractModel):
    path: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    sample_count: int = Field(ge=1)
    parse_rate: StrictFloat = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    schema_validity_rate: StrictFloat = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    exact_match_rate: StrictFloat = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    prompt_truncation_rate: StrictFloat = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    generation_cap_exhaustion_rate: StrictFloat = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mean_latency_seconds: StrictFloat = Field(ge=0.0, allow_inf_nan=False)


class TaskBehaviorMetrics(ContractModel):
    task_name: TaskName
    support: int = Field(ge=1)
    exact_match_rate: StrictFloat = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    macro_f1: StrictFloat | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)


class FailureCategory(ContractModel):
    category: str = Field(pattern=r"^[a-z][a-z0-9_]*$", min_length=1, max_length=80)
    count: int = Field(ge=0)


class SemanticEvaluationReport(ContractModel):
    report_version: str
    evaluation_view: RemediationView
    example_count: int = Field(ge=1)
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    constrained: PathBehaviorMetrics
    unconstrained: PathBehaviorMetrics
    task_metrics: tuple[TaskBehaviorMetrics, ...]
    failures: tuple[FailureCategory, ...]
    view_metrics: DevelopmentViewMetrics
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def shape_and_checksum_match(self) -> SemanticEvaluationReport:
        if self.constrained.sample_count != self.example_count or (
            self.unconstrained.sample_count != self.example_count
        ):
            raise ValueError("path metrics do not cover the evaluation view")
        tasks = tuple(item.task_name for item in self.task_metrics)
        if tasks != tuple(task for task in TaskName if task in set(tasks)):
            raise ValueError("task metrics must use canonical order")
        categories = tuple(item.category for item in self.failures)
        if categories != tuple(sorted(categories)) or len(categories) != len(set(categories)):
            raise ValueError("failure categories must be unique and sorted")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("semantic evaluation report checksum mismatch")
        return self


def _target_label(target: ProjectionTaskTargetValue) -> str | None:
    if type(target) is FaultDiagnosisTarget:
        if target.diagnosis_status is not DiagnosisStatus.DIAGNOSED:
            return target.diagnosis_status.value
        return "DIAGNOSED:" + "+".join(label.value for label in target.fault_labels)
    if type(target) is NextActionTarget:
        return target.immediate_action.value
    if type(target) is PromptContinuationTarget:
        return target.next_event_type.value
    return None


def _optional_target_label(target: ProjectionTaskTargetValue | None) -> str | None:
    return None if target is None else _target_label(target)


def _evidence_refs(target: ProjectionTaskTargetValue | None) -> tuple[str, ...]:
    return target.fact_refs if type(target) is PromptEvidenceTarget else ()


def _is_no_fault(target: ProjectionTaskTargetValue | None) -> bool:
    return (
        type(target) is FaultDiagnosisTarget and target.diagnosis_status is DiagnosisStatus.NO_FAULT
    )


def _prediction_target(
    example: RemediationExample, prediction: CompactPathPrediction
) -> ProjectionTaskTargetValue | None:
    if not prediction.schema_valid:
        return None
    try:
        return parse_compact_target(
            prediction.generated_text,
            context=example.compact_context,
        )
    except (TypeError, ValueError):
        return None


def _macro_f1(truth: tuple[str, ...], predicted: tuple[str, ...]) -> float:
    if not truth or len(truth) != len(predicted):
        raise ValueError("macro F1 requires aligned non-empty labels")
    labels = tuple(sorted(set(truth) | set(predicted)))
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            expected == label and actual == label
            for expected, actual in zip(truth, predicted, strict=True)
        )
        false_positive = sum(
            expected != label and actual == label
            for expected, actual in zip(truth, predicted, strict=True)
        )
        false_negative = sum(
            expected == label and actual != label
            for expected, actual in zip(truth, predicted, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores)


def _set_f1(truth: tuple[tuple[str, ...], ...], predicted: tuple[tuple[str, ...], ...]) -> float:
    true_positive = false_positive = false_negative = 0
    for expected, actual in zip(truth, predicted, strict=True):
        expected_set = set(expected)
        actual_set = set(actual)
        true_positive += len(expected_set & actual_set)
        false_positive += len(actual_set - expected_set)
        false_negative += len(expected_set - actual_set)
    precision = (
        1.0
        if true_positive + false_positive == 0
        else true_positive / (true_positive + false_positive)
    )
    recall = (
        1.0
        if true_positive + false_negative == 0
        else true_positive / (true_positive + false_negative)
    )
    return 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)


def _calibration(correct: tuple[bool, ...], confidence: tuple[float, ...]) -> tuple[float, float]:
    count = len(correct)
    ece = 0.0
    for index in range(CALIBRATION_BINS):
        lower = index / CALIBRATION_BINS
        upper = (index + 1) / CALIBRATION_BINS
        members = tuple(
            position
            for position, value in enumerate(confidence)
            if lower <= value < upper or (index == CALIBRATION_BINS - 1 and value == 1.0)
        )
        if members:
            accuracy = sum(correct[position] for position in members) / len(members)
            mean_confidence = sum(confidence[position] for position in members) / len(members)
            ece += len(members) / count * abs(accuracy - mean_confidence)
    retained = max(1, math.ceil(count * SELECTIVE_COVERAGE))
    ordering = sorted(range(count), key=lambda index: (-confidence[index], index))
    risk = 1.0 - sum(correct[index] for index in ordering[:retained]) / retained
    return ece, risk


def _bootstrap(
    support: int,
    evaluator: Callable[[tuple[int, ...]], float],
) -> tuple[float, float, float]:
    if support == 0:
        return 0.0, 0.0, 0.0
    identity = tuple(range(support))
    estimate = float(evaluator(identity))
    if not math.isfinite(estimate) or not 0.0 <= estimate <= 1.0:
        raise ValueError("metric estimate is not a finite probability")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for iteration in range(BOOTSTRAP_RESAMPLES):
        indices = tuple(int(item) for item in generator.integers(0, support, size=support))
        values[iteration] = evaluator(indices)
    lower, upper = np.quantile(values, (0.025, 0.975), method="linear")
    return estimate, min(float(lower), estimate), max(float(upper), estimate)


def _estimate(
    support: int,
    evaluator: Callable[[tuple[int, ...]], float],
) -> MetricEstimate:
    estimate, lower, upper = _bootstrap(support, evaluator)
    return MetricEstimate(
        support=support,
        estimate=estimate,
        interval_lower=lower,
        interval_upper=upper,
    )


def _rate(values: tuple[bool, ...]) -> MetricEstimate:
    return _estimate(
        len(values),
        lambda indices: sum(values[index] for index in indices) / len(indices),
    )


def _classification_estimate(truth: tuple[str, ...], predicted: tuple[str, ...]) -> MetricEstimate:
    return _estimate(
        len(truth),
        lambda indices: _macro_f1(
            tuple(truth[index] for index in indices),
            tuple(predicted[index] for index in indices),
        ),
    )


def _evidence_estimate(
    truth: tuple[tuple[str, ...], ...], predicted: tuple[tuple[str, ...], ...]
) -> MetricEstimate:
    return _estimate(
        len(truth),
        lambda indices: _set_f1(
            tuple(truth[index] for index in indices),
            tuple(predicted[index] for index in indices),
        ),
    )


def _baseline_pairs(
    report: RemediationBaselineReport, task: TaskName
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    candidates = tuple(
        result
        for result in report.results
        if result.task_name == task.value and result.classification is not None
    )
    if not candidates:
        return (), ()
    best = min(
        candidates,
        key=lambda result: (
            -float(result.classification.macro_f1),  # type: ignore[union-attr]
            result.baseline_name,
        ),
    )
    metrics = best.classification
    if metrics is None:
        raise RuntimeError("selected classification baseline has no metrics")
    truth: list[str] = []
    predicted: list[str] = []
    for row, expected in enumerate(metrics.labels):
        for column, actual in enumerate(metrics.labels):
            count = metrics.confusion_matrix[row][column]
            truth.extend(expected for _ in range(count))
            predicted.extend(actual for _ in range(count))
    if len(truth) != metrics.sample_count:
        raise ValueError("baseline confusion matrix does not reconstruct its support")
    return tuple(truth), tuple(predicted)


def _requires_abstention(target: ProjectionTaskTargetValue) -> bool:
    if type(target) is FaultDiagnosisTarget:
        return target.diagnosis_status is DiagnosisStatus.UNRESOLVED
    if type(target) is NextActionTarget:
        return target.immediate_action is ActionLabel.INSUFFICIENT_EVIDENCE
    if type(target) is IncidentSummaryTarget:
        return (
            target.diagnosis_status is DiagnosisStatus.UNRESOLVED
            or target.immediate_action is ActionLabel.INSUFFICIENT_EVIDENCE
        )
    return False


def _predicted_abstention(target: ProjectionTaskTargetValue | None) -> bool:
    return target is not None and _requires_abstention(target)


def _path_behavior(
    name: str,
    examples: tuple[RemediationExample, ...],
    paths: tuple[CompactPathPrediction, ...],
) -> PathBehaviorMetrics:
    exact = tuple(
        path.canonical_target_json == example.canonical_target_json
        for example, path in zip(examples, paths, strict=True)
    )
    count = len(paths)
    return PathBehaviorMetrics(
        path=name,
        sample_count=count,
        parse_rate=sum(path.compact_parse_success for path in paths) / count,
        schema_validity_rate=sum(path.schema_valid for path in paths) / count,
        exact_match_rate=sum(exact) / count,
        prompt_truncation_rate=sum(path.prompt_truncated for path in paths) / count,
        generation_cap_exhaustion_rate=sum(path.generation_cap_exhausted for path in paths) / count,
        mean_latency_seconds=sum(path.elapsed_seconds for path in paths) / count,
    )


def semantic_composite_score(
    examples: tuple[RemediationExample, ...],
    predictions: tuple[DualPathCompactPrediction, ...],
) -> float:
    """Return the frozen equal-weight semantic checkpoint-selection score.

    Schema validity is a hard prerequisite.  The remaining score averages exact
    match, every supported classification/evidence/abstention/specificity metric,
    calibration quality, and selective-risk quality. Comparator margins are constant
    across checkpoints on one view and are therefore reserved for the full gate.
    """

    if (
        type(examples) is not tuple
        or type(predictions) is not tuple
        or not examples
        or len(examples) != len(predictions)
    ):
        raise ValueError("semantic composite requires aligned non-empty tuples")
    ordered_examples = tuple(sorted(examples, key=lambda item: item.example_id))
    ordered_predictions = tuple(sorted(predictions, key=lambda item: item.example_id))
    if any(
        example.example_id != prediction.example_id
        or example.checksum_sha256 != prediction.example_checksum_sha256
        for example, prediction in zip(ordered_examples, ordered_predictions, strict=True)
    ):
        raise ValueError("semantic composite prediction provenance mismatch")
    paths = tuple(item.constrained for item in ordered_predictions)
    if not all(path.schema_valid for path in paths):
        return 0.0
    predicted_targets = tuple(
        _prediction_target(example, path)
        for example, path in zip(ordered_examples, paths, strict=True)
    )
    expected_targets = tuple(
        parse_compact_target(example.compact_target, context=example.compact_context)
        for example in ordered_examples
    )
    exact = tuple(
        path.canonical_target_json == example.canonical_target_json
        for example, path in zip(ordered_examples, paths, strict=True)
    )
    values: list[float] = [sum(exact) / len(exact)]
    classification_tasks = frozenset(
        {TaskName.FAULT_FAMILY, TaskName.NEXT_ACTION, TaskName.CONTINUE_LOG}
    )
    for task in (item for item in TaskName if item in classification_tasks):
        positions = tuple(
            index for index, example in enumerate(ordered_examples) if example.task_name is task
        )
        if positions:
            truth = tuple(str(ordered_examples[index].classification_label) for index in positions)
            predicted = tuple(
                _optional_target_label(predicted_targets[index]) or "__INVALID__"
                for index in positions
            )
            values.append(_macro_f1(truth, predicted))
    evidence_positions = tuple(
        index
        for index, target in enumerate(expected_targets)
        if type(target) is PromptEvidenceTarget
    )
    if evidence_positions:
        values.append(
            _set_f1(
                tuple(
                    expected_targets[index].fact_refs  # type: ignore[union-attr]
                    for index in evidence_positions
                ),
                tuple(_evidence_refs(predicted_targets[index]) for index in evidence_positions),
            )
        )
    abstention_positions = tuple(
        index for index, target in enumerate(expected_targets) if _requires_abstention(target)
    )
    if abstention_positions:
        values.append(
            sum(_predicted_abstention(predicted_targets[index]) for index in abstention_positions)
            / len(abstention_positions)
        )
    no_fault_positions = tuple(
        index
        for index, target in enumerate(expected_targets)
        if type(target) is FaultDiagnosisTarget
        and target.diagnosis_status is DiagnosisStatus.NO_FAULT
    )
    if no_fault_positions:
        values.append(
            sum(_is_no_fault(predicted_targets[index]) for index in no_fault_positions)
            / len(no_fault_positions)
        )
    ece, risk = _calibration(
        exact,
        tuple(float(path.selected_token_geometric_mean_probability) for path in paths),
    )
    values.extend((1.0 - ece, 1.0 - risk))
    score = sum(values) / len(values)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError("semantic composite produced an invalid score")
    return score


def classification_macro_f1_score(
    examples: tuple[RemediationExample, ...],
    predictions: tuple[DualPathCompactPrediction, ...],
    *,
    task_name: TaskName,
) -> float:
    """Return one provenance-checked constrained classification-task macro F1."""

    classification_tasks = {
        TaskName.FAULT_FAMILY,
        TaskName.NEXT_ACTION,
        TaskName.CONTINUE_LOG,
    }
    if type(task_name) is not TaskName or task_name not in classification_tasks:
        raise ValueError("macro F1 selection requires a classification task")
    if (
        type(examples) is not tuple
        or type(predictions) is not tuple
        or not examples
        or len(examples) != len(predictions)
    ):
        raise ValueError("macro F1 selection requires aligned non-empty tuples")
    ordered_examples = tuple(sorted(examples, key=lambda item: item.example_id))
    ordered_predictions = tuple(sorted(predictions, key=lambda item: item.example_id))
    if any(
        example.example_id != prediction.example_id
        or example.checksum_sha256 != prediction.example_checksum_sha256
        for example, prediction in zip(ordered_examples, ordered_predictions, strict=True)
    ):
        raise ValueError("macro F1 selection prediction provenance mismatch")
    positions = tuple(
        index for index, example in enumerate(ordered_examples) if example.task_name is task_name
    )
    if not positions:
        raise ValueError("macro F1 selection task has no examples")
    predicted_targets = tuple(
        _prediction_target(example, prediction.constrained)
        for example, prediction in zip(ordered_examples, ordered_predictions, strict=True)
    )
    truth = tuple(str(ordered_examples[index].classification_label) for index in positions)
    predicted = tuple(
        _optional_target_label(predicted_targets[index]) or "__INVALID__" for index in positions
    )
    return _macro_f1(truth, predicted)


def evaluate_semantic_predictions(
    *,
    view: RemediationView,
    examples: tuple[RemediationExample, ...],
    predictions: tuple[DualPathCompactPrediction, ...],
    baseline_report: RemediationBaselineReport,
    artifacts: DevelopmentArtifactBinding,
    confidence_transform: Callable[[float], float] | None = None,
) -> SemanticEvaluationReport:
    """Calculate supported task metrics and bind them to exact development evidence."""

    if type(view) is not RemediationView or view not in _VIEW_MAP:
        raise ValueError("semantic evaluation requires a validation or shadow view")
    if (
        type(examples) is not tuple
        or type(predictions) is not tuple
        or not examples
        or len(examples) != len(predictions)
    ):
        raise ValueError("semantic examples and predictions must be aligned and non-empty")
    if type(baseline_report) is not RemediationBaselineReport or (
        baseline_report.evaluation_view is not view
    ):
        raise ValueError("baseline report is bound to another evaluation view")
    if type(artifacts) is not DevelopmentArtifactBinding:
        raise TypeError("semantic evaluation requires an exact artifact binding")
    if artifacts.comparator_artifact_sha256 != baseline_report.checksum_sha256:
        raise ValueError("comparator artifact hash differs from the baseline report")
    if artifacts.dataset_manifest_sha256 != baseline_report.dataset_manifest_sha256:
        raise ValueError("dataset artifact hash differs from the baseline report")
    if artifacts.tokenizer_manifest_sha256 != baseline_report.tokenizer_manifest_sha256:
        raise ValueError("tokenizer artifact hash differs from the baseline report")
    if artifacts.output_contract_sha256 != compact_output_contract_byte_sha256():
        raise ValueError("output-contract artifact hash differs from the frozen contract")
    ordered_examples = tuple(sorted(examples, key=lambda item: item.example_id))
    ordered_predictions = tuple(sorted(predictions, key=lambda item: item.example_id))
    if len({item.example_id for item in ordered_examples}) != len(ordered_examples) or len(
        {item.example_id for item in ordered_predictions}
    ) != len(ordered_predictions):
        raise ValueError("semantic examples and predictions must have unique IDs")
    for example, prediction in zip(ordered_examples, ordered_predictions, strict=True):
        if (
            example.view is not view
            or prediction.example_id != example.example_id
            or prediction.example_checksum_sha256 != example.checksum_sha256
            or prediction.task_name is not example.task_name
        ):
            raise ValueError("semantic prediction provenance differs from its example")
        if prediction.tokenizer_manifest_sha256 != artifacts.tokenizer_manifest_sha256:
            raise ValueError("prediction tokenizer hash differs from its artifact binding")
    if artifacts.prediction_artifact_sha256 != prediction_artifact_byte_sha256(predictions):
        raise ValueError("prediction artifact byte hash differs from canonical JSONL")

    constrained_paths = tuple(item.constrained for item in ordered_predictions)
    unconstrained_paths = tuple(item.unconstrained for item in ordered_predictions)
    predicted_targets = tuple(
        _prediction_target(example, path)
        for example, path in zip(ordered_examples, constrained_paths, strict=True)
    )
    expected_targets = tuple(
        parse_compact_target(example.compact_target, context=example.compact_context)
        for example in ordered_examples
    )
    exact = tuple(
        path.canonical_target_json == example.canonical_target_json
        for example, path in zip(ordered_examples, constrained_paths, strict=True)
    )
    raw_confidence = tuple(
        float(path.selected_token_geometric_mean_probability) for path in constrained_paths
    )
    confidence = (
        raw_confidence
        if confidence_transform is None
        else tuple(confidence_transform(value) for value in raw_confidence)
    )
    if any(
        type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in confidence
    ):
        raise ValueError("confidence transform must return finite probabilities")

    classification: dict[TaskName, MetricEstimate] = {}
    task_metrics: list[TaskBehaviorMetrics] = []
    classification_tasks = frozenset(
        {TaskName.FAULT_FAMILY, TaskName.NEXT_ACTION, TaskName.CONTINUE_LOG}
    )
    for task in (item for item in TaskName if item in classification_tasks):
        positions = tuple(
            index for index, example in enumerate(ordered_examples) if example.task_name is task
        )
        if not positions:
            classification[task] = MetricEstimate(
                support=0,
                estimate=0.0,
                interval_lower=0.0,
                interval_upper=0.0,
            )
            continue
        truth = tuple(str(ordered_examples[index].classification_label) for index in positions)
        predicted = tuple(_optional_target_label(predicted_targets[index]) for index in positions)
        predicted_labels = tuple(
            value if value is not None else "__INVALID__" for value in predicted
        )
        estimate = _classification_estimate(truth, predicted_labels)
        classification[task] = estimate
        task_metrics.append(
            TaskBehaviorMetrics(
                task_name=task,
                support=len(positions),
                exact_match_rate=sum(exact[index] for index in positions) / len(positions),
                macro_f1=estimate.estimate,
            )
        )

    evidence_positions = tuple(
        index
        for index, target in enumerate(expected_targets)
        if type(target) is PromptEvidenceTarget
    )
    evidence_truth = tuple(
        expected_targets[index].fact_refs  # type: ignore[union-attr]
        for index in evidence_positions
    )
    evidence_predicted = tuple(
        _evidence_refs(predicted_targets[index]) for index in evidence_positions
    )
    evidence = (
        _evidence_estimate(evidence_truth, evidence_predicted)
        if evidence_positions
        else MetricEstimate(support=0, estimate=0.0, interval_lower=0.0, interval_upper=0.0)
    )

    abstention_positions = tuple(
        index for index, target in enumerate(expected_targets) if _requires_abstention(target)
    )
    abstention_correct = tuple(
        _predicted_abstention(predicted_targets[index]) for index in abstention_positions
    )
    abstention = (
        _rate(abstention_correct)
        if abstention_correct
        else MetricEstimate(support=0, estimate=0.0, interval_lower=0.0, interval_upper=0.0)
    )

    no_fault_positions = tuple(
        index
        for index, target in enumerate(expected_targets)
        if type(target) is FaultDiagnosisTarget
        and target.diagnosis_status is DiagnosisStatus.NO_FAULT
    )
    false_positives = tuple(
        not _is_no_fault(predicted_targets[index]) for index in no_fault_positions
    )
    false_positive_rate = (
        _rate(false_positives)
        if false_positives
        else MetricEstimate(support=0, estimate=0.0, interval_lower=0.0, interval_upper=0.0)
    )

    calibration_ece = _estimate(
        len(exact),
        lambda indices: _calibration(
            tuple(exact[index] for index in indices),
            tuple(confidence[index] for index in indices),
        )[0],
    )
    selective_risk = _estimate(
        len(exact),
        lambda indices: _calibration(
            tuple(exact[index] for index in indices),
            tuple(confidence[index] for index in indices),
        )[1],
    )
    constrained_parse = _rate(tuple(path.compact_parse_success for path in constrained_paths))
    constrained_schema = _rate(tuple(path.schema_valid for path in constrained_paths))
    unconstrained_parse = _rate(tuple(path.compact_parse_success for path in unconstrained_paths))
    unconstrained_schema = _rate(tuple(path.schema_valid for path in unconstrained_paths))
    fault_baseline_truth, fault_baseline_prediction = _baseline_pairs(
        baseline_report, TaskName.FAULT_FAMILY
    )
    action_baseline_truth, action_baseline_prediction = _baseline_pairs(
        baseline_report, TaskName.NEXT_ACTION
    )
    fault_comparator = (
        _classification_estimate(fault_baseline_truth, fault_baseline_prediction)
        if fault_baseline_truth
        else MetricEstimate(support=0, estimate=0.0, interval_lower=0.0, interval_upper=0.0)
    )
    action_comparator = (
        _classification_estimate(action_baseline_truth, action_baseline_prediction)
        if action_baseline_truth
        else MetricEstimate(support=0, estimate=0.0, interval_lower=0.0, interval_upper=0.0)
    )
    if fault_comparator.support != classification[TaskName.FAULT_FAMILY].support or (
        action_comparator.support != classification[TaskName.NEXT_ACTION].support
    ):
        raise ValueError("model and comparator classification supports differ")
    semantic = SemanticMetricSet(
        constrained_parse_rate=constrained_parse,
        constrained_schema_validity_rate=constrained_schema,
        unconstrained_parse_rate=unconstrained_parse,
        unconstrained_schema_validity_rate=unconstrained_schema,
        fault_family_macro_f1=classification[TaskName.FAULT_FAMILY],
        strongest_fault_comparator_macro_f1=fault_comparator,
        next_action_macro_f1=classification[TaskName.NEXT_ACTION],
        strongest_action_comparator_macro_f1=action_comparator,
        continuation_macro_f1=classification[TaskName.CONTINUE_LOG],
        evidence_f1=evidence,
        required_abstention_accuracy=abstention,
        no_fault_false_positive_rate=false_positive_rate,
        expected_calibration_error=calibration_ece,
        selective_risk_at_80_percent_coverage=selective_risk,
    )
    development_view = _VIEW_MAP[view]
    version = (
        RemediationVersion.V03 if view is RemediationView.IID_VALIDATION else RemediationVersion.V04
    )
    composition = _rate(exact) if view is RemediationView.SHADOW_COMPOSITION else None
    view_metrics = bind_development_view_metrics(
        contract_version=version,
        view=development_view,
        sample_count=len(ordered_examples),
        artifacts=artifacts,
        metrics=semantic,
        composition_score_interval=composition,
    )
    failures = Counter[str]()
    for index, (example, path, target) in enumerate(
        zip(ordered_examples, constrained_paths, predicted_targets, strict=True)
    ):
        if path.generation_cap_exhausted:
            failures["generation_cap_exhausted"] += 1
        if not path.schema_valid:
            failures["invalid_schema"] += 1
        elif not exact[index]:
            failures["semantic_mismatch"] += 1
        if _requires_abstention(expected_targets[index]) and not _predicted_abstention(target):
            failures["required_abstention_missed"] += 1
        if example.task_name is TaskName.EXTRACT_EVIDENCE and not exact[index]:
            failures["evidence_mismatch"] += 1
    unconstrained_invalid = sum(not path.schema_valid for path in unconstrained_paths)
    failures["unconstrained_invalid_schema"] = unconstrained_invalid
    draft = SemanticEvaluationReport.model_construct(
        report_version=version.value,
        evaluation_view=view,
        example_count=len(ordered_examples),
        predictions_sha256=artifacts.prediction_artifact_sha256,
        baseline_report_sha256=artifacts.comparator_artifact_sha256,
        constrained=_path_behavior("constrained", ordered_examples, constrained_paths),
        unconstrained=_path_behavior("unconstrained", ordered_examples, unconstrained_paths),
        task_metrics=tuple(task_metrics),
        failures=tuple(
            FailureCategory(category=category, count=failures[category])
            for category in sorted(failures)
        ),
        view_metrics=view_metrics,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return SemanticEvaluationReport(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
        checksum_sha256=checksum,
    )


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "FailureCategory",
    "PathBehaviorMetrics",
    "SemanticEvaluationReport",
    "TaskBehaviorMetrics",
    "canonical_prediction_jsonl_bytes",
    "classification_macro_f1_score",
    "compact_output_contract_byte_sha256",
    "evaluate_semantic_predictions",
    "prediction_artifact_byte_sha256",
    "semantic_composite_score",
]
