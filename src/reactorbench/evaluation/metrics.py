"""Deterministic Phase 5 metric contracts and calculations."""

from __future__ import annotations

import math

import numpy as np
from pydantic import Field, StrictFloat, model_validator

from reactorbench.schemas.base import ContractModel, canonical_sha256
from reactorbench.schemas.enums import TaskName


class ClassificationMetrics(ContractModel):
    task_name: TaskName
    sample_count: int = Field(strict=True, ge=1)
    labels: tuple[str, ...]
    supports: tuple[int, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    accuracy: StrictFloat
    macro_f1: StrictFloat
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def shape_and_checksum_are_valid(self) -> ClassificationMetrics:
        size = len(self.labels)
        if not size or len(set(self.labels)) != size or self.labels != tuple(sorted(self.labels)):
            raise ValueError("classification labels must be unique canonical order")
        if len(self.supports) != size or sum(self.supports) != self.sample_count:
            raise ValueError("classification supports do not match sample count")
        if len(self.confusion_matrix) != size or any(
            len(row) != size for row in self.confusion_matrix
        ):
            raise ValueError("confusion matrix shape does not match labels")
        if sum(sum(row) for row in self.confusion_matrix) != self.sample_count:
            raise ValueError("confusion matrix does not match sample count")
        if not (0.0 <= self.accuracy <= 1.0 and 0.0 <= self.macro_f1 <= 1.0):
            raise ValueError("classification metrics must be in [0, 1]")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("classification metric checksum mismatch")
        return self


class LanguageModelMetrics(ContractModel):
    sample_count: int = Field(strict=True, ge=1)
    target_token_count: int = Field(strict=True, ge=1)
    negative_log_likelihood: StrictFloat
    perplexity: StrictFloat

    @model_validator(mode="after")
    def values_are_finite(self) -> LanguageModelMetrics:
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (self.negative_log_likelihood, self.perplexity)
        ):
            raise ValueError("language-model metrics must be finite and non-negative")
        return self


class BootstrapInterval(ContractModel):
    sample_count: int = Field(strict=True, ge=1)
    resamples: int = Field(strict=True, ge=100, le=100_000)
    seed: int = Field(strict=True, ge=0, le=4_294_967_295)
    confidence_level: StrictFloat
    estimate: StrictFloat
    lower: StrictFloat
    upper: StrictFloat

    @model_validator(mode="after")
    def interval_is_ordered_and_finite(self) -> BootstrapInterval:
        values = (self.confidence_level, self.estimate, self.lower, self.upper)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("bootstrap interval values must be finite")
        if not 0.0 < self.confidence_level < 1.0 or not self.lower <= self.estimate <= self.upper:
            raise ValueError("bootstrap interval is invalid")
        return self


class CalibrationMetrics(ContractModel):
    sample_count: int = Field(strict=True, ge=1)
    bin_count: int = Field(strict=True, ge=2, le=100)
    expected_calibration_error: StrictFloat
    selective_coverage: StrictFloat
    selective_risk: StrictFloat

    @model_validator(mode="after")
    def values_are_probabilities(self) -> CalibrationMetrics:
        values = (
            self.expected_calibration_error,
            self.selective_coverage,
            self.selective_risk,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("calibration values must be finite probabilities")
        return self


class SetF1Metrics(ContractModel):
    sample_count: int = Field(strict=True, ge=1)
    true_positive: int = Field(strict=True, ge=0)
    false_positive: int = Field(strict=True, ge=0)
    false_negative: int = Field(strict=True, ge=0)
    precision: StrictFloat
    recall: StrictFloat
    f1: StrictFloat

    @model_validator(mode="after")
    def counts_and_rates_are_valid(self) -> SetF1Metrics:
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in (self.precision, self.recall, self.f1)
        ):
            raise ValueError("set metrics must be finite probabilities")
        return self


def classification_metrics(
    task_name: TaskName,
    truth: tuple[str, ...],
    predicted: tuple[str, ...],
) -> ClassificationMetrics:
    if type(task_name) is not TaskName or type(truth) is not tuple or type(predicted) is not tuple:
        raise TypeError("classification metrics require exact typed tuples")
    if not truth or len(truth) != len(predicted):
        raise ValueError("classification truth and predictions must be non-empty and aligned")
    if any(type(value) is not str or not value for value in truth + predicted):
        raise ValueError("classification labels must be non-empty strings")
    labels = tuple(sorted(set(truth) | set(predicted)))
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for expected, actual in zip(truth, predicted, strict=True):
        matrix[index[expected]][index[actual]] += 1
    supports = tuple(sum(row) for row in matrix)
    correct = sum(matrix[position][position] for position in range(len(labels)))
    f1_values: list[float] = []
    for position in range(len(labels)):
        true_positive = matrix[position][position]
        false_positive = sum(row[position] for row in matrix) - true_positive
        false_negative = sum(matrix[position]) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    draft = ClassificationMetrics.model_construct(
        task_name=task_name,
        sample_count=len(truth),
        labels=labels,
        supports=supports,
        confusion_matrix=tuple(tuple(row) for row in matrix),
        accuracy=correct / len(truth),
        macro_f1=sum(f1_values) / len(f1_values),
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return ClassificationMetrics(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
        checksum_sha256=checksum,
    )


def language_model_metrics(
    *, sample_count: int, target_token_count: int, negative_log_likelihood: float
) -> LanguageModelMetrics:
    if not math.isfinite(negative_log_likelihood) or negative_log_likelihood < 0.0:
        raise ValueError("negative_log_likelihood must be finite and non-negative")
    return LanguageModelMetrics(
        sample_count=sample_count,
        target_token_count=target_token_count,
        negative_log_likelihood=float(negative_log_likelihood),
        perplexity=float(math.exp(min(negative_log_likelihood, 80.0))),
    )


def bootstrap_mean_interval(
    values: tuple[float, ...],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> BootstrapInterval:
    if (
        type(values) is not tuple
        or not values
        or any(type(value) is not float or not math.isfinite(value) for value in values)
    ):
        raise TypeError("bootstrap values must be a non-empty exact float tuple")
    if (
        type(resamples) is not int
        or not 100 <= resamples <= 100_000
        or type(seed) is not int
        or not 0 <= seed <= 4_294_967_295
        or type(confidence_level) is not float
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError("bootstrap configuration is invalid")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    estimates = array[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(estimates, (alpha, 1.0 - alpha), method="linear")
    estimate = float(array.mean())
    return BootstrapInterval(
        sample_count=len(values),
        resamples=resamples,
        seed=seed,
        confidence_level=confidence_level,
        estimate=estimate,
        lower=min(float(lower), estimate),
        upper=max(float(upper), estimate),
    )


def calibration_metrics(
    correct: tuple[bool, ...],
    confidence: tuple[float, ...],
    *,
    bin_count: int,
    selective_coverage: float,
) -> CalibrationMetrics:
    if (
        type(correct) is not tuple
        or type(confidence) is not tuple
        or not correct
        or len(correct) != len(confidence)
        or any(type(item) is not bool for item in correct)
        or any(
            type(item) is not float or not math.isfinite(item) or not 0.0 <= item <= 1.0
            for item in confidence
        )
    ):
        raise TypeError("calibration inputs must be aligned exact tuples")
    if type(bin_count) is not int or not 2 <= bin_count <= 100:
        raise ValueError("bin_count is invalid")
    if type(selective_coverage) is not float or not 0.0 < selective_coverage <= 1.0:
        raise ValueError("selective coverage is invalid")
    ece = 0.0
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        members = tuple(
            position
            for position, value in enumerate(confidence)
            if lower <= value < upper or (index == bin_count - 1 and value == 1.0)
        )
        if not members:
            continue
        accuracy = sum(correct[position] for position in members) / len(members)
        mean_confidence = sum(confidence[position] for position in members) / len(members)
        ece += len(members) / len(correct) * abs(accuracy - mean_confidence)
    retained = max(1, math.ceil(len(correct) * selective_coverage))
    ordering = sorted(range(len(correct)), key=lambda index: (-confidence[index], index))
    risk = 1.0 - sum(correct[index] for index in ordering[:retained]) / retained
    return CalibrationMetrics(
        sample_count=len(correct),
        bin_count=bin_count,
        expected_calibration_error=ece,
        selective_coverage=retained / len(correct),
        selective_risk=risk,
    )


def set_f1_metrics(
    truth: tuple[tuple[str, ...], ...], predicted: tuple[tuple[str, ...], ...]
) -> SetF1Metrics:
    if (
        type(truth) is not tuple
        or type(predicted) is not tuple
        or not truth
        or len(truth) != len(predicted)
    ):
        raise ValueError("set metrics require non-empty aligned tuples")
    if any(type(values) is not tuple for values in truth + predicted):
        raise TypeError("set metric rows must be exact tuples")
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
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return SetF1Metrics(
        sample_count=len(truth),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
    )


__all__ = [
    "BootstrapInterval",
    "CalibrationMetrics",
    "ClassificationMetrics",
    "LanguageModelMetrics",
    "SetF1Metrics",
    "bootstrap_mean_interval",
    "calibration_metrics",
    "classification_metrics",
    "language_model_metrics",
    "set_f1_metrics",
]
