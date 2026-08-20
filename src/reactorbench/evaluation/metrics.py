"""Deterministic Phase 5 metric contracts and calculations."""

from __future__ import annotations

import math

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


__all__ = [
    "ClassificationMetrics",
    "LanguageModelMetrics",
    "classification_metrics",
    "language_model_metrics",
]
