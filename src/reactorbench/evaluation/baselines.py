"""Preregistered deterministic Phase 5 baseline implementations."""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from typing import Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import Field, StrictFloat, model_validator
from torch import Tensor, nn

from reactorbench.schemas.base import ContractModel, canonical_sha256
from reactorbench.schemas.enums import TaskName
from reactorbench.tokenizer import PAD_ID, ProjectTokenizer

from .config import BaselineConfig
from .data import ExperimentData, ExperimentExample, examples_for_task
from .metrics import (
    ClassificationMetrics,
    LanguageModelMetrics,
    classification_metrics,
    language_model_metrics,
)
from .serialization import TokenizedExample

_WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class BaselineResult(ContractModel):
    baseline_name: str = Field(min_length=1, max_length=64)
    task_name: str = Field(min_length=1, max_length=64)
    result_kind: Literal["classification", "language_model"]
    parameter_count: int = Field(strict=True, ge=0)
    elapsed_seconds: StrictFloat
    classification: ClassificationMetrics | None = None
    language_model: LanguageModelMetrics | None = None
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def result_shape_and_checksum_are_valid(self) -> BaselineResult:
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("baseline elapsed time must be finite and non-negative")
        expected_classification = self.result_kind == "classification"
        if expected_classification != (self.classification is not None):
            raise ValueError("classification result shape is inconsistent")
        if expected_classification == (self.language_model is not None):
            raise ValueError("language-model result shape is inconsistent")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("baseline result checksum mismatch")
        return self


def _result(
    *,
    name: str,
    task_name: str,
    parameter_count: int,
    elapsed_seconds: float,
    classification: ClassificationMetrics | None = None,
    language_model: LanguageModelMetrics | None = None,
) -> BaselineResult:
    kind: Literal["classification", "language_model"] = (
        "classification" if classification is not None else "language_model"
    )
    draft = BaselineResult.model_construct(
        baseline_name=name,
        task_name=task_name,
        result_kind=kind,
        parameter_count=parameter_count,
        elapsed_seconds=float(elapsed_seconds),
        classification=classification,
        language_model=language_model,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return BaselineResult(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
        checksum_sha256=checksum,
    )


def _labels(records: tuple[ExperimentExample, ...]) -> tuple[str, ...]:
    values = tuple(record.classification_label for record in records)
    if any(value is None for value in values):
        raise ValueError("classification baseline received a non-classification example")
    return tuple(value for value in values if value is not None)


def _majority_label(records: tuple[ExperimentExample, ...]) -> str:
    counts = Counter(_labels(records))
    return min(counts, key=lambda label: (-counts[label], label))


def _majority_results(data: ExperimentData) -> tuple[BaselineResult, ...]:
    results: list[BaselineResult] = []
    for task in (TaskName.FAULT_FAMILY, TaskName.NEXT_ACTION, TaskName.CONTINUE_LOG):
        train = examples_for_task(data.train, task)
        validation = examples_for_task(data.validation, task)
        started = time.perf_counter()
        prediction = _majority_label(train)
        predicted = tuple(prediction for _ in validation)
        metrics = classification_metrics(task, _labels(validation), predicted)
        results.append(
            _result(
                name="majority_frequency",
                task_name=task.value,
                parameter_count=0,
                elapsed_seconds=time.perf_counter() - started,
                classification=metrics,
            )
        )
    return tuple(results)


def _fault_rule(text: str, fallback: str) -> str:
    folded = text.casefold()
    transitions = folded.count("channel observation transition")
    if "changed from available to degraded" in folded:
        return "DIAGNOSED:PUMP_DEGRADATION"
    if "primary inventory" in folded:
        return "DIAGNOSED:ABSTRACT_INVENTORY_LOSS"
    if "secondary flow" in folded and "secondary inventory" in folded:
        return "DIAGNOSED:FLOW_IMBALANCE"
    if "transfer efficiency" in folded:
        return "DIAGNOSED:TRANSFER_EFFICIENCY_LOSS"
    if "redundant channel disagreement" in folded:
        return "DIAGNOSED:SENSOR_NOISE" if transitions >= 3 else "DIAGNOSED:SENSOR_DRIFT"
    if "[e-" not in folded:
        return "NO_FAULT"
    return fallback


def _action_rule(text: str, fallback: str) -> str:
    folded = text.casefold()
    transitions = folded.count("channel observation transition")
    if "changed from available to degraded" in folded:
        return "REQUEST_COMPONENT_INSPECTION"
    if "redundant channel disagreement" in folded:
        return "FLAG_SENSOR_SUSPECT" if transitions >= 3 else "VERIFY_REDUNDANT_CHANNEL"
    if "primary inventory" in folded or "transfer efficiency" in folded:
        return "REDUCE_SIMULATED_LOAD"
    if "secondary flow" in folded and "secondary inventory" in folded:
        return "COMPARE_RELATED_TRENDS"
    if "[e-" not in folded:
        return "CONTINUE_MONITORING"
    return fallback


def _rule_results(data: ExperimentData) -> tuple[BaselineResult, ...]:
    results: list[BaselineResult] = []
    for task, rule in (
        (TaskName.FAULT_FAMILY, _fault_rule),
        (TaskName.NEXT_ACTION, _action_rule),
    ):
        train = examples_for_task(data.train, task)
        validation = examples_for_task(data.validation, task)
        fallback = _majority_label(train)
        started = time.perf_counter()
        predicted = tuple(rule(record.prompt_text, fallback) for record in validation)
        metrics = classification_metrics(task, _labels(validation), predicted)
        results.append(
            _result(
                name="deterministic_keyword_rules",
                task_name=task.value,
                parameter_count=0,
                elapsed_seconds=time.perf_counter() - started,
                classification=metrics,
            )
        )
    return tuple(results)


def _words(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(text.casefold()))


def _next_event_ngram(data: ExperimentData, order: int) -> BaselineResult:
    task = TaskName.CONTINUE_LOG
    train = examples_for_task(data.train, task)
    validation = examples_for_task(data.validation, task)
    fallback = _majority_label(train)
    contexts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for record in train:
        words = _words(record.prompt_text)
        contexts[words[-(order - 1) :]][str(record.classification_label)] += 1
    started = time.perf_counter()
    predicted: list[str] = []
    for record in validation:
        key = _words(record.prompt_text)[-(order - 1) :]
        counts = contexts.get(key)
        predicted.append(
            fallback if not counts else min(counts, key=lambda label: (-counts[label], label))
        )
    metrics = classification_metrics(task, _labels(validation), tuple(predicted))
    return _result(
        name="word_ngram_suffix",
        task_name=task.value,
        parameter_count=sum(len(counts) for counts in contexts.values()),
        elapsed_seconds=time.perf_counter() - started,
        classification=metrics,
    )


def _token_ngram_language_model(
    train: tuple[TokenizedExample, ...],
    validation: tuple[TokenizedExample, ...],
    *,
    order: int,
    alpha: float,
    vocab_size: int,
) -> BaselineResult:
    contexts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    unigram: Counter[int] = Counter()
    for record in train:
        for position, is_target in enumerate(record.target_mask):
            if not is_target or position == 0:
                continue
            token = record.token_ids[position]
            context = record.token_ids[max(0, position - order + 1) : position]
            contexts[context][token] += 1
            unigram[token] += 1
    started = time.perf_counter()
    total_loss = 0.0
    target_count = 0
    unigram_total = sum(unigram.values())
    for record in validation:
        for position, is_target in enumerate(record.target_mask):
            if not is_target or position == 0:
                continue
            token = record.token_ids[position]
            context = record.token_ids[max(0, position - order + 1) : position]
            counts = contexts.get(context)
            if counts:
                probability = (counts[token] + alpha) / (sum(counts.values()) + alpha * vocab_size)
            else:
                probability = (unigram[token] + alpha) / (unigram_total + alpha * vocab_size)
            total_loss -= math.log(probability)
            target_count += 1
    metrics = language_model_metrics(
        sample_count=len(validation),
        target_token_count=target_count,
        negative_log_likelihood=total_loss / target_count,
    )
    return _result(
        name="token_trigram_additive",
        task_name="target_language_modeling",
        parameter_count=sum(len(counts) for counts in contexts.values()) + len(unigram),
        elapsed_seconds=time.perf_counter() - started,
        language_model=metrics,
    )


def _bow_matrix(
    records: tuple[ExperimentExample, ...], vocabulary: dict[str, int]
) -> NDArray[np.float64]:
    matrix = np.zeros((len(records), len(vocabulary) + 1), dtype=np.float64)
    matrix[:, -1] = 1.0
    for row, record in enumerate(records):
        counts = Counter(_words(record.prompt_text))
        for word, count in counts.items():
            column = vocabulary.get(word)
            if column is not None:
                matrix[row, column] = math.log1p(count)
    norms = np.linalg.norm(matrix[:, :-1], axis=1, keepdims=True)
    matrix[:, :-1] /= np.maximum(norms, 1.0)
    return matrix


def _bow_logistic(data: ExperimentData, config: BaselineConfig) -> BaselineResult:
    task = TaskName.FAULT_FAMILY
    train = examples_for_task(data.train, task)
    validation = examples_for_task(data.validation, task)
    frequencies = Counter(word for record in train for word in set(_words(record.prompt_text)))
    chosen = sorted(frequencies, key=lambda word: (-frequencies[word], word))[
        : config.bow_max_features
    ]
    vocabulary = {word: index for index, word in enumerate(chosen)}
    labels = tuple(sorted(set(_labels(train))))
    label_index = {label: index for index, label in enumerate(labels)}
    x_train = _bow_matrix(train, vocabulary)
    x_validation = _bow_matrix(validation, vocabulary)
    y = np.array([label_index[label] for label in _labels(train)], dtype=np.int64)
    weights = np.zeros((x_train.shape[1], len(labels)), dtype=np.float64)
    started = time.perf_counter()
    for _ in range(config.bow_steps):
        logits = x_train @ weights
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        probabilities[np.arange(len(train)), y] -= 1.0
        gradient = x_train.T @ probabilities / len(train)
        gradient[:-1] += config.bow_l2 * weights[:-1]
        weights -= config.bow_learning_rate * gradient
    predictions = tuple(labels[index] for index in np.argmax(x_validation @ weights, axis=1))
    metrics = classification_metrics(task, _labels(validation), predictions)
    return _result(
        name="bag_of_words_logistic_regression",
        task_name=task.value,
        parameter_count=weights.size,
        elapsed_seconds=time.perf_counter() - started,
        classification=metrics,
    )


class _GruClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_width: int, hidden_width: int, classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_width, padding_idx=PAD_ID)
        self.gru = nn.GRU(embedding_width, hidden_width, batch_first=True)
        self.output = nn.Linear(hidden_width, classes)

    def forward(self, input_ids: Tensor, lengths: Tensor) -> Tensor:
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return cast(Tensor, self.output(hidden[-1]))


def _prompt_token_rows(
    records: tuple[ExperimentExample, ...], tokenizer: ProjectTokenizer, maximum: int
) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for record in records:
        encoded = tokenizer.encode(record.prompt_text)
        rows.append((encoded[0], *encoded[-(maximum - 1) :]) if len(encoded) > maximum else encoded)
    return tuple(rows)


def _gru_result(
    data: ExperimentData,
    tokenizer: ProjectTokenizer,
    config: BaselineConfig,
    task: TaskName,
    *,
    seed: int,
) -> BaselineResult:
    train = examples_for_task(data.train, task)
    validation = examples_for_task(data.validation, task)
    labels = tuple(sorted(set(_labels(train))))
    label_index = {label: index for index, label in enumerate(labels)}
    train_rows = _prompt_token_rows(train, tokenizer, config.gru_max_tokens)
    validation_rows = _prompt_token_rows(validation, tokenizer, config.gru_max_tokens)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _GruClassifier(
            tokenizer.vocab_size,
            config.gru_embedding_width,
            config.gru_hidden_width,
            len(labels),
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.gru_learning_rate)
    rng = np.random.default_rng(seed)

    def batch(
        rows: tuple[tuple[int, ...], ...], indices: NDArray[np.int64]
    ) -> tuple[Tensor, Tensor]:
        width = max(len(rows[int(index)]) for index in indices)
        tokens = torch.full((len(indices), width), PAD_ID, dtype=torch.long)
        lengths = torch.empty(len(indices), dtype=torch.long)
        for position, index in enumerate(indices):
            row = rows[int(index)]
            tokens[position, : len(row)] = torch.tensor(row, dtype=torch.long)
            lengths[position] = len(row)
        return tokens, lengths

    started = time.perf_counter()
    model.train()
    for _ in range(config.gru_epochs):
        order = rng.permutation(len(train)).astype(np.int64)
        for start in range(0, len(order), config.gru_batch_size):
            indices = order[start : start + config.gru_batch_size]
            tokens, lengths = batch(train_rows, indices)
            targets = torch.tensor(
                [label_index[str(train[int(index)].classification_label)] for index in indices],
                dtype=torch.long,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(tokens, lengths), targets)
            torch.autograd.backward(loss)
            optimizer.step()
    model.eval()
    predicted: list[str] = []
    with torch.no_grad():
        for start in range(0, len(validation), config.gru_batch_size):
            indices = np.arange(
                start, min(start + config.gru_batch_size, len(validation)), dtype=np.int64
            )
            tokens, lengths = batch(validation_rows, indices)
            predicted.extend(
                labels[index] for index in model(tokens, lengths).argmax(dim=1).tolist()
            )
    metrics = classification_metrics(task, _labels(validation), tuple(predicted))
    return _result(
        name="gru_sequence_classifier",
        task_name=task.value,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        elapsed_seconds=time.perf_counter() - started,
        classification=metrics,
    )


def run_preregistered_baselines(
    data: ExperimentData,
    tokenizer: ProjectTokenizer,
    config: BaselineConfig,
    *,
    tokenized_train: tuple[TokenizedExample, ...],
    tokenized_validation: tuple[TokenizedExample, ...],
) -> tuple[BaselineResult, ...]:
    if type(data) is not ExperimentData or type(tokenizer) is not ProjectTokenizer:
        raise TypeError("baseline runner requires exact data and tokenizer objects")
    results = [*_majority_results(data), *_rule_results(data)]
    results.append(_next_event_ngram(data, config.ngram_order))
    results.append(
        _token_ngram_language_model(
            tokenized_train,
            tokenized_validation,
            order=config.ngram_order,
            alpha=config.ngram_additive_smoothing,
            vocab_size=tokenizer.vocab_size,
        )
    )
    results.append(_bow_logistic(data, config))
    results.append(_gru_result(data, tokenizer, config, TaskName.FAULT_FAMILY, seed=5511))
    results.append(_gru_result(data, tokenizer, config, TaskName.CONTINUE_LOG, seed=5512))
    return tuple(results)


__all__ = ["BaselineResult", "run_preregistered_baselines"]
