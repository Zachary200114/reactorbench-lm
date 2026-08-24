"""Preregistered simple-comparator adapter for compact remediation examples."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import Field, model_validator
from torch import Tensor

from reactorbench.evaluation.baselines import (
    BaselineResult,
    _bow_logistic,
    _bow_matrix,
    _gru_result,
    _GruClassifier,
    _labels,
    _majority_label,
    _majority_results,
    _next_event_ngram,
    _prompt_token_rows,
    _result,
    _rule_results,
    _token_ngram_language_model,
    _words,
    run_preregistered_baselines,
)
from reactorbench.evaluation.config import BaselineConfig
from reactorbench.evaluation.data import ExperimentData, ExperimentExample, examples_for_task
from reactorbench.evaluation.metrics import classification_metrics
from reactorbench.evaluation.serialization import TokenizedExample
from reactorbench.schemas.base import ContractModel, canonical_sha256
from reactorbench.schemas.enums import SplitName, TaskName
from reactorbench.tokenizer import PAD_ID, ProjectTokenizer

from .config import RemediationView
from .data import RemediationExample, SafeDevelopmentDataset
from .serialization import CompactTokenizedExample

StopCallback = Callable[[], bool]


def _poll_stop(stop_requested: StopCallback | None) -> None:
    """Raise cooperatively while rejecting ambiguous callback values."""

    if stop_requested is None:
        return
    requested = stop_requested()
    if type(requested) is not bool:
        raise TypeError("baseline stop callback must return an exact boolean")
    if requested:
        raise KeyboardInterrupt


class RemediationBaselineReport(ContractModel):
    report_version: str = "0.3.0"
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_view: RemediationView
    tokenizer_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_count: int = Field(ge=1, le=64)
    results: tuple[BaselineResult, ...]
    strongest_fault_comparator_macro_f1: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    strongest_action_comparator_macro_f1: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def results_and_checksum_match(self) -> RemediationBaselineReport:
        if self.evaluation_view is RemediationView.IID_TRAIN:
            raise ValueError("baseline report cannot evaluate its training view")
        if len(self.results) != self.result_count:
            raise ValueError("baseline report count differs from its results")
        fault = tuple(
            result.classification.macro_f1
            for result in self.results
            if result.task_name == TaskName.FAULT_FAMILY.value and result.classification is not None
        )
        action = tuple(
            result.classification.macro_f1
            for result in self.results
            if result.task_name == TaskName.NEXT_ACTION.value and result.classification is not None
        )
        if not fault or not action:
            raise ValueError("baseline report lacks required classification comparators")
        if self.strongest_fault_comparator_macro_f1 != max(
            fault
        ) or self.strongest_action_comparator_macro_f1 != max(action):
            raise ValueError("baseline report strongest-comparator values are inconsistent")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("baseline report checksum mismatch")
        return self


def _experiment_example(example: RemediationExample, *, split: SplitName) -> ExperimentExample:
    if split not in {SplitName.IID_TRAIN, SplitName.IID_VALIDATION}:
        raise ValueError("legacy baseline adapter supports only train/evaluation roles")
    return ExperimentExample(
        example_id=example.example_id,
        split_name=split,
        task_name=example.task_name,
        prompt_text=example.prompt_text,
        target_text=example.compact_target,
        classification_label=example.classification_label,
        source_checksum_sha256=example.checksum_sha256,
    )


def _legacy_tokens(example: CompactTokenizedExample) -> TokenizedExample:
    return TokenizedExample(
        example_id=example.example_id,
        token_ids=example.token_ids,
        target_mask=example.target_mask,
        truncated_prompt=example.prompt_truncated,
    )


def _cooperative_bow_logistic(
    data: ExperimentData,
    config: BaselineConfig,
    *,
    stop_requested: StopCallback,
) -> BaselineResult:
    """Run the frozen BOW comparator with one stop boundary per optimizer step."""

    _poll_stop(stop_requested)
    task = TaskName.FAULT_FAMILY
    train = examples_for_task(data.train, task)
    validation = examples_for_task(data.validation, task)
    frequencies: dict[str, int] = {}
    for record in train:
        for word in set(_words(record.prompt_text)):
            frequencies[word] = frequencies.get(word, 0) + 1
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
        _poll_stop(stop_requested)
        logits = x_train @ weights
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        probabilities[np.arange(len(train)), y] -= 1.0
        gradient = x_train.T @ probabilities / len(train)
        gradient[:-1] += config.bow_l2 * weights[:-1]
        weights -= config.bow_learning_rate * gradient
    _poll_stop(stop_requested)
    predictions = tuple(labels[index] for index in np.argmax(x_validation @ weights, axis=1))
    metrics = classification_metrics(task, _labels(validation), predictions)
    return _result(
        name="bag_of_words_logistic_regression",
        task_name=task.value,
        parameter_count=weights.size,
        elapsed_seconds=time.perf_counter() - started,
        classification=metrics,
    )


def _cooperative_gru_result(
    data: ExperimentData,
    tokenizer: ProjectTokenizer,
    config: BaselineConfig,
    task: TaskName,
    *,
    seed: int,
    stop_requested: StopCallback,
) -> BaselineResult:
    """Run the frozen GRU comparator with stop boundaries at every batch."""

    _poll_stop(stop_requested)
    train = examples_for_task(data.train, task)
    validation = examples_for_task(data.validation, task)
    labels = tuple(sorted(set(_labels(train))))
    label_index = {label: index for index, label in enumerate(labels)}
    train_rows = _prompt_token_rows(train, tokenizer, config.gru_max_tokens)
    validation_rows = _prompt_token_rows(validation, tokenizer, config.gru_max_tokens)
    _poll_stop(stop_requested)
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
            _poll_stop(stop_requested)
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
            _poll_stop(stop_requested)
            indices = np.arange(
                start, min(start + config.gru_batch_size, len(validation)), dtype=np.int64
            )
            tokens, lengths = batch(validation_rows, indices)
            predicted.extend(
                labels[index] for index in model(tokens, lengths).argmax(dim=1).tolist()
            )
    _poll_stop(stop_requested)
    metrics = classification_metrics(task, _labels(validation), tuple(predicted))
    return _result(
        name="gru_sequence_classifier",
        task_name=task.value,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        elapsed_seconds=time.perf_counter() - started,
        classification=metrics,
    )


def _cooperative_full_matrix(
    data: ExperimentData,
    tokenizer: ProjectTokenizer,
    config: BaselineConfig,
    *,
    tokenized_train: tuple[TokenizedExample, ...],
    tokenized_validation: tuple[TokenizedExample, ...],
    stop_requested: StopCallback,
) -> tuple[BaselineResult, ...]:
    """Execute the existing comparator matrix with boundaries between fits."""

    results: list[BaselineResult] = []
    _poll_stop(stop_requested)
    results.extend(_majority_results(data))
    _poll_stop(stop_requested)
    results.extend(_rule_results(data))
    _poll_stop(stop_requested)
    results.append(_next_event_ngram(data, config.ngram_order))
    _poll_stop(stop_requested)
    results.append(
        _token_ngram_language_model(
            tokenized_train,
            tokenized_validation,
            order=config.ngram_order,
            alpha=config.ngram_additive_smoothing,
            vocab_size=tokenizer.vocab_size,
        )
    )
    _poll_stop(stop_requested)
    results.append(_cooperative_bow_logistic(data, config, stop_requested=stop_requested))
    _poll_stop(stop_requested)
    results.append(
        _cooperative_gru_result(
            data,
            tokenizer,
            config,
            TaskName.FAULT_FAMILY,
            seed=5511,
            stop_requested=stop_requested,
        )
    )
    _poll_stop(stop_requested)
    results.append(
        _cooperative_gru_result(
            data,
            tokenizer,
            config,
            TaskName.CONTINUE_LOG,
            seed=5512,
            stop_requested=stop_requested,
        )
    )
    _poll_stop(stop_requested)
    return tuple(results)


def _baselines_with_optional_continuation(
    data: ExperimentData,
    tokenizer: ProjectTokenizer,
    config: BaselineConfig,
    *,
    tokenized_train: tuple[TokenizedExample, ...],
    tokenized_validation: tuple[TokenizedExample, ...],
    stop_requested: StopCallback | None = None,
) -> tuple[BaselineResult, ...]:
    """Preserve the matrix while marking an absent continuation task as N/A."""

    validation_tasks = {item.task_name for item in data.validation}
    required = {
        TaskName.FAULT_FAMILY,
        TaskName.NEXT_ACTION,
        TaskName.CONTINUE_LOG,
    }
    if required.issubset(validation_tasks):
        if stop_requested is not None:
            return _cooperative_full_matrix(
                data,
                tokenizer,
                config,
                tokenized_train=tokenized_train,
                tokenized_validation=tokenized_validation,
                stop_requested=stop_requested,
            )
        return run_preregistered_baselines(
            data,
            tokenizer,
            config,
            tokenized_train=tokenized_train,
            tokenized_validation=tokenized_validation,
        )
    if not {TaskName.FAULT_FAMILY, TaskName.NEXT_ACTION}.issubset(validation_tasks):
        raise ValueError("evaluation view lacks a required diagnosis or action comparator task")
    results: list[BaselineResult] = []
    for task in (TaskName.FAULT_FAMILY, TaskName.NEXT_ACTION):
        _poll_stop(stop_requested)
        train = examples_for_task(data.train, task)
        validation = examples_for_task(data.validation, task)
        prediction = _majority_label(train)
        metrics = classification_metrics(
            task,
            _labels(validation),
            tuple(prediction for _ in validation),
        )
        results.append(
            _result(
                name="majority_frequency",
                task_name=task.value,
                parameter_count=0,
                elapsed_seconds=0.0,
                classification=metrics,
            )
        )
        _poll_stop(stop_requested)
    _poll_stop(stop_requested)
    results.extend(_rule_results(data))
    _poll_stop(stop_requested)
    results.append(
        _token_ngram_language_model(
            tokenized_train,
            tokenized_validation,
            order=config.ngram_order,
            alpha=config.ngram_additive_smoothing,
            vocab_size=tokenizer.vocab_size,
        )
    )
    _poll_stop(stop_requested)
    results.append(
        _bow_logistic(data, config)
        if stop_requested is None
        else _cooperative_bow_logistic(data, config, stop_requested=stop_requested)
    )
    _poll_stop(stop_requested)
    results.append(
        _gru_result(data, tokenizer, config, TaskName.FAULT_FAMILY, seed=5511)
        if stop_requested is None
        else _cooperative_gru_result(
            data,
            tokenizer,
            config,
            TaskName.FAULT_FAMILY,
            seed=5511,
            stop_requested=stop_requested,
        )
    )
    _poll_stop(stop_requested)
    return tuple(results)


def run_remediation_baselines(
    dataset: SafeDevelopmentDataset,
    tokenizer: ProjectTokenizer,
    baseline_config: BaselineConfig,
    *,
    tokenized_train: tuple[CompactTokenizedExample, ...],
    tokenized_validation: tuple[CompactTokenizedExample, ...],
    evaluation_view: RemediationView = RemediationView.IID_VALIDATION,
    stop_requested: StopCallback | None = None,
) -> RemediationBaselineReport:
    """Run the existing majority/rule/N-gram/BOW/GRU comparator matrix."""

    if (
        type(dataset) is not SafeDevelopmentDataset
        or type(tokenizer) is not ProjectTokenizer
        or type(baseline_config) is not BaselineConfig
    ):
        raise TypeError("baseline adapter requires exact project contracts")
    train_source = tuple(
        item for item in dataset.examples if item.view is RemediationView.IID_TRAIN
    )
    if type(evaluation_view) is not RemediationView or evaluation_view is RemediationView.IID_TRAIN:
        raise ValueError("baseline evaluation view must be a validation or shadow view")
    validation_source = tuple(item for item in dataset.examples if item.view is evaluation_view)
    if not train_source or not validation_source:
        raise ValueError("baseline adapter requires IID train and validation examples")
    if {item.example_id for item in tokenized_train} != {
        item.example_id for item in train_source
    } or {item.example_id for item in tokenized_validation} != {
        item.example_id for item in validation_source
    }:
        raise ValueError("baseline token inventories differ from the compact dataset")
    data = ExperimentData(
        train=tuple(_experiment_example(item, split=SplitName.IID_TRAIN) for item in train_source),
        validation=tuple(
            _experiment_example(item, split=SplitName.IID_VALIDATION) for item in validation_source
        ),
        inventory_sha256=canonical_sha256(
            tuple((item.example_id, item.checksum_sha256) for item in dataset.examples)
        ),
    )
    legacy_train = tuple(_legacy_tokens(item) for item in tokenized_train)
    legacy_validation = tuple(_legacy_tokens(item) for item in tokenized_validation)
    results = _baselines_with_optional_continuation(
        data,
        tokenizer,
        baseline_config,
        tokenized_train=legacy_train,
        tokenized_validation=legacy_validation,
        stop_requested=stop_requested,
    )
    fault = tuple(
        result.classification.macro_f1
        for result in results
        if result.task_name == TaskName.FAULT_FAMILY.value and result.classification is not None
    )
    action = tuple(
        result.classification.macro_f1
        for result in results
        if result.task_name == TaskName.NEXT_ACTION.value and result.classification is not None
    )
    if not fault or not action:
        raise RuntimeError("preregistered baseline matrix omitted a required comparator")
    draft = RemediationBaselineReport.model_construct(
        dataset_manifest_sha256=dataset.manifest.checksum_sha256,
        evaluation_view=evaluation_view,
        tokenizer_manifest_sha256=tokenizer.manifest.checksum_sha256,
        baseline_config_sha256=canonical_sha256(
            baseline_config.model_dump(mode="json", round_trip=True)
        ),
        result_count=len(results),
        results=results,
        strongest_fault_comparator_macro_f1=max(fault),
        strongest_action_comparator_macro_f1=max(action),
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return RemediationBaselineReport(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
        checksum_sha256=checksum,
    )


__all__ = ["RemediationBaselineReport", "run_remediation_baselines"]
