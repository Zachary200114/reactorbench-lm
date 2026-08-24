"""Preregistered simple-comparator adapter for compact remediation examples."""

from __future__ import annotations

from pydantic import Field, model_validator

from reactorbench.evaluation.baselines import (
    BaselineResult,
    _bow_logistic,
    _gru_result,
    _labels,
    _majority_label,
    _result,
    _rule_results,
    _token_ngram_language_model,
    run_preregistered_baselines,
)
from reactorbench.evaluation.config import BaselineConfig
from reactorbench.evaluation.data import ExperimentData, ExperimentExample, examples_for_task
from reactorbench.evaluation.metrics import classification_metrics
from reactorbench.evaluation.serialization import TokenizedExample
from reactorbench.schemas.base import ContractModel, canonical_sha256
from reactorbench.schemas.enums import SplitName, TaskName
from reactorbench.tokenizer import ProjectTokenizer

from .config import RemediationView
from .data import RemediationExample, SafeDevelopmentDataset
from .serialization import CompactTokenizedExample


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


def _baselines_with_optional_continuation(
    data: ExperimentData,
    tokenizer: ProjectTokenizer,
    config: BaselineConfig,
    *,
    tokenized_train: tuple[TokenizedExample, ...],
    tokenized_validation: tuple[TokenizedExample, ...],
) -> tuple[BaselineResult, ...]:
    """Preserve the matrix while marking an absent continuation task as N/A."""

    validation_tasks = {item.task_name for item in data.validation}
    required = {
        TaskName.FAULT_FAMILY,
        TaskName.NEXT_ACTION,
        TaskName.CONTINUE_LOG,
    }
    if required.issubset(validation_tasks):
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
    results.extend(_rule_results(data))
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
    return tuple(results)


def run_remediation_baselines(
    dataset: SafeDevelopmentDataset,
    tokenizer: ProjectTokenizer,
    baseline_config: BaselineConfig,
    *,
    tokenized_train: tuple[CompactTokenizedExample, ...],
    tokenized_validation: tuple[CompactTokenizedExample, ...],
    evaluation_view: RemediationView = RemediationView.IID_VALIDATION,
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
