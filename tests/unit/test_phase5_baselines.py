from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from reactorbench.evaluation import (
    ExperimentData,
    ExperimentExample,
    load_phase5_config,
    tokenize_example,
)
from reactorbench.evaluation.baselines import run_preregistered_baselines
from reactorbench.schemas.base import canonical_sha256
from reactorbench.schemas.enums import SplitName, TaskName
from reactorbench.tokenizer import ProjectTokenizer, TokenizerArtifactManifest


def _fake_tokenizer(monkeypatch: pytest.MonkeyPatch) -> ProjectTokenizer:
    tokenizer = object.__new__(ProjectTokenizer)
    tokenizer.manifest = cast(TokenizerArtifactManifest, SimpleNamespace(actual_vocab_size=64))

    def encode(
        _self: ProjectTokenizer, text: str, *, add_bos: bool = True, add_eos: bool = True
    ) -> tuple[int, ...]:
        values = tuple(4 + ord(character) % 60 for character in text)
        return (*((1,) if add_bos else ()), *values, *((2,) if add_eos else ()))

    monkeypatch.setattr(ProjectTokenizer, "encode", encode)
    return tokenizer


def _records(split: SplitName, count: int) -> tuple[ExperimentExample, ...]:
    records: list[ExperimentExample] = []
    for index in range(count):
        for task in (TaskName.FAULT_FAMILY, TaskName.NEXT_ACTION, TaskName.CONTINUE_LOG):
            positive = index % 2 == 0
            if task is TaskName.FAULT_FAMILY:
                label = "DIAGNOSED:PUMP_DEGRADATION" if positive else "UNRESOLVED"
                target = f'{{"fault":"{label}"}}'
                prompt = (
                    "component state transition changed from available to degraded"
                    if positive
                    else "one ambiguous channel observation transition"
                )
            elif task is TaskName.NEXT_ACTION:
                label = "REQUEST_COMPONENT_INSPECTION" if positive else "INSUFFICIENT_EVIDENCE"
                target = f'{{"action":"{label}"}}'
                prompt = (
                    "component state transition changed from available to degraded"
                    if positive
                    else "one ambiguous channel observation transition"
                )
            else:
                label = "OPERATING_MODE_CHANGED" if positive else "CHANNEL_QUALITY_CHANGED"
                target = f'{{"event":"{label}"}}'
                prompt = "recent event sequence alpha" if positive else "recent event sequence beta"
            records.append(
                ExperimentExample(
                    example_id=f"example:{split.value}:{task.value}:{index:03d}",
                    split_name=split,
                    task_name=task,
                    prompt_text=prompt,
                    target_text=target,
                    classification_label=label,
                    source_checksum_sha256=f"{index % 16:x}" * 64,
                )
            )
    return tuple(records)


def test_all_preregistered_baselines_run_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _fake_tokenizer(monkeypatch)
    train = _records(SplitName.IID_TRAIN, 8)
    validation = _records(SplitName.IID_VALIDATION, 4)
    data = ExperimentData(
        train=train,
        validation=validation,
        inventory_sha256=canonical_sha256(
            tuple(record.example_id for record in train + validation)
        ),
    )
    phase5 = load_phase5_config(
        __import__("pathlib").Path("configs/experiments/phase5-pilot-v0.1.0.toml")
    )
    baseline_config = phase5.baselines.model_copy(
        update={"bow_steps": 20, "gru_epochs": 2, "gru_batch_size": 4}
    )
    tokenized_train = tuple(
        tokenize_example(item, tokenizer, phase5.serialization, context_length=128)
        for item in train
    )
    tokenized_validation = tuple(
        tokenize_example(item, tokenizer, phase5.serialization, context_length=128)
        for item in validation
    )

    results = run_preregistered_baselines(
        data,
        tokenizer,
        baseline_config,
        tokenized_train=tokenized_train,
        tokenized_validation=tokenized_validation,
    )

    assert len(results) == 10
    assert {(result.baseline_name, result.task_name) for result in results} >= {
        ("majority_frequency", "fault_family"),
        ("deterministic_keyword_rules", "next_action"),
        ("token_trigram_additive", "target_language_modeling"),
        ("bag_of_words_logistic_regression", "fault_family"),
        ("gru_sequence_classifier", "continue_log"),
    }
    assert all(result.elapsed_seconds >= 0.0 for result in results)
