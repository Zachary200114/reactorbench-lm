from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from pydantic import ValidationError

from reactorbench.dataset import VerifiedDevelopmentCandidateArtifact
from reactorbench.dataset.contracts import PromptContinuationTarget
from reactorbench.evaluation import (
    ExperimentExample,
    Phase5Config,
    batch_tensors,
    classification_metrics,
    examples_for_task,
    load_phase5_config,
    materialize_experiment_data,
    supervised_causal_loss,
    tokenize_example,
)
from reactorbench.model import TransformerConfig, initialized_model
from reactorbench.schemas.enums import (
    ActionLabel,
    DiagnosisStatus,
    EventType,
    FaultFamily,
    SplitName,
    TaskName,
)
from reactorbench.schemas.target import FaultDiagnosisTarget, NextActionTarget
from reactorbench.tokenizer import ProjectTokenizer

CONFIG_PATH = Path("configs/experiments/phase5-pilot-v0.1.0.toml")


def _fake_tokenizer(monkeypatch: pytest.MonkeyPatch) -> ProjectTokenizer:
    tokenizer = object.__new__(ProjectTokenizer)

    def encode(
        _self: ProjectTokenizer, text: str, *, add_bos: bool = True, add_eos: bool = True
    ) -> tuple[int, ...]:
        values = tuple(4 + ord(character) % 60 for character in text)
        return (*((1,) if add_bos else ()), *values, *((2,) if add_eos else ()))

    monkeypatch.setattr(ProjectTokenizer, "encode", encode)
    return tokenizer


def _example(prompt: str = "fictional prompt evidence") -> ExperimentExample:
    return ExperimentExample(
        example_id="example:test",
        split_name=SplitName.IID_TRAIN,
        task_name=TaskName.NEXT_ACTION,
        prompt_text=prompt,
        target_text='{"immediate_action":"CONTINUE_MONITORING","task_name":"next_action"}',
        classification_label="CONTINUE_MONITORING",
        source_checksum_sha256="a" * 64,
    )


def test_phase5_config_loads_the_preregistered_contract() -> None:
    config = load_phase5_config(CONFIG_PATH)

    assert config.pilot_transformer.steps == 500
    assert config.smaller_transformer.steps == 300
    assert config.data.train_split is SplitName.IID_TRAIN
    assert config.data.validation_split is SplitName.IID_VALIDATION
    assert config.data.classification_tasks == (
        TaskName.FAULT_FAMILY,
        TaskName.NEXT_ACTION,
        TaskName.CONTINUE_LOG,
    )


def test_phase5_config_rejects_coercion_unknown_fields_and_split_omissions() -> None:
    payload = load_phase5_config(CONFIG_PATH).model_dump(mode="python", round_trip=True)
    cast(dict[str, object], payload["pilot_transformer"])["steps"] = "500"
    with pytest.raises(ValidationError, match="int_type"):
        Phase5Config.model_validate(payload)

    payload = load_phase5_config(CONFIG_PATH).model_dump(mode="python", round_trip=True)
    prohibited = cast(
        tuple[object, ...], cast(dict[str, object], payload["data"])["prohibited_splits"]
    )
    cast(dict[str, object], payload["data"])["prohibited_splits"] = prohibited[:-1]
    with pytest.raises(ValidationError, match="every non-train"):
        Phase5Config.model_validate(payload)

    payload = load_phase5_config(CONFIG_PATH).model_dump(mode="python", round_trip=True)
    payload["unknown"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Phase5Config.model_validate(payload)


def test_classification_metrics_are_exact_and_checksum_bound() -> None:
    metrics = classification_metrics(
        TaskName.FAULT_FAMILY,
        ("A", "A", "B", "B"),
        ("A", "B", "B", "B"),
    )

    assert metrics.accuracy == 0.75
    assert metrics.labels == ("A", "B")
    assert metrics.confusion_matrix == ((1, 1), (0, 2))
    assert metrics.macro_f1 == pytest.approx((2 / 3 + 0.8) / 2)
    tampered = metrics.model_copy(update={"accuracy": 1.0})
    with pytest.raises(ValidationError, match="checksum"):
        type(metrics).model_validate(tampered.model_dump(mode="python"))


def test_target_only_serialization_truncates_prompt_and_masks_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _fake_tokenizer(monkeypatch)
    config = load_phase5_config(CONFIG_PATH)
    tokenized = tokenize_example(
        _example("long fictional prompt " * 40),
        tokenizer,
        config.serialization,
        context_length=128,
    )

    assert tokenized.truncated_prompt
    assert tokenized.token_ids[0] == 1
    assert any(tokenized.target_mask)
    assert not tokenized.target_mask[0]
    inputs, attention, targets = batch_tensors((tokenized,), context_length=128)
    model_config = TransformerConfig(
        model_version="0.1.0",
        layers=1,
        width=16,
        heads=4,
        context_length=128,
        feed_forward_multiplier=2,
        dropout=0.0,
        tie_embeddings=True,
        bias=True,
    )
    model = initialized_model(model_config, vocab_size=64, seed=7)
    loss = supervised_causal_loss(model, inputs, attention, targets)

    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="target mask"):
        supervised_causal_loss(model, inputs, attention, targets[:, :-1])


def test_experiment_materialization_keeps_only_train_and_validation() -> None:
    targets = (
        (
            TaskName.FAULT_FAMILY,
            FaultDiagnosisTarget(
                diagnosis_status=DiagnosisStatus.NO_FAULT,
                fault_labels=(),
                abstention_reason=None,
            ),
        ),
        (
            TaskName.FAULT_FAMILY,
            FaultDiagnosisTarget(
                diagnosis_status=DiagnosisStatus.DIAGNOSED,
                fault_labels=(FaultFamily.SENSOR_DRIFT,),
                abstention_reason=None,
            ),
        ),
        (
            TaskName.NEXT_ACTION,
            NextActionTarget(immediate_action=ActionLabel.CONTINUE_MONITORING),
        ),
        (
            TaskName.CONTINUE_LOG,
            PromptContinuationTarget(next_event_type=EventType.OPERATING_MODE_CHANGED),
        ),
        (
            TaskName.NEXT_ACTION,
            NextActionTarget(immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE),
        ),
    )
    renders: list[SimpleNamespace] = []
    examples: list[SimpleNamespace] = []
    for index, split in enumerate(
        (
            SplitName.IID_TRAIN,
            SplitName.IID_TRAIN,
            SplitName.IID_VALIDATION,
            SplitName.IID_VALIDATION,
            SplitName.IID_TEST,
        )
    ):
        task, target = targets[index]
        render_id = f"render-{index}"
        renders.append(
            SimpleNamespace(render_id=render_id, split_name=split, text=f"prompt {index}")
        )
        examples.append(
            SimpleNamespace(
                example_id=f"example-{index}",
                split_name=split,
                prompt_render_ids=(render_id,),
                corruption_ids=(),
                task_target=SimpleNamespace(target=target),
                task_name=task,
                checksum_sha256=f"{index + 1}" * 64,
            )
        )
    verified = VerifiedDevelopmentCandidateArtifact(
        manifest=cast(Any, None),
        metadata=cast(Any, None),
        candidate=cast(
            Any,
            SimpleNamespace(rendered_candidates=tuple(renders), task_examples=tuple(examples)),
        ),
    )

    data = materialize_experiment_data(verified, maximum_prompt_utf8_bytes=1024)

    assert len(data.train) == 2
    assert len(data.validation) == 2
    assert data.train[0].classification_label == "NO_FAULT"
    assert data.train[1].classification_label == "DIAGNOSED:SENSOR_DRIFT"
    assert {record.classification_label for record in data.validation} == {
        "CONTINUE_MONITORING",
        "OPERATING_MODE_CHANGED",
    }
    assert len(examples_for_task(data.validation, TaskName.CONTINUE_LOG)) == 1
    with pytest.raises(TypeError, match="exact record tuple"):
        examples_for_task(cast(Any, list(data.validation)), TaskName.CONTINUE_LOG)
    assert all(
        record.split_name is not SplitName.IID_TEST for record in data.train + data.validation
    )
