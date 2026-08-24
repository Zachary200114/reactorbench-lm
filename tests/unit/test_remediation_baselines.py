"""Focused contract tests for the compact remediation baseline adapter."""

from __future__ import annotations

import hashlib
from typing import Any, cast

import pytest
from pydantic import ValidationError

import reactorbench.remediation.baselines as remediation_baselines
from reactorbench.evaluation.baselines import BaselineResult, _result
from reactorbench.evaluation.compact import (
    CompactTargetContext,
    compact_target_json,
    serialize_compact_target,
)
from reactorbench.evaluation.config import BaselineConfig
from reactorbench.evaluation.data import ExperimentData, ExperimentExample, examples_for_task
from reactorbench.evaluation.metrics import (
    classification_metrics,
    language_model_metrics,
)
from reactorbench.remediation.baselines import (
    RemediationBaselineReport,
    _baselines_with_optional_continuation,
    run_remediation_baselines,
)
from reactorbench.remediation.config import VIEW_SOURCE_SPLIT, RemediationView
from reactorbench.remediation.data import (
    RemediationExample,
    SafeDevelopmentDataset,
    SafeDevelopmentManifest,
)
from reactorbench.remediation.serialization import CompactTokenizedExample
from reactorbench.schemas.base import canonical_json_bytes, canonical_sha256
from reactorbench.schemas.enums import (
    ActionLabel,
    DiagnosisStatus,
    FaultFamily,
    TaskName,
)
from reactorbench.schemas.target import FaultDiagnosisTarget, NextActionTarget
from reactorbench.tokenizer import ProjectTokenizer
from reactorbench.tokenizer.core import TokenizerArtifactManifest, TrainingCorpusManifest


def _target(task: TaskName, variant: int) -> tuple[FaultDiagnosisTarget | NextActionTarget, str]:
    if task is TaskName.FAULT_FAMILY:
        if variant == 0:
            target = FaultDiagnosisTarget(diagnosis_status=DiagnosisStatus.NO_FAULT)
            return target, DiagnosisStatus.NO_FAULT.value
        target = FaultDiagnosisTarget(
            diagnosis_status=DiagnosisStatus.DIAGNOSED,
            fault_labels=(FaultFamily.SENSOR_DRIFT,),
        )
        return target, "DIAGNOSED:SENSOR_DRIFT"
    if task is TaskName.NEXT_ACTION:
        action = (
            ActionLabel.CONTINUE_MONITORING
            if variant == 0
            else ActionLabel.VERIFY_REDUNDANT_CHANNEL
        )
        return NextActionTarget(immediate_action=action), action.value
    raise AssertionError(f"unsupported task: {task}")


def _example(
    *,
    index: int,
    view: RemediationView,
    task: TaskName,
    variant: int,
) -> RemediationExample:
    target, label = _target(task, variant)
    context = CompactTargetContext(task_name=task, visible_fact_refs=("o-0000",))
    compact = serialize_compact_target(target, context=context)
    prompt = f"[o-0000] fictional {view.value} {task.value} variant {variant}"
    values: dict[str, Any] = {
        "artifact_version": "0.3.0",
        "example_id": f"baseline:{index:04d}",
        "view": view,
        "source_split": VIEW_SOURCE_SPLIT[view],
        "task_name": task,
        "group_id": f"baseline-group:{index:04d}",
        "source_record_ids": (f"projection:{index:04d}",),
        "parent_record_sha256": f"{index + 1:064x}",
        "prompt_text": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "template_family_id": "compact-log-v1",
        "alias_family_id": "canonical-v1",
        "compact_context": context,
        "compact_target": compact,
        "canonical_target_json": compact_target_json(compact, context=context),
        "classification_label": label,
        "augmentation": "none",
    }
    draft = RemediationExample.model_construct(**values, checksum_sha256="0" * 64)
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return RemediationExample(**values, checksum_sha256=checksum)


def _dataset() -> SafeDevelopmentDataset:
    examples = tuple(
        sorted(
            (
                _example(
                    index=0,
                    view=RemediationView.IID_TRAIN,
                    task=TaskName.FAULT_FAMILY,
                    variant=0,
                ),
                _example(
                    index=1,
                    view=RemediationView.IID_TRAIN,
                    task=TaskName.FAULT_FAMILY,
                    variant=1,
                ),
                _example(
                    index=2,
                    view=RemediationView.IID_TRAIN,
                    task=TaskName.NEXT_ACTION,
                    variant=0,
                ),
                _example(
                    index=3,
                    view=RemediationView.IID_TRAIN,
                    task=TaskName.NEXT_ACTION,
                    variant=1,
                ),
                _example(
                    index=4,
                    view=RemediationView.SHADOW_COMPONENT,
                    task=TaskName.FAULT_FAMILY,
                    variant=0,
                ),
                _example(
                    index=5,
                    view=RemediationView.SHADOW_COMPONENT,
                    task=TaskName.FAULT_FAMILY,
                    variant=1,
                ),
                _example(
                    index=6,
                    view=RemediationView.SHADOW_COMPONENT,
                    task=TaskName.NEXT_ACTION,
                    variant=0,
                ),
                _example(
                    index=7,
                    view=RemediationView.SHADOW_COMPONENT,
                    task=TaskName.NEXT_ACTION,
                    variant=1,
                ),
            ),
            key=lambda item: item.example_id,
        )
    )
    inventory = canonical_sha256(
        tuple((item.example_id, item.checksum_sha256) for item in examples)
    )
    payload = b"".join(
        canonical_json_bytes(item.model_dump(mode="json", round_trip=True)) + b"\n"
        for item in examples
    )
    values: dict[str, Any] = {
        "artifact_version": "0.3.0",
        "boundary": "development_only_no_final_or_golden_payloads",
        "source_commit": "abcdef0",
        "dataset_version": "test-v0.3",
        "dataset_config_sha256": "a" * 64,
        "compact_contract_version": "0.2.0",
        "views": (
            RemediationView.IID_TRAIN,
            RemediationView.SHADOW_COMPONENT,
        ),
        "example_count": len(examples),
        "counts_by_view": (
            (RemediationView.IID_TRAIN, 4),
            (RemediationView.SHADOW_COMPONENT, 4),
        ),
        "counts_by_task": (
            (TaskName.FAULT_FAMILY, 4),
            (TaskName.NEXT_ACTION, 4),
        ),
        "examples_sha256": hashlib.sha256(payload).hexdigest(),
        "examples_size_bytes": len(payload),
        "inventory_sha256": inventory,
    }
    draft = SafeDevelopmentManifest.model_construct(**values, checksum_sha256="0" * 64)
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    manifest = SafeDevelopmentManifest(**values, checksum_sha256=checksum)
    return SafeDevelopmentDataset(manifest=manifest, examples=examples)


def _tokenizer() -> ProjectTokenizer:
    corpus = TrainingCorpusManifest(
        candidate_bundle_sha256="1" * 64,
        candidate_artifact_manifest_sha256="2" * 64,
        postrender_packet_sha256="3" * 64,
        postrender_approval_record_sha256="4" * 64,
        document_count=1,
        utf8_bytes=1,
        document_inventory_sha256="5" * 64,
        corpus_sha256="6" * 64,
    )
    values: dict[str, Any] = {
        "artifact_version": "0.1.0",
        "tokenizer_version": "0.1.0",
        "algorithm": "sentencepiece_bpe",
        "sentencepiece_version": "test",
        "requested_vocab_size": 512,
        "actual_vocab_size": 512,
        "unk_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 3,
        "special_symbols": ("<|prompt|>", "<|target|>", "<|sep|>"),
        "model_sha256": "7" * 64,
        "vocab_sha256": "8" * 64,
        "model_size_bytes": 1,
        "vocab_size_bytes": 1,
        "corpus": corpus,
    }
    draft = TokenizerArtifactManifest.model_construct(**values, checksum_sha256="0" * 64)
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    manifest = TokenizerArtifactManifest(**values, checksum_sha256=checksum)
    return ProjectTokenizer(cast(Any, object()), manifest)


def _config() -> BaselineConfig:
    return BaselineConfig(
        ngram_order=3,
        ngram_additive_smoothing=0.1,
        bow_max_features=128,
        bow_steps=10,
        bow_learning_rate=0.1,
        bow_l2=0.0,
        gru_embedding_width=8,
        gru_hidden_width=8,
        gru_epochs=1,
        gru_batch_size=2,
        gru_learning_rate=0.1,
        gru_max_tokens=8,
    )


def _tokenized(example: RemediationExample) -> CompactTokenizedExample:
    return CompactTokenizedExample(
        example_id=example.example_id,
        task_name=example.task_name,
        group_id=example.group_id,
        token_ids=(1, 4, 2),
        target_mask=(False, True, True),
        prompt_token_count=2,
        target_token_count=2,
        prompt_tokens_retained=2,
        prompt_truncated=False,
    )


def _perfect_classification_result(
    data: ExperimentData,
    *,
    task: TaskName,
    name: str,
) -> BaselineResult:
    records = examples_for_task(data.validation, task)
    labels = tuple(cast(str, item.classification_label) for item in records)
    return _result(
        name=name,
        task_name=task.value,
        parameter_count=1,
        elapsed_seconds=0.0,
        classification=classification_metrics(task, labels, labels),
    )


def test_public_baseline_wrapper_supports_component_view_without_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset()
    train = tuple(
        _tokenized(item) for item in dataset.examples if item.view is RemediationView.IID_TRAIN
    )
    validation = tuple(
        _tokenized(item)
        for item in dataset.examples
        if item.view is RemediationView.SHADOW_COMPONENT
    )
    observed_gru_tasks: list[TaskName] = []

    def fake_ngram(*_args: object, **_kwargs: object) -> BaselineResult:
        return _result(
            name="token_ngram_language_model",
            task_name="all_targets",
            parameter_count=1,
            elapsed_seconds=0.0,
            language_model=language_model_metrics(
                sample_count=len(validation),
                target_token_count=len(validation) * 2,
                negative_log_likelihood=0.5,
            ),
        )

    def fake_bow(data: ExperimentData, _config: BaselineConfig) -> BaselineResult:
        return _perfect_classification_result(
            data,
            task=TaskName.FAULT_FAMILY,
            name="bag_of_words_logistic_regression",
        )

    def fake_gru(
        data: ExperimentData,
        _tokenizer: ProjectTokenizer,
        _config: BaselineConfig,
        task: TaskName,
        *,
        seed: int,
    ) -> BaselineResult:
        assert seed == 5511
        observed_gru_tasks.append(task)
        return _perfect_classification_result(
            data,
            task=task,
            name="gru_sequence_classifier",
        )

    monkeypatch.setattr(remediation_baselines, "_token_ngram_language_model", fake_ngram)
    monkeypatch.setattr(remediation_baselines, "_bow_logistic", fake_bow)
    monkeypatch.setattr(remediation_baselines, "_gru_result", fake_gru)
    monkeypatch.setattr(
        remediation_baselines,
        "run_preregistered_baselines",
        lambda *_args, **_kwargs: pytest.fail(
            "continuation-required legacy matrix must not run for component shadow"
        ),
    )

    report = run_remediation_baselines(
        dataset,
        _tokenizer(),
        _config(),
        tokenized_train=train,
        tokenized_validation=validation,
        evaluation_view=RemediationView.SHADOW_COMPONENT,
    )

    assert report.evaluation_view is RemediationView.SHADOW_COMPONENT
    assert report.result_count == 7
    assert observed_gru_tasks == [TaskName.FAULT_FAMILY]
    assert all(result.task_name != TaskName.CONTINUE_LOG.value for result in report.results)
    assert report.strongest_fault_comparator_macro_f1 == 1.0
    assert report.strongest_action_comparator_macro_f1 == 0.3333333333333333
    assert report.checksum_sha256 != "0" * 64

    payload = report.model_dump(mode="python", round_trip=True)
    payload["strongest_fault_comparator_macro_f1"] = 0.0
    with pytest.raises(ValidationError, match=r"strongest-comparator|checksum"):
        RemediationBaselineReport.model_validate(payload)


def test_optional_matrix_rejects_view_missing_a_required_action_task() -> None:
    record = ExperimentExample(
        example_id="fault-only",
        split_name=VIEW_SOURCE_SPLIT[RemediationView.IID_VALIDATION],
        task_name=TaskName.FAULT_FAMILY,
        prompt_text="fictional prompt",
        target_text="target",
        classification_label="NO_FAULT",
        source_checksum_sha256="a" * 64,
    )
    data = ExperimentData(
        train=(record,),
        validation=(record,),
        inventory_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="lacks a required diagnosis or action"):
        _baselines_with_optional_continuation(
            data,
            _tokenizer(),
            _config(),
            tokenized_train=(),
            tokenized_validation=(),
        )


def test_public_wrapper_rejects_wrong_view_and_token_inventory() -> None:
    dataset = _dataset()
    tokenizer = _tokenizer()
    config = _config()
    train = tuple(
        _tokenized(item) for item in dataset.examples if item.view is RemediationView.IID_TRAIN
    )
    validation = tuple(
        _tokenized(item)
        for item in dataset.examples
        if item.view is RemediationView.SHADOW_COMPONENT
    )

    with pytest.raises(ValueError, match="validation or shadow"):
        run_remediation_baselines(
            dataset,
            tokenizer,
            config,
            tokenized_train=train,
            tokenized_validation=train,
            evaluation_view=RemediationView.IID_TRAIN,
        )
    with pytest.raises(ValueError, match="token inventories"):
        run_remediation_baselines(
            dataset,
            tokenizer,
            config,
            tokenized_train=train,
            tokenized_validation=validation[:-1],
            evaluation_view=RemediationView.SHADOW_COMPONENT,
        )
