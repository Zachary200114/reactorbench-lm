"""Leakage-resistant Phase 5 experiment example materialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from reactorbench.dataset import VerifiedDevelopmentCandidateArtifact
from reactorbench.dataset.contracts import ProjectionTaskTargetValue, PromptContinuationTarget
from reactorbench.schemas.base import canonical_json_bytes, canonical_sha256
from reactorbench.schemas.enums import DiagnosisStatus, SplitName, TaskName
from reactorbench.schemas.target import FaultDiagnosisTarget, NextActionTarget

from .config import Phase6TestFreezeConfig


@dataclass(frozen=True)
class ExperimentExample:
    example_id: str
    split_name: SplitName
    task_name: TaskName
    prompt_text: str
    target_text: str
    classification_label: str | None
    source_checksum_sha256: str


@dataclass(frozen=True)
class ExperimentData:
    train: tuple[ExperimentExample, ...]
    validation: tuple[ExperimentExample, ...]
    inventory_sha256: str


@dataclass(frozen=True)
class Phase6ExperimentData:
    """Verified examples for every frozen split, including paired/corrupted prompts."""

    by_split: Mapping[SplitName, tuple[ExperimentExample, ...]]
    inventory_sha256_by_split: Mapping[SplitName, str]
    all_records: tuple[ExperimentExample, ...]


_PHASE6_SPLIT_HASH_FIELD: dict[SplitName, str] = {
    SplitName.COMPONENT_TEST: "component_test_inventory_sha256",
    SplitName.COMPOSITION_TEST: "composition_test_inventory_sha256",
    SplitName.COUNTERFACTUAL_TEST: "counterfactual_test_inventory_sha256",
    SplitName.IID_TEST: "iid_test_inventory_sha256",
    SplitName.NOISE_TEST: "noise_test_inventory_sha256",
    SplitName.SEVERITY_TEST: "severity_test_inventory_sha256",
    SplitName.TEMPLATE_TEST: "template_test_inventory_sha256",
}


def _classification_label(task_name: TaskName, target: ProjectionTaskTargetValue) -> str | None:
    if task_name is TaskName.FAULT_FAMILY:
        if type(target) is not FaultDiagnosisTarget:
            raise TypeError("fault-family example has the wrong target type")
        diagnosis = target.diagnosis_status
        if diagnosis is not DiagnosisStatus.DIAGNOSED:
            return str(diagnosis.value)
        return "DIAGNOSED:" + "+".join(label.value for label in target.fault_labels)
    if task_name is TaskName.NEXT_ACTION:
        if type(target) is not NextActionTarget:
            raise TypeError("next-action example has the wrong target type")
        return str(target.immediate_action.value)
    if task_name is TaskName.CONTINUE_LOG:
        if type(target) is not PromptContinuationTarget:
            raise TypeError("continue-log example has the wrong target type")
        return str(target.next_event_type.value)
    return None


def materialize_experiment_data(
    verified: VerifiedDevelopmentCandidateArtifact,
    *,
    maximum_prompt_utf8_bytes: int,
) -> ExperimentData:
    """Materialize only IID train/validation examples after full artifact verification."""

    if type(verified) is not VerifiedDevelopmentCandidateArtifact:
        raise TypeError("verified must be an exact VerifiedDevelopmentCandidateArtifact")
    if type(maximum_prompt_utf8_bytes) is not int or not 1024 <= maximum_prompt_utf8_bytes:
        raise ValueError("maximum_prompt_utf8_bytes is invalid")
    renders = {record.render_id: record for record in verified.candidate.rendered_candidates}
    permitted = {SplitName.IID_TRAIN, SplitName.IID_VALIDATION}
    records: list[ExperimentExample] = []
    for example in verified.candidate.task_examples:
        if example.split_name not in permitted:
            continue
        if len(example.prompt_render_ids) != 1 or example.corruption_ids:
            raise ValueError("Phase 5 train/validation examples must use one uncorrupted prompt")
        rendered = renders.get(example.prompt_render_ids[0])
        if rendered is None or rendered.split_name is not example.split_name:
            raise ValueError("task example references a missing or cross-split render")
        if len(rendered.text.encode("utf-8")) > maximum_prompt_utf8_bytes:
            raise ValueError("Phase 5 prompt exceeds its configured byte bound")
        target = example.task_target.target
        target_text = canonical_json_bytes(target.model_dump(mode="json", round_trip=True)).decode(
            "utf-8"
        )
        records.append(
            ExperimentExample(
                example_id=example.example_id,
                split_name=example.split_name,
                task_name=example.task_name,
                prompt_text=rendered.text,
                target_text=target_text,
                classification_label=_classification_label(example.task_name, target),
                source_checksum_sha256=example.checksum_sha256,
            )
        )
    records.sort(key=lambda item: item.example_id)
    if len({record.example_id for record in records}) != len(records):
        raise ValueError("Phase 5 experiment example IDs are not unique")
    train = tuple(record for record in records if record.split_name is SplitName.IID_TRAIN)
    validation = tuple(
        record for record in records if record.split_name is SplitName.IID_VALIDATION
    )
    if not train or not validation:
        raise ValueError("Phase 5 requires non-empty IID train and validation inventories")
    inventory = tuple(
        (
            record.example_id,
            record.split_name.value,
            record.task_name.value,
            record.source_checksum_sha256,
            canonical_sha256(record.target_text),
        )
        for record in records
    )
    return ExperimentData(
        train=train,
        validation=validation,
        inventory_sha256=canonical_sha256(inventory),
    )


def materialize_phase6_data(
    verified: VerifiedDevelopmentCandidateArtifact,
    *,
    freeze: Phase6TestFreezeConfig,
    maximum_prompt_utf8_bytes: int,
) -> Phase6ExperimentData:
    """Materialize all splits and prove the held-out inventory is unchanged."""

    if type(verified) is not VerifiedDevelopmentCandidateArtifact:
        raise TypeError("verified must be an exact VerifiedDevelopmentCandidateArtifact")
    if type(freeze) is not Phase6TestFreezeConfig:
        raise TypeError("freeze must be an exact Phase6TestFreezeConfig")
    if type(maximum_prompt_utf8_bytes) is not int or maximum_prompt_utf8_bytes < 1024:
        raise ValueError("maximum_prompt_utf8_bytes is invalid")
    renders = {record.render_id: record for record in verified.candidate.rendered_candidates}
    corruptions = {record.corruption_id: record for record in verified.candidate.corruption_records}
    records: list[ExperimentExample] = []
    for example in verified.candidate.task_examples:
        prompt_texts: list[str] = []
        if example.corruption_ids:
            for render_id, corruption_id in zip(
                example.prompt_render_ids, example.corruption_ids, strict=True
            ):
                rendered = renders.get(render_id)
                corruption = corruptions.get(corruption_id)
                if (
                    rendered is None
                    or corruption is None
                    or rendered.split_name is not example.split_name
                    or rendered.model_input_sha256 != corruption.model_input_sha256
                    or rendered.text_sha256 != corruption.text_sha256
                ):
                    raise ValueError("task example references a missing or misbound corruption")
                prompt_texts.append(rendered.text)
        else:
            for render_id in example.prompt_render_ids:
                rendered = renders.get(render_id)
                if rendered is None or rendered.split_name is not example.split_name:
                    raise ValueError("task example references a missing or cross-split render")
                prompt_texts.append(rendered.text)
        if len(prompt_texts) == 1:
            prompt_text = prompt_texts[0]
        elif len(prompt_texts) == 2 and example.task_name is TaskName.COUNTERFACTUAL_COMPARE:
            prompt_text = (
                "[BASELINE]\n" + prompt_texts[0] + "\n[COUNTERFACTUAL]\n" + prompt_texts[1]
            )
        else:
            raise ValueError("task example has an unsupported prompt arity")
        if len(prompt_text.encode("utf-8")) > maximum_prompt_utf8_bytes:
            raise ValueError("Phase 6 prompt exceeds its configured byte bound")
        target = example.task_target.target
        records.append(
            ExperimentExample(
                example_id=example.example_id,
                split_name=example.split_name,
                task_name=example.task_name,
                prompt_text=prompt_text,
                target_text=canonical_json_bytes(
                    target.model_dump(mode="json", round_trip=True)
                ).decode("utf-8"),
                classification_label=_classification_label(example.task_name, target),
                source_checksum_sha256=example.checksum_sha256,
            )
        )
    records.sort(key=lambda item: item.example_id)
    if len(records) != len({record.example_id for record in records}):
        raise ValueError("Phase 6 example IDs are not unique")
    grouped = {
        split: tuple(record for record in records if record.split_name is split)
        for split in SplitName
    }
    expected_counts = {
        SplitName.IID_TRAIN: freeze.train_example_count,
        SplitName.IID_VALIDATION: freeze.validation_example_count,
    }
    for split, expected in expected_counts.items():
        if len(grouped[split]) != expected:
            raise ValueError(f"{split.value} count differs from the frozen inventory")
    test_count = sum(len(grouped[split]) for split in _PHASE6_SPLIT_HASH_FIELD)
    if test_count != freeze.test_example_count:
        raise ValueError("held-out example count differs from the frozen inventory")
    inventory_hashes: dict[SplitName, str] = {}
    for split, split_records in grouped.items():
        digest = canonical_sha256(
            tuple(
                (record.example_id, record.task_name.value, record.source_checksum_sha256)
                for record in split_records
            )
        )
        inventory_hashes[split] = digest
        field_name = _PHASE6_SPLIT_HASH_FIELD.get(split)
        if field_name is not None and digest != getattr(freeze, field_name):
            raise ValueError(f"{split.value} checksum differs from the frozen inventory")
    return Phase6ExperimentData(
        by_split=MappingProxyType(grouped),
        inventory_sha256_by_split=MappingProxyType(inventory_hashes),
        all_records=tuple(records),
    )


def examples_for_task(
    records: tuple[ExperimentExample, ...], task_name: TaskName
) -> tuple[ExperimentExample, ...]:
    if type(records) is not tuple or type(task_name) is not TaskName:
        raise TypeError("task filtering requires exact record tuple and TaskName")
    selected = tuple(record for record in records if record.task_name is task_name)
    if any(record.classification_label is None for record in selected):
        raise ValueError("selected classification task contains a non-classification target")
    return selected


__all__ = [
    "ExperimentData",
    "ExperimentExample",
    "Phase6ExperimentData",
    "examples_for_task",
    "materialize_experiment_data",
    "materialize_phase6_data",
]
