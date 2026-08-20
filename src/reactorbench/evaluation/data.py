"""Leakage-resistant Phase 5 experiment example materialization."""

from __future__ import annotations

from dataclasses import dataclass

from reactorbench.dataset import VerifiedDevelopmentCandidateArtifact
from reactorbench.dataset.contracts import ProjectionTaskTargetValue, PromptContinuationTarget
from reactorbench.schemas.base import canonical_json_bytes, canonical_sha256
from reactorbench.schemas.enums import DiagnosisStatus, SplitName, TaskName
from reactorbench.schemas.target import FaultDiagnosisTarget, NextActionTarget


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
    "examples_for_task",
    "materialize_experiment_data",
]
