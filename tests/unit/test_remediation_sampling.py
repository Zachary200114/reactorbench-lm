"""Unit tests for resumable task-balanced batch selection."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from reactorbench.remediation.sampling import (
    MAX_BATCH_SIZE,
    MAX_GROUP_ID_UTF8_BYTES,
    MAX_SAMPLING_INTEGER,
    SamplingMetadataRecord,
    SamplingRecord,
    TaskBalancedSamplingError,
    sampling_metadata_inventory_sha256,
    task_balanced_batch,
    task_balanced_batch_indices,
    task_class_balanced_batch_indices,
)
from reactorbench.schemas.enums import TaskName


@dataclass(frozen=True)
class _Record:
    record_id: str
    task_name: TaskName
    group_id: str


@dataclass(frozen=True)
class _ClassRecord:
    example_id: str
    task_name: TaskName
    group_id: str


def _class_inventory() -> tuple[tuple[_ClassRecord, ...], tuple[SamplingMetadataRecord, ...]]:
    classification = {
        TaskName.FAULT_FAMILY,
        TaskName.NEXT_ACTION,
        TaskName.CONTINUE_LOG,
    }
    records: list[_ClassRecord] = []
    metadata: list[SamplingMetadataRecord] = []
    for task in TaskName:
        for index in range(6):
            example_id = f"{task.value}:{index}"
            records.append(_ClassRecord(example_id, task, f"group:{example_id}"))
            metadata.append(
                SamplingMetadataRecord(
                    example_id=example_id,
                    task_name=task,
                    classification_label=(f"label-{index % 3}" if task in classification else None),
                    augmentation=f"augmentation-{index % 2}",
                )
            )
    return tuple(records), tuple(metadata)


def test_task_class_sampler_is_deterministic_and_balances_supervised_strata() -> None:
    records, metadata = _class_inventory()
    first = task_class_balanced_batch_indices(
        records, metadata=metadata, batch_size=18, seed=91, step=4
    )
    replay = task_class_balanced_batch_indices(
        records, metadata=metadata, batch_size=18, seed=91, step=4
    )
    assert first == replay
    assert Counter(records[index].task_name for index in first) == Counter(
        dict.fromkeys(TaskName, 3)
    )
    metadata_by_id = {item.example_id: item for item in metadata}
    for task in (TaskName.FAULT_FAMILY, TaskName.NEXT_ACTION, TaskName.CONTINUE_LOG):
        labels = {
            metadata_by_id[records[index].example_id].classification_label
            for index in first
            if records[index].task_name is task
        }
        assert labels == {"label-0", "label-1", "label-2"}


def test_task_class_metadata_hash_is_order_independent_and_mismatch_is_rejected() -> None:
    records, metadata = _class_inventory()
    assert sampling_metadata_inventory_sha256(metadata) == sampling_metadata_inventory_sha256(
        tuple(reversed(metadata))
    )
    with pytest.raises(ValueError, match="exactly cover"):
        task_class_balanced_batch_indices(
            records,
            metadata=metadata[:-1],
            batch_size=6,
            seed=1,
            step=0,
        )


def _singletons(*, tasks: tuple[TaskName, ...], per_task: int) -> tuple[_Record, ...]:
    return tuple(
        _Record(
            record_id=f"{task_name.value}:{index}",
            task_name=task_name,
            group_id=f"group:{task_name.value}:{index}",
        )
        for task_name in tasks
        for index in range(per_task)
    )


def test_exact_balanced_batch_is_deterministic_resumable_and_rng_isolated() -> None:
    records = _singletons(tasks=tuple(TaskName), per_task=12)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    expected = task_balanced_batch(records, batch_size=14, seed=1729, step=9)
    task_balanced_batch(records, batch_size=14, seed=1729, step=2)
    resumed = task_balanced_batch(records, batch_size=14, seed=1729, step=9)

    assert resumed == expected
    assert len(resumed) == 14
    counts = Counter(record.task_name for record in resumed)
    assert set(counts) == set(TaskName)
    assert max(counts.values()) - min(counts.values()) == 1
    assert len({record.record_id for record in resumed}) == len(resumed)
    assert random.getstate() == python_state
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_state[0]
    assert np.array_equal(numpy_after[1], numpy_state[1])
    assert numpy_after[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


@given(
    seed=st.integers(min_value=0, max_value=MAX_SAMPLING_INTEGER),
    step=st.integers(min_value=0, max_value=MAX_SAMPLING_INTEGER),
    task_count=st.integers(min_value=1, max_value=len(TaskName)),
    batch_size=st.integers(min_value=1, max_value=60),
)
def test_singleton_inventory_property_is_exact_balanced_unique_and_replayable(
    seed: int,
    step: int,
    task_count: int,
    batch_size: int,
) -> None:
    tasks = tuple(TaskName)[:task_count]
    records = _singletons(tasks=tasks, per_task=(batch_size + task_count - 1) // task_count)

    first = task_balanced_batch(records, batch_size=batch_size, seed=seed, step=step)
    replay = task_balanced_batch(records, batch_size=batch_size, seed=seed, step=step)

    assert first == replay
    assert len(first) == batch_size
    assert len({record.record_id for record in first}) == batch_size
    counts = Counter(record.task_name for record in first)
    quotas = tuple(counts[task_name] for task_name in tasks)
    assert max(quotas) - min(quotas) <= 1


def test_remainder_quota_and_records_rotate_by_step() -> None:
    tasks = (
        TaskName.CONTINUE_LOG,
        TaskName.FAULT_FAMILY,
        TaskName.NEXT_ACTION,
    )
    records = _singletons(tasks=tasks, per_task=8)
    batches = tuple(
        task_balanced_batch(records, batch_size=4, seed=44, step=step) for step in range(3)
    )

    bonus_tasks = []
    for batch in batches:
        counts = Counter(record.task_name for record in batch)
        bonus_tasks.append(next(task for task, count in counts.items() if count == 2))
    assert set(bonus_tasks) == set(tasks)
    assert len({tuple(record.record_id for record in batch) for batch in batches}) == 3


def test_v03_six_task_batches_sample_shared_groups_at_record_level() -> None:
    records = tuple(
        _Record(
            record_id=f"{task_name.value}:{member}",
            task_name=task_name,
            group_id=f"augmentation:{task_name.value}",
        )
        for task_name in TaskName
        for member in range(3)
    )

    batches = tuple(
        task_balanced_batch(records, batch_size=6, seed=700, step=step) for step in range(6)
    )
    assert batches == tuple(
        task_balanced_batch(records, batch_size=6, seed=700, step=step) for step in range(6)
    )
    for batch in batches:
        counts = Counter(record.task_name for record in batch)
        assert counts == Counter(dict.fromkeys(TaskName, 1))
        assert len(batch) == 6

    for task_name in TaskName:
        first_cycle = {
            record.record_id
            for batch in batches[:3]
            for record in batch
            if record.task_name is task_name
        }
        assert first_cycle == {f"{task_name.value}:{member}" for member in range(3)}


def test_sparse_task_quotas_advance_by_actual_prior_draws() -> None:
    tasks = (TaskName.CONTINUE_LOG, TaskName.FAULT_FAMILY)
    records = _singletons(tasks=tasks, per_task=4)
    batches = tuple(
        task_balanced_batch(records, batch_size=1, seed=17, step=step) for step in range(8)
    )

    for task_name in tasks:
        selected = [batch[0].record_id for batch in batches if batch[0].task_name is task_name]
        assert len(selected) == 4
        assert len(set(selected)) == 4


def test_duplicate_records_are_used_only_when_task_inventory_is_too_small() -> None:
    records = (
        _Record("a", TaskName.EXTRACT_EVIDENCE, "shared"),
        _Record("b", TaskName.EXTRACT_EVIDENCE, "shared"),
    )
    indices = task_balanced_batch_indices(records, batch_size=4, seed=1, step=0)
    batch = task_balanced_batch(records, batch_size=4, seed=1, step=0)

    assert len(indices) == 4
    assert Counter(indices) == Counter({0: 2, 1: 2})
    assert Counter(record.record_id for record in batch) == Counter({"a": 2, "b": 2})
    assert {record.group_id for record in batch} == {"shared"}

    singleton_groups = (
        _Record("c", TaskName.EXTRACT_EVIDENCE, "c"),
        _Record("d", TaskName.EXTRACT_EVIDENCE, "d"),
    )
    cycled = task_balanced_batch(singleton_groups, batch_size=4, seed=1, step=0)
    assert Counter(record.record_id for record in cycled) == Counter({"c": 2, "d": 2})


def test_record_selection_avoids_duplicates_when_inventory_permits() -> None:
    records = (
        _Record("a0", TaskName.INCIDENT_SUMMARY, "a"),
        _Record("a1", TaskName.INCIDENT_SUMMARY, "a"),
        _Record("b0", TaskName.INCIDENT_SUMMARY, "b"),
        _Record("b1", TaskName.INCIDENT_SUMMARY, "b"),
        _Record("c0", TaskName.INCIDENT_SUMMARY, "c"),
    )
    indices = task_balanced_batch_indices(records, batch_size=3, seed=5, step=0)

    assert len(indices) == len(set(indices)) == 3
    assert len({records[index].record_id for index in indices}) == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", True),
        ("batch_size", 0),
        ("batch_size", MAX_BATCH_SIZE + 1),
        ("seed", False),
        ("seed", -1),
        ("seed", MAX_SAMPLING_INTEGER + 1),
        ("step", False),
        ("step", -1),
        ("step", MAX_SAMPLING_INTEGER + 1),
    ],
)
def test_integer_bounds_fail_closed(field: str, value: object) -> None:
    records = (_Record("one", TaskName.CONTINUE_LOG, "one"),)
    arguments: dict[str, object] = {"batch_size": 1, "seed": 0, "step": 0}
    arguments[field] = value
    with pytest.raises(ValueError, match=field):
        task_balanced_batch(records, **arguments)  # type: ignore[arg-type]


def test_inventory_must_be_an_exact_nonempty_tuple() -> None:
    record = _Record("one", TaskName.CONTINUE_LOG, "one")
    with pytest.raises(TypeError, match="exact tuple"):
        task_balanced_batch([record], batch_size=1, seed=0, step=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        task_balanced_batch((), batch_size=1, seed=0, step=0)
    with pytest.raises(ValueError, match="same object"):
        task_balanced_batch((record, record), batch_size=1, seed=0, step=0)


@pytest.mark.parametrize(
    "group_id",
    [
        "",
        " leading",
        "trailing ",
        "control\ncharacter",
        "x" * (MAX_GROUP_ID_UTF8_BYTES + 1),
    ],
)
def test_group_identifiers_are_strictly_bounded(group_id: str) -> None:
    records = (_Record("one", TaskName.CONTINUE_LOG, group_id),)
    with pytest.raises(ValueError, match="group_id"):
        task_balanced_batch(records, batch_size=1, seed=0, step=0)


def test_record_surface_and_group_task_boundaries_fail_closed() -> None:
    @dataclass(frozen=True)
    class _WrongTask:
        task_name: str
        group_id: str

    @dataclass(frozen=True)
    class _NoGroup:
        task_name: TaskName

    with pytest.raises(TypeError, match="exact TaskName"):
        task_balanced_batch(
            (cast(SamplingRecord, _WrongTask(TaskName.CONTINUE_LOG.value, "g")),),
            batch_size=1,
            seed=0,
            step=0,
        )
    with pytest.raises(TypeError, match="expose"):
        task_balanced_batch(
            (cast(SamplingRecord, _NoGroup(TaskName.CONTINUE_LOG)),),
            batch_size=1,
            seed=0,
            step=0,
        )

    cross_task = (
        _Record("one", TaskName.CONTINUE_LOG, "shared"),
        _Record("two", TaskName.FAULT_FAMILY, "shared"),
    )
    with pytest.raises(ValueError, match="crosses task boundaries"):
        task_balanced_batch(cross_task, batch_size=2, seed=0, step=0)


def test_group_size_does_not_constrain_batches_and_replacement_is_bounded() -> None:
    oversized = tuple(_Record(f"r{index}", TaskName.CONTINUE_LOG, "shared") for index in range(3))
    selected = task_balanced_batch(oversized, batch_size=2, seed=0, step=0)
    assert len(selected) == len({record.record_id for record in selected}) == 2

    shared_siblings = (
        _Record("a0", TaskName.CONTINUE_LOG, "a"),
        _Record("a1", TaskName.CONTINUE_LOG, "a"),
        _Record("b0", TaskName.FAULT_FAMILY, "b"),
        _Record("b1", TaskName.FAULT_FAMILY, "b"),
    )
    balanced = task_balanced_batch(shared_siblings, batch_size=2, seed=0, step=0)
    assert Counter(record.task_name for record in balanced) == Counter(
        {TaskName.CONTINUE_LOG: 1, TaskName.FAULT_FAMILY: 1}
    )

    replacement = (
        _Record("a0", TaskName.NEXT_ACTION, "a"),
        _Record("a1", TaskName.NEXT_ACTION, "a"),
    )
    repeated = task_balanced_batch(replacement, batch_size=3, seed=0, step=0)
    assert len(repeated) == 3
    assert {record.record_id for record in repeated} == {"a0", "a1"}
    assert sorted(Counter(record.record_id for record in repeated).values()) == [1, 2]


def test_sampler_specific_error_remains_a_public_value_error() -> None:
    assert issubclass(TaskBalancedSamplingError, ValueError)
