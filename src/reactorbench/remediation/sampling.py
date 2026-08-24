"""Deterministic, record-level task-balanced batch selection.

The sampler deliberately owns no mutable cursor. A batch is an exact function of
the validated inventory, ``seed``, and zero-based ``step``; callers can therefore
resume at a recorded step without serializing Python, NumPy, or Torch RNG state.

``group_id`` is validated lineage metadata. Group membership is kept atomic by the
dataset split, not by optimization batches, so records that share a group may be
sampled independently here.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import NamedTuple, Protocol

from reactorbench.schemas.enums import TaskName

MAX_BATCH_SIZE = 4096
MAX_GROUP_ID_UTF8_BYTES = 512
MAX_INVENTORY_RECORDS = 1_000_000
MAX_SAMPLING_INTEGER = (1 << 63) - 1


class SamplingRecord(Protocol):
    """Minimum immutable surface consumed by the sampler."""

    @property
    def task_name(self) -> TaskName: ...

    @property
    def group_id(self) -> str: ...


class TaskBalancedSamplingError(ValueError):
    """A validated inventory cannot satisfy a sampler invariant."""


class _ValidatedRecord(NamedTuple):
    position: int
    group_id: str


def _bounded_integer(value: object, *, field_name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field_name} must be an exact integer in [{minimum}, {maximum}]")
    return value


def _rank(domain: str, seed: int, *parts: str) -> int:
    """Return a process-stable rank without touching any global RNG."""

    digest = hashlib.sha256()
    for value in (domain, str(seed), *parts):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big")


def _validate_group_id(value: object, *, record_index: int) -> str:
    if type(value) is not str:
        raise TypeError(f"record {record_index} group_id must be an exact string")
    if not value or value != value.strip():
        raise ValueError(f"record {record_index} group_id must be non-empty and trimmed")
    if any(not character.isprintable() for character in value):
        raise ValueError(f"record {record_index} group_id contains a control character")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"record {record_index} group_id is not valid UTF-8") from exc
    if len(encoded) > MAX_GROUP_ID_UTF8_BYTES:
        raise ValueError(f"record {record_index} group_id exceeds its UTF-8 byte bound")
    return value


def _validated_inventory[RecordT: SamplingRecord](
    records: tuple[RecordT, ...],
) -> dict[TaskName, tuple[_ValidatedRecord, ...]]:
    if type(records) is not tuple:
        raise TypeError("records must be an exact tuple")
    if not records:
        raise ValueError("records must not be empty")
    if len(records) > MAX_INVENTORY_RECORDS:
        raise ValueError("record inventory exceeds its bound")

    object_ids: set[int] = set()
    group_tasks: dict[str, TaskName] = {}
    by_task: dict[TaskName, list[_ValidatedRecord]] = defaultdict(list)
    for index, record in enumerate(records):
        identity = id(record)
        if identity in object_ids:
            raise ValueError("records must not repeat the same object instance")
        object_ids.add(identity)
        try:
            task_name = record.task_name
            raw_group_id = record.group_id
        except AttributeError as exc:
            raise TypeError("every record must expose task_name and group_id") from exc
        if type(task_name) is not TaskName:
            raise TypeError(f"record {index} task_name must be an exact TaskName")
        group_id = _validate_group_id(raw_group_id, record_index=index)
        prior_task = group_tasks.setdefault(group_id, task_name)
        if prior_task is not task_name:
            raise ValueError(f"group {group_id!r} crosses task boundaries")
        by_task[task_name].append(_ValidatedRecord(position=index, group_id=group_id))

    return {task_name: tuple(indices) for task_name, indices in by_task.items()}


def _task_quotas(
    task_names: tuple[TaskName, ...],
    *,
    batch_size: int,
    seed: int,
    step: int,
) -> dict[TaskName, int]:
    base, remainder = divmod(batch_size, len(task_names))
    start = (_rank("task-quota", seed) + step) % len(task_names)
    bonus = {task_names[(start + offset) % len(task_names)] for offset in range(remainder)}
    return {task_name: base + int(task_name in bonus) for task_name in task_names}


def _draws_before_step(
    task_names: tuple[TaskName, ...],
    *,
    task_name: TaskName,
    batch_size: int,
    seed: int,
    step: int,
) -> int:
    """Return the task's exact cumulative quota before ``step`` in constant space."""

    task_count = len(task_names)
    base, remainder = divmod(batch_size, task_count)
    total = base * step
    complete_cycles, partial_cycle = divmod(step, task_count)
    total += complete_cycles * remainder
    initial_start = _rank("task-quota", seed) % task_count
    for prior_step in range(partial_cycle):
        start = (initial_start + prior_step) % task_count
        if any(
            task_names[(start + offset) % task_count] is task_name for offset in range(remainder)
        ):
            total += 1
    return total


def _ordered_task_records(
    records: tuple[_ValidatedRecord, ...],
    *,
    seed: int,
    task_name: TaskName,
) -> tuple[_ValidatedRecord, ...]:
    return tuple(
        sorted(
            records,
            key=lambda record: (
                _rank(
                    "record-order",
                    seed,
                    task_name.value,
                    record.group_id,
                    str(record.position),
                ),
                record.group_id,
                record.position,
            ),
        )
    )


def task_balanced_batch_indices[RecordT: SamplingRecord](
    records: tuple[RecordT, ...],
    *,
    batch_size: int,
    seed: int,
    step: int,
) -> tuple[int, ...]:
    """Select one exact, deterministic, task-balanced batch of input positions.

    Quotas across tasks present in ``records`` differ by at most one, and any
    remainder rotates across tasks by step. Each task has a stable seeded record
    order and an exact stateless cursor derived from all prior quotas. Consequently,
    records rotate fairly across interrupted and resumed runs without mutable RNG
    state. A task's records are not duplicated within a batch when its inventory can
    satisfy its quota; deterministic cyclic replacement begins only after all of
    that task's records have been selected.

    ``group_id`` is checked for safe syntax and task-consistent lineage, but it does
    not make sibling records indivisible inside a training batch.
    """

    batch_size = _bounded_integer(
        batch_size,
        field_name="batch_size",
        minimum=1,
        maximum=MAX_BATCH_SIZE,
    )
    seed = _bounded_integer(
        seed,
        field_name="seed",
        minimum=0,
        maximum=MAX_SAMPLING_INTEGER,
    )
    step = _bounded_integer(
        step,
        field_name="step",
        minimum=0,
        maximum=MAX_SAMPLING_INTEGER,
    )
    records_by_task = _validated_inventory(records)
    canonical_tasks = tuple(task for task in TaskName if task in records_by_task)
    quotas = _task_quotas(
        canonical_tasks,
        batch_size=batch_size,
        seed=seed,
        step=step,
    )

    selected: list[tuple[int, TaskName, _ValidatedRecord]] = []
    for task_name in canonical_tasks:
        ordered = _ordered_task_records(
            records_by_task[task_name],
            seed=seed,
            task_name=task_name,
        )
        quota = quotas[task_name]
        cursor = _draws_before_step(
            canonical_tasks,
            task_name=task_name,
            batch_size=batch_size,
            seed=seed,
            step=step,
        )
        for draw_position in range(quota):
            record = ordered[(cursor + draw_position) % len(ordered)]
            selected.append((draw_position, task_name, record))

    selected.sort(
        key=lambda item: (
            _rank(
                "batch-record",
                seed,
                str(step),
                item[1].value,
                str(item[0]),
                item[2].group_id,
                str(item[2].position),
            ),
            item[1].value,
            item[0],
            item[2].position,
        )
    )
    indices = tuple(record.position for _draw_position, _task_name, record in selected)
    if len(indices) != batch_size:
        raise TaskBalancedSamplingError(
            "task-balanced sampler violated its exact batch-size invariant"
        )
    return indices


def task_balanced_batch[RecordT: SamplingRecord](
    records: tuple[RecordT, ...],
    *,
    batch_size: int,
    seed: int,
    step: int,
) -> tuple[RecordT, ...]:
    """Return records selected by :func:`task_balanced_batch_indices`."""

    indices = task_balanced_batch_indices(
        records,
        batch_size=batch_size,
        seed=seed,
        step=step,
    )
    return tuple(records[index] for index in indices)


__all__ = [
    "MAX_BATCH_SIZE",
    "MAX_GROUP_ID_UTF8_BYTES",
    "MAX_INVENTORY_RECORDS",
    "MAX_SAMPLING_INTEGER",
    "SamplingRecord",
    "TaskBalancedSamplingError",
    "task_balanced_batch",
    "task_balanced_batch_indices",
]
