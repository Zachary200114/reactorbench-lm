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


class SamplingMetadataRecord(NamedTuple):
    """Non-tokenized, permitted sampling metadata for one IID-train row.

    This deliberately lives beside, rather than inside, ``CompactTokenizedExample``:
    adding semantic fields to that frozen serialization would alter the preserved
    tokenized-inventory checksum used by the v0.2 and historical v0.3 runs.
    """

    example_id: str
    task_name: TaskName
    classification_label: str | None
    augmentation: str


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


def sampling_metadata_inventory_sha256(records: tuple[SamplingMetadataRecord, ...]) -> str:
    """Return the canonical provenance hash for the separate sampler inventory."""

    validated = _validated_sampling_metadata(records)
    digest = hashlib.sha256()
    for row in sorted(validated, key=lambda item: item.example_id):
        for value in (
            row.example_id,
            row.task_name.value,
            row.classification_label or "",
            row.augmentation,
        ):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _validated_sampling_metadata(
    records: tuple[SamplingMetadataRecord, ...],
) -> tuple[SamplingMetadataRecord, ...]:
    if type(records) is not tuple or not records or len(records) > MAX_INVENTORY_RECORDS:
        raise ValueError("sampling metadata must be a bounded non-empty exact tuple")
    identifiers: set[str] = set()
    for index, row in enumerate(records):
        if type(row) is not SamplingMetadataRecord:
            raise TypeError("sampling metadata must use exact SamplingMetadataRecord rows")
        if (
            type(row.example_id) is not str
            or not row.example_id
            or row.example_id != row.example_id.strip()
            or type(row.task_name) is not TaskName
            or type(row.augmentation) is not str
            or not row.augmentation
            or row.augmentation != row.augmentation.strip()
            or (row.classification_label is not None and type(row.classification_label) is not str)
        ):
            raise ValueError(f"sampling metadata row {index} is invalid")
        if row.example_id in identifiers:
            raise ValueError("sampling metadata example IDs must be unique")
        identifiers.add(row.example_id)
        classification = row.task_name in {
            TaskName.FAULT_FAMILY,
            TaskName.NEXT_ACTION,
            TaskName.CONTINUE_LOG,
        }
        if classification != (
            row.classification_label is not None and bool(row.classification_label)
        ):
            raise ValueError("sampling metadata label presence differs from task contract")
    return records


def task_class_balanced_batch_indices[RecordT: SamplingRecord](
    records: tuple[RecordT, ...],
    *,
    metadata: tuple[SamplingMetadataRecord, ...],
    batch_size: int,
    seed: int,
    step: int,
) -> tuple[int, ...]:
    """Choose a stateless task-and-stratum-balanced batch.

    Task quotas use the existing rotating allocation.  A classification task's
    stratum is its *already supervised* exact label; other tasks use augmentation.
    Both level rotations are hash ranked and indexed from ``step`` so a resumed run
    produces exactly the same next batch without consuming process RNG state.
    """

    batch_size = _bounded_integer(
        batch_size, field_name="batch_size", minimum=1, maximum=MAX_BATCH_SIZE
    )
    seed = _bounded_integer(seed, field_name="seed", minimum=0, maximum=MAX_SAMPLING_INTEGER)
    step = _bounded_integer(step, field_name="step", minimum=0, maximum=MAX_SAMPLING_INTEGER)
    by_task = _validated_inventory(records)
    rows = _validated_sampling_metadata(metadata)
    position_by_id = {
        getattr(record, "example_id", None): index for index, record in enumerate(records)
    }
    if None in position_by_id or len(position_by_id) != len(records):
        raise ValueError("sampling records must expose unique example_id values")
    if set(position_by_id) != {row.example_id for row in rows}:
        raise ValueError("sampling metadata does not exactly cover the tokenized inventory")
    metadata_by_position: dict[int, SamplingMetadataRecord] = {}
    for row in rows:
        position = position_by_id[row.example_id]
        if records[position].task_name is not row.task_name:
            raise ValueError("sampling metadata task differs from tokenized record")
        metadata_by_position[position] = row
    tasks = tuple(task for task in TaskName if task in by_task)
    quotas = _task_quotas(tasks, batch_size=batch_size, seed=seed, step=step)
    selected: list[tuple[int, TaskName, _ValidatedRecord]] = []
    for task in tasks:
        strata: dict[str, list[_ValidatedRecord]] = defaultdict(list)
        for record in by_task[task]:
            row = metadata_by_position[record.position]
            stratum = (
                row.classification_label
                if row.classification_label is not None
                else row.augmentation
            )
            strata[stratum].append(record)
        ordered_strata = tuple(
            sorted(
                strata, key=lambda value: (_rank("stratum-order", seed, task.value, value), value)
            )
        )
        if not ordered_strata:
            raise TaskBalancedSamplingError("task-class sampler has an empty task stratum")
        # Rotating by global step ensures each task's strata get the remainder fairly.
        for draw in range(quotas[task]):
            stratum = ordered_strata[(step + draw) % len(ordered_strata)]
            ordered_rows = _ordered_task_records(tuple(strata[stratum]), seed=seed, task_name=task)
            cursor = (step * max(1, quotas[task]) + draw) // len(ordered_strata)
            selected.append((draw, task, ordered_rows[cursor % len(ordered_rows)]))
    selected.sort(
        key=lambda item: (
            _rank(
                "task-class-batch",
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
    indices = tuple(item[2].position for item in selected)
    if len(indices) != batch_size:
        raise TaskBalancedSamplingError(
            "task-class sampler violated its exact batch-size invariant"
        )
    return indices


def fault_continuation_focused_batch_indices[RecordT: SamplingRecord](
    records: tuple[RecordT, ...],
    *,
    metadata: tuple[SamplingMetadataRecord, ...],
    batch_size: int,
    seed: int,
    step: int,
) -> tuple[int, ...]:
    """Choose the frozen v0.3.2 weak-task-focused six-row batch.

    Each batch contains two fault-diagnosis rows, two continuation rows, and one
    row from each of two rotating non-focus tasks. Fault rows retain the empirical
    training distribution; continuation rows rotate uniformly across supervised
    labels because the development errors are confined to its rare event types.
    The other four tasks alternate in deterministic pairs, so each receives one
    row every two steps without disappearing from training.
    """

    batch_size = _bounded_integer(batch_size, field_name="batch_size", minimum=6, maximum=6)
    seed = _bounded_integer(seed, field_name="seed", minimum=0, maximum=MAX_SAMPLING_INTEGER)
    step = _bounded_integer(step, field_name="step", minimum=0, maximum=MAX_SAMPLING_INTEGER)
    by_task = _validated_inventory(records)
    if set(by_task) != set(TaskName):
        raise TaskBalancedSamplingError("focused sampler requires all six task inventories")
    rows = _validated_sampling_metadata(metadata)
    position_by_id = {
        getattr(record, "example_id", None): index for index, record in enumerate(records)
    }
    if None in position_by_id or len(position_by_id) != len(records):
        raise ValueError("sampling records must expose unique example_id values")
    if set(position_by_id) != {row.example_id for row in rows}:
        raise ValueError("sampling metadata does not exactly cover the tokenized inventory")
    metadata_by_position: dict[int, SamplingMetadataRecord] = {}
    for row in rows:
        position = position_by_id[row.example_id]
        if records[position].task_name is not row.task_name:
            raise ValueError("sampling metadata task differs from tokenized record")
        metadata_by_position[position] = row

    selected: list[tuple[int, TaskName, _ValidatedRecord]] = []
    fault_rows = _ordered_task_records(
        by_task[TaskName.FAULT_FAMILY], seed=seed, task_name=TaskName.FAULT_FAMILY
    )
    for draw in range(2):
        selected.append(
            (draw, TaskName.FAULT_FAMILY, fault_rows[(step * 2 + draw) % len(fault_rows)])
        )

    continuation_strata: dict[str, list[_ValidatedRecord]] = defaultdict(list)
    for record in by_task[TaskName.CONTINUE_LOG]:
        label = metadata_by_position[record.position].classification_label
        if label is None:
            raise TaskBalancedSamplingError("continuation sampling metadata lacks its label")
        continuation_strata[label].append(record)
    ordered_labels = tuple(
        sorted(
            continuation_strata,
            key=lambda value: (_rank("focused-continuation-label", seed, value), value),
        )
    )
    for draw in range(2):
        sequence_position = step * 2 + draw
        label = ordered_labels[sequence_position % len(ordered_labels)]
        label_rows = _ordered_task_records(
            tuple(continuation_strata[label]), seed=seed, task_name=TaskName.CONTINUE_LOG
        )
        selected.append(
            (
                draw,
                TaskName.CONTINUE_LOG,
                label_rows[(sequence_position // len(ordered_labels)) % len(label_rows)],
            )
        )

    other_tasks = (
        TaskName.EXTRACT_EVIDENCE,
        TaskName.NEXT_ACTION,
        TaskName.INCIDENT_SUMMARY,
        TaskName.COUNTERFACTUAL_COMPARE,
    )
    start = _rank("focused-other-task-order", seed) % len(other_tasks)
    pair_offset = 0 if step % 2 == 0 else 2
    for draw in range(2):
        task = other_tasks[(start + pair_offset + draw) % len(other_tasks)]
        task_rows = _ordered_task_records(by_task[task], seed=seed, task_name=task)
        selected.append((draw, task, task_rows[(step // 2) % len(task_rows)]))

    selected.sort(
        key=lambda item: (
            _rank(
                "focused-batch-record",
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
    indices = tuple(item[2].position for item in selected)
    if len(indices) != 6:
        raise TaskBalancedSamplingError("focused sampler violated its six-row invariant")
    return indices


def hierarchical_task_label_balanced_batch_indices[RecordT: SamplingRecord](
    records: tuple[RecordT, ...],
    *,
    metadata: tuple[SamplingMetadataRecord, ...],
    batch_size: int,
    seed: int,
    step: int,
) -> tuple[int, ...]:
    """Choose one row per task with bounded label-aware classification cycles.

    Continuation and action labels rotate uniformly. Fault diagnosis preserves the
    high-level 50% unresolved, 10% no-fault, and 40% diagnosed mix while rotating
    diagnosed fault labels uniformly inside the diagnosed tier. This prevents the
    global class flattening that damaged abstention while restoring every task to
    every six-row batch.
    """

    batch_size = _bounded_integer(batch_size, field_name="batch_size", minimum=6, maximum=6)
    seed = _bounded_integer(seed, field_name="seed", minimum=0, maximum=MAX_SAMPLING_INTEGER)
    step = _bounded_integer(step, field_name="step", minimum=0, maximum=MAX_SAMPLING_INTEGER)
    by_task = _validated_inventory(records)
    if set(by_task) != set(TaskName):
        raise TaskBalancedSamplingError("hierarchical sampler requires all six task inventories")
    rows = _validated_sampling_metadata(metadata)
    position_by_id = {
        getattr(record, "example_id", None): index for index, record in enumerate(records)
    }
    if None in position_by_id or len(position_by_id) != len(records):
        raise ValueError("sampling records must expose unique example_id values")
    if set(position_by_id) != {row.example_id for row in rows}:
        raise ValueError("sampling metadata does not exactly cover the tokenized inventory")
    metadata_by_position: dict[int, SamplingMetadataRecord] = {}
    for row in rows:
        position = position_by_id[row.example_id]
        if records[position].task_name is not row.task_name:
            raise ValueError("sampling metadata task differs from tokenized record")
        metadata_by_position[position] = row

    def label_strata(task: TaskName) -> dict[str, tuple[_ValidatedRecord, ...]]:
        mutable: dict[str, list[_ValidatedRecord]] = defaultdict(list)
        for record in by_task[task]:
            label = metadata_by_position[record.position].classification_label
            if label is None:
                raise TaskBalancedSamplingError(
                    f"hierarchical {task.value} metadata lacks its classification label"
                )
            mutable[label].append(record)
        return {label: tuple(values) for label, values in mutable.items()}

    def selected_from_label(
        task: TaskName,
        strata: dict[str, tuple[_ValidatedRecord, ...]],
        label: str,
        occurrence: int,
    ) -> _ValidatedRecord:
        candidates = strata.get(label)
        if not candidates:
            raise TaskBalancedSamplingError(
                f"hierarchical {task.value} metadata lacks required label {label}"
            )
        ordered = _ordered_task_records(candidates, seed=seed, task_name=task)
        return ordered[occurrence % len(ordered)]

    selected: list[tuple[int, TaskName, _ValidatedRecord]] = []
    for task in TaskName:
        if task in {TaskName.CONTINUE_LOG, TaskName.NEXT_ACTION}:
            strata = label_strata(task)
            ordered_labels = tuple(
                sorted(
                    strata,
                    key=lambda value: (
                        _rank("hierarchical-label", seed, task.value, value),
                        value,
                    ),
                )
            )
            label_position = step % len(ordered_labels)
            label = ordered_labels[label_position]
            record = selected_from_label(
                task,
                strata,
                label,
                step // len(ordered_labels),
            )
        elif task is TaskName.FAULT_FAMILY:
            strata = label_strata(task)
            diagnosed_labels = tuple(
                sorted(
                    (label for label in strata if label.startswith("DIAGNOSED:")),
                    key=lambda value: (_rank("hierarchical-fault-label", seed, value), value),
                )
            )
            if set(strata) != {"UNRESOLVED", "NO_FAULT", *diagnosed_labels}:
                raise TaskBalancedSamplingError(
                    "hierarchical fault metadata contains an unsupported label tier"
                )
            if not diagnosed_labels:
                raise TaskBalancedSamplingError(
                    "hierarchical fault metadata lacks diagnosed labels"
                )
            cycle = step % 10
            block = step // 10
            if cycle < 5:
                label = "UNRESOLVED"
                occurrence = block * 5 + cycle
            elif cycle == 5:
                label = "NO_FAULT"
                occurrence = block
            else:
                diagnosed_position = block * 4 + (cycle - 6)
                label = diagnosed_labels[diagnosed_position % len(diagnosed_labels)]
                occurrence = diagnosed_position // len(diagnosed_labels)
            record = selected_from_label(task, strata, label, occurrence)
        else:
            ordered = _ordered_task_records(by_task[task], seed=seed, task_name=task)
            record = ordered[step % len(ordered)]
        selected.append((0, task, record))

    selected.sort(
        key=lambda item: (
            _rank(
                "hierarchical-batch-record",
                seed,
                str(step),
                item[1].value,
                item[2].group_id,
                str(item[2].position),
            ),
            item[1].value,
            item[2].position,
        )
    )
    indices = tuple(item[2].position for item in selected)
    if len(indices) != 6 or {records[index].task_name for index in indices} != set(TaskName):
        raise TaskBalancedSamplingError("hierarchical sampler violated its six-task invariant")
    return indices


__all__ = [
    "MAX_BATCH_SIZE",
    "MAX_GROUP_ID_UTF8_BYTES",
    "MAX_INVENTORY_RECORDS",
    "MAX_SAMPLING_INTEGER",
    "SamplingMetadataRecord",
    "SamplingRecord",
    "TaskBalancedSamplingError",
    "fault_continuation_focused_batch_indices",
    "hierarchical_task_label_balanced_batch_indices",
    "sampling_metadata_inventory_sha256",
    "task_balanced_batch",
    "task_balanced_batch_indices",
    "task_class_balanced_batch_indices",
]
