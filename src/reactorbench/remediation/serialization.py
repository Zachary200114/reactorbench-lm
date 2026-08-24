"""Compact-target serialization with task-specific generation reservations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from reactorbench.evaluation.serialization import supervised_causal_loss
from reactorbench.schemas.enums import TaskName
from reactorbench.tokenizer import BOS_ID, PAD_ID, ProjectTokenizer

from .data import RemediationExample

PROMPT_PREFIX = "<|prompt|>"
TARGET_PREFIX = "<|target|>"


@dataclass(frozen=True, slots=True)
class CompactTokenizedExample:
    example_id: str
    task_name: TaskName
    group_id: str
    token_ids: tuple[int, ...]
    target_mask: tuple[bool, ...]
    prompt_token_count: int
    target_token_count: int
    prompt_tokens_retained: int
    prompt_truncated: bool


def compact_serialized_parts(example: RemediationExample) -> tuple[str, str]:
    if type(example) is not RemediationExample:
        raise TypeError("compact serialization requires an exact RemediationExample")
    prompt = (
        f"{PROMPT_PREFIX}\n{example.prompt_text}\nTASK={example.task_name.value}\n{TARGET_PREFIX}\n"
    )
    return prompt, example.compact_target


def compact_prompt_footer(task_name: TaskName) -> str:
    """Return the task-conditioning suffix that truncation must preserve exactly."""

    if type(task_name) is not TaskName:
        raise TypeError("compact prompt footer requires an exact TaskName")
    return f"TASK={task_name.value}\n{TARGET_PREFIX}\n"


def retained_compact_prompt_tokens(
    example: RemediationExample,
    tokenizer: ProjectTokenizer,
    *,
    context_length: int,
    generation_cap: int,
) -> tuple[tuple[int, ...], int, bool]:
    """Tokenize and left-truncate a prompt while proving its task footer survived."""

    if type(example) is not RemediationExample or type(tokenizer) is not ProjectTokenizer:
        raise TypeError("compact prompt tokenization requires exact project contracts")
    if type(context_length) is not int or context_length < 32:
        raise ValueError("context length must be an integer of at least 32")
    if type(generation_cap) is not int or not 1 <= generation_cap < context_length:
        raise ValueError("generation cap must leave a valid prompt boundary")
    prompt_text, _target_text = compact_serialized_parts(example)
    original = tuple(tokenizer.encode(prompt_text, add_bos=True, add_eos=False))
    if not original or original[0] != BOS_ID:
        raise ValueError("tokenized compact prompt does not begin with BOS")
    maximum_prefix = context_length - generation_cap
    if maximum_prefix < 2:
        raise ValueError("generation cap leaves no valid prompt boundary")
    truncated = len(original) > maximum_prefix
    retained = (BOS_ID, *original[-(maximum_prefix - 1) :]) if truncated else original
    if len(retained) < 2:
        raise ValueError("tokenized compact prompt is too short")
    decoded_retained = tokenizer.decode(tuple(retained[1:]))
    if not decoded_retained.endswith(compact_prompt_footer(example.task_name)):
        raise ValueError("context allocation does not retain the complete task footer")
    return tuple(retained), len(original), truncated


def tokenize_compact_example(
    example: RemediationExample,
    tokenizer: ProjectTokenizer,
    *,
    context_length: int,
    generation_caps: Mapping[TaskName, int],
) -> CompactTokenizedExample:
    if type(example) is not RemediationExample or type(tokenizer) is not ProjectTokenizer:
        raise TypeError("compact tokenization requires exact project contracts")
    if type(context_length) is not int or context_length < 32:
        raise ValueError("context length must be an integer of at least 32")
    cap = generation_caps.get(example.task_name)
    if type(cap) is not int or not 1 <= cap < context_length:
        raise ValueError("example task has no valid frozen generation cap")
    _prompt_text, target_text = compact_serialized_parts(example)
    target_ids = tokenizer.encode(target_text, add_bos=False, add_eos=True)
    if len(target_ids) > cap:
        raise ValueError("compact target exceeds its frozen task generation cap")
    prompt_ids, original_prompt_count, truncated = retained_compact_prompt_tokens(
        example,
        tokenizer,
        context_length=context_length,
        generation_cap=cap,
    )
    token_ids = (*prompt_ids, *target_ids)
    if len(token_ids) > context_length:
        raise RuntimeError("compact serialization exceeded its model context")
    target_mask = (*([False] * len(prompt_ids)), *([True] * len(target_ids)))
    return CompactTokenizedExample(
        example_id=example.example_id,
        task_name=example.task_name,
        group_id=example.group_id,
        token_ids=token_ids,
        target_mask=target_mask,
        prompt_token_count=original_prompt_count,
        target_token_count=len(target_ids),
        prompt_tokens_retained=len(prompt_ids),
        prompt_truncated=truncated,
    )


def compact_batch_tensors(
    examples: tuple[CompactTokenizedExample, ...], *, context_length: int
) -> tuple[Tensor, Tensor, Tensor]:
    if type(examples) is not tuple or not examples:
        raise ValueError("compact batch requires a non-empty exact tuple")
    if any(type(item) is not CompactTokenizedExample for item in examples):
        raise TypeError("compact batch contains an invalid record")
    width = min(context_length, max(len(item.token_ids) for item in examples))
    input_ids = torch.full((len(examples), width), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.bool)
    target_mask = torch.zeros((len(examples), width), dtype=torch.bool)
    for row, item in enumerate(examples):
        if len(item.token_ids) > width:
            raise ValueError("compact example exceeds its batch context")
        length = len(item.token_ids)
        input_ids[row, :length] = torch.tensor(item.token_ids, dtype=torch.long)
        attention_mask[row, :length] = True
        target_mask[row, :length] = torch.tensor(item.target_mask, dtype=torch.bool)
    return input_ids, attention_mask, target_mask


__all__ = [
    "CompactTokenizedExample",
    "compact_batch_tensors",
    "compact_prompt_footer",
    "compact_serialized_parts",
    "retained_compact_prompt_tokens",
    "supervised_causal_loss",
    "tokenize_compact_example",
]
