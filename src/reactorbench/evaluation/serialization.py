"""Canonical Phase 5 prompt/target serialization and loss masking."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor

from reactorbench.model import TransformerLM, shift_next_token_targets
from reactorbench.tokenizer import BOS_ID, PAD_ID, ProjectTokenizer

from .config import SerializationConfig
from .data import ExperimentExample


@dataclass(frozen=True)
class TokenizedExample:
    example_id: str
    token_ids: tuple[int, ...]
    target_mask: tuple[bool, ...]
    truncated_prompt: bool


def serialized_parts(example: ExperimentExample, config: SerializationConfig) -> tuple[str, str]:
    if type(example) is not ExperimentExample or type(config) is not SerializationConfig:
        raise TypeError("serialization requires exact example and config objects")
    prefix = (
        f"{config.prompt_prefix}\nTASK={example.task_name.value}\n"
        f"{example.prompt_text}\n{config.target_prefix}\n"
    )
    target = f"{example.target_text}\n{config.record_separator}"
    return prefix, target


def tokenize_example(
    example: ExperimentExample,
    tokenizer: ProjectTokenizer,
    config: SerializationConfig,
    *,
    context_length: int,
) -> TokenizedExample:
    if type(tokenizer) is not ProjectTokenizer:
        raise TypeError("tokenizer must be an exact ProjectTokenizer")
    if type(context_length) is not int or context_length < 8:
        raise ValueError("context_length must be an integer of at least eight")
    prefix_text, target_text = serialized_parts(example, config)
    prefix_ids = tokenizer.encode(prefix_text, add_bos=True, add_eos=False)
    target_ids = tokenizer.encode(target_text, add_bos=False, add_eos=True)
    if len(target_ids) >= context_length:
        raise ValueError("complete target does not fit the configured model context")
    truncated = len(prefix_ids) + len(target_ids) > context_length
    if truncated:
        available = context_length - len(target_ids)
        if available < 2:
            raise ValueError("model context cannot retain a valid prompt boundary")
        prefix_ids = (BOS_ID, *prefix_ids[-(available - 1) :])
    token_ids = (*prefix_ids, *target_ids)
    target_mask = (*([False] * len(prefix_ids)), *([True] * len(target_ids)))
    if len(token_ids) > context_length or len(token_ids) != len(target_mask):
        raise RuntimeError("serialized example violated its context or mask boundary")
    return TokenizedExample(
        example_id=example.example_id,
        token_ids=token_ids,
        target_mask=target_mask,
        truncated_prompt=truncated,
    )


def batch_tensors(
    examples: tuple[TokenizedExample, ...],
    *,
    context_length: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if type(examples) is not tuple or not examples:
        raise ValueError("batch examples must be a non-empty exact tuple")
    if any(type(item) is not TokenizedExample for item in examples):
        raise TypeError("batch contains a non-TokenizedExample item")
    width = min(context_length, max(len(item.token_ids) for item in examples))
    input_ids = torch.full((len(examples), width), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.bool)
    target_mask = torch.zeros((len(examples), width), dtype=torch.bool)
    for row, item in enumerate(examples):
        if len(item.token_ids) > width:
            raise ValueError("tokenized example exceeds batch context")
        length = len(item.token_ids)
        input_ids[row, :length] = torch.tensor(item.token_ids, dtype=torch.long)
        attention_mask[row, :length] = True
        target_mask[row, :length] = torch.tensor(item.target_mask, dtype=torch.bool)
    return input_ids, attention_mask, target_mask


def supervised_causal_loss(
    model: TransformerLM,
    input_ids: Tensor,
    attention_mask: Tensor,
    target_mask: Tensor,
) -> Tensor:
    if type(target_mask) is not Tensor or target_mask.dtype != torch.bool:
        raise TypeError("target_mask must be a boolean tensor")
    if target_mask.shape != input_ids.shape or torch.any(target_mask & ~attention_mask):
        raise ValueError("target mask must align with visible input tokens")
    shifted = shift_next_token_targets(input_ids, attention_mask)
    selected_targets = target_mask[:, 1:] & shifted.target_mask
    if not selected_targets.any():
        raise ValueError("supervised loss requires at least one target token")
    logits = model(shifted.input_ids, shifted.input_mask)
    losses = functional.cross_entropy(
        logits.reshape(-1, model.vocab_size),
        shifted.target_ids.reshape(-1),
        reduction="none",
    ).view_as(shifted.target_ids)
    return losses.masked_select(selected_targets).mean()


__all__ = [
    "TokenizedExample",
    "batch_tensors",
    "serialized_parts",
    "supervised_causal_loss",
    "tokenize_example",
]
