"""Project-defined decoder-only causal Transformer built from PyTorch primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .config import TransformerConfig


@dataclass(frozen=True)
class CausalBatch:
    input_ids: Tensor
    target_ids: Tensor
    input_mask: Tensor
    target_mask: Tensor


@dataclass(frozen=True)
class AttentionCache:
    """Per-layer inference-only keys and values for bounded autoregressive decoding."""

    key: Tensor
    value: Tensor


def shift_next_token_targets(
    input_ids: Tensor, attention_mask: Tensor | None = None
) -> CausalBatch:
    """Shift one token exactly once and preserve the padding-loss boundary."""

    if type(input_ids) is not Tensor or input_ids.dtype != torch.long or input_ids.ndim != 2:
        raise TypeError("input_ids must be a rank-2 torch.long tensor")
    if input_ids.shape[1] < 2:
        raise ValueError("next-token targets require at least two tokens")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    if (
        type(attention_mask) is not Tensor
        or attention_mask.dtype != torch.bool
        or attention_mask.shape != input_ids.shape
    ):
        raise TypeError("attention_mask must be a matching boolean tensor")
    return CausalBatch(
        input_ids=input_ids[:, :-1],
        target_ids=input_ids[:, 1:],
        input_mask=attention_mask[:, :-1],
        target_mask=attention_mask[:, 1:],
    )


class CausalSelfAttention(nn.Module):
    """Explicit masked multi-head self-attention without a pretrained component."""

    causal_mask: Tensor

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=config.bias)
        self.output = nn.Linear(config.width, config.width, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", mask.view(1, 1, *mask.shape), persistent=False)

    def _project(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, sequence, width = hidden.shape
        qkv = self.qkv(hidden)
        query, key, value = qkv.split(width, dim=-1)

        def heads(value_: Tensor) -> Tensor:
            return value_.view(batch, sequence, self.heads, self.head_width).transpose(1, 2)

        return heads(query), heads(key), heads(value)

    def _output(self, attended: Tensor) -> Tensor:
        batch, _heads, sequence, _head_width = attended.shape
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, -1)
        return cast(Tensor, self.residual_dropout(self.output(attended)))

    def forward_with_cache(
        self, hidden: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, AttentionCache]:
        query, key, value = self._project(hidden)
        sequence = hidden.shape[1]
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_width)
        allowed = self.causal_mask[:, :, :sequence, :sequence]
        allowed = allowed & attention_mask[:, None, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = functional.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        attended = weights @ value
        return self._output(attended), AttentionCache(key=key, value=value)

    def forward_step(
        self,
        hidden: Tensor,
        cache: AttentionCache,
        key_mask: Tensor,
    ) -> tuple[Tensor, AttentionCache]:
        if hidden.shape[1] != 1 or type(cache) is not AttentionCache:
            raise ValueError("cached attention step requires one token and an exact cache")
        query, key, value = self._project(hidden)
        combined_key = torch.cat((cache.key, key), dim=2)
        combined_value = torch.cat((cache.value, value), dim=2)
        if key_mask.shape != (hidden.shape[0], combined_key.shape[2]):
            raise ValueError("cached attention key mask has the wrong shape")
        scores = query @ combined_key.transpose(-2, -1) / math.sqrt(self.head_width)
        scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = self.attention_dropout(functional.softmax(scores, dim=-1))
        return self._output(weights @ combined_value), AttentionCache(
            key=combined_key,
            value=combined_value,
        )

    def forward(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        output, _cache = self.forward_with_cache(hidden, attention_mask)
        return output


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        hidden_width = config.width * config.feed_forward_multiplier
        self.input = nn.Linear(config.width, hidden_width, bias=config.bias)
        self.output = nn.Linear(hidden_width, config.width, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        return cast(
            Tensor,
            self.dropout(self.output(functional.gelu(self.input(hidden), approximate="tanh"))),
        )


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.width, bias=config.bias)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden), attention_mask)
        return cast(Tensor, hidden + self.feed_forward(self.feed_forward_norm(hidden)))

    def forward_with_cache(
        self, hidden: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, AttentionCache]:
        attended, cache = self.attention.forward_with_cache(
            self.attention_norm(hidden), attention_mask
        )
        hidden = hidden + attended
        return cast(Tensor, hidden + self.feed_forward(self.feed_forward_norm(hidden))), cache

    def forward_step(
        self, hidden: Tensor, cache: AttentionCache, key_mask: Tensor
    ) -> tuple[Tensor, AttentionCache]:
        attended, updated = self.attention.forward_step(
            self.attention_norm(hidden), cache, key_mask
        )
        hidden = hidden + attended
        return cast(Tensor, hidden + self.feed_forward(self.feed_forward_norm(hidden))), updated


class TransformerLM(nn.Module):
    """A randomly initialized, pre-normalization decoder-only language model."""

    def __init__(self, config: TransformerConfig, *, vocab_size: int) -> None:
        super().__init__()
        if type(config) is not TransformerConfig:
            raise TypeError("config must be an exact TransformerConfig")
        if type(vocab_size) is not int or not 8 <= vocab_size <= 65_536:
            raise ValueError("vocab_size must be an integer in [8, 65536]")
        self.config = config
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.context_length, config.width)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.width, bias=config.bias)
        self.lm_head = nn.Linear(config.width, vocab_size, bias=False)
        self.apply(self._initialize)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _validate_inputs(self, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        if type(input_ids) is not Tensor or input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise TypeError("input_ids must be a rank-2 torch.long tensor")
        if not 1 <= input_ids.shape[1] <= self.config.context_length:
            raise ValueError("input sequence length is outside the configured context")
        if input_ids.numel() and (
            input_ids.min().item() < 0 or input_ids.max().item() >= self.vocab_size
        ):
            raise ValueError("input token ID is outside the configured vocabulary")
        if attention_mask is None:
            return torch.ones_like(input_ids, dtype=torch.bool)
        if (
            type(attention_mask) is not Tensor
            or attention_mask.dtype != torch.bool
            or attention_mask.shape != input_ids.shape
        ):
            raise TypeError("attention_mask must be a matching boolean tensor")
        if not torch.all(attention_mask.any(dim=1)):
            raise ValueError("every sequence must expose at least one non-padding token")
        return attention_mask

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        attention_mask = self._validate_inputs(input_ids, attention_mask)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        return cast(Tensor, self.lm_head(self.final_norm(hidden)))

    @torch.no_grad()
    def prefill_cache(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, tuple[AttentionCache, ...]]:
        """Encode a prompt once and return next-token logits plus layer caches."""

        attention_mask = self._validate_inputs(input_ids, attention_mask)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        hidden = self.embedding_dropout(hidden)
        caches: list[AttentionCache] = []
        for block_module in self.blocks:
            block = cast(TransformerBlock, block_module)
            hidden, cache = block.forward_with_cache(hidden, attention_mask)
            caches.append(cache)
        logits = self.lm_head(self.final_norm(hidden))[:, -1, :]
        return cast(Tensor, logits), tuple(caches)

    @torch.no_grad()
    def decode_step(
        self,
        input_ids: Tensor,
        *,
        position: int,
        caches: tuple[AttentionCache, ...],
        key_mask: Tensor,
    ) -> tuple[Tensor, tuple[AttentionCache, ...]]:
        """Advance an exact cache by one token without recomputing its prefix."""

        if (
            type(input_ids) is not Tensor
            or input_ids.dtype != torch.long
            or input_ids.ndim != 2
            or input_ids.shape[1] != 1
        ):
            raise TypeError("cached decode input must be a rank-2 one-token long tensor")
        if type(position) is not int or not 0 <= position < self.config.context_length:
            raise ValueError("cached decode position is outside the model context")
        if len(caches) != len(self.blocks) or any(
            type(cache) is not AttentionCache for cache in caches
        ):
            raise ValueError("cached decode requires one exact cache per layer")
        hidden = (
            self.token_embedding(input_ids)
            + self.position_embedding.weight[position][None, None, :]
        )
        hidden = self.embedding_dropout(hidden)
        updated: list[AttentionCache] = []
        for block_module, cache in zip(self.blocks, caches, strict=True):
            block = cast(TransformerBlock, block_module)
            hidden, next_cache = block.forward_step(hidden, cache, key_mask)
            updated.append(next_cache)
        logits = self.lm_head(self.final_norm(hidden))[:, -1, :]
        return cast(Tensor, logits), tuple(updated)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float | None = None,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Bounded autoregressive decoding with greedy or temperature/top-k sampling."""

        if type(max_new_tokens) is not int or not 0 <= max_new_tokens <= self.config.context_length:
            raise ValueError("max_new_tokens is outside the configured context")
        if input_ids.shape[1] + max_new_tokens > self.config.context_length:
            raise ValueError("generation would exceed the configured context")
        if temperature is not None and (type(temperature) is not float or temperature <= 0.0):
            raise ValueError("temperature must be a positive float or None")
        if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= self.vocab_size):
            raise ValueError("top_k is outside the vocabulary")
        generated = input_ids
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits = self(generated)[:, -1, :]
                if temperature is None:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                        logits = logits.masked_fill(logits < threshold, float("-inf"))
                    probabilities = functional.softmax(logits, dim=-1)
                    next_token = torch.multinomial(
                        probabilities, num_samples=1, generator=generator
                    )
                generated = torch.cat((generated, next_token), dim=1)
        finally:
            self.train(was_training)
        return generated


def causal_language_model_loss(
    model: TransformerLM,
    input_ids: Tensor,
    *,
    attention_mask: Tensor | None = None,
) -> Tensor:
    shifted = shift_next_token_targets(input_ids, attention_mask)
    logits = model(shifted.input_ids, shifted.input_mask)
    losses = functional.cross_entropy(
        logits.reshape(-1, model.vocab_size),
        shifted.target_ids.reshape(-1),
        reduction="none",
    ).view_as(shifted.target_ids)
    selected = losses.masked_select(shifted.target_mask)
    if not selected.numel():
        raise ValueError("next-token loss requires at least one visible target")
    return selected.mean()


def exact_parameter_count(config: TransformerConfig, *, vocab_size: int) -> int:
    """Calculate the architecture parameter count without allocating a model."""

    width = config.width
    multiplier = config.feed_forward_multiplier
    embeddings = vocab_size * width + config.context_length * width
    per_block = (4 + 2 * multiplier) * width * width
    if config.bias:
        per_block += (multiplier + 9) * width
    else:
        per_block += 2 * width
    final_norm = 2 * width if config.bias else width
    untied_head = 0 if config.tie_embeddings else vocab_size * width
    return embeddings + config.layers * per_block + final_norm + untied_head


def initialized_model(config: TransformerConfig, *, vocab_size: int, seed: int) -> TransformerLM:
    """Initialize deterministically without advancing the caller's CPU RNG stream."""

    if type(seed) is not int or not 0 <= seed <= 4_294_967_295:
        raise ValueError("seed must be a uint32 integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TransformerLM(config, vocab_size=vocab_size)
    return model


__all__ = [
    "CausalBatch",
    "TransformerLM",
    "causal_language_model_loss",
    "exact_parameter_count",
    "initialized_model",
    "shift_next_token_targets",
]
