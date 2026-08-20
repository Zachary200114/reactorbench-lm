"""Correctness tests for the project-defined causal Transformer."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from reactorbench.model import (
    TransformerConfig,
    causal_language_model_loss,
    exact_parameter_count,
    initialized_model,
    shift_next_token_targets,
)


def _config(**changes: object) -> TransformerConfig:
    payload: dict[str, object] = {
        "model_version": "0.1.0",
        "layers": 1,
        "width": 32,
        "heads": 4,
        "context_length": 16,
        "feed_forward_multiplier": 2,
        "dropout": 0.0,
        "tie_embeddings": True,
        "bias": True,
    }
    payload.update(changes)
    return TransformerConfig.model_validate(payload)


def test_shifted_targets_and_padding_boundary_are_exact() -> None:
    tokens = torch.tensor(((1, 2, 3, 4), (5, 6, 7, 0)), dtype=torch.long)
    mask = torch.tensor(((True, True, True, True), (True, True, True, False)))

    batch = shift_next_token_targets(tokens, mask)

    assert torch.equal(batch.input_ids, torch.tensor(((1, 2, 3), (5, 6, 7))))
    assert torch.equal(batch.target_ids, torch.tensor(((2, 3, 4), (6, 7, 0))))
    assert torch.equal(
        batch.target_mask,
        torch.tensor(((True, True, True), (True, True, False))),
    )


def test_causal_mask_prevents_future_token_influence() -> None:
    model = initialized_model(_config(), vocab_size=64, seed=17)
    model.eval()
    original = torch.tensor(((1, 2, 3, 4, 5),), dtype=torch.long)
    changed = original.clone()
    changed[0, -1] = 9

    with torch.no_grad():
        original_logits = model(original)
        changed_logits = model(changed)

    assert torch.equal(original_logits[:, :-1], changed_logits[:, :-1])
    assert not torch.equal(original_logits[:, -1], changed_logits[:, -1])


def test_padding_tokens_do_not_change_causal_loss() -> None:
    model = initialized_model(_config(), vocab_size=64, seed=23)
    mask = torch.tensor(((True, True, True, False),))
    first = torch.tensor(((1, 2, 3, 4),), dtype=torch.long)
    second = torch.tensor(((1, 2, 3, 55),), dtype=torch.long)

    first_loss = causal_language_model_loss(model, first, attention_mask=mask)
    second_loss = causal_language_model_loss(model, second, attention_mask=mask)

    assert torch.equal(first_loss, second_loss)


def test_exact_parameter_formula_matches_allocated_models() -> None:
    for config in (
        _config(),
        _config(layers=2, width=48, heads=6, tie_embeddings=False),
        _config(layers=2, width=64, heads=8, bias=False),
    ):
        model = initialized_model(config, vocab_size=512, seed=1)
        allocated = sum(parameter.numel() for parameter in model.parameters())
        assert exact_parameter_count(config, vocab_size=512) == allocated


def test_initialization_is_reproducible_without_advancing_global_rng() -> None:
    torch.manual_seed(101)
    before = torch.random.get_rng_state().clone()
    first = initialized_model(_config(), vocab_size=64, seed=7)
    after = torch.random.get_rng_state().clone()
    second = initialized_model(_config(), vocab_size=64, seed=7)

    assert torch.equal(before, after)
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )


def test_bounded_generation_is_deterministic() -> None:
    model = initialized_model(_config(), vocab_size=64, seed=31)
    prefix = torch.tensor(((1, 2, 3),), dtype=torch.long)

    first = model.generate(prefix, max_new_tokens=4)
    second = model.generate(prefix, max_new_tokens=4)

    assert torch.equal(first, second)
    assert first.shape == (1, 7)


def test_cached_decode_matches_full_forward_logits() -> None:
    model = initialized_model(_config(), vocab_size=64, seed=37)
    model.eval()
    prefix = torch.tensor(((1, 2, 3), (4, 5, 6)), dtype=torch.long)
    next_tokens = torch.tensor(((7,), (8,)), dtype=torch.long)

    with torch.no_grad():
        full_prefix = model(prefix)[:, -1, :]
        cached_prefix, caches = model.prefill_cache(prefix)
        key_mask = torch.ones((2, 4), dtype=torch.bool)
        cached_next, updated = model.decode_step(
            next_tokens,
            position=3,
            caches=caches,
            key_mask=key_mask,
        )
        full_next = model(torch.cat((prefix, next_tokens), dim=1))[:, -1, :]

    assert torch.equal(cached_prefix, full_prefix)
    assert torch.allclose(cached_next, full_next, atol=1e-6, rtol=1e-6)
    assert all(cache.key.shape[2] == 4 for cache in updated)


def test_cached_decode_rejects_malformed_steps_and_cache_shapes() -> None:
    model = initialized_model(_config(), vocab_size=64, seed=41)
    prefix = torch.tensor(((1, 2, 3),), dtype=torch.long)
    _logits, caches = model.prefill_cache(prefix)
    with pytest.raises(TypeError, match="one-token long tensor"):
        model.decode_step(
            torch.tensor(((4, 5),), dtype=torch.long),
            position=3,
            caches=caches,
            key_mask=torch.ones((1, 4), dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="outside the model context"):
        model.decode_step(
            torch.tensor(((4,),), dtype=torch.long),
            position=16,
            caches=caches,
            key_mask=torch.ones((1, 4), dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="one exact cache per layer"):
        model.decode_step(
            torch.tensor(((4,),), dtype=torch.long),
            position=3,
            caches=cast(Any, ()),
            key_mask=torch.ones((1, 4), dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="wrong shape"):
        model.decode_step(
            torch.tensor(((4,),), dtype=torch.long),
            position=3,
            caches=caches,
            key_mask=torch.ones((1, 3), dtype=torch.bool),
        )
