"""Tiny-shard learning proof for the from-scratch model."""

from __future__ import annotations

import torch

from reactorbench.model import (
    TransformerConfig,
    causal_language_model_loss,
    initialized_model,
)


def test_tiny_model_overfits_a_two_sequence_shard() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = TransformerConfig(
        model_version="0.1.0",
        layers=1,
        width=32,
        heads=4,
        context_length=8,
        feed_forward_multiplier=2,
        dropout=0.0,
        tie_embeddings=True,
        bias=True,
    )
    model = initialized_model(config, vocab_size=32, seed=9)
    input_ids = torch.tensor(
        ((1, 4, 5, 6, 7, 2, 3, 3), (1, 8, 9, 10, 11, 2, 3, 3)),
        dtype=torch.long,
    )
    attention_mask = torch.tensor(((True, True, True, True, True, True, False, False),) * 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    initial = float(
        causal_language_model_loss(model, input_ids, attention_mask=attention_mask).detach()
    )

    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = causal_language_model_loss(model, input_ids, attention_mask=attention_mask)
        torch.autograd.backward(loss)
        optimizer.step()

    final = float(
        causal_language_model_loss(model, input_ids, attention_mask=attention_mask).detach()
    )
    assert final < 0.20
    assert (initial - final) / initial > 0.90
