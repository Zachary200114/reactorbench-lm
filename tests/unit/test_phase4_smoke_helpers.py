"""Focused tests for the Phase 4 smoke-run orchestration boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from reactorbench.model import exact_parameter_count, initialized_model, load_phase4_config
from reactorbench.tokenizer import ProjectTokenizer
from reactorbench.training.smoke import (
    ModelTierParameterCounts,
    SmokeRunReport,
    _causal_mask_probe,
    _device,
    _report,
    _smoke_batch,
    _strict_json_object,
    _tensor_sha256,
)

CONFIG_PATH = Path("configs/model/phase4-smoke-v0.1.0.toml")


def test_smoke_batch_is_bounded_and_padding_aware(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = object.__new__(ProjectTokenizer)

    def encode(_self: ProjectTokenizer, text: str, **_kwargs: object) -> tuple[int, ...]:
        return tuple(int(value) for value in text.split(","))

    monkeypatch.setattr(ProjectTokenizer, "encode", encode)
    input_ids, attention_mask = _smoke_batch(
        tokenizer,
        ("1,2,3,4,5", "6,7,8"),
        document_count=2,
        context_length=4,
    )

    assert torch.equal(input_ids, torch.tensor(((1, 2, 3, 4), (6, 7, 8, 3))))
    assert torch.equal(
        attention_mask,
        torch.tensor(((True, True, True, True), (True, True, True, False))),
    )

    with pytest.raises(ValueError, match="document count"):
        _smoke_batch(tokenizer, ("1,2,3",), document_count=2, context_length=4)
    with pytest.raises(ValueError, match="fewer than two"):
        _smoke_batch(tokenizer, ("1",), document_count=1, context_length=4)


def test_causal_probe_and_tensor_hash_are_deterministic() -> None:
    config = load_phase4_config(CONFIG_PATH).smoke_model.model_copy(
        update={"layers": 1, "width": 32, "heads": 4, "context_length": 16}
    )
    model = initialized_model(config, vocab_size=64, seed=5)
    input_ids = torch.tensor(((1, 2, 3, 4, 5),), dtype=torch.long)

    assert _causal_mask_probe(model, input_ids)
    assert _tensor_sha256(input_ids) == _tensor_sha256(input_ids.clone())
    assert _tensor_sha256(input_ids) != _tensor_sha256(input_ids + 1)
    with pytest.raises(ValueError, match="at least three"):
        _causal_mask_probe(model, input_ids[:, :2])


def test_smoke_report_binds_measurements_dependencies_and_parameter_tiers() -> None:
    config = load_phase4_config(CONFIG_PATH)
    counts = ModelTierParameterCounts(
        smoke=exact_parameter_count(config.smoke_model, vocab_size=2048),
        pilot=exact_parameter_count(config.pilot_model, vocab_size=2048),
        main=exact_parameter_count(config.main_model, vocab_size=2048),
    )
    report = _report(
        config=config,
        source_commit="abcdef0",
        candidate_sha256="1" * 64,
        corpus_sha256="2" * 64,
        tokenizer_sha256="3" * 64,
        checkpoint_sha256="4" * 64,
        smoke_inputs_sha256="5" * 64,
        logits_sha256="6" * 64,
        counts=counts,
        sequence_length=128,
        target_tokens=508,
        initial_loss=8.0,
        final_loss=0.5,
        loss_curve=(8.0, 2.0, 0.5),
        elapsed_seconds=2.0,
    )

    assert report.parameter_counts == counts
    assert report.loss_reduction_fraction == pytest.approx(0.9375)
    assert report.tokens_per_second == pytest.approx(76_200.0)
    assert SmokeRunReport.model_validate_json(report.model_dump_json(round_trip=True)) == report

    tampered = report.model_dump(mode="python", round_trip=True)
    tampered["final_loss"] = 0.25
    with pytest.raises(ValidationError, match="checksum"):
        SmokeRunReport.model_validate(tampered)
    with pytest.raises(ValidationError, match="increase monotonically"):
        ModelTierParameterCounts(smoke=2, pilot=2, main=3)


def test_safe_json_and_device_boundaries_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _strict_json_object(b'{"value":1}') == {"value": 1}
    with pytest.raises(ValueError, match="duplicate"):
        _strict_json_object(b'{"value":1,"value":2}')
    with pytest.raises(ValueError, match="non-finite"):
        _strict_json_object(b'{"value":NaN}')
    with pytest.raises(ValueError, match="one object"):
        _strict_json_object(b"[]")

    assert _device("cpu") == torch.device("cpu")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        _device("mps")
