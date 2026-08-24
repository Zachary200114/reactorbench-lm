from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from pydantic import ValidationError
from torch import Tensor

from reactorbench.evaluation.compact import CompactTargetContext, compact_target_json
from reactorbench.model import TransformerConfig, TransformerLM, initialized_model
from reactorbench.remediation.data import RemediationExample
from reactorbench.remediation.decoding import (
    CompactDecodeError,
    CompactPathPrediction,
    DecodePath,
    DualPathCompactPrediction,
    compact_wire_parse_success,
    decode_compact_example,
    decode_compact_examples,
)
from reactorbench.schemas.base import canonical_sha256
from reactorbench.schemas.enums import ActionLabel, TaskName
from reactorbench.schemas.target import NextActionTarget
from reactorbench.tokenizer import BOS_ID, EOS_ID, ProjectTokenizer

_TARGET = "RB2|next_action|7"
_PROMPT_FOOTER = "TASK=next_action\n<|target|>\n"


class StepClock:
    def __init__(self, *, step: float = 0.25) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


def _context() -> CompactTargetContext:
    return CompactTargetContext(
        task_name=TaskName.NEXT_ACTION,
        visible_fact_refs=("o-0000",),
    )


def _example(
    identifier: str = "example:decode",
    *,
    prompt_text: str = "fictional visible evidence",
    hidden_target: str = "unused truth A",
) -> RemediationExample:
    # Deliberately omit every unused RemediationExample field. If the decoder crosses
    # the narrow prompt/context boundary, this test fixture fails with AttributeError.
    return RemediationExample.model_construct(
        example_id=identifier,
        task_name=TaskName.NEXT_ACTION,
        prompt_text=prompt_text,
        compact_context=_context(),
        compact_target=hidden_target,
        checksum_sha256=("a" if identifier.endswith("decode") else "b") * 64,
    )


def _tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    long_prompt: bool = False,
) -> tuple[ProjectTokenizer, dict[str, int]]:
    characters = tuple(sorted({*_TARGET, *_PROMPT_FOOTER, "P", "X"}))
    by_character = {character: index + 4 for index, character in enumerate(characters)}
    by_token = {token_id: character for character, token_id in by_character.items()}
    tokenizer = object.__new__(ProjectTokenizer)
    tokenizer.manifest = cast(
        Any,
        SimpleNamespace(
            actual_vocab_size=len(characters) + 4,
            checksum_sha256="c" * 64,
        ),
    )

    def encode(
        _self: ProjectTokenizer,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> tuple[int, ...]:
        if text.startswith("<|prompt|>"):
            prompt_body = (
                "P" * (100 - len(_PROMPT_FOOTER)) + _PROMPT_FOOTER
                if long_prompt
                else _PROMPT_FOOTER
            )
            body = tuple(by_character[character] for character in prompt_body)
        else:
            body = tuple(by_character[character] for character in text)
        return (
            *((BOS_ID,) if add_bos else ()),
            *body,
            *((EOS_ID,) if add_eos else ()),
        )

    def decode(_self: ProjectTokenizer, token_ids: tuple[int, ...]) -> str:
        return "".join(by_token[token_id] for token_id in token_ids)

    monkeypatch.setattr(ProjectTokenizer, "encode", encode)
    monkeypatch.setattr(ProjectTokenizer, "decode", decode)
    return tokenizer, by_character


def _model(vocab_size: int, *, context_length: int = 64) -> TransformerLM:
    return initialized_model(
        TransformerConfig(
            model_version="0.3.0",
            layers=1,
            width=16,
            heads=4,
            context_length=context_length,
            feed_forward_multiplier=2,
            dropout=0.0,
            tie_embeddings=True,
            bias=True,
        ),
        vocab_size=vocab_size,
        seed=73,
    )


def _script_logits(
    monkeypatch: pytest.MonkeyPatch,
    model: TransformerLM,
    by_character: dict[str, int],
    *,
    prompt_length: int = len(_PROMPT_FOOTER) + 1,
    prefer_eos: bool = False,
    nonfinite: bool = False,
) -> None:
    target_ids = tuple(by_character[character] for character in _TARGET)
    prefix_lengths: dict[int, int] = {}

    def logits(index: int, device: torch.device) -> Tensor:
        result = torch.zeros((1, model.vocab_size), dtype=torch.float32, device=device)
        result[0, by_character["X"]] = 10.0
        if index < len(target_ids):
            result[0, target_ids[index]] = 9.0
        else:
            result[0, EOS_ID] = 9.0
        if prefer_eos:
            result[0, EOS_ID] = 20.0
        if nonfinite:
            result[0, by_character["X"]] = torch.nan
        return result

    def prefill(
        self: TransformerLM,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[()]]:
        del attention_mask
        prefix_lengths[id(self)] = input_ids.shape[1]
        return logits(0, input_ids.device), ()

    def step(
        self: TransformerLM,
        input_ids: Tensor,
        *,
        position: int,
        caches: tuple[object, ...],
        key_mask: Tensor,
    ) -> tuple[Tensor, tuple[()]]:
        del caches, key_mask
        index = position - prefix_lengths[id(self)] + 1
        return logits(index, input_ids.device), ()

    def forward(
        self: TransformerLM,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        del attention_mask
        output = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], self.vocab_size),
            dtype=torch.float32,
            device=input_ids.device,
        )
        output[:, -1, :] = logits(input_ids.shape[1] - prompt_length, input_ids.device)
        return output

    monkeypatch.setattr(TransformerLM, "prefill_cache", prefill)
    monkeypatch.setattr(TransformerLM, "decode_step", step)
    monkeypatch.setattr(TransformerLM, "forward", forward)


def _decode(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_cache: bool = True,
    cap: int | None = None,
    prefer_eos: bool = False,
) -> DualPathCompactPrediction:
    tokenizer, by_character = _tokenizer(monkeypatch)
    model = _model(tokenizer.vocab_size)
    _script_logits(monkeypatch, model, by_character, prefer_eos=prefer_eos)
    return decode_compact_example(
        model,
        tokenizer,
        _example(),
        generation_caps={TaskName.NEXT_ACTION: cap or len(_TARGET) + 1},
        device=torch.device("cpu"),
        use_cache=use_cache,
        clock=StepClock(),
    )


def test_dual_path_reports_unconstrained_failure_and_constrained_canonical_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _decode(monkeypatch)

    assert result.unconstrained.path is DecodePath.UNCONSTRAINED
    assert result.unconstrained.generated_text == "X" * (len(_TARGET) + 1)
    assert result.unconstrained.generation_cap_exhausted
    assert not result.unconstrained.eos_emitted
    assert not result.unconstrained.compact_parse_success
    assert not result.unconstrained.schema_valid
    assert result.unconstrained.canonical_target_json is None

    assert result.constrained.path is DecodePath.CONSTRAINED
    assert result.constrained.generated_text == _TARGET
    assert result.constrained.eos_emitted
    assert not result.constrained.generation_cap_exhausted
    assert result.constrained.compact_parse_success
    assert result.constrained.schema_valid
    assert result.constrained.canonical_target_json == compact_target_json(
        _TARGET,
        context=_context(),
    )
    assert result.constrained.generated_token_count == len(_TARGET)
    assert result.constrained.selected_token_count == len(_TARGET) + 1
    assert 0.0 < result.constrained.selected_token_geometric_mean_probability < 1.0
    assert result.unconstrained.elapsed_seconds == 0.25
    assert result.constrained.elapsed_seconds == 0.25


def test_wire_parse_is_distinct_from_strict_schema_validation() -> None:
    assert compact_wire_parse_success(
        "RB2|next_action|NOT_AN_ACTION",
        task_name=TaskName.NEXT_ACTION,
    )
    assert not compact_wire_parse_success(
        "RB2|fault_family|NOT_AN_ACTION",
        task_name=TaskName.NEXT_ACTION,
    )
    assert not compact_wire_parse_success(
        "RB2|next_action|",
        task_name=TaskName.NEXT_ACTION,
    )
    assert not compact_wire_parse_success(
        "RB2|next_action|CONTINUE_MONITORING\n",
        task_name=TaskName.NEXT_ACTION,
    )
    with pytest.raises(TypeError, match="exact text and TaskName"):
        compact_wire_parse_success(cast(Any, b"RB2"), task_name=TaskName.NEXT_ACTION)


def test_cached_and_full_forward_decoding_have_token_and_probability_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached = _decode(monkeypatch, use_cache=True)
    uncached = _decode(monkeypatch, use_cache=False)

    for cached_path, uncached_path in (
        (cached.unconstrained, uncached.unconstrained),
        (cached.constrained, uncached.constrained),
    ):
        assert cached_path.generated_token_ids == uncached_path.generated_token_ids
        assert cached_path.generated_text == uncached_path.generated_text
        assert cached_path.eos_emitted == uncached_path.eos_emitted
        assert cached_path.schema_valid == uncached_path.schema_valid
        assert cached_path.selected_token_geometric_mean_probability == pytest.approx(
            uncached_path.selected_token_geometric_mean_probability,
            abs=1e-7,
        )
        assert cached_path.used_cache
        assert not uncached_path.used_cache


def test_decoder_does_not_mutate_global_rng_and_restores_training_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer, by_character = _tokenizer(monkeypatch)
    model = _model(tokenizer.vocab_size)
    _script_logits(monkeypatch, model, by_character)
    model.train()
    torch_before = torch.random.get_rng_state().clone()
    python_before = random.getstate()

    first = decode_compact_example(
        model,
        tokenizer,
        _example(),
        generation_caps={TaskName.NEXT_ACTION: len(_TARGET) + 1},
        device=torch.device("cpu"),
        clock=StepClock(),
    )
    torch_after = torch.random.get_rng_state().clone()
    python_after = random.getstate()
    second = decode_compact_example(
        model,
        tokenizer,
        _example(),
        generation_caps={TaskName.NEXT_ACTION: len(_TARGET) + 1},
        device=torch.device("cpu"),
        clock=StepClock(),
    )

    assert torch.equal(torch_before, torch_after)
    assert python_before == python_after
    assert model.training
    assert first == second


def test_same_visible_context_with_different_hidden_truth_decodes_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer, by_character = _tokenizer(monkeypatch)
    model = _model(tokenizer.vocab_size)
    _script_logits(monkeypatch, model, by_character)
    examples = (
        _example("example:first", hidden_target="truth one"),
        _example("example:second", hidden_target="completely different truth"),
    )

    results = decode_compact_examples(
        model,
        tokenizer,
        examples,
        generation_caps={TaskName.NEXT_ACTION: len(_TARGET) + 1},
        device=torch.device("cpu"),
        clock=StepClock(),
    )

    assert tuple(result.example_id for result in results) == (
        "example:first",
        "example:second",
    )
    assert results[0].constrained == results[1].constrained
    assert results[0].unconstrained == results[1].unconstrained


def test_premature_unconstrained_eos_is_safe_but_constraint_delays_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _decode(monkeypatch, prefer_eos=True)

    assert result.unconstrained.eos_emitted
    assert result.unconstrained.generated_token_ids == ()
    assert result.unconstrained.generated_text == ""
    assert not result.unconstrained.compact_parse_success
    assert not result.unconstrained.schema_valid
    assert result.unconstrained.selected_token_count == 1
    assert result.constrained.generated_text == _TARGET
    assert result.constrained.schema_valid
    assert result.constrained.eos_emitted


def test_short_cap_reports_exhaustion_without_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _decode(monkeypatch, cap=2)

    for path in (result.unconstrained, result.constrained):
        assert path.generation_cap_exhausted
        assert not path.eos_emitted
        assert path.selected_token_count == 2
        assert not path.compact_parse_success
        assert not path.schema_valid
        assert path.canonical_target_json is None


def test_prompt_suffix_retention_uses_task_cap_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer, by_character = _tokenizer(monkeypatch, long_prompt=True)
    model = _model(tokenizer.vocab_size)
    _script_logits(monkeypatch, model, by_character)
    cap = len(_TARGET) + 1
    result = decode_compact_example(
        model,
        tokenizer,
        _example(prompt_text="LONG fictional prompt"),
        generation_caps={TaskName.NEXT_ACTION: cap},
        device=torch.device("cpu"),
        clock=StepClock(),
    )

    assert result.unconstrained.prompt_token_count == 101
    assert result.unconstrained.prompt_tokens_retained == model.config.context_length - cap
    assert result.unconstrained.prompt_truncated
    assert result.constrained.prompt_tokens_retained == (
        result.unconstrained.prompt_tokens_retained
    )


@pytest.mark.parametrize(
    "caps",
    [
        cast(Any, []),
        {},
        {TaskName.NEXT_ACTION: 0},
        {TaskName.NEXT_ACTION: 63},
        cast(Any, {"next_action": 10}),
        cast(Any, {TaskName.NEXT_ACTION: True}),
    ],
)
def test_generation_cap_mapping_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    caps: dict[TaskName, int],
) -> None:
    tokenizer, by_character = _tokenizer(monkeypatch)
    model = _model(tokenizer.vocab_size)
    _script_logits(monkeypatch, model, by_character)

    with pytest.raises((TypeError, ValueError), match=r"generation cap|generation-cap"):
        decode_compact_example(
            model,
            tokenizer,
            _example(),
            generation_caps=caps,
            device=torch.device("cpu"),
            clock=StepClock(),
        )


def test_batch_and_runtime_boundaries_reject_invalid_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer, by_character = _tokenizer(monkeypatch)
    model = _model(tokenizer.vocab_size)
    _script_logits(monkeypatch, model, by_character)
    caps = {TaskName.NEXT_ACTION: len(_TARGET) + 1}

    with pytest.raises(TypeError, match="non-empty bounded"):
        decode_compact_examples(
            model,
            tokenizer,
            (),
            generation_caps=caps,
            device=torch.device("cpu"),
            clock=StepClock(),
        )
    duplicate = _example()
    with pytest.raises(ValueError, match="unique IDs"):
        decode_compact_examples(
            model,
            tokenizer,
            (duplicate, duplicate),
            generation_caps=caps,
            device=torch.device("cpu"),
            clock=StepClock(),
        )
    with pytest.raises(TypeError, match="exact RemediationExample"):
        decode_compact_example(
            model,
            tokenizer,
            cast(Any, object()),
            generation_caps=caps,
            device=torch.device("cpu"),
            clock=StepClock(),
        )
    with pytest.raises(ValueError, match="parameter device"):
        decode_compact_example(
            model,
            tokenizer,
            _example(),
            generation_caps=caps,
            device=torch.device("meta"),
            clock=StepClock(),
        )


def test_nonfinite_logits_and_backward_clock_fail_safely_and_restore_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer, by_character = _tokenizer(monkeypatch)
    model = _model(tokenizer.vocab_size)
    _script_logits(monkeypatch, model, by_character, nonfinite=True)
    model.train()
    with pytest.raises(CompactDecodeError, match="non-finite"):
        decode_compact_example(
            model,
            tokenizer,
            _example(),
            generation_caps={TaskName.NEXT_ACTION: len(_TARGET) + 1},
            device=torch.device("cpu"),
            clock=StepClock(),
        )
    assert model.training

    _script_logits(monkeypatch, model, by_character)
    values = iter((1.0, 0.0))
    with pytest.raises(CompactDecodeError, match="monotonic finite"):
        decode_compact_example(
            model,
            tokenizer,
            _example(),
            generation_caps={TaskName.NEXT_ACTION: len(_TARGET) + 1},
            device=torch.device("cpu"),
            clock=lambda: next(values),
        )
    assert model.training


def test_prediction_contracts_reject_tampering_and_unknown_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _decode(monkeypatch)
    path_payload = result.constrained.model_dump(mode="python", round_trip=True)
    path_payload["generated_token_count"] += 1
    with pytest.raises(ValidationError, match="token count"):
        CompactPathPrediction.model_validate(path_payload)

    path_payload = result.constrained.model_dump(mode="python", round_trip=True)
    path_payload["unknown"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CompactPathPrediction.model_validate(path_payload)

    result_payload = result.model_dump(mode="python", round_trip=True)
    result_payload["checksum_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="checksum"):
        DualPathCompactPrediction.model_validate(result_payload)

    result_payload = result.model_dump(mode="python", round_trip=True)
    result_payload["unconstrained"] = result.constrained.model_dump(mode="python", round_trip=True)
    with pytest.raises(ValidationError, match="wrong unconstrained"):
        DualPathCompactPrediction.model_validate(result_payload)


def test_path_contract_rejects_every_cross_field_inconsistency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _decode(monkeypatch)
    base = result.constrained.model_dump(mode="python", round_trip=True)
    mutations: tuple[tuple[str, object, str], ...] = (
        ("selected_token_count", 1, "selected token count"),
        ("generation_cap", len(_TARGET), "exceeds the frozen"),
        ("generation_cap_exhausted", True, "exhaustion differs"),
        (
            "prompt_tokens_retained",
            cast(int, base["prompt_token_count"]) + 1,
            "exceed the original",
        ),
        ("prompt_truncated", True, "truncation flag"),
        ("compact_parse_success", False, "must first parse"),
        ("canonical_target_json", None, "must exist exactly"),
        ("checksum_sha256", "0" * 64, "checksum mismatch"),
        ("generated_text", "é" * 40_000, "byte bound"),
    )
    for field, value, error in mutations:
        payload = {**base, field: value}
        with pytest.raises(ValidationError, match=error):
            CompactPathPrediction.model_validate(payload)


def test_dual_contract_rejects_path_task_cap_and_prompt_misalignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _decode(monkeypatch)
    base = result.model_dump(mode="python", round_trip=True)

    payload = {**base, "constrained": result.unconstrained}
    with pytest.raises(ValidationError, match="wrong constrained"):
        DualPathCompactPrediction.model_validate(payload)

    payload = {**base, "task_name": TaskName.FAULT_FAMILY}
    with pytest.raises(ValidationError, match="task differs"):
        DualPathCompactPrediction.model_validate(payload)

    path_payload = result.constrained.model_dump(mode="python", round_trip=True)
    path_payload["generation_cap"] = result.constrained.generation_cap + 1
    path_payload["checksum_sha256"] = canonical_sha256(
        {key: value for key, value in path_payload.items() if key != "checksum_sha256"}
    )
    mismatched_cap = CompactPathPrediction.model_validate(path_payload)
    payload = {**base, "constrained": mismatched_cap}
    with pytest.raises(ValidationError, match="generation caps differ"):
        DualPathCompactPrediction.model_validate(payload)

    path_payload = result.constrained.model_dump(mode="python", round_trip=True)
    path_payload["prompt_token_count"] = result.constrained.prompt_token_count + 1
    path_payload["prompt_truncated"] = True
    path_payload["checksum_sha256"] = canonical_sha256(
        {key: value for key, value in path_payload.items() if key != "checksum_sha256"}
    )
    mismatched_prompt = CompactPathPrediction.model_validate(path_payload)
    payload = {**base, "constrained": mismatched_prompt}
    with pytest.raises(ValidationError, match="prompt boundaries differ"):
        DualPathCompactPrediction.model_validate(payload)


def test_prompt_logit_tokenizer_and_context_invariants_fail_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer, by_character = _tokenizer(monkeypatch)
    model = _model(tokenizer.vocab_size)
    _script_logits(monkeypatch, model, by_character)
    caps = {TaskName.NEXT_ACTION: len(_TARGET) + 1}

    monkeypatch.setattr(
        ProjectTokenizer,
        "encode",
        lambda _self, _text, *, add_bos=True, add_eos=True: (4, 5),
    )
    with pytest.raises(CompactDecodeError, match="does not begin"):
        decode_compact_example(
            model,
            tokenizer,
            _example(),
            generation_caps=caps,
            device=torch.device("cpu"),
            clock=StepClock(),
        )

    monkeypatch.setattr(
        ProjectTokenizer,
        "encode",
        lambda _self, _text, *, add_bos=True, add_eos=True: (BOS_ID,),
    )
    with pytest.raises(CompactDecodeError, match="too short"):
        decode_compact_example(
            model,
            tokenizer,
            _example(),
            generation_caps=caps,
            device=torch.device("cpu"),
            clock=StepClock(),
        )

    _tokenizer(monkeypatch)

    def malformed_prefill(
        _self: TransformerLM,
        _input_ids: Tensor,
        _attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[()]]:
        return torch.zeros((1, tokenizer.vocab_size - 1)), ()

    monkeypatch.setattr(TransformerLM, "prefill_cache", malformed_prefill)
    with pytest.raises(CompactDecodeError, match="logit shape"):
        decode_compact_example(
            model,
            tokenizer,
            _example(),
            generation_caps=caps,
            device=torch.device("cpu"),
            clock=StepClock(),
        )

    tokenizer.manifest.checksum_sha256 = "not-a-checksum"
    with pytest.raises(CompactDecodeError, match="checksum-bound manifest"):
        decode_compact_example(
            model,
            tokenizer,
            _example(),
            generation_caps=caps,
            device=torch.device("cpu"),
            clock=StepClock(),
        )

    tokenizer.manifest.checksum_sha256 = "c" * 64
    mismatched_context = RemediationExample.model_construct(
        example_id="example:mismatch",
        task_name=TaskName.NEXT_ACTION,
        prompt_text="fictional",
        compact_context=CompactTargetContext(
            task_name=TaskName.FAULT_FAMILY,
            visible_fact_refs=("o-0000",),
        ),
        checksum_sha256="d" * 64,
    )
    with pytest.raises(ValueError, match="prompt-local compact context"):
        decode_compact_example(
            model,
            tokenizer,
            mismatched_context,
            generation_caps=caps,
            device=torch.device("cpu"),
            clock=StepClock(),
        )


def test_constrained_canonical_json_matches_the_strict_target_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _decode(monkeypatch)
    expected = NextActionTarget(immediate_action=ActionLabel.CONTINUE_MONITORING)

    assert result.constrained.canonical_target_json is not None
    assert result.constrained.canonical_target_json == (
        '{"immediate_action":"CONTINUE_MONITORING","task_name":"next_action"}'
    )
    assert compact_target_json(_TARGET, context=_context()) == (
        result.constrained.canonical_target_json
    )
    assert expected.task_name is result.task_name
