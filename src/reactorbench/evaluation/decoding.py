"""Bounded cached greedy decoding and strict structured-target validation."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import torch
from pydantic import Field, StrictFloat, TypeAdapter, model_validator

from reactorbench.dataset.contracts import ProjectionTaskTargetValue
from reactorbench.model import TransformerLM
from reactorbench.schemas.base import ContractModel, canonical_json_bytes, canonical_sha256
from reactorbench.schemas.enums import TaskName
from reactorbench.tokenizer import BOS_ID, EOS_ID, ProjectTokenizer

from .config import SerializationConfig
from .data import ExperimentExample, _classification_label
from .serialization import serialized_parts

_TARGET_ADAPTER: TypeAdapter[ProjectionTaskTargetValue] = TypeAdapter(ProjectionTaskTargetValue)


class DecodedPrediction(ContractModel):
    example_id: str = Field(min_length=1, max_length=128)
    task_name: TaskName
    generated_text: str = Field(max_length=65_536)
    generated_token_count: int = Field(strict=True, ge=0, le=4096)
    prompt_truncated: bool
    generation_truncated: bool
    json_parse_success: bool
    schema_valid: bool
    predicted_target_json: str | None = Field(default=None, max_length=65_536)
    classification_label: str | None = Field(default=None, max_length=512)
    confidence: StrictFloat
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def shape_and_checksum_match(self) -> DecodedPrediction:
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("prediction confidence must be finite in [0,1]")
        if self.schema_valid and not self.json_parse_success:
            raise ValueError("schema-valid output must first parse as JSON")
        if self.schema_valid != (self.predicted_target_json is not None):
            raise ValueError("canonical prediction JSON must exist exactly for valid outputs")
        if not self.schema_valid and self.classification_label is not None:
            raise ValueError("invalid output cannot carry a classification label")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("prediction checksum mismatch")
        return self


def _strict_json(payload: str) -> object:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError("prediction contains a duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"prediction contains non-finite JSON: {value}")

    return json.loads(payload, object_pairs_hook=pairs, parse_constant=reject_constant)


def parse_structured_prediction(
    generated_text: str,
    *,
    task_name: TaskName,
) -> tuple[bool, bool, ProjectionTaskTargetValue | None, str | None]:
    """Parse one exact JSON object, then apply the task-specific strict contract."""

    if type(generated_text) is not str or type(task_name) is not TaskName:
        raise TypeError("prediction parsing requires exact text and TaskName")
    try:
        decoded = _strict_json(generated_text)
    except (UnicodeError, ValueError, json.JSONDecodeError):
        return False, False, None, None
    if type(decoded) is not dict:
        return True, False, None, None
    try:
        target = _TARGET_ADAPTER.validate_json(canonical_json_bytes(decoded), strict=True)
    except ValueError:
        return True, False, None, None
    if target.task_name is not task_name:
        return True, False, None, None
    canonical = canonical_json_bytes(target.model_dump(mode="json", round_trip=True)).decode(
        "utf-8"
    )
    return True, True, target, canonical


def _prediction(
    example: ExperimentExample,
    *,
    generated_text: str,
    generated_token_count: int,
    prompt_truncated: bool,
    generation_truncated: bool,
    confidence: float,
) -> DecodedPrediction:
    parsed, valid, target, canonical = parse_structured_prediction(
        generated_text, task_name=example.task_name
    )
    label = None if target is None else _classification_label(example.task_name, target)
    draft = DecodedPrediction.model_construct(
        example_id=example.example_id,
        task_name=example.task_name,
        generated_text=generated_text,
        generated_token_count=generated_token_count,
        prompt_truncated=prompt_truncated,
        generation_truncated=generation_truncated,
        json_parse_success=parsed,
        schema_valid=valid,
        predicted_target_json=canonical,
        classification_label=label,
        confidence=float(confidence if valid else 0.0),
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return DecodedPrediction(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
        checksum_sha256=checksum,
    )


def _prompt_tokens(
    example: ExperimentExample,
    tokenizer: ProjectTokenizer,
    serialization: SerializationConfig,
    *,
    context_length: int,
    maximum_generated_tokens: int,
) -> tuple[tuple[int, ...], bool]:
    prefix, _target = serialized_parts(example, serialization)
    tokens = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    maximum_prefix = context_length - maximum_generated_tokens
    if maximum_prefix < 2:
        raise ValueError("decoder reserve leaves no valid prompt boundary")
    truncated = len(tokens) > maximum_prefix
    if truncated:
        tokens = (BOS_ID, *tokens[-(maximum_prefix - 1) :])
    return tokens, truncated


def greedy_decode_predictions(
    model: TransformerLM,
    tokenizer: ProjectTokenizer,
    examples: tuple[ExperimentExample, ...],
    serialization: SerializationConfig,
    *,
    maximum_generated_tokens: int,
    batch_size: int,
    device: torch.device,
) -> tuple[DecodedPrediction, ...]:
    """Decode examples with exact per-layer caches and no sampling."""

    if type(model) is not TransformerLM or type(tokenizer) is not ProjectTokenizer:
        raise TypeError("decoding requires exact project model and tokenizer objects")
    if type(examples) is not tuple or any(type(item) is not ExperimentExample for item in examples):
        raise TypeError("examples must be an exact ExperimentExample tuple")
    if type(serialization) is not SerializationConfig:
        raise TypeError("serialization must be an exact SerializationConfig")
    if (
        type(maximum_generated_tokens) is not int
        or not 1 <= maximum_generated_tokens < model.config.context_length
        or type(batch_size) is not int
        or not 1 <= batch_size <= 128
    ):
        raise ValueError("decoder token or batch bound is invalid")
    prepared = [
        (
            example,
            *_prompt_tokens(
                example,
                tokenizer,
                serialization,
                context_length=model.config.context_length,
                maximum_generated_tokens=maximum_generated_tokens,
            ),
        )
        for example in examples
    ]
    grouped: dict[int, list[tuple[ExperimentExample, tuple[int, ...], bool]]] = defaultdict(list)
    for example, tokens, truncated in prepared:
        grouped[len(tokens)].append((example, tokens, truncated))
    results: list[DecodedPrediction] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for prefix_length in sorted(grouped):
                group = grouped[prefix_length]
                for start in range(0, len(group), batch_size):
                    chunk = group[start : start + batch_size]
                    input_ids = torch.tensor(
                        tuple(item[1] for item in chunk), dtype=torch.long, device=device
                    )
                    logits, caches = model.prefill_cache(input_ids)
                    generated: list[list[int]] = [[] for _ in chunk]
                    log_probabilities: list[float] = [0.0 for _ in chunk]
                    finished = [False for _ in chunk]
                    exhausted = True
                    for offset in range(maximum_generated_tokens):
                        probabilities = torch.softmax(logits, dim=-1)
                        next_tokens = logits.argmax(dim=-1)
                        chosen = probabilities.gather(1, next_tokens[:, None]).squeeze(1)
                        for row, token in enumerate(next_tokens.tolist()):
                            if finished[row]:
                                continue
                            if token == EOS_ID:
                                finished[row] = True
                                continue
                            generated[row].append(token)
                            log_probabilities[row] += math.log(
                                max(float(chosen[row].item()), 1e-30)
                            )
                        if all(finished):
                            exhausted = False
                            break
                        if offset + 1 == maximum_generated_tokens:
                            break
                        key_mask = torch.ones(
                            (len(chunk), prefix_length + offset + 1),
                            dtype=torch.bool,
                            device=device,
                        )
                        logits, caches = model.decode_step(
                            next_tokens[:, None],
                            position=prefix_length + offset,
                            caches=caches,
                            key_mask=key_mask,
                        )
                    for row, (example, _tokens, truncated) in enumerate(chunk):
                        count = len(generated[row])
                        confidence = math.exp(log_probabilities[row] / count) if count else 0.0
                        results.append(
                            _prediction(
                                example,
                                generated_text=tokenizer.decode(tuple(generated[row])),
                                generated_token_count=count,
                                prompt_truncated=truncated,
                                generation_truncated=exhausted and not finished[row],
                                confidence=confidence,
                            )
                        )
    finally:
        model.train(was_training)
    results.sort(key=lambda item: item.example_id)
    return tuple(results)


__all__ = [
    "DecodedPrediction",
    "greedy_decode_predictions",
    "parse_structured_prediction",
]
