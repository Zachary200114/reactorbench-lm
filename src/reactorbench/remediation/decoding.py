"""Deterministic dual-path compact decoding for development-only examples.

This module is deliberately computation-only: it accepts already validated model,
tokenizer, example, device, and generation-cap objects and never loads a dataset or
artifact.  The constrained path is constructed exclusively from the example's
prompt-local :class:`CompactTargetContext`; target text, labels, lineage, and IDs are
not inputs to token selection.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Annotated, Literal

import torch
from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator
from torch import Tensor

from reactorbench.evaluation.compact import (
    COMPACT_WIRE_PREFIX,
    MAX_COMPACT_TARGET_BYTES,
    MAX_CONSTRAINED_GENERATED_TOKENS,
    CompactDecodingError,
    CompactTargetConstraint,
    CompactTargetError,
    compact_target_json,
)
from reactorbench.model import AttentionCache, TransformerLM
from reactorbench.schemas.base import ContractId, ContractModel, canonical_sha256
from reactorbench.schemas.enums import TaskName
from reactorbench.tokenizer import EOS_ID, ProjectTokenizer

from .data import RemediationExample
from .serialization import retained_compact_prompt_tokens

MAX_DECODE_BATCH_SIZE = 1024
MAX_GENERATED_TEXT_BYTES = max(65_536, MAX_COMPACT_TARGET_BYTES)
_COMPACT_WIRE_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_|,~-"
)
_TASK_FIELD_COUNTS: dict[TaskName, int] = {
    TaskName.CONTINUE_LOG: 1,
    TaskName.FAULT_FAMILY: 3,
    TaskName.EXTRACT_EVIDENCE: 2,
    TaskName.NEXT_ACTION: 1,
    TaskName.INCIDENT_SUMMARY: 7,
    TaskName.COUNTERFACTUAL_COMPARE: 6,
}

Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
Probability = Annotated[StrictFloat, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[StrictFloat, Field(ge=0.0, allow_inf_nan=False)]
TokenId = Annotated[StrictInt, Field(ge=0, le=65_535)]


class CompactDecodeError(RuntimeError):
    """Safe public failure for an invalid or internally inconsistent decode."""


class DecodePath(StrEnum):
    UNCONSTRAINED = "unconstrained_greedy"
    CONSTRAINED = "truth_independent_constrained_greedy"


class CompactPathPrediction(ContractModel):
    """One immutable compact-generation result."""

    result_version: Literal["0.3.0"] = "0.3.0"
    path: DecodePath
    task_name: TaskName
    generation_cap: int = Field(strict=True, ge=1, le=MAX_CONSTRAINED_GENERATED_TOKENS)
    prompt_token_count: int = Field(strict=True, ge=1)
    prompt_tokens_retained: int = Field(strict=True, ge=2)
    prompt_truncated: StrictBool
    generated_token_ids: tuple[TokenId, ...] = Field(max_length=MAX_CONSTRAINED_GENERATED_TOKENS)
    generated_token_count: int = Field(strict=True, ge=0, le=MAX_CONSTRAINED_GENERATED_TOKENS)
    selected_token_count: int = Field(strict=True, ge=1, le=MAX_CONSTRAINED_GENERATED_TOKENS)
    generated_text: str = Field(max_length=MAX_GENERATED_TEXT_BYTES)
    eos_emitted: StrictBool
    generation_cap_exhausted: StrictBool
    compact_parse_success: StrictBool
    schema_valid: StrictBool
    canonical_target_json: str | None = Field(default=None, max_length=65_536)
    selected_token_geometric_mean_probability: Probability
    elapsed_seconds: NonNegativeFiniteFloat
    used_cache: StrictBool
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def shape_and_checksum_match(self) -> CompactPathPrediction:
        if len(self.generated_text.encode("utf-8")) > MAX_GENERATED_TEXT_BYTES:
            raise ValueError("generated compact text exceeds its byte bound")
        if self.generated_token_count != len(self.generated_token_ids):
            raise ValueError("generated token count differs from its token inventory")
        expected_selected = self.generated_token_count + int(self.eos_emitted)
        if self.selected_token_count != expected_selected:
            raise ValueError("selected token count must include an emitted EOS")
        if self.selected_token_count > self.generation_cap:
            raise ValueError("selected token count exceeds the frozen generation cap")
        expected_exhausted = not self.eos_emitted
        if self.generation_cap_exhausted is not expected_exhausted:
            raise ValueError("generation-cap exhaustion differs from EOS completion")
        if self.prompt_tokens_retained > self.prompt_token_count:
            raise ValueError("retained prompt tokens exceed the original prompt")
        if self.prompt_truncated is not (self.prompt_tokens_retained < self.prompt_token_count):
            raise ValueError("prompt truncation flag differs from retained-token counts")
        if self.schema_valid and not self.compact_parse_success:
            raise ValueError("schema-valid compact output must first parse")
        if self.schema_valid != (self.canonical_target_json is not None):
            raise ValueError("canonical target JSON must exist exactly for schema-valid output")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("compact path prediction checksum mismatch")
        return self


class DualPathCompactPrediction(ContractModel):
    """Checksum-bound unconstrained/constrained report for one example."""

    result_version: Literal["0.3.0"] = "0.3.0"
    example_id: ContractId
    example_checksum_sha256: Sha256
    task_name: TaskName
    model_config_sha256: Sha256
    tokenizer_manifest_sha256: Sha256
    generation_caps_sha256: Sha256
    unconstrained: CompactPathPrediction
    constrained: CompactPathPrediction
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def paths_and_checksum_match(self) -> DualPathCompactPrediction:
        if self.unconstrained.path is not DecodePath.UNCONSTRAINED:
            raise ValueError("dual-path report has the wrong unconstrained path")
        if self.constrained.path is not DecodePath.CONSTRAINED:
            raise ValueError("dual-path report has the wrong constrained path")
        paths = (self.unconstrained, self.constrained)
        if any(path.task_name is not self.task_name for path in paths):
            raise ValueError("dual-path task differs from its path result")
        if self.unconstrained.generation_cap != self.constrained.generation_cap:
            raise ValueError("dual-path generation caps differ")
        prompt_fields = (
            "prompt_token_count",
            "prompt_tokens_retained",
            "prompt_truncated",
        )
        if any(
            getattr(self.unconstrained, field) != getattr(self.constrained, field)
            for field in prompt_fields
        ):
            raise ValueError("dual-path prompt boundaries differ")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("dual-path prediction checksum mismatch")
        return self


def compact_wire_parse_success(text: str, *, task_name: TaskName) -> bool:
    """Check bounded compact wire structure without granting schema validity.

    This lexical/field-shape tier intentionally does not validate enum values,
    reference membership, or semantic relationships.  Those belong to the strict
    compact compiler and are reported separately as ``schema_valid``.
    """

    if type(text) is not str or type(task_name) is not TaskName:
        raise TypeError("compact wire parsing requires exact text and TaskName")
    try:
        encoded = text.encode("utf-8")
    except UnicodeError:
        return False
    if (
        not encoded
        or len(encoded) > MAX_COMPACT_TARGET_BYTES
        or any(character not in _COMPACT_WIRE_CHARACTERS for character in text)
    ):
        return False
    pieces = text.split("|")
    expected_length = 2 + _TASK_FIELD_COUNTS[task_name]
    return (
        len(pieces) == expected_length
        and pieces[0] == COMPACT_WIRE_PREFIX
        and pieces[1] == task_name.value
        and all(field != "" for field in pieces[2:])
    )


def _validated_cap_snapshot(
    generation_caps: Mapping[TaskName, int],
    *,
    required_tasks: frozenset[TaskName],
    context_length: int,
) -> tuple[tuple[TaskName, int], ...]:
    if not isinstance(generation_caps, Mapping):
        raise TypeError("generation caps must be an explicit TaskName mapping")
    items = tuple(generation_caps.items())
    if any(type(task) is not TaskName or type(cap) is not int for task, cap in items):
        raise TypeError("generation caps require exact TaskName and integer members")
    if len(items) != len({task for task, _cap in items}):
        raise ValueError("generation-cap mapping contains duplicate tasks")
    available = {task for task, _cap in items}
    if not required_tasks.issubset(available):
        raise ValueError("an example task has no explicit frozen generation cap")
    if any(
        not 1 <= cap <= MAX_CONSTRAINED_GENERATED_TOKENS or context_length - cap < 2
        for _task, cap in items
    ):
        raise ValueError("generation cap is outside the decoder/context bounds")
    order = {task: index for index, task in enumerate(TaskName)}
    return tuple(sorted(items, key=lambda item: order[item[0]]))


def _prompt_tokens(
    example: RemediationExample,
    tokenizer: ProjectTokenizer,
    *,
    context_length: int,
    generation_cap: int,
) -> tuple[tuple[int, ...], int, bool]:
    try:
        return retained_compact_prompt_tokens(
            example,
            tokenizer,
            context_length=context_length,
            generation_cap=generation_cap,
        )
    except (TypeError, ValueError) as exc:
        raise CompactDecodeError(str(exc)) from exc


def _validated_logits(logits: Tensor, *, tokenizer: ProjectTokenizer) -> Tensor:
    if (
        type(logits) is not Tensor
        or logits.ndim != 2
        or logits.shape != (1, tokenizer.vocab_size)
        or not logits.is_floating_point()
    ):
        raise CompactDecodeError("model returned an invalid next-token logit shape")
    if not bool(torch.isfinite(logits).all().item()):
        raise CompactDecodeError("model returned non-finite next-token logits")
    return logits[0]


def _full_forward_logits(
    model: TransformerLM,
    token_ids: tuple[int, ...],
    *,
    device: torch.device,
    tokenizer: ProjectTokenizer,
) -> Tensor:
    input_ids = torch.tensor((token_ids,), dtype=torch.long, device=device)
    output = model(input_ids)
    if (
        type(output) is not Tensor
        or output.ndim != 3
        or output.shape[:2] != input_ids.shape
        or output.shape[2] != tokenizer.vocab_size
    ):
        raise CompactDecodeError("model returned an invalid full-forward logit shape")
    return _validated_logits(output[:, -1, :], tokenizer=tokenizer)


def _elapsed(start: float, end: float) -> float:
    if (
        type(start) is not float
        or type(end) is not float
        or not math.isfinite(start)
        or not math.isfinite(end)
        or end < start
    ):
        raise CompactDecodeError("decoder clock did not provide monotonic finite values")
    return end - start


def _synchronize_for_latency(device: torch.device) -> None:
    """Make MPS latency include queued device work; CPU execution is synchronous."""

    if device.type == "mps":
        torch.mps.synchronize()


def _path_prediction(
    *,
    model: TransformerLM,
    tokenizer: ProjectTokenizer,
    example: RemediationExample,
    prompt_token_ids: tuple[int, ...],
    prompt_token_count: int,
    prompt_truncated: bool,
    generation_cap: int,
    path: DecodePath,
    device: torch.device,
    use_cache: bool,
    clock: Callable[[], float],
) -> CompactPathPrediction:
    constraint = (
        CompactTargetConstraint(
            example.compact_context,
            maximum_generated_tokens=generation_cap,
        )
        if path is DecodePath.CONSTRAINED
        else None
    )
    _synchronize_for_latency(device)
    start = clock()
    generated: list[int] = []
    log_probability_sum = 0.0
    selected_count = 0
    eos_emitted = False

    caches: tuple[AttentionCache, ...] = ()
    if use_cache:
        input_ids = torch.tensor((prompt_token_ids,), dtype=torch.long, device=device)
        prefill_logits, caches = model.prefill_cache(input_ids)
        logits = _validated_logits(prefill_logits, tokenizer=tokenizer)
    else:
        logits = _full_forward_logits(
            model,
            prompt_token_ids,
            device=device,
            tokenizer=tokenizer,
        )

    for _offset in range(generation_cap):
        if constraint is None:
            selected = int(torch.argmax(logits).item())
        else:
            try:
                selected = constraint.select_next_token_id(
                    logits,
                    tokenizer,
                    tuple(generated),
                )
            except (CompactDecodingError, TypeError, ValueError) as error:
                raise CompactDecodeError("constrained decoder reached a safe failure") from error
        selected_count += 1
        selected_log_probability = float(torch.log_softmax(logits, dim=0)[selected].item())
        if not math.isfinite(selected_log_probability):
            raise CompactDecodeError("selected-token probability is non-finite")
        log_probability_sum += selected_log_probability
        if selected == EOS_ID:
            eos_emitted = True
            break
        generated.append(selected)
        if selected_count == generation_cap:
            break
        if use_cache:
            next_input = torch.tensor(((selected,),), dtype=torch.long, device=device)
            key_mask = torch.ones(
                (1, len(prompt_token_ids) + len(generated)),
                dtype=torch.bool,
                device=device,
            )
            try:
                step_logits, caches = model.decode_step(
                    next_input,
                    position=len(prompt_token_ids) + len(generated) - 1,
                    caches=caches,
                    key_mask=key_mask,
                )
            except (TypeError, ValueError) as error:
                raise CompactDecodeError("cached decoder step failed safely") from error
            logits = _validated_logits(step_logits, tokenizer=tokenizer)
        else:
            logits = _full_forward_logits(
                model,
                (*prompt_token_ids, *generated),
                device=device,
                tokenizer=tokenizer,
            )

    _synchronize_for_latency(device)
    end = clock()
    elapsed_seconds = _elapsed(start, end)
    generated_ids = tuple(generated)
    generated_text = tokenizer.decode(generated_ids) if generated_ids else ""
    if len(generated_text.encode("utf-8")) > MAX_GENERATED_TEXT_BYTES:
        raise CompactDecodeError("generated compact text exceeds its safe byte bound")
    compact_parse_success = compact_wire_parse_success(
        generated_text,
        task_name=example.task_name,
    )
    schema_valid = False
    canonical: str | None = None
    if compact_parse_success:
        try:
            canonical = compact_target_json(
                generated_text,
                context=example.compact_context,
            )
            schema_valid = True
        except (CompactTargetError, TypeError, ValueError):
            schema_valid = False
            canonical = None
    confidence = math.exp(log_probability_sum / selected_count)
    draft = CompactPathPrediction.model_construct(
        result_version="0.3.0",
        path=path,
        task_name=example.task_name,
        generation_cap=generation_cap,
        prompt_token_count=prompt_token_count,
        prompt_tokens_retained=len(prompt_token_ids),
        prompt_truncated=prompt_truncated,
        generated_token_ids=generated_ids,
        generated_token_count=len(generated_ids),
        selected_token_count=selected_count,
        generated_text=generated_text,
        eos_emitted=eos_emitted,
        generation_cap_exhausted=not eos_emitted,
        compact_parse_success=compact_parse_success,
        schema_valid=schema_valid,
        canonical_target_json=canonical,
        selected_token_geometric_mean_probability=confidence,
        elapsed_seconds=elapsed_seconds,
        used_cache=use_cache,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return CompactPathPrediction(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
        checksum_sha256=checksum,
    )


def _tokenizer_manifest_checksum(tokenizer: ProjectTokenizer) -> str:
    checksum = getattr(tokenizer.manifest, "checksum_sha256", None)
    if (
        type(checksum) is not str
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise CompactDecodeError("tokenizer has no valid checksum-bound manifest")
    return checksum


def decode_compact_examples(
    model: TransformerLM,
    tokenizer: ProjectTokenizer,
    examples: tuple[RemediationExample, ...],
    *,
    generation_caps: Mapping[TaskName, int],
    device: torch.device,
    use_cache: bool = True,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[DualPathCompactPrediction, ...]:
    """Decode a bounded example batch through both preregistered greedy paths."""

    if type(model) is not TransformerLM or type(tokenizer) is not ProjectTokenizer:
        raise TypeError("compact decoding requires exact project model/tokenizer objects")
    if (
        type(examples) is not tuple
        or not examples
        or len(examples) > MAX_DECODE_BATCH_SIZE
        or any(type(example) is not RemediationExample for example in examples)
    ):
        raise TypeError("examples must be a non-empty bounded exact RemediationExample tuple")
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("decode examples must have unique IDs")
    if type(device) is not torch.device or type(use_cache) is not bool or not callable(clock):
        raise TypeError("decoder device, cache flag, or clock has an invalid type")
    if model.config.context_length < 4:
        raise ValueError("model context is too short for bounded compact decoding")
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("model and tokenizer vocabulary sizes differ")
    try:
        parameter_device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("decoder model contains no parameters") from error
    if parameter_device != device:
        raise ValueError("decoder device differs from the model parameter device")
    if any(example.task_name is not example.compact_context.task_name for example in examples):
        raise ValueError("example task differs from its prompt-local compact context")

    cap_snapshot = _validated_cap_snapshot(
        generation_caps,
        required_tasks=frozenset(example.task_name for example in examples),
        context_length=model.config.context_length,
    )
    caps = dict(cap_snapshot)
    caps_sha256 = canonical_sha256(tuple((task.value, cap) for task, cap in cap_snapshot))
    model_config_sha256 = canonical_sha256(model.config.model_dump(mode="json", round_trip=True))
    tokenizer_manifest_sha256 = _tokenizer_manifest_checksum(tokenizer)
    was_training = model.training
    results: list[DualPathCompactPrediction] = []
    model.eval()
    try:
        with torch.no_grad():
            for example in examples:
                cap = caps[example.task_name]
                prompt_ids, original_prompt_count, truncated = _prompt_tokens(
                    example,
                    tokenizer,
                    context_length=model.config.context_length,
                    generation_cap=cap,
                )
                unconstrained = _path_prediction(
                    model=model,
                    tokenizer=tokenizer,
                    example=example,
                    prompt_token_ids=prompt_ids,
                    prompt_token_count=original_prompt_count,
                    prompt_truncated=truncated,
                    generation_cap=cap,
                    path=DecodePath.UNCONSTRAINED,
                    device=device,
                    use_cache=use_cache,
                    clock=clock,
                )
                constrained = _path_prediction(
                    model=model,
                    tokenizer=tokenizer,
                    example=example,
                    prompt_token_ids=prompt_ids,
                    prompt_token_count=original_prompt_count,
                    prompt_truncated=truncated,
                    generation_cap=cap,
                    path=DecodePath.CONSTRAINED,
                    device=device,
                    use_cache=use_cache,
                    clock=clock,
                )
                draft = DualPathCompactPrediction.model_construct(
                    result_version="0.3.0",
                    example_id=example.example_id,
                    example_checksum_sha256=example.checksum_sha256,
                    task_name=example.task_name,
                    model_config_sha256=model_config_sha256,
                    tokenizer_manifest_sha256=tokenizer_manifest_sha256,
                    generation_caps_sha256=caps_sha256,
                    unconstrained=unconstrained,
                    constrained=constrained,
                    checksum_sha256="0" * 64,
                )
                checksum = canonical_sha256(
                    draft.model_dump(
                        mode="json",
                        round_trip=True,
                        exclude={"checksum_sha256"},
                    )
                )
                results.append(
                    DualPathCompactPrediction(
                        **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
                        checksum_sha256=checksum,
                    )
                )
    finally:
        model.train(was_training)
    return tuple(results)


def decode_compact_example(
    model: TransformerLM,
    tokenizer: ProjectTokenizer,
    example: RemediationExample,
    *,
    generation_caps: Mapping[TaskName, int],
    device: torch.device,
    use_cache: bool = True,
    clock: Callable[[], float] = time.perf_counter,
) -> DualPathCompactPrediction:
    """Decode one example through unconstrained and constrained greedy paths."""

    if type(example) is not RemediationExample:
        raise TypeError("single compact decoding requires an exact RemediationExample")
    return decode_compact_examples(
        model,
        tokenizer,
        (example,),
        generation_caps=generation_caps,
        device=device,
        use_cache=use_cache,
        clock=clock,
    )[0]


__all__ = [
    "MAX_DECODE_BATCH_SIZE",
    "CompactDecodeError",
    "CompactPathPrediction",
    "DecodePath",
    "DualPathCompactPrediction",
    "compact_wire_parse_success",
    "decode_compact_example",
    "decode_compact_examples",
]
