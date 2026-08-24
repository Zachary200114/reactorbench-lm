from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from reactorbench.remediation.data import RemediationExample
from reactorbench.remediation.serialization import (
    compact_serialized_parts,
    tokenize_compact_example,
)
from reactorbench.schemas.enums import TaskName
from reactorbench.tokenizer import BOS_ID, EOS_ID, ProjectTokenizer


def _example(*, prompt_text: str = "fictional evidence") -> RemediationExample:
    return RemediationExample.model_construct(
        example_id="serialization:test",
        task_name=TaskName.NEXT_ACTION,
        group_id="serialization:test",
        prompt_text=prompt_text,
        compact_target="RB2|next_action|7",
    )


def test_task_conditioning_is_immediately_before_the_target_boundary() -> None:
    prompt, target = compact_serialized_parts(_example())

    assert prompt == ("<|prompt|>\nfictional evidence\nTASK=next_action\n<|target|>\n")
    assert target == "RB2|next_action|7"


def test_left_truncation_preserves_task_conditioning_footer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = object.__new__(ProjectTokenizer)
    tokenizer.manifest = cast(
        Any,
        SimpleNamespace(actual_vocab_size=256, checksum_sha256="a" * 64),
    )

    def encode(
        _self: ProjectTokenizer,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> tuple[int, ...]:
        body = tuple(ord(character) + 4 for character in text)
        return (
            *((BOS_ID,) if add_bos else ()),
            *body,
            *((EOS_ID,) if add_eos else ()),
        )

    monkeypatch.setattr(ProjectTokenizer, "encode", encode)
    monkeypatch.setattr(
        ProjectTokenizer,
        "decode",
        lambda _self, token_ids: "".join(chr(token_id - 4) for token_id in token_ids),
    )
    tokenized = tokenize_compact_example(
        _example(prompt_text="x" * 300),
        tokenizer,
        context_length=96,
        generation_caps={TaskName.NEXT_ACTION: 32},
    )

    assert tokenized.prompt_truncated
    retained_prompt = tokenized.token_ids[: tokenized.prompt_tokens_retained]
    assert retained_prompt[0] == BOS_ID
    decoded_suffix = "".join(chr(token_id - 4) for token_id in retained_prompt[1:])
    assert decoded_suffix.endswith("TASK=next_action\n<|target|>\n")


def test_tokenization_fails_when_context_cannot_retain_the_complete_footer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = object.__new__(ProjectTokenizer)
    tokenizer.manifest = cast(
        Any,
        SimpleNamespace(actual_vocab_size=256, checksum_sha256="a" * 64),
    )

    def encode(
        _self: ProjectTokenizer,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> tuple[int, ...]:
        body = tuple(ord(character) + 4 for character in text)
        return (
            *((BOS_ID,) if add_bos else ()),
            *body,
            *((EOS_ID,) if add_eos else ()),
        )

    monkeypatch.setattr(ProjectTokenizer, "encode", encode)
    monkeypatch.setattr(
        ProjectTokenizer,
        "decode",
        lambda _self, token_ids: "".join(chr(token_id - 4) for token_id in token_ids),
    )

    with pytest.raises(ValueError, match="complete task footer"):
        tokenize_compact_example(
            _example(prompt_text="x" * 300),
            tokenizer,
            context_length=64,
            generation_caps={TaskName.NEXT_ACTION: 48},
        )
