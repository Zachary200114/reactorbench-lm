from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from reactorbench.evaluation import ExperimentData, ExperimentExample, load_phase5_config
from reactorbench.model import TransformerConfig
from reactorbench.schemas.base import canonical_sha256
from reactorbench.schemas.enums import SplitName, TaskName
from reactorbench.tokenizer import (
    ProjectTokenizer,
    TokenizerArtifactManifest,
    TrainingCorpusManifest,
)
from reactorbench.training.pilot import _strict_json_object, _train_transformer


def _manifest() -> TokenizerArtifactManifest:
    corpus = TrainingCorpusManifest(
        candidate_bundle_sha256="1" * 64,
        candidate_artifact_manifest_sha256="2" * 64,
        postrender_packet_sha256="3" * 64,
        postrender_approval_record_sha256="4" * 64,
        document_count=1,
        utf8_bytes=1,
        document_inventory_sha256="5" * 64,
        corpus_sha256="6" * 64,
    )
    draft = TokenizerArtifactManifest.model_construct(
        tokenizer_version="0.1.0",
        algorithm="sentencepiece_bpe",
        sentencepiece_version="test",
        requested_vocab_size=512,
        actual_vocab_size=512,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        special_symbols=("<|prompt|>", "<|target|>", "<|sep|>"),
        model_sha256="7" * 64,
        vocab_sha256="8" * 64,
        model_size_bytes=1,
        vocab_size_bytes=1,
        corpus=corpus,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return TokenizerArtifactManifest(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
        checksum_sha256=checksum,
    )


def _data() -> ExperimentData:
    def records(split: SplitName, count: int) -> tuple[ExperimentExample, ...]:
        return tuple(
            ExperimentExample(
                example_id=f"example:{split.value}:{index:03d}",
                split_name=split,
                task_name=TaskName.NEXT_ACTION,
                prompt_text=f"fictional trend {index % 3}",
                target_text='{"immediate_action":"CONTINUE_MONITORING"}',
                classification_label="CONTINUE_MONITORING",
                source_checksum_sha256=hashlib.sha256(str(index).encode()).hexdigest(),
            )
            for index in range(count)
        )

    train = records(SplitName.IID_TRAIN, 12)
    validation = records(SplitName.IID_VALIDATION, 6)
    return ExperimentData(
        train=train,
        validation=validation,
        inventory_sha256=canonical_sha256(tuple(item.example_id for item in train + validation)),
    )


def test_tiny_validation_selected_transformer_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    tokenizer = object.__new__(ProjectTokenizer)
    tokenizer.manifest = manifest

    def encode(
        _self: ProjectTokenizer, text: str, *, add_bos: bool = True, add_eos: bool = True
    ) -> tuple[int, ...]:
        values = tuple(4 + ord(character) % 60 for character in text)
        return (*((1,) if add_bos else ()), *values, *((2,) if add_eos else ()))

    monkeypatch.setattr(ProjectTokenizer, "encode", encode)
    phase5 = load_phase5_config(Path("configs/experiments/phase5-pilot-v0.1.0.toml"))
    training = phase5.smaller_transformer.model_copy(
        update={
            "device": "cpu",
            "steps": 4,
            "batch_size": 4,
            "evaluation_interval": 2,
            "learning_rate": 0.005,
        }
    )
    model = TransformerConfig(
        model_version="0.1.0",
        layers=1,
        width=16,
        heads=4,
        context_length=96,
        feed_forward_multiplier=2,
        dropout=0.0,
        tie_embeddings=True,
        bias=True,
    )

    result = _train_transformer(
        tier="smaller_transformer",
        model_config=model,
        training_config=training,
        tokenizer_manifest=manifest,
        tokenizer=tokenizer,
        data=_data(),
        phase5_config=phase5,
        output_directory=tmp_path / "checkpoint",
        source_commit="abcdef0",
    )

    assert result.device == "cpu"
    assert result.selected_step in {0, 2, 4}
    assert result.validation_curve[0].step == 0
    assert result.checkpoint_size_bytes > 0
    assert {path.name for path in (tmp_path / "checkpoint").iterdir()} == {
        "manifest.json",
        "model.safetensors",
    }

    assert _strict_json_object(b'{"ok":true}') == {"ok": True}
    with pytest.raises(ValueError, match="duplicate"):
        _strict_json_object(b'{"ok":true,"ok":false}')
