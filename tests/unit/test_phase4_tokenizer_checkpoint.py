"""Tokenizer isolation, determinism, and safe checkpoint tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from reactorbench.dataset.catalog import AliasFamily, TemplateFamily
from reactorbench.dataset.renderer import RenderedCandidate
from reactorbench.model import (
    TransformerConfig,
    initialized_model,
    load_checkpoint,
    save_checkpoint,
)
from reactorbench.model.config import TokenizerConfig
from reactorbench.schemas.base import canonical_json_bytes, canonical_sha256
from reactorbench.schemas.enums import SplitName
from reactorbench.tokenizer import (
    ProjectTokenizer,
    TrainingCorpus,
    TrainingCorpusManifest,
    train_tokenizer,
    training_corpus_from_rendered_candidates,
)


def _tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig(
        tokenizer_version="0.1.0",
        algorithm="sentencepiece_bpe",
        vocab_size=512,
        character_coverage=1.0,
        byte_fallback=True,
        normalization_rule="identity",
        special_symbols=("<|prompt|>", "<|target|>", "<|sep|>"),
    )


def _corpus() -> TrainingCorpus:
    documents = tuple(
        f"Research log {index:03d}: component-{index % 17} trend {index / 127:.4f}; "
        f"fictional state Δ café Ω signal-{index * index % 101}."
        for index in range(128)
    )
    corpus_bytes = "\n\n".join(documents).encode("utf-8")
    inventory = tuple(
        (str(index), hashlib.sha256(document.encode("utf-8")).hexdigest())
        for index, document in enumerate(documents)
    )
    return TrainingCorpus(
        documents=documents,
        manifest=TrainingCorpusManifest(
            candidate_bundle_sha256="1" * 64,
            candidate_artifact_manifest_sha256="2" * 64,
            postrender_packet_sha256="3" * 64,
            postrender_approval_record_sha256="4" * 64,
            document_count=len(documents),
            utf8_bytes=len(corpus_bytes),
            document_inventory_sha256=canonical_sha256(inventory),
            corpus_sha256=hashlib.sha256(corpus_bytes).hexdigest(),
        ),
    )


def _candidate(split_name: SplitName, text: str, identity: str) -> RenderedCandidate:
    draft = RenderedCandidate.model_construct(
        split_name=split_name,
        template_family_id=TemplateFamily.COMPACT_LOG,
        alias_family_id=AliasFamily.CANONICAL,
        model_input_sha256=identity * 64,
        template_ids=(f"template-{identity}",),
        lines=(text,),
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        render_id="render-" + "0" * 24,
    )
    payload = draft.model_dump(mode="json", round_trip=True, exclude={"render_id", "text_sha256"})
    return RenderedCandidate(
        **draft.model_dump(mode="python", exclude={"render_id"}),
        render_id=f"render-{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:24]}",
    )


def test_training_corpus_excludes_every_holdout_split() -> None:
    records = tuple(
        sorted(
            (
                _candidate(SplitName.IID_TRAIN, "approved training prose", "a"),
                _candidate(SplitName.IID_TEST, "forbidden test prose", "b"),
                _candidate(SplitName.TEMPLATE_TEST, "forbidden template holdout", "c"),
            ),
            key=lambda record: record.render_id,
        )
    )

    corpus = training_corpus_from_rendered_candidates(
        records,
        candidate_bundle_sha256="1" * 64,
        candidate_artifact_manifest_sha256="2" * 64,
        postrender_packet_sha256="3" * 64,
        postrender_approval_record_sha256="4" * 64,
    )

    assert corpus.documents == ("approved training prose",)
    assert corpus.manifest.source_split is SplitName.IID_TRAIN
    assert corpus.manifest.document_count == 1


def test_tokenizer_training_is_byte_deterministic_and_unicode_complete(tmp_path: Path) -> None:
    corpus = _corpus()
    first = train_tokenizer(corpus, _tokenizer_config(), output_directory=tmp_path / "first")
    second = train_tokenizer(corpus, _tokenizer_config(), output_directory=tmp_path / "second")

    assert first.model_sha256 == second.model_sha256
    assert first.vocab_sha256 == second.vocab_sha256
    tokenizer = ProjectTokenizer.load(tmp_path / "first", expected_checksum=first.checksum_sha256)
    token_ids = tokenizer.encode("Aster fictional Δ channel café")
    assert tokenizer.decode(token_ids) == "Aster fictional Δ channel café"
    assert tokenizer.vocab_size == 512

    with pytest.raises(FileExistsError):
        train_tokenizer(corpus, _tokenizer_config(), output_directory=tmp_path / "first")


def test_tokenizer_load_rejects_tampering(tmp_path: Path) -> None:
    manifest = train_tokenizer(
        _corpus(), _tokenizer_config(), output_directory=tmp_path / "tokenizer"
    )
    with (tmp_path / "tokenizer" / "tokenizer.vocab").open("a", encoding="utf-8") as file:
        file.write("tampered\n")

    with pytest.raises(ValueError, match="checksum or size"):
        ProjectTokenizer.load(tmp_path / "tokenizer", expected_checksum=manifest.checksum_sha256)


def test_safetensors_checkpoint_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    tokenizer_manifest = train_tokenizer(
        _corpus(), _tokenizer_config(), output_directory=tmp_path / "tokenizer"
    )
    config = TransformerConfig(
        model_version="0.1.0",
        layers=1,
        width=32,
        heads=4,
        context_length=16,
        feed_forward_multiplier=2,
        dropout=0.0,
        tie_embeddings=True,
        bias=True,
    )
    model = initialized_model(config, vocab_size=512, seed=7)
    manifest = save_checkpoint(
        model,
        output_directory=tmp_path / "checkpoint",
        tokenizer_manifest=tokenizer_manifest,
        source_commit="abcdef0",
        seed=7,
        training_steps=0,
        initial_loss=1.0,
        final_loss=1.0,
    )
    loaded, loaded_manifest = load_checkpoint(
        tmp_path / "checkpoint",
        expected_manifest_sha256=manifest.checksum_sha256,
        expected_tokenizer_sha256=tokenizer_manifest.checksum_sha256,
        device=torch.device("cpu"),
    )
    inputs = torch.tensor(((1, 2, 3, 4),), dtype=torch.long)

    assert loaded_manifest == manifest
    assert torch.equal(model(inputs), loaded(inputs))
    assert {path.name for path in (tmp_path / "checkpoint").iterdir()} == {
        "manifest.json",
        "model.safetensors",
    }

    weights = tmp_path / "checkpoint" / "model.safetensors"
    with weights.open("ab") as file:
        file.write(b"tamper")
    with pytest.raises(ValueError, match="checksum or size"):
        load_checkpoint(
            tmp_path / "checkpoint",
            expected_manifest_sha256=manifest.checksum_sha256,
            expected_tokenizer_sha256=tokenizer_manifest.checksum_sha256,
            device=torch.device("cpu"),
        )
