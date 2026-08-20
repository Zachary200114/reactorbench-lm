"""Approved-corpus extraction and deterministic project SentencePiece tokenizer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import sentencepiece as sentencepiece
from pydantic import Field, model_validator

from reactorbench.dataset import RenderedCandidate, VerifiedDevelopmentCandidateArtifact
from reactorbench.dataset.review import HumanReviewRecord, verify_review_record
from reactorbench.model.config import TokenizerConfig
from reactorbench.schemas.base import (
    ContractModel,
    SemanticVersion,
    canonical_json_bytes,
    canonical_sha256,
)
from reactorbench.schemas.enums import SplitName

TOKENIZER_ARTIFACT_VERSION: SemanticVersion = "0.1.0"
UNK_ID = 0
BOS_ID = 1
EOS_ID = 2
PAD_ID = 3
MAX_TOKENIZER_MODEL_BYTES = 64 * 1024 * 1024
MAX_TOKENIZER_VOCAB_BYTES = 8 * 1024 * 1024
MAX_TOKENIZER_MANIFEST_BYTES = 1024 * 1024
_TRAINING_CWD_LOCK = threading.Lock()


class TrainingCorpusManifest(ContractModel):
    corpus_version: Literal["0.1.0"] = "0.1.0"
    source_split: Literal[SplitName.IID_TRAIN] = SplitName.IID_TRAIN
    candidate_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_artifact_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    postrender_packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    postrender_approval_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_count: int = Field(strict=True, ge=1, le=10_000)
    utf8_bytes: int = Field(strict=True, ge=1, le=64 * 1024 * 1024)
    document_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TrainingCorpus:
    documents: tuple[str, ...]
    manifest: TrainingCorpusManifest

    def canonical_bytes(self) -> bytes:
        return "\n\n".join(self.documents).encode("utf-8")


class TokenizerArtifactManifest(ContractModel):
    artifact_version: Literal["0.1.0"] = "0.1.0"
    tokenizer_version: SemanticVersion
    algorithm: Literal["sentencepiece_bpe"]
    sentencepiece_version: str = Field(min_length=1, max_length=32)
    requested_vocab_size: int = Field(strict=True, ge=512, le=16_384)
    actual_vocab_size: int = Field(strict=True, ge=512, le=16_384)
    unk_id: Literal[0]
    bos_id: Literal[1]
    eos_id: Literal[2]
    pad_id: Literal[3]
    special_symbols: tuple[str, ...]
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vocab_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_size_bytes: int = Field(strict=True, ge=1, le=MAX_TOKENIZER_MODEL_BYTES)
    vocab_size_bytes: int = Field(strict=True, ge=1, le=MAX_TOKENIZER_VOCAB_BYTES)
    corpus: TrainingCorpusManifest
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def checksum_matches(self) -> TokenizerArtifactManifest:
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("tokenizer manifest checksum mismatch")
        return self


def approved_training_corpus(
    verified: VerifiedDevelopmentCandidateArtifact,
    approval_record: HumanReviewRecord,
) -> TrainingCorpus:
    """Extract only approved IID-training prose from a fully verified candidate."""

    if type(verified) is not VerifiedDevelopmentCandidateArtifact:
        raise TypeError("verified must be an exact verified development artifact")
    if type(approval_record) is not HumanReviewRecord:
        raise TypeError("approval_record must be an exact HumanReviewRecord")
    verify_review_record(
        verified.candidate.postrender_review_packet,
        approval_record,
        require_approved=True,
    )
    return training_corpus_from_rendered_candidates(
        verified.candidate.rendered_candidates,
        candidate_bundle_sha256=verified.candidate.checksum_sha256,
        candidate_artifact_manifest_sha256=verified.manifest.checksum(),
        postrender_packet_sha256=verified.candidate.postrender_review_packet.packet_sha256,
        postrender_approval_record_sha256=approval_record.review_record_sha256,
    )


def training_corpus_from_rendered_candidates(
    records: tuple[RenderedCandidate, ...],
    *,
    candidate_bundle_sha256: str,
    candidate_artifact_manifest_sha256: str,
    postrender_packet_sha256: str,
    postrender_approval_record_sha256: str,
) -> TrainingCorpus:
    """Build the tokenizer corpus through an explicit IID-training-only boundary."""

    if type(records) is not tuple or any(
        type(record) is not RenderedCandidate for record in records
    ):
        raise TypeError("records must be an exact tuple of RenderedCandidate objects")
    validated = tuple(
        RenderedCandidate.model_validate(record.model_dump(mode="python", round_trip=True))
        for record in records
    )
    identifiers = tuple(record.render_id for record in validated)
    if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(set(identifiers)):
        raise ValueError("rendered candidate inventory must have unique canonical ID order")
    records = tuple(record for record in validated if record.split_name is SplitName.IID_TRAIN)
    if not records:
        raise ValueError("approved candidate contains no IID-training prose")
    if any(record.split_name is not SplitName.IID_TRAIN for record in records):
        raise ValueError("tokenizer corpus contains a non-training split")
    inventory = tuple((record.render_id, record.text_sha256) for record in records)
    documents = tuple(record.text for record in records)
    corpus_bytes = "\n\n".join(documents).encode("utf-8")
    manifest = TrainingCorpusManifest(
        candidate_bundle_sha256=candidate_bundle_sha256,
        candidate_artifact_manifest_sha256=candidate_artifact_manifest_sha256,
        postrender_packet_sha256=postrender_packet_sha256,
        postrender_approval_record_sha256=postrender_approval_record_sha256,
        document_count=len(documents),
        utf8_bytes=len(corpus_bytes),
        document_inventory_sha256=canonical_sha256(inventory),
        corpus_sha256=hashlib.sha256(corpus_bytes).hexdigest(),
    )
    return TrainingCorpus(documents=documents, manifest=manifest)


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError("tokenizer manifest contains a duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"tokenizer manifest contains non-finite JSON: {value}")

    decoded = json.loads(
        payload.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject_constant
    )
    if type(decoded) is not dict:
        raise ValueError("tokenizer manifest must contain one JSON object")
    return decoded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        while chunk := artifact.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def train_tokenizer(
    corpus: TrainingCorpus,
    config: TokenizerConfig,
    *,
    output_directory: Path,
) -> TokenizerArtifactManifest:
    """Train one deterministic tokenizer and atomically publish its data-only files."""

    if type(corpus) is not TrainingCorpus or type(config) is not TokenizerConfig:
        raise TypeError("tokenizer training requires exact corpus and config objects")
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be a pathlib.Path")
    parent = output_directory.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    if parent.is_symlink() or not parent.is_dir() or output_directory.exists():
        raise FileExistsError("tokenizer output must be a new path below a regular directory")
    sentences = tuple(
        line for document in corpus.documents for line in document.splitlines() if line
    )
    if not sentences:
        raise ValueError("tokenizer corpus contains no non-empty sentences")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.tmp-", dir=parent))
    try:
        with _TRAINING_CWD_LOCK:
            previous = Path.cwd()
            os.chdir(temporary)
            try:
                sentencepiece.SentencePieceTrainer.train(
                    sentence_iterator=iter(sentences),
                    model_prefix="tokenizer",
                    model_type="bpe",
                    vocab_size=config.vocab_size,
                    character_coverage=config.character_coverage,
                    byte_fallback=config.byte_fallback,
                    normalization_rule_name=config.normalization_rule,
                    remove_extra_whitespaces=False,
                    split_by_whitespace=True,
                    shuffle_input_sentence=False,
                    input_sentence_size=0,
                    num_threads=1,
                    hard_vocab_limit=True,
                    max_sentence_length=65_536,
                    unk_id=UNK_ID,
                    bos_id=BOS_ID,
                    eos_id=EOS_ID,
                    pad_id=PAD_ID,
                    user_defined_symbols=list(config.special_symbols),
                    minloglevel=2,
                )
            finally:
                os.chdir(previous)
        model_path = temporary / "tokenizer.model"
        vocab_path = temporary / "tokenizer.vocab"
        processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
        if (
            processor.unk_id() != UNK_ID
            or processor.bos_id() != BOS_ID
            or processor.eos_id() != EOS_ID
            or processor.pad_id() != PAD_ID
        ):
            raise ValueError("trained tokenizer special IDs do not match the contract")
        actual_vocab_size = processor.vocab_size()
        if actual_vocab_size != config.vocab_size:
            raise ValueError("trained tokenizer did not produce the requested vocabulary size")
        draft = TokenizerArtifactManifest.model_construct(
            tokenizer_version=config.tokenizer_version,
            algorithm=config.algorithm,
            sentencepiece_version=sentencepiece.__version__,
            requested_vocab_size=config.vocab_size,
            actual_vocab_size=actual_vocab_size,
            unk_id=UNK_ID,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            pad_id=PAD_ID,
            special_symbols=config.special_symbols,
            model_sha256=_sha256(model_path),
            vocab_sha256=_sha256(vocab_path),
            model_size_bytes=model_path.stat().st_size,
            vocab_size_bytes=vocab_path.stat().st_size,
            corpus=corpus.manifest,
            checksum_sha256="0" * 64,
        )
        checksum = canonical_sha256(
            draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        manifest = TokenizerArtifactManifest(
            **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
            checksum_sha256=checksum,
        )
        (temporary / "manifest.json").write_bytes(
            canonical_json_bytes(manifest.model_dump(mode="json", round_trip=True)) + b"\n"
        )
        os.rename(temporary, output_directory)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class ProjectTokenizer:
    """Checksum-verified tokenizer loaded only from a trusted project artifact."""

    def __init__(
        self,
        processor: sentencepiece.SentencePieceProcessor,
        manifest: TokenizerArtifactManifest,
    ) -> None:
        self._processor = processor
        self.manifest = manifest

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        expected_checksum: str | None = None,
    ) -> ProjectTokenizer:
        if not isinstance(directory, Path) or directory.is_symlink() or not directory.is_dir():
            raise ValueError("tokenizer directory must be a regular non-symlink directory")
        expected_files = {"manifest.json", "tokenizer.model", "tokenizer.vocab"}
        if {path.name for path in directory.iterdir()} != expected_files:
            raise ValueError("tokenizer directory contains an unexpected file inventory")
        manifest_path = directory / "manifest.json"
        model_path = directory / "tokenizer.model"
        vocab_path = directory / "tokenizer.vocab"
        for path in (manifest_path, model_path, vocab_path):
            if path.is_symlink() or not path.is_file():
                raise ValueError("tokenizer artifact contains a symlink or non-file")
        if manifest_path.stat().st_size > MAX_TOKENIZER_MANIFEST_BYTES:
            raise ValueError("tokenizer manifest exceeds its size limit")
        manifest_payload = manifest_path.read_bytes()
        _strict_json_object(manifest_payload)
        manifest = TokenizerArtifactManifest.model_validate_json(manifest_payload)
        if expected_checksum is not None and manifest.checksum_sha256 != expected_checksum:
            raise ValueError("tokenizer manifest does not match the expected checksum")
        if (
            model_path.stat().st_size != manifest.model_size_bytes
            or _sha256(model_path) != manifest.model_sha256
        ):
            raise ValueError("tokenizer model checksum or size mismatch")
        if (
            vocab_path.stat().st_size != manifest.vocab_size_bytes
            or _sha256(vocab_path) != manifest.vocab_sha256
        ):
            raise ValueError("tokenizer vocabulary checksum or size mismatch")
        processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
        if processor.vocab_size() != manifest.actual_vocab_size:
            raise ValueError("tokenizer runtime vocabulary differs from its manifest")
        return cls(processor, manifest)

    @property
    def vocab_size(self) -> int:
        return self.manifest.actual_vocab_size

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> tuple[int, ...]:
        if type(text) is not str or not text or len(text.encode("utf-8")) > 1024 * 1024:
            raise ValueError("tokenizer input must be non-empty and at most 1 MiB")
        token_ids = tuple(
            self._processor.encode(text, out_type=int, add_bos=add_bos, add_eos=add_eos)
        )
        if UNK_ID in token_ids:
            raise ValueError("byte-fallback tokenizer unexpectedly emitted an unknown token")
        return token_ids

    def decode(self, token_ids: tuple[int, ...]) -> str:
        if type(token_ids) is not tuple or not token_ids:
            raise ValueError("token IDs must be a non-empty tuple")
        if any(
            type(token_id) is not int or not 0 <= token_id < self.vocab_size
            for token_id in token_ids
        ):
            raise ValueError("token ID is outside the tokenizer vocabulary")
        return cast(str, self._processor.decode(token_ids))


__all__ = [
    "BOS_ID",
    "EOS_ID",
    "PAD_ID",
    "UNK_ID",
    "ProjectTokenizer",
    "TokenizerArtifactManifest",
    "TrainingCorpus",
    "TrainingCorpusManifest",
    "approved_training_corpus",
    "train_tokenizer",
    "training_corpus_from_rendered_candidates",
]
