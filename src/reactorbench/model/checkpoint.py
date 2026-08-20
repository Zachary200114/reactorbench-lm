"""Checksum-bound, data-only Transformer checkpoint artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field, StrictFloat, model_validator
from safetensors.torch import load_file, save_file

from reactorbench.schemas.base import (
    ContractModel,
    SemanticVersion,
    canonical_json_bytes,
    canonical_sha256,
)
from reactorbench.tokenizer import TokenizerArtifactManifest

from .config import TransformerConfig
from .transformer import TransformerLM, exact_parameter_count, initialized_model

CHECKPOINT_ARTIFACT_VERSION: SemanticVersion = "0.1.0"
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024


class CheckpointManifest(ContractModel):
    artifact_version: Literal["0.1.0"] = "0.1.0"
    model_version: SemanticVersion
    architecture: Literal["decoder_only_causal_transformer"] = "decoder_only_causal_transformer"
    initialization: Literal["project_random_normal_0.02"] = "project_random_normal_0.02"
    weights_format: Literal["safetensors"] = "safetensors"
    transformer_config: TransformerConfig
    vocab_size: int = Field(strict=True, ge=8, le=65_536)
    parameter_count: int = Field(strict=True, ge=1)
    tokenizer_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    seed: int = Field(strict=True, ge=0, le=4_294_967_295)
    training_steps: int = Field(strict=True, ge=0)
    initial_loss: StrictFloat
    final_loss: StrictFloat
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weights_size_bytes: int = Field(strict=True, ge=1, le=MAX_CHECKPOINT_BYTES)
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def derived_values_are_exact(self) -> CheckpointManifest:
        if not (self.initial_loss >= 0.0 and self.final_loss >= 0.0):
            raise ValueError("checkpoint losses must be finite and non-negative")
        expected_parameters = exact_parameter_count(
            self.transformer_config, vocab_size=self.vocab_size
        )
        if self.parameter_count != expected_parameters:
            raise ValueError("checkpoint parameter count does not match its configuration")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("checkpoint manifest checksum mismatch")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        while chunk := artifact.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError("checkpoint manifest contains a duplicate key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"checkpoint manifest contains non-finite JSON: {value}")

    decoded = json.loads(
        payload.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject_constant
    )
    if type(decoded) is not dict:
        raise ValueError("checkpoint manifest must contain one object")
    return decoded


def save_checkpoint(
    model: TransformerLM,
    *,
    output_directory: Path,
    tokenizer_manifest: TokenizerArtifactManifest,
    source_commit: str,
    seed: int,
    training_steps: int,
    initial_loss: float,
    final_loss: float,
) -> CheckpointManifest:
    """Atomically save cloned tensors without pickle or arbitrary Python objects."""

    if (
        type(model) is not TransformerLM
        or type(tokenizer_manifest) is not TokenizerArtifactManifest
    ):
        raise TypeError("checkpoint save requires exact model and tokenizer manifest objects")
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be a pathlib.Path")
    parent = output_directory.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    if parent.is_symlink() or not parent.is_dir() or output_directory.exists():
        raise FileExistsError("checkpoint output must be a new contained directory")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.tmp-", dir=parent))
    try:
        weights_path = temporary / "model.safetensors"
        state = {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in model.state_dict().items()
        }
        save_file(state, str(weights_path), metadata={"format": "reactorbench-lm-v0.1.0"})
        weights_size = weights_path.stat().st_size
        if weights_size > MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint exceeds its size limit")
        draft = CheckpointManifest.model_construct(
            model_version=model.config.model_version,
            transformer_config=model.config,
            vocab_size=model.vocab_size,
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            tokenizer_manifest_sha256=tokenizer_manifest.checksum_sha256,
            corpus_sha256=tokenizer_manifest.corpus.corpus_sha256,
            dataset_candidate_sha256=tokenizer_manifest.corpus.candidate_bundle_sha256,
            source_commit=source_commit,
            seed=seed,
            training_steps=training_steps,
            initial_loss=float(initial_loss),
            final_loss=float(final_loss),
            weights_sha256=_sha256(weights_path),
            weights_size_bytes=weights_size,
            checksum_sha256="0" * 64,
        )
        checksum = canonical_sha256(
            draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        manifest = CheckpointManifest(
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


def load_checkpoint(
    directory: Path,
    *,
    expected_manifest_sha256: str,
    expected_tokenizer_sha256: str,
    device: torch.device,
) -> tuple[TransformerLM, CheckpointManifest]:
    """Verify and load only safetensors from one trusted checkpoint directory."""

    if not isinstance(directory, Path) or directory.is_symlink() or not directory.is_dir():
        raise ValueError("checkpoint directory must be a regular non-symlink directory")
    if {path.name for path in directory.iterdir()} != {"manifest.json", "model.safetensors"}:
        raise ValueError("checkpoint directory contains an unexpected file inventory")
    manifest_path = directory / "manifest.json"
    weights_path = directory / "model.safetensors"
    if any(path.is_symlink() or not path.is_file() for path in (manifest_path, weights_path)):
        raise ValueError("checkpoint contains a symlink or non-file")
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("checkpoint manifest exceeds its size limit")
    manifest_payload = manifest_path.read_bytes()
    _strict_json_object(manifest_payload)
    manifest = CheckpointManifest.model_validate_json(manifest_payload)
    if manifest.checksum_sha256 != expected_manifest_sha256:
        raise ValueError("checkpoint manifest does not match the expected checksum")
    if manifest.tokenizer_manifest_sha256 != expected_tokenizer_sha256:
        raise ValueError("checkpoint is bound to a different tokenizer")
    if (
        weights_path.stat().st_size != manifest.weights_size_bytes
        or _sha256(weights_path) != manifest.weights_sha256
    ):
        raise ValueError("checkpoint weight checksum or size mismatch")
    model = initialized_model(
        manifest.transformer_config, vocab_size=manifest.vocab_size, seed=manifest.seed
    )
    state = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, manifest


__all__ = ["CheckpointManifest", "load_checkpoint", "save_checkpoint"]
