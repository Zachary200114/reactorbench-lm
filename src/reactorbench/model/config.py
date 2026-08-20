"""Strict Phase 4 tokenizer, model, and smoke-training configuration."""

from __future__ import annotations

import tomllib
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from reactorbench.schemas.base import ContractId, SemanticVersion

NonNegativeFloat = Annotated[StrictFloat, Field(ge=0.0, allow_inf_nan=False)]
PositiveFloat = Annotated[StrictFloat, Field(gt=0.0, allow_inf_nan=False)]
UnitFloat = Annotated[StrictFloat, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


def _relative_project_path(value: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise ValueError("project paths must be non-empty POSIX relative paths")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("project paths must be canonical and contained")
    return value


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class Phase4Paths(StrictConfigModel):
    config_version: Literal["0.1.0"]
    dataset_root: str
    dataset_artifact_name: ContractId
    postrender_approval_record: str
    run_root: str
    run_name: ContractId

    @field_validator("dataset_root", "postrender_approval_record", "run_root", mode="after")
    @classmethod
    def paths_are_project_relative(cls, value: str) -> str:
        return _relative_project_path(value)


class TokenizerConfig(StrictConfigModel):
    tokenizer_version: SemanticVersion
    algorithm: Literal["sentencepiece_bpe"]
    vocab_size: Annotated[StrictInt, Field(ge=512, le=16_384)]
    character_coverage: Annotated[StrictFloat, Field(gt=0.0, le=1.0, allow_inf_nan=False)]
    byte_fallback: Literal[True]
    normalization_rule: Literal["identity"]
    special_symbols: tuple[str, ...]

    @field_validator("special_symbols", mode="before")
    @classmethod
    def toml_array_becomes_tuple(cls, value: object) -> object:
        if type(value) is list:
            return tuple(value)
        if type(value) is tuple:
            return value
        raise ValueError("special_symbols must be an exact list or tuple")

    @field_validator("special_symbols", mode="after")
    @classmethod
    def symbols_are_exact_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        expected = ("<|prompt|>", "<|target|>", "<|sep|>")
        if value != expected:
            raise ValueError(f"special_symbols must be exactly {expected!r}")
        return value


class TransformerConfig(StrictConfigModel):
    model_version: SemanticVersion
    layers: Annotated[StrictInt, Field(ge=1, le=32)]
    width: Annotated[StrictInt, Field(ge=16, le=2048)]
    heads: Annotated[StrictInt, Field(ge=1, le=32)]
    context_length: Annotated[StrictInt, Field(ge=8, le=4096)]
    feed_forward_multiplier: Annotated[StrictInt, Field(ge=2, le=8)]
    dropout: UnitFloat
    tie_embeddings: StrictBool
    bias: StrictBool

    @model_validator(mode="after")
    def attention_dimensions_are_valid(self) -> TransformerConfig:
        if self.width % self.heads:
            raise ValueError("model width must be divisible by the number of heads")
        return self


class SmokeTrainingConfig(StrictConfigModel):
    seed: Annotated[StrictInt, Field(ge=0, le=4_294_967_295)]
    device: Literal["cpu", "mps"]
    document_count: Annotated[StrictInt, Field(ge=1, le=64)]
    batch_size: Annotated[StrictInt, Field(ge=1, le=128)]
    steps: Annotated[StrictInt, Field(ge=1, le=10_000)]
    learning_rate: PositiveFloat
    weight_decay: NonNegativeFloat
    gradient_clip_norm: PositiveFloat
    maximum_final_loss: PositiveFloat
    minimum_loss_reduction_fraction: Annotated[
        StrictFloat, Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    ]


class Phase4Config(StrictConfigModel):
    phase4: Phase4Paths
    tokenizer: TokenizerConfig
    smoke_model: TransformerConfig
    pilot_model: TransformerConfig
    main_model: TransformerConfig
    smoke_training: SmokeTrainingConfig

    @model_validator(mode="after")
    def model_tiers_are_monotonic(self) -> Phase4Config:
        tiers = (self.smoke_model, self.pilot_model, self.main_model)
        for previous, current in pairwise(tiers):
            if not (
                previous.layers < current.layers
                and previous.width < current.width
                and previous.context_length < current.context_length
            ):
                raise ValueError("smoke, pilot, and main tiers must increase monotonically")
        return self


def load_phase4_config(path: Path) -> Phase4Config:
    """Load strict Phase 4 TOML with unknown-field and coercion rejection."""

    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise ValueError("Phase 4 config must be a regular non-symlink file")
    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    return Phase4Config.model_validate(raw)


def resolve_project_path(project_root: Path, relative: str, *, must_exist: bool) -> Path:
    """Resolve one reviewed relative path below a trusted project root."""

    if not isinstance(project_root, Path) or project_root.is_symlink() or not project_root.is_dir():
        raise ValueError("project_root must be an existing non-symlink directory")
    relative = _relative_project_path(relative)
    root = project_root.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("project path contains a symlink")
    resolved = candidate.resolve(strict=must_exist)
    if not resolved.is_relative_to(root):
        raise ValueError("project path escapes the repository")
    return resolved


__all__ = [
    "Phase4Config",
    "Phase4Paths",
    "SmokeTrainingConfig",
    "TokenizerConfig",
    "TransformerConfig",
    "load_phase4_config",
    "resolve_project_path",
]
