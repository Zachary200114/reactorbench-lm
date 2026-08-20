"""Strict Phase 5 baseline and pilot configuration."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, field_validator, model_validator

from reactorbench.model.config import StrictConfigModel, _relative_project_path
from reactorbench.schemas.base import ContractId
from reactorbench.schemas.enums import SplitName, TaskName

PositiveFloat = Annotated[StrictFloat, Field(gt=0.0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[StrictFloat, Field(ge=0.0, allow_inf_nan=False)]


class Phase5Paths(StrictConfigModel):
    config_version: Literal["0.1.0"]
    phase4_config_path: str
    phase4_run_path: str
    run_root: str
    run_name: ContractId

    @field_validator("phase4_config_path", "phase4_run_path", "run_root", mode="after")
    @classmethod
    def paths_are_contained(cls, value: str) -> str:
        return _relative_project_path(value)


class Phase5DataConfig(StrictConfigModel):
    train_split: Literal[SplitName.IID_TRAIN]
    validation_split: Literal[SplitName.IID_VALIDATION]
    prohibited_splits: tuple[SplitName, ...]
    classification_tasks: tuple[TaskName, ...]

    @field_validator("train_split", "validation_split", mode="before")
    @classmethod
    def split_literals_become_enums(cls, value: object) -> object:
        if type(value) is str:
            return SplitName(value)
        return value

    @field_validator("prohibited_splits", "classification_tasks", mode="before")
    @classmethod
    def arrays_become_tuples(cls, value: object) -> object:
        if type(value) is list:
            values: tuple[object, ...] = tuple(value)
        elif type(value) is tuple:
            values = value
        else:
            raise ValueError("Phase 5 enum collections must be exact lists or tuples")
        if not values:
            return values
        enum_type: type[SplitName] | type[TaskName]
        enum_type = SplitName if values[0] in {item.value for item in SplitName} else TaskName
        if any(type(item) is not str and not isinstance(item, enum_type) for item in values):
            raise ValueError("Phase 5 enum collections contain an invalid member")
        return tuple(enum_type(item) if type(item) is str else item for item in values)

    @model_validator(mode="after")
    def split_and_task_sets_are_exact(self) -> Phase5DataConfig:
        expected_splits = tuple(
            split
            for split in SplitName
            if split not in {SplitName.IID_TRAIN, SplitName.IID_VALIDATION}
        )
        if self.prohibited_splits != expected_splits:
            raise ValueError("prohibited_splits must list every non-train/non-validation split")
        expected_tasks = (
            TaskName.FAULT_FAMILY,
            TaskName.NEXT_ACTION,
            TaskName.CONTINUE_LOG,
        )
        if self.classification_tasks != expected_tasks:
            raise ValueError("classification_tasks must match the preregistered order")
        return self


class SerializationConfig(StrictConfigModel):
    serialization_version: Literal["0.1.0"]
    prompt_prefix: Literal["<|prompt|>"]
    target_prefix: Literal["<|target|>"]
    record_separator: Literal["<|sep|>"]
    truncation: Literal["retain_prompt_suffix"]
    maximum_prompt_utf8_bytes: Annotated[StrictInt, Field(ge=1024, le=1024 * 1024)]


class BaselineConfig(StrictConfigModel):
    ngram_order: Literal[3]
    ngram_additive_smoothing: PositiveFloat
    bow_max_features: Annotated[StrictInt, Field(ge=128, le=8192)]
    bow_steps: Annotated[StrictInt, Field(ge=10, le=10_000)]
    bow_learning_rate: PositiveFloat
    bow_l2: NonNegativeFloat
    gru_embedding_width: Annotated[StrictInt, Field(ge=8, le=512)]
    gru_hidden_width: Annotated[StrictInt, Field(ge=8, le=1024)]
    gru_epochs: Annotated[StrictInt, Field(ge=1, le=1000)]
    gru_batch_size: Annotated[StrictInt, Field(ge=1, le=256)]
    gru_learning_rate: PositiveFloat
    gru_max_tokens: Annotated[StrictInt, Field(ge=8, le=4096)]


class TransformerTrainingConfig(StrictConfigModel):
    seed: Annotated[StrictInt, Field(ge=0, le=4_294_967_295)]
    device: Literal["cpu", "mps"]
    allow_cpu_fallback: StrictBool
    steps: Annotated[StrictInt, Field(ge=1, le=10_000)]
    batch_size: Annotated[StrictInt, Field(ge=1, le=128)]
    learning_rate: PositiveFloat
    weight_decay: NonNegativeFloat
    gradient_clip_norm: PositiveFloat
    evaluation_interval: Annotated[StrictInt, Field(ge=1, le=1000)]

    @model_validator(mode="after")
    def evaluation_schedule_is_exact(self) -> TransformerTrainingConfig:
        if self.steps % self.evaluation_interval:
            raise ValueError("training steps must be divisible by evaluation_interval")
        return self


class Phase5AcceptanceConfig(StrictConfigModel):
    minimum_validation_nll_reduction_fraction: Annotated[
        StrictFloat, Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    ]
    maximum_report_bytes: Annotated[StrictInt, Field(ge=1024, le=64 * 1024 * 1024)]
    maximum_run_bytes: Annotated[StrictInt, Field(ge=1024 * 1024, le=4 * 1024**3)]
    require_all_baselines: Literal[True]
    require_validation_only_selection: Literal[True]


class Phase5Config(StrictConfigModel):
    phase5: Phase5Paths
    data: Phase5DataConfig
    serialization: SerializationConfig
    baselines: BaselineConfig
    smaller_transformer: TransformerTrainingConfig
    pilot_transformer: TransformerTrainingConfig
    acceptance: Phase5AcceptanceConfig

    @model_validator(mode="after")
    def tier_schedule_is_preregistered(self) -> Phase5Config:
        if self.smaller_transformer.steps != 300 or self.pilot_transformer.steps != 500:
            raise ValueError("Phase 5 Transformer schedules must remain 300/500 steps")
        if self.smaller_transformer.seed == self.pilot_transformer.seed:
            raise ValueError("Transformer tiers require distinct fixed seeds")
        return self


def load_phase5_config(path: Path) -> Phase5Config:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise ValueError("Phase 5 config must be a regular non-symlink file")
    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    return Phase5Config.model_validate(raw)


__all__ = [
    "BaselineConfig",
    "Phase5AcceptanceConfig",
    "Phase5Config",
    "Phase5DataConfig",
    "Phase5Paths",
    "SerializationConfig",
    "TransformerTrainingConfig",
    "load_phase5_config",
]
