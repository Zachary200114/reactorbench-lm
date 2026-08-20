"""Strict Phase 5 baseline and pilot configuration."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from reactorbench.model.config import StrictConfigModel, TransformerConfig, _relative_project_path
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
    model_context_length: Literal[512]
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


class Phase6Paths(StrictConfigModel):
    contract_version: Literal["0.1.0"]
    phase4_config_path: str
    phase5_report_path: str
    phase5_report_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    phase5_config_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    run_root: str
    run_name: ContractId

    @field_validator("phase4_config_path", "phase5_report_path", "run_root", mode="after")
    @classmethod
    def paths_are_contained(cls, value: str) -> str:
        return _relative_project_path(value)


class Phase6SelectionConfig(StrictConfigModel):
    metric: Literal["iid_validation_target_nll"]
    direction: Literal["lower"]
    tie_break: Literal["earlier_step"]
    early_stopping: Literal[False]
    minimum_validation_nll_reduction_fraction: PositiveFloat
    maximum_selected_validation_nll: PositiveFloat
    minimum_relative_nll_improvement_over_smaller: PositiveFloat

    @model_validator(mode="after")
    def thresholds_are_frozen(self) -> Phase6SelectionConfig:
        if (
            self.minimum_validation_nll_reduction_fraction,
            self.maximum_selected_validation_nll,
            self.minimum_relative_nll_improvement_over_smaller,
        ) != (0.90, 0.50, 0.10):
            raise ValueError("Phase 6 selection thresholds differ from the pilot freeze")
        return self


class Phase6EvaluationConfig(StrictConfigModel):
    bootstrap_resamples: Literal[2000]
    bootstrap_seed: Literal[6602]
    confidence_level: PositiveFloat
    minimum_fault_macro_f1_margin_over_best_simple: NonNegativeFloat
    minimum_next_action_macro_f1_margin_over_best_simple: NonNegativeFloat
    minimum_continue_log_macro_f1: NonNegativeFloat
    maximum_target_nll_fraction_of_trigram: PositiveFloat
    minimum_evidence_f1: NonNegativeFloat
    minimum_parse_success_rate: NonNegativeFloat
    minimum_schema_validity_rate: NonNegativeFloat
    maximum_no_fault_false_positive_rate: NonNegativeFloat
    minimum_required_abstention_accuracy: NonNegativeFloat
    maximum_expected_calibration_error: NonNegativeFloat
    selective_risk_coverage: PositiveFloat
    maximum_selective_risk: NonNegativeFloat
    composition_has_pass_threshold: Literal[False]
    report_every_split_separately: Literal[True]
    require_golden_suite_approval_before_test: Literal[True]

    @model_validator(mode="after")
    def thresholds_are_frozen(self) -> Phase6EvaluationConfig:
        observed = (
            self.confidence_level,
            self.minimum_fault_macro_f1_margin_over_best_simple,
            self.minimum_next_action_macro_f1_margin_over_best_simple,
            self.minimum_continue_log_macro_f1,
            self.maximum_target_nll_fraction_of_trigram,
            self.minimum_evidence_f1,
            self.minimum_parse_success_rate,
            self.minimum_schema_validity_rate,
            self.maximum_no_fault_false_positive_rate,
            self.minimum_required_abstention_accuracy,
            self.maximum_expected_calibration_error,
            self.selective_risk_coverage,
            self.maximum_selective_risk,
        )
        expected = (0.95, 0.02, 0.02, 0.90, 0.75, 0.70, 0.99, 0.99, 0.10, 0.80, 0.15, 0.80, 0.20)
        if observed != expected:
            raise ValueError("Phase 6 evaluation thresholds differ from the pilot freeze")
        return self


class Phase6ExperimentMatrix(StrictConfigModel):
    required: tuple[StrictStr, ...]

    @field_validator("required", mode="before")
    @classmethod
    def array_becomes_tuple(cls, value: object) -> object:
        if type(value) is list:
            values = tuple(value)
        elif type(value) is tuple:
            values = value
        else:
            raise ValueError("Phase 6 experiment matrix must be an exact string array")
        if any(type(item) is not str for item in values):
            raise ValueError("Phase 6 experiment matrix must be an exact string array")
        return values

    @model_validator(mode="after")
    def matrix_is_exact(self) -> Phase6ExperimentMatrix:
        expected = tuple(
            f"E{index}_{name}"
            for index, name in enumerate(
                (
                    "simple_baselines",
                    "recurrent_baselines",
                    "smaller_transformer",
                    "main_transformer",
                    "event_order_ablation",
                    "renderer_diversity_ablation",
                    "abstention_ablation",
                    "compound_training_ablation",
                )
            )
        )
        if self.required != expected:
            raise ValueError("Phase 6 experiment matrix must contain exact E0-E7 order")
        return self


class Phase6Config(StrictConfigModel):
    phase6: Phase6Paths
    data: Phase5DataConfig
    model: TransformerConfig
    training: TransformerTrainingConfig
    selection: Phase6SelectionConfig
    evaluation: Phase6EvaluationConfig
    experiments: Phase6ExperimentMatrix

    @model_validator(mode="after")
    def pilot_informed_contract_is_exact(self) -> Phase6Config:
        expected_model = {
            "model_version": "0.1.0",
            "layers": 8,
            "width": 384,
            "heads": 8,
            "context_length": 512,
            "feed_forward_multiplier": 4,
            "dropout": 0.1,
            "tie_embeddings": True,
            "bias": True,
        }
        if self.model.model_dump(mode="python") != expected_model:
            raise ValueError("Phase 6 main model must match the frozen 8x384 tier")
        expected_training = {
            "seed": 6601,
            "device": "mps",
            "allow_cpu_fallback": False,
            "steps": 1500,
            "batch_size": 4,
            "learning_rate": 0.00025,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "evaluation_interval": 100,
        }
        if self.training.model_dump(mode="python") != expected_training:
            raise ValueError("Phase 6 training schedule differs from the frozen pilot decision")
        return self


def load_phase5_config(path: Path) -> Phase5Config:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise ValueError("Phase 5 config must be a regular non-symlink file")
    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    return Phase5Config.model_validate(raw)


def load_phase6_config(path: Path) -> Phase6Config:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise ValueError("Phase 6 config must be a regular non-symlink file")
    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    return Phase6Config.model_validate(raw)


__all__ = [
    "BaselineConfig",
    "Phase5AcceptanceConfig",
    "Phase5Config",
    "Phase5DataConfig",
    "Phase5Paths",
    "Phase6Config",
    "Phase6EvaluationConfig",
    "Phase6ExperimentMatrix",
    "Phase6Paths",
    "Phase6SelectionConfig",
    "SerializationConfig",
    "TransformerTrainingConfig",
    "load_phase5_config",
    "load_phase6_config",
]
