"""Strict, explicit configuration for the Phase 3 development-data pipeline."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reactorbench.schemas.base import ContractId, SemanticVersion

Seed = Annotated[int, Field(strict=True, ge=0, le=4_294_967_295)]
PositiveBound = Annotated[int, Field(strict=True, gt=0)]


class DatasetSplitConfig(BaseModel):
    """Renderer plan and globally exclusive seeds for one named split."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    seeds: tuple[Seed, ...]
    template_families: tuple[ContractId, ...]
    alias_families: tuple[ContractId, ...]

    @field_validator("seeds", "template_families", "alias_families", mode="before")
    @classmethod
    def toml_arrays_become_tuples(cls, value: object) -> object:
        if type(value) is not list:
            raise ValueError("dataset split arrays must be TOML arrays")
        return tuple(value)

    @model_validator(mode="after")
    def members_are_nonempty_and_unique(self) -> DatasetSplitConfig:
        for name in ("seeds", "template_families", "alias_families"):
            values = getattr(self, name)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be nonempty and unique")
        if self.seeds != tuple(sorted(self.seeds)):
            raise ValueError("seeds must be in increasing order")
        return self


class DatasetSplitPlan(BaseModel):
    """Exact closed set of Phase 3 split plans."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    iid_train: DatasetSplitConfig
    iid_validation: DatasetSplitConfig
    iid_test: DatasetSplitConfig
    template_test: DatasetSplitConfig
    component_test: DatasetSplitConfig
    severity_test: DatasetSplitConfig
    composition_test: DatasetSplitConfig
    counterfactual_test: DatasetSplitConfig
    noise_test: DatasetSplitConfig

    @model_validator(mode="after")
    def seeds_are_globally_exclusive(self) -> DatasetSplitPlan:
        owners: dict[int, str] = {}
        for split_name in type(self).model_fields:
            for seed in getattr(self, split_name).seeds:
                previous = owners.setdefault(seed, split_name)
                if previous != split_name:
                    raise ValueError(f"seed {seed} appears in both {previous} and {split_name}")
        return self


class DatasetBuildSettings(BaseModel):
    """Bounded artifact and version settings."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    dataset_version: SemanticVersion
    renderer_version: Literal["0.1.0"]
    projection_version: Literal["0.1.0"]
    manifest_version: Literal["0.1.0"]
    schema_snapshot_version: Literal["0.1.0"]
    artifact_name: ContractId
    duration_ticks: Annotated[int, Field(strict=True, ge=9, le=64)]
    minimum_trajectories: PositiveBound
    maximum_trajectories: PositiveBound
    maximum_task_records: PositiveBound
    maximum_rendered_bytes: PositiveBound
    golden_reserved_seed_max: Seed
    overwrite: Literal[False]

    @model_validator(mode="after")
    def trajectory_range_is_ordered(self) -> DatasetBuildSettings:
        if self.minimum_trajectories > self.maximum_trajectories:
            raise ValueError("minimum_trajectories cannot exceed maximum_trajectories")
        return self


class DatasetReviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    required_pre_render_status: Literal["APPROVED"]
    generated_status: Literal["PENDING_POSTRENDER_HUMAN_REVIEW"]
    reviewer_role: Literal["project-owner"]
    require_catalog_hash_match: Literal[True]
    require_rules_hash_match: Literal[True]


class DatasetQualityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ngram_sizes: tuple[Literal[3, 4, 5], ...]
    fail_on_exact_duplicate: Literal[True]
    fail_on_task_scoped_model_input_duplicate: Literal[True]
    fail_on_forbidden_skeleton_overlap: Literal[True]
    fail_on_prohibited_content: Literal[True]
    report_ngram_overlap_without_threshold: Literal[True]

    @field_validator("ngram_sizes", mode="before")
    @classmethod
    def toml_array_becomes_tuple(cls, value: object) -> object:
        if type(value) is not list:
            raise ValueError("ngram_sizes must be a TOML array")
        return tuple(value)

    @field_validator("ngram_sizes", mode="after")
    @classmethod
    def sizes_are_exact(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value != (3, 4, 5):
            raise ValueError("ngram_sizes must be exactly [3, 4, 5]")
        return value


class DevelopmentDatasetConfig(BaseModel):
    """Complete reviewed Phase 3 development-candidate configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    dataset: DatasetBuildSettings
    splits: DatasetSplitPlan
    review: DatasetReviewConfig
    quality: DatasetQualityConfig

    @model_validator(mode="after")
    def reserved_seeds_are_absent(self) -> DevelopmentDatasetConfig:
        for split_name in type(self.splits).model_fields:
            if min(getattr(self.splits, split_name).seeds) <= self.dataset.golden_reserved_seed_max:
                raise ValueError("golden-reserved seeds cannot enter the development candidate")
        return self


def load_development_dataset_config(path: Path) -> DevelopmentDatasetConfig:
    """Load one explicit TOML file with unknown-field rejection."""

    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    return DevelopmentDatasetConfig.model_validate(raw)


def canonical_dataset_config_bytes(config: DevelopmentDatasetConfig) -> bytes:
    payload = config.model_dump(mode="json")
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def dataset_config_sha256(config: DevelopmentDatasetConfig) -> str:
    return hashlib.sha256(canonical_dataset_config_bytes(config)).hexdigest()


__all__ = [
    "DatasetBuildSettings",
    "DatasetQualityConfig",
    "DatasetReviewConfig",
    "DatasetSplitConfig",
    "DatasetSplitPlan",
    "DevelopmentDatasetConfig",
    "canonical_dataset_config_bytes",
    "dataset_config_sha256",
    "load_development_dataset_config",
]
