"""Reproducible example-provenance contract."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StrictStr, field_validator

from .base import (
    SCHEMA_VERSION,
    ContractId,
    ContractModel,
    SchemaVersion,
    SeedInt,
    SemanticVersion,
    canonical_enum_tuple,
    canonical_sha256,
    canonical_string_tuple,
)
from .enums import FaultFamily, PlantVariant, SplitName, TaskName

GitCommit = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{7,64}$")]


class ProvenanceRecord(ContractModel):
    dataset_version: SemanticVersion
    generator_commit: GitCommit
    scenario_schema_version: SchemaVersion = SCHEMA_VERSION
    renderer_version: SemanticVersion
    seed: SeedInt
    trajectory_id: ContractId
    scenario_id: ContractId
    plant_variant_id: PlantVariant
    fault_family_ids: tuple[FaultFamily, ...] = ()
    template_family_ids: tuple[ContractId, ...]
    split_name: SplitName
    task_name: TaskName

    @field_validator("fault_family_ids", mode="after")
    @classmethod
    def faults_are_a_canonical_set(cls, values: tuple[FaultFamily, ...]) -> tuple[FaultFamily, ...]:
        return canonical_enum_tuple(values, enum_type=FaultFamily, field_name="fault_family_ids")

    @field_validator("template_family_ids", mode="after")
    @classmethod
    def templates_are_a_canonical_set(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return canonical_string_tuple(values, field_name="template_family_ids")

    def stable_hash(self) -> str:
        """Return the stable provenance identifier for this exact record."""

        return canonical_sha256(self.model_dump(mode="json", round_trip=True))
