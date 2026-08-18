"""Latent process-state contracts.

These records contain only the invented state-machine truth.  Fault labels and
rendered language are intentionally absent so they cannot leak into model-visible
data through this layer.
"""

from __future__ import annotations

from pydantic import StrictBool, field_validator, model_validator

from .base import (
    SCHEMA_VERSION,
    ContractId,
    ContractModel,
    NonNegativeInt,
    NormalizedFloat,
    SchemaVersion,
    canonical_string_tuple,
)
from .enums import ComponentState, OperatingMode


class PlantValues(ContractModel):
    heat_source_level: NormalizedFloat
    primary_flow: NormalizedFloat
    primary_thermal_state: NormalizedFloat
    primary_inventory: NormalizedFloat
    transfer_efficiency: NormalizedFloat
    secondary_flow: NormalizedFloat
    secondary_inventory: NormalizedFloat
    steam_state: NormalizedFloat
    condenser_function: NormalizedFloat
    heat_rejection: NormalizedFloat
    turbine_output: NormalizedFloat
    electrical_output: NormalizedFloat
    load_demand: NormalizedFloat
    support_power: NormalizedFloat


class ComponentLatentState(ContractModel):
    component_id: ContractId
    state: ComponentState
    health: NormalizedFloat
    commanded_position: NormalizedFloat | None = None
    actual_position: NormalizedFloat | None = None
    pending_maintenance: StrictBool = False

    @model_validator(mode="after")
    def positions_are_paired(self) -> ComponentLatentState:
        if (self.commanded_position is None) != (self.actual_position is None):
            raise ValueError("commanded_position and actual_position must be present together")
        return self


class LatentPlantState(ContractModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    tick: NonNegativeInt
    operating_mode: OperatingMode
    values: PlantValues
    components: tuple[ComponentLatentState, ...]

    @field_validator("components", mode="after")
    @classmethod
    def components_are_canonical(
        cls, values: tuple[ComponentLatentState, ...]
    ) -> tuple[ComponentLatentState, ...]:
        component_ids = canonical_string_tuple(
            tuple(item.component_id for item in values), field_name="component_ids"
        )
        by_id = {item.component_id: item for item in values}
        return tuple(by_id[component_id] for component_id in component_ids)
