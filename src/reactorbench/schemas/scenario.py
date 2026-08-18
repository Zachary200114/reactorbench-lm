"""Scenario-definition and scheduled-action contracts."""

from __future__ import annotations

from pydantic import field_validator, model_validator

from .base import (
    SCHEMA_VERSION,
    ContractId,
    ContractModel,
    NonNegativeInt,
    PositiveInt,
    SchemaVersion,
    SeedInt,
)
from .enums import (
    ActionLabel,
    FaultFamily,
    PlantVariant,
    ScenarioDriver,
    SeverityBand,
)

_FAULT_ORDER = {fault: index for index, fault in enumerate(FaultFamily)}
_SENSOR_FAULTS = {
    FaultFamily.SENSOR_DRIFT,
    FaultFamily.SENSOR_STUCK,
    FaultFamily.SENSOR_NOISE,
}


class FaultInjection(ContractModel):
    fault_family: FaultFamily
    component_id: ContractId
    onset_tick: NonNegativeInt
    severity: SeverityBand
    channel_id: ContractId | None = None
    duration_ticks: PositiveInt | None = None

    @model_validator(mode="after")
    def channel_scope_matches_fault_layer(self) -> FaultInjection:
        is_sensor_fault = self.fault_family in _SENSOR_FAULTS
        if is_sensor_fault != (self.channel_id is not None):
            raise ValueError(
                "sensor faults require channel_id; process faults must not set channel_id"
            )
        return self


class ScenarioAction(ContractModel):
    decision_tick: NonNegativeInt
    action: ActionLabel


class ScenarioDefinition(ContractModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    scenario_id: ContractId
    plant_variant_id: PlantVariant
    seed: SeedInt
    duration_ticks: PositiveInt
    driver: ScenarioDriver
    fault_injections: tuple[FaultInjection, ...] = ()
    action_sequence: tuple[ScenarioAction, ...] = ()

    @field_validator("fault_injections", mode="after")
    @classmethod
    def injections_are_a_canonical_set(
        cls, values: tuple[FaultInjection, ...]
    ) -> tuple[FaultInjection, ...]:
        keys = tuple(
            (
                item.onset_tick,
                _FAULT_ORDER[item.fault_family],
                item.component_id,
                item.channel_id or "",
            )
            for item in values
        )
        if len(keys) != len(set(keys)):
            raise ValueError("fault_injections must not contain duplicates")
        by_key = dict(zip(keys, values, strict=True))
        return tuple(by_key[key] for key in sorted(keys))

    @field_validator("action_sequence", mode="after")
    @classmethod
    def actions_are_strictly_ordered(
        cls, values: tuple[ScenarioAction, ...]
    ) -> tuple[ScenarioAction, ...]:
        ticks = tuple(item.decision_tick for item in values)
        if ticks != tuple(sorted(ticks)):
            raise ValueError("action_sequence must be ordered by decision_tick")
        if len(ticks) != len(set(ticks)):
            raise ValueError("only one scenario action is allowed per decision tick")
        return values

    @model_validator(mode="after")
    def scheduled_items_fit_the_trajectory(self) -> ScenarioDefinition:
        if any(injection.onset_tick >= self.duration_ticks for injection in self.fault_injections):
            raise ValueError("fault onset must be before duration_ticks")
        if any(
            injection.duration_ticks is not None
            and injection.onset_tick + injection.duration_ticks > self.duration_ticks
            for injection in self.fault_injections
        ):
            raise ValueError("fault duration must fit within the scenario window")
        if any(action.decision_tick >= self.duration_ticks for action in self.action_sequence):
            raise ValueError("action decision tick must be before duration_ticks")
        return self
