"""Scenario-definition and scheduled-action contracts."""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

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
    ComponentState,
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


class StandbyContext(ContractModel):
    """Bounded dependency context for a fictional standby train."""

    context_id: ContractId
    active_train_id: ContractId
    standby_train_id: ContractId
    standby_state: ComponentState
    standby_support_bus_id: ContractId
    support_bus_state: ComponentState
    standby_start_delay_ticks: PositiveInt

    @model_validator(mode="after")
    def train_ids_and_states_are_bounded(self) -> StandbyContext:
        if self.active_train_id == self.standby_train_id:
            raise ValueError("active_train_id and standby_train_id must be different")

        permitted_states = {ComponentState.AVAILABLE, ComponentState.UNAVAILABLE}
        if self.standby_state not in permitted_states:
            raise ValueError("standby_state must be AVAILABLE or UNAVAILABLE")
        if self.support_bus_state not in permitted_states:
            raise ValueError("support_bus_state must be AVAILABLE or UNAVAILABLE")
        return self


class DependencyLink(ContractModel):
    """One fictional support-bus dependency edge exposed to a scenario."""

    support_bus_id: ContractId
    dependent_component_id: ContractId

    @model_validator(mode="after")
    def endpoints_are_distinct(self) -> DependencyLink:
        if self.support_bus_id == self.dependent_component_id:
            raise ValueError("support_bus_id and dependent_component_id must be different")
        return self


class DependencyMapContext(ContractModel):
    """Canonical, bounded dependency map for a fictional plant variant.

    This contract deliberately validates only the structure of the map.  The
    simulator owns validation that the supplied links match a variant's reviewed
    component registry exactly.
    """

    plant_variant_id: PlantVariant
    links: tuple[DependencyLink, ...] = Field(min_length=1)

    @field_validator("links", mode="after")
    @classmethod
    def links_are_a_canonical_function(
        cls, values: tuple[DependencyLink, ...]
    ) -> tuple[DependencyLink, ...]:
        pairs = tuple((link.support_bus_id, link.dependent_component_id) for link in values)
        if len(pairs) != len(set(pairs)):
            raise ValueError("links must not contain duplicate support-bus/dependent pairs")

        dependents = tuple(link.dependent_component_id for link in values)
        if len(dependents) != len(set(dependents)):
            raise ValueError("each dependent_component_id must have exactly one support_bus_id")

        canonical_pairs = tuple(sorted(pairs))
        if pairs != canonical_pairs:
            raise ValueError(
                "links must use canonical (support_bus_id, dependent_component_id) order"
            )
        return values


class ScenarioDefinition(ContractModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    scenario_id: ContractId
    plant_variant_id: PlantVariant
    seed: SeedInt
    duration_ticks: PositiveInt
    driver: ScenarioDriver
    fault_injections: tuple[FaultInjection, ...] = ()
    action_sequence: tuple[ScenarioAction, ...] = ()
    standby_context: StandbyContext | None = None
    dependency_map_context: DependencyMapContext | None = None

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
        if (
            self.dependency_map_context is not None
            and self.dependency_map_context.plant_variant_id is not self.plant_variant_id
        ):
            raise ValueError("dependency_map_context.plant_variant_id must match plant_variant_id")
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
