"""Target-independent counterfactual grouping from strict scenario structure."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from enum import StrEnum

from pydantic import field_validator, model_validator

from reactorbench.schemas.base import (
    SCHEMA_VERSION,
    ContractId,
    ContractModel,
    SchemaVersion,
    canonical_sha256,
    require_unique,
)
from reactorbench.schemas.enums import (
    ActionLabel,
    ComponentState,
    FaultFamily,
    StateVariable,
)
from reactorbench.schemas.scenario import FaultInjection, ScenarioDefinition
from reactorbench.simulator.variants import get_variant_spec


class GroupingError(ValueError):
    """Raised when a related scenario cannot be grouped without guessing."""


class CounterfactualFamily(StrEnum):
    G07_STANDBY = "G07_STANDBY"
    G08_G09_VALVE = "G08_G09_VALVE"
    G12_DEPENDENCY_MAP = "G12_DEPENDENCY_MAP"
    G14_COMPOSITION = "G14_COMPOSITION"
    G15_EVIDENCE_SUFFICIENCY = "G15_EVIDENCE_SUFFICIENCY"


class CounterfactualVariant(StrEnum):
    STANDBY_AVAILABLE = "standby_available"
    STANDBY_UNAVAILABLE = "standby_unavailable"
    VALVE_LAG_3 = "valve_lag_3"
    VALVE_LAG_4 = "valve_lag_4"
    VALVE_STUCK = "valve_stuck"
    MAP_INCLUDED = "map_included"
    MAP_WITHHELD = "map_withheld"
    PUMP_ONLY = "pump_only"
    SENSOR_ONLY = "sensor_only"
    COMPOUND = "compound"
    SPARSE = "sparse"


_EXPECTED_VARIANTS: dict[CounterfactualFamily, tuple[CounterfactualVariant, ...]] = {
    CounterfactualFamily.G07_STANDBY: (
        CounterfactualVariant.STANDBY_AVAILABLE,
        CounterfactualVariant.STANDBY_UNAVAILABLE,
    ),
    CounterfactualFamily.G08_G09_VALVE: (
        CounterfactualVariant.VALVE_LAG_3,
        CounterfactualVariant.VALVE_LAG_4,
        CounterfactualVariant.VALVE_STUCK,
    ),
    CounterfactualFamily.G12_DEPENDENCY_MAP: (
        CounterfactualVariant.MAP_INCLUDED,
        CounterfactualVariant.MAP_WITHHELD,
    ),
    CounterfactualFamily.G14_COMPOSITION: (
        CounterfactualVariant.PUMP_ONLY,
        CounterfactualVariant.SENSOR_ONLY,
        CounterfactualVariant.COMPOUND,
    ),
    # Phase 2 has only the sparse fixture. Expanded G15-A/G15-B relatives remain deferred.
    CounterfactualFamily.G15_EVIDENCE_SUFFICIENCY: (CounterfactualVariant.SPARSE,),
}


class GroupAssignment(ContractModel):
    """Audit-only family assignment derived without targets, text, or outcomes."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    scenario_id: ContractId
    counterfactual_group_id: ContractId
    family: CounterfactualFamily
    variant_id: CounterfactualVariant
    shared_factors_sha256: str
    affected_component_ids: tuple[ContractId, ...] = ()
    affected_channel_ids: tuple[ContractId, ...] = ()
    expected_variants: tuple[CounterfactualVariant, ...]
    expanded_siblings_supported: bool
    incomplete_reason: str | None = None

    @field_validator("affected_component_ids", "affected_channel_ids", mode="after")
    @classmethod
    def role_ids_are_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        require_unique(values, field_name="affected role IDs")
        return tuple(sorted(values))

    @model_validator(mode="after")
    def family_contract_is_exact(self) -> GroupAssignment:
        if len(self.shared_factors_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.shared_factors_sha256
        ):
            raise ValueError("shared_factors_sha256 must be a lowercase SHA-256")
        expected = _EXPECTED_VARIANTS[self.family]
        if self.expected_variants != expected:
            raise ValueError("expected_variants must match the preregistered family")
        if self.variant_id not in expected:
            raise ValueError("variant_id does not belong to the declared family")
        expected_prefix = self.family.value.lower().replace("_", "-")
        if not self.counterfactual_group_id.startswith(f"cf:{expected_prefix}:"):
            raise ValueError("counterfactual_group_id must use the strict family prefix")
        is_g15 = self.family is CounterfactualFamily.G15_EVIDENCE_SUFFICIENCY
        if is_g15:
            if self.expanded_siblings_supported or self.incomplete_reason is None:
                raise ValueError("G15 must remain explicitly incomplete until siblings exist")
        elif not self.expanded_siblings_supported or self.incomplete_reason is not None:
            raise ValueError("implemented counterfactual families must declare supported siblings")
        return self


class CounterfactualGroup(ContractModel):
    """One audited group, complete only when every implemented semantic role exists."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    counterfactual_group_id: ContractId
    family: CounterfactualFamily
    members: tuple[GroupAssignment, ...]
    is_complete: bool
    exclusion_reason: str | None = None

    @model_validator(mode="after")
    def members_are_consistent(self) -> CounterfactualGroup:
        if not self.members:
            raise ValueError("counterfactual group cannot be empty")
        if any(
            member.counterfactual_group_id != self.counterfactual_group_id
            for member in self.members
        ):
            raise ValueError("all members must share counterfactual_group_id")
        if any(member.family is not self.family for member in self.members):
            raise ValueError("all members must share counterfactual family")
        hashes = {member.shared_factors_sha256 for member in self.members}
        if len(hashes) != 1:
            raise ValueError("all members must share the same preregistered factors")
        scenario_ids = tuple(member.scenario_id for member in self.members)
        require_unique(scenario_ids, field_name="counterfactual scenario IDs")
        variants = tuple(member.variant_id for member in self.members)
        require_unique(variants, field_name="counterfactual variant roles")
        role_order = {role: index for index, role in enumerate(_EXPECTED_VARIANTS[self.family])}
        order = tuple(
            (role_order[member.variant_id], member.scenario_id) for member in self.members
        )
        if order != tuple(sorted(order)):
            raise ValueError("group members must use canonical variant/scenario order")
        observed = {member.variant_id for member in self.members}
        expected = set(_EXPECTED_VARIANTS[self.family])
        supported = all(member.expanded_siblings_supported for member in self.members)
        actually_complete = supported and observed == expected
        if self.is_complete != actually_complete:
            raise ValueError("is_complete must reflect exact role coverage and generator support")
        if self.is_complete != (self.exclusion_reason is None):
            raise ValueError(
                "incomplete groups require an exclusion reason, complete groups forbid one"
            )
        return self


def _single_injection(
    scenario: ScenarioDefinition, families: frozenset[FaultFamily]
) -> FaultInjection | None:
    if len(scenario.fault_injections) != 1:
        return None
    injection = scenario.fault_injections[0]
    return injection if injection.fault_family in families else None


def _assignment(
    *,
    scenario: ScenarioDefinition,
    family: CounterfactualFamily,
    variant: CounterfactualVariant,
    shared: Mapping[str, object],
    component_ids: tuple[str, ...] = (),
    channel_ids: tuple[str, ...] = (),
    expanded_siblings_supported: bool = True,
    incomplete_reason: str | None = None,
) -> GroupAssignment:
    shared_hash = canonical_sha256(shared)
    prefix = family.value.lower().replace("_", "-")
    return GroupAssignment(
        scenario_id=scenario.scenario_id,
        counterfactual_group_id=f"cf:{prefix}:{shared_hash[:24]}",
        family=family,
        variant_id=variant,
        shared_factors_sha256=shared_hash,
        affected_component_ids=component_ids,
        affected_channel_ids=channel_ids,
        expected_variants=_EXPECTED_VARIANTS[family],
        expanded_siblings_supported=expanded_siblings_supported,
        incomplete_reason=incomplete_reason,
    )


def _g07_assignment(scenario: ScenarioDefinition) -> GroupAssignment | None:
    injection = _single_injection(scenario, frozenset({FaultFamily.PUMP_TRIP}))
    context = scenario.standby_context
    if injection is None or context is None:
        return None
    if context.active_train_id != injection.component_id:
        raise GroupingError("G07 active component must match its trip injection")
    variant = (
        CounterfactualVariant.STANDBY_AVAILABLE
        if context.standby_state is ComponentState.AVAILABLE
        else CounterfactualVariant.STANDBY_UNAVAILABLE
    )
    shared = {
        "family": CounterfactualFamily.G07_STANDBY.value,
        "plant_variant": scenario.plant_variant_id.value,
        "seed": scenario.seed,
        "duration_ticks": scenario.duration_ticks,
        "driver": scenario.driver.value,
        "fault_family": injection.fault_family.value,
        "active_component_id": injection.component_id,
        "standby_component_id": context.standby_train_id,
        "support_component_id": context.standby_support_bus_id,
        "support_state": context.support_bus_state.value,
        "start_delay_ticks": context.standby_start_delay_ticks,
        "onset_tick": injection.onset_tick,
        "severity": injection.severity.value,
    }
    return _assignment(
        scenario=scenario,
        family=CounterfactualFamily.G07_STANDBY,
        variant=variant,
        shared=shared,
        component_ids=(context.active_train_id, context.standby_train_id),
    )


def _g08_g09_assignment(scenario: ScenarioDefinition) -> GroupAssignment | None:
    injection = _single_injection(
        scenario, frozenset({FaultFamily.VALVE_LAG, FaultFamily.VALVE_STUCK})
    )
    if injection is None:
        return None
    if injection.fault_family is FaultFamily.VALVE_LAG:
        if injection.duration_ticks == 3:
            variant = CounterfactualVariant.VALVE_LAG_3
        elif injection.duration_ticks == 4:
            variant = CounterfactualVariant.VALVE_LAG_4
        else:
            raise GroupingError("G08 lag duration must be exactly 3 or 4 ticks")
    else:
        variant = CounterfactualVariant.VALVE_STUCK
    # Fault family and duration are the decisive varied temporal factor and are excluded.
    shared = {
        "family": CounterfactualFamily.G08_G09_VALVE.value,
        "plant_variant": scenario.plant_variant_id.value,
        "seed": scenario.seed,
        "duration_ticks": scenario.duration_ticks,
        "driver": scenario.driver.value,
        "component_id": injection.component_id,
        "onset_tick": injection.onset_tick,
        "severity": injection.severity.value,
    }
    return _assignment(
        scenario=scenario,
        family=CounterfactualFamily.G08_G09_VALVE,
        variant=variant,
        shared=shared,
        component_ids=(injection.component_id,),
    )


def _g12_assignment(scenario: ScenarioDefinition) -> GroupAssignment | None:
    injection = _single_injection(scenario, frozenset({FaultFamily.SUPPORT_POWER_INTERRUPTION}))
    if injection is None:
        return None
    variant = (
        CounterfactualVariant.MAP_INCLUDED
        if scenario.dependency_map_context is not None
        else CounterfactualVariant.MAP_WITHHELD
    )
    # The map and its presence are the varied context, so neither enters the shared hash.
    shared = {
        "family": CounterfactualFamily.G12_DEPENDENCY_MAP.value,
        "plant_variant": scenario.plant_variant_id.value,
        "seed": scenario.seed,
        "duration_ticks": scenario.duration_ticks,
        "driver": scenario.driver.value,
        "fault_family": injection.fault_family.value,
        "component_id": injection.component_id,
        "onset_tick": injection.onset_tick,
        "severity": injection.severity.value,
    }
    return _assignment(
        scenario=scenario,
        family=CounterfactualFamily.G12_DEPENDENCY_MAP,
        variant=variant,
        shared=shared,
        component_ids=(injection.component_id,),
    )


def _g14_assignment(scenario: ScenarioDefinition) -> GroupAssignment | None:
    signature = tuple(injection.fault_family for injection in scenario.fault_injections)
    allowed = {
        (FaultFamily.PUMP_DEGRADATION,),
        (FaultFamily.SENSOR_DRIFT,),
        (FaultFamily.SENSOR_DRIFT, FaultFamily.PUMP_DEGRADATION),
    }
    if signature not in allowed:
        return None
    spec = get_variant_spec(scenario.plant_variant_id)
    thermal_channels = spec.channels_for(StateVariable.PRIMARY_THERMAL_STATE)
    default_pump = spec.primary_train_ids[scenario.seed % len(spec.primary_train_ids)]
    default_channel = thermal_channels[(scenario.seed // 2) % len(thermal_channels)].channel_id
    pump = next(
        (
            injection
            for injection in scenario.fault_injections
            if injection.fault_family is FaultFamily.PUMP_DEGRADATION
        ),
        None,
    )
    sensor = next(
        (
            injection
            for injection in scenario.fault_injections
            if injection.fault_family is FaultFamily.SENSOR_DRIFT
        ),
        None,
    )
    if sensor is not None and sensor.channel_id not in {
        channel.channel_id for channel in thermal_channels
    }:
        return None
    pump_id = pump.component_id if pump is not None else default_pump
    channel_id = sensor.channel_id if sensor is not None else default_channel
    if channel_id is None:
        raise GroupingError("G14 sensor role must resolve to a thermal channel")
    injections = tuple(item for item in (pump, sensor) if item is not None)
    onset_ticks = {injection.onset_tick for injection in injections}
    severities = {injection.severity.value for injection in injections}
    if len(onset_ticks) != 1 or len(severities) != 1:
        raise GroupingError("G14 factors must share one onset and severity")
    variant = {
        (FaultFamily.PUMP_DEGRADATION,): CounterfactualVariant.PUMP_ONLY,
        (FaultFamily.SENSOR_DRIFT,): CounterfactualVariant.SENSOR_ONLY,
        (
            FaultFamily.SENSOR_DRIFT,
            FaultFamily.PUMP_DEGRADATION,
        ): CounterfactualVariant.COMPOUND,
    }[signature]
    # Missing factor identities are deterministically completed from the same seed.
    shared = {
        "family": CounterfactualFamily.G14_COMPOSITION.value,
        "plant_variant": scenario.plant_variant_id.value,
        "seed": scenario.seed,
        "duration_ticks": scenario.duration_ticks,
        "driver": scenario.driver.value,
        "pump_component_id": pump_id,
        "sensor_channel_id": channel_id,
        "onset_tick": next(iter(onset_ticks)),
        "severity": next(iter(severities)),
    }
    return _assignment(
        scenario=scenario,
        family=CounterfactualFamily.G14_COMPOSITION,
        variant=variant,
        shared=shared,
        component_ids=(pump_id,),
        channel_ids=(channel_id,),
    )


def _g15_assignment(scenario: ScenarioDefinition) -> GroupAssignment | None:
    is_sparse_shape = (
        not scenario.fault_injections
        and scenario.standby_context is None
        and scenario.dependency_map_context is None
        and scenario.driver.value == "STEADY_OPERATION"
        and len(scenario.action_sequence) == 1
        and scenario.action_sequence[0].decision_tick == 2
        and scenario.action_sequence[0].action is ActionLabel.INSUFFICIENT_EVIDENCE
    )
    if not is_sparse_shape:
        return None
    spec = get_variant_spec(scenario.plant_variant_id)
    flow_channels = spec.channels_for(StateVariable.PRIMARY_FLOW)
    selected = flow_channels[scenario.seed % len(flow_channels)].channel_id
    shared = {
        "family": CounterfactualFamily.G15_EVIDENCE_SUFFICIENCY.value,
        "plant_variant": scenario.plant_variant_id.value,
        "seed": scenario.seed,
        "duration_ticks": scenario.duration_ticks,
        "driver": scenario.driver.value,
        "selected_channel_id": selected,
        "decision_tick": 2,
    }
    return _assignment(
        scenario=scenario,
        family=CounterfactualFamily.G15_EVIDENCE_SUFFICIENCY,
        variant=CounterfactualVariant.SPARSE,
        shared=shared,
        channel_ids=(selected,),
        expanded_siblings_supported=False,
        incomplete_reason="G15-A/G15-B evidence-expanded siblings are not generator-supported",
    )


def derive_group_assignment(scenario: ScenarioDefinition) -> GroupAssignment | None:
    """Return strict family metadata or ``None`` for a non-counterfactual scenario."""

    if type(scenario) is not ScenarioDefinition:
        raise TypeError("scenario must be a ScenarioDefinition")
    assignments = tuple(
        assignment
        for assignment in (
            _g07_assignment(scenario),
            _g08_g09_assignment(scenario),
            _g12_assignment(scenario),
            _g14_assignment(scenario),
            _g15_assignment(scenario),
        )
        if assignment is not None
    )
    if len(assignments) > 1:
        raise GroupingError("scenario shape matched more than one counterfactual family")
    return assignments[0] if assignments else None


def group_scenarios(scenarios: Iterable[ScenarioDefinition]) -> tuple[CounterfactualGroup, ...]:
    """Build deterministic family groups and report incomplete coverage explicitly."""

    assignments = tuple(
        assignment
        for scenario in scenarios
        if (assignment := derive_group_assignment(scenario)) is not None
    )
    scenario_ids = tuple(assignment.scenario_id for assignment in assignments)
    require_unique(scenario_ids, field_name="grouped scenario IDs")
    by_group: dict[str, list[GroupAssignment]] = defaultdict(list)
    for assignment in assignments:
        by_group[assignment.counterfactual_group_id].append(assignment)
    groups: list[CounterfactualGroup] = []
    for group_id in sorted(by_group):
        members = by_group[group_id]
        family = members[0].family
        role_order = {role: index for index, role in enumerate(_EXPECTED_VARIANTS[family])}
        ordered = tuple(
            sorted(members, key=lambda item: (role_order[item.variant_id], item.scenario_id))
        )
        observed = {member.variant_id for member in ordered}
        supported = all(member.expanded_siblings_supported for member in ordered)
        complete = supported and observed == set(_EXPECTED_VARIANTS[family])
        reason = None if complete else "missing or unsupported counterfactual variant roles"
        groups.append(
            CounterfactualGroup(
                counterfactual_group_id=group_id,
                family=family,
                members=ordered,
                is_complete=complete,
                exclusion_reason=reason,
            )
        )
    return tuple(groups)


def require_complete_group(group: CounterfactualGroup) -> CounterfactualGroup:
    """Fail closed before comparison construction or counterfactual-test assignment."""

    if type(group) is not CounterfactualGroup:
        raise TypeError("group must be a CounterfactualGroup")
    if not group.is_complete:
        raise GroupingError(
            "counterfactual operation requires a complete generator-supported group"
        )
    return group
