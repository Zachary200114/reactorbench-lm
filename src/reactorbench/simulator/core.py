"""Deterministic first-fault Aster-A structured simulator."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import TypedDict

from reactorbench.schemas import (
    SCHEMA_VERSION,
    AbstentionReason,
    ActionLabel,
    CanonicalEvent,
    ChannelQuality,
    ComponentLatentState,
    ComponentState,
    DecisionTarget,
    DependencyLink,
    DependencyMapContext,
    DiagnosisStatus,
    EventType,
    EvidenceSlot,
    FaultFamily,
    FaultInjection,
    LatentPlantState,
    ObservationFrame,
    ObservationStatus,
    OperatingMode,
    PlantValues,
    PlantVariant,
    ProvenanceRecord,
    ScenarioAction,
    ScenarioDefinition,
    ScenarioDriver,
    ScenarioTargets,
    SensorChannelObservation,
    SeverityBand,
    StandbyContext,
    StateVariable,
    StructuredTrajectory,
)

from .content_guard import assert_no_prohibited_content
from .variants import (
    ASTER_A_SPEC,
    AsterVariantSpec,
    ChannelRole,
    ComponentRole,
    get_variant_spec,
)
from .variants import (
    GENERATOR_VERSION as _VARIANT_GENERATOR_VERSION,
)

GENERATOR_VERSION = "0.1.0"
if GENERATOR_VERSION != _VARIANT_GENERATOR_VERSION:
    raise RuntimeError("simulator and variant registry generator versions must match")
_MIN_DURATION = 8
_MAX_DURATION = 64
_MAX_SEED = 4_294_967_295
_NOISE_BOUND = 0.006
_MAX_TICK_STEP = 0.03
_LOAD_ONSET_TICK = 2
_LOAD_RAMP_TICKS = 3
_LOAD_STEAM_TICK = _LOAD_ONSET_TICK + 1
_LOAD_OUTPUT_TICK = _LOAD_STEAM_TICK + 1
_LOAD_SETTLE_TICK = _LOAD_OUTPUT_TICK + _LOAD_RAMP_TICKS
_STUCK_VERIFY_TICK = _LOAD_OUTPUT_TICK + 1
_STUCK_FLAG_TICK = _STUCK_VERIFY_TICK + 1
_STUCK_FLAG_APPLY_TICK = _STUCK_FLAG_TICK + 1
_SENSOR_NOISE_ONSET_TICK = 2
_SENSOR_NOISE_FIRST_DECISION_TICK = 3
_SENSOR_NOISE_SECOND_DECISION_TICK = 4
_SENSOR_NOISE_DIAGNOSIS_TICK = 5
_SENSOR_NOISE_FLAG_TICK = 6
_SENSOR_NOISE_FLAG_APPLY_TICK = 7
_PUMP_DEGRADATION_ONSET_TICK = 2
_PUMP_FLOW_TICK = 3
_PUMP_THERMAL_TICK = 4
_PUMP_STEAM_TICK = 5
_PUMP_OUTPUT_TICK = 6
_PUMP_INSPECTION_DECISION_TICK = 6
_PUMP_INSPECTION_APPLY_TICK = 7
_PUMP_LOAD_DECISION_TICK = 7
_PUMP_LOAD_APPLY_TICK = 8
_PUMP_MIN_DURATION = 9
_PUMP_MODE_SUBJECT = "aster-operating-domain"
_PUMP_LOAD_SUBJECT = "aster-load-domain"
_PUMP_TRIP_ONSET_TICK = 2
_PUMP_TRIP_FLOW_TICK = 3
_PUMP_TRIP_THERMAL_TICK = 4
_PUMP_TRIP_STEAM_TICK = 5
_PUMP_TRIP_DECISION_TICK = 5
_PUMP_TRIP_ACTION_TICK = 6
_PUMP_TRIP_RECOVERY_TICK = 7
_PUMP_TRIP_MIN_DURATION = 8
_VALVE_ONSET_TICK = 2
_VALVE_EARLY_DECISION_TICK = 3
_VALVE_EARLY_ACTION_TICK = 4
_VALVE_DECISIVE_TICK = 6
_VALVE_ACTION_TICK = 7
_VALVE_LAG_DURATION_TICKS = 4
_VALVE_MIN_DURATION = 8
_TRANSFER_ONSET_TICK = 2
_TRANSFER_THERMAL_TICK = 3
_TRANSFER_STEAM_TICK = 4
_TRANSFER_OUTPUT_TICK = 5
_TRANSFER_LOAD_DECISION_TICK = 5
_TRANSFER_LOAD_APPLY_TICK = 6
_TRANSFER_MIN_DURATION = 8
_FLOW_IMBALANCE_ONSET_TICK = 2
_FLOW_IMBALANCE_INVENTORY_TICK = 3
_FLOW_IMBALANCE_COMPARE_TICK = 4
_FLOW_IMBALANCE_STEAM_TICK = 5
_FLOW_IMBALANCE_OUTPUT_TICK = 6
_FLOW_IMBALANCE_STABILIZE_APPLY_TICK = 7
_FLOW_IMBALANCE_MIN_DURATION = 8
_SUPPORT_POWER_ONSET_TICK = 2
_SUPPORT_POWER_COMPONENT_TICK = 3
_SUPPORT_POWER_EFFECT_TICK = 4
_SUPPORT_POWER_DELAYED_TICK = 5
_SUPPORT_POWER_DECISION_TICK = 5
_SUPPORT_POWER_ACTION_TICK = 6
_SUPPORT_POWER_STABILIZED_TICK = 7
_SUPPORT_POWER_MIN_DURATION = 8
_LOAD_RESPONSE_STAGES: dict[StateVariable, tuple[float, int]] = {
    StateVariable.LOAD_DEMAND: (1.0, _LOAD_ONSET_TICK),
    StateVariable.HEAT_SOURCE_LEVEL: (0.8, _LOAD_ONSET_TICK),
    StateVariable.PRIMARY_FLOW: (0.55, _LOAD_ONSET_TICK),
    StateVariable.STEAM_STATE: (0.75, _LOAD_STEAM_TICK),
    StateVariable.TURBINE_OUTPUT: (0.75, _LOAD_OUTPUT_TICK),
    StateVariable.ELECTRICAL_OUTPUT: (0.75, _LOAD_OUTPUT_TICK),
}


class UnsupportedScenarioError(ValueError):
    """Raised when a valid generic schema record is outside this milestone."""


class _ComponentPositionFields(TypedDict):
    commanded_position: float | None
    actual_position: float | None


_INSTRUMENTATION = ASTER_A_SPEC.instrumentation_id
_PRIMARY_TRAINS = ASTER_A_SPEC.primary_train_ids
_SUPPORT_BUSES = ASTER_A_SPEC.support_bus_ids


@dataclass(frozen=True)
class SimulationTrace:
    """One fully separated audit trace generated from a supported scenario."""

    scenario: ScenarioDefinition
    latent_states: tuple[LatentPlantState, ...]
    observations: tuple[ObservationFrame, ...]
    events: tuple[CanonicalEvent, ...]
    targets: ScenarioTargets

    def visible_payload(self) -> dict[str, object]:
        """Return only model-visible observations and canonical events."""

        return {
            "schema_version": SCHEMA_VERSION,
            "plant_variant_id": self.scenario.plant_variant_id.value,
            "dependency_map_context": (
                self.scenario.dependency_map_context.model_dump(mode="json")
                if self.scenario.dependency_map_context is not None
                else None
            ),
            "standby_context": (
                self.scenario.standby_context.model_dump(mode="json")
                if self.scenario.standby_context is not None
                else None
            ),
            "observations": [frame.model_dump(mode="json") for frame in self.observations],
            "events": [event.model_dump(mode="json") for event in self.events],
        }

    def to_structured_trajectory(
        self, *, trajectory_id: str, provenance: ProvenanceRecord
    ) -> StructuredTrajectory:
        return StructuredTrajectory(
            trajectory_id=trajectory_id,
            scenario_id=self.scenario.scenario_id,
            scenario=self.scenario,
            provenance=provenance,
            latent_states=self.latent_states,
            observations=self.observations,
            events=self.events,
            targets=self.targets,
        )


def dependency_map_context_for(spec: AsterVariantSpec) -> DependencyMapContext:
    """Derive the sole canonical visible dependency map from a reviewed card."""

    if type(spec) is not AsterVariantSpec:
        raise TypeError("spec must be an AsterVariantSpec")
    return DependencyMapContext(
        plant_variant_id=spec.plant_variant,
        links=tuple(
            DependencyLink(
                support_bus_id=link.supplier_component_id,
                dependent_component_id=link.dependent_component_id,
            )
            for link in sorted(
                spec.dependency_links,
                key=lambda link: (link.supplier_component_id, link.dependent_component_id),
            )
        ),
    )


def _spec_for(scenario: ScenarioDefinition) -> AsterVariantSpec:
    """Resolve only an exact registered variant selected by a canonical scenario."""

    if type(scenario.plant_variant_id) is not PlantVariant:
        raise UnsupportedScenarioError("plant variant must use the canonical enum")
    try:
        return get_variant_spec(scenario.plant_variant_id)
    except (KeyError, TypeError) as error:
        raise UnsupportedScenarioError("plant variant is not registered") from error


def _injections_for(
    scenario: ScenarioDefinition, fault_family: FaultFamily
) -> tuple[FaultInjection, ...]:
    """Return only exact canonical injections for one family without index assumptions."""

    if type(fault_family) is not FaultFamily:
        raise TypeError("fault_family must be a FaultFamily")
    return tuple(
        injection
        for injection in scenario.fault_injections
        if type(injection) is FaultInjection and injection.fault_family is fault_family
    )


def _fault_signature(scenario: ScenarioDefinition) -> tuple[FaultFamily, ...]:
    """Return the canonical ordered family signature for strict dispatch."""

    return tuple(injection.fault_family for injection in scenario.fault_injections)


def _single_injection_for(
    scenario: ScenarioDefinition, fault_family: FaultFamily
) -> FaultInjection | None:
    injections = _injections_for(scenario, fault_family)
    if _fault_signature(scenario) == (fault_family,) and len(injections) == 1:
        return injections[0]
    return None


def _require_uint32(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_SEED:
        raise ValueError(f"{name} must be a uint32 integer")


def _require_duration(duration_ticks: int) -> None:
    _require_uint32(duration_ticks, name="duration_ticks")
    if not _MIN_DURATION <= duration_ticks <= _MAX_DURATION:
        raise ValueError(f"duration_ticks must be between {_MIN_DURATION} and {_MAX_DURATION}")


def _drift_channels() -> tuple[str, ...]:
    return tuple(
        channel.channel_id
        for channel in ASTER_A_SPEC.channels
        if channel.variable is StateVariable.PRIMARY_FLOW
    )


def _stuck_channels() -> tuple[str, ...]:
    return tuple(
        channel.channel_id
        for channel in ASTER_A_SPEC.channels
        if channel.variable is StateVariable.ELECTRICAL_OUTPUT
    )


def _sensor_noise_channels() -> tuple[str, ...]:
    return tuple(
        channel.channel_id
        for channel in ASTER_A_SPEC.channels
        if channel.variable is StateVariable.PRIMARY_THERMAL_STATE
    )


def build_stable_scenario(
    *, seed: int, duration_ticks: int = 12, plant_variant: PlantVariant = PlantVariant.ASTER_A
) -> ScenarioDefinition:
    """Build one deterministic no-fault scenario from a reviewed Aster variant card."""

    _require_uint32(seed, name="seed")
    _require_duration(duration_ticks)
    if type(plant_variant) is not PlantVariant:
        raise ValueError("plant_variant must be a PlantVariant")
    spec = get_variant_spec(plant_variant)
    scenario = ScenarioDefinition(
        scenario_id=f"{spec.plant_variant.value.lower()}-stable-{seed}-{duration_ticks}",
        plant_variant_id=spec.plant_variant,
        seed=seed,
        duration_ticks=duration_ticks,
        driver=ScenarioDriver.STEADY_OPERATION,
        action_sequence=(
            ScenarioAction(
                decision_tick=duration_ticks - 1,
                action=ActionLabel.CONTINUE_MONITORING,
            ),
        ),
    )
    assert_no_prohibited_content(scenario)
    return scenario


def build_load_transient_scenario(*, seed: int, duration_ticks: int = 12) -> ScenarioDefinition:
    """Build the canonical benign load transition derived only from seed and duration.

    The change starts at a fixed early tick.  Its sign and bounded magnitude are
    deterministic functions of the seed, so no unrepresented scenario parameter
    is needed to reproduce or reverse the fictional transition.
    """

    _require_uint32(seed, name="seed")
    _require_duration(duration_ticks)
    if _LOAD_SETTLE_TICK >= duration_ticks:
        raise ValueError("load transient needs a completed bounded response")
    scenario = ScenarioDefinition(
        scenario_id=f"aster-a-load-{seed}-{duration_ticks}",
        plant_variant_id=PlantVariant.ASTER_A,
        seed=seed,
        duration_ticks=duration_ticks,
        driver=ScenarioDriver.LOAD_TRANSIENT,
        action_sequence=(
            ScenarioAction(
                decision_tick=duration_ticks - 1,
                action=ActionLabel.CONTINUE_MONITORING,
            ),
        ),
    )
    assert_no_prohibited_content(scenario)
    return scenario


def build_pump_degradation_scenario(
    *, seed: int, duration_ticks: int = 12, component_id: str | None = None
) -> ScenarioDefinition:
    """Build the canonical gradual primary-train degradation case."""

    _require_uint32(seed, name="seed")
    _require_duration(duration_ticks)
    if duration_ticks < _PUMP_MIN_DURATION:
        raise ValueError("pump degradation needs at least 9 ticks")
    if component_id is None:
        selected_component = ASTER_A_SPEC.primary_train_ids[
            seed % len(ASTER_A_SPEC.primary_train_ids)
        ]
    elif type(component_id) is str and component_id in ASTER_A_SPEC.primary_train_ids:
        selected_component = component_id
    else:
        raise ValueError("component_id must be an Aster-A primary-train id or None")
    scenario = ScenarioDefinition(
        scenario_id=(
            f"aster-a-pump-degradation-{seed}-{duration_ticks}-"
            f"{_PUMP_DEGRADATION_ONSET_TICK}-low-{selected_component}"
        ),
        plant_variant_id=PlantVariant.ASTER_A,
        seed=seed,
        duration_ticks=duration_ticks,
        driver=ScenarioDriver.STEADY_OPERATION,
        fault_injections=(
            FaultInjection(
                fault_family=FaultFamily.PUMP_DEGRADATION,
                component_id=selected_component,
                onset_tick=_PUMP_DEGRADATION_ONSET_TICK,
                severity=SeverityBand.LOW,
            ),
        ),
        action_sequence=(
            ScenarioAction(
                decision_tick=_PUMP_THERMAL_TICK,
                action=ActionLabel.INSUFFICIENT_EVIDENCE,
            ),
            ScenarioAction(
                decision_tick=_PUMP_INSPECTION_DECISION_TICK,
                action=ActionLabel.REQUEST_COMPONENT_INSPECTION,
            ),
            ScenarioAction(
                decision_tick=_PUMP_LOAD_DECISION_TICK,
                action=ActionLabel.REDUCE_SIMULATED_LOAD,
            ),
        ),
    )
    assert_no_prohibited_content(scenario)
    return scenario


def _process_scenario_id(
    *,
    spec: AsterVariantSpec,
    family: FaultFamily,
    seed: int,
    duration_ticks: int,
    component_id: str,
) -> str:
    return (
        f"{spec.plant_variant.value.lower()}-{family.value.lower().replace('_', '-')}-"
        f"{seed}-{duration_ticks}-{_TRANSFER_ONSET_TICK}-low-{component_id}"
    )


def build_transfer_efficiency_loss_scenario(
    *,
    seed: int,
    duration_ticks: int = 12,
    plant_variant: PlantVariant = PlantVariant.ASTER_A,
) -> ScenarioDefinition:
    """Build G10's normalized, gradual transfer-efficiency-loss scenario."""

    _require_uint32(seed, name="seed")
    _require_duration(duration_ticks)
    if type(plant_variant) is not PlantVariant:
        raise ValueError("plant_variant must be a PlantVariant")
    if duration_ticks < _TRANSFER_MIN_DURATION:
        raise ValueError("transfer-efficiency loss needs at least 8 ticks")
    spec = get_variant_spec(plant_variant)
    component_id = spec.transfer_unit_id
    scenario = ScenarioDefinition(
        scenario_id=_process_scenario_id(
            spec=spec,
            family=FaultFamily.TRANSFER_EFFICIENCY_LOSS,
            seed=seed,
            duration_ticks=duration_ticks,
            component_id=component_id,
        ),
        plant_variant_id=spec.plant_variant,
        seed=seed,
        duration_ticks=duration_ticks,
        driver=ScenarioDriver.STEADY_OPERATION,
        fault_injections=(
            FaultInjection(
                fault_family=FaultFamily.TRANSFER_EFFICIENCY_LOSS,
                component_id=component_id,
                onset_tick=_TRANSFER_ONSET_TICK,
                severity=SeverityBand.LOW,
            ),
        ),
        action_sequence=(
            ScenarioAction(
                decision_tick=_TRANSFER_THERMAL_TICK,
                action=ActionLabel.INSUFFICIENT_EVIDENCE,
            ),
            ScenarioAction(
                decision_tick=_TRANSFER_LOAD_DECISION_TICK,
                action=ActionLabel.REDUCE_SIMULATED_LOAD,
            ),
        ),
    )
    assert_no_prohibited_content(scenario)
    return scenario


def build_flow_imbalance_scenario(
    *,
    seed: int,
    duration_ticks: int = 12,
    plant_variant: PlantVariant = PlantVariant.ASTER_A,
) -> ScenarioDefinition:
    """Build G11's normalized secondary-flow-imbalance scenario."""

    _require_uint32(seed, name="seed")
    _require_duration(duration_ticks)
    if type(plant_variant) is not PlantVariant:
        raise ValueError("plant_variant must be a PlantVariant")
    if duration_ticks < _FLOW_IMBALANCE_MIN_DURATION:
        raise ValueError("flow imbalance needs at least 8 ticks")
    spec = get_variant_spec(plant_variant)
    component_id = spec.secondary_feed_id
    scenario = ScenarioDefinition(
        scenario_id=_process_scenario_id(
            spec=spec,
            family=FaultFamily.FLOW_IMBALANCE,
            seed=seed,
            duration_ticks=duration_ticks,
            component_id=component_id,
        ),
        plant_variant_id=spec.plant_variant,
        seed=seed,
        duration_ticks=duration_ticks,
        driver=ScenarioDriver.STEADY_OPERATION,
        fault_injections=(
            FaultInjection(
                fault_family=FaultFamily.FLOW_IMBALANCE,
                component_id=component_id,
                onset_tick=_FLOW_IMBALANCE_ONSET_TICK,
                severity=SeverityBand.LOW,
            ),
        ),
        action_sequence=(
            ScenarioAction(
                decision_tick=_FLOW_IMBALANCE_COMPARE_TICK,
                action=ActionLabel.COMPARE_RELATED_TRENDS,
            ),
            ScenarioAction(
                decision_tick=_FLOW_IMBALANCE_OUTPUT_TICK,
                action=ActionLabel.ENTER_SIMULATED_STABLE_STATE,
            ),
        ),
    )
    assert_no_prohibited_content(scenario)
    return scenario


def _support_power_scenario_id(
    *,
    spec: AsterVariantSpec,
    seed: int,
    duration_ticks: int,
    bus_id: str,
    include_dependency_map: bool,
) -> str:
    context_suffix = "included" if include_dependency_map else "withheld"
    return (
        f"{spec.plant_variant.value.lower()}-support-power-interruption-{seed}-"
        f"{duration_ticks}-{_SUPPORT_POWER_ONSET_TICK}-low-{bus_id}-map-{context_suffix}"
    )


def build_support_power_interruption_scenario(
    *,
    seed: int,
    duration_ticks: int = 12,
    plant_variant: PlantVariant = PlantVariant.ASTER_B,
    include_dependency_map: bool = True,
) -> ScenarioDefinition:
    """Build G12's bounded support-bus interruption fixture.

    The dependency map is selected only from the immutable variant registry.
    ``include_dependency_map`` changes the model-visible context and target
    branch, never the latent physics or observations before the decision tick.
    """

    _require_uint32(seed, name="seed")
    _require_duration(duration_ticks)
    if type(plant_variant) is not PlantVariant:
        raise ValueError("plant_variant must be a PlantVariant")
    if plant_variant not in {PlantVariant.ASTER_A, PlantVariant.ASTER_B}:
        raise ValueError("support-power interruption supports only Aster-A and Aster-B")
    if type(include_dependency_map) is not bool:
        raise ValueError("include_dependency_map must be a bool")
    if duration_ticks < _SUPPORT_POWER_MIN_DURATION:
        raise ValueError("support-power interruption needs at least 8 ticks")

    spec = get_variant_spec(plant_variant)
    bus_id = spec.component_for_role(ComponentRole.SUPPORT_BUS_TWO).component_id
    scenario = ScenarioDefinition(
        scenario_id=_support_power_scenario_id(
            spec=spec,
            seed=seed,
            duration_ticks=duration_ticks,
            bus_id=bus_id,
            include_dependency_map=include_dependency_map,
        ),
        plant_variant_id=spec.plant_variant,
        seed=seed,
        duration_ticks=duration_ticks,
        driver=ScenarioDriver.STEADY_OPERATION,
        fault_injections=(
            FaultInjection(
                fault_family=FaultFamily.SUPPORT_POWER_INTERRUPTION,
                component_id=bus_id,
                onset_tick=_SUPPORT_POWER_ONSET_TICK,
                severity=SeverityBand.LOW,
            ),
        ),
        action_sequence=(
            ScenarioAction(
                decision_tick=_SUPPORT_POWER_DECISION_TICK,
                action=(
                    ActionLabel.ENTER_SIMULATED_STABLE_STATE
                    if include_dependency_map
                    else ActionLabel.INSUFFICIENT_EVIDENCE
                ),
            ),
        ),
        dependency_map_context=(
            dependency_map_context_for(spec) if include_dependency_map else None
        ),
    )
    assert_no_prohibited_content(scenario)
    return scenario


def _support_bus_for_train(train_id: str) -> str:
    dependency_map = dict(ASTER_A_SPEC.primary_train_support_bus_pairs)
    try:
        return dependency_map[train_id]
    except KeyError as error:
        raise ValueError("train_id must have one Aster-A support-bus dependency") from error


def _trip_standby_context(*, active_train_id: str, standby_state: ComponentState) -> StandbyContext:
    standby_train_id = next(
        train_id for train_id in ASTER_A_SPEC.primary_train_ids if train_id != active_train_id
    )
    support_bus_id = _support_bus_for_train(standby_train_id)
    return StandbyContext(
        context_id=(
            f"aster-a-standby-{active_train_id}-{standby_train_id}-{standby_state.value.lower()}"
        ),
        active_train_id=active_train_id,
        standby_train_id=standby_train_id,
        standby_state=standby_state,
        standby_support_bus_id=support_bus_id,
        support_bus_state=ComponentState.AVAILABLE,
        standby_start_delay_ticks=ASTER_A_SPEC.standby_start_delay_ticks,
    )


def build_pump_trip_scenario(
    *,
    seed: int,
    duration_ticks: int = 12,
    component_id: str | None = None,
    standby_state: ComponentState = ComponentState.AVAILABLE,
) -> ScenarioDefinition:
    """Build one matched fictional pump-trip case with bounded standby context."""

    _require_uint32(seed, name="seed")
    _require_duration(duration_ticks)
    if duration_ticks < _PUMP_TRIP_MIN_DURATION:
        raise ValueError("pump trip needs at least 8 ticks")
    if component_id is None:
        active_train_id = ASTER_A_SPEC.primary_train_ids[seed % len(ASTER_A_SPEC.primary_train_ids)]
    elif type(component_id) is str and component_id in ASTER_A_SPEC.primary_train_ids:
        active_train_id = component_id
    else:
        raise ValueError("component_id must be an Aster-A primary-train id or None")
    if type(standby_state) is not ComponentState or standby_state not in {
        ComponentState.AVAILABLE,
        ComponentState.UNAVAILABLE,
    }:
        raise ValueError("standby_state must be AVAILABLE or UNAVAILABLE")

    context = _trip_standby_context(active_train_id=active_train_id, standby_state=standby_state)
    actions = (
        (
            ScenarioAction(
                decision_tick=_PUMP_TRIP_DECISION_TICK,
                action=ActionLabel.SELECT_SYNTHETIC_STANDBY_TRAIN,
            ),
        )
        if standby_state is ComponentState.AVAILABLE
        else (
            ScenarioAction(
                decision_tick=_PUMP_TRIP_DECISION_TICK,
                action=ActionLabel.REDUCE_SIMULATED_LOAD,
            ),
            ScenarioAction(
                decision_tick=_PUMP_TRIP_ACTION_TICK,
                action=ActionLabel.ENTER_SIMULATED_STABLE_STATE,
            ),
        )
    )
    scenario = ScenarioDefinition(
        scenario_id=(
            f"aster-a-pump-trip-{seed}-{duration_ticks}-{_PUMP_TRIP_ONSET_TICK}-low-"
            f"{active_train_id}-{standby_state.value.lower()}"
        ),
        plant_variant_id=PlantVariant.ASTER_A,
        seed=seed,
        duration_ticks=duration_ticks,
        driver=ScenarioDriver.STEADY_OPERATION,
        fault_injections=(
            FaultInjection(
                fault_family=FaultFamily.PUMP_TRIP,
                component_id=active_train_id,
                onset_tick=_PUMP_TRIP_ONSET_TICK,
                severity=SeverityBand.LOW,
            ),
        ),
        action_sequence=actions,
        standby_context=context,
    )
    assert_no_prohibited_content(scenario)
    return scenario


def _build_valve_scenario(
    *,
    seed: int,
    fault_family: FaultFamily,
    duration_ticks: int,
    component_id: str | None,
    lag_ticks: int | None,
) -> ScenarioDefinition:
    """Build one fixed-duration member of the G08/G09 temporal contrast pair."""

    _require_uint32(seed, name="seed")
    _require_duration(duration_ticks)
    if duration_ticks < _VALVE_MIN_DURATION:
        raise ValueError("valve scenarios need decision and action application history")
    if fault_family not in {FaultFamily.VALVE_LAG, FaultFamily.VALVE_STUCK}:
        raise ValueError("fault_family must be VALVE_LAG or VALVE_STUCK")
    if fault_family is FaultFamily.VALVE_LAG:
        if type(lag_ticks) is not int or lag_ticks not in {3, 4}:
            raise ValueError("lag_ticks must be an integer in the declared {3, 4} lag band")
        injection_duration = lag_ticks
    elif lag_ticks is not None:
        raise ValueError("lag_ticks is only supported for VALVE_LAG")
    else:
        injection_duration = None
    if component_id is None:
        selected_component = ASTER_A_SPEC.primary_flow_valve_ids[
            seed % len(ASTER_A_SPEC.primary_flow_valve_ids)
        ]
    elif type(component_id) is str and component_id in ASTER_A_SPEC.primary_flow_valve_ids:
        selected_component = component_id
    else:
        raise ValueError("component_id must be an Aster-A primary-flow valve id or None")

    decisive_tick = (
        _VALVE_ONSET_TICK + injection_duration if injection_duration else _VALVE_DECISIVE_TICK
    )
    scenario = ScenarioDefinition(
        scenario_id=(
            f"aster-a-{fault_family.value.lower().replace('_', '-')}-{seed}-{duration_ticks}-"
            f"{_VALVE_ONSET_TICK}-low-{selected_component}"
            f"-lag-{injection_duration}"
            if injection_duration is not None
            else f"aster-a-{fault_family.value.lower().replace('_', '-')}-{seed}-{duration_ticks}-"
            f"{_VALVE_ONSET_TICK}-low-{selected_component}"
        ),
        plant_variant_id=PlantVariant.ASTER_A,
        seed=seed,
        duration_ticks=duration_ticks,
        driver=ScenarioDriver.STEADY_OPERATION,
        fault_injections=(
            FaultInjection(
                fault_family=fault_family,
                component_id=selected_component,
                onset_tick=_VALVE_ONSET_TICK,
                severity=SeverityBand.LOW,
                duration_ticks=injection_duration,
            ),
        ),
        action_sequence=(
            ScenarioAction(
                decision_tick=_VALVE_EARLY_DECISION_TICK,
                action=ActionLabel.INSUFFICIENT_EVIDENCE,
            ),
            ScenarioAction(
                decision_tick=decisive_tick,
                action=(
                    ActionLabel.CONTINUE_MONITORING
                    if fault_family is FaultFamily.VALVE_LAG
                    else ActionLabel.REQUEST_COMPONENT_INSPECTION
                ),
            ),
        ),
    )
    assert_no_prohibited_content(scenario)
    return scenario


def build_valve_lag_scenario(
    *,
    seed: int,
    duration_ticks: int = 12,
    component_id: str | None = None,
    lag_ticks: int = _VALVE_LAG_DURATION_TICKS,
) -> ScenarioDefinition:
    """Build G08: a fictional command lag that resolves after a bounded interval."""

    return _build_valve_scenario(
        seed=seed,
        fault_family=FaultFamily.VALVE_LAG,
        duration_ticks=duration_ticks,
        component_id=component_id,
        lag_ticks=lag_ticks,
    )


def build_valve_stuck_scenario(
    *, seed: int, duration_ticks: int = 12, component_id: str | None = None
) -> ScenarioDefinition:
    """Build G09: the matched fictional command mismatch that remains unresolved."""

    return _build_valve_scenario(
        seed=seed,
        fault_family=FaultFamily.VALVE_STUCK,
        duration_ticks=duration_ticks,
        component_id=component_id,
        lag_ticks=None,
    )


def build_sensor_stuck_load_scenario(
    *, seed: int, duration_ticks: int = 12, channel_id: str | None = None
) -> ScenarioDefinition:
    """Build the sole supported sensor-stuck case over the benign load transient."""

    _require_uint32(seed, name="seed")
    _require_duration(duration_ticks)
    if _STUCK_FLAG_APPLY_TICK >= duration_ticks:
        raise ValueError("sensor stuck needs verification and flag application history")
    if channel_id is None:
        selected_channel = _stuck_channels()[seed % len(_stuck_channels())]
    elif isinstance(channel_id, str) and channel_id:
        selected_channel = channel_id
    else:
        raise ValueError("channel_id must be a non-empty string or None")
    if selected_channel not in _stuck_channels():
        raise ValueError("channel_id must be an Aster-A electrical-output channel")
    scenario = ScenarioDefinition(
        scenario_id=f"aster-a-stuck-load-{seed}-{duration_ticks}-{selected_channel}",
        plant_variant_id=PlantVariant.ASTER_A,
        seed=seed,
        duration_ticks=duration_ticks,
        driver=ScenarioDriver.LOAD_TRANSIENT,
        fault_injections=(
            FaultInjection(
                fault_family=FaultFamily.SENSOR_STUCK,
                component_id=_INSTRUMENTATION,
                onset_tick=_LOAD_ONSET_TICK,
                severity=SeverityBand.LOW,
                channel_id=selected_channel,
            ),
        ),
        action_sequence=(
            ScenarioAction(
                decision_tick=_STUCK_VERIFY_TICK,
                action=ActionLabel.VERIFY_REDUNDANT_CHANNEL,
            ),
            ScenarioAction(
                decision_tick=_STUCK_FLAG_TICK,
                action=ActionLabel.FLAG_SENSOR_SUSPECT,
            ),
        ),
    )
    assert_no_prohibited_content(scenario)
    return scenario


def build_sensor_noise_scenario(
    *, seed: int, duration_ticks: int = 12, channel_id: str | None = None
) -> ScenarioDefinition:
    """Build the sole supported alternating primary-thermal sensor-noise case."""

    _require_uint32(seed, name="seed")
    _require_duration(duration_ticks)
    if _SENSOR_NOISE_FLAG_APPLY_TICK >= duration_ticks:
        raise ValueError("sensor noise needs diagnosis and flag application history")
    if channel_id is None:
        selected_channel = _sensor_noise_channels()[seed % len(_sensor_noise_channels())]
    elif isinstance(channel_id, str) and channel_id:
        selected_channel = channel_id
    else:
        raise ValueError("channel_id must be a non-empty string or None")
    if selected_channel not in _sensor_noise_channels():
        raise ValueError("channel_id must be an Aster-A primary-thermal-state channel")
    scenario = ScenarioDefinition(
        scenario_id=(
            f"aster-a-noise-{seed}-{duration_ticks}-{_SENSOR_NOISE_ONSET_TICK}-low-"
            f"{selected_channel}"
        ),
        plant_variant_id=PlantVariant.ASTER_A,
        seed=seed,
        duration_ticks=duration_ticks,
        driver=ScenarioDriver.STEADY_OPERATION,
        fault_injections=(
            FaultInjection(
                fault_family=FaultFamily.SENSOR_NOISE,
                component_id=_INSTRUMENTATION,
                onset_tick=_SENSOR_NOISE_ONSET_TICK,
                severity=SeverityBand.LOW,
                channel_id=selected_channel,
            ),
        ),
        action_sequence=(
            ScenarioAction(
                decision_tick=_SENSOR_NOISE_FIRST_DECISION_TICK,
                action=ActionLabel.INSUFFICIENT_EVIDENCE,
            ),
            ScenarioAction(
                decision_tick=_SENSOR_NOISE_SECOND_DECISION_TICK,
                action=ActionLabel.INSUFFICIENT_EVIDENCE,
            ),
            ScenarioAction(
                decision_tick=_SENSOR_NOISE_DIAGNOSIS_TICK,
                action=ActionLabel.COMPARE_RELATED_TRENDS,
            ),
            ScenarioAction(
                decision_tick=_SENSOR_NOISE_FLAG_TICK,
                action=ActionLabel.FLAG_SENSOR_SUSPECT,
            ),
        ),
    )
    assert_no_prohibited_content(scenario)
    return scenario


def build_sensor_drift_scenario(
    *,
    seed: int,
    duration_ticks: int = 12,
    onset_tick: int = 3,
    severity: SeverityBand = SeverityBand.LOW,
    channel_id: str | None = None,
) -> ScenarioDefinition:
    """Build a bounded, single-channel primary-flow observation drift case."""

    _require_uint32(seed, name="seed")
    _require_duration(duration_ticks)
    _require_uint32(onset_tick, name="onset_tick")
    if not isinstance(severity, SeverityBand):
        raise ValueError("severity must be a SeverityBand")
    if onset_tick + 4 >= duration_ticks:
        raise ValueError("sensor drift needs three decisions and flag application after onset")
    if channel_id is None:
        selected_channel = _drift_channels()[seed % len(_drift_channels())]
    elif isinstance(channel_id, str) and channel_id:
        selected_channel = channel_id
    else:
        raise ValueError("channel_id must be a non-empty string or None")
    if selected_channel not in _drift_channels():
        raise ValueError("channel_id must be an Aster-A primary-flow channel")
    scenario = ScenarioDefinition(
        scenario_id=(
            f"aster-a-drift-{seed}-{duration_ticks}-{onset_tick}-"
            f"{severity.value.lower()}-{selected_channel}"
        ),
        plant_variant_id=PlantVariant.ASTER_A,
        seed=seed,
        duration_ticks=duration_ticks,
        driver=ScenarioDriver.STEADY_OPERATION,
        fault_injections=(
            FaultInjection(
                fault_family=FaultFamily.SENSOR_DRIFT,
                component_id=_PRIMARY_TRAINS[0],
                onset_tick=onset_tick,
                severity=severity,
                channel_id=selected_channel,
            ),
        ),
        action_sequence=(
            ScenarioAction(
                decision_tick=onset_tick + 1,
                action=ActionLabel.INSUFFICIENT_EVIDENCE,
            ),
            ScenarioAction(
                decision_tick=onset_tick + 2,
                action=ActionLabel.VERIFY_REDUNDANT_CHANNEL,
            ),
            ScenarioAction(
                decision_tick=onset_tick + 3,
                action=ActionLabel.FLAG_SENSOR_SUSPECT,
            ),
        ),
    )
    assert_no_prohibited_content(scenario)
    return scenario


def _pump_degradation_injection(scenario: ScenarioDefinition) -> FaultInjection | None:
    return _single_injection_for(scenario, FaultFamily.PUMP_DEGRADATION)


def _pump_trip_injection(scenario: ScenarioDefinition) -> FaultInjection | None:
    return _single_injection_for(scenario, FaultFamily.PUMP_TRIP)


def _transfer_injection(scenario: ScenarioDefinition) -> FaultInjection | None:
    return _single_injection_for(scenario, FaultFamily.TRANSFER_EFFICIENCY_LOSS)


def _flow_imbalance_injection(scenario: ScenarioDefinition) -> FaultInjection | None:
    return _single_injection_for(scenario, FaultFamily.FLOW_IMBALANCE)


def _support_power_injection(scenario: ScenarioDefinition) -> FaultInjection | None:
    return _single_injection_for(scenario, FaultFamily.SUPPORT_POWER_INTERRUPTION)


def _valve_injection(scenario: ScenarioDefinition) -> FaultInjection | None:
    signature = _fault_signature(scenario)
    if signature == (FaultFamily.VALVE_LAG,):
        return _single_injection_for(scenario, FaultFamily.VALVE_LAG)
    if signature == (FaultFamily.VALVE_STUCK,):
        return _single_injection_for(scenario, FaultFamily.VALVE_STUCK)
    return None


def _valve_decisive_tick(scenario: ScenarioDefinition) -> int:
    injection = _valve_injection(scenario)
    if injection is not None and injection.fault_family is FaultFamily.VALVE_LAG:
        if injection.duration_ticks is None:
            raise ValueError("valve lag requires a finite declared duration")
        return injection.onset_tick + injection.duration_ticks
    return _VALVE_DECISIVE_TICK


def _valve_action_tick(scenario: ScenarioDefinition) -> int:
    return _valve_decisive_tick(scenario) + 1


def _valve_initial_position(seed: int) -> float:
    """Return a seeded fictional valve position away from normalized boundaries."""

    return round(Random(seed * 7_000_001 + 191).uniform(0.44, 0.56), 6)  # noqa: S311


def _valve_commanded_position(seed: int) -> float:
    initial = _valve_initial_position(seed)
    magnitude = Random(seed * 7_000_003 + 193).uniform(0.018, 0.026)  # noqa: S311
    direction = 1.0 if seed % 2 == 0 else -1.0
    return _clip(initial + direction * magnitude)


def _valve_positions(
    *, scenario: ScenarioDefinition, tick: int, component_id: str
) -> tuple[float | None, float | None]:
    """Keep fictional command truth separate from effective component position."""

    if component_id not in ASTER_A_SPEC.primary_flow_valve_ids:
        return None, None
    initial = _valve_initial_position(scenario.seed)
    injection = _valve_injection(scenario)
    if injection is None or tick < _VALVE_ONSET_TICK:
        return initial, initial
    commanded = _valve_commanded_position(scenario.seed)
    if (
        injection.fault_family is FaultFamily.VALVE_LAG
        and injection.duration_ticks is not None
        and tick >= injection.onset_tick + injection.duration_ticks
    ):
        return commanded, commanded
    return commanded, initial


def _component_positions(
    *, scenario: ScenarioDefinition, tick: int, component_id: str
) -> _ComponentPositionFields:
    commanded_position, actual_position = _valve_positions(
        scenario=scenario, tick=tick, component_id=component_id
    )
    return {
        "commanded_position": commanded_position,
        "actual_position": actual_position,
    }


def _valve_flow_delta(scenario: ScenarioDefinition, tick: int) -> float:
    injection = _valve_injection(scenario)
    if (
        injection is None
        or injection.fault_family is not FaultFamily.VALVE_LAG
        or injection.duration_ticks is None
        or tick < _valve_decisive_tick(scenario)
    ):
        return 0.0
    return round(
        (_valve_commanded_position(scenario.seed) - _valve_initial_position(scenario.seed)) * 0.55,
        6,
    )


def _valve_values(scenario: ScenarioDefinition, tick: int) -> PlantValues:
    baseline = _baseline_values(scenario.seed)
    values = baseline.model_dump()
    flow_delta = _valve_flow_delta(scenario, tick)
    if flow_delta:
        values[StateVariable.PRIMARY_FLOW.value] = _clip(baseline.primary_flow + flow_delta)
    return PlantValues(**values)


def _baseline_values(seed: int) -> PlantValues:
    values: dict[str, float] = {}
    for index, variable in enumerate(StateVariable):
        stream = seed * 1_000_003 + index * 7_919 + 17
        values[variable.value] = round(
            0.47 + Random(stream).uniform(-0.045, 0.045),  # noqa: S311
            6,
        )
    return PlantValues(**values)


def _load_direction(seed: int) -> float:
    return 1.0 if seed % 2 == 0 else -1.0


def _load_magnitude(seed: int) -> float:
    return 0.036 + Random(seed * 3_000_017 + 43).uniform(-0.006, 0.006)  # noqa: S311


def _load_progress(tick: int, start_tick: int) -> float:
    if tick < start_tick:
        return 0.0
    return min(1.0, (tick - start_tick + 1) / _LOAD_RAMP_TICKS)


def _load_transient_values(seed: int, tick: int) -> PlantValues:
    baseline = _baseline_values(seed)
    values = baseline.model_dump()
    for variable, (weight, start_tick) in _LOAD_RESPONSE_STAGES.items():
        change = _load_direction(seed) * _load_magnitude(seed) * _load_progress(tick, start_tick)
        values[variable.value] = _clip(values[variable.value] + change * weight)
    return PlantValues(**values)


def _pump_health_step(seed: int) -> float:
    """Return a seed-derived fictional health step in the closed [0.010, 0.014] range."""

    return round(Random(seed * 4_000_019 + 97).uniform(0.010, 0.014), 6)  # noqa: S311


def _process_loss_step(seed: int, *, stream_offset: int) -> float:
    """Return one local, seed-derived normalized process step in [0.010, 0.014]."""

    return round(Random(seed * 7_000_013 + stream_offset).uniform(0.010, 0.014), 6)  # noqa: S311


def _transfer_values(seed: int, tick: int) -> PlantValues:
    """G10 latent values; untouched variables remain exactly at baseline."""

    baseline = _baseline_values(seed)
    values = baseline.model_dump()
    step = _process_loss_step(seed, stream_offset=211)
    if tick >= _TRANSFER_ONSET_TICK:
        values[StateVariable.TRANSFER_EFFICIENCY.value] = _clip(
            baseline.transfer_efficiency - min(0.12, step * (tick - 1))
        )
    if tick >= _TRANSFER_THERMAL_TICK:
        values[StateVariable.PRIMARY_THERMAL_STATE.value] = _clip(
            baseline.primary_thermal_state + min(0.10, 0.8 * step * (tick - 2))
        )
    if tick >= _TRANSFER_STEAM_TICK:
        values[StateVariable.STEAM_STATE.value] = _clip(
            baseline.steam_state - min(0.10, 0.8 * step * (tick - 3))
        )
    if tick >= _TRANSFER_OUTPUT_TICK:
        loss = min(0.09, 0.7 * step * (tick - 4))
        values[StateVariable.TURBINE_OUTPUT.value] = _clip(baseline.turbine_output - loss)
        values[StateVariable.ELECTRICAL_OUTPUT.value] = _clip(baseline.electrical_output - loss)
    if tick >= _TRANSFER_LOAD_APPLY_TICK:
        reduction = min(0.036, 0.012 * (tick - 5))
        values[StateVariable.LOAD_DEMAND.value] = _clip(baseline.load_demand - reduction)
        values[StateVariable.HEAT_SOURCE_LEVEL.value] = _clip(
            baseline.heat_source_level - reduction
        )
    return PlantValues(**values)


def _flow_imbalance_values(seed: int, tick: int) -> PlantValues:
    """G11 latent values; secondary effects arrive after the initiating flow loss."""

    baseline = _baseline_values(seed)
    values = baseline.model_dump()
    step = _process_loss_step(seed, stream_offset=307)
    if tick >= _FLOW_IMBALANCE_ONSET_TICK:
        values[StateVariable.SECONDARY_FLOW.value] = _clip(
            baseline.secondary_flow - min(0.12, step * (tick - 1))
        )
    if tick >= _FLOW_IMBALANCE_INVENTORY_TICK:
        values[StateVariable.SECONDARY_INVENTORY.value] = _clip(
            baseline.secondary_inventory - min(0.10, 0.85 * step * (tick - 2))
        )
    if tick >= _FLOW_IMBALANCE_STEAM_TICK:
        values[StateVariable.STEAM_STATE.value] = _clip(
            baseline.steam_state - min(0.09, 0.75 * step * (tick - 4))
        )
    if tick >= _FLOW_IMBALANCE_OUTPUT_TICK:
        loss = min(0.09, 0.7 * step * (tick - 5))
        values[StateVariable.TURBINE_OUTPUT.value] = _clip(baseline.turbine_output - loss)
        values[StateVariable.ELECTRICAL_OUTPUT.value] = _clip(baseline.electrical_output - loss)
    if tick >= _FLOW_IMBALANCE_STABILIZE_APPLY_TICK:
        reduction = min(0.036, 0.012 * (tick - 6))
        values[StateVariable.LOAD_DEMAND.value] = _clip(baseline.load_demand - reduction)
        values[StateVariable.HEAT_SOURCE_LEVEL.value] = _clip(
            baseline.heat_source_level - reduction
        )
    return PlantValues(**values)


def _pump_health_loss(*, step: float, tick: int) -> float:
    if tick < _PUMP_DEGRADATION_ONSET_TICK:
        return 0.0
    return min(0.24, step * (tick - 1))


def _pump_action_ramp(tick: int) -> float:
    if tick < _PUMP_LOAD_APPLY_TICK:
        return 0.0
    return min(0.036, 0.012 * (tick - 7))


def _pump_values(seed: int, tick: int) -> PlantValues:
    baseline = _baseline_values(seed)
    values = baseline.model_dump()
    step = _pump_health_step(seed)
    if tick >= _PUMP_FLOW_TICK:
        values[StateVariable.PRIMARY_FLOW.value] = _clip(
            baseline.primary_flow - min(0.18, 1.35 * step * (tick - 2))
        )
    if tick >= _PUMP_THERMAL_TICK:
        values[StateVariable.PRIMARY_THERMAL_STATE.value] = _clip(
            baseline.primary_thermal_state + min(0.12, 1.35 * step * (tick - 3))
        )
    if tick >= _PUMP_STEAM_TICK:
        values[StateVariable.STEAM_STATE.value] = _clip(
            baseline.steam_state - min(0.14, 1.35 * step * (tick - 4))
        )
    if tick >= _PUMP_OUTPUT_TICK:
        output_loss = min(0.14, 1.35 * step * (tick - 5))
        values[StateVariable.TURBINE_OUTPUT.value] = _clip(baseline.turbine_output - output_loss)
        values[StateVariable.ELECTRICAL_OUTPUT.value] = _clip(
            baseline.electrical_output - output_loss
        )
    action_ramp = _pump_action_ramp(tick)
    if action_ramp:
        values[StateVariable.LOAD_DEMAND.value] = _clip(baseline.load_demand - action_ramp)
        values[StateVariable.HEAT_SOURCE_LEVEL.value] = _clip(
            baseline.heat_source_level - action_ramp
        )
    return PlantValues(**values)


def _pump_components(
    *, scenario: ScenarioDefinition, selected_component: str, step: float, tick: int
) -> tuple[ComponentLatentState, ...]:
    health_loss = _pump_health_loss(step=step, tick=tick)
    return tuple(
        ComponentLatentState(
            component_id=component.component_id,
            state=(
                ComponentState.DEGRADED
                if component.component_id == selected_component
                and tick >= _PUMP_DEGRADATION_ONSET_TICK
                else ComponentState.AVAILABLE
            ),
            health=(
                _clip(1.0 - health_loss) if component.component_id == selected_component else 1.0
            ),
            pending_maintenance=(
                component.component_id == selected_component and tick >= _PUMP_INSPECTION_APPLY_TICK
            ),
            **_component_positions(
                scenario=scenario,
                tick=tick,
                component_id=component.component_id,
            ),
        )
        for component in ASTER_A_SPEC.components
    )


def _trip_flow_drop(seed: int) -> float:
    """Return the sole seed-derived abrupt G07 process step."""

    return round(Random(seed * 6_000_017 + 131).uniform(0.096, 0.12), 6)  # noqa: S311


def _trip_action_ramp(tick: int) -> float:
    if tick < _PUMP_TRIP_ACTION_TICK:
        return 0.0
    return min(0.036, 0.012 * (tick - _PUMP_TRIP_ACTION_TICK + 1))


def _trip_values(scenario: ScenarioDefinition, tick: int) -> PlantValues:
    context = scenario.standby_context
    if context is None:
        raise ValueError("pump-trip values require standby context")
    baseline = _baseline_values(scenario.seed)
    values = baseline.model_dump()
    if tick >= _PUMP_TRIP_FLOW_TICK:
        flow = baseline.primary_flow - _trip_flow_drop(scenario.seed)
        if context.standby_state is ComponentState.AVAILABLE and tick >= _PUMP_TRIP_RECOVERY_TICK:
            flow += min(0.054, 0.018 * (tick - _PUMP_TRIP_RECOVERY_TICK + 1))
        values[StateVariable.PRIMARY_FLOW.value] = _clip(flow)
    if tick >= _PUMP_TRIP_THERMAL_TICK:
        values[StateVariable.PRIMARY_THERMAL_STATE.value] = _clip(
            baseline.primary_thermal_state + 0.018
        )
    if tick >= _PUMP_TRIP_STEAM_TICK:
        values[StateVariable.STEAM_STATE.value] = _clip(baseline.steam_state - 0.018)
        values[StateVariable.TURBINE_OUTPUT.value] = _clip(baseline.turbine_output - 0.016)
        values[StateVariable.ELECTRICAL_OUTPUT.value] = _clip(baseline.electrical_output - 0.016)
    if context.standby_state is ComponentState.UNAVAILABLE:
        action_ramp = _trip_action_ramp(tick)
        if action_ramp:
            values[StateVariable.LOAD_DEMAND.value] = _clip(baseline.load_demand - action_ramp)
            values[StateVariable.HEAT_SOURCE_LEVEL.value] = _clip(
                baseline.heat_source_level - action_ramp
            )
    return PlantValues(**values)


def _trip_components(
    *, scenario: ScenarioDefinition, tick: int
) -> tuple[ComponentLatentState, ...]:
    context = scenario.standby_context
    if context is None:
        raise ValueError("pump-trip components require standby context")

    def state_for(component_id: str) -> ComponentState:
        if component_id == context.active_train_id:
            return (
                ComponentState.AVAILABLE
                if tick < _PUMP_TRIP_ONSET_TICK
                else ComponentState.UNAVAILABLE
            )
        if component_id != context.standby_train_id:
            return ComponentState.AVAILABLE
        if context.standby_state is ComponentState.UNAVAILABLE:
            return ComponentState.UNAVAILABLE
        if tick < _PUMP_TRIP_ACTION_TICK:
            return ComponentState.AVAILABLE
        if tick < _PUMP_TRIP_RECOVERY_TICK:
            return ComponentState.STARTING
        return ComponentState.RECOVERING

    return tuple(
        ComponentLatentState(
            component_id=component.component_id,
            state=state_for(component.component_id),
            health=1.0,
            **_component_positions(
                scenario=scenario, tick=tick, component_id=component.component_id
            ),
        )
        for component in ASTER_A_SPEC.components
    )


def _healthy_components(
    *, scenario: ScenarioDefinition, tick: int, spec: AsterVariantSpec
) -> tuple[ComponentLatentState, ...]:
    """Return the fully available component topology for process-only G10/G11 cases."""

    return tuple(
        ComponentLatentState(
            component_id=component.component_id,
            state=ComponentState.AVAILABLE,
            health=1.0,
            **_component_positions(
                scenario=scenario, tick=tick, component_id=component.component_id
            ),
        )
        for component in spec.components
    )


def _support_power_drop(seed: int) -> float:
    """Return one deterministic normalized support-power loss in the reviewed band."""

    return round(Random(seed * 6_000_029 + 401).uniform(0.018, 0.024), 6)  # noqa: S311


def _support_power_roles(*, spec: AsterVariantSpec, bus_id: str) -> frozenset[ComponentRole]:
    return frozenset(
        component.role
        for component in spec.components
        if component.component_id in spec.dependents_for(bus_id)
    )


def _support_power_values(
    *, scenario: ScenarioDefinition, tick: int, spec: AsterVariantSpec
) -> PlantValues:
    """Return G12 values with effects derived exclusively from the registry roles."""

    baseline = _baseline_values(scenario.seed)
    values = baseline.model_dump()
    injection = _support_power_injection(scenario)
    if injection is None:
        return baseline
    bus_id = injection.component_id
    roles = _support_power_roles(spec=spec, bus_id=bus_id)

    if tick >= _SUPPORT_POWER_ONSET_TICK:
        values[StateVariable.SUPPORT_POWER.value] = _clip(
            baseline.support_power - _support_power_drop(scenario.seed)
        )
    if tick >= _SUPPORT_POWER_EFFECT_TICK:
        primary_flow_delta = 0.0
        if ComponentRole.PRIMARY_TRAIN_ONE in roles:
            primary_flow_delta -= 0.010
        if ComponentRole.PRIMARY_TRAIN_TWO in roles:
            primary_flow_delta -= 0.010
        if ComponentRole.PRIMARY_FLOW_VALVE in roles:
            primary_flow_delta -= 0.008
        primary_flow_delta = -min(-primary_flow_delta, spec.max_per_tick_step)
        if primary_flow_delta:
            values[StateVariable.PRIMARY_FLOW.value] = _clip(
                baseline.primary_flow + primary_flow_delta
            )
        if ComponentRole.TRANSFER_UNIT in roles:
            values[StateVariable.TRANSFER_EFFICIENCY.value] = _clip(
                baseline.transfer_efficiency - 0.012
            )
        if ComponentRole.SECONDARY_FEED in roles:
            values[StateVariable.SECONDARY_FLOW.value] = _clip(baseline.secondary_flow - 0.012)
    if tick >= _SUPPORT_POWER_DELAYED_TICK:
        if ComponentRole.TRANSFER_UNIT in roles:
            values[StateVariable.PRIMARY_THERMAL_STATE.value] = _clip(
                baseline.primary_thermal_state + 0.008
            )
            values[StateVariable.STEAM_STATE.value] = _clip(baseline.steam_state - 0.008)
        if ComponentRole.SECONDARY_FEED in roles:
            values[StateVariable.SECONDARY_INVENTORY.value] = _clip(
                baseline.secondary_inventory - 0.008
            )
    if scenario.dependency_map_context is not None and tick >= _SUPPORT_POWER_ACTION_TICK:
        reduction = min(0.036, 0.012 * (tick - _SUPPORT_POWER_ACTION_TICK + 1))
        values[StateVariable.LOAD_DEMAND.value] = _clip(baseline.load_demand - reduction)
        values[StateVariable.HEAT_SOURCE_LEVEL.value] = _clip(
            baseline.heat_source_level - reduction
        )
    return PlantValues(**values)


def _support_power_components(
    *, scenario: ScenarioDefinition, tick: int, spec: AsterVariantSpec
) -> tuple[ComponentLatentState, ...]:
    injection = _support_power_injection(scenario)
    if injection is None:
        return _healthy_components(scenario=scenario, tick=tick, spec=spec)
    unavailable = {injection.component_id} if tick >= _SUPPORT_POWER_ONSET_TICK else set()
    if tick >= _SUPPORT_POWER_COMPONENT_TICK:
        unavailable.update(spec.dependents_for(injection.component_id))
    return tuple(
        ComponentLatentState(
            component_id=component.component_id,
            state=(
                ComponentState.UNAVAILABLE
                if component.component_id in unavailable
                else ComponentState.AVAILABLE
            ),
            health=1.0,
            **_component_positions(
                scenario=scenario, tick=tick, component_id=component.component_id
            ),
        )
        for component in spec.components
    )


def _latent_states(scenario: ScenarioDefinition) -> tuple[LatentPlantState, ...]:
    spec = _spec_for(scenario)
    baseline = _baseline_values(scenario.seed)
    trip = _pump_trip_injection(scenario)
    if trip is not None:
        context = scenario.standby_context
        if context is None:
            raise ValueError("pump-trip latent states require standby context")
        return tuple(
            LatentPlantState(
                tick=tick,
                operating_mode=(
                    OperatingMode.STABLE
                    if tick < _PUMP_TRIP_ONSET_TICK
                    else OperatingMode.DISTURBED
                    if tick < _PUMP_TRIP_RECOVERY_TICK
                    else OperatingMode.RECOVERY
                    if context.standby_state is ComponentState.AVAILABLE
                    else OperatingMode.STABILIZED
                ),
                values=_trip_values(scenario, tick),
                components=_trip_components(scenario=scenario, tick=tick),
            )
            for tick in range(scenario.duration_ticks)
        )
    transfer = _transfer_injection(scenario)
    if transfer is not None:
        return tuple(
            LatentPlantState(
                tick=tick,
                operating_mode=(
                    OperatingMode.STABLE
                    if tick < _TRANSFER_ONSET_TICK
                    else OperatingMode.DISTURBED
                    if tick < _TRANSFER_LOAD_APPLY_TICK
                    else OperatingMode.RECOVERY
                ),
                values=_transfer_values(scenario.seed, tick),
                components=_healthy_components(scenario=scenario, tick=tick, spec=spec),
            )
            for tick in range(scenario.duration_ticks)
        )
    flow_imbalance = _flow_imbalance_injection(scenario)
    if flow_imbalance is not None:
        return tuple(
            LatentPlantState(
                tick=tick,
                operating_mode=(
                    OperatingMode.STABLE
                    if tick < _FLOW_IMBALANCE_ONSET_TICK
                    else OperatingMode.DISTURBED
                    if tick < _FLOW_IMBALANCE_STABILIZE_APPLY_TICK
                    else OperatingMode.STABILIZED
                ),
                values=_flow_imbalance_values(scenario.seed, tick),
                components=_healthy_components(scenario=scenario, tick=tick, spec=spec),
            )
            for tick in range(scenario.duration_ticks)
        )
    support_power = _support_power_injection(scenario)
    if support_power is not None:
        return tuple(
            LatentPlantState(
                tick=tick,
                operating_mode=(
                    OperatingMode.STABLE
                    if tick < _SUPPORT_POWER_ONSET_TICK
                    else OperatingMode.DISTURBED
                    if tick < _SUPPORT_POWER_ACTION_TICK or scenario.dependency_map_context is None
                    else OperatingMode.RECOVERY
                    if tick < _SUPPORT_POWER_STABILIZED_TICK
                    else OperatingMode.STABILIZED
                ),
                values=_support_power_values(scenario=scenario, tick=tick, spec=spec),
                components=_support_power_components(scenario=scenario, tick=tick, spec=spec),
            )
            for tick in range(scenario.duration_ticks)
        )
    pump = _pump_degradation_injection(scenario)
    if pump is not None:
        step = _pump_health_step(scenario.seed)
        return tuple(
            LatentPlantState(
                tick=tick,
                operating_mode=(
                    OperatingMode.STABLE
                    if tick < _PUMP_DEGRADATION_ONSET_TICK
                    else OperatingMode.DISTURBED
                    if tick < _PUMP_LOAD_APPLY_TICK
                    else OperatingMode.RECOVERY
                ),
                values=_pump_values(scenario.seed, tick),
                components=_pump_components(
                    scenario=scenario,
                    selected_component=pump.component_id,
                    step=step,
                    tick=tick,
                ),
            )
            for tick in range(scenario.duration_ticks)
        )
    valve = _valve_injection(scenario)
    if valve is not None:
        decisive_tick = _valve_decisive_tick(scenario)
        action_tick = _valve_action_tick(scenario)
        return tuple(
            LatentPlantState(
                tick=tick,
                operating_mode=(
                    OperatingMode.STABLE
                    if tick < _VALVE_ONSET_TICK
                    else OperatingMode.DISTURBED
                    if valve.fault_family is FaultFamily.VALVE_STUCK or tick < decisive_tick
                    else OperatingMode.RECOVERY
                ),
                values=_valve_values(scenario, tick),
                components=tuple(
                    ComponentLatentState(
                        component_id=component.component_id,
                        state=ComponentState.AVAILABLE,
                        health=1.0,
                        pending_maintenance=(
                            valve.fault_family is FaultFamily.VALVE_STUCK
                            and component.component_id == valve.component_id
                            and tick >= action_tick
                        ),
                        **_component_positions(
                            scenario=scenario,
                            tick=tick,
                            component_id=component.component_id,
                        ),
                    )
                    for component in ASTER_A_SPEC.components
                ),
            )
            for tick in range(scenario.duration_ticks)
        )
    components = tuple(
        ComponentLatentState(
            component_id=component.component_id,
            state=ComponentState.AVAILABLE,
            health=1.0,
            **_component_positions(scenario=scenario, tick=0, component_id=component.component_id),
        )
        for component in spec.components
    )
    return tuple(
        LatentPlantState(
            tick=tick,
            operating_mode=(
                OperatingMode.LOAD_CHANGE
                if _LOAD_ONSET_TICK <= tick < _LOAD_SETTLE_TICK
                and scenario.driver is ScenarioDriver.LOAD_TRANSIENT
                else OperatingMode.STABLE
            ),
            values=(
                _load_transient_values(scenario.seed, tick)
                if scenario.driver is ScenarioDriver.LOAD_TRANSIENT
                else baseline
            ),
            components=components,
        )
        for tick in range(scenario.duration_ticks)
    )


def _noise(seed: int, tick: int, variable_index: int, *, bound: float = _NOISE_BOUND) -> float:
    stream = seed * 2_000_033 + tick * 4_099 + variable_index * 101 + 29
    return Random(stream).uniform(-bound, bound)  # noqa: S311


def _clip(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 6)


def _drift_bias(*, scenario: ScenarioDefinition, tick: int, channel_id: str) -> float:
    injection = _single_injection_for(scenario, FaultFamily.SENSOR_DRIFT)
    if injection is None:
        return 0.0
    if tick <= injection.onset_tick or channel_id != injection.channel_id:
        return 0.0
    plateau = {SeverityBand.LOW: 0.042, SeverityBand.MEDIUM: 0.056, SeverityBand.HIGH: 0.07}[
        injection.severity
    ]
    magnitude = min(plateau, plateau * (tick - injection.onset_tick) / 3)
    direction = 1.0 if scenario.seed % 2 == 0 else -1.0
    return direction * magnitude


def _sensor_stuck_injection(scenario: ScenarioDefinition) -> FaultInjection | None:
    return _single_injection_for(scenario, FaultFamily.SENSOR_STUCK)


def _sensor_noise_injection(scenario: ScenarioDefinition) -> FaultInjection | None:
    return _single_injection_for(scenario, FaultFamily.SENSOR_NOISE)


def _sensor_noise_offset(*, seed: int, tick: int) -> float:
    if tick <= _SENSOR_NOISE_ONSET_TICK:
        return 0.0
    pair_index = (tick - _SENSOR_NOISE_FIRST_DECISION_TICK) // 2
    amplitude = Random(seed * 5_000_011 + pair_index * 17_777 + 71).uniform(  # noqa: S311
        0.018, 0.024
    )
    phase = 1.0 if seed % 2 == 0 else -1.0
    alternating = -1.0 if (tick - _SENSOR_NOISE_FIRST_DECISION_TICK) % 2 else 1.0
    return phase * amplitude * alternating


def _pump_channel_status(tick: int, variable: StateVariable) -> ObservationStatus:
    if variable is StateVariable.PRIMARY_FLOW:
        if tick < _PUMP_FLOW_TICK:
            return ObservationStatus.NORMAL
        return ObservationStatus.WATCH if tick <= 5 else ObservationStatus.ABNORMAL
    if variable is StateVariable.PRIMARY_THERMAL_STATE:
        if tick < _PUMP_THERMAL_TICK:
            return ObservationStatus.NORMAL
        return ObservationStatus.WATCH if tick <= 5 else ObservationStatus.ABNORMAL
    if variable is StateVariable.STEAM_STATE:
        if tick < _PUMP_STEAM_TICK:
            return ObservationStatus.NORMAL
        return ObservationStatus.WATCH if tick == 5 else ObservationStatus.ABNORMAL
    if variable in {StateVariable.TURBINE_OUTPUT, StateVariable.ELECTRICAL_OUTPUT}:
        return ObservationStatus.NORMAL if tick < _PUMP_OUTPUT_TICK else ObservationStatus.ABNORMAL
    return ObservationStatus.NORMAL


def _pump_overall_status(tick: int) -> ObservationStatus:
    if tick < _PUMP_DEGRADATION_ONSET_TICK:
        return ObservationStatus.NORMAL
    return ObservationStatus.WATCH if tick <= 5 else ObservationStatus.ABNORMAL


def _trip_channel_status(
    scenario: ScenarioDefinition, tick: int, variable: StateVariable
) -> ObservationStatus:
    context = scenario.standby_context
    if context is None:
        raise ValueError("pump-trip status requires standby context")
    if variable is StateVariable.PRIMARY_FLOW:
        if tick < _PUMP_TRIP_FLOW_TICK:
            return ObservationStatus.NORMAL
        if context.standby_state is ComponentState.AVAILABLE and tick >= _PUMP_TRIP_RECOVERY_TICK:
            return ObservationStatus.WATCH
        return ObservationStatus.ABNORMAL
    if variable is StateVariable.PRIMARY_THERMAL_STATE:
        if tick < _PUMP_TRIP_THERMAL_TICK:
            return ObservationStatus.NORMAL
        return (
            ObservationStatus.WATCH
            if tick == _PUMP_TRIP_THERMAL_TICK
            else ObservationStatus.ABNORMAL
        )
    if variable is StateVariable.STEAM_STATE:
        if tick < _PUMP_TRIP_STEAM_TICK:
            return ObservationStatus.NORMAL
        return (
            ObservationStatus.WATCH if tick == _PUMP_TRIP_STEAM_TICK else ObservationStatus.ABNORMAL
        )
    if variable in {StateVariable.TURBINE_OUTPUT, StateVariable.ELECTRICAL_OUTPUT}:
        if tick < _PUMP_TRIP_STEAM_TICK:
            return ObservationStatus.NORMAL
        return (
            ObservationStatus.WATCH if tick == _PUMP_TRIP_STEAM_TICK else ObservationStatus.ABNORMAL
        )
    return ObservationStatus.NORMAL


def _trip_overall_status(tick: int) -> ObservationStatus:
    return ObservationStatus.NORMAL if tick < _PUMP_TRIP_ONSET_TICK else ObservationStatus.ABNORMAL


def _transfer_channel_status(tick: int, variable: StateVariable) -> ObservationStatus:
    if variable is StateVariable.TRANSFER_EFFICIENCY:
        return (
            ObservationStatus.NORMAL
            if tick < _TRANSFER_ONSET_TICK
            else ObservationStatus.WATCH
            if tick <= 4
            else ObservationStatus.ABNORMAL
        )
    if variable is StateVariable.PRIMARY_THERMAL_STATE:
        return (
            ObservationStatus.NORMAL
            if tick < _TRANSFER_THERMAL_TICK
            else ObservationStatus.WATCH
            if tick <= 4
            else ObservationStatus.ABNORMAL
        )
    if variable is StateVariable.STEAM_STATE:
        return (
            ObservationStatus.NORMAL
            if tick < _TRANSFER_STEAM_TICK
            else ObservationStatus.WATCH
            if tick == _TRANSFER_STEAM_TICK
            else ObservationStatus.ABNORMAL
        )
    if variable in {StateVariable.TURBINE_OUTPUT, StateVariable.ELECTRICAL_OUTPUT}:
        return (
            ObservationStatus.NORMAL
            if tick < _TRANSFER_OUTPUT_TICK
            else ObservationStatus.WATCH
            if tick == _TRANSFER_OUTPUT_TICK
            else ObservationStatus.ABNORMAL
        )
    return ObservationStatus.NORMAL


def _transfer_overall_status(tick: int) -> ObservationStatus:
    if tick < _TRANSFER_ONSET_TICK:
        return ObservationStatus.NORMAL
    return ObservationStatus.WATCH if tick <= 4 else ObservationStatus.ABNORMAL


def _flow_imbalance_channel_status(tick: int, variable: StateVariable) -> ObservationStatus:
    if variable is StateVariable.SECONDARY_FLOW:
        return (
            ObservationStatus.NORMAL
            if tick < _FLOW_IMBALANCE_ONSET_TICK
            else ObservationStatus.WATCH
            if tick <= 3
            else ObservationStatus.ABNORMAL
        )
    if variable is StateVariable.SECONDARY_INVENTORY:
        return (
            ObservationStatus.NORMAL
            if tick < _FLOW_IMBALANCE_INVENTORY_TICK
            else ObservationStatus.WATCH
            if tick <= 4
            else ObservationStatus.ABNORMAL
        )
    if variable is StateVariable.STEAM_STATE:
        return (
            ObservationStatus.NORMAL
            if tick < _FLOW_IMBALANCE_STEAM_TICK
            else ObservationStatus.WATCH
            if tick == _FLOW_IMBALANCE_STEAM_TICK
            else ObservationStatus.ABNORMAL
        )
    if variable in {StateVariable.TURBINE_OUTPUT, StateVariable.ELECTRICAL_OUTPUT}:
        return (
            ObservationStatus.NORMAL
            if tick < _FLOW_IMBALANCE_OUTPUT_TICK
            else ObservationStatus.WATCH
            if tick == _FLOW_IMBALANCE_OUTPUT_TICK
            else ObservationStatus.ABNORMAL
        )
    return ObservationStatus.NORMAL


def _flow_imbalance_overall_status(tick: int) -> ObservationStatus:
    if tick < _FLOW_IMBALANCE_ONSET_TICK:
        return ObservationStatus.NORMAL
    return ObservationStatus.WATCH if tick <= 3 else ObservationStatus.ABNORMAL


def _support_power_channel_status(
    scenario: ScenarioDefinition, tick: int, variable: StateVariable
) -> ObservationStatus:
    injection = _support_power_injection(scenario)
    if injection is None:
        return ObservationStatus.NORMAL
    spec = get_variant_spec(scenario.plant_variant_id)
    roles = _support_power_roles(spec=spec, bus_id=injection.component_id)
    if tick < _SUPPORT_POWER_ONSET_TICK:
        return ObservationStatus.NORMAL
    affected = (
        variable is StateVariable.SUPPORT_POWER
        or (
            variable is StateVariable.PRIMARY_FLOW
            and bool(
                roles
                & {
                    ComponentRole.PRIMARY_TRAIN_ONE,
                    ComponentRole.PRIMARY_TRAIN_TWO,
                    ComponentRole.PRIMARY_FLOW_VALVE,
                }
            )
        )
        or (variable is StateVariable.TRANSFER_EFFICIENCY and ComponentRole.TRANSFER_UNIT in roles)
        or (
            variable
            in {
                StateVariable.PRIMARY_THERMAL_STATE,
                StateVariable.STEAM_STATE,
            }
            and ComponentRole.TRANSFER_UNIT in roles
        )
        or (
            variable
            in {
                StateVariable.SECONDARY_FLOW,
                StateVariable.SECONDARY_INVENTORY,
            }
            and ComponentRole.SECONDARY_FEED in roles
        )
    )
    if not affected:
        return ObservationStatus.NORMAL
    effect_tick = (
        _SUPPORT_POWER_ONSET_TICK
        if variable is StateVariable.SUPPORT_POWER
        else _SUPPORT_POWER_EFFECT_TICK
        if variable
        in {
            StateVariable.PRIMARY_FLOW,
            StateVariable.TRANSFER_EFFICIENCY,
            StateVariable.SECONDARY_FLOW,
        }
        else _SUPPORT_POWER_DELAYED_TICK
    )
    if tick < effect_tick:
        return ObservationStatus.NORMAL
    return (
        ObservationStatus.WATCH if tick < _SUPPORT_POWER_ACTION_TICK else ObservationStatus.ABNORMAL
    )


def _support_power_overall_status(tick: int) -> ObservationStatus:
    if tick < _SUPPORT_POWER_ONSET_TICK:
        return ObservationStatus.NORMAL
    return (
        ObservationStatus.WATCH if tick < _SUPPORT_POWER_ACTION_TICK else ObservationStatus.ABNORMAL
    )


def _valve_channel_status(
    scenario: ScenarioDefinition, tick: int, variable: StateVariable
) -> ObservationStatus:
    if variable is not StateVariable.PRIMARY_FLOW:
        return ObservationStatus.NORMAL
    injection = _valve_injection(scenario)
    if injection is None or tick < _valve_decisive_tick(scenario):
        return ObservationStatus.NORMAL
    return (
        ObservationStatus.WATCH
        if injection.fault_family is FaultFamily.VALVE_LAG
        else ObservationStatus.NORMAL
    )


def _valve_overall_status(scenario: ScenarioDefinition, tick: int) -> ObservationStatus:
    injection = _valve_injection(scenario)
    if injection is None or tick < _valve_decisive_tick(scenario):
        return ObservationStatus.NORMAL
    return (
        ObservationStatus.WATCH
        if injection.fault_family is FaultFamily.VALVE_LAG
        else ObservationStatus.ABNORMAL
    )


def _selected_channel_status(
    scenario: ScenarioDefinition, tick: int, channel_id: str
) -> ObservationStatus:
    if len(scenario.fault_injections) != 1:
        return ObservationStatus.NORMAL
    injection = scenario.fault_injections[0]
    if channel_id != injection.channel_id:
        return ObservationStatus.NORMAL
    if injection.fault_family is FaultFamily.SENSOR_STUCK:
        if tick < _LOAD_OUTPUT_TICK:
            return ObservationStatus.NORMAL
        if tick == _LOAD_OUTPUT_TICK:
            return ObservationStatus.WATCH
        return ObservationStatus.CONFLICTING
    if injection.fault_family is FaultFamily.SENSOR_NOISE:
        if tick <= _SENSOR_NOISE_ONSET_TICK:
            return ObservationStatus.NORMAL
        if tick <= _SENSOR_NOISE_SECOND_DECISION_TICK:
            return ObservationStatus.WATCH
        return ObservationStatus.CONFLICTING
    if injection.fault_family is not FaultFamily.SENSOR_DRIFT:
        return ObservationStatus.NORMAL
    onset_tick = injection.onset_tick
    if tick <= onset_tick:
        return ObservationStatus.NORMAL
    if tick == onset_tick + 1:
        return ObservationStatus.WATCH
    return ObservationStatus.CONFLICTING


def _selected_channel_quality(
    scenario: ScenarioDefinition, tick: int, channel_id: str
) -> ChannelQuality:
    injection = _sensor_stuck_injection(scenario)
    if (
        injection is not None
        and channel_id == injection.channel_id
        and tick >= _STUCK_FLAG_APPLY_TICK
    ):
        return ChannelQuality.SUSPECT
    sensor_noise = _sensor_noise_injection(scenario)
    if (
        sensor_noise is not None
        and channel_id == sensor_noise.channel_id
        and tick >= _SENSOR_NOISE_FLAG_APPLY_TICK
    ):
        return ChannelQuality.SUSPECT
    drift = _single_injection_for(scenario, FaultFamily.SENSOR_DRIFT)
    if drift is not None and channel_id == drift.channel_id and tick >= drift.onset_tick + 4:
        return ChannelQuality.SUSPECT
    return ChannelQuality.GOOD


def _observations(
    scenario: ScenarioDefinition, latent_states: tuple[LatentPlantState, ...]
) -> tuple[ObservationFrame, ...]:
    spec = _spec_for(scenario)
    frames: list[ObservationFrame] = []
    pump = _pump_degradation_injection(scenario)
    trip = _pump_trip_injection(scenario)
    transfer = _transfer_injection(scenario)
    flow_imbalance = _flow_imbalance_injection(scenario)
    support_power = _support_power_injection(scenario)
    valve = _valve_injection(scenario)
    stuck = _sensor_stuck_injection(scenario)
    sensor_noise = _sensor_noise_injection(scenario)
    for latent in latent_states:
        channels: list[SensorChannelObservation] = []
        for index, variable in enumerate(StateVariable):
            base = getattr(latent.values, variable.value) + _noise(
                scenario.seed, latent.tick, index, bound=spec.baseline_noise_bound
            )
            for channel in (channel for channel in spec.channels if channel.variable is variable):
                observed_value = _clip(
                    base
                    + _drift_bias(
                        scenario=scenario, tick=latent.tick, channel_id=channel.channel_id
                    )
                )
                if (
                    stuck is not None
                    and channel.channel_id == stuck.channel_id
                    and latent.tick >= stuck.onset_tick
                ):
                    reference_latent = latent_states[stuck.onset_tick - 1]
                    reference = getattr(reference_latent.values, variable.value) + _noise(
                        scenario.seed,
                        stuck.onset_tick - 1,
                        index,
                        bound=spec.baseline_noise_bound,
                    )
                    observed_value = _clip(reference)
                if (
                    sensor_noise is not None
                    and channel.channel_id == sensor_noise.channel_id
                    and latent.tick > sensor_noise.onset_tick
                ):
                    observed_value = _clip(
                        observed_value + _sensor_noise_offset(seed=scenario.seed, tick=latent.tick)
                    )
                channel_status = (
                    _trip_channel_status(scenario, latent.tick, variable)
                    if trip is not None
                    else _pump_channel_status(latent.tick, variable)
                    if pump is not None
                    else _transfer_channel_status(latent.tick, variable)
                    if transfer is not None
                    else _flow_imbalance_channel_status(latent.tick, variable)
                    if flow_imbalance is not None
                    else _support_power_channel_status(scenario, latent.tick, variable)
                    if support_power is not None
                    else _valve_channel_status(scenario, latent.tick, variable)
                    if valve is not None
                    else _selected_channel_status(scenario, latent.tick, channel.channel_id)
                )
                channel_quality = (
                    ChannelQuality.GOOD
                    if (
                        pump is not None
                        or trip is not None
                        or transfer is not None
                        or flow_imbalance is not None
                        or support_power is not None
                        or valve is not None
                    )
                    else _selected_channel_quality(scenario, latent.tick, channel.channel_id)
                )
                channels.append(
                    SensorChannelObservation(
                        channel_id=channel.channel_id,
                        variable=variable,
                        value=observed_value,
                        quality=channel_quality,
                        status=channel_status,
                    )
                )
        overall_status = (
            _trip_overall_status(latent.tick)
            if trip is not None
            else _pump_overall_status(latent.tick)
            if pump is not None
            else _transfer_overall_status(latent.tick)
            if transfer is not None
            else _flow_imbalance_overall_status(latent.tick)
            if flow_imbalance is not None
            else _support_power_overall_status(latent.tick)
            if support_power is not None
            else _valve_overall_status(scenario, latent.tick)
            if valve is not None
            else ObservationStatus.NORMAL
        )
        if (
            pump is None
            and trip is None
            and transfer is None
            and flow_imbalance is None
            and support_power is None
            and valve is None
            and scenario.fault_injections
        ):
            selected = scenario.fault_injections[0].channel_id or spec.instrumentation_id
            overall_status = _selected_channel_status(scenario, latent.tick, selected)
        frames.append(
            ObservationFrame(
                tick=latent.tick,
                overall_status=overall_status,
                channels=tuple(channels),
            )
        )
    return tuple(frames)


def _event(
    events: list[CanonicalEvent],
    *,
    sim_time: int,
    event_type: EventType,
    subject_id: str,
    **payload: object,
) -> CanonicalEvent:
    event = CanonicalEvent.model_validate(
        {
            "event_id": f"e-{len(events):04d}",
            "event_index": len(events),
            "sim_time": sim_time,
            "event_type": event_type,
            "subject_id": subject_id,
            **payload,
        }
    )
    events.append(event)
    return event


def _first_channel_id(variable: StateVariable, *, spec: AsterVariantSpec = ASTER_A_SPEC) -> str:
    return spec.channel_for(variable, ChannelRole.PRIMARY).channel_id


def _observed_value(
    observations: tuple[ObservationFrame, ...], *, tick: int, variable: StateVariable
) -> float:
    frame = observations[tick]
    value = next(channel.value for channel in frame.channels if channel.variable is variable)
    if value is None:
        raise ValueError("pump process observations must remain available")
    return value


def _pump_trip_events_and_targets(
    scenario: ScenarioDefinition, observations: tuple[ObservationFrame, ...]
) -> tuple[tuple[CanonicalEvent, ...], ScenarioTargets]:
    """Emit the matched G07 trip chain and its context-driven action branch."""

    injection = _pump_trip_injection(scenario)
    context = scenario.standby_context
    if injection is None or context is None:
        raise ValueError("pump-trip event generation requires injection and context")
    standby_available = context.standby_state is ComponentState.AVAILABLE
    context_slot = (
        EvidenceSlot.STANDBY_AVAILABLE if standby_available else EvidenceSlot.COMPONENT_UNAVAILABLE
    )
    events: list[CanonicalEvent] = []
    context_fact = _event(
        events,
        sim_time=0,
        event_type=EventType.BENIGN_NOTE,
        subject_id=context.standby_train_id,
        evidence_slots=(EvidenceSlot.STABLE_OPERATION, context_slot),
    )
    active_unavailable = _event(
        events,
        sim_time=_PUMP_TRIP_ONSET_TICK,
        event_type=EventType.COMPONENT_STATE_CHANGED,
        subject_id=context.active_train_id,
        component_state_before=ComponentState.AVAILABLE,
        component_state_after=ComponentState.UNAVAILABLE,
        evidence_slots=(EvidenceSlot.COMPONENT_UNAVAILABLE,),
        related_event_ids=(context_fact.event_id,),
    )
    _event(
        events,
        sim_time=_PUMP_TRIP_ONSET_TICK,
        event_type=EventType.OPERATING_MODE_CHANGED,
        subject_id=_PUMP_MODE_SUBJECT,
        operating_mode_before=OperatingMode.STABLE,
        operating_mode_after=OperatingMode.DISTURBED,
        related_event_ids=(active_unavailable.event_id,),
    )
    flow = _event(
        events,
        sim_time=_PUMP_TRIP_FLOW_TICK,
        event_type=EventType.OBSERVATION_CHANGED,
        subject_id=_first_channel_id(StateVariable.PRIMARY_FLOW),
        variable=StateVariable.PRIMARY_FLOW,
        value_before=_observed_value(
            observations,
            tick=_PUMP_TRIP_FLOW_TICK - 1,
            variable=StateVariable.PRIMARY_FLOW,
        ),
        value_after=_observed_value(
            observations,
            tick=_PUMP_TRIP_FLOW_TICK,
            variable=StateVariable.PRIMARY_FLOW,
        ),
        observation_status=ObservationStatus.ABNORMAL,
        evidence_slots=(EvidenceSlot.FLOW_DECLINING, EvidenceSlot.MULTIPLE_CHANNELS_AGREE),
        related_event_ids=(active_unavailable.event_id,),
    )
    thermal = _event(
        events,
        sim_time=_PUMP_TRIP_THERMAL_TICK,
        event_type=EventType.OBSERVATION_CHANGED,
        subject_id=_first_channel_id(StateVariable.PRIMARY_THERMAL_STATE),
        variable=StateVariable.PRIMARY_THERMAL_STATE,
        value_before=_observed_value(
            observations,
            tick=_PUMP_TRIP_THERMAL_TICK - 1,
            variable=StateVariable.PRIMARY_THERMAL_STATE,
        ),
        value_after=_observed_value(
            observations,
            tick=_PUMP_TRIP_THERMAL_TICK,
            variable=StateVariable.PRIMARY_THERMAL_STATE,
        ),
        observation_status=ObservationStatus.WATCH,
        evidence_slots=(EvidenceSlot.CORRELATED_STATE_CHANGE,),
        related_event_ids=(flow.event_id,),
    )
    steam = _event(
        events,
        sim_time=_PUMP_TRIP_STEAM_TICK,
        event_type=EventType.OBSERVATION_CHANGED,
        subject_id=_first_channel_id(StateVariable.STEAM_STATE),
        variable=StateVariable.STEAM_STATE,
        value_before=_observed_value(
            observations,
            tick=_PUMP_TRIP_STEAM_TICK - 1,
            variable=StateVariable.STEAM_STATE,
        ),
        value_after=_observed_value(
            observations,
            tick=_PUMP_TRIP_STEAM_TICK,
            variable=StateVariable.STEAM_STATE,
        ),
        observation_status=ObservationStatus.WATCH,
        evidence_slots=(EvidenceSlot.DEPENDENT_TREND_DELAY,),
        related_event_ids=(thermal.event_id,),
    )
    electrical = _event(
        events,
        sim_time=_PUMP_TRIP_STEAM_TICK,
        event_type=EventType.OBSERVATION_CHANGED,
        subject_id=_first_channel_id(StateVariable.ELECTRICAL_OUTPUT),
        variable=StateVariable.ELECTRICAL_OUTPUT,
        value_before=_observed_value(
            observations,
            tick=_PUMP_TRIP_STEAM_TICK - 1,
            variable=StateVariable.ELECTRICAL_OUTPUT,
        ),
        value_after=_observed_value(
            observations,
            tick=_PUMP_TRIP_STEAM_TICK,
            variable=StateVariable.ELECTRICAL_OUTPUT,
        ),
        observation_status=ObservationStatus.WATCH,
        evidence_slots=(
            EvidenceSlot.CORRELATED_STATE_CHANGE,
            EvidenceSlot.DEPENDENT_TREND_DELAY,
        ),
        related_event_ids=(steam.event_id,),
    )
    evidence_slots = (
        *((context_slot,) if context_slot is EvidenceSlot.STANDBY_AVAILABLE else ()),
        EvidenceSlot.COMPONENT_UNAVAILABLE,
        EvidenceSlot.FLOW_DECLINING,
        EvidenceSlot.MULTIPLE_CHANNELS_AGREE,
        EvidenceSlot.CORRELATED_STATE_CHANGE,
        EvidenceSlot.DEPENDENT_TREND_DELAY,
    )
    action = (
        ActionLabel.SELECT_SYNTHETIC_STANDBY_TRAIN
        if standby_available
        else ActionLabel.REDUCE_SIMULATED_LOAD
    )
    evidence_event_ids = (
        context_fact.event_id,
        active_unavailable.event_id,
        flow.event_id,
        thermal.event_id,
        steam.event_id,
        electrical.event_id,
    )
    decision = DecisionTarget(
        scenario_id=scenario.scenario_id,
        decision_tick=_PUMP_TRIP_DECISION_TICK,
        diagnosis_status=DiagnosisStatus.DIAGNOSED,
        fault_labels=(FaultFamily.PUMP_TRIP,),
        evidence_event_ids=evidence_event_ids,
        evidence_slots=evidence_slots,
        immediate_action=action,
    )
    decisions = [decision]
    applied = _event(
        events,
        sim_time=_PUMP_TRIP_ACTION_TICK,
        event_type=EventType.ACTION_APPLIED,
        subject_id=(context.standby_train_id if standby_available else _PUMP_LOAD_SUBJECT),
        action_label=action,
        related_event_ids=(electrical.event_id,),
    )
    if standby_available:
        starting = _event(
            events,
            sim_time=_PUMP_TRIP_ACTION_TICK,
            event_type=EventType.COMPONENT_STATE_CHANGED,
            subject_id=context.standby_train_id,
            component_state_before=ComponentState.AVAILABLE,
            component_state_after=ComponentState.STARTING,
            related_event_ids=(applied.event_id,),
        )
        recovering = _event(
            events,
            sim_time=_PUMP_TRIP_RECOVERY_TICK,
            event_type=EventType.COMPONENT_STATE_CHANGED,
            subject_id=context.standby_train_id,
            component_state_before=ComponentState.STARTING,
            component_state_after=ComponentState.RECOVERING,
            related_event_ids=(starting.event_id,),
        )
        recovery_flow = _event(
            events,
            sim_time=_PUMP_TRIP_RECOVERY_TICK,
            event_type=EventType.OBSERVATION_CHANGED,
            subject_id=_first_channel_id(StateVariable.PRIMARY_FLOW),
            variable=StateVariable.PRIMARY_FLOW,
            value_before=_observed_value(
                observations,
                tick=_PUMP_TRIP_RECOVERY_TICK - 1,
                variable=StateVariable.PRIMARY_FLOW,
            ),
            value_after=_observed_value(
                observations,
                tick=_PUMP_TRIP_RECOVERY_TICK,
                variable=StateVariable.PRIMARY_FLOW,
            ),
            observation_status=ObservationStatus.WATCH,
            evidence_slots=(EvidenceSlot.CORRELATED_STATE_CHANGE,),
            related_event_ids=(recovering.event_id,),
        )
        _event(
            events,
            sim_time=_PUMP_TRIP_RECOVERY_TICK,
            event_type=EventType.OPERATING_MODE_CHANGED,
            subject_id=_PUMP_MODE_SUBJECT,
            operating_mode_before=OperatingMode.DISTURBED,
            operating_mode_after=OperatingMode.RECOVERY,
            related_event_ids=(recovery_flow.event_id,),
        )
    else:
        target = _event(
            events,
            sim_time=_PUMP_TRIP_ACTION_TICK,
            event_type=EventType.TARGET_CHANGED,
            subject_id=_PUMP_LOAD_SUBJECT,
            variable=StateVariable.LOAD_DEMAND,
            value_before=_trip_values(scenario, _PUMP_TRIP_ACTION_TICK - 1).load_demand,
            value_after=_trip_values(scenario, _PUMP_TRIP_ACTION_TICK).load_demand,
            related_event_ids=(applied.event_id,),
        )
        decisions.append(
            DecisionTarget(
                scenario_id=scenario.scenario_id,
                decision_tick=_PUMP_TRIP_ACTION_TICK,
                diagnosis_status=DiagnosisStatus.DIAGNOSED,
                fault_labels=(FaultFamily.PUMP_TRIP,),
                evidence_event_ids=(*evidence_event_ids, target.event_id),
                evidence_slots=evidence_slots,
                immediate_action=ActionLabel.ENTER_SIMULATED_STABLE_STATE,
            )
        )
        stabilize = _event(
            events,
            sim_time=_PUMP_TRIP_RECOVERY_TICK,
            event_type=EventType.ACTION_APPLIED,
            subject_id=_PUMP_MODE_SUBJECT,
            action_label=ActionLabel.ENTER_SIMULATED_STABLE_STATE,
            related_event_ids=(target.event_id,),
        )
        _event(
            events,
            sim_time=_PUMP_TRIP_RECOVERY_TICK,
            event_type=EventType.OPERATING_MODE_CHANGED,
            subject_id=_PUMP_MODE_SUBJECT,
            operating_mode_before=OperatingMode.DISTURBED,
            operating_mode_after=OperatingMode.STABILIZED,
            related_event_ids=(stabilize.event_id,),
        )
    return tuple(events), ScenarioTargets(
        scenario_id=scenario.scenario_id, decisions=tuple(decisions)
    )


def _pump_events_and_targets(
    scenario: ScenarioDefinition, observations: tuple[ObservationFrame, ...]
) -> tuple[tuple[CanonicalEvent, ...], ScenarioTargets]:
    """Emit the fixed, backward-linked visible causal chain for G06."""

    injection = _pump_degradation_injection(scenario)
    if injection is None:
        raise ValueError("pump event generation requires a pump-degradation injection")
    events: list[CanonicalEvent] = []
    stable = _event(
        events,
        sim_time=0,
        event_type=EventType.BENIGN_NOTE,
        subject_id=_INSTRUMENTATION,
        evidence_slots=(EvidenceSlot.STABLE_OPERATION, EvidenceSlot.RELATED_STATE_STABLE),
    )
    component = _event(
        events,
        sim_time=_PUMP_DEGRADATION_ONSET_TICK,
        event_type=EventType.COMPONENT_STATE_CHANGED,
        subject_id=injection.component_id,
        component_state_before=ComponentState.AVAILABLE,
        component_state_after=ComponentState.DEGRADED,
        evidence_slots=(EvidenceSlot.COMPONENT_HEALTH_DECLINING,),
        related_event_ids=(stable.event_id,),
    )
    _event(
        events,
        sim_time=_PUMP_DEGRADATION_ONSET_TICK,
        event_type=EventType.OPERATING_MODE_CHANGED,
        subject_id=_PUMP_MODE_SUBJECT,
        operating_mode_before=OperatingMode.STABLE,
        operating_mode_after=OperatingMode.DISTURBED,
        related_event_ids=(component.event_id,),
    )
    flow = _event(
        events,
        sim_time=_PUMP_FLOW_TICK,
        event_type=EventType.OBSERVATION_CHANGED,
        subject_id=_first_channel_id(StateVariable.PRIMARY_FLOW),
        variable=StateVariable.PRIMARY_FLOW,
        value_before=_observed_value(
            observations, tick=_PUMP_FLOW_TICK - 1, variable=StateVariable.PRIMARY_FLOW
        ),
        value_after=_observed_value(
            observations, tick=_PUMP_FLOW_TICK, variable=StateVariable.PRIMARY_FLOW
        ),
        observation_status=ObservationStatus.WATCH,
        evidence_slots=(EvidenceSlot.FLOW_DECLINING, EvidenceSlot.MULTIPLE_CHANNELS_AGREE),
        related_event_ids=(component.event_id,),
    )
    thermal = _event(
        events,
        sim_time=_PUMP_THERMAL_TICK,
        event_type=EventType.OBSERVATION_CHANGED,
        subject_id=_first_channel_id(StateVariable.PRIMARY_THERMAL_STATE),
        variable=StateVariable.PRIMARY_THERMAL_STATE,
        value_before=_observed_value(
            observations,
            tick=_PUMP_THERMAL_TICK - 1,
            variable=StateVariable.PRIMARY_THERMAL_STATE,
        ),
        value_after=_observed_value(
            observations, tick=_PUMP_THERMAL_TICK, variable=StateVariable.PRIMARY_THERMAL_STATE
        ),
        observation_status=ObservationStatus.WATCH,
        evidence_slots=(EvidenceSlot.CORRELATED_STATE_CHANGE,),
        related_event_ids=(flow.event_id,),
    )
    missing_evidence = _event(
        events,
        sim_time=_PUMP_THERMAL_TICK,
        event_type=EventType.BENIGN_NOTE,
        subject_id=_INSTRUMENTATION,
        evidence_slots=(EvidenceSlot.MISSING_DECISIVE_EVIDENCE,),
        related_event_ids=(thermal.event_id,),
    )
    early_decision = DecisionTarget(
        scenario_id=scenario.scenario_id,
        decision_tick=_PUMP_THERMAL_TICK,
        diagnosis_status=DiagnosisStatus.UNRESOLVED,
        evidence_event_ids=(missing_evidence.event_id,),
        evidence_slots=(EvidenceSlot.MISSING_DECISIVE_EVIDENCE,),
        immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE,
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
    )
    _event(
        events,
        sim_time=_PUMP_STEAM_TICK,
        event_type=EventType.ACTION_APPLIED,
        subject_id=_INSTRUMENTATION,
        action_label=ActionLabel.INSUFFICIENT_EVIDENCE,
        related_event_ids=(missing_evidence.event_id,),
    )
    steam = _event(
        events,
        sim_time=_PUMP_STEAM_TICK,
        event_type=EventType.OBSERVATION_CHANGED,
        subject_id=_first_channel_id(StateVariable.STEAM_STATE),
        variable=StateVariable.STEAM_STATE,
        value_before=_observed_value(
            observations, tick=_PUMP_STEAM_TICK - 1, variable=StateVariable.STEAM_STATE
        ),
        value_after=_observed_value(
            observations, tick=_PUMP_STEAM_TICK, variable=StateVariable.STEAM_STATE
        ),
        observation_status=ObservationStatus.WATCH,
        evidence_slots=(EvidenceSlot.DEPENDENT_TREND_DELAY,),
        related_event_ids=(thermal.event_id,),
    )
    electrical = _event(
        events,
        sim_time=_PUMP_OUTPUT_TICK,
        event_type=EventType.OBSERVATION_CHANGED,
        subject_id=_first_channel_id(StateVariable.ELECTRICAL_OUTPUT),
        variable=StateVariable.ELECTRICAL_OUTPUT,
        value_before=_observed_value(
            observations,
            tick=_PUMP_OUTPUT_TICK - 1,
            variable=StateVariable.ELECTRICAL_OUTPUT,
        ),
        value_after=_observed_value(
            observations, tick=_PUMP_OUTPUT_TICK, variable=StateVariable.ELECTRICAL_OUTPUT
        ),
        observation_status=ObservationStatus.ABNORMAL,
        evidence_slots=(EvidenceSlot.CORRELATED_STATE_CHANGE, EvidenceSlot.DEPENDENT_TREND_DELAY),
        related_event_ids=(steam.event_id,),
    )
    mature_evidence_slots = (
        EvidenceSlot.COMPONENT_HEALTH_DECLINING,
        EvidenceSlot.FLOW_DECLINING,
        EvidenceSlot.MULTIPLE_CHANNELS_AGREE,
        EvidenceSlot.CORRELATED_STATE_CHANGE,
        EvidenceSlot.DEPENDENT_TREND_DELAY,
    )
    mature_decision = DecisionTarget(
        scenario_id=scenario.scenario_id,
        decision_tick=_PUMP_INSPECTION_DECISION_TICK,
        diagnosis_status=DiagnosisStatus.DIAGNOSED,
        fault_labels=(FaultFamily.PUMP_DEGRADATION,),
        evidence_event_ids=(
            component.event_id,
            flow.event_id,
            thermal.event_id,
            steam.event_id,
            electrical.event_id,
        ),
        evidence_slots=mature_evidence_slots,
        immediate_action=ActionLabel.REQUEST_COMPONENT_INSPECTION,
    )
    _event(
        events,
        sim_time=_PUMP_INSPECTION_APPLY_TICK,
        event_type=EventType.ACTION_APPLIED,
        subject_id=injection.component_id,
        action_label=ActionLabel.REQUEST_COMPONENT_INSPECTION,
        related_event_ids=(electrical.event_id,),
    )
    persistent = _event(
        events,
        sim_time=_PUMP_INSPECTION_APPLY_TICK,
        event_type=EventType.BENIGN_NOTE,
        subject_id=_PUMP_MODE_SUBJECT,
        evidence_slots=(EvidenceSlot.CORRELATED_STATE_CHANGE, EvidenceSlot.DEPENDENT_TREND_DELAY),
        related_event_ids=(electrical.event_id,),
    )
    persistent_decision = DecisionTarget(
        scenario_id=scenario.scenario_id,
        decision_tick=_PUMP_LOAD_DECISION_TICK,
        diagnosis_status=DiagnosisStatus.DIAGNOSED,
        fault_labels=(FaultFamily.PUMP_DEGRADATION,),
        evidence_event_ids=(
            component.event_id,
            flow.event_id,
            thermal.event_id,
            steam.event_id,
            electrical.event_id,
            persistent.event_id,
        ),
        evidence_slots=mature_evidence_slots,
        immediate_action=ActionLabel.REDUCE_SIMULATED_LOAD,
    )
    reduce_load = _event(
        events,
        sim_time=_PUMP_LOAD_APPLY_TICK,
        event_type=EventType.ACTION_APPLIED,
        subject_id=_PUMP_LOAD_SUBJECT,
        action_label=ActionLabel.REDUCE_SIMULATED_LOAD,
        related_event_ids=(persistent.event_id,),
    )
    target_load = _event(
        events,
        sim_time=_PUMP_LOAD_APPLY_TICK,
        event_type=EventType.TARGET_CHANGED,
        subject_id=_PUMP_LOAD_SUBJECT,
        variable=StateVariable.LOAD_DEMAND,
        value_before=_pump_values(scenario.seed, _PUMP_LOAD_APPLY_TICK - 1).load_demand,
        value_after=_pump_values(scenario.seed, _PUMP_LOAD_APPLY_TICK).load_demand,
        related_event_ids=(reduce_load.event_id,),
    )
    _event(
        events,
        sim_time=_PUMP_LOAD_APPLY_TICK,
        event_type=EventType.OPERATING_MODE_CHANGED,
        subject_id=_PUMP_MODE_SUBJECT,
        operating_mode_before=OperatingMode.DISTURBED,
        operating_mode_after=OperatingMode.RECOVERY,
        related_event_ids=(target_load.event_id,),
    )
    return tuple(events), ScenarioTargets(
        scenario_id=scenario.scenario_id,
        decisions=(early_decision, mature_decision, persistent_decision),
    )


def _valve_events_and_targets(
    scenario: ScenarioDefinition, observations: tuple[ObservationFrame, ...]
) -> tuple[tuple[CanonicalEvent, ...], ScenarioTargets]:
    """Emit the G08/G09 shared prefix and their minimal decisive contrast."""

    injection = _valve_injection(scenario)
    if injection is None:
        raise ValueError("valve event generation requires a valve injection")
    is_lag = injection.fault_family is FaultFamily.VALVE_LAG
    decisive_tick = _valve_decisive_tick(scenario)
    action_tick = _valve_action_tick(scenario)
    events: list[CanonicalEvent] = []
    initial_position = _valve_initial_position(scenario.seed)
    commanded_position = _valve_commanded_position(scenario.seed)
    stable = _event(
        events,
        sim_time=0,
        event_type=EventType.BENIGN_NOTE,
        subject_id=_INSTRUMENTATION,
        evidence_slots=(EvidenceSlot.STABLE_OPERATION, EvidenceSlot.RELATED_STATE_STABLE),
    )
    command = _event(
        events,
        sim_time=_VALVE_ONSET_TICK,
        event_type=EventType.COMMAND_RECORDED,
        subject_id=injection.component_id,
        variable=StateVariable.PRIMARY_FLOW,
        commanded_value=commanded_position,
        related_event_ids=(stable.event_id,),
    )

    def mismatch(
        *, tick: int, slots: tuple[EvidenceSlot, ...], previous: CanonicalEvent
    ) -> CanonicalEvent:
        return _event(
            events,
            sim_time=tick,
            event_type=EventType.COMMAND_POSITION_MISMATCH,
            subject_id=injection.component_id,
            variable=StateVariable.PRIMARY_FLOW,
            commanded_value=commanded_position,
            observed_value=initial_position,
            evidence_slots=slots,
            related_event_ids=(previous.event_id,),
        )

    first_mismatch = mismatch(
        tick=_VALVE_ONSET_TICK,
        slots=(EvidenceSlot.COMMAND_POSITION_MISMATCH,),
        previous=command,
    )
    early_mismatch = mismatch(
        tick=_VALVE_EARLY_DECISION_TICK,
        slots=(EvidenceSlot.COMMAND_POSITION_MISMATCH,),
        previous=first_mismatch,
    )
    missing = _event(
        events,
        sim_time=_VALVE_EARLY_DECISION_TICK,
        event_type=EventType.BENIGN_NOTE,
        subject_id=_INSTRUMENTATION,
        evidence_slots=(EvidenceSlot.MISSING_DECISIVE_EVIDENCE,),
        related_event_ids=(early_mismatch.event_id,),
    )
    early_decision = DecisionTarget(
        scenario_id=scenario.scenario_id,
        decision_tick=_VALVE_EARLY_DECISION_TICK,
        diagnosis_status=DiagnosisStatus.UNRESOLVED,
        evidence_event_ids=(first_mismatch.event_id, early_mismatch.event_id, missing.event_id),
        evidence_slots=(
            EvidenceSlot.COMMAND_POSITION_MISMATCH,
            EvidenceSlot.MISSING_DECISIVE_EVIDENCE,
        ),
        immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE,
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
    )
    _event(
        events,
        sim_time=_VALVE_EARLY_ACTION_TICK,
        event_type=EventType.ACTION_APPLIED,
        subject_id=_INSTRUMENTATION,
        action_label=ActionLabel.INSUFFICIENT_EVIDENCE,
        related_event_ids=(missing.event_id,),
    )
    applied_mismatch = mismatch(
        tick=_VALVE_EARLY_ACTION_TICK,
        slots=(EvidenceSlot.COMMAND_POSITION_MISMATCH,),
        previous=early_mismatch,
    )
    prior = (
        applied_mismatch
        if decisive_tick - 1 == _VALVE_EARLY_ACTION_TICK
        else mismatch(
            tick=decisive_tick - 1,
            slots=(EvidenceSlot.COMMAND_POSITION_MISMATCH,),
            previous=applied_mismatch,
        )
    )

    if is_lag:
        flow = _event(
            events,
            sim_time=decisive_tick,
            event_type=EventType.OBSERVATION_CHANGED,
            subject_id=_first_channel_id(StateVariable.PRIMARY_FLOW),
            variable=StateVariable.PRIMARY_FLOW,
            value_before=_observed_value(
                observations,
                tick=decisive_tick - 1,
                variable=StateVariable.PRIMARY_FLOW,
            ),
            value_after=_observed_value(
                observations,
                tick=decisive_tick,
                variable=StateVariable.PRIMARY_FLOW,
            ),
            observation_status=ObservationStatus.WATCH,
            evidence_slots=(EvidenceSlot.CORRELATED_STATE_CHANGE,),
            related_event_ids=(prior.event_id,),
        )
        decisive = _event(
            events,
            sim_time=decisive_tick,
            event_type=EventType.COMMAND_POSITION_ALIGNED,
            subject_id=injection.component_id,
            variable=StateVariable.PRIMARY_FLOW,
            commanded_value=commanded_position,
            observed_value=commanded_position,
            evidence_slots=(EvidenceSlot.MISMATCH_RESOLVED,),
            related_event_ids=(flow.event_id,),
        )
        fault = FaultFamily.VALVE_LAG
        action = ActionLabel.CONTINUE_MONITORING
        slots = (
            EvidenceSlot.COMMAND_POSITION_MISMATCH,
            EvidenceSlot.CORRELATED_STATE_CHANGE,
            EvidenceSlot.MISMATCH_RESOLVED,
        )
        evidence = (command.event_id, prior.event_id, flow.event_id, decisive.event_id)
    else:
        decisive = mismatch(
            tick=decisive_tick,
            slots=(
                EvidenceSlot.COMMAND_POSITION_MISMATCH,
                EvidenceSlot.MISMATCH_PERSISTED,
            ),
            previous=prior,
        )
        related_stable = _event(
            events,
            sim_time=_VALVE_DECISIVE_TICK,
            event_type=EventType.BENIGN_NOTE,
            subject_id=_INSTRUMENTATION,
            evidence_slots=(EvidenceSlot.RELATED_STATE_STABLE,),
            related_event_ids=(decisive.event_id,),
        )
        fault = FaultFamily.VALVE_STUCK
        action = ActionLabel.REQUEST_COMPONENT_INSPECTION
        slots = (
            EvidenceSlot.COMMAND_POSITION_MISMATCH,
            EvidenceSlot.MISMATCH_PERSISTED,
            EvidenceSlot.RELATED_STATE_STABLE,
        )
        evidence = (command.event_id, prior.event_id, decisive.event_id, related_stable.event_id)
    mature_decision = DecisionTarget(
        scenario_id=scenario.scenario_id,
        decision_tick=decisive_tick,
        diagnosis_status=DiagnosisStatus.DIAGNOSED,
        fault_labels=(fault,),
        evidence_event_ids=evidence,
        evidence_slots=slots,
        immediate_action=action,
    )
    _event(
        events,
        sim_time=action_tick,
        event_type=EventType.ACTION_APPLIED,
        subject_id=injection.component_id,
        action_label=action,
        related_event_ids=(decisive.event_id,),
    )
    return tuple(events), ScenarioTargets(
        scenario_id=scenario.scenario_id,
        decisions=(early_decision, mature_decision),
    )


def _process_observation_event(
    events: list[CanonicalEvent],
    *,
    observations: tuple[ObservationFrame, ...],
    tick: int,
    variable: StateVariable,
    status: ObservationStatus,
    evidence_slots: tuple[EvidenceSlot, ...],
    related_event_ids: tuple[str, ...],
    spec: AsterVariantSpec,
) -> CanonicalEvent:
    return _event(
        events,
        sim_time=tick,
        event_type=EventType.OBSERVATION_CHANGED,
        subject_id=_first_channel_id(variable, spec=spec),
        variable=variable,
        value_before=_observed_value(observations, tick=tick - 1, variable=variable),
        value_after=_observed_value(observations, tick=tick, variable=variable),
        observation_status=status,
        evidence_slots=evidence_slots,
        related_event_ids=related_event_ids,
    )


def _decision_from_process_evidence(
    *,
    scenario_id: str,
    decision_tick: int,
    events: tuple[CanonicalEvent, ...],
) -> DecisionTarget:
    """Infer only G10/G11 conclusions from a canonical visible-event prefix.

    This deliberately has no access to a scenario, injection, latent state, or
    hidden family hint.  Anything other than a complete, internally canonical
    prefix is treated as insufficient evidence instead of being guessed from.
    """

    if type(scenario_id) is not str or type(decision_tick) is not int or decision_tick < 0:
        raise ValueError("scenario_id and decision_tick must be canonical non-negative inputs")
    if type(events) is not tuple:
        raise TypeError("process evidence must use a tuple of canonical events")
    prior_ids: set[str] = set()
    for index, event in enumerate(events):
        if type(event) is not CanonicalEvent:
            raise TypeError("process evidence must contain canonical events")
        if (
            event.event_index != index
            or event.event_id != f"e-{index:04d}"
            or event.sim_time > decision_tick
            or any(related_id not in prior_ids for related_id in event.related_event_ids)
        ):
            raise ValueError("process evidence must be an ordered visible-event prefix")
        prior_ids.add(event.event_id)

    def observed(
        *, tick: int, variable: StateVariable, slots: tuple[EvidenceSlot, ...]
    ) -> CanonicalEvent | None:
        for event in events:
            if (
                event.sim_time == tick
                and event.event_type is EventType.OBSERVATION_CHANGED
                and event.variable is variable
                and all(slot in event.evidence_slots for slot in slots)
            ):
                return event
        return None

    def note(*, tick: int, slot: EvidenceSlot) -> CanonicalEvent | None:
        return next(
            (
                event
                for event in events
                if event.sim_time == tick
                and event.event_type is EventType.BENIGN_NOTE
                and slot in event.evidence_slots
            ),
            None,
        )

    g10_events = (
        observed(
            tick=_TRANSFER_ONSET_TICK,
            variable=StateVariable.TRANSFER_EFFICIENCY,
            slots=(EvidenceSlot.MULTIPLE_CHANNELS_AGREE,),
        ),
        observed(
            tick=_TRANSFER_THERMAL_TICK,
            variable=StateVariable.PRIMARY_THERMAL_STATE,
            slots=(EvidenceSlot.CORRELATED_STATE_CHANGE,),
        ),
        observed(
            tick=_TRANSFER_STEAM_TICK,
            variable=StateVariable.STEAM_STATE,
            slots=(
                EvidenceSlot.DEPENDENT_TREND_DELAY,
                EvidenceSlot.UPSTREAM_DOWNSTREAM_DIVERGENCE,
            ),
        ),
        observed(
            tick=_TRANSFER_OUTPUT_TICK,
            variable=StateVariable.TURBINE_OUTPUT,
            slots=(EvidenceSlot.DEPENDENT_TREND_DELAY,),
        ),
        observed(
            tick=_TRANSFER_OUTPUT_TICK,
            variable=StateVariable.ELECTRICAL_OUTPUT,
            slots=(
                EvidenceSlot.CORRELATED_STATE_CHANGE,
                EvidenceSlot.DEPENDENT_TREND_DELAY,
            ),
        ),
        note(tick=_TRANSFER_OUTPUT_TICK, slot=EvidenceSlot.RELATED_STATE_STABLE),
    )
    if decision_tick == _TRANSFER_LOAD_DECISION_TICK and all(g10_events):
        return DecisionTarget(
            scenario_id=scenario_id,
            decision_tick=decision_tick,
            diagnosis_status=DiagnosisStatus.DIAGNOSED,
            fault_labels=(FaultFamily.TRANSFER_EFFICIENCY_LOSS,),
            evidence_event_ids=tuple(event.event_id for event in g10_events if event is not None),
            evidence_slots=(
                EvidenceSlot.MULTIPLE_CHANNELS_AGREE,
                EvidenceSlot.CORRELATED_STATE_CHANGE,
                EvidenceSlot.DEPENDENT_TREND_DELAY,
                EvidenceSlot.UPSTREAM_DOWNSTREAM_DIVERGENCE,
                EvidenceSlot.RELATED_STATE_STABLE,
            ),
            immediate_action=ActionLabel.REDUCE_SIMULATED_LOAD,
        )

    g11_initial = (
        observed(
            tick=_FLOW_IMBALANCE_ONSET_TICK,
            variable=StateVariable.SECONDARY_FLOW,
            slots=(EvidenceSlot.SECONDARY_TREND_MISMATCH, EvidenceSlot.MULTIPLE_CHANNELS_AGREE),
        ),
        observed(
            tick=_FLOW_IMBALANCE_INVENTORY_TICK,
            variable=StateVariable.SECONDARY_INVENTORY,
            slots=(
                EvidenceSlot.SECONDARY_TREND_MISMATCH,
                EvidenceSlot.MULTIPLE_CHANNELS_AGREE,
                EvidenceSlot.INVENTORY_TREND_DECLINING,
            ),
        ),
        observed(
            tick=_FLOW_IMBALANCE_COMPARE_TICK,
            variable=StateVariable.SECONDARY_INVENTORY,
            slots=(
                EvidenceSlot.UPSTREAM_DOWNSTREAM_DIVERGENCE,
                EvidenceSlot.SECONDARY_TREND_MISMATCH,
            ),
        ),
        note(tick=_FLOW_IMBALANCE_COMPARE_TICK, slot=EvidenceSlot.RELATED_STATE_STABLE),
    )
    g11_slots = (
        EvidenceSlot.SECONDARY_TREND_MISMATCH,
        EvidenceSlot.MULTIPLE_CHANNELS_AGREE,
        EvidenceSlot.INVENTORY_TREND_DECLINING,
        EvidenceSlot.UPSTREAM_DOWNSTREAM_DIVERGENCE,
        EvidenceSlot.RELATED_STATE_STABLE,
    )
    if decision_tick == _FLOW_IMBALANCE_COMPARE_TICK and all(g11_initial):
        return DecisionTarget(
            scenario_id=scenario_id,
            decision_tick=decision_tick,
            diagnosis_status=DiagnosisStatus.DIAGNOSED,
            fault_labels=(FaultFamily.FLOW_IMBALANCE,),
            evidence_event_ids=tuple(event.event_id for event in g11_initial if event is not None),
            evidence_slots=g11_slots,
            immediate_action=ActionLabel.COMPARE_RELATED_TRENDS,
        )

    g11_mature = (
        *g11_initial,
        observed(
            tick=_FLOW_IMBALANCE_STEAM_TICK,
            variable=StateVariable.STEAM_STATE,
            slots=(EvidenceSlot.DEPENDENT_TREND_DELAY, EvidenceSlot.CORRELATED_STATE_CHANGE),
        ),
        observed(
            tick=_FLOW_IMBALANCE_OUTPUT_TICK,
            variable=StateVariable.TURBINE_OUTPUT,
            slots=(EvidenceSlot.DEPENDENT_TREND_DELAY,),
        ),
        observed(
            tick=_FLOW_IMBALANCE_OUTPUT_TICK,
            variable=StateVariable.ELECTRICAL_OUTPUT,
            slots=(EvidenceSlot.CORRELATED_STATE_CHANGE, EvidenceSlot.DEPENDENT_TREND_DELAY),
        ),
    )
    if decision_tick == _FLOW_IMBALANCE_OUTPUT_TICK and all(g11_mature):
        return DecisionTarget(
            scenario_id=scenario_id,
            decision_tick=decision_tick,
            diagnosis_status=DiagnosisStatus.DIAGNOSED,
            fault_labels=(FaultFamily.FLOW_IMBALANCE,),
            evidence_event_ids=tuple(event.event_id for event in g11_mature if event is not None),
            evidence_slots=(
                *g11_slots,
                EvidenceSlot.DEPENDENT_TREND_DELAY,
                EvidenceSlot.CORRELATED_STATE_CHANGE,
            ),
            immediate_action=ActionLabel.ENTER_SIMULATED_STABLE_STATE,
        )

    return DecisionTarget(
        scenario_id=scenario_id,
        decision_tick=decision_tick,
        diagnosis_status=DiagnosisStatus.UNRESOLVED,
        evidence_slots=(EvidenceSlot.MISSING_DECISIVE_EVIDENCE,),
        immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE,
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
    )


def _transfer_events_and_targets(
    scenario: ScenarioDefinition, observations: tuple[ObservationFrame, ...]
) -> tuple[tuple[CanonicalEvent, ...], ScenarioTargets]:
    """Emit G10's causal transfer-loss chain without exposing its latent label."""

    injection = _transfer_injection(scenario)
    if injection is None:
        raise ValueError("transfer events require a transfer-efficiency-loss injection")
    spec = _spec_for(scenario)
    events: list[CanonicalEvent] = []
    stable = _event(
        events,
        sim_time=0,
        event_type=EventType.BENIGN_NOTE,
        subject_id=spec.instrumentation_id,
        evidence_slots=(EvidenceSlot.STABLE_OPERATION, EvidenceSlot.RELATED_STATE_STABLE),
    )
    efficiency = _process_observation_event(
        events,
        observations=observations,
        tick=_TRANSFER_ONSET_TICK,
        variable=StateVariable.TRANSFER_EFFICIENCY,
        status=ObservationStatus.WATCH,
        evidence_slots=(EvidenceSlot.MULTIPLE_CHANNELS_AGREE,),
        related_event_ids=(stable.event_id,),
        spec=spec,
    )
    _event(
        events,
        sim_time=_TRANSFER_ONSET_TICK,
        event_type=EventType.OPERATING_MODE_CHANGED,
        subject_id=spec.transfer_unit_id,
        operating_mode_before=OperatingMode.STABLE,
        operating_mode_after=OperatingMode.DISTURBED,
        related_event_ids=(efficiency.event_id,),
    )
    thermal = _process_observation_event(
        events,
        observations=observations,
        tick=_TRANSFER_THERMAL_TICK,
        variable=StateVariable.PRIMARY_THERMAL_STATE,
        status=ObservationStatus.WATCH,
        evidence_slots=(EvidenceSlot.CORRELATED_STATE_CHANGE,),
        related_event_ids=(efficiency.event_id,),
        spec=spec,
    )
    missing = _event(
        events,
        sim_time=_TRANSFER_THERMAL_TICK,
        event_type=EventType.BENIGN_NOTE,
        subject_id=spec.instrumentation_id,
        evidence_slots=(EvidenceSlot.MISSING_DECISIVE_EVIDENCE,),
        related_event_ids=(thermal.event_id,),
    )
    early = DecisionTarget(
        scenario_id=scenario.scenario_id,
        decision_tick=_TRANSFER_THERMAL_TICK,
        diagnosis_status=DiagnosisStatus.UNRESOLVED,
        evidence_event_ids=(missing.event_id,),
        evidence_slots=(EvidenceSlot.MISSING_DECISIVE_EVIDENCE,),
        immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE,
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
    )
    abstain = _event(
        events,
        sim_time=_TRANSFER_STEAM_TICK,
        event_type=EventType.ACTION_APPLIED,
        subject_id=spec.instrumentation_id,
        action_label=ActionLabel.INSUFFICIENT_EVIDENCE,
        related_event_ids=(missing.event_id,),
    )
    steam = _process_observation_event(
        events,
        observations=observations,
        tick=_TRANSFER_STEAM_TICK,
        variable=StateVariable.STEAM_STATE,
        status=ObservationStatus.WATCH,
        evidence_slots=(
            EvidenceSlot.DEPENDENT_TREND_DELAY,
            EvidenceSlot.UPSTREAM_DOWNSTREAM_DIVERGENCE,
        ),
        related_event_ids=(thermal.event_id, abstain.event_id),
        spec=spec,
    )
    turbine = _process_observation_event(
        events,
        observations=observations,
        tick=_TRANSFER_OUTPUT_TICK,
        variable=StateVariable.TURBINE_OUTPUT,
        status=ObservationStatus.WATCH,
        evidence_slots=(EvidenceSlot.DEPENDENT_TREND_DELAY,),
        related_event_ids=(steam.event_id,),
        spec=spec,
    )
    electrical = _process_observation_event(
        events,
        observations=observations,
        tick=_TRANSFER_OUTPUT_TICK,
        variable=StateVariable.ELECTRICAL_OUTPUT,
        status=ObservationStatus.WATCH,
        evidence_slots=(EvidenceSlot.CORRELATED_STATE_CHANGE, EvidenceSlot.DEPENDENT_TREND_DELAY),
        related_event_ids=(turbine.event_id,),
        spec=spec,
    )
    preserved = _event(
        events,
        sim_time=_TRANSFER_OUTPUT_TICK,
        event_type=EventType.BENIGN_NOTE,
        subject_id=spec.primary_loop_domain_id,
        evidence_slots=(EvidenceSlot.RELATED_STATE_STABLE, EvidenceSlot.MULTIPLE_CHANNELS_AGREE),
        related_event_ids=(electrical.event_id,),
    )
    mature = _decision_from_process_evidence(
        scenario_id=scenario.scenario_id,
        decision_tick=_TRANSFER_LOAD_DECISION_TICK,
        events=tuple(events),
    )
    reduce_load = _event(
        events,
        sim_time=_TRANSFER_LOAD_APPLY_TICK,
        event_type=EventType.ACTION_APPLIED,
        subject_id=spec.primary_loop_domain_id,
        action_label=ActionLabel.REDUCE_SIMULATED_LOAD,
        related_event_ids=(preserved.event_id,),
    )
    target = _event(
        events,
        sim_time=_TRANSFER_LOAD_APPLY_TICK,
        event_type=EventType.TARGET_CHANGED,
        subject_id=spec.primary_loop_domain_id,
        variable=StateVariable.LOAD_DEMAND,
        value_before=_transfer_values(scenario.seed, _TRANSFER_LOAD_APPLY_TICK - 1).load_demand,
        value_after=_transfer_values(scenario.seed, _TRANSFER_LOAD_APPLY_TICK).load_demand,
        related_event_ids=(reduce_load.event_id,),
    )
    _event(
        events,
        sim_time=_TRANSFER_LOAD_APPLY_TICK,
        event_type=EventType.OPERATING_MODE_CHANGED,
        subject_id=spec.transfer_unit_id,
        operating_mode_before=OperatingMode.DISTURBED,
        operating_mode_after=OperatingMode.RECOVERY,
        related_event_ids=(target.event_id,),
    )
    return tuple(events), ScenarioTargets(
        scenario_id=scenario.scenario_id, decisions=(early, mature)
    )


def _flow_imbalance_events_and_targets(
    scenario: ScenarioDefinition, observations: tuple[ObservationFrame, ...]
) -> tuple[tuple[CanonicalEvent, ...], ScenarioTargets]:
    """Emit G11's secondary mismatch, delayed dependent effects, and two actions."""

    injection = _flow_imbalance_injection(scenario)
    if injection is None:
        raise ValueError("flow-imbalance events require a flow-imbalance injection")
    spec = _spec_for(scenario)
    events: list[CanonicalEvent] = []
    stable = _event(
        events,
        sim_time=0,
        event_type=EventType.BENIGN_NOTE,
        subject_id=spec.instrumentation_id,
        evidence_slots=(EvidenceSlot.STABLE_OPERATION, EvidenceSlot.RELATED_STATE_STABLE),
    )
    flow = _process_observation_event(
        events,
        observations=observations,
        tick=_FLOW_IMBALANCE_ONSET_TICK,
        variable=StateVariable.SECONDARY_FLOW,
        status=ObservationStatus.WATCH,
        evidence_slots=(
            EvidenceSlot.SECONDARY_TREND_MISMATCH,
            EvidenceSlot.MULTIPLE_CHANNELS_AGREE,
        ),
        related_event_ids=(stable.event_id,),
        spec=spec,
    )
    _event(
        events,
        sim_time=_FLOW_IMBALANCE_ONSET_TICK,
        event_type=EventType.OPERATING_MODE_CHANGED,
        subject_id=spec.secondary_feed_id,
        operating_mode_before=OperatingMode.STABLE,
        operating_mode_after=OperatingMode.DISTURBED,
        related_event_ids=(flow.event_id,),
    )
    inventory = _process_observation_event(
        events,
        observations=observations,
        tick=_FLOW_IMBALANCE_INVENTORY_TICK,
        variable=StateVariable.SECONDARY_INVENTORY,
        status=ObservationStatus.WATCH,
        evidence_slots=(
            EvidenceSlot.SECONDARY_TREND_MISMATCH,
            EvidenceSlot.MULTIPLE_CHANNELS_AGREE,
            EvidenceSlot.INVENTORY_TREND_DECLINING,
        ),
        related_event_ids=(flow.event_id,),
        spec=spec,
    )
    persistence = _process_observation_event(
        events,
        observations=observations,
        tick=_FLOW_IMBALANCE_COMPARE_TICK,
        variable=StateVariable.SECONDARY_INVENTORY,
        status=ObservationStatus.WATCH,
        evidence_slots=(
            EvidenceSlot.UPSTREAM_DOWNSTREAM_DIVERGENCE,
            EvidenceSlot.SECONDARY_TREND_MISMATCH,
        ),
        related_event_ids=(inventory.event_id,),
        spec=spec,
    )
    _event(
        events,
        sim_time=_FLOW_IMBALANCE_COMPARE_TICK,
        event_type=EventType.BENIGN_NOTE,
        subject_id=spec.primary_loop_domain_id,
        evidence_slots=(EvidenceSlot.RELATED_STATE_STABLE,),
        related_event_ids=(persistence.event_id,),
    )
    compare = _decision_from_process_evidence(
        scenario_id=scenario.scenario_id,
        decision_tick=_FLOW_IMBALANCE_COMPARE_TICK,
        events=tuple(events),
    )
    applied_compare = _event(
        events,
        sim_time=_FLOW_IMBALANCE_STEAM_TICK,
        event_type=EventType.ACTION_APPLIED,
        subject_id=spec.secondary_feed_id,
        action_label=ActionLabel.COMPARE_RELATED_TRENDS,
        related_event_ids=(persistence.event_id,),
    )
    steam = _process_observation_event(
        events,
        observations=observations,
        tick=_FLOW_IMBALANCE_STEAM_TICK,
        variable=StateVariable.STEAM_STATE,
        status=ObservationStatus.WATCH,
        evidence_slots=(EvidenceSlot.DEPENDENT_TREND_DELAY, EvidenceSlot.CORRELATED_STATE_CHANGE),
        related_event_ids=(applied_compare.event_id,),
        spec=spec,
    )
    turbine = _process_observation_event(
        events,
        observations=observations,
        tick=_FLOW_IMBALANCE_OUTPUT_TICK,
        variable=StateVariable.TURBINE_OUTPUT,
        status=ObservationStatus.WATCH,
        evidence_slots=(EvidenceSlot.DEPENDENT_TREND_DELAY,),
        related_event_ids=(steam.event_id,),
        spec=spec,
    )
    electrical = _process_observation_event(
        events,
        observations=observations,
        tick=_FLOW_IMBALANCE_OUTPUT_TICK,
        variable=StateVariable.ELECTRICAL_OUTPUT,
        status=ObservationStatus.WATCH,
        evidence_slots=(EvidenceSlot.CORRELATED_STATE_CHANGE, EvidenceSlot.DEPENDENT_TREND_DELAY),
        related_event_ids=(turbine.event_id,),
        spec=spec,
    )
    stabilize = _decision_from_process_evidence(
        scenario_id=scenario.scenario_id,
        decision_tick=_FLOW_IMBALANCE_OUTPUT_TICK,
        events=tuple(events),
    )
    applied_stabilize = _event(
        events,
        sim_time=_FLOW_IMBALANCE_STABILIZE_APPLY_TICK,
        event_type=EventType.ACTION_APPLIED,
        subject_id=spec.secondary_feed_id,
        action_label=ActionLabel.ENTER_SIMULATED_STABLE_STATE,
        related_event_ids=(electrical.event_id,),
    )
    target = _event(
        events,
        sim_time=_FLOW_IMBALANCE_STABILIZE_APPLY_TICK,
        event_type=EventType.TARGET_CHANGED,
        subject_id=spec.primary_loop_domain_id,
        variable=StateVariable.LOAD_DEMAND,
        value_before=_flow_imbalance_values(
            scenario.seed, _FLOW_IMBALANCE_STABILIZE_APPLY_TICK - 1
        ).load_demand,
        value_after=_flow_imbalance_values(
            scenario.seed, _FLOW_IMBALANCE_STABILIZE_APPLY_TICK
        ).load_demand,
        related_event_ids=(applied_stabilize.event_id,),
    )
    _event(
        events,
        sim_time=_FLOW_IMBALANCE_STABILIZE_APPLY_TICK,
        event_type=EventType.OPERATING_MODE_CHANGED,
        subject_id=spec.secondary_feed_id,
        operating_mode_before=OperatingMode.DISTURBED,
        operating_mode_after=OperatingMode.STABILIZED,
        related_event_ids=(target.event_id,),
    )
    return tuple(events), ScenarioTargets(
        scenario_id=scenario.scenario_id, decisions=(compare, stabilize)
    )


def _support_power_events_and_targets(
    scenario: ScenarioDefinition, observations: tuple[ObservationFrame, ...]
) -> tuple[tuple[CanonicalEvent, ...], ScenarioTargets]:
    """Emit G12's map-aware bus-to-dependent causal chain."""

    injection = _support_power_injection(scenario)
    if injection is None:
        raise ValueError("support-power events require a support-power injection")
    spec = _spec_for(scenario)
    events: list[CanonicalEvent] = []
    stable = _event(
        events,
        sim_time=0,
        event_type=EventType.BENIGN_NOTE,
        subject_id=spec.instrumentation_id,
        evidence_slots=(EvidenceSlot.STABLE_OPERATION, EvidenceSlot.RELATED_STATE_STABLE),
    )
    bus = _event(
        events,
        sim_time=_SUPPORT_POWER_ONSET_TICK,
        event_type=EventType.COMPONENT_STATE_CHANGED,
        subject_id=injection.component_id,
        component_state_before=ComponentState.AVAILABLE,
        component_state_after=ComponentState.UNAVAILABLE,
        evidence_slots=(EvidenceSlot.SUPPORT_BUS_CHANGE,),
        related_event_ids=(stable.event_id,),
    )
    _event(
        events,
        sim_time=_SUPPORT_POWER_ONSET_TICK,
        event_type=EventType.OPERATING_MODE_CHANGED,
        subject_id=spec.primary_loop_domain_id,
        operating_mode_before=OperatingMode.STABLE,
        operating_mode_after=OperatingMode.DISTURBED,
        related_event_ids=(bus.event_id,),
    )
    support_power = _process_observation_event(
        events,
        observations=observations,
        tick=_SUPPORT_POWER_ONSET_TICK,
        variable=StateVariable.SUPPORT_POWER,
        status=ObservationStatus.WATCH,
        evidence_slots=(EvidenceSlot.SUPPORT_BUS_CHANGE,),
        related_event_ids=(bus.event_id,),
        spec=spec,
    )

    dependency_events: list[CanonicalEvent] = []
    dependency_events_by_role: dict[ComponentRole, list[CanonicalEvent]] = {}
    for dependent_id in sorted(spec.dependents_for(injection.component_id)):
        dependent_role = next(
            component.role
            for component in spec.components
            if component.component_id == dependent_id
        )
        dependent_event = _event(
            events,
            sim_time=_SUPPORT_POWER_COMPONENT_TICK,
            event_type=EventType.COMPONENT_STATE_CHANGED,
            subject_id=dependent_id,
            component_state_before=ComponentState.AVAILABLE,
            component_state_after=ComponentState.UNAVAILABLE,
            evidence_slots=(EvidenceSlot.MAPPED_COMPONENT_CHANGE,),
            related_event_ids=(bus.event_id,),
        )
        dependency_events.append(dependent_event)
        dependency_events_by_role.setdefault(dependent_role, []).append(dependent_event)
    dependency_ids = tuple(event.event_id for event in dependency_events)
    primary_flow = _support_power_roles(spec=spec, bus_id=injection.component_id)
    effect_events: list[CanonicalEvent] = [support_power]
    effect_events_by_role: dict[ComponentRole, CanonicalEvent] = {}
    primary_flow_dependency_ids = tuple(
        event.event_id
        for role in (
            ComponentRole.PRIMARY_TRAIN_ONE,
            ComponentRole.PRIMARY_TRAIN_TWO,
            ComponentRole.PRIMARY_FLOW_VALVE,
        )
        for event in dependency_events_by_role.get(role, ())
    )
    if primary_flow & {
        ComponentRole.PRIMARY_TRAIN_ONE,
        ComponentRole.PRIMARY_TRAIN_TWO,
        ComponentRole.PRIMARY_FLOW_VALVE,
    }:
        effect_events.append(
            _process_observation_event(
                events,
                observations=observations,
                tick=_SUPPORT_POWER_EFFECT_TICK,
                variable=StateVariable.PRIMARY_FLOW,
                status=ObservationStatus.WATCH,
                evidence_slots=(EvidenceSlot.MAPPED_COMPONENT_CHANGE,),
                related_event_ids=primary_flow_dependency_ids,
                spec=spec,
            )
        )
        effect_events_by_role[ComponentRole.PRIMARY_FLOW_VALVE] = effect_events[-1]
        effect_events_by_role[ComponentRole.PRIMARY_TRAIN_ONE] = effect_events[-1]
        effect_events_by_role[ComponentRole.PRIMARY_TRAIN_TWO] = effect_events[-1]
    if ComponentRole.TRANSFER_UNIT in primary_flow:
        effect_events.append(
            _process_observation_event(
                events,
                observations=observations,
                tick=_SUPPORT_POWER_EFFECT_TICK,
                variable=StateVariable.TRANSFER_EFFICIENCY,
                status=ObservationStatus.WATCH,
                evidence_slots=(EvidenceSlot.MAPPED_COMPONENT_CHANGE,),
                related_event_ids=tuple(
                    event.event_id
                    for event in dependency_events_by_role.get(ComponentRole.TRANSFER_UNIT, ())
                ),
                spec=spec,
            )
        )
        effect_events_by_role[ComponentRole.TRANSFER_UNIT] = effect_events[-1]
    if ComponentRole.SECONDARY_FEED in primary_flow:
        effect_events.append(
            _process_observation_event(
                events,
                observations=observations,
                tick=_SUPPORT_POWER_EFFECT_TICK,
                variable=StateVariable.SECONDARY_FLOW,
                status=ObservationStatus.WATCH,
                evidence_slots=(EvidenceSlot.MAPPED_COMPONENT_CHANGE,),
                related_event_ids=tuple(
                    event.event_id
                    for event in dependency_events_by_role.get(ComponentRole.SECONDARY_FEED, ())
                ),
                spec=spec,
            )
        )
        effect_events_by_role[ComponentRole.SECONDARY_FEED] = effect_events[-1]

    delayed_events: list[CanonicalEvent] = []
    effect_ids = tuple(event.event_id for event in effect_events)
    if ComponentRole.TRANSFER_UNIT in primary_flow:
        delayed_events.append(
            _process_observation_event(
                events,
                observations=observations,
                tick=_SUPPORT_POWER_DELAYED_TICK,
                variable=StateVariable.PRIMARY_THERMAL_STATE,
                status=ObservationStatus.WATCH,
                evidence_slots=(EvidenceSlot.DEPENDENT_TREND_DELAY,),
                related_event_ids=(effect_events_by_role[ComponentRole.TRANSFER_UNIT].event_id,),
                spec=spec,
            )
        )
        delayed_events.append(
            _process_observation_event(
                events,
                observations=observations,
                tick=_SUPPORT_POWER_DELAYED_TICK,
                variable=StateVariable.STEAM_STATE,
                status=ObservationStatus.WATCH,
                evidence_slots=(EvidenceSlot.DEPENDENT_TREND_DELAY,),
                related_event_ids=(effect_events_by_role[ComponentRole.TRANSFER_UNIT].event_id,),
                spec=spec,
            )
        )
    if ComponentRole.SECONDARY_FEED in primary_flow:
        delayed_events.append(
            _process_observation_event(
                events,
                observations=observations,
                tick=_SUPPORT_POWER_DELAYED_TICK,
                variable=StateVariable.SECONDARY_INVENTORY,
                status=ObservationStatus.WATCH,
                evidence_slots=(EvidenceSlot.DEPENDENT_TREND_DELAY,),
                related_event_ids=(effect_events_by_role[ComponentRole.SECONDARY_FEED].event_id,),
                spec=spec,
            )
        )
    delayed_ids = tuple(event.event_id for event in delayed_events)
    physical_evidence_ids = (bus.event_id, *dependency_ids, *effect_ids, *delayed_ids)
    physical_evidence_slots = (
        EvidenceSlot.SUPPORT_BUS_CHANGE,
        EvidenceSlot.MAPPED_COMPONENT_CHANGE,
        EvidenceSlot.DEPENDENT_TREND_DELAY,
    )
    if scenario.dependency_map_context is not None:
        decision = DecisionTarget(
            scenario_id=scenario.scenario_id,
            decision_tick=_SUPPORT_POWER_DECISION_TICK,
            diagnosis_status=DiagnosisStatus.DIAGNOSED,
            fault_labels=(FaultFamily.SUPPORT_POWER_INTERRUPTION,),
            evidence_event_ids=physical_evidence_ids,
            evidence_slots=physical_evidence_slots,
            immediate_action=ActionLabel.ENTER_SIMULATED_STABLE_STATE,
        )
        applied = _event(
            events,
            sim_time=_SUPPORT_POWER_ACTION_TICK,
            event_type=EventType.ACTION_APPLIED,
            subject_id=spec.primary_loop_domain_id,
            action_label=ActionLabel.ENTER_SIMULATED_STABLE_STATE,
            related_event_ids=physical_evidence_ids,
        )
        target = _event(
            events,
            sim_time=_SUPPORT_POWER_ACTION_TICK,
            event_type=EventType.TARGET_CHANGED,
            subject_id=spec.primary_loop_domain_id,
            variable=StateVariable.LOAD_DEMAND,
            value_before=_support_power_values(
                scenario=scenario, tick=_SUPPORT_POWER_ACTION_TICK - 1, spec=spec
            ).load_demand,
            value_after=_support_power_values(
                scenario=scenario, tick=_SUPPORT_POWER_ACTION_TICK, spec=spec
            ).load_demand,
            related_event_ids=(applied.event_id,),
        )
        recovery = _event(
            events,
            sim_time=_SUPPORT_POWER_ACTION_TICK,
            event_type=EventType.OPERATING_MODE_CHANGED,
            subject_id=spec.primary_loop_domain_id,
            operating_mode_before=OperatingMode.DISTURBED,
            operating_mode_after=OperatingMode.RECOVERY,
            related_event_ids=(target.event_id,),
        )
        _event(
            events,
            sim_time=_SUPPORT_POWER_STABILIZED_TICK,
            event_type=EventType.OPERATING_MODE_CHANGED,
            subject_id=spec.primary_loop_domain_id,
            operating_mode_before=OperatingMode.RECOVERY,
            operating_mode_after=OperatingMode.STABILIZED,
            related_event_ids=(recovery.event_id,),
        )
    else:
        decision = DecisionTarget(
            scenario_id=scenario.scenario_id,
            decision_tick=_SUPPORT_POWER_DECISION_TICK,
            diagnosis_status=DiagnosisStatus.UNRESOLVED,
            evidence_event_ids=physical_evidence_ids,
            evidence_slots=physical_evidence_slots,
            immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE,
            abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
        )
        _event(
            events,
            sim_time=_SUPPORT_POWER_ACTION_TICK,
            event_type=EventType.ACTION_APPLIED,
            subject_id=spec.instrumentation_id,
            action_label=ActionLabel.INSUFFICIENT_EVIDENCE,
            related_event_ids=physical_evidence_ids,
        )
    return tuple(events), ScenarioTargets(scenario_id=scenario.scenario_id, decisions=(decision,))


def _events_and_targets(
    scenario: ScenarioDefinition, observations: tuple[ObservationFrame, ...]
) -> tuple[tuple[CanonicalEvent, ...], ScenarioTargets]:
    spec = _spec_for(scenario)
    events: list[CanonicalEvent] = []
    stable = _event(
        events,
        sim_time=0,
        event_type=EventType.BENIGN_NOTE,
        subject_id=spec.instrumentation_id,
        evidence_slots=(EvidenceSlot.STABLE_OPERATION, EvidenceSlot.RELATED_STATE_STABLE),
    )
    if scenario.driver is ScenarioDriver.STEADY_OPERATION and not scenario.fault_injections:
        return (
            tuple(events),
            ScenarioTargets(
                scenario_id=scenario.scenario_id,
                decisions=(
                    DecisionTarget(
                        scenario_id=scenario.scenario_id,
                        decision_tick=scenario.duration_ticks - 1,
                        diagnosis_status=DiagnosisStatus.NO_FAULT,
                        evidence_event_ids=(stable.event_id,),
                        evidence_slots=(EvidenceSlot.STABLE_OPERATION,),
                        immediate_action=ActionLabel.CONTINUE_MONITORING,
                    ),
                ),
            ),
        )

    trip = _pump_trip_injection(scenario)
    if scenario.driver is ScenarioDriver.STEADY_OPERATION and trip is not None:
        return _pump_trip_events_and_targets(scenario, observations)

    pump = _pump_degradation_injection(scenario)
    if scenario.driver is ScenarioDriver.STEADY_OPERATION and pump is not None:
        return _pump_events_and_targets(scenario, observations)

    transfer = _transfer_injection(scenario)
    if scenario.driver is ScenarioDriver.STEADY_OPERATION and transfer is not None:
        return _transfer_events_and_targets(scenario, observations)

    flow_imbalance = _flow_imbalance_injection(scenario)
    if scenario.driver is ScenarioDriver.STEADY_OPERATION and flow_imbalance is not None:
        return _flow_imbalance_events_and_targets(scenario, observations)

    support_power = _support_power_injection(scenario)
    if scenario.driver is ScenarioDriver.STEADY_OPERATION and support_power is not None:
        return _support_power_events_and_targets(scenario, observations)

    valve = _valve_injection(scenario)
    if scenario.driver is ScenarioDriver.STEADY_OPERATION and valve is not None:
        return _valve_events_and_targets(scenario, observations)

    sensor_noise = _sensor_noise_injection(scenario)
    if scenario.driver is ScenarioDriver.STEADY_OPERATION and sensor_noise is not None:
        selected_channel = sensor_noise.channel_id or _INSTRUMENTATION
        observed = {
            frame.tick: next(
                channel.value
                for channel in frame.channels
                if channel.channel_id == selected_channel
            )
            for frame in observations
        }
        first = _event(
            events,
            sim_time=_SENSOR_NOISE_FIRST_DECISION_TICK,
            event_type=EventType.OBSERVATION_CHANGED,
            subject_id=selected_channel,
            variable=StateVariable.PRIMARY_THERMAL_STATE,
            value_before=observed[_SENSOR_NOISE_ONSET_TICK],
            value_after=observed[_SENSOR_NOISE_FIRST_DECISION_TICK],
            observation_status=ObservationStatus.WATCH,
            evidence_slots=(EvidenceSlot.MISSING_DECISIVE_EVIDENCE,),
            related_event_ids=(stable.event_id,),
        )
        _event(
            events,
            sim_time=_SENSOR_NOISE_SECOND_DECISION_TICK,
            event_type=EventType.ACTION_APPLIED,
            subject_id=_INSTRUMENTATION,
            action_label=ActionLabel.INSUFFICIENT_EVIDENCE,
            related_event_ids=(first.event_id,),
        )
        second_observation = _event(
            events,
            sim_time=_SENSOR_NOISE_SECOND_DECISION_TICK,
            event_type=EventType.OBSERVATION_CHANGED,
            subject_id=selected_channel,
            variable=StateVariable.PRIMARY_THERMAL_STATE,
            value_before=observed[_SENSOR_NOISE_FIRST_DECISION_TICK],
            value_after=observed[_SENSOR_NOISE_SECOND_DECISION_TICK],
            observation_status=ObservationStatus.WATCH,
            evidence_slots=(EvidenceSlot.MISSING_DECISIVE_EVIDENCE,),
            related_event_ids=(first.event_id,),
        )
        _event(
            events,
            sim_time=_SENSOR_NOISE_DIAGNOSIS_TICK,
            event_type=EventType.ACTION_APPLIED,
            subject_id=_INSTRUMENTATION,
            action_label=ActionLabel.INSUFFICIENT_EVIDENCE,
            related_event_ids=(second_observation.event_id,),
        )
        alternating = _event(
            events,
            sim_time=_SENSOR_NOISE_DIAGNOSIS_TICK,
            event_type=EventType.OBSERVATION_CHANGED,
            subject_id=selected_channel,
            variable=StateVariable.PRIMARY_THERMAL_STATE,
            value_before=observed[_SENSOR_NOISE_SECOND_DECISION_TICK],
            value_after=observed[_SENSOR_NOISE_DIAGNOSIS_TICK],
            observation_status=ObservationStatus.CONFLICTING,
            evidence_slots=(EvidenceSlot.RAPID_INCONSISTENT_READINGS,),
            related_event_ids=(second_observation.event_id,),
        )
        disagreement = _event(
            events,
            sim_time=_SENSOR_NOISE_DIAGNOSIS_TICK,
            event_type=EventType.CHANNEL_DISAGREEMENT,
            subject_id=selected_channel,
            variable=StateVariable.PRIMARY_THERMAL_STATE,
            observation_status=ObservationStatus.CONFLICTING,
            evidence_slots=(
                EvidenceSlot.CHANNEL_DISAGREEMENT,
                EvidenceSlot.RAPID_INCONSISTENT_READINGS,
            ),
            related_event_ids=(first.event_id, alternating.event_id),
        )
        _event(
            events,
            sim_time=_SENSOR_NOISE_FLAG_TICK,
            event_type=EventType.ACTION_APPLIED,
            subject_id=_INSTRUMENTATION,
            action_label=ActionLabel.COMPARE_RELATED_TRENDS,
            related_event_ids=(disagreement.event_id,),
        )
        related = _event(
            events,
            sim_time=_SENSOR_NOISE_FLAG_TICK,
            event_type=EventType.BENIGN_NOTE,
            subject_id=_INSTRUMENTATION,
            evidence_slots=(EvidenceSlot.RELATED_STATE_STABLE,),
            related_event_ids=(stable.event_id,),
        )
        flag = _event(
            events,
            sim_time=_SENSOR_NOISE_FLAG_APPLY_TICK,
            event_type=EventType.ACTION_APPLIED,
            subject_id=_INSTRUMENTATION,
            action_label=ActionLabel.FLAG_SENSOR_SUSPECT,
            related_event_ids=(related.event_id,),
        )
        _event(
            events,
            sim_time=_SENSOR_NOISE_FLAG_APPLY_TICK,
            event_type=EventType.CHANNEL_QUALITY_CHANGED,
            subject_id=selected_channel,
            channel_quality_before=ChannelQuality.GOOD,
            channel_quality=ChannelQuality.SUSPECT,
            related_event_ids=(flag.event_id,),
        )
        noise_decisions: tuple[DecisionTarget, ...] = (
            DecisionTarget(
                scenario_id=scenario.scenario_id,
                decision_tick=_SENSOR_NOISE_FIRST_DECISION_TICK,
                diagnosis_status=DiagnosisStatus.UNRESOLVED,
                evidence_event_ids=(first.event_id,),
                evidence_slots=(EvidenceSlot.MISSING_DECISIVE_EVIDENCE,),
                immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE,
                abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
            ),
            DecisionTarget(
                scenario_id=scenario.scenario_id,
                decision_tick=_SENSOR_NOISE_SECOND_DECISION_TICK,
                diagnosis_status=DiagnosisStatus.UNRESOLVED,
                evidence_event_ids=(first.event_id, second_observation.event_id),
                evidence_slots=(EvidenceSlot.MISSING_DECISIVE_EVIDENCE,),
                immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE,
                abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
            ),
            DecisionTarget(
                scenario_id=scenario.scenario_id,
                decision_tick=_SENSOR_NOISE_DIAGNOSIS_TICK,
                diagnosis_status=DiagnosisStatus.DIAGNOSED,
                fault_labels=(FaultFamily.SENSOR_NOISE,),
                evidence_event_ids=(disagreement.event_id,),
                evidence_slots=(
                    EvidenceSlot.CHANNEL_DISAGREEMENT,
                    EvidenceSlot.RAPID_INCONSISTENT_READINGS,
                ),
                immediate_action=ActionLabel.COMPARE_RELATED_TRENDS,
            ),
            DecisionTarget(
                scenario_id=scenario.scenario_id,
                decision_tick=_SENSOR_NOISE_FLAG_TICK,
                diagnosis_status=DiagnosisStatus.DIAGNOSED,
                fault_labels=(FaultFamily.SENSOR_NOISE,),
                evidence_event_ids=(disagreement.event_id, related.event_id),
                evidence_slots=(
                    EvidenceSlot.CHANNEL_DISAGREEMENT,
                    EvidenceSlot.RAPID_INCONSISTENT_READINGS,
                    EvidenceSlot.RELATED_STATE_STABLE,
                ),
                immediate_action=ActionLabel.FLAG_SENSOR_SUSPECT,
            ),
        )
        return tuple(events), ScenarioTargets(
            scenario_id=scenario.scenario_id, decisions=noise_decisions
        )

    stuck = _sensor_stuck_injection(scenario)
    if scenario.driver is ScenarioDriver.LOAD_TRANSIENT and stuck is not None:
        selected_channel = stuck.channel_id or _INSTRUMENTATION
        baseline_load = _baseline_values(scenario.seed).load_demand
        target_load = _clip(
            baseline_load + _load_direction(scenario.seed) * _load_magnitude(scenario.seed)
        )
        target = _event(
            events,
            sim_time=_LOAD_ONSET_TICK,
            event_type=EventType.TARGET_CHANGED,
            subject_id="aster-load-domain",
            variable=StateVariable.LOAD_DEMAND,
            value_before=baseline_load,
            value_after=target_load,
            related_event_ids=(stable.event_id,),
        )
        _event(
            events,
            sim_time=_LOAD_ONSET_TICK,
            event_type=EventType.OPERATING_MODE_CHANGED,
            subject_id="aster-load-domain",
            operating_mode_before=OperatingMode.STABLE,
            operating_mode_after=OperatingMode.LOAD_CHANGE,
            related_event_ids=(target.event_id,),
        )
        redundant_channel = next(
            channel.channel_id
            for channel in ASTER_A_SPEC.channels
            if channel.variable is StateVariable.ELECTRICAL_OUTPUT
            and channel.channel_id != selected_channel
        )
        observed_electrical_output = {
            frame.tick: next(
                channel.value
                for channel in frame.channels
                if channel.channel_id == redundant_channel
            )
            for frame in observations
        }
        correlated = _event(
            events,
            sim_time=_STUCK_VERIFY_TICK,
            event_type=EventType.OBSERVATION_CHANGED,
            subject_id=redundant_channel,
            variable=StateVariable.ELECTRICAL_OUTPUT,
            value_before=observed_electrical_output[_LOAD_ONSET_TICK - 1],
            value_after=observed_electrical_output[_STUCK_VERIFY_TICK],
            observation_status=ObservationStatus.NORMAL,
            evidence_slots=(EvidenceSlot.CORRELATED_STATE_CHANGE,),
            related_event_ids=(target.event_id,),
        )
        disagreement = _event(
            events,
            sim_time=_STUCK_VERIFY_TICK,
            event_type=EventType.CHANNEL_DISAGREEMENT,
            subject_id=selected_channel,
            variable=StateVariable.ELECTRICAL_OUTPUT,
            observation_status=ObservationStatus.CONFLICTING,
            evidence_slots=(
                EvidenceSlot.CHANNEL_DISAGREEMENT,
                EvidenceSlot.CHANNEL_FROZEN,
            ),
            related_event_ids=(target.event_id, correlated.event_id),
        )
        _event(
            events,
            sim_time=_STUCK_FLAG_TICK,
            event_type=EventType.ACTION_APPLIED,
            subject_id=_INSTRUMENTATION,
            action_label=ActionLabel.VERIFY_REDUNDANT_CHANNEL,
            related_event_ids=(disagreement.event_id,),
        )
        coordinated = _event(
            events,
            sim_time=_STUCK_FLAG_TICK,
            event_type=EventType.BENIGN_NOTE,
            subject_id="aster-load-domain",
            evidence_slots=(EvidenceSlot.COORDINATED_LOAD_RESPONSE,),
            related_event_ids=(target.event_id,),
        )
        flag = _event(
            events,
            sim_time=_STUCK_FLAG_APPLY_TICK,
            event_type=EventType.ACTION_APPLIED,
            subject_id=_INSTRUMENTATION,
            action_label=ActionLabel.FLAG_SENSOR_SUSPECT,
            related_event_ids=(disagreement.event_id,),
        )
        _event(
            events,
            sim_time=_LOAD_SETTLE_TICK,
            event_type=EventType.OPERATING_MODE_CHANGED,
            subject_id="aster-load-domain",
            operating_mode_before=OperatingMode.LOAD_CHANGE,
            operating_mode_after=OperatingMode.STABLE,
            related_event_ids=(coordinated.event_id,),
        )
        _event(
            events,
            sim_time=_STUCK_FLAG_APPLY_TICK,
            event_type=EventType.CHANNEL_QUALITY_CHANGED,
            subject_id=selected_channel,
            channel_quality_before=ChannelQuality.GOOD,
            channel_quality=ChannelQuality.SUSPECT,
            related_event_ids=(flag.event_id,),
        )
        decisions: tuple[DecisionTarget, ...] = (
            DecisionTarget(
                scenario_id=scenario.scenario_id,
                decision_tick=_STUCK_VERIFY_TICK,
                diagnosis_status=DiagnosisStatus.DIAGNOSED,
                fault_labels=(FaultFamily.SENSOR_STUCK,),
                evidence_event_ids=(correlated.event_id, disagreement.event_id),
                evidence_slots=(
                    EvidenceSlot.CHANNEL_FROZEN,
                    EvidenceSlot.CORRELATED_STATE_CHANGE,
                    EvidenceSlot.CHANNEL_DISAGREEMENT,
                ),
                immediate_action=ActionLabel.VERIFY_REDUNDANT_CHANNEL,
            ),
            DecisionTarget(
                scenario_id=scenario.scenario_id,
                decision_tick=_STUCK_FLAG_TICK,
                diagnosis_status=DiagnosisStatus.DIAGNOSED,
                fault_labels=(FaultFamily.SENSOR_STUCK,),
                evidence_event_ids=(correlated.event_id, disagreement.event_id),
                evidence_slots=(
                    EvidenceSlot.CHANNEL_FROZEN,
                    EvidenceSlot.CORRELATED_STATE_CHANGE,
                    EvidenceSlot.CHANNEL_DISAGREEMENT,
                ),
                immediate_action=ActionLabel.FLAG_SENSOR_SUSPECT,
            ),
        )
        return tuple(events), ScenarioTargets(scenario_id=scenario.scenario_id, decisions=decisions)

    if scenario.driver is ScenarioDriver.LOAD_TRANSIENT:
        baseline_load = _baseline_values(scenario.seed).load_demand
        target_load = _clip(
            baseline_load + _load_direction(scenario.seed) * _load_magnitude(scenario.seed)
        )
        target = _event(
            events,
            sim_time=_LOAD_ONSET_TICK,
            event_type=EventType.TARGET_CHANGED,
            subject_id="aster-load-domain",
            variable=StateVariable.LOAD_DEMAND,
            value_before=baseline_load,
            value_after=target_load,
            related_event_ids=(stable.event_id,),
        )
        _event(
            events,
            sim_time=_LOAD_ONSET_TICK,
            event_type=EventType.OPERATING_MODE_CHANGED,
            subject_id="aster-load-domain",
            operating_mode_before=OperatingMode.STABLE,
            operating_mode_after=OperatingMode.LOAD_CHANGE,
            related_event_ids=(target.event_id,),
        )
        coordinated = _event(
            events,
            sim_time=_LOAD_SETTLE_TICK - 1,
            event_type=EventType.BENIGN_NOTE,
            subject_id="aster-load-domain",
            evidence_slots=(EvidenceSlot.COORDINATED_LOAD_RESPONSE,),
            related_event_ids=(target.event_id,),
        )
        _event(
            events,
            sim_time=_LOAD_SETTLE_TICK,
            event_type=EventType.OPERATING_MODE_CHANGED,
            subject_id="aster-load-domain",
            operating_mode_before=OperatingMode.LOAD_CHANGE,
            operating_mode_after=OperatingMode.STABLE,
            related_event_ids=(coordinated.event_id,),
        )
        return (
            tuple(events),
            ScenarioTargets(
                scenario_id=scenario.scenario_id,
                decisions=(
                    DecisionTarget(
                        scenario_id=scenario.scenario_id,
                        decision_tick=scenario.duration_ticks - 1,
                        diagnosis_status=DiagnosisStatus.NO_FAULT,
                        evidence_event_ids=(stable.event_id, coordinated.event_id),
                        evidence_slots=(
                            EvidenceSlot.STABLE_OPERATION,
                            EvidenceSlot.COORDINATED_LOAD_RESPONSE,
                        ),
                        immediate_action=ActionLabel.CONTINUE_MONITORING,
                    ),
                ),
            ),
        )

    injection = scenario.fault_injections[0]
    early_tick = injection.onset_tick + 1
    mature_tick = injection.onset_tick + 2
    flag_tick = injection.onset_tick + 3
    selected_channel = injection.channel_id or _INSTRUMENTATION
    observed_by_tick = {
        frame.tick: next(
            channel for channel in frame.channels if channel.channel_id == selected_channel
        )
        for frame in observations
    }
    early = _event(
        events,
        sim_time=early_tick,
        event_type=EventType.OBSERVATION_CHANGED,
        subject_id=selected_channel,
        variable=StateVariable.PRIMARY_FLOW,
        value_before=observed_by_tick[injection.onset_tick].value,
        value_after=observed_by_tick[early_tick].value,
        observation_status=ObservationStatus.WATCH,
        evidence_slots=(EvidenceSlot.MISSING_DECISIVE_EVIDENCE,),
        related_event_ids=(stable.event_id,),
    )
    _event(
        events,
        sim_time=mature_tick,
        event_type=EventType.ACTION_APPLIED,
        subject_id=_INSTRUMENTATION,
        action_label=ActionLabel.INSUFFICIENT_EVIDENCE,
        related_event_ids=(early.event_id,),
    )
    mature = _event(
        events,
        sim_time=mature_tick,
        event_type=EventType.CHANNEL_DISAGREEMENT,
        subject_id=selected_channel,
        variable=StateVariable.PRIMARY_FLOW,
        observation_status=ObservationStatus.CONFLICTING,
        evidence_slots=(EvidenceSlot.CHANNEL_DISAGREEMENT,),
        related_event_ids=(stable.event_id, early.event_id),
    )
    _event(
        events,
        sim_time=flag_tick,
        event_type=EventType.ACTION_APPLIED,
        subject_id=_INSTRUMENTATION,
        action_label=ActionLabel.VERIFY_REDUNDANT_CHANNEL,
        related_event_ids=(mature.event_id,),
    )
    flag = _event(
        events,
        sim_time=flag_tick + 1,
        event_type=EventType.ACTION_APPLIED,
        subject_id=_INSTRUMENTATION,
        action_label=ActionLabel.FLAG_SENSOR_SUSPECT,
        related_event_ids=(mature.event_id,),
    )
    _event(
        events,
        sim_time=flag_tick + 1,
        event_type=EventType.CHANNEL_QUALITY_CHANGED,
        subject_id=selected_channel,
        channel_quality_before=ChannelQuality.GOOD,
        channel_quality=ChannelQuality.SUSPECT,
        related_event_ids=(flag.event_id,),
    )
    decisions = (
        DecisionTarget(
            scenario_id=scenario.scenario_id,
            decision_tick=early_tick,
            diagnosis_status=DiagnosisStatus.UNRESOLVED,
            evidence_event_ids=(early.event_id,),
            evidence_slots=(EvidenceSlot.MISSING_DECISIVE_EVIDENCE,),
            immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE,
            abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
        ),
        DecisionTarget(
            scenario_id=scenario.scenario_id,
            decision_tick=mature_tick,
            diagnosis_status=DiagnosisStatus.DIAGNOSED,
            fault_labels=(FaultFamily.SENSOR_DRIFT,),
            evidence_event_ids=(stable.event_id, mature.event_id),
            evidence_slots=(EvidenceSlot.CHANNEL_DISAGREEMENT, EvidenceSlot.RELATED_STATE_STABLE),
            immediate_action=ActionLabel.VERIFY_REDUNDANT_CHANNEL,
        ),
        DecisionTarget(
            scenario_id=scenario.scenario_id,
            decision_tick=flag_tick,
            diagnosis_status=DiagnosisStatus.DIAGNOSED,
            fault_labels=(FaultFamily.SENSOR_DRIFT,),
            evidence_event_ids=(stable.event_id, mature.event_id),
            evidence_slots=(EvidenceSlot.CHANNEL_DISAGREEMENT, EvidenceSlot.RELATED_STATE_STABLE),
            immediate_action=ActionLabel.FLAG_SENSOR_SUSPECT,
        ),
    )
    return tuple(events), ScenarioTargets(scenario_id=scenario.scenario_id, decisions=decisions)


def _validate_supported_scenario(scenario: ScenarioDefinition) -> None:
    if type(scenario) is not ScenarioDefinition:
        raise UnsupportedScenarioError("scenario must use the canonical contract")
    if scenario.schema_version != SCHEMA_VERSION:
        raise UnsupportedScenarioError("unsupported scenario schema version")
    if type(scenario.plant_variant_id) is not PlantVariant:
        raise UnsupportedScenarioError("plant variant must use the canonical enum")
    try:
        spec = get_variant_spec(scenario.plant_variant_id)
    except (KeyError, TypeError) as error:
        raise UnsupportedScenarioError("plant variant is not registered") from error
    if type(scenario.driver) is not ScenarioDriver:
        raise UnsupportedScenarioError("scenario driver must use the canonical enum")
    if scenario.driver not in {ScenarioDriver.STEADY_OPERATION, ScenarioDriver.LOAD_TRANSIENT}:
        raise UnsupportedScenarioError("unsupported scenario driver")
    try:
        _require_uint32(scenario.seed, name="seed")
        _require_duration(scenario.duration_ticks)
    except ValueError as error:
        raise UnsupportedScenarioError(str(error)) from error
    if type(scenario.fault_injections) is not tuple:
        raise UnsupportedScenarioError("fault_injections must use a tuple container")
    if len(scenario.fault_injections) > 1:
        raise UnsupportedScenarioError("only zero or one injection is supported")
    if any(type(injection) is not FaultInjection for injection in scenario.fault_injections):
        raise UnsupportedScenarioError("fault injection must use the canonical contract")
    if type(scenario.action_sequence) is not tuple:
        raise UnsupportedScenarioError("action_sequence must use a tuple container")
    is_pump_trip = (
        len(scenario.fault_injections) == 1
        and scenario.fault_injections[0].fault_family is FaultFamily.PUMP_TRIP
    )
    is_support_power = (
        len(scenario.fault_injections) == 1
        and scenario.fault_injections[0].fault_family is FaultFamily.SUPPORT_POWER_INTERRUPTION
    )
    if scenario.dependency_map_context is not None:
        if type(scenario.dependency_map_context) is not DependencyMapContext:
            raise UnsupportedScenarioError("dependency map context must use the canonical contract")
        if not is_support_power:
            raise UnsupportedScenarioError("dependency map context is only supported for G12")
    for action in scenario.action_sequence:
        if type(action) is not ScenarioAction:
            raise UnsupportedScenarioError("action sequence must use the canonical contract")
        if type(action.action) is not ActionLabel:
            raise UnsupportedScenarioError("scenario action must use the canonical enum")
        try:
            _require_uint32(action.decision_tick, name="action decision tick")
        except ValueError as error:
            raise UnsupportedScenarioError(str(error)) from error
    for injection in scenario.fault_injections:
        if type(injection.fault_family) is not FaultFamily:
            raise UnsupportedScenarioError("fault family must use the canonical enum")
        if type(injection.severity) is not SeverityBand:
            raise UnsupportedScenarioError("fault severity must use the canonical enum")
        try:
            _require_uint32(injection.onset_tick, name="fault onset")
            if injection.duration_ticks is not None:
                _require_uint32(injection.duration_ticks, name="fault duration")
        except ValueError as error:
            raise UnsupportedScenarioError(str(error)) from error
    if is_pump_trip:
        context = scenario.standby_context
        if type(context) is not StandbyContext:
            raise UnsupportedScenarioError("pump trip requires canonical standby context")
        if (
            type(context.context_id) is not str
            or type(context.active_train_id) is not str
            or type(context.standby_train_id) is not str
            or type(context.standby_support_bus_id) is not str
            or type(context.standby_state) is not ComponentState
            or type(context.support_bus_state) is not ComponentState
            or isinstance(context.standby_start_delay_ticks, bool)
            or type(context.standby_start_delay_ticks) is not int
        ):
            raise UnsupportedScenarioError("pump-trip standby context is noncanonical")
    elif scenario.standby_context is not None:
        raise UnsupportedScenarioError("standby context is only supported for pump trip")
    if not scenario.fault_injections:
        if scenario.plant_variant_id is not PlantVariant.ASTER_A and (
            scenario.driver is not ScenarioDriver.STEADY_OPERATION
        ):
            raise UnsupportedScenarioError("only steady no-fault scenarios support Aster-B/C")
        expected: tuple[ScenarioAction, ...] = (
            ScenarioAction(
                decision_tick=scenario.duration_ticks - 1,
                action=ActionLabel.CONTINUE_MONITORING,
            ),
        )
        if scenario.action_sequence != expected:
            raise UnsupportedScenarioError("no-fault scenario action sequence is noncanonical")
        expected_id = (
            f"{spec.plant_variant.value.lower()}-stable-{scenario.seed}-{scenario.duration_ticks}"
            if scenario.driver is ScenarioDriver.STEADY_OPERATION
            else f"aster-a-load-{scenario.seed}-{scenario.duration_ticks}"
        )
        if scenario.scenario_id != expected_id:
            raise UnsupportedScenarioError("no-fault scenario id is noncanonical")
        if scenario.driver is ScenarioDriver.LOAD_TRANSIENT and (
            _LOAD_SETTLE_TICK >= scenario.duration_ticks
        ):
            raise UnsupportedScenarioError("load transient has insufficient response history")
        return
    process_faults = {
        FaultFamily.TRANSFER_EFFICIENCY_LOSS,
        FaultFamily.FLOW_IMBALANCE,
    }
    if (
        scenario.plant_variant_id is not PlantVariant.ASTER_A
        and scenario.fault_injections[0].fault_family not in process_faults
        and not is_support_power
    ):
        raise UnsupportedScenarioError("this fault scenario currently supports only ASTER-A")
    if is_support_power:
        injection = scenario.fault_injections[0]
        if scenario.plant_variant_id not in {PlantVariant.ASTER_A, PlantVariant.ASTER_B}:
            raise UnsupportedScenarioError("G12 supports only Aster-A and Aster-B")
        expected_bus_id = spec.component_for_role(ComponentRole.SUPPORT_BUS_TWO).component_id
        expected_included_id = _support_power_scenario_id(
            spec=spec,
            seed=scenario.seed,
            duration_ticks=scenario.duration_ticks,
            bus_id=expected_bus_id,
            include_dependency_map=True,
        )
        expected_withheld_id = _support_power_scenario_id(
            spec=spec,
            seed=scenario.seed,
            duration_ticks=scenario.duration_ticks,
            bus_id=expected_bus_id,
            include_dependency_map=False,
        )
        if scenario.scenario_id == expected_included_id:
            expected_action = ActionLabel.ENTER_SIMULATED_STABLE_STATE
            g12_context: DependencyMapContext | None = dependency_map_context_for(spec)
        elif scenario.scenario_id == expected_withheld_id:
            expected_action = ActionLabel.INSUFFICIENT_EVIDENCE
            g12_context = None
        else:
            raise UnsupportedScenarioError("support-power scenario id is noncanonical")
        if (
            scenario.driver is not ScenarioDriver.STEADY_OPERATION
            or type(injection.component_id) is not str
            or injection.component_id != expected_bus_id
            or injection.channel_id is not None
            or injection.severity is not SeverityBand.LOW
            or injection.onset_tick != _SUPPORT_POWER_ONSET_TICK
            or injection.duration_ticks is not None
            or scenario.duration_ticks < _SUPPORT_POWER_MIN_DURATION
            or scenario.standby_context is not None
            or scenario.action_sequence
            != (
                ScenarioAction(
                    decision_tick=_SUPPORT_POWER_DECISION_TICK,
                    action=expected_action,
                ),
            )
            or scenario.dependency_map_context != g12_context
        ):
            raise UnsupportedScenarioError("unsupported support-power interruption scenario")
        return
    if scenario.driver is ScenarioDriver.LOAD_TRANSIENT:
        injection = scenario.fault_injections[0]
        if injection.fault_family is not FaultFamily.SENSOR_STUCK:
            raise UnsupportedScenarioError(
                "load transient cannot include unsupported fault injection"
            )
        channel_id = injection.channel_id
        if not isinstance(channel_id, str):
            raise UnsupportedScenarioError(
                "sensor-stuck load requires an electrical-output channel"
            )
        expected_channel_components = {
            channel.channel_id: channel.component_id
            for channel in ASTER_A_SPEC.channels
            if channel.variable is StateVariable.ELECTRICAL_OUTPUT
        }
        stuck_expected_actions = (
            ScenarioAction(
                decision_tick=_STUCK_VERIFY_TICK,
                action=ActionLabel.VERIFY_REDUNDANT_CHANNEL,
            ),
            ScenarioAction(
                decision_tick=_STUCK_FLAG_TICK,
                action=ActionLabel.FLAG_SENSOR_SUSPECT,
            ),
        )
        if (
            channel_id not in _stuck_channels()
            or injection.component_id != expected_channel_components.get(channel_id)
            or injection.severity is not SeverityBand.LOW
            or injection.onset_tick != _LOAD_ONSET_TICK
            or injection.duration_ticks is not None
            or scenario.action_sequence != stuck_expected_actions
            or _STUCK_FLAG_APPLY_TICK >= scenario.duration_ticks
        ):
            raise UnsupportedScenarioError("unsupported sensor-stuck load scenario")
        expected_id = f"aster-a-stuck-load-{scenario.seed}-{scenario.duration_ticks}-{channel_id}"
        if scenario.scenario_id != expected_id:
            raise UnsupportedScenarioError("sensor-stuck load scenario id is noncanonical")
        return
    injection = scenario.fault_injections[0]
    if injection.fault_family in process_faults:
        is_transfer = injection.fault_family is FaultFamily.TRANSFER_EFFICIENCY_LOSS
        expected_process_component = (
            spec.transfer_unit_id if is_transfer else spec.secondary_feed_id
        )
        expected_actions = (
            (
                ScenarioAction(
                    decision_tick=_TRANSFER_THERMAL_TICK,
                    action=ActionLabel.INSUFFICIENT_EVIDENCE,
                ),
                ScenarioAction(
                    decision_tick=_TRANSFER_LOAD_DECISION_TICK,
                    action=ActionLabel.REDUCE_SIMULATED_LOAD,
                ),
            )
            if is_transfer
            else (
                ScenarioAction(
                    decision_tick=_FLOW_IMBALANCE_COMPARE_TICK,
                    action=ActionLabel.COMPARE_RELATED_TRENDS,
                ),
                ScenarioAction(
                    decision_tick=_FLOW_IMBALANCE_OUTPUT_TICK,
                    action=ActionLabel.ENTER_SIMULATED_STABLE_STATE,
                ),
            )
        )
        minimum_duration = _TRANSFER_MIN_DURATION if is_transfer else _FLOW_IMBALANCE_MIN_DURATION
        expected_id = _process_scenario_id(
            spec=spec,
            family=injection.fault_family,
            seed=scenario.seed,
            duration_ticks=scenario.duration_ticks,
            component_id=expected_process_component,
        )
        if (
            scenario.driver is not ScenarioDriver.STEADY_OPERATION
            or type(injection.component_id) is not str
            or injection.component_id != expected_process_component
            or injection.channel_id is not None
            or injection.severity is not SeverityBand.LOW
            or injection.onset_tick != _TRANSFER_ONSET_TICK
            or injection.duration_ticks is not None
            or scenario.action_sequence != expected_actions
            or scenario.duration_ticks < minimum_duration
            or scenario.standby_context is not None
            or scenario.dependency_map_context is not None
            or scenario.scenario_id != expected_id
        ):
            raise UnsupportedScenarioError("unsupported transfer/flow process scenario")
        return
    if injection.fault_family in {FaultFamily.VALVE_LAG, FaultFamily.VALVE_STUCK}:
        lag_duration = injection.duration_ticks
        if injection.fault_family is FaultFamily.VALVE_LAG:
            if type(lag_duration) is not int or lag_duration not in {3, 4}:
                raise UnsupportedScenarioError(
                    "valve lag must use the declared {3, 4} duration band"
                )
            decisive_tick = injection.onset_tick + lag_duration
        else:
            decisive_tick = _VALVE_DECISIVE_TICK
        expected_action = (
            ActionLabel.CONTINUE_MONITORING
            if injection.fault_family is FaultFamily.VALVE_LAG
            else ActionLabel.REQUEST_COMPONENT_INSPECTION
        )
        valve_expected_actions = (
            ScenarioAction(
                decision_tick=_VALVE_EARLY_DECISION_TICK,
                action=ActionLabel.INSUFFICIENT_EVIDENCE,
            ),
            ScenarioAction(
                decision_tick=decisive_tick,
                action=expected_action,
            ),
        )
        expected_duration = (
            lag_duration if injection.fault_family is FaultFamily.VALVE_LAG else None
        )
        if (
            scenario.driver is not ScenarioDriver.STEADY_OPERATION
            or type(injection.component_id) is not str
            or injection.component_id not in ASTER_A_SPEC.primary_flow_valve_ids
            or injection.channel_id is not None
            or injection.severity is not SeverityBand.LOW
            or injection.onset_tick != _VALVE_ONSET_TICK
            or injection.duration_ticks != expected_duration
            or scenario.action_sequence != valve_expected_actions
            or scenario.duration_ticks < _VALVE_MIN_DURATION
            or scenario.standby_context is not None
        ):
            raise UnsupportedScenarioError("unsupported valve lag/stuck scenario")
        expected_id = (
            f"aster-a-{injection.fault_family.value.lower().replace('_', '-')}-"
            f"{scenario.seed}-{scenario.duration_ticks}-{_VALVE_ONSET_TICK}-low-"
            f"{injection.component_id}-lag-{expected_duration}"
            if expected_duration is not None
            else f"aster-a-{injection.fault_family.value.lower().replace('_', '-')}-"
            f"{scenario.seed}-{scenario.duration_ticks}-{_VALVE_ONSET_TICK}-low-"
            f"{injection.component_id}"
        )
        if scenario.scenario_id != expected_id:
            raise UnsupportedScenarioError("valve lag/stuck scenario id is noncanonical")
        return
    if injection.fault_family is FaultFamily.PUMP_TRIP:
        context = scenario.standby_context
        if type(context) is not StandbyContext:
            raise UnsupportedScenarioError("pump trip requires canonical standby context")
        if (
            type(injection.component_id) is not str
            or injection.component_id not in ASTER_A_SPEC.primary_train_ids
        ):
            raise UnsupportedScenarioError("pump trip requires an Aster-A primary train")
        if context.standby_state not in {
            ComponentState.AVAILABLE,
            ComponentState.UNAVAILABLE,
        }:
            raise UnsupportedScenarioError("unsupported standby state for pump trip")
        expected_context = _trip_standby_context(
            active_train_id=injection.component_id,
            standby_state=context.standby_state,
        )
        trip_expected_actions = (
            (
                ScenarioAction(
                    decision_tick=_PUMP_TRIP_DECISION_TICK,
                    action=ActionLabel.SELECT_SYNTHETIC_STANDBY_TRAIN,
                ),
            )
            if context.standby_state is ComponentState.AVAILABLE
            else (
                ScenarioAction(
                    decision_tick=_PUMP_TRIP_DECISION_TICK,
                    action=ActionLabel.REDUCE_SIMULATED_LOAD,
                ),
                ScenarioAction(
                    decision_tick=_PUMP_TRIP_ACTION_TICK,
                    action=ActionLabel.ENTER_SIMULATED_STABLE_STATE,
                ),
            )
        )
        if (
            type(injection.component_id) is not str
            or injection.component_id not in ASTER_A_SPEC.primary_train_ids
            or injection.channel_id is not None
            or injection.severity is not SeverityBand.LOW
            or injection.onset_tick != _PUMP_TRIP_ONSET_TICK
            or injection.duration_ticks is not None
            or scenario.action_sequence != trip_expected_actions
            or scenario.duration_ticks < _PUMP_TRIP_MIN_DURATION
            or context != expected_context
            or context.support_bus_state is not ComponentState.AVAILABLE
            or context.standby_start_delay_ticks != ASTER_A_SPEC.standby_start_delay_ticks
            or ASTER_A_SPEC.primary_train_support_bus_pairs
            != tuple(zip(_PRIMARY_TRAINS, _SUPPORT_BUSES, strict=True))
        ):
            raise UnsupportedScenarioError("unsupported pump-trip scenario")
        expected_id = (
            f"aster-a-pump-trip-{scenario.seed}-{scenario.duration_ticks}-"
            f"{_PUMP_TRIP_ONSET_TICK}-low-{injection.component_id}-"
            f"{context.standby_state.value.lower()}"
        )
        if scenario.scenario_id != expected_id:
            raise UnsupportedScenarioError("pump-trip scenario id is noncanonical")
        return
    if injection.fault_family is FaultFamily.PUMP_DEGRADATION:
        pump_expected_actions = (
            ScenarioAction(
                decision_tick=_PUMP_THERMAL_TICK,
                action=ActionLabel.INSUFFICIENT_EVIDENCE,
            ),
            ScenarioAction(
                decision_tick=_PUMP_INSPECTION_DECISION_TICK,
                action=ActionLabel.REQUEST_COMPONENT_INSPECTION,
            ),
            ScenarioAction(
                decision_tick=_PUMP_LOAD_DECISION_TICK,
                action=ActionLabel.REDUCE_SIMULATED_LOAD,
            ),
        )
        if (
            scenario.driver is not ScenarioDriver.STEADY_OPERATION
            or type(injection.component_id) is not str
            or injection.component_id not in ASTER_A_SPEC.primary_train_ids
            or injection.channel_id is not None
            or injection.severity is not SeverityBand.LOW
            or injection.onset_tick != _PUMP_DEGRADATION_ONSET_TICK
            or injection.duration_ticks is not None
            or scenario.action_sequence != pump_expected_actions
            or scenario.duration_ticks < _PUMP_MIN_DURATION
            or scenario.standby_context is not None
        ):
            raise UnsupportedScenarioError("unsupported pump-degradation scenario")
        expected_id = (
            f"aster-a-pump-degradation-{scenario.seed}-{scenario.duration_ticks}-"
            f"{_PUMP_DEGRADATION_ONSET_TICK}-low-{injection.component_id}"
        )
        if scenario.scenario_id != expected_id:
            raise UnsupportedScenarioError("pump-degradation scenario id is noncanonical")
        return
    if injection.fault_family is FaultFamily.SENSOR_NOISE:
        channel_id = injection.channel_id
        noise_expected_actions = (
            ScenarioAction(
                decision_tick=_SENSOR_NOISE_FIRST_DECISION_TICK,
                action=ActionLabel.INSUFFICIENT_EVIDENCE,
            ),
            ScenarioAction(
                decision_tick=_SENSOR_NOISE_SECOND_DECISION_TICK,
                action=ActionLabel.INSUFFICIENT_EVIDENCE,
            ),
            ScenarioAction(
                decision_tick=_SENSOR_NOISE_DIAGNOSIS_TICK,
                action=ActionLabel.COMPARE_RELATED_TRENDS,
            ),
            ScenarioAction(
                decision_tick=_SENSOR_NOISE_FLAG_TICK,
                action=ActionLabel.FLAG_SENSOR_SUSPECT,
            ),
        )
        if (
            not isinstance(channel_id, str)
            or channel_id not in _sensor_noise_channels()
            or injection.component_id != _INSTRUMENTATION
            or injection.severity is not SeverityBand.LOW
            or injection.onset_tick != _SENSOR_NOISE_ONSET_TICK
            or injection.duration_ticks is not None
            or scenario.action_sequence != noise_expected_actions
            or _SENSOR_NOISE_FLAG_APPLY_TICK >= scenario.duration_ticks
            or scenario.standby_context is not None
        ):
            raise UnsupportedScenarioError("unsupported sensor-noise scenario")
        expected_id = (
            f"aster-a-noise-{scenario.seed}-{scenario.duration_ticks}-{_SENSOR_NOISE_ONSET_TICK}-low-"
            f"{channel_id}"
        )
        if scenario.scenario_id != expected_id:
            raise UnsupportedScenarioError("sensor-noise scenario id is noncanonical")
        return
    if injection.fault_family is not FaultFamily.SENSOR_DRIFT:
        raise UnsupportedScenarioError("only SENSOR_DRIFT is supported")
    channel_id = injection.channel_id
    if channel_id is None:
        raise UnsupportedScenarioError("sensor drift requires a primary-flow channel")
    if not isinstance(injection.severity, SeverityBand):
        raise UnsupportedScenarioError("sensor drift requires a canonical severity")
    try:
        _require_uint32(injection.onset_tick, name="fault onset")
    except ValueError as error:
        raise UnsupportedScenarioError(str(error)) from error
    expected_component = {
        channel.channel_id: channel.component_id
        for channel in ASTER_A_SPEC.channels
        if channel.variable is StateVariable.PRIMARY_FLOW
    }.get(channel_id)
    if channel_id not in _drift_channels() or injection.component_id != expected_component:
        raise UnsupportedScenarioError("sensor drift must use an Aster-A primary-flow mapping")
    if injection.duration_ticks is not None:
        raise UnsupportedScenarioError("finite-duration sensor drift is not supported")
    if injection.onset_tick + 4 >= scenario.duration_ticks:
        raise UnsupportedScenarioError("sensor drift has insufficient post-onset history")
    expected = (
        ScenarioAction(
            decision_tick=injection.onset_tick + 1,
            action=ActionLabel.INSUFFICIENT_EVIDENCE,
        ),
        ScenarioAction(
            decision_tick=injection.onset_tick + 2,
            action=ActionLabel.VERIFY_REDUNDANT_CHANNEL,
        ),
        ScenarioAction(
            decision_tick=injection.onset_tick + 3,
            action=ActionLabel.FLAG_SENSOR_SUSPECT,
        ),
    )
    if scenario.action_sequence != expected:
        raise UnsupportedScenarioError("sensor drift action sequence is noncanonical")
    expected_id = (
        f"aster-a-drift-{scenario.seed}-{scenario.duration_ticks}-{injection.onset_tick}-"
        f"{injection.severity.value.lower()}-{channel_id}"
    )
    if scenario.scenario_id != expected_id:
        raise UnsupportedScenarioError("sensor drift scenario id is noncanonical")


def generate_trace(scenario: ScenarioDefinition) -> SimulationTrace:
    """Generate a deterministic, validated separated trace for one supported scenario."""

    _validate_supported_scenario(scenario)
    latent_states = _latent_states(scenario)
    observations = _observations(scenario, latent_states)
    events, targets = _events_and_targets(scenario, observations)
    trace = SimulationTrace(
        scenario=scenario,
        latent_states=latent_states,
        observations=observations,
        events=events,
        targets=targets,
    )
    assert_no_prohibited_content(_spec_for(scenario))
    assert_no_prohibited_content(trace)
    return trace
