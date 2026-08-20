"""Read-only, fail-closed projection from audit trajectories to renderer input."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from reactorbench.dataset.contracts import (
    DATASET_CONTRACT_VERSION,
    CounterfactualPairInput,
    CounterfactualProjectionLineage,
    CounterfactualProjectionRecord,
    DependencyLinkContextFact,
    ModelInput,
    ProjectedContextFact,
    ProjectedEventFact,
    ProjectedObservationFact,
    ProjectionLineage,
    ProjectionRecord,
    ProjectionTaskTarget,
    ProjectionView,
    PromptCounterfactualComparisonTarget,
    StandbyRelationshipContextFact,
    make_action_target,
    make_continuation_target,
    make_evidence_target,
    make_fault_target,
    make_incident_summary_target,
)
from reactorbench.dataset.grouping import CounterfactualGroup, require_complete_group
from reactorbench.schemas.base import SCHEMA_VERSION, canonical_sha256
from reactorbench.schemas.enums import (
    AsterSubsystem,
    ChannelQuality,
    ComponentState,
    CounterfactualChange,
    DiagnosisStatus,
    EventType,
    EvidenceSlot,
    FaultFamily,
    ObservationStatus,
    ObservedTrend,
    OperatingMode,
    StateVariable,
    TaskName,
)
from reactorbench.schemas.events import CanonicalEvent
from reactorbench.schemas.observation import SensorChannelObservation
from reactorbench.schemas.target import CounterfactualConclusion, DecisionTarget
from reactorbench.schemas.trajectory import StructuredTrajectory
from reactorbench.simulator.variants import get_variant_spec


class ProjectionError(ValueError):
    """Raised when audit truth cannot be safely projected under a declared view."""


_G07_VARIABLES = frozenset(
    {
        StateVariable.PRIMARY_FLOW,
        StateVariable.PRIMARY_THERMAL_STATE,
        StateVariable.STEAM_STATE,
        StateVariable.TURBINE_OUTPUT,
        StateVariable.ELECTRICAL_OUTPUT,
        StateVariable.SUPPORT_POWER,
    }
)
_G12_VARIABLES = frozenset(
    {
        StateVariable.SUPPORT_POWER,
        StateVariable.PRIMARY_FLOW,
        StateVariable.TRANSFER_EFFICIENCY,
        StateVariable.SECONDARY_FLOW,
        StateVariable.PRIMARY_THERMAL_STATE,
        StateVariable.STEAM_STATE,
        StateVariable.SECONDARY_INVENTORY,
    }
)
_G13_VARIABLES = frozenset(
    {
        StateVariable.PRIMARY_INVENTORY,
        StateVariable.PRIMARY_FLOW,
        StateVariable.PRIMARY_THERMAL_STATE,
    }
)
_G14_VARIABLES = frozenset(
    {
        StateVariable.PRIMARY_FLOW,
        StateVariable.PRIMARY_THERMAL_STATE,
        StateVariable.STEAM_STATE,
        StateVariable.ELECTRICAL_OUTPUT,
    }
)
_OBSERVATION_EVENT_TYPES = frozenset(
    {
        EventType.OBSERVATION_CHANGED,
        EventType.CHANNEL_QUALITY_CHANGED,
        EventType.CHANNEL_DISAGREEMENT,
    }
)


@dataclass(frozen=True, slots=True)
class _ProjectionRecipe:
    recipe_id: str
    all_channel_variables: frozenset[StateVariable]
    single_channel_variables: frozenset[StateVariable]
    event_types: frozenset[EventType]


_SUBSYSTEM_BY_FAULT: dict[FaultFamily, AsterSubsystem] = {
    FaultFamily.SENSOR_DRIFT: AsterSubsystem.INSTRUMENTATION,
    FaultFamily.SENSOR_STUCK: AsterSubsystem.INSTRUMENTATION,
    FaultFamily.SENSOR_NOISE: AsterSubsystem.INSTRUMENTATION,
    FaultFamily.PUMP_DEGRADATION: AsterSubsystem.PRIMARY_LOOP,
    FaultFamily.PUMP_TRIP: AsterSubsystem.PRIMARY_LOOP,
    FaultFamily.VALVE_LAG: AsterSubsystem.PRIMARY_LOOP,
    FaultFamily.VALVE_STUCK: AsterSubsystem.PRIMARY_LOOP,
    FaultFamily.TRANSFER_EFFICIENCY_LOSS: AsterSubsystem.TRANSFER_UNIT,
    FaultFamily.FLOW_IMBALANCE: AsterSubsystem.SECONDARY_LOOP,
    FaultFamily.SUPPORT_POWER_INTERRUPTION: AsterSubsystem.SUPPORT_POWER,
    FaultFamily.ABSTRACT_INVENTORY_LOSS: AsterSubsystem.PRIMARY_LOOP,
}


def _fault_signature(trajectory: StructuredTrajectory) -> tuple[FaultFamily, ...]:
    return tuple(item.fault_family for item in trajectory.scenario.fault_injections)


def _is_thermal_drift(trajectory: StructuredTrajectory) -> bool:
    signature = _fault_signature(trajectory)
    if signature != (FaultFamily.SENSOR_DRIFT,):
        return False
    injection = trajectory.scenario.fault_injections[0]
    if injection.channel_id is None:
        return False
    spec = get_variant_spec(trajectory.scenario.plant_variant_id)
    return any(
        channel.channel_id == injection.channel_id
        and channel.variable is StateVariable.PRIMARY_THERMAL_STATE
        for channel in spec.channels
    )


def _is_g15_sparse(trajectory: StructuredTrajectory) -> bool:
    scenario = trajectory.scenario
    if scenario.fault_injections or scenario.standby_context or scenario.dependency_map_context:
        return False
    if len(trajectory.targets.decisions) != 1:
        return False
    decision = trajectory.targets.decisions[0]
    if decision.decision_tick != 2 or decision.diagnosis_status is not DiagnosisStatus.UNRESOLVED:
        return False
    sparse_events = tuple(
        event
        for event in trajectory.events
        if event.sim_time == decision.decision_tick
        and event.event_type is EventType.OBSERVATION_CHANGED
        and event.variable is StateVariable.PRIMARY_FLOW
    )
    return len(sparse_events) == 1


def infer_projection_view(trajectory: StructuredTrajectory) -> ProjectionView:
    """Infer the sole safe policy from source structure, never target contents."""

    signature = _fault_signature(trajectory)
    scenario = trajectory.scenario
    if signature == (FaultFamily.PUMP_TRIP,) and scenario.standby_context is not None:
        return ProjectionView.G07_STANDBY_DECISION
    if signature == (FaultFamily.SUPPORT_POWER_INTERRUPTION,):
        return (
            ProjectionView.G12_MAP_INCLUDED_DECISION
            if scenario.dependency_map_context is not None
            else ProjectionView.G12_MAP_WITHHELD_DECISION
        )
    if signature == (FaultFamily.ABSTRACT_INVENTORY_LOSS,):
        return ProjectionView.G13_INVENTORY_DECISION
    if signature in {
        (FaultFamily.PUMP_DEGRADATION,),
        (FaultFamily.SENSOR_DRIFT, FaultFamily.PUMP_DEGRADATION),
    } or _is_thermal_drift(trajectory):
        return ProjectionView.G14_FACTORED_DECISION
    if _is_g15_sparse(trajectory):
        return ProjectionView.G15_SPARSE_DECISION
    return ProjectionView.STANDARD_DECISION


def _require_declared_view(
    trajectory: StructuredTrajectory, declared: ProjectionView
) -> ProjectionView:
    if type(declared) is not ProjectionView:
        raise ProjectionError("view must be a ProjectionView")
    expected = infer_projection_view(trajectory)
    if declared is not expected:
        raise ProjectionError(
            f"declared view {declared.value} does not match required policy {expected.value}"
        )
    return declared


def _decision_at(trajectory: StructuredTrajectory, tick: int) -> DecisionTarget:
    matches = tuple(
        decision for decision in trajectory.targets.decisions if decision.decision_tick == tick
    )
    if len(matches) != 1:
        raise ProjectionError("decision_tick must identify exactly one source decision")
    return matches[0]


def _standard_recipe(trajectory: StructuredTrajectory, *, decision_tick: int) -> _ProjectionRecipe:
    scenario = trajectory.scenario
    signature = _fault_signature(trajectory)
    decision_ticks = tuple(item.decision_tick for item in trajectory.targets.decisions)
    try:
        ordinal = decision_ticks.index(decision_tick) + 1
    except ValueError as error:
        raise ProjectionError("decision tick has no declared projection recipe") from error
    suffix = f"d{ordinal}-v1"
    state_events = _OBSERVATION_EVENT_TYPES | {
        EventType.OPERATING_MODE_CHANGED,
        EventType.TARGET_CHANGED,
        EventType.COMPONENT_STATE_CHANGED,
    }
    if not signature and scenario.driver.value == "STEADY_OPERATION":
        return _ProjectionRecipe(
            f"g01-stable-{suffix}",
            frozenset(),
            frozenset(
                {
                    StateVariable.PRIMARY_FLOW,
                    StateVariable.PRIMARY_THERMAL_STATE,
                    StateVariable.STEAM_STATE,
                    StateVariable.ELECTRICAL_OUTPUT,
                }
            ),
            frozenset(),
        )
    if not signature and scenario.driver.value == "LOAD_TRANSIENT":
        return _ProjectionRecipe(
            f"g02-load-{suffix}",
            frozenset(),
            frozenset(
                {
                    StateVariable.LOAD_DEMAND,
                    StateVariable.HEAT_SOURCE_LEVEL,
                    StateVariable.PRIMARY_FLOW,
                    StateVariable.STEAM_STATE,
                    StateVariable.TURBINE_OUTPUT,
                    StateVariable.ELECTRICAL_OUTPUT,
                }
            ),
            state_events,
        )
    if signature in {(FaultFamily.SENSOR_DRIFT,), (FaultFamily.SENSOR_NOISE,)}:
        injection = scenario.fault_injections[0]
        if injection.channel_id is None:
            raise ProjectionError("sensor recipe requires a channel-scoped injection")
        spec = get_variant_spec(scenario.plant_variant_id)
        variable = next(
            (
                channel.variable
                for channel in spec.channels
                if channel.channel_id == injection.channel_id
            ),
            None,
        )
        if variable is None:
            raise ProjectionError("sensor channel is absent from the variant registry")
        related = (
            frozenset({StateVariable.PRIMARY_THERMAL_STATE})
            if variable is not StateVariable.PRIMARY_THERMAL_STATE
            else frozenset({StateVariable.PRIMARY_FLOW})
        )
        return _ProjectionRecipe(
            f"g03-g05-observation-fault-{suffix}",
            frozenset({variable}),
            related,
            _OBSERVATION_EVENT_TYPES,
        )
    if signature == (FaultFamily.SENSOR_STUCK,):
        return _ProjectionRecipe(
            f"g04-stuck-load-{suffix}",
            frozenset({StateVariable.ELECTRICAL_OUTPUT}),
            frozenset(
                {
                    StateVariable.LOAD_DEMAND,
                    StateVariable.TURBINE_OUTPUT,
                    StateVariable.PRIMARY_FLOW,
                }
            ),
            state_events,
        )
    if signature in {(FaultFamily.VALVE_LAG,), (FaultFamily.VALVE_STUCK,)}:
        return _ProjectionRecipe(
            f"g08-g09-valve-{suffix}",
            frozenset({StateVariable.PRIMARY_FLOW}),
            frozenset(),
            _OBSERVATION_EVENT_TYPES
            | {
                EventType.COMMAND_RECORDED,
                EventType.COMMAND_POSITION_MISMATCH,
                EventType.COMMAND_POSITION_ALIGNED,
            },
        )
    if signature == (FaultFamily.TRANSFER_EFFICIENCY_LOSS,):
        return _ProjectionRecipe(
            f"g10-transfer-{suffix}",
            frozenset({StateVariable.PRIMARY_FLOW}),
            frozenset(
                {
                    StateVariable.TRANSFER_EFFICIENCY,
                    StateVariable.PRIMARY_THERMAL_STATE,
                    StateVariable.STEAM_STATE,
                    StateVariable.TURBINE_OUTPUT,
                    StateVariable.ELECTRICAL_OUTPUT,
                }
            ),
            state_events,
        )
    if signature == (FaultFamily.FLOW_IMBALANCE,):
        return _ProjectionRecipe(
            f"g11-flow-imbalance-{suffix}",
            frozenset({StateVariable.SECONDARY_FLOW}),
            frozenset(
                {
                    StateVariable.SECONDARY_INVENTORY,
                    StateVariable.STEAM_STATE,
                    StateVariable.ELECTRICAL_OUTPUT,
                    StateVariable.PRIMARY_FLOW,
                    StateVariable.TRANSFER_EFFICIENCY,
                }
            ),
            state_events,
        )
    raise ProjectionError("standard view has no preregistered scenario-family recipe")


def _projection_recipe(
    trajectory: StructuredTrajectory, *, decision_tick: int, view: ProjectionView
) -> _ProjectionRecipe:
    decision_ticks = tuple(item.decision_tick for item in trajectory.targets.decisions)
    try:
        ordinal = decision_ticks.index(decision_tick) + 1
    except ValueError as error:
        raise ProjectionError("decision tick has no declared projection recipe") from error
    if view is ProjectionView.G07_STANDBY_DECISION:
        return _ProjectionRecipe(
            f"g07-standby-d{ordinal}-v1",
            frozenset({StateVariable.PRIMARY_FLOW}),
            _G07_VARIABLES - {StateVariable.PRIMARY_FLOW},
            _OBSERVATION_EVENT_TYPES
            | {EventType.OPERATING_MODE_CHANGED, EventType.COMPONENT_STATE_CHANGED},
        )
    if view in {
        ProjectionView.G12_MAP_INCLUDED_DECISION,
        ProjectionView.G12_MAP_WITHHELD_DECISION,
    }:
        return _ProjectionRecipe(
            f"g12-dependency-map-d{ordinal}-v1",
            frozenset({StateVariable.SUPPORT_POWER}),
            _G12_VARIABLES - {StateVariable.SUPPORT_POWER},
            _OBSERVATION_EVENT_TYPES
            | {EventType.OPERATING_MODE_CHANGED, EventType.COMPONENT_STATE_CHANGED},
        )
    if view is ProjectionView.G13_INVENTORY_DECISION:
        return _ProjectionRecipe(
            f"g13-inventory-d{ordinal}-v1",
            frozenset(),
            _G13_VARIABLES,
            _OBSERVATION_EVENT_TYPES | {EventType.OPERATING_MODE_CHANGED},
        )
    if view is ProjectionView.G14_FACTORED_DECISION:
        return _ProjectionRecipe(
            f"g14-factored-d{ordinal}-v1",
            frozenset({StateVariable.PRIMARY_FLOW, StateVariable.PRIMARY_THERMAL_STATE}),
            _G14_VARIABLES - {StateVariable.PRIMARY_FLOW, StateVariable.PRIMARY_THERMAL_STATE},
            _OBSERVATION_EVENT_TYPES | {EventType.COMPONENT_STATE_CHANGED},
        )
    if view is ProjectionView.G15_SPARSE_DECISION:
        return _ProjectionRecipe(
            "g15-sparse-d1-v1",
            frozenset(),
            frozenset({StateVariable.PRIMARY_FLOW}),
            frozenset({EventType.OBSERVATION_CHANGED}),
        )
    return _standard_recipe(trajectory, decision_tick=decision_tick)


def _project_context(
    trajectory: StructuredTrajectory, view: ProjectionView, task_name: TaskName
) -> tuple[ProjectedContextFact, ...]:
    scenario = trajectory.scenario
    facts: list[ProjectedContextFact] = []
    if view is ProjectionView.G07_STANDBY_DECISION and task_name in {
        TaskName.NEXT_ACTION,
        TaskName.INCIDENT_SUMMARY,
        TaskName.EXTRACT_EVIDENCE,
    }:
        standby_context = scenario.standby_context
        if standby_context is None:
            raise ProjectionError("G07 projection requires strict standby context")
        facts.append(
            StandbyRelationshipContextFact(
                fact_ref=f"c-{len(facts):04d}",
                active_component_id=standby_context.active_train_id,
                standby_component_id=standby_context.standby_train_id,
                standby_state=standby_context.standby_state,
                support_component_id=standby_context.standby_support_bus_id,
                support_state=standby_context.support_bus_state,
                start_delay_ticks=standby_context.standby_start_delay_ticks,
            )
        )
    elif view is ProjectionView.G12_MAP_INCLUDED_DECISION:
        dependency_context = scenario.dependency_map_context
        if dependency_context is None:
            raise ProjectionError("map-included G12 projection requires dependency context")
        for link in dependency_context.links:
            facts.append(
                DependencyLinkContextFact(
                    fact_ref=f"c-{len(facts):04d}",
                    support_component_id=link.support_bus_id,
                    dependent_component_id=link.dependent_component_id,
                )
            )
    elif view is ProjectionView.G12_MAP_WITHHELD_DECISION:
        if scenario.dependency_map_context is not None:
            raise ProjectionError("map-withheld G12 projection cannot expose dependency context")
    return tuple(facts)


def _event_fact(event: CanonicalEvent, *, index: int) -> ProjectedEventFact:
    return ProjectedEventFact(
        fact_ref=f"e-{index:04d}",
        tick=event.sim_time,
        event_type=event.event_type,
        subject_id=event.subject_id,
        operating_mode_before=event.operating_mode_before,
        operating_mode_after=event.operating_mode_after,
        component_state_before=event.component_state_before,
        component_state_after=event.component_state_after,
        variable=event.variable,
        value_before=event.value_before,
        value_after=event.value_after,
        observation_status=event.observation_status,
        channel_quality_before=event.channel_quality_before,
        channel_quality=event.channel_quality,
        commanded_value=event.commanded_value,
        observed_value=event.observed_value,
    )


def _decision_events(
    trajectory: StructuredTrajectory,
    *,
    cut_tick: int,
    view: ProjectionView,
    recipe: _ProjectionRecipe,
) -> tuple[CanonicalEvent, ...]:
    selected = tuple(
        event
        for event in trajectory.events
        if event.sim_time <= cut_tick and event.event_type in recipe.event_types
    )
    if view is ProjectionView.G15_SPARSE_DECISION:
        selected = tuple(
            event
            for event in selected
            if event.sim_time == cut_tick
            and event.event_type is EventType.OBSERVATION_CHANGED
            and event.variable is StateVariable.PRIMARY_FLOW
        )
        if len(selected) != 1:
            raise ProjectionError("G15 must project exactly one sparse primary-flow event")
    return selected


def _project_event_facts(
    events: Iterable[CanonicalEvent],
) -> tuple[ProjectedEventFact, ...]:
    ordered = tuple(sorted(events, key=lambda event: event.event_index))
    return tuple(_event_fact(event, index=index) for index, event in enumerate(ordered))


def _channel_is_selected(
    channel: SensorChannelObservation,
    *,
    selected_channel_ids: frozenset[str],
) -> bool:
    return channel.channel_id in selected_channel_ids


def _recipe_channel_ids(
    trajectory: StructuredTrajectory, recipe: _ProjectionRecipe
) -> frozenset[str]:
    spec = get_variant_spec(trajectory.scenario.plant_variant_id)
    selected: set[str] = set()
    for variable in recipe.all_channel_variables:
        selected.update(channel.channel_id for channel in spec.channels_for(variable))
    for variable in recipe.single_channel_variables:
        channels = spec.channels_for(variable)
        if not channels:
            raise ProjectionError(f"projection recipe has no channel for {variable.value}")
        selected.add(channels[0].channel_id)
    return frozenset(selected)


def _project_observations(
    trajectory: StructuredTrajectory,
    *,
    cut_tick: int,
    view: ProjectionView,
    selected_events: tuple[CanonicalEvent, ...],
    recipe: _ProjectionRecipe,
) -> tuple[ProjectedObservationFact, ...]:
    selected_channel_ids = _recipe_channel_ids(trajectory, recipe)
    selected_ticks: frozenset[int] | None = None
    if view is ProjectionView.G13_INVENTORY_DECISION:
        selected_channel_ids = frozenset(
            event.subject_id
            for event in selected_events
            if event.event_type is EventType.OBSERVATION_CHANGED
            and event.variable in _G13_VARIABLES
        )
        if not selected_channel_ids:
            raise ProjectionError("G13 decision prefix contains no allowed observation channel")
    elif view is ProjectionView.G15_SPARSE_DECISION:
        selected_channel_ids = frozenset(event.subject_id for event in selected_events)
        selected_ticks = frozenset({cut_tick})

    cells = tuple(
        (frame.tick, channel)
        for frame in trajectory.observations
        if frame.tick <= cut_tick and (selected_ticks is None or frame.tick in selected_ticks)
        for channel in frame.channels
        if _channel_is_selected(
            channel,
            selected_channel_ids=selected_channel_ids,
        )
    )
    return tuple(
        ProjectedObservationFact(
            fact_ref=f"o-{index:04d}",
            tick=tick,
            channel_id=channel.channel_id,
            variable=channel.variable,
            value=channel.value,
            quality=channel.quality,
            status=channel.status,
        )
        for index, (tick, channel) in enumerate(cells)
    )


def _observed_trend(events: tuple[ProjectedEventFact, ...]) -> ObservedTrend:
    deltas = tuple(
        event.value_after - event.value_before
        for event in events
        if event.value_before is not None
        and event.value_after is not None
        and event.value_before != event.value_after
    )
    if not deltas:
        return ObservedTrend.STABLE
    if all(delta > 0 for delta in deltas):
        return ObservedTrend.RISING
    if all(delta < 0 for delta in deltas):
        return ObservedTrend.FALLING
    return ObservedTrend.MIXED


def _operating_mode(events: tuple[ProjectedEventFact, ...]) -> OperatingMode:
    mode = OperatingMode.STABLE
    for event in events:
        if event.event_type is EventType.OPERATING_MODE_CHANGED:
            if event.operating_mode_after is None:
                raise ProjectionError("operating-mode event is missing its after value")
            mode = event.operating_mode_after
    return mode


def _unique_refs(*groups: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(ref for group in groups for ref in group))


def _events_of_type(
    model_input: ModelInput, *event_types: EventType
) -> tuple[ProjectedEventFact, ...]:
    allowed = frozenset(event_types)
    return tuple(event for event in model_input.event_facts if event.event_type in allowed)


def _observation_series_by_channel(
    model_input: ModelInput,
) -> dict[str, tuple[ProjectedObservationFact, ...]]:
    mutable: dict[str, list[ProjectedObservationFact]] = {}
    for fact in model_input.observation_facts:
        mutable.setdefault(fact.channel_id, []).append(fact)
    return {channel_id: tuple(facts) for channel_id, facts in mutable.items()}


def _stable_series_refs(
    model_input: ModelInput,
    *,
    variables: frozenset[StateVariable] | None = None,
) -> tuple[str, ...]:
    """Return one visibly stable channel series without consulting audit truth.

    A series is stable only when it has at least two good, normal observations and
    no visible change event for that channel.  The first and last cells are both
    cited because a single normal cell cannot establish stability over time.
    """

    visibly_changed_channels = {
        event.subject_id
        for event in model_input.event_facts
        if event.event_type
        in {
            EventType.OBSERVATION_CHANGED,
            EventType.CHANNEL_QUALITY_CHANGED,
            EventType.CHANNEL_DISAGREEMENT,
        }
    }
    for channel_id, series in _observation_series_by_channel(model_input).items():
        if channel_id in visibly_changed_channels or len(series) < 2:
            continue
        if variables is not None and series[0].variable not in variables:
            continue
        if all(
            fact.value is not None
            and fact.quality is ChannelQuality.GOOD
            and fact.status is ObservationStatus.NORMAL
            for fact in series
        ):
            return (series[0].fact_ref, series[-1].fact_ref)
    return ()


def _falling_event_refs(
    model_input: ModelInput, variables: frozenset[StateVariable]
) -> tuple[str, ...]:
    return tuple(
        event.fact_ref
        for event in model_input.event_facts
        if event.event_type is EventType.OBSERVATION_CHANGED
        and event.variable in variables
        and event.value_before is not None
        and event.value_after is not None
        and event.value_after < event.value_before
    )


def _falling_series_refs(
    model_input: ModelInput, variables: frozenset[StateVariable]
) -> tuple[str, ...]:
    for series in _observation_series_by_channel(model_input).values():
        first = series[0]
        last = series[-1]
        if (
            len(series) >= 2
            and first.variable in variables
            and first.value is not None
            and last.value is not None
            and last.value < first.value
        ):
            return (first.fact_ref, last.fact_ref)
    return ()


def _matching_channel_pair_refs(
    model_input: ModelInput,
) -> tuple[str, ...]:
    by_tick_and_variable: dict[tuple[int, StateVariable], list[ProjectedObservationFact]] = {}
    for fact in model_input.observation_facts:
        if fact.value is None or fact.quality is not ChannelQuality.GOOD:
            continue
        by_tick_and_variable.setdefault((fact.tick, fact.variable), []).append(fact)
    for key in sorted(by_tick_and_variable, reverse=True):
        facts = by_tick_and_variable[key]
        for index, left in enumerate(facts):
            for right in facts[index + 1 :]:
                left_value = left.value
                right_value = right.value
                if (
                    left.channel_id != right.channel_id
                    and left.status is right.status
                    and left_value is not None
                    and right_value is not None
                    and abs(left_value - right_value) <= 1e-6
                ):
                    return (left.fact_ref, right.fact_ref)
    return ()


def visible_evidence_fact_refs(model_input: ModelInput, slot: EvidenceSlot) -> tuple[str, ...]:
    """Ground one evidence label in renderer-visible semantics only.

    This closed matcher consumes no scenario identifier, fault injection, target
    label, source event identifier, or source evidence annotation.  It therefore
    cannot use audit-only truth to choose prompt facts.  A caller may retain only
    declared target slots for which this function returns references.
    """

    if type(model_input) is not ModelInput:
        raise TypeError("model_input must be a ModelInput")
    if type(slot) is not EvidenceSlot:
        raise TypeError("slot must be an EvidenceSlot")

    events = model_input.event_facts
    observations = model_input.observation_facts
    changed_observations = _events_of_type(model_input, EventType.OBSERVATION_CHANGED)
    component_changes = _events_of_type(model_input, EventType.COMPONENT_STATE_CHANGED)

    if slot is EvidenceSlot.STABLE_OPERATION:
        return _stable_series_refs(model_input)

    if slot is EvidenceSlot.COORDINATED_LOAD_RESPONSE:
        target_refs = tuple(
            event.fact_ref
            for event in events
            if event.event_type is EventType.TARGET_CHANGED
            and event.variable is StateVariable.LOAD_DEMAND
        )
        response_variables = {
            StateVariable.HEAT_SOURCE_LEVEL,
            StateVariable.PRIMARY_FLOW,
            StateVariable.TURBINE_OUTPUT,
            StateVariable.ELECTRICAL_OUTPUT,
        }
        response_refs = tuple(
            fact.fact_ref
            for fact in observations
            if fact.tick == model_input.cut_tick
            and fact.variable in response_variables
            and fact.status is ObservationStatus.NORMAL
            and fact.quality is ChannelQuality.GOOD
        )
        if (
            target_refs
            and len({fact.variable for fact in observations if fact.fact_ref in response_refs}) >= 2
        ):
            return _unique_refs(target_refs, response_refs)
        return ()

    if slot is EvidenceSlot.CHANNEL_DISAGREEMENT:
        return tuple(
            event.fact_ref
            for event in events
            if event.event_type is EventType.CHANNEL_DISAGREEMENT
            and event.observation_status is ObservationStatus.CONFLICTING
        )

    if slot is EvidenceSlot.RELATED_STATE_STABLE:
        return _stable_series_refs(model_input)

    if slot is EvidenceSlot.CHANNEL_FROZEN:
        for series in _observation_series_by_channel(model_input).values():
            if len(series) < 3 or any(fact.value is None for fact in series):
                continue
            recent = series[-3:]
            values = tuple(fact.value for fact in recent)
            if values[0] == values[1] == values[2]:
                return tuple(fact.fact_ref for fact in recent)
        return ()

    if slot is EvidenceSlot.CORRELATED_STATE_CHANGE:
        by_variable: dict[StateVariable, ProjectedEventFact] = {}
        for event in changed_observations:
            if event.variable is not None:
                by_variable[event.variable] = event
        if len(by_variable) >= 2:
            return tuple(event.fact_ref for event in tuple(by_variable.values())[-2:])
        if changed_observations and component_changes:
            return (component_changes[-1].fact_ref, changed_observations[-1].fact_ref)
        aligned = _events_of_type(model_input, EventType.COMMAND_POSITION_ALIGNED)
        if changed_observations and aligned:
            return (changed_observations[-1].fact_ref, aligned[-1].fact_ref)
        targets = _events_of_type(model_input, EventType.TARGET_CHANGED)
        changed_series: list[tuple[str, str]] = []
        for series in _observation_series_by_channel(model_input).values():
            first = series[0]
            last = series[-1]
            if (
                len(series) >= 2
                and first.value is not None
                and last.value is not None
                and first.value != last.value
            ):
                changed_series.append((first.fact_ref, last.fact_ref))
        if targets and len(changed_series) >= 2:
            return _unique_refs(
                (targets[-1].fact_ref,),
                changed_series[-2],
                changed_series[-1],
            )
        return ()

    if slot is EvidenceSlot.RAPID_INCONSISTENT_READINGS:
        by_subject: dict[str, list[ProjectedEventFact]] = {}
        for event in changed_observations:
            if event.value_before is not None and event.value_after is not None:
                by_subject.setdefault(event.subject_id, []).append(event)
        for subject_events in by_subject.values():
            signs: list[int] = []
            for event in subject_events:
                before = event.value_before
                after = event.value_after
                if before is not None and after is not None and after != before:
                    signs.append(1 if after > before else -1)
            if len(signs) >= 2 and len(set(signs)) > 1:
                return tuple(event.fact_ref for event in subject_events[-3:])
        return ()

    if slot is EvidenceSlot.COMPONENT_HEALTH_DECLINING:
        return tuple(
            event.fact_ref
            for event in component_changes
            if event.component_state_after is ComponentState.DEGRADED
        )

    if slot is EvidenceSlot.FLOW_DECLINING:
        variables = frozenset({StateVariable.PRIMARY_FLOW})
        return _unique_refs(
            _falling_event_refs(model_input, variables),
            _falling_series_refs(model_input, variables),
        )

    if slot is EvidenceSlot.DEPENDENT_TREND_DELAY:
        semantic_changes = tuple(
            event
            for event in (*component_changes, *changed_observations)
            if event.variable is not None or event.event_type is EventType.COMPONENT_STATE_CHANGED
        )
        for left_index, left in enumerate(semantic_changes):
            for right in semantic_changes[left_index + 1 :]:
                if right.tick > left.tick and (
                    left.subject_id != right.subject_id or left.variable is not right.variable
                ):
                    return (left.fact_ref, right.fact_ref)
        return ()

    if slot is EvidenceSlot.COMPONENT_UNAVAILABLE:
        event_refs = tuple(
            event.fact_ref
            for event in component_changes
            if event.component_state_after is ComponentState.UNAVAILABLE
        )
        context_refs = tuple(
            context.fact_ref
            for context in model_input.context_facts
            if isinstance(context, StandbyRelationshipContextFact)
            and (
                context.standby_state is ComponentState.UNAVAILABLE
                or context.support_state is ComponentState.UNAVAILABLE
            )
        )
        return _unique_refs(event_refs, context_refs)

    if slot is EvidenceSlot.STANDBY_AVAILABLE:
        return tuple(
            context.fact_ref
            for context in model_input.context_facts
            if isinstance(context, StandbyRelationshipContextFact)
            and context.standby_state is ComponentState.AVAILABLE
            and context.support_state is ComponentState.AVAILABLE
        )

    if slot is EvidenceSlot.COMMAND_POSITION_MISMATCH:
        return tuple(
            event.fact_ref
            for event in events
            if event.event_type is EventType.COMMAND_POSITION_MISMATCH
            and event.commanded_value != event.observed_value
        )

    if slot is EvidenceSlot.MISMATCH_RESOLVED:
        return tuple(
            event.fact_ref
            for event in events
            if event.event_type is EventType.COMMAND_POSITION_ALIGNED
            and event.commanded_value == event.observed_value
        )

    if slot is EvidenceSlot.MISMATCH_PERSISTED:
        mismatch = _events_of_type(model_input, EventType.COMMAND_POSITION_MISMATCH)
        if len({event.tick for event in mismatch}) >= 2:
            return (mismatch[0].fact_ref, mismatch[-1].fact_ref)
        return ()

    if slot is EvidenceSlot.UPSTREAM_DOWNSTREAM_DIVERGENCE:
        stable_upstream = _stable_series_refs(
            model_input,
            variables=frozenset({StateVariable.PRIMARY_FLOW, StateVariable.TRANSFER_EFFICIENCY}),
        )
        downstream_change = tuple(
            event.fact_ref
            for event in changed_observations
            if event.variable
            in {
                StateVariable.SECONDARY_FLOW,
                StateVariable.SECONDARY_INVENTORY,
                StateVariable.STEAM_STATE,
                StateVariable.TURBINE_OUTPUT,
                StateVariable.ELECTRICAL_OUTPUT,
            }
        )
        if stable_upstream and downstream_change:
            return _unique_refs(stable_upstream, downstream_change[-1:])
        stable_downstream = _stable_series_refs(
            model_input,
            variables=frozenset(
                {
                    StateVariable.STEAM_STATE,
                    StateVariable.TURBINE_OUTPUT,
                    StateVariable.ELECTRICAL_OUTPUT,
                }
            ),
        )
        upstream_change = tuple(
            event.fact_ref
            for event in changed_observations
            if event.variable
            in {
                StateVariable.PRIMARY_FLOW,
                StateVariable.TRANSFER_EFFICIENCY,
                StateVariable.SECONDARY_FLOW,
            }
        )
        if stable_downstream and upstream_change:
            return _unique_refs(upstream_change[-1:], stable_downstream)
        return ()

    if slot is EvidenceSlot.SECONDARY_TREND_MISMATCH:
        falling_secondary = _falling_event_refs(
            model_input, frozenset({StateVariable.SECONDARY_FLOW})
        )
        related_stable = _stable_series_refs(
            model_input,
            variables=frozenset(
                {
                    StateVariable.PRIMARY_FLOW,
                    StateVariable.STEAM_STATE,
                    StateVariable.ELECTRICAL_OUTPUT,
                }
            ),
        )
        if falling_secondary and related_stable:
            return _unique_refs(falling_secondary[-1:], related_stable)
        return ()

    if slot is EvidenceSlot.SUPPORT_BUS_CHANGE:
        support_changes = tuple(
            event.fact_ref
            for event in changed_observations
            if event.variable is StateVariable.SUPPORT_POWER
        )
        unavailable = tuple(
            event.fact_ref
            for event in component_changes
            if event.component_state_after is ComponentState.UNAVAILABLE
        )
        return _unique_refs(unavailable[:1], support_changes[:1])

    if slot is EvidenceSlot.MAPPED_COMPONENT_CHANGE:
        links = tuple(
            context
            for context in model_input.context_facts
            if isinstance(context, DependencyLinkContextFact)
        )
        changed_by_subject = {event.subject_id: event for event in component_changes}
        for link in links:
            changed = changed_by_subject.get(link.dependent_component_id)
            if changed is not None:
                return (link.fact_ref, changed.fact_ref)
        return ()

    if slot is EvidenceSlot.INVENTORY_TREND_DECLINING:
        return _falling_event_refs(
            model_input,
            frozenset({StateVariable.PRIMARY_INVENTORY, StateVariable.SECONDARY_INVENTORY}),
        )

    if slot is EvidenceSlot.MULTIPLE_CHANNELS_AGREE:
        return _matching_channel_pair_refs(model_input)

    if slot is EvidenceSlot.MISSING_DECISIVE_EVIDENCE:
        tentative_event_refs = tuple(
            event.fact_ref
            for event in events
            if event.event_type
            in {
                EventType.OBSERVATION_CHANGED,
                EventType.CHANNEL_QUALITY_CHANGED,
                EventType.CHANNEL_DISAGREEMENT,
                EventType.COMPONENT_STATE_CHANGED,
                EventType.COMMAND_POSITION_MISMATCH,
            }
        )
        tentative_observation_refs = tuple(
            fact.fact_ref
            for fact in observations
            if fact.tick == model_input.cut_tick
            and (
                fact.status is not ObservationStatus.NORMAL
                or fact.quality is not ChannelQuality.GOOD
            )
        )
        return _unique_refs(tentative_event_refs[-2:], tentative_observation_refs[-2:])

    if slot is EvidenceSlot.CONFLICTING_OBSERVATIONS:
        disagreement = tuple(
            event.fact_ref for event in events if event.event_type is EventType.CHANNEL_DISAGREEMENT
        )
        conflicting = tuple(
            fact.fact_ref for fact in observations if fact.status is ObservationStatus.CONFLICTING
        )
        if disagreement and conflicting:
            return _unique_refs(disagreement[-1:], conflicting[-2:])
        return ()

    raise ProjectionError(f"evidence slot has no closed visible matcher: {slot.value}")


def _task_target(
    *,
    task_name: TaskName,
    decision: DecisionTarget,
    model_input: ModelInput,
    event_facts: tuple[ProjectedEventFact, ...],
) -> ProjectionTaskTarget:
    if task_name is TaskName.FAULT_FAMILY:
        return make_fault_target(
            diagnosis_status=decision.diagnosis_status,
            fault_labels=decision.fault_labels,
            abstention_reason=decision.abstention_reason,
        )
    if task_name is TaskName.NEXT_ACTION:
        return make_action_target(action=decision.immediate_action)
    if task_name is TaskName.EXTRACT_EVIDENCE:
        slot_refs = tuple(
            (slot, visible_evidence_fact_refs(model_input, slot))
            for slot in decision.evidence_slots
        )
        visible_slots = tuple(slot for slot, refs in slot_refs if refs)
        refs = _unique_refs(*(refs for _, refs in slot_refs if refs))
        return make_evidence_target(
            fact_refs=refs,
            evidence_slots=visible_slots,
        )
    if task_name is TaskName.INCIDENT_SUMMARY:
        affected = tuple(
            subsystem
            for subsystem in AsterSubsystem
            if subsystem in {_SUBSYSTEM_BY_FAULT[fault] for fault in decision.fault_labels}
        )
        return make_incident_summary_target(
            affected_subsystems=affected,
            observed_trend=_observed_trend(event_facts),
            diagnosis_status=decision.diagnosis_status,
            fault_labels=decision.fault_labels,
            operating_mode=_operating_mode(event_facts),
            immediate_action=decision.immediate_action,
            abstention_reason=decision.abstention_reason,
        )
    if task_name is TaskName.CONTINUE_LOG:
        raise ProjectionError("continue_log requires project_continuation")
    if task_name is TaskName.COUNTERFACTUAL_COMPARE:
        raise ProjectionError("counterfactual_compare requires a complete grouped pair")
    raise ProjectionError(f"unsupported task: {task_name.value}")


def _record(
    *,
    trajectory: StructuredTrajectory,
    view: ProjectionView,
    model_input: ModelInput,
    task_target: ProjectionTaskTarget,
    decision_tick: int | None,
    source_event_index_exclusive: int | None,
    projection_recipe_id: str,
) -> ProjectionRecord:
    source_hash = canonical_sha256(trajectory.model_dump(mode="json", round_trip=True))
    fingerprint = model_input.structured_fingerprint()
    identity = canonical_sha256(
        {
            "trajectory_id": trajectory.trajectory_id,
            "view": view.value,
            "task": task_target.task_name.value,
            "decision_tick": decision_tick,
            "source_event_index_exclusive": source_event_index_exclusive,
            "projection_recipe_id": projection_recipe_id,
        }
    )
    lineage = ProjectionLineage(
        trajectory_id=trajectory.trajectory_id,
        scenario_id=trajectory.scenario_id,
        seed=trajectory.scenario.seed,
        decision_tick=decision_tick,
        source_event_index_exclusive=source_event_index_exclusive,
        task_name=task_target.task_name,
        projection_recipe_id=projection_recipe_id,
        source_trajectory_sha256=source_hash,
        provenance_sha256=trajectory.provenance.stable_hash(),
        structured_fingerprint_sha256=fingerprint,
    )
    projection_id = f"projection:{identity[:24]}"
    draft = ProjectionRecord.model_construct(
        schema_version=SCHEMA_VERSION,
        dataset_contract_version=DATASET_CONTRACT_VERSION,
        projection_id=projection_id,
        projection_view=view,
        lineage=lineage,
        model_input=model_input,
        task_target=task_target,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return ProjectionRecord(
        schema_version=SCHEMA_VERSION,
        dataset_contract_version=DATASET_CONTRACT_VERSION,
        projection_id=projection_id,
        projection_view=view,
        lineage=lineage,
        model_input=model_input,
        task_target=task_target,
        checksum_sha256=checksum,
    )


def project_trajectory(
    trajectory: StructuredTrajectory,
    *,
    decision_tick: int,
    task_name: TaskName,
    view: ProjectionView,
) -> ProjectionRecord:
    """Project one exact decision prefix under an explicitly declared safe view."""

    if type(trajectory) is not StructuredTrajectory:
        raise TypeError("trajectory must be a StructuredTrajectory")
    if type(decision_tick) is not int or decision_tick < 0:
        raise ProjectionError("decision_tick must be a non-negative integer")
    if type(task_name) is not TaskName:
        raise ProjectionError("task_name must be a TaskName")
    declared = _require_declared_view(trajectory, view)
    decision = _decision_at(trajectory, decision_tick)
    recipe = _projection_recipe(trajectory, decision_tick=decision_tick, view=declared)
    source_events = _decision_events(
        trajectory,
        cut_tick=decision_tick,
        view=declared,
        recipe=recipe,
    )
    event_facts = _project_event_facts(source_events)
    observation_facts = _project_observations(
        trajectory,
        cut_tick=decision_tick,
        view=declared,
        selected_events=source_events,
        recipe=recipe,
    )
    context_facts = _project_context(trajectory, declared, task_name)
    model_input = ModelInput(
        cut_tick=decision_tick,
        observation_facts=observation_facts,
        event_facts=event_facts,
        context_facts=context_facts,
    )
    target = _task_target(
        task_name=task_name,
        decision=decision,
        model_input=model_input,
        event_facts=event_facts,
    )
    return _record(
        trajectory=trajectory,
        view=declared,
        model_input=model_input,
        task_target=target,
        decision_tick=decision_tick,
        source_event_index_exclusive=None,
        projection_recipe_id=recipe.recipe_id,
    )


def project_continuation(
    trajectory: StructuredTrajectory,
    *,
    next_event_index: int,
    view: ProjectionView,
) -> ProjectionRecord:
    """Project an event-index prefix and its next non-action event target.

    Observation frames are intentionally omitted: an entire frame at the target
    event's tick could reveal a same-tick next event.  The event-index cut is kept
    in audit lineage and the renderer receives only the already-observed event facts.
    """

    if type(trajectory) is not StructuredTrajectory:
        raise TypeError("trajectory must be a StructuredTrajectory")
    if type(next_event_index) is not int or next_event_index <= 0:
        raise ProjectionError("next_event_index must identify an event after a non-empty prefix")
    if next_event_index >= len(trajectory.events):
        raise ProjectionError("next_event_index must be inside the event sequence")
    declared = _require_declared_view(trajectory, view)
    next_event = trajectory.events[next_event_index]
    if next_event.event_type is EventType.ACTION_APPLIED:
        raise ProjectionError("ACTION_APPLIED cannot be a continuation target")
    source_events = tuple(
        event
        for event in trajectory.events[:next_event_index]
        if event.event_type is not EventType.ACTION_APPLIED
    )
    if not source_events:
        raise ProjectionError("continuation prefix must contain a visible non-action event")
    event_facts = _project_event_facts(source_events)
    cut_tick = event_facts[-1].tick
    model_input = ModelInput(
        cut_tick=cut_tick,
        source_event_index_exclusive=next_event_index,
        observation_facts=(),
        event_facts=event_facts,
        context_facts=_project_context(trajectory, declared, TaskName.CONTINUE_LOG),
    )
    return _record(
        trajectory=trajectory,
        view=declared,
        model_input=model_input,
        task_target=make_continuation_target(next_event_type=next_event.event_type),
        decision_tick=None,
        source_event_index_exclusive=next_event_index,
        projection_recipe_id="continue-events-index-v1",
    )


def _conclusion(decision: DecisionTarget, *, model_input: ModelInput) -> CounterfactualConclusion:
    return CounterfactualConclusion(
        diagnosis_status=decision.diagnosis_status,
        fault_labels=decision.fault_labels,
        evidence_slots=tuple(
            slot
            for slot in decision.evidence_slots
            if visible_evidence_fact_refs(model_input, slot)
        ),
        immediate_action=decision.immediate_action,
        abstention_reason=decision.abstention_reason,
    )


def _changed_conclusion_fields(
    baseline: CounterfactualConclusion,
    counterfactual: CounterfactualConclusion,
) -> tuple[CounterfactualChange, ...]:
    comparisons = {
        CounterfactualChange.DIAGNOSIS_STATUS: (
            baseline.diagnosis_status,
            counterfactual.diagnosis_status,
        ),
        CounterfactualChange.FAULT_LABELS: (
            baseline.fault_labels,
            counterfactual.fault_labels,
        ),
        CounterfactualChange.EVIDENCE_SLOTS: (
            baseline.evidence_slots,
            counterfactual.evidence_slots,
        ),
        CounterfactualChange.IMMEDIATE_ACTION: (
            baseline.immediate_action,
            counterfactual.immediate_action,
        ),
    }
    return tuple(
        field for field in CounterfactualChange if comparisons[field][0] != comparisons[field][1]
    )


def _visible_fact_signatures(model_input: ModelInput) -> tuple[tuple[str, str], ...]:
    facts = (
        *model_input.observation_facts,
        *model_input.event_facts,
        *model_input.context_facts,
    )
    return tuple(
        (
            fact.fact_ref,
            canonical_sha256(fact.model_dump(mode="json", exclude={"fact_ref"})),
        )
        for fact in facts
    )


def project_counterfactual_pair(
    baseline_trajectory: StructuredTrajectory,
    counterfactual_trajectory: StructuredTrajectory,
    *,
    baseline_decision_tick: int,
    counterfactual_decision_tick: int,
    group: CounterfactualGroup,
    baseline_view: ProjectionView,
    counterfactual_view: ProjectionView,
) -> CounterfactualProjectionRecord:
    """Build a paired comparison only from a complete generator-supported group.

    Each group changes one preregistered causal factor.  That intervention may
    produce several renderer-visible consequences, so decisive reference counts
    are intentionally allowed to differ between the two members.
    """

    require_complete_group(group)
    member_ids = {member.scenario_id for member in group.members}
    selected_ids = {
        baseline_trajectory.scenario_id,
        counterfactual_trajectory.scenario_id,
    }
    if len(selected_ids) != 2 or not selected_ids.issubset(member_ids):
        raise ProjectionError("counterfactual pair must select two distinct members of its group")
    baseline_projection = project_trajectory(
        baseline_trajectory,
        decision_tick=baseline_decision_tick,
        task_name=TaskName.INCIDENT_SUMMARY,
        view=baseline_view,
    )
    counterfactual_projection = project_trajectory(
        counterfactual_trajectory,
        decision_tick=counterfactual_decision_tick,
        task_name=TaskName.INCIDENT_SUMMARY,
        view=counterfactual_view,
    )
    baseline_decision = _decision_at(baseline_trajectory, baseline_decision_tick)
    counterfactual_decision = _decision_at(counterfactual_trajectory, counterfactual_decision_tick)
    baseline_conclusion = _conclusion(
        baseline_decision, model_input=baseline_projection.model_input
    )
    counterfactual_conclusion = _conclusion(
        counterfactual_decision, model_input=counterfactual_projection.model_input
    )
    changed_fields = _changed_conclusion_fields(
        baseline_conclusion,
        counterfactual_conclusion,
    )
    if not changed_fields:
        raise ProjectionError("selected counterfactual decisions must have different conclusions")

    baseline_signatures = _visible_fact_signatures(baseline_projection.model_input)
    counterfactual_signatures = _visible_fact_signatures(counterfactual_projection.model_input)
    baseline_other = {signature for _, signature in counterfactual_signatures}
    counterfactual_other = {signature for _, signature in baseline_signatures}
    baseline_refs = tuple(
        fact_ref for fact_ref, signature in baseline_signatures if signature not in baseline_other
    )
    counterfactual_refs = tuple(
        fact_ref
        for fact_ref, signature in counterfactual_signatures
        if signature not in counterfactual_other
    )
    if not baseline_refs and not counterfactual_refs:
        raise ProjectionError("counterfactual conclusions differ without a visible changed fact")
    conclusion_slot_delta = set(baseline_conclusion.evidence_slots).symmetric_difference(
        counterfactual_conclusion.evidence_slots
    )
    decisive_slots = tuple(
        slot
        for slot in EvidenceSlot
        if slot in conclusion_slot_delta
        and bool(visible_evidence_fact_refs(baseline_projection.model_input, slot))
        != bool(visible_evidence_fact_refs(counterfactual_projection.model_input, slot))
    )
    target = ProjectionTaskTarget(
        task_name=TaskName.COUNTERFACTUAL_COMPARE,
        target=PromptCounterfactualComparisonTarget(
            baseline=baseline_conclusion,
            counterfactual=counterfactual_conclusion,
            changed_fields=changed_fields,
            baseline_decisive_fact_refs=baseline_refs,
            counterfactual_decisive_fact_refs=counterfactual_refs,
            decisive_evidence_slots=decisive_slots,
        ),
    )
    pair_input = CounterfactualPairInput(
        baseline=baseline_projection.model_input,
        counterfactual=counterfactual_projection.model_input,
    )
    identity = canonical_sha256(
        {
            "group_id": group.counterfactual_group_id,
            "baseline_projection_id": baseline_projection.projection_id,
            "counterfactual_projection_id": counterfactual_projection.projection_id,
        }
    )
    lineage = CounterfactualProjectionLineage(
        counterfactual_group_id=group.counterfactual_group_id,
        baseline_projection_id=baseline_projection.projection_id,
        counterfactual_projection_id=counterfactual_projection.projection_id,
    )
    pair_id = f"counterfactual:{identity[:24]}"
    draft = CounterfactualProjectionRecord.model_construct(
        schema_version=SCHEMA_VERSION,
        dataset_contract_version=DATASET_CONTRACT_VERSION,
        pair_id=pair_id,
        lineage=lineage,
        model_input=pair_input,
        task_target=target,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return CounterfactualProjectionRecord(
        schema_version=SCHEMA_VERSION,
        dataset_contract_version=DATASET_CONTRACT_VERSION,
        pair_id=pair_id,
        lineage=lineage,
        model_input=pair_input,
        task_target=target,
        checksum_sha256=checksum,
    )
