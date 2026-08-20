"""Contract tests for the developmental G12 support-power scenario."""

from __future__ import annotations

import json

import pytest

from reactorbench.schemas import (
    ActionLabel,
    CanonicalEvent,
    ComponentLatentState,
    ComponentState,
    DiagnosisStatus,
    EventType,
    EvidenceSlot,
    FaultFamily,
    OperatingMode,
    PlantVariant,
    ScenarioAction,
    StateVariable,
)
from reactorbench.simulator import (
    ASTER_A_SPEC,
    ASTER_B_SPEC,
    AsterVariantSpec,
    ComponentRole,
    UnsupportedScenarioError,
    build_stable_scenario,
    build_support_power_interruption_scenario,
    dependency_map_context_for,
    generate_trace,
    scan_prohibited_content,
)
from reactorbench.simulator.core import SimulationTrace


def _bus_id(variant: PlantVariant) -> str:
    spec = ASTER_A_SPEC if variant is PlantVariant.ASTER_A else ASTER_B_SPEC
    return spec.component_for_role(ComponentRole.SUPPORT_BUS_TWO).component_id


def _spec_for(variant: PlantVariant) -> AsterVariantSpec:
    return ASTER_A_SPEC if variant is PlantVariant.ASTER_A else ASTER_B_SPEC


def _component(trace: SimulationTrace, tick: int, component_id: str) -> ComponentLatentState:
    return next(
        component
        for component in trace.latent_states[tick].components
        if component.component_id == component_id
    )


def _value(trace: SimulationTrace, tick: int, variable: StateVariable) -> float:
    return float(getattr(trace.latent_states[tick].values, variable.value))


def _events_at(trace: SimulationTrace, tick: int) -> tuple[CanonicalEvent, ...]:
    return tuple(event for event in trace.events if event.sim_time == tick)


def _first_event(
    trace: SimulationTrace, *, tick: int, event_type: EventType, subject_id: str
) -> CanonicalEvent:
    return next(
        event
        for event in _events_at(trace, tick)
        if event.event_type is event_type and event.subject_id == subject_id
    )


def _unavailable_ids(trace: SimulationTrace, tick: int) -> set[str]:
    return {
        component.component_id
        for component in trace.latent_states[tick].components
        if component.state is ComponentState.UNAVAILABLE
    }


def _state_change_ticks(trace: SimulationTrace, component_id: str) -> tuple[int, ...]:
    return tuple(
        state.tick
        for state, previous in zip(trace.latent_states[1:], trace.latent_states[:-1], strict=True)
        if _component(trace, state.tick, component_id).state
        != _component(trace, previous.tick, component_id).state
    )


@pytest.mark.parametrize("variant", [PlantVariant.ASTER_A, PlantVariant.ASTER_B])
@pytest.mark.parametrize("include_dependency_map", [True, False])
def test_builder_emits_strict_variant_scoped_scenario(
    variant: PlantVariant, include_dependency_map: bool
) -> None:
    scenario = build_support_power_interruption_scenario(
        seed=23,
        duration_ticks=12,
        plant_variant=variant,
        include_dependency_map=include_dependency_map,
    )
    spec = _spec_for(variant)
    bus_id = _bus_id(variant)

    assert scenario.plant_variant_id is variant
    assert scenario.driver.value == "STEADY_OPERATION"
    assert scenario.duration_ticks == 12
    assert scenario.scenario_id.startswith(
        f"{variant.value.lower()}-support-power-interruption-23-12-2-low-{bus_id}"
    )
    assert scenario.scenario_id.endswith(
        "-map-included" if include_dependency_map else "-map-withheld"
    )
    assert len(scenario.fault_injections) == 1
    injection = scenario.fault_injections[0]
    assert injection.fault_family is FaultFamily.SUPPORT_POWER_INTERRUPTION
    assert injection.component_id == bus_id
    assert injection.onset_tick == 2
    assert injection.severity.value == "LOW"
    assert injection.channel_id is None
    assert injection.duration_ticks is None
    expected_action = (
        ActionLabel.ENTER_SIMULATED_STABLE_STATE
        if include_dependency_map
        else ActionLabel.INSUFFICIENT_EVIDENCE
    )
    assert scenario.action_sequence == (ScenarioAction(decision_tick=5, action=expected_action),)
    if include_dependency_map:
        assert scenario.dependency_map_context == dependency_map_context_for(spec)
    else:
        assert scenario.dependency_map_context is None


@pytest.mark.parametrize("variant", [PlantVariant.ASTER_A, PlantVariant.ASTER_B])
def test_bus_event_precedes_exact_mapped_component_availability_loss(variant: PlantVariant) -> None:
    scenario = build_support_power_interruption_scenario(
        seed=23, duration_ticks=12, plant_variant=variant
    )
    trace = generate_trace(scenario)
    spec = _spec_for(variant)
    bus_id = _bus_id(variant)
    dependent_ids = set(spec.dependents_for(bus_id))
    all_component_ids = {component.component_id for component in spec.components}

    assert _state_change_ticks(trace, bus_id) == (2,)
    assert _unavailable_ids(trace, 2) == {bus_id}
    assert _unavailable_ids(trace, 3) == {bus_id, *dependent_ids}
    assert _unavailable_ids(trace, 7) == {bus_id, *dependent_ids}
    assert all(
        _component(trace, 3, component_id).state is ComponentState.AVAILABLE
        for component_id in all_component_ids - {bus_id} - dependent_ids
    )

    bus_event = _first_event(
        trace,
        tick=2,
        event_type=EventType.COMPONENT_STATE_CHANGED,
        subject_id=bus_id,
    )
    dependent_events = tuple(
        _first_event(
            trace,
            tick=3,
            event_type=EventType.COMPONENT_STATE_CHANGED,
            subject_id=component_id,
        )
        for component_id in sorted(dependent_ids)
    )
    assert bus_event.component_state_before is ComponentState.AVAILABLE
    assert bus_event.component_state_after is ComponentState.UNAVAILABLE
    assert all(
        event.component_state_after is ComponentState.UNAVAILABLE for event in dependent_events
    )
    assert bus_event.event_index < min(event.event_index for event in dependent_events)
    assert bus_event.sim_time < min(event.sim_time for event in dependent_events)


@pytest.mark.parametrize("variant", [PlantVariant.ASTER_A, PlantVariant.ASTER_B])
def test_role_derived_effects_are_bounded_and_delayed(variant: PlantVariant) -> None:
    trace = generate_trace(
        build_support_power_interruption_scenario(seed=23, duration_ticks=12, plant_variant=variant)
    )
    stable = generate_trace(
        build_stable_scenario(seed=23, duration_ticks=12, plant_variant=variant)
    )
    spec = _spec_for(variant)
    bus_id = _bus_id(variant)
    dependent_ids = set(spec.dependents_for(bus_id))
    expected_primary_flow_delta = 0.0

    support_power_drop = _value(stable, 2, StateVariable.SUPPORT_POWER) - _value(
        trace, 2, StateVariable.SUPPORT_POWER
    )
    assert 0.018 <= support_power_drop <= 0.024

    for component_id in dependent_ids:
        role = next(item.role for item in spec.components if item.component_id == component_id)
        if role in {ComponentRole.PRIMARY_TRAIN_ONE, ComponentRole.PRIMARY_TRAIN_TWO}:
            expected_primary_flow_delta -= 0.010
        elif role is ComponentRole.PRIMARY_FLOW_VALVE:
            expected_primary_flow_delta -= 0.008
        elif role is ComponentRole.TRANSFER_UNIT:
            assert _value(trace, 4, StateVariable.TRANSFER_EFFICIENCY) == pytest.approx(
                _value(stable, 4, StateVariable.TRANSFER_EFFICIENCY) - 0.012
            )
            assert _value(trace, 5, StateVariable.PRIMARY_THERMAL_STATE) == pytest.approx(
                _value(stable, 5, StateVariable.PRIMARY_THERMAL_STATE) + 0.008
            )
            assert _value(trace, 5, StateVariable.STEAM_STATE) == pytest.approx(
                _value(stable, 5, StateVariable.STEAM_STATE) - 0.008
            )
        elif role is ComponentRole.SECONDARY_FEED:
            assert _value(trace, 4, StateVariable.SECONDARY_FLOW) == pytest.approx(
                _value(stable, 4, StateVariable.SECONDARY_FLOW) - 0.012
            )
            assert _value(trace, 5, StateVariable.SECONDARY_INVENTORY) == pytest.approx(
                _value(stable, 5, StateVariable.SECONDARY_INVENTORY) - 0.008
            )

    assert _value(trace, 4, StateVariable.PRIMARY_FLOW) == pytest.approx(
        _value(stable, 4, StateVariable.PRIMARY_FLOW) + expected_primary_flow_delta
    )

    assert all(
        abs(after - before) <= spec.max_per_tick_step + 1e-9
        for before, after in zip(
            (_value(trace, tick, StateVariable.PRIMARY_FLOW) for tick in range(11)),
            (_value(trace, tick, StateVariable.PRIMARY_FLOW) for tick in range(1, 12)),
            strict=True,
        )
    )


@pytest.mark.parametrize("variant", [PlantVariant.ASTER_A, PlantVariant.ASTER_B])
def test_causal_edges_follow_only_relevant_dependency_roles(variant: PlantVariant) -> None:
    trace = generate_trace(
        build_support_power_interruption_scenario(seed=23, duration_ticks=12, plant_variant=variant)
    )
    spec = _spec_for(variant)
    bus_id = _bus_id(variant)
    dependent_ids = set(spec.dependents_for(bus_id))
    event_by_id = {event.event_id: event for event in trace.events}

    def related_dependents(event: CanonicalEvent) -> set[str]:
        return {
            event_by_id[related_id].subject_id
            for related_id in event.related_event_ids
            if event_by_id[related_id].subject_id in dependent_ids
        }

    roles_by_id = {component.component_id: component.role for component in spec.components}
    primary_flow = next(
        event
        for event in trace.events
        if event.sim_time == 4
        and event.event_type is EventType.OBSERVATION_CHANGED
        and event.variable is StateVariable.PRIMARY_FLOW
    )
    assert related_dependents(primary_flow) == {
        component_id
        for component_id, role in roles_by_id.items()
        if component_id in dependent_ids
        and role
        in {
            ComponentRole.PRIMARY_TRAIN_ONE,
            ComponentRole.PRIMARY_TRAIN_TWO,
            ComponentRole.PRIMARY_FLOW_VALVE,
        }
    }
    transfer_events = tuple(
        event
        for event in trace.events
        if event.sim_time == 4
        and event.event_type is EventType.OBSERVATION_CHANGED
        and event.variable is StateVariable.TRANSFER_EFFICIENCY
    )
    if ComponentRole.TRANSFER_UNIT in {roles_by_id[component_id] for component_id in dependent_ids}:
        assert len(transfer_events) == 1
        assert related_dependents(transfer_events[0]) == {
            component_id
            for component_id in dependent_ids
            if roles_by_id[component_id] is ComponentRole.TRANSFER_UNIT
        }
    else:
        assert transfer_events == ()
    secondary_events = tuple(
        event
        for event in trace.events
        if event.sim_time == 4
        and event.event_type is EventType.OBSERVATION_CHANGED
        and event.variable is StateVariable.SECONDARY_FLOW
    )
    if ComponentRole.SECONDARY_FEED in {
        roles_by_id[component_id] for component_id in dependent_ids
    }:
        assert len(secondary_events) == 1
        assert related_dependents(secondary_events[0]) == {
            component_id
            for component_id in dependent_ids
            if roles_by_id[component_id] is ComponentRole.SECONDARY_FEED
        }
    else:
        assert secondary_events == ()


def test_map_context_changes_only_diagnosis_and_action_after_decision() -> None:
    included = generate_trace(
        build_support_power_interruption_scenario(
            seed=41,
            duration_ticks=12,
            plant_variant=PlantVariant.ASTER_B,
            include_dependency_map=True,
        )
    )
    withheld = generate_trace(
        build_support_power_interruption_scenario(
            seed=41,
            duration_ticks=12,
            plant_variant=PlantVariant.ASTER_B,
            include_dependency_map=False,
        )
    )

    assert included.latent_states[:6] == withheld.latent_states[:6]
    assert included.observations[:6] == withheld.observations[:6]
    assert tuple(event for event in included.events if event.sim_time <= 5) == tuple(
        event for event in withheld.events if event.sim_time <= 5
    )
    assert (
        included.scenario.scenario_id.rsplit("-map-", 1)[0]
        == withheld.scenario.scenario_id.rsplit("-map-", 1)[0]
    )

    included_decision = next(
        decision for decision in included.targets.decisions if decision.decision_tick == 5
    )
    withheld_decision = next(
        decision for decision in withheld.targets.decisions if decision.decision_tick == 5
    )
    assert included_decision.diagnosis_status is DiagnosisStatus.DIAGNOSED
    assert included_decision.fault_labels == (FaultFamily.SUPPORT_POWER_INTERRUPTION,)
    assert included_decision.evidence_slots == (
        EvidenceSlot.SUPPORT_BUS_CHANGE,
        EvidenceSlot.MAPPED_COMPONENT_CHANGE,
        EvidenceSlot.DEPENDENT_TREND_DELAY,
    )
    assert withheld_decision.evidence_event_ids == included_decision.evidence_event_ids
    assert withheld_decision.evidence_slots == included_decision.evidence_slots
    assert EvidenceSlot.MISSING_DECISIVE_EVIDENCE not in {
        slot for event in included.events for slot in event.evidence_slots
    }
    assert included_decision.immediate_action is ActionLabel.ENTER_SIMULATED_STABLE_STATE
    assert withheld_decision.diagnosis_status is DiagnosisStatus.UNRESOLVED
    assert withheld_decision.fault_labels == ()
    assert withheld_decision.abstention_reason is not None
    assert withheld_decision.immediate_action is ActionLabel.INSUFFICIENT_EVIDENCE

    applied = tuple(
        event
        for event in included.events
        if event.event_type is EventType.ACTION_APPLIED
        and event.action_label is ActionLabel.ENTER_SIMULATED_STABLE_STATE
    )
    assert len(applied) == 1
    assert applied[0].sim_time == 6
    assert included.latent_states[6].operating_mode is OperatingMode.RECOVERY
    stable = generate_trace(
        build_stable_scenario(seed=41, duration_ticks=12, plant_variant=PlantVariant.ASTER_B)
    )
    assert included.latent_states[6].values.load_demand == pytest.approx(
        stable.latent_states[6].values.load_demand - 0.012
    )
    assert included.latent_states[6].values.heat_source_level == pytest.approx(
        stable.latent_states[6].values.heat_source_level - 0.012
    )
    assert included.latent_states[7].operating_mode is OperatingMode.STABILIZED
    withheld_abstention_applied = tuple(
        event
        for event in withheld.events
        if event.event_type is EventType.ACTION_APPLIED
        and event.action_label is ActionLabel.INSUFFICIENT_EVIDENCE
    )
    assert len(withheld_abstention_applied) == 1
    assert withheld_abstention_applied[0].sim_time == 6
    assert withheld.latent_states[6].values == withheld.latent_states[5].values
    assert withheld.latent_states[6].components == withheld.latent_states[5].components
    assert withheld.latent_states[6].operating_mode is OperatingMode.DISTURBED
    assert not any(
        event.event_type is EventType.ACTION_APPLIED
        and event.action_label is ActionLabel.ENTER_SIMULATED_STABLE_STATE
        for event in withheld.events
    )


def test_visible_payload_has_only_allowlisted_context_and_no_fault_truth() -> None:
    for include_dependency_map in (True, False):
        trace = generate_trace(
            build_support_power_interruption_scenario(
                seed=23,
                duration_ticks=12,
                plant_variant=PlantVariant.ASTER_B,
                include_dependency_map=include_dependency_map,
            )
        )
        payload = trace.visible_payload()
        serialized = json.dumps(payload, sort_keys=True)
        assert set(payload) == {
            "schema_version",
            "plant_variant_id",
            "dependency_map_context",
            "standby_context",
            "observations",
            "events",
        }
        assert payload["standby_context"] is None
        context = payload["dependency_map_context"]
        if include_dependency_map:
            assert isinstance(context, dict)
            assert set(context) == {"plant_variant_id", "links"}
            assert context["plant_variant_id"] == PlantVariant.ASTER_B.value
            assert all(
                set(link) == {"support_bus_id", "dependent_component_id"}
                for link in context["links"]
            )
        else:
            assert context is None
        for forbidden in (
            "SUPPORT_POWER_INTERRUPTION",
            "fault_family",
            "fault_injection",
            "driver",
            "severity",
            "onset_tick",
            "scenario_id",
            "latent",
            "targets",
            "provenance",
        ):
            assert forbidden not in serialized


def test_generated_trace_passes_prohibited_content_gate() -> None:
    trace = generate_trace(
        build_support_power_interruption_scenario(
            seed=0, duration_ticks=8, plant_variant=PlantVariant.ASTER_A
        )
    )
    assert scan_prohibited_content(trace) == ()


def test_model_copy_and_raw_context_mutations_fail_closed() -> None:
    scenario = build_support_power_interruption_scenario(seed=23, duration_ticks=12)
    wrong_bus = ASTER_A_SPEC.component_for_role(ComponentRole.SUPPORT_BUS_ONE).component_id
    tampered_injection = scenario.fault_injections[0].model_copy(update={"component_id": wrong_bus})
    tampered = scenario.model_copy(update={"fault_injections": (tampered_injection,)})
    with pytest.raises(UnsupportedScenarioError):
        generate_trace(tampered)

    with pytest.raises(TypeError):
        build_support_power_interruption_scenario(  # type: ignore[call-arg, unused-ignore]
            seed=23, duration_ticks=12, dependency_map={"arbitrary": "caller-data"}
        )


def test_map_variant_context_is_immutable_and_role_derived() -> None:
    scenario = build_support_power_interruption_scenario(
        seed=23, duration_ticks=12, plant_variant=PlantVariant.ASTER_B
    )
    assert scenario.dependency_map_context == dependency_map_context_for(ASTER_B_SPEC)
    assert tuple(
        (link.support_bus_id, link.dependent_component_id)
        for link in scenario.dependency_map_context.links  # type: ignore[union-attr, unused-ignore]
    ) == tuple(
        sorted(
            (link.supplier_component_id, link.dependent_component_id)
            for link in ASTER_B_SPEC.dependency_links
        )
    )


def test_map_group_has_same_generation_key_for_split_assignment() -> None:
    included = build_support_power_interruption_scenario(
        seed=99, duration_ticks=12, plant_variant=PlantVariant.ASTER_A, include_dependency_map=True
    )
    withheld = build_support_power_interruption_scenario(
        seed=99, duration_ticks=12, plant_variant=PlantVariant.ASTER_A, include_dependency_map=False
    )
    assert included.scenario_id.rsplit("-map-", 1)[0] == withheld.scenario_id.rsplit("-map-", 1)[0]
    assert included.seed == withheld.seed
    assert included.duration_ticks == withheld.duration_ticks
    assert included.plant_variant_id is withheld.plant_variant_id
