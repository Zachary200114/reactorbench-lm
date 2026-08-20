"""Public contract and fail-closed tests for developmental G12."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from reactorbench.schemas import (
    ActionLabel,
    ComponentState,
    DependencyLink,
    DependencyMapContext,
    FaultFamily,
    PlantVariant,
    ProvenanceRecord,
    ScenarioAction,
    ScenarioDefinition,
    SeverityBand,
    SplitName,
    StandbyContext,
    TaskName,
)
from reactorbench.simulator import (
    ASTER_A_SPEC,
    ASTER_B_SPEC,
    ComponentRole,
    UnsupportedScenarioError,
    build_support_power_interruption_scenario,
    dependency_map_context_for,
    generate_trace,
)
from reactorbench.simulator.core import SimulationTrace


def _bus_id(variant: PlantVariant) -> str:
    spec = ASTER_A_SPEC if variant is PlantVariant.ASTER_A else ASTER_B_SPEC
    return spec.component_for_role(ComponentRole.SUPPORT_BUS_TWO).component_id


def _valid_trace(*, include_dependency_map: bool = True) -> SimulationTrace:
    return generate_trace(
        build_support_power_interruption_scenario(
            seed=23,
            duration_ticks=12,
            plant_variant=PlantVariant.ASTER_B,
            include_dependency_map=include_dependency_map,
        )
    )


def test_visible_payload_schema_and_context_are_explicitly_allowlisted() -> None:
    for include_dependency_map in (True, False):
        trace = _valid_trace(include_dependency_map=include_dependency_map)
        payload = trace.visible_payload()
        assert set(payload) == {
            "schema_version",
            "plant_variant_id",
            "dependency_map_context",
            "standby_context",
            "observations",
            "events",
        }
        assert payload["plant_variant_id"] == PlantVariant.ASTER_B.value
        assert payload["standby_context"] is None
        context = payload["dependency_map_context"]
        if include_dependency_map:
            assert context == dependency_map_context_for(ASTER_B_SPEC).model_dump(mode="json")
            assert set(context) == {"plant_variant_id", "links"}
            assert all(
                set(link) == {"support_bus_id", "dependent_component_id"}
                for link in context["links"]
            )
        else:
            assert context is None


def test_visible_payload_cannot_leak_injection_driver_targets_or_provenance() -> None:
    for include_dependency_map in (True, False):
        serialized = json.dumps(
            _valid_trace(include_dependency_map=include_dependency_map).visible_payload()
        )
        for forbidden in (
            "SUPPORT_POWER_INTERRUPTION",
            "fault_family",
            "fault_injection",
            "driver",
            "severity",
            "onset_tick",
            "duration_ticks",
            "scenario_id",
            "latent_states",
            "targets",
            "provenance",
            "pending_maintenance",
        ):
            assert forbidden not in serialized


def test_structured_trajectory_preserves_truth_only_in_audit_source() -> None:
    trace = _valid_trace()
    trajectory_id = "g12-contract-trace-23"
    trajectory = trace.to_structured_trajectory(
        trajectory_id=trajectory_id,
        provenance=ProvenanceRecord(
            dataset_version="0.1.0",
            generator_commit="abcdef1",
            renderer_version="0.1.0",
            seed=23,
            trajectory_id=trajectory_id,
            scenario_id=trace.scenario.scenario_id,
            plant_variant_id=PlantVariant.ASTER_B,
            fault_family_ids=(FaultFamily.SUPPORT_POWER_INTERRUPTION,),
            template_family_ids=("template-g12",),
            split_name=SplitName.COMPOSITION_TEST,
            task_name=TaskName.FAULT_FAMILY,
        ),
    )
    assert trajectory.targets.decisions[-1].fault_labels == (
        FaultFamily.SUPPORT_POWER_INTERRUPTION,
    )
    assert "SUPPORT_POWER_INTERRUPTION" not in json.dumps(trace.visible_payload())
    assert trajectory.provenance.fault_family_ids == (FaultFamily.SUPPORT_POWER_INTERRUPTION,)


def test_events_are_causal_and_action_is_applied_one_tick_after_decision() -> None:
    trace = _valid_trace()
    event_by_id = {event.event_id: event for event in trace.events}
    assert tuple(event.event_index for event in trace.events) == tuple(range(len(trace.events)))
    assert tuple(event.sim_time for event in trace.events) == tuple(
        sorted(event.sim_time for event in trace.events)
    )
    for event in trace.events:
        assert all(
            event_by_id[related].event_index < event.event_index
            for related in event.related_event_ids
        )

    decision = next(item for item in trace.targets.decisions if item.decision_tick == 5)
    assert decision.immediate_action is ActionLabel.ENTER_SIMULATED_STABLE_STATE
    applied = next(
        event
        for event in trace.events
        if event.action_label is ActionLabel.ENTER_SIMULATED_STABLE_STATE
    )
    assert applied.sim_time == decision.decision_tick + 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"plant_variant": PlantVariant.ASTER_C},
        {"plant_variant": "ASTER-B"},
        {"seed": True},
        {"seed": 23.0},
        {"seed": -1},
        {"seed": 2**32},
        {"duration_ticks": True},
        {"duration_ticks": 12.0},
        {"duration_ticks": 7},
        {"duration_ticks": 65},
        {"include_dependency_map": 1},
        {"include_dependency_map": None},
    ],
)
def test_builder_rejects_out_of_scope_or_noncanonical_inputs(kwargs: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "seed": 23,
        "duration_ticks": 12,
        "plant_variant": PlantVariant.ASTER_B,
        "include_dependency_map": True,
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError, UnsupportedScenarioError)):
        cast(Any, build_support_power_interruption_scenario)(**arguments)


def test_builder_does_not_accept_a_caller_supplied_dependency_map() -> None:
    with pytest.raises(TypeError):
        build_support_power_interruption_scenario(  # type: ignore[call-arg, unused-ignore]
            seed=23,
            duration_ticks=12,
            plant_variant=PlantVariant.ASTER_B,
            dependency_map={"aster-b-bus-sonnet": ("caller-component",)},
        )


def test_model_copy_tampering_and_malformed_containers_fail_closed() -> None:
    scenario = build_support_power_interruption_scenario(
        seed=23, duration_ticks=12, plant_variant=PlantVariant.ASTER_B
    )
    bus_id = _bus_id(PlantVariant.ASTER_B)
    wrong_bus = ASTER_B_SPEC.component_for_role(ComponentRole.SUPPORT_BUS_ONE).component_id
    injection = scenario.fault_injections[0]

    def rejected(candidate: ScenarioDefinition) -> None:
        with pytest.raises(UnsupportedScenarioError):
            generate_trace(candidate)

    for update in (
        {"component_id": wrong_bus},
        {"component_id": "sonnet"},
        {"onset_tick": 3},
        {"severity": SeverityBand.MEDIUM},
        {"severity": "LOW"},
        {"duration_ticks": 1},
        {"channel_id": "boreal-primary-flow-a"},
        {"channel_id": 1},
    ):
        tampered_injection = injection.model_copy(update=update)
        rejected(scenario.model_copy(update={"fault_injections": (tampered_injection,)}))

    rejected(scenario.model_copy(update={"fault_injections": [injection]}))
    rejected(
        scenario.model_copy(update={"scenario_id": "aster-b-support-power-interruption-lookalike"})
    )
    rejected(
        scenario.model_copy(
            update={
                "standby_context": StandbyContext(
                    context_id="g12-standby",
                    active_train_id="aster-b-train-nomad",
                    standby_train_id="aster-b-train-saffron",
                    standby_state=ComponentState.AVAILABLE,
                    standby_support_bus_id="aster-b-bus-sonnet",
                    support_bus_state=ComponentState.AVAILABLE,
                    standby_start_delay_ticks=2,
                )
            }
        )
    )

    action = scenario.action_sequence[0]
    rejected(
        scenario.model_copy(
            update={"action_sequence": (ScenarioAction(decision_tick=4, action=action.action),)}
        )
    )
    rejected(
        scenario.model_copy(
            update={"action_sequence": (action.model_copy(update={"action": "NOT_AN_ACTION"}),)}
        )
    )
    rejected(scenario.model_copy(update={"action_sequence": [action]}))
    rejected(scenario.model_copy(update={"action_sequence": ()}))

    assert injection.component_id == bus_id


def test_context_cannot_be_cross_variant_or_structurally_rewritten() -> None:
    scenario = build_support_power_interruption_scenario(
        seed=23, duration_ticks=12, plant_variant=PlantVariant.ASTER_B
    )
    context = dependency_map_context_for(ASTER_A_SPEC)
    with pytest.raises((ValueError, UnsupportedScenarioError)):
        generate_trace(scenario.model_copy(update={"dependency_map_context": context}))

    assert isinstance(scenario.dependency_map_context, DependencyMapContext)
    links = scenario.dependency_map_context.links
    contexts = (
        scenario.dependency_map_context.model_copy(update={"links": links[:-1]}),
        scenario.dependency_map_context.model_copy(update={"links": tuple(reversed(links))}),
        scenario.dependency_map_context.model_copy(
            update={"plant_variant_id": PlantVariant.ASTER_A}
        ),
        scenario.dependency_map_context.model_copy(
            update={
                "links": (
                    DependencyLink(
                        support_bus_id="aster-b-bus-sonnet",
                        dependent_component_id="caller-component",
                    ),
                )
            }
        ),
        scenario.dependency_map_context.model_copy(
            update={
                "links": (DependencyLink(support_bus_id="sonnet", dependent_component_id="nomad"),)
            }
        ),
        scenario.dependency_map_context.model_copy(update={"links": ()}),
    )
    for context in contexts:
        with pytest.raises((ValueError, UnsupportedScenarioError)):
            generate_trace(scenario.model_copy(update={"dependency_map_context": context}))

    with pytest.raises(UnsupportedScenarioError):
        generate_trace(scenario.model_copy(update={"dependency_map_context": None}))


def test_matched_context_pair_has_same_group_key_and_different_visible_context() -> None:
    included = build_support_power_interruption_scenario(
        seed=23,
        duration_ticks=12,
        plant_variant=PlantVariant.ASTER_B,
        include_dependency_map=True,
    )
    withheld = build_support_power_interruption_scenario(
        seed=23,
        duration_ticks=12,
        plant_variant=PlantVariant.ASTER_B,
        include_dependency_map=False,
    )
    assert included.scenario_id.rsplit("-map-", 1)[0] == withheld.scenario_id.rsplit("-map-", 1)[0]
    assert generate_trace(included).visible_payload()["dependency_map_context"] is not None
    assert generate_trace(withheld).visible_payload()["dependency_map_context"] is None


def test_state_remains_unavailable_after_the_selected_action() -> None:
    trace = _valid_trace()
    bus_id = _bus_id(PlantVariant.ASTER_B)
    dependent_ids = set(ASTER_B_SPEC.dependents_for(bus_id))
    expected = {bus_id, *dependent_ids}
    for tick in (6, 7, 11):
        unavailable = {
            component.component_id
            for component in trace.latent_states[tick].components
            if component.state is ComponentState.UNAVAILABLE
        }
        assert unavailable == expected
