"""Acceptance tests for developmental G13 and G15 structured fixtures.

These tests intentionally describe the public scenario-builder contract before
the corresponding implementation is present.  They are kept separate from
the older simulator tests so the two new cases can mature without weakening
the already-reviewed Phase 2 fixtures.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from itertools import pairwise

import pytest

from reactorbench.schemas import (
    AbstentionReason,
    ActionLabel,
    CanonicalEvent,
    ChannelQuality,
    ComponentState,
    DecisionTarget,
    DiagnosisStatus,
    EventType,
    EvidenceSlot,
    FaultFamily,
    FaultInjection,
    ObservationStatus,
    OperatingMode,
    PlantVariant,
    ProvenanceRecord,
    ScenarioAction,
    ScenarioDefinition,
    SensorChannelObservation,
    SeverityBand,
    SplitName,
    StandbyContext,
    StateVariable,
    TaskName,
)
from reactorbench.simulator import (
    AsterVariantSpec,
    ChannelRole,
    UnsupportedScenarioError,
    build_abstract_inventory_loss_scenario,
    build_sparse_primary_flow_scenario,
    build_stable_scenario,
    dependency_map_context_for,
    generate_trace,
    get_variant_spec,
    scan_prohibited_content,
)
from reactorbench.simulator.core import SimulationTrace, _decision_from_process_evidence

_VARIANTS = (PlantVariant.ASTER_A, PlantVariant.ASTER_B, PlantVariant.ASTER_C)
_G13_MIN_DURATION = 10
_G15_MIN_DURATION = 8
_Builder = Callable[..., ScenarioDefinition]


def _spec_for(variant: PlantVariant) -> AsterVariantSpec:
    return get_variant_spec(variant)


def _channel(trace: SimulationTrace, tick: int, channel_id: str) -> SensorChannelObservation:
    return next(
        channel for channel in trace.observations[tick].channels if channel.channel_id == channel_id
    )


def _event_at(
    trace: SimulationTrace,
    *,
    tick: int,
    event_type: EventType,
    subject_id: str | None = None,
) -> CanonicalEvent:
    matches = tuple(
        event
        for event in trace.events
        if event.sim_time == tick
        and event.event_type is event_type
        and (subject_id is None or event.subject_id == subject_id)
    )
    assert len(matches) == 1
    return matches[0]


def _decision(trace: SimulationTrace, tick: int) -> DecisionTarget:
    return next(decision for decision in trace.targets.decisions if decision.decision_tick == tick)


def _event_actions(trace: SimulationTrace) -> tuple[tuple[int, ActionLabel], ...]:
    return tuple(
        (event.sim_time, event.action_label)
        for event in trace.events
        if event.event_type is EventType.ACTION_APPLIED and event.action_label is not None
    )


def _canonicalize_events(
    events: tuple[CanonicalEvent, ...], *, omitted_ids: set[str] | None = None
) -> tuple[CanonicalEvent, ...]:
    omitted = omitted_ids or set()
    kept = tuple(event for event in events if event.event_id not in omitted)
    remapped_ids = {event.event_id: f"e-{index:04d}" for index, event in enumerate(kept)}
    return tuple(
        event.model_copy(
            update={
                "event_id": remapped_ids[event.event_id],
                "event_index": index,
                "related_event_ids": tuple(
                    remapped_ids[related_id]
                    for related_id in event.related_event_ids
                    if related_id in remapped_ids
                ),
            }
        )
        for index, event in enumerate(kept)
    )


@pytest.mark.parametrize("variant", _VARIANTS)
def test_g13_builder_is_variant_scoped_and_has_the_fixed_schedule(
    variant: PlantVariant,
) -> None:
    scenario = build_abstract_inventory_loss_scenario(
        seed=23, duration_ticks=12, plant_variant=variant
    )
    spec = _spec_for(variant)

    assert scenario.plant_variant_id is variant
    assert scenario.driver.value == "STEADY_OPERATION"
    assert scenario.scenario_id.startswith(f"{variant.value.lower()}-inventory-loss-23-12-2-low-")
    assert scenario.scenario_id.endswith(spec.primary_loop_domain_id)
    assert len(scenario.fault_injections) == 1
    injection = scenario.fault_injections[0]
    assert injection.fault_family is FaultFamily.ABSTRACT_INVENTORY_LOSS
    assert injection.component_id == spec.primary_loop_domain_id
    assert injection.onset_tick == 2
    assert injection.severity.value == "LOW"
    assert injection.channel_id is None
    assert injection.duration_ticks is None
    assert scenario.action_sequence == (
        ScenarioAction(decision_tick=3, action=ActionLabel.INSUFFICIENT_EVIDENCE),
        ScenarioAction(decision_tick=6, action=ActionLabel.REDUCE_SIMULATED_LOAD),
        ScenarioAction(decision_tick=7, action=ActionLabel.ENTER_SIMULATED_STABLE_STATE),
    )


@pytest.mark.parametrize("variant", _VARIANTS)
def test_g13_latent_and_observation_causality_is_explicit(variant: PlantVariant) -> None:
    trace = generate_trace(
        build_abstract_inventory_loss_scenario(seed=23, duration_ticks=12, plant_variant=variant)
    )
    stable = generate_trace(
        build_stable_scenario(seed=23, duration_ticks=12, plant_variant=variant)
    )
    spec = _spec_for(variant)
    primary_pi = spec.channel_for(
        StateVariable.PRIMARY_INVENTORY, role=ChannelRole.PRIMARY
    ).channel_id
    redundant_pi = spec.channel_for(
        StateVariable.PRIMARY_INVENTORY, role=ChannelRole.REDUNDANT
    ).channel_id
    primary_pf = spec.channel_for(StateVariable.PRIMARY_FLOW, role=ChannelRole.PRIMARY).channel_id
    primary_pt = spec.channel_for(
        StateVariable.PRIMARY_THERMAL_STATE, role=ChannelRole.PRIMARY
    ).channel_id

    assert trace.latent_states[:3] == stable.latent_states[:3]
    assert trace.observations[:3] == stable.observations[:3]

    # The latent process starts at onset. Full frames carry both channels;
    # the redundant *event* follows one tick after the primary event.
    assert (
        trace.latent_states[3].values.primary_inventory
        < trace.latent_states[2].values.primary_inventory
    )
    assert (
        trace.latent_states[4].values.primary_inventory
        < trace.latent_states[3].values.primary_inventory
    )
    assert _channel(trace, 3, primary_pi) != _channel(stable, 3, primary_pi)
    assert _channel(trace, 3, redundant_pi) != _channel(stable, 3, redundant_pi)
    assert _channel(trace, 3, primary_pi).value == _channel(trace, 3, redundant_pi).value
    assert _channel(trace, 3, primary_pi).quality is ChannelQuality.GOOD
    assert _channel(trace, 3, redundant_pi).quality is ChannelQuality.GOOD
    assert _channel(trace, 4, redundant_pi) != _channel(stable, 4, redundant_pi)
    assert _channel(trace, 4, primary_pi).value == _channel(trace, 4, redundant_pi).value

    assert _channel(trace, 5, primary_pf) != _channel(stable, 5, primary_pf)
    assert _channel(trace, 6, primary_pt) != _channel(stable, 6, primary_pt)
    assert _channel(trace, 5, primary_pf).status is ObservationStatus.WATCH
    assert _channel(trace, 6, primary_pt).status is ObservationStatus.WATCH

    assert _decision(trace, 3).diagnosis_status is DiagnosisStatus.UNRESOLVED
    assert _decision(trace, 3).abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE
    assert _decision(trace, 3).immediate_action is ActionLabel.INSUFFICIENT_EVIDENCE
    assert _decision(trace, 3).fault_labels == ()
    mature = _decision(trace, 6)
    assert mature.diagnosis_status is DiagnosisStatus.DIAGNOSED
    assert mature.fault_labels == (FaultFamily.ABSTRACT_INVENTORY_LOSS,)
    assert {
        EvidenceSlot.INVENTORY_TREND_DECLINING,
        EvidenceSlot.MULTIPLE_CHANNELS_AGREE,
        EvidenceSlot.FLOW_DECLINING,
        EvidenceSlot.CORRELATED_STATE_CHANGE,
    }.issubset(mature.evidence_slots)
    assert mature.immediate_action is ActionLabel.REDUCE_SIMULATED_LOAD
    assert _decision(trace, 7).immediate_action is ActionLabel.ENTER_SIMULATED_STABLE_STATE

    assert _event_actions(trace) == (
        (4, ActionLabel.INSUFFICIENT_EVIDENCE),
        (7, ActionLabel.REDUCE_SIMULATED_LOAD),
        (8, ActionLabel.ENTER_SIMULATED_STABLE_STATE),
    )
    assert trace.latent_states[8].operating_mode is OperatingMode.RECOVERY
    assert trace.latent_states[9].operating_mode is OperatingMode.STABILIZED
    assert trace.latent_states[7].values.load_demand < trace.latent_states[6].values.load_demand
    assert (
        trace.latent_states[7].values.heat_source_level
        < trace.latent_states[6].values.heat_source_level
    )
    assert (
        trace.latent_states[8].values.primary_inventory
        == trace.latent_states[7].values.primary_inventory
    )
    assert all(
        state.values.primary_inventory == trace.latent_states[7].values.primary_inventory
        for state in trace.latent_states[8:]
    )

    events_by_id = {event.event_id: event for event in trace.events}
    assert tuple(event.event_index for event in trace.events) == tuple(range(len(trace.events)))
    for event in trace.events:
        assert all(
            events_by_id[related].event_index < event.event_index
            for related in event.related_event_ids
        )
    first_pi = _event_at(
        trace,
        tick=3,
        event_type=EventType.OBSERVATION_CHANGED,
        subject_id=primary_pi,
    )
    second_pi = _event_at(
        trace,
        tick=4,
        event_type=EventType.OBSERVATION_CHANGED,
        subject_id=redundant_pi,
    )
    assert first_pi.event_index < second_pi.event_index
    assert first_pi.sim_time < second_pi.sim_time


@pytest.mark.parametrize("variant", _VARIANTS)
def test_g13_values_are_normalized_and_steps_are_bounded(variant: PlantVariant) -> None:
    trace = generate_trace(
        build_abstract_inventory_loss_scenario(
            seed=2**32 - 1, duration_ticks=64, plant_variant=variant
        )
    )
    spec = _spec_for(variant)
    for state in trace.latent_states:
        assert all(0.0 <= value <= 1.0 for value in state.values.model_dump().values())
    for previous, current in pairwise(trace.latent_states):
        for variable in StateVariable:
            before = getattr(previous.values, variable.value)
            after = getattr(current.values, variable.value)
            assert abs(after - before) <= spec.max_per_tick_step + 1e-9
    assert scan_prohibited_content(trace) == ()


def test_g13_provenance_is_audit_only_and_visible_payload_is_truth_free() -> None:
    trace = generate_trace(build_abstract_inventory_loss_scenario(seed=23, duration_ticks=12))
    payload = json.dumps(trace.visible_payload(), sort_keys=True)
    for forbidden in (
        "ABSTRACT_INVENTORY_LOSS",
        "fault_family",
        "fault_injection",
        "onset_tick",
        "scenario_id",
        "latent_states",
        "targets",
        "provenance",
    ):
        assert forbidden not in payload

    trajectory = trace.to_structured_trajectory(
        trajectory_id="g13-trace-23",
        provenance=ProvenanceRecord(
            dataset_version="0.1.0",
            generator_commit="abcdef1",
            renderer_version="0.1.0",
            seed=23,
            trajectory_id="g13-trace-23",
            scenario_id=trace.scenario.scenario_id,
            plant_variant_id=trace.scenario.plant_variant_id,
            fault_family_ids=(FaultFamily.ABSTRACT_INVENTORY_LOSS,),
            template_family_ids=("template-g13",),
            split_name=SplitName.COMPOSITION_TEST,
            task_name=TaskName.FAULT_FAMILY,
        ),
    )
    assert trajectory.provenance.fault_family_ids == (FaultFamily.ABSTRACT_INVENTORY_LOSS,)
    assert trajectory.targets.decisions[-1].fault_labels == (FaultFamily.ABSTRACT_INVENTORY_LOSS,)


def test_g13_visible_evidence_helper_abstains_after_canonical_ablations() -> None:
    trace = generate_trace(build_abstract_inventory_loss_scenario(seed=23, duration_ticks=12))
    prefix = tuple(event for event in trace.events if event.sim_time <= 6)
    complete = _decision_from_process_evidence(
        scenario_id=trace.scenario.scenario_id,
        decision_tick=6,
        events=_canonicalize_events(prefix),
    )
    assert complete.diagnosis_status is DiagnosisStatus.DIAGNOSED
    assert complete.fault_labels == (FaultFamily.ABSTRACT_INVENTORY_LOSS,)

    redundant_id = next(
        event.event_id
        for event in prefix
        if event.sim_time == 4 and event.variable is StateVariable.PRIMARY_INVENTORY
    )
    thermal_id = next(
        event.event_id
        for event in prefix
        if event.sim_time == 6 and event.variable is StateVariable.PRIMARY_THERMAL_STATE
    )
    for omitted_id in (redundant_id, thermal_id):
        ablated = _decision_from_process_evidence(
            scenario_id=trace.scenario.scenario_id,
            decision_tick=6,
            events=_canonicalize_events(prefix, omitted_ids={omitted_id}),
        )
        assert ablated.diagnosis_status is DiagnosisStatus.UNRESOLVED
        assert ablated.fault_labels == ()
        assert ablated.evidence_event_ids == ()
        assert ablated.evidence_slots == (EvidenceSlot.MISSING_DECISIVE_EVIDENCE,)
        assert ablated.immediate_action is ActionLabel.INSUFFICIENT_EVIDENCE
        assert ablated.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE

    reverse_time_prefix = tuple(
        event.model_copy(
            update={
                "event_id": f"e-{index:04d}",
                "event_index": index,
                "related_event_ids": (),
            }
        )
        for index, event in enumerate(reversed(prefix))
    )
    with pytest.raises(ValueError, match="ordered visible-event prefix"):
        _decision_from_process_evidence(
            scenario_id="neutral-visible-prefix",
            decision_tick=6,
            events=reverse_time_prefix,
        )


@pytest.mark.parametrize("variant", _VARIANTS)
def test_g15_exposes_one_sparse_primary_flow_cell_without_latent_truth(
    variant: PlantVariant,
) -> None:
    sparse = generate_trace(
        build_sparse_primary_flow_scenario(seed=23, duration_ticks=8, plant_variant=variant)
    )
    stable = generate_trace(build_stable_scenario(seed=23, duration_ticks=8, plant_variant=variant))
    spec = _spec_for(variant)
    selected = spec.channels_for(StateVariable.PRIMARY_FLOW)[23 % 2].channel_id

    assert sparse.scenario.plant_variant_id is variant
    assert sparse.scenario.scenario_id.startswith(f"{variant.value.lower()}-")
    assert "sparse" in sparse.scenario.scenario_id
    assert "primary-flow" in sparse.scenario.scenario_id
    assert sparse.scenario.fault_injections == ()
    assert sparse.latent_states == stable.latent_states
    assert len(sparse.observations) == 8
    expected_channels = tuple(sorted(channel.channel_id for channel in spec.channels))
    assert all(
        tuple(channel.channel_id for channel in frame.channels) == expected_channels
        for frame in sparse.observations
    )
    for tick, (sparse_frame, stable_frame) in enumerate(
        zip(sparse.observations, stable.observations, strict=True)
    ):
        if tick == 2:
            sparse_cell = _channel(sparse, tick, selected)
            stable_cell = _channel(stable, tick, selected)
            assert sparse_cell != stable_cell
            assert sparse_cell.status is ObservationStatus.WATCH
            assert sparse_cell.quality is ChannelQuality.GOOD
            assert sparse_cell.value is not None
            assert stable_cell.value is not None
            fictional_offset = stable_cell.value - sparse_cell.value
            assert 0.018 - 1e-6 <= fictional_offset <= 0.024 + 1e-6
            assert all(
                _channel(sparse, tick, channel.channel_id)
                == _channel(stable, tick, channel.channel_id)
                for channel in spec.channels
                if channel.channel_id != selected
            )
        else:
            assert sparse_frame.channels == stable_frame.channels

    assert _decision(sparse, 2).diagnosis_status is DiagnosisStatus.UNRESOLVED
    assert _decision(sparse, 2).fault_labels == ()
    assert _decision(sparse, 2).abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE
    assert _decision(sparse, 2).immediate_action is ActionLabel.INSUFFICIENT_EVIDENCE
    assert _event_actions(sparse) == ((3, ActionLabel.INSUFFICIENT_EVIDENCE),)
    assert all(
        event.event_type
        in {EventType.BENIGN_NOTE, EventType.OBSERVATION_CHANGED, EventType.ACTION_APPLIED}
        for event in sparse.events
    )
    assert all(
        event.sim_time <= 2
        for event in sparse.events
        if event.event_type is not EventType.ACTION_APPLIED
    )


@pytest.mark.parametrize("variant", _VARIANTS)
def test_g15_is_seeded_prefix_preserving_and_global_rng_independent(variant: PlantVariant) -> None:
    for seed in (0, 2**32 - 1):
        first = generate_trace(
            build_sparse_primary_flow_scenario(seed=seed, duration_ticks=8, plant_variant=variant)
        )
        second = generate_trace(
            build_sparse_primary_flow_scenario(seed=seed, duration_ticks=8, plant_variant=variant)
        )
        assert first == second
        random.seed(11)
        third = generate_trace(
            build_sparse_primary_flow_scenario(seed=seed, duration_ticks=8, plant_variant=variant)
        )
        random.seed(91)
        fourth = generate_trace(
            build_sparse_primary_flow_scenario(seed=seed, duration_ticks=8, plant_variant=variant)
        )
        assert third == fourth


@pytest.mark.parametrize(
    "builder", [build_abstract_inventory_loss_scenario, build_sparse_primary_flow_scenario]
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"seed": 23.0},
        {"seed": -1},
        {"seed": 2**32},
        {"duration_ticks": True},
        {"duration_ticks": 12.0},
        {"plant_variant": "ASTER-A"},
        {"plant_variant": PlantVariant.ASTER_A.value},
    ],
)
def test_new_builders_reject_noncanonical_inputs(
    builder: _Builder, kwargs: dict[str, object]
) -> None:
    arguments: dict[str, object] = {"seed": 23, "duration_ticks": 12}
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError, UnsupportedScenarioError)):
        builder(**arguments)


def test_new_builders_reject_out_of_window_durations_and_unknown_kwargs() -> None:
    with pytest.raises((TypeError, ValueError)):
        build_abstract_inventory_loss_scenario(seed=23, duration_ticks=_G13_MIN_DURATION - 1)
    with pytest.raises((TypeError, ValueError)):
        build_sparse_primary_flow_scenario(seed=23, duration_ticks=_G15_MIN_DURATION - 1)
    for builder in (build_abstract_inventory_loss_scenario, build_sparse_primary_flow_scenario):
        with pytest.raises(TypeError):
            builder(  # type: ignore[call-arg]
                seed=23, duration_ticks=12, channel_id="caller-selected"
            )


def test_model_copy_tampering_fails_closed_for_both_new_scenarios() -> None:
    g13 = build_abstract_inventory_loss_scenario(seed=23, duration_ticks=12)
    g15 = build_sparse_primary_flow_scenario(seed=23, duration_ticks=8)
    for scenario in (g13, g15):
        spec = _spec_for(scenario.plant_variant_id)
        active_train, standby_train = spec.primary_train_ids
        standby_context = StandbyContext(
            context_id=f"{scenario.plant_variant_id.value.lower()}-test-standby",
            active_train_id=active_train,
            standby_train_id=standby_train,
            standby_state=ComponentState.AVAILABLE,
            standby_support_bus_id=spec.support_for(standby_train),
            support_bus_state=ComponentState.AVAILABLE,
            standby_start_delay_ticks=1,
        )
        dependency_context = dependency_map_context_for(spec)
        first_action = scenario.action_sequence[0]
        malformed: tuple[ScenarioDefinition, ...] = (
            scenario.model_copy(update={"scenario_id": "caller-lookalike"}),
            scenario.model_copy(update={"driver": "STEADY_OPERATION"}),
            scenario.model_copy(
                update={
                    "action_sequence": (
                        first_action.model_copy(
                            update={"decision_tick": first_action.decision_tick + 1}
                        ),
                        *scenario.action_sequence[1:],
                    )
                }
            ),
            scenario.model_copy(
                update={
                    "action_sequence": (
                        first_action.model_copy(update={"action": "INSUFFICIENT_EVIDENCE"}),
                        *scenario.action_sequence[1:],
                    )
                }
            ),
            scenario.model_copy(update={"action_sequence": [*scenario.action_sequence]}),
            scenario.model_copy(update={"fault_injections": [*scenario.fault_injections]}),
            scenario.model_copy(update={"standby_context": standby_context}),
            scenario.model_copy(update={"dependency_map_context": dependency_context}),
            scenario.model_copy(update={"plant_variant_id": PlantVariant.ASTER_B}),
        )
        if scenario.fault_injections:
            injection = scenario.fault_injections[0]
            malformed += (
                scenario.model_copy(
                    update={
                        "fault_injections": (
                            injection.model_copy(
                                update={"component_id": spec.primary_train_ids[0]}
                            ),
                        )
                    }
                ),
                scenario.model_copy(
                    update={"fault_injections": (injection.model_copy(update={"onset_tick": 3}),)}
                ),
                scenario.model_copy(
                    update={
                        "fault_injections": (injection.model_copy(update={"severity": "MEDIUM"}),)
                    }
                ),
                scenario.model_copy(
                    update={
                        "fault_injections": (
                            injection.model_copy(
                                update={
                                    "channel_id": spec.channels_for(
                                        StateVariable.PRIMARY_INVENTORY
                                    )[0].channel_id
                                }
                            ),
                        )
                    }
                ),
                scenario.model_copy(
                    update={
                        "fault_injections": (injection.model_copy(update={"duration_ticks": 1}),)
                    }
                ),
            )
        else:
            sparse_injection = FaultInjection(
                fault_family=FaultFamily.SENSOR_DRIFT,
                component_id=spec.instrumentation_id,
                onset_tick=2,
                severity=SeverityBand.LOW,
                channel_id=spec.channels_for(StateVariable.PRIMARY_FLOW)[23 % 2].channel_id,
            )
            malformed += (
                scenario.model_copy(update={"fault_injections": (sparse_injection,)}),
                scenario.model_copy(update={"action_sequence": ()}),
                scenario.model_copy(
                    update={
                        "action_sequence": (
                            first_action.model_copy(
                                update={"action": ActionLabel.CONTINUE_MONITORING}
                            ),
                        )
                    }
                ),
            )
        for candidate in malformed:
            with pytest.raises(UnsupportedScenarioError):
                generate_trace(candidate)
