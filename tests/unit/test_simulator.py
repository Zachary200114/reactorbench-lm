from __future__ import annotations

import random
from collections.abc import Callable
from itertools import pairwise
from typing import cast

import pytest

from reactorbench.schemas import (
    AbstentionReason,
    ActionLabel,
    CanonicalEvent,
    ChannelQuality,
    ComponentState,
    DiagnosisStatus,
    EventType,
    EvidenceSlot,
    FaultFamily,
    ObservationStatus,
    OperatingMode,
    PlantVariant,
    ScenarioAction,
    ScenarioDefinition,
    ScenarioDriver,
    SeverityBand,
    StateVariable,
)
from reactorbench.schemas.base import canonical_sha256
from reactorbench.simulator import (
    ASTER_A_SPEC,
    ASTER_B_SPEC,
    ASTER_C_SPEC,
    VARIANT_REGISTRY,
    AsterVariantSpec,
    SimulationTrace,
    UnsupportedScenarioError,
    build_flow_imbalance_scenario,
    build_load_transient_scenario,
    build_pump_degradation_scenario,
    build_pump_trip_scenario,
    build_sensor_drift_scenario,
    build_sensor_noise_scenario,
    build_sensor_stuck_load_scenario,
    build_stable_scenario,
    build_transfer_efficiency_loss_scenario,
    build_valve_lag_scenario,
    build_valve_stuck_scenario,
    dependency_map_context_for,
    generate_trace,
    get_variant_spec,
    scan_prohibited_content,
)
from reactorbench.simulator.core import _decision_from_process_evidence


def _channel_values(trace: SimulationTrace, channel_id: str) -> tuple[float | None, ...]:
    return tuple(
        next(channel.value for channel in frame.channels if channel.channel_id == channel_id)
        for frame in trace.observations
    )


def test_stable_trace_is_bounded_available_and_resolved() -> None:
    trace = generate_trace(build_stable_scenario(seed=19, duration_ticks=10))

    assert len(trace.latent_states) == len(trace.observations) == 10
    assert trace.scenario.fault_injections == ()
    assert trace.targets.decisions[-1].diagnosis_status is DiagnosisStatus.NO_FAULT
    assert trace.targets.decisions[-1].immediate_action is ActionLabel.CONTINUE_MONITORING
    first_latent = trace.latent_states[0]
    assert all(
        0.425 <= value <= 0.515
        for latent in trace.latent_states
        for value in latent.values.model_dump().values()
    )
    assert all(
        latent.operating_mode == first_latent.operating_mode
        and latent.values == first_latent.values
        and latent.components == first_latent.components
        for latent in trace.latent_states
    )
    assert all(
        component.state is ComponentState.AVAILABLE and component.health == 1.0
        for latent in trace.latent_states
        for component in latent.components
    )
    assert all(len(frame.channels) == 2 * len(StateVariable) for frame in trace.observations)
    assert all(frame.overall_status is ObservationStatus.NORMAL for frame in trace.observations)
    for frame in trace.observations:
        for variable in StateVariable:
            pair = [channel for channel in frame.channels if channel.variable is variable]
            assert len(pair) == 2
            assert pair[0].value == pair[1].value
        assert all(
            channel.status is ObservationStatus.NORMAL and channel.quality is ChannelQuality.GOOD
            for channel in frame.channels
        )


@pytest.mark.parametrize(
    ("builder", "fault", "first_action", "second_action"),
    [
        (
            build_transfer_efficiency_loss_scenario,
            FaultFamily.TRANSFER_EFFICIENCY_LOSS,
            ActionLabel.INSUFFICIENT_EVIDENCE,
            ActionLabel.REDUCE_SIMULATED_LOAD,
        ),
        (
            build_flow_imbalance_scenario,
            FaultFamily.FLOW_IMBALANCE,
            ActionLabel.COMPARE_RELATED_TRENDS,
            ActionLabel.ENTER_SIMULATED_STABLE_STATE,
        ),
    ],
)
def test_g10_g11_variant_builders_are_deterministic_bounded_and_causal(
    builder: Callable[..., ScenarioDefinition],
    fault: FaultFamily,
    first_action: ActionLabel,
    second_action: ActionLabel,
) -> None:
    for variant in PlantVariant:
        scenario = builder(seed=20, duration_ticks=12, plant_variant=variant)
        trace = generate_trace(scenario)
        max_step = get_variant_spec(variant).max_per_tick_step
        stable = generate_trace(
            build_stable_scenario(seed=20, duration_ticks=12, plant_variant=variant)
        )

        assert trace == generate_trace(scenario)
        assert trace.latent_states[:2] == stable.latent_states[:2]
        assert trace.observations[:2] == stable.observations[:2]
        assert all(
            component.state is ComponentState.AVAILABLE and component.health == 1.0
            for state in trace.latent_states
            for component in state.components
        )
        for variable in StateVariable:
            values = [getattr(state.values, variable.value) for state in trace.latent_states]
            assert all(abs(second - first) <= max_step + 1e-9 for first, second in pairwise(values))
        decisions = trace.targets.decisions
        assert decisions[-1].fault_labels == (fault,)
        assert decisions[-1].immediate_action is second_action
        if fault is FaultFamily.TRANSFER_EFFICIENCY_LOSS:
            assert decisions[0].diagnosis_status is DiagnosisStatus.UNRESOLVED
            assert decisions[0].immediate_action is first_action
            assert (
                trace.latent_states[2].values.transfer_efficiency
                < stable.latent_states[2].values.transfer_efficiency
            )
        else:
            assert decisions[0].immediate_action is first_action
            assert (
                trace.latent_states[2].values.secondary_flow
                < stable.latent_states[2].values.secondary_flow
            )


def test_g10_g11_direct_model_copies_fail_closed() -> None:
    transfer = build_transfer_efficiency_loss_scenario(seed=7, plant_variant=PlantVariant.ASTER_B)
    copied = transfer.model_copy(
        update={"scenario_id": "aster-b-transfer-efficiency-loss-7-12-2-low-foreign"}
    )
    with pytest.raises(UnsupportedScenarioError):
        generate_trace(copied)

    imbalance = build_flow_imbalance_scenario(seed=7, plant_variant=PlantVariant.ASTER_C)
    cross_variant = imbalance.model_copy(update={"plant_variant_id": PlantVariant.ASTER_A})
    with pytest.raises(UnsupportedScenarioError):
        generate_trace(cross_variant)

    for builder in (build_transfer_efficiency_loss_scenario, build_flow_imbalance_scenario):
        for raw_variant in ("ASTER-A", True, 1.0):
            with pytest.raises(ValueError, match="plant_variant"):
                builder(seed=7, plant_variant=cast(PlantVariant, raw_variant))
        for malformed_duration in (True, 8.0):
            with pytest.raises(ValueError, match="duration_ticks"):
                builder(seed=7, duration_ticks=cast(int, malformed_duration))

        base = builder(seed=7)
        injection = base.fault_injections[0]
        wrong_action_tick = base.action_sequence[0].model_copy(
            update={"decision_tick": base.action_sequence[0].decision_tick + 1}
        )
        invalid_copies = (
            base.model_copy(update={"scenario_id": "noncanonical-process-case"}),
            base.model_copy(update={"seed": True}),
            base.model_copy(update={"seed": 7.0}),
            base.model_copy(update={"fault_injections": [injection]}),
            base.model_copy(
                update={
                    "fault_injections": (
                        injection.model_copy(update={"component_id": "unknown-process-part"}),
                    )
                }
            ),
            base.model_copy(
                update={
                    "fault_injections": (
                        injection.model_copy(update={"channel_id": "bad-channel"}),
                    )
                }
            ),
            base.model_copy(
                update={"fault_injections": (injection.model_copy(update={"severity": "LOW"}),)}
            ),
            base.model_copy(
                update={"fault_injections": (injection.model_copy(update={"onset_tick": 3}),)}
            ),
            base.model_copy(
                update={"fault_injections": (injection.model_copy(update={"duration_ticks": 1}),)}
            ),
            base.model_copy(update={"standby_context": {}}),
            base.model_copy(update={"dependency_map_context": {}}),
            base.model_copy(update={"action_sequence": [base.action_sequence[0]]}),
            base.model_copy(
                update={"action_sequence": (wrong_action_tick, base.action_sequence[1])}
            ),
            base.model_copy(
                update={
                    "action_sequence": (
                        ScenarioAction(
                            decision_tick=base.action_sequence[0].decision_tick,
                            action=ActionLabel.INSUFFICIENT_EVIDENCE,
                        ).model_copy(update={"action": "INSUFFICIENT_EVIDENCE"}),
                        base.action_sequence[1],
                    )
                }
            ),
        )
        for copied_shape in invalid_copies:
            with pytest.raises(UnsupportedScenarioError):
                generate_trace(copied_shape)


def _canonical_prefix_without(
    events: tuple[CanonicalEvent, ...], *, event_id: str
) -> tuple[CanonicalEvent, ...]:
    """Reindex an ablated visible prefix without retaining hidden-event references."""

    retained = [event for event in events if getattr(event, "event_id", None) != event_id]
    id_map = {event.event_id: f"e-{index:04d}" for index, event in enumerate(retained)}
    return tuple(
        CanonicalEvent.model_validate(
            {
                **event.model_dump(mode="python"),
                "event_id": id_map[event.event_id],
                "event_index": index,
                "related_event_ids": tuple(
                    id_map[related_id]
                    for related_id in event.related_event_ids
                    if related_id in id_map
                ),
            }
        )
        for index, event in enumerate(retained)
    )


def test_g10_g11_visible_evidence_helper_is_prefix_only_and_scenario_id_neutral() -> None:
    transfer = generate_trace(build_transfer_efficiency_loss_scenario(seed=23))
    imbalance = generate_trace(build_flow_imbalance_scenario(seed=23))
    cases = (
        (
            transfer,
            5,
            FaultFamily.TRANSFER_EFFICIENCY_LOSS,
            ActionLabel.REDUCE_SIMULATED_LOAD,
        ),
        (imbalance, 4, FaultFamily.FLOW_IMBALANCE, ActionLabel.COMPARE_RELATED_TRENDS),
        (imbalance, 6, FaultFamily.FLOW_IMBALANCE, ActionLabel.ENTER_SIMULATED_STABLE_STATE),
    )
    for trace, tick, family, action in cases:
        prefix = tuple(event for event in trace.events if event.sim_time <= tick)
        neutral = _decision_from_process_evidence(
            scenario_id="neutral-visible-prefix", decision_tick=tick, events=prefix
        )
        spoofed = _decision_from_process_evidence(
            scenario_id="spoofed-visible-prefix", decision_tick=tick, events=prefix
        )
        assert neutral.fault_labels == spoofed.fault_labels == (family,)
        assert neutral.immediate_action is spoofed.immediate_action is action
        assert neutral.scenario_id != spoofed.scenario_id

        ablated = _decision_from_process_evidence(
            scenario_id="neutral-visible-prefix",
            decision_tick=tick,
            events=prefix[:-1],
        )
        assert ablated.diagnosis_status is DiagnosisStatus.UNRESOLVED
        assert ablated.fault_labels == ()
        assert ablated.immediate_action is ActionLabel.INSUFFICIENT_EVIDENCE

        with pytest.raises(TypeError):
            _decision_from_process_evidence(
                scenario_id="neutral-visible-prefix",
                decision_tick=tick,
                events=list(prefix),  # type: ignore[arg-type]
            )
        with pytest.raises(TypeError):
            _decision_from_process_evidence(
                scenario_id="neutral-visible-prefix",
                decision_tick=tick,
                events=({"event_id": "e-0000"},),  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="ordered visible-event prefix"):
            _decision_from_process_evidence(
                scenario_id="neutral-visible-prefix",
                decision_tick=tick,
                events=trace.events,
            )

    transfer_prefix = tuple(event for event in transfer.events if event.sim_time <= 5)
    transfer_te = next(
        event
        for event in transfer_prefix
        if event.sim_time == 2 and event.variable is StateVariable.TRANSFER_EFFICIENCY
    )
    assert (
        _decision_from_process_evidence(
            scenario_id="neutral-visible-prefix",
            decision_tick=5,
            events=_canonical_prefix_without(transfer_prefix, event_id=transfer_te.event_id),
        ).diagnosis_status
        is DiagnosisStatus.UNRESOLVED
    )

    imbalance_prefix = tuple(event for event in imbalance.events if event.sim_time <= 4)
    imbalance_persistence = next(
        event
        for event in imbalance_prefix
        if event.sim_time == 4 and event.variable is StateVariable.SECONDARY_INVENTORY
    )
    assert (
        _decision_from_process_evidence(
            scenario_id="neutral-visible-prefix",
            decision_tick=4,
            events=_canonical_prefix_without(
                imbalance_prefix, event_id=imbalance_persistence.event_id
            ),
        ).diagnosis_status
        is DiagnosisStatus.UNRESOLVED
    )


def test_g10_g11_duration_prefixes_and_global_rng_state_are_stable() -> None:
    original_state = random.getstate()
    try:
        random.seed(7331)
        before = random.getstate()
        for builder in (
            build_transfer_efficiency_loss_scenario,
            build_flow_imbalance_scenario,
        ):
            for variant in PlantVariant:
                traces = [
                    generate_trace(builder(seed=20, duration_ticks=duration, plant_variant=variant))
                    for duration in (8, 12, 64)
                ]
                assert random.getstate() == before
                assert (
                    traces[0].latent_states[:8]
                    == traces[1].latent_states[:8]
                    == traces[2].latent_states[:8]
                )
                assert (
                    traces[0].observations[:8]
                    == traces[1].observations[:8]
                    == traces[2].observations[:8]
                )
                prefixes = [
                    tuple(event for event in trace.events if event.sim_time <= 7)
                    for trace in traces
                ]
                assert prefixes[0] == prefixes[1] == prefixes[2]
    finally:
        random.setstate(original_state)


def test_load_transient_is_benign_coordinated_and_causally_ordered() -> None:
    scenario = build_load_transient_scenario(seed=20, duration_ticks=12)
    trace = generate_trace(scenario)
    stable = generate_trace(build_stable_scenario(seed=20, duration_ticks=12))

    assert scenario.driver is ScenarioDriver.LOAD_TRANSIENT
    assert scenario.fault_injections == ()
    decision = trace.targets.decisions[-1]
    assert decision.diagnosis_status is DiagnosisStatus.NO_FAULT
    assert decision.fault_labels == ()
    assert decision.immediate_action is ActionLabel.CONTINUE_MONITORING
    assert all(
        component.state is ComponentState.AVAILABLE
        for state in trace.latent_states
        for component in state.components
    )
    assert all(frame.overall_status is ObservationStatus.NORMAL for frame in trace.observations)
    assert all(
        channel.status is ObservationStatus.NORMAL and channel.quality is ChannelQuality.GOOD
        for frame in trace.observations
        for channel in frame.channels
    )
    assert all(latent.operating_mode is OperatingMode.STABLE for latent in trace.latent_states[:2])
    assert all(
        latent.operating_mode is OperatingMode.LOAD_CHANGE for latent in trace.latent_states[2:5]
    )
    assert all(
        latent.operating_mode is OperatingMode.LOAD_CHANGE for latent in trace.latent_states[5:7]
    )
    assert all(latent.operating_mode is OperatingMode.STABLE for latent in trace.latent_states[7:])
    assert trace.latent_states[:2] == stable.latent_states[:2]
    assert trace.observations[:2] == stable.observations[:2]
    assert trace.latent_states[-1].values.load_demand > stable.latent_states[-1].values.load_demand
    for variable in (
        StateVariable.HEAT_SOURCE_LEVEL,
        StateVariable.PRIMARY_FLOW,
        StateVariable.STEAM_STATE,
        StateVariable.TURBINE_OUTPUT,
        StateVariable.ELECTRICAL_OUTPUT,
    ):
        assert getattr(trace.latent_states[-1].values, variable.value) > getattr(
            stable.latent_states[-1].values, variable.value
        )
    assert all(
        transient.values.transfer_efficiency == baseline.values.transfer_efficiency
        for transient, baseline in zip(trace.latent_states, stable.latent_states, strict=True)
    )
    for frame in trace.observations:
        for variable in StateVariable:
            pair = [channel for channel in frame.channels if channel.variable is variable]
            assert len(pair) == 2
            assert pair[0].value == pair[1].value
    assert [(event.sim_time, event.event_type.value) for event in trace.events] == [
        (0, "BENIGN_NOTE"),
        (2, "TARGET_CHANGED"),
        (2, "OPERATING_MODE_CHANGED"),
        (6, "BENIGN_NOTE"),
        (7, "OPERATING_MODE_CHANGED"),
    ]
    assert not any(event.action_label is not None for event in trace.events)
    target_event, coordinated_event = trace.events[1], trace.events[3]
    assert target_event.sim_time < coordinated_event.sim_time
    assert coordinated_event.related_event_ids == (target_event.event_id,)


def test_load_transient_stages_latent_response_in_causal_order() -> None:
    transient = generate_trace(build_load_transient_scenario(seed=20, duration_ticks=12))
    stable = generate_trace(build_stable_scenario(seed=20, duration_ticks=12))

    def first_change(variable: StateVariable) -> int:
        return next(
            state.tick
            for state, baseline in zip(transient.latent_states, stable.latent_states, strict=True)
            if getattr(state.values, variable.value) != getattr(baseline.values, variable.value)
        )

    assert first_change(StateVariable.LOAD_DEMAND) == 2
    assert first_change(StateVariable.HEAT_SOURCE_LEVEL) == 2
    assert first_change(StateVariable.PRIMARY_FLOW) == 2
    assert first_change(StateVariable.STEAM_STATE) == 3
    assert first_change(StateVariable.TURBINE_OUTPUT) == 4
    assert first_change(StateVariable.ELECTRICAL_OUTPUT) == 4


def test_load_transient_direction_is_seed_derived_and_reversible() -> None:
    rising = generate_trace(build_load_transient_scenario(seed=20))
    falling = generate_trace(build_load_transient_scenario(seed=21))
    rising_stable = generate_trace(build_stable_scenario(seed=20))
    falling_stable = generate_trace(build_stable_scenario(seed=21))

    assert (
        rising.latent_states[-1].values.load_demand
        > rising_stable.latent_states[-1].values.load_demand
    )
    assert (
        falling.latent_states[-1].values.load_demand
        < falling_stable.latent_states[-1].values.load_demand
    )


def test_load_transient_builder_and_generator_fail_closed() -> None:
    with pytest.raises(ValueError, match="between"):
        build_load_transient_scenario(seed=1, duration_ticks=65)
    scenario = build_load_transient_scenario(seed=4)
    drift_injection = build_sensor_drift_scenario(seed=4).fault_injections[0]
    unsupported = scenario.model_copy(update={"fault_injections": (drift_injection,)})

    with pytest.raises(UnsupportedScenarioError, match="cannot include"):
        generate_trace(unsupported)


def test_sensor_stuck_load_preserves_latent_truth_and_freezes_one_channel() -> None:
    scenario = build_sensor_stuck_load_scenario(seed=20, duration_ticks=12)
    stuck = generate_trace(scenario)
    load = generate_trace(build_load_transient_scenario(seed=20, duration_ticks=12))
    injection = scenario.fault_injections[0]
    selected_channel = injection.channel_id
    assert selected_channel is not None

    assert stuck.latent_states == load.latent_states
    frozen_value = _channel_values(load, selected_channel)[1]
    for stuck_frame, load_frame in zip(stuck.observations, load.observations, strict=True):
        for stuck_channel, load_channel in zip(
            stuck_frame.channels, load_frame.channels, strict=True
        ):
            if stuck_channel.channel_id != selected_channel:
                assert stuck_channel == load_channel
            elif stuck_frame.tick < 2:
                assert stuck_channel == load_channel
            else:
                assert stuck_channel.value == frozen_value
    selected = [
        next(channel for channel in frame.channels if channel.channel_id == selected_channel)
        for frame in stuck.observations
    ]
    assert [channel.status for channel in selected[:4]] == [ObservationStatus.NORMAL] * 4
    assert selected[4].status is ObservationStatus.WATCH
    assert all(channel.status is ObservationStatus.CONFLICTING for channel in selected[5:])
    assert [frame.overall_status for frame in stuck.observations[:4]] == [
        ObservationStatus.NORMAL
    ] * 4
    assert stuck.observations[4].overall_status is ObservationStatus.WATCH
    assert all(
        frame.overall_status is ObservationStatus.CONFLICTING for frame in stuck.observations[5:]
    )
    assert all(channel.quality is ChannelQuality.GOOD for channel in selected[:7])
    assert selected[7].quality is ChannelQuality.SUSPECT
    assert all(
        decision.diagnosis_status is DiagnosisStatus.DIAGNOSED
        and decision.fault_labels == (FaultFamily.SENSOR_STUCK,)
        for decision in stuck.targets.decisions
    )


def test_sensor_stuck_load_events_are_causal_and_evidence_backed() -> None:
    trace = generate_trace(build_sensor_stuck_load_scenario(seed=21, duration_ticks=12))
    events_by_id = {event.event_id: event for event in trace.events}

    assert [
        (event.sim_time, event.event_type.value, event.action_label) for event in trace.events
    ] == [
        (0, "BENIGN_NOTE", None),
        (2, "TARGET_CHANGED", None),
        (2, "OPERATING_MODE_CHANGED", None),
        (5, "OBSERVATION_CHANGED", None),
        (5, "CHANNEL_DISAGREEMENT", None),
        (6, "ACTION_APPLIED", ActionLabel.VERIFY_REDUNDANT_CHANNEL),
        (6, "BENIGN_NOTE", None),
        (7, "ACTION_APPLIED", ActionLabel.FLAG_SENSOR_SUSPECT),
        (7, "OPERATING_MODE_CHANGED", None),
        (7, "CHANNEL_QUALITY_CHANGED", None),
    ]
    assert [decision.immediate_action for decision in trace.targets.decisions] == [
        ActionLabel.VERIFY_REDUNDANT_CHANNEL,
        ActionLabel.FLAG_SENSOR_SUSPECT,
    ]
    assert all(
        EvidenceSlot.RELATED_STATE_STABLE not in decision.evidence_slots
        and set(decision.evidence_slots)
        == {
            EvidenceSlot.CHANNEL_FROZEN,
            EvidenceSlot.CORRELATED_STATE_CHANGE,
            EvidenceSlot.CHANNEL_DISAGREEMENT,
        }
        for decision in trace.targets.decisions
    )
    assert all(
        event.sim_time <= decision.decision_tick
        for decision in trace.targets.decisions
        for evidence_id in decision.evidence_event_ids
        for event in (events_by_id[evidence_id],)
    )
    coordinated = next(
        event
        for event in trace.events
        if event.evidence_slots == (EvidenceSlot.COORDINATED_LOAD_RESPONSE,)
    )
    settlement = next(
        event
        for event in trace.events
        if event.event_type.value == "OPERATING_MODE_CHANGED" and event.sim_time == 7
    )
    assert settlement.related_event_ids == (coordinated.event_id,)


def test_sensor_stuck_load_supports_both_channels_and_load_directions() -> None:
    for seed, channel_id, direction in (
        (20, "aster-electrical-output-a", 1.0),
        (21, "aster-electrical-output-b", -1.0),
    ):
        stuck = generate_trace(build_sensor_stuck_load_scenario(seed=seed, channel_id=channel_id))
        load = generate_trace(build_load_transient_scenario(seed=seed))
        stable = generate_trace(build_stable_scenario(seed=seed))

        assert stuck.latent_states == load.latent_states
        assert (
            stuck.latent_states[-1].values.load_demand - stable.latent_states[-1].values.load_demand
        ) * direction > 0.0
        assert (
            _channel_values(stuck, channel_id)[2:] == (_channel_values(load, channel_id)[1],) * 10
        )


def test_sensor_stuck_load_builder_and_generator_fail_closed() -> None:
    scenario = build_sensor_stuck_load_scenario(seed=4, duration_ticks=8)
    injection = scenario.fault_injections[0]
    malformed = (
        scenario.model_copy(update={"driver": ScenarioDriver.STEADY_OPERATION}),
        scenario.model_copy(update={"plant_variant_id": PlantVariant.ASTER_B}),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"channel_id": "aster-electrical-output-x"}),
                )
            }
        ),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"component_id": "aster-train-cirrus"}),
                )
            }
        ),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"severity": SeverityBand.MEDIUM}),
                )
            }
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"onset_tick": 3}),)}
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"duration_ticks": 2}),)}
        ),
        scenario.model_copy(update={"action_sequence": ()}),
        scenario.model_copy(update={"fault_injections": (injection, injection)}),
        scenario.model_copy(update={"fault_injections": [injection]}),
    )

    for invalid in malformed:
        with pytest.raises(UnsupportedScenarioError):
            generate_trace(invalid)
    with pytest.raises(ValueError, match="between"):
        build_sensor_stuck_load_scenario(seed=4, duration_ticks=7)
    with pytest.raises(ValueError, match="electrical-output"):
        build_sensor_stuck_load_scenario(seed=4, channel_id="aster-primary-flow-a")


def test_sensor_noise_is_prefix_preserving_alternating_and_evidence_gated() -> None:
    scenario = build_sensor_noise_scenario(seed=20, duration_ticks=12)
    trace = generate_trace(scenario)
    stable = generate_trace(build_stable_scenario(seed=20, duration_ticks=12))
    selected_channel = scenario.fault_injections[0].channel_id
    assert selected_channel is not None
    assert scenario.scenario_id == f"aster-a-noise-20-12-2-low-{selected_channel}"

    assert trace.latent_states == stable.latent_states
    for noise_frame, stable_frame in zip(trace.observations, stable.observations, strict=True):
        for noise_channel, stable_channel in zip(
            noise_frame.channels, stable_frame.channels, strict=True
        ):
            if noise_channel.channel_id != selected_channel:
                assert noise_channel == stable_channel
            elif noise_frame.tick <= 2:
                assert noise_channel == stable_channel
    selected_values = _channel_values(trace, selected_channel)
    stable_values = _channel_values(stable, selected_channel)
    deviations = tuple(
        value - baseline
        for value, baseline in zip(selected_values[3:], stable_values[3:], strict=True)
        if value is not None and baseline is not None
    )
    assert all(0.018 - 1e-6 <= abs(value) <= 0.024 + 1e-6 for value in deviations)
    assert all(left * right < 0.0 for left, right in pairwise(deviations))
    assert all(
        abs(deviations[index]) == pytest.approx(abs(deviations[index + 1]))
        for index in range(0, len(deviations) - 1, 2)
    )
    assert [decision.immediate_action for decision in trace.targets.decisions] == [
        ActionLabel.INSUFFICIENT_EVIDENCE,
        ActionLabel.INSUFFICIENT_EVIDENCE,
        ActionLabel.COMPARE_RELATED_TRENDS,
        ActionLabel.FLAG_SENSOR_SUSPECT,
    ]
    assert [decision.diagnosis_status for decision in trace.targets.decisions[:2]] == [
        DiagnosisStatus.UNRESOLVED,
        DiagnosisStatus.UNRESOLVED,
    ]
    assert all(
        decision.fault_labels == (FaultFamily.SENSOR_NOISE,)
        for decision in trace.targets.decisions[2:]
    )
    selected = [
        next(channel for channel in frame.channels if channel.channel_id == selected_channel)
        for frame in trace.observations
    ]
    assert [channel.status for channel in selected[:3]] == [ObservationStatus.NORMAL] * 3
    assert [channel.status for channel in selected[3:5]] == [ObservationStatus.WATCH] * 2
    assert all(channel.status is ObservationStatus.CONFLICTING for channel in selected[5:])
    assert all(channel.quality is ChannelQuality.GOOD for channel in selected[:7])
    assert selected[7].quality is ChannelQuality.SUSPECT
    assert all(channel.quality is not ChannelQuality.NOISY for channel in selected)
    events_by_id = {event.event_id: event for event in trace.events}
    second_observation = next(
        event
        for event in trace.events
        if event.sim_time == 4 and event.event_type.value == "OBSERVATION_CHANGED"
    )
    second_abstention = next(
        event
        for event in trace.events
        if event.sim_time == 5
        and event.event_type.value == "ACTION_APPLIED"
        and event.action_label is ActionLabel.INSUFFICIENT_EVIDENCE
    )
    third_deviation = next(
        event
        for event in trace.events
        if event.sim_time == 5 and event.event_type.value == "OBSERVATION_CHANGED"
    )
    assert second_observation.event_id in trace.targets.decisions[1].evidence_event_ids
    assert second_abstention.related_event_ids == (second_observation.event_id,)
    assert EvidenceSlot.RAPID_INCONSISTENT_READINGS in third_deviation.evidence_slots
    assert all(
        events_by_id[evidence_id].sim_time <= decision.decision_tick
        for decision in trace.targets.decisions
        for evidence_id in decision.evidence_event_ids
    )


def test_sensor_noise_actions_and_fail_closed_constraints() -> None:
    scenario = build_sensor_noise_scenario(seed=21, duration_ticks=8)
    trace = generate_trace(scenario)
    injection = scenario.fault_injections[0]
    assert [
        (event.sim_time, event.action_label) for event in trace.events if event.action_label
    ] == [
        (4, ActionLabel.INSUFFICIENT_EVIDENCE),
        (5, ActionLabel.INSUFFICIENT_EVIDENCE),
        (6, ActionLabel.COMPARE_RELATED_TRENDS),
        (7, ActionLabel.FLAG_SENSOR_SUSPECT),
    ]
    assert [
        (event.sim_time, event.event_type.value) for event in trace.events if event.sim_time >= 3
    ] == [
        (3, "OBSERVATION_CHANGED"),
        (4, "ACTION_APPLIED"),
        (4, "OBSERVATION_CHANGED"),
        (5, "ACTION_APPLIED"),
        (5, "OBSERVATION_CHANGED"),
        (5, "CHANNEL_DISAGREEMENT"),
        (6, "ACTION_APPLIED"),
        (6, "BENIGN_NOTE"),
        (7, "ACTION_APPLIED"),
        (7, "CHANNEL_QUALITY_CHANGED"),
    ]
    invalid = (
        scenario.model_copy(update={"driver": ScenarioDriver.LOAD_TRANSIENT}),
        scenario.model_copy(update={"plant_variant_id": PlantVariant.ASTER_B}),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"channel_id": "aster-primary-flow-a"}),
                )
            }
        ),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"component_id": "aster-train-cirrus"}),
                )
            }
        ),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"severity": SeverityBand.MEDIUM}),
                )
            }
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"onset_tick": 3}),)}
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"duration_ticks": 2}),)}
        ),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"fault_family": FaultFamily.SENSOR_DRIFT}),
                )
            }
        ),
        scenario.model_copy(update={"action_sequence": ()}),
        scenario.model_copy(update={"fault_injections": (injection, injection)}),
        scenario.model_copy(update={"fault_injections": [injection]}),
        scenario.model_copy(update={"scenario_id": "spoofed-scenario"}),
        scenario.model_copy(update={"schema_version": "9.9.9"}),
    )
    for malformed in invalid:
        with pytest.raises(UnsupportedScenarioError):
            generate_trace(malformed)
    for channel_id in ("", True, 7, "aster-primary-thermal-state-x"):
        with pytest.raises(ValueError, match=r".+"):
            build_sensor_noise_scenario(seed=1, channel_id=channel_id)  # type: ignore[arg-type]


def test_model_copy_lookalikes_are_rejected_before_fault_dispatch() -> None:
    noise = build_sensor_noise_scenario(seed=31, duration_ticks=8)
    stuck = build_sensor_stuck_load_scenario(seed=31, duration_ticks=8)

    for scenario in (noise, stuck):
        injection = scenario.fault_injections[0]
        first_action = scenario.action_sequence[0]
        canonical_actions = scenario.action_sequence
        invalid = (
            scenario.model_copy(update={"driver": scenario.driver.value}),
            scenario.model_copy(
                update={"fault_injections": (injection.model_copy(update={"onset_tick": 2.0}),)}
            ),
            scenario.model_copy(
                update={"fault_injections": (injection.model_copy(update={"onset_tick": True}),)}
            ),
            scenario.model_copy(
                update={
                    "action_sequence": (
                        first_action.model_copy(
                            update={"decision_tick": float(first_action.decision_tick)}
                        ),
                        *canonical_actions[1:],
                    )
                }
            ),
            scenario.model_copy(
                update={
                    "action_sequence": (
                        first_action.model_copy(update={"decision_tick": True}),
                        *canonical_actions[1:],
                    )
                }
            ),
            scenario.model_copy(
                update={
                    "action_sequence": (
                        first_action.model_copy(update={"action": first_action.action.value}),
                        *canonical_actions[1:],
                    )
                }
            ),
            scenario.model_copy(update={"action_sequence": list(canonical_actions)}),
            scenario.model_copy(update={"action_sequence": ("malformed-action",)}),
        )
        for malformed in invalid:
            with pytest.raises(UnsupportedScenarioError):
                generate_trace(malformed)

    valid_noise = generate_trace(noise)
    assert all(
        decision.fault_labels == (FaultFamily.SENSOR_NOISE,)
        for decision in valid_noise.targets.decisions[2:]
    )


@pytest.mark.parametrize("channel_id", ["", True, 7, "aster-electrical-output-x"])
def test_sensor_stuck_builder_requires_strict_allowlisted_channel_input(channel_id: object) -> None:
    with pytest.raises(ValueError, match=r".+"):
        build_sensor_stuck_load_scenario(seed=1, channel_id=channel_id)  # type: ignore[arg-type]


def test_sensor_drift_separates_only_observation_layer_and_actions() -> None:
    drift_scenario = build_sensor_drift_scenario(seed=20, onset_tick=3, duration_ticks=12)
    drift = generate_trace(drift_scenario)
    stable = generate_trace(build_stable_scenario(seed=20, duration_ticks=12))
    injection = drift_scenario.fault_injections[0]
    selected_channel = injection.channel_id
    assert selected_channel is not None

    assert drift.latent_states == stable.latent_states
    for drift_frame, stable_frame in zip(drift.observations, stable.observations, strict=True):
        for drift_channel, stable_channel in zip(
            drift_frame.channels, stable_frame.channels, strict=True
        ):
            if drift_channel.channel_id != selected_channel:
                assert drift_channel == stable_channel
            elif drift_frame.tick <= injection.onset_tick:
                assert drift_channel == stable_channel

    selected_values = _channel_values(drift, selected_channel)
    paired_channel = next(
        channel.channel_id
        for channel in ASTER_A_SPEC.channels
        if channel.variable is StateVariable.PRIMARY_FLOW and channel.channel_id != selected_channel
    )
    paired_values = _channel_values(drift, paired_channel)
    separations = tuple(
        abs(selected - paired)
        for selected, paired in zip(selected_values, paired_values, strict=True)
        if selected is not None and paired is not None
    )
    assert separations[injection.onset_tick] == 0.0
    assert all(
        separations[index + 1] + 1e-12 >= separations[index]
        for index in range(injection.onset_tick + 1, len(separations) - 1)
    )
    assert all(
        later - earlier <= ASTER_A_SPEC.max_per_tick_step
        for earlier, later in zip(
            separations[injection.onset_tick + 1 :],
            separations[injection.onset_tick + 2 :],
            strict=False,
        )
    )
    assert [decision.immediate_action for decision in drift.targets.decisions] == [
        ActionLabel.INSUFFICIENT_EVIDENCE,
        ActionLabel.VERIFY_REDUNDANT_CHANNEL,
        ActionLabel.FLAG_SENSOR_SUSPECT,
    ]
    early, mature, flagged = drift.targets.decisions
    assert early.diagnosis_status is DiagnosisStatus.UNRESOLVED
    assert early.fault_labels == ()
    assert early.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE
    assert mature.fault_labels == flagged.fault_labels == (FaultFamily.SENSOR_DRIFT,)
    assert set(mature.evidence_slots) == {
        EvidenceSlot.CHANNEL_DISAGREEMENT,
        EvidenceSlot.RELATED_STATE_STABLE,
    }
    selected_at_flag = next(
        channel
        for channel in drift.observations[flagged.decision_tick].channels
        if channel.channel_id == selected_channel
    )
    selected_after_flag = next(
        channel
        for channel in drift.observations[flagged.decision_tick + 1].channels
        if channel.channel_id == selected_channel
    )
    assert selected_at_flag.quality is ChannelQuality.GOOD
    assert selected_after_flag.quality is ChannelQuality.SUSPECT
    assert selected_at_flag.status is ObservationStatus.CONFLICTING
    selected_statuses = tuple(
        next(channel.status for channel in frame.channels if channel.channel_id == selected_channel)
        for frame in drift.observations
    )
    assert all(
        status is ObservationStatus.NORMAL
        for status in selected_statuses[: injection.onset_tick + 1]
    )
    assert selected_statuses[injection.onset_tick + 1] is ObservationStatus.WATCH
    assert all(
        status is ObservationStatus.CONFLICTING
        for status in selected_statuses[injection.onset_tick + 2 :]
    )


def test_drift_action_events_follow_their_decisions_in_causal_order() -> None:
    scenario = build_sensor_drift_scenario(seed=20, onset_tick=3, duration_ticks=12)
    trace = generate_trace(scenario)
    early, mature, flagged = trace.targets.decisions
    action_events = [event for event in trace.events if event.action_label is not None]

    assert [(event.sim_time, event.action_label) for event in action_events] == [
        (early.decision_tick + 1, ActionLabel.INSUFFICIENT_EVIDENCE),
        (mature.decision_tick + 1, ActionLabel.VERIFY_REDUNDANT_CHANNEL),
        (flagged.decision_tick + 1, ActionLabel.FLAG_SENSOR_SUSPECT),
    ]
    mature_events = [event for event in trace.events if event.sim_time == mature.decision_tick]
    assert mature_events[0].action_label is ActionLabel.INSUFFICIENT_EVIDENCE
    assert mature_events[1].evidence_slots == (EvidenceSlot.CHANNEL_DISAGREEMENT,)
    flag_application = flagged.decision_tick + 1
    assert [
        event.event_type.value for event in trace.events if event.sim_time == flag_application
    ] == [
        "ACTION_APPLIED",
        "CHANNEL_QUALITY_CHANGED",
    ]


def test_diagnosis_references_stable_and_mature_evidence_without_future_events() -> None:
    trace = generate_trace(build_sensor_drift_scenario(seed=20))
    event_by_id = {event.event_id: event for event in trace.events}
    stable = trace.events[0]
    mature = next(
        event
        for event in trace.events
        if event.evidence_slots == (EvidenceSlot.CHANNEL_DISAGREEMENT,)
    )

    assert mature.related_event_ids == (stable.event_id, trace.events[1].event_id)
    for decision in trace.targets.decisions[1:]:
        assert decision.evidence_event_ids == (stable.event_id, mature.event_id)
        observed_slots = {
            slot
            for event_id in decision.evidence_event_ids
            for slot in event_by_id[event_id].evidence_slots
        }
        assert set(decision.evidence_slots).issubset(observed_slots)
        assert all(
            event_by_id[event_id].sim_time <= decision.decision_tick
            for event_id in decision.evidence_event_ids
        )


@pytest.mark.parametrize(
    ("builder", "kwargs"),
    [
        (build_stable_scenario, {"seed": True}),
        (build_stable_scenario, {"seed": 2**32}),
        (build_stable_scenario, {"seed": 1, "duration_ticks": True}),
        (build_stable_scenario, {"seed": 1, "duration_ticks": 7}),
        (build_sensor_drift_scenario, {"seed": 1, "onset_tick": True}),
        (build_sensor_drift_scenario, {"seed": 1, "duration_ticks": 8, "onset_tick": 4}),
        (build_sensor_noise_scenario, {"seed": True}),
        (build_sensor_noise_scenario, {"seed": 1, "duration_ticks": True}),
        (build_sensor_noise_scenario, {"seed": 1, "duration_ticks": 7}),
    ],
)
def test_builders_reject_loose_or_unsupported_inputs(builder: object, kwargs: object) -> None:
    assert callable(builder)
    assert isinstance(kwargs, dict)
    with pytest.raises(ValueError, match=r"must|between|needs"):
        builder(**kwargs)


def test_drift_builder_rejects_nonprimary_channel() -> None:
    with pytest.raises(ValueError, match="primary-flow"):
        build_sensor_drift_scenario(
            seed=1,
            channel_id="aster-secondary-flow-a",
            severity=SeverityBand.LOW,
        )


@pytest.mark.parametrize("channel_id", ["", True, 7, "aster-primary-flow-x"])
def test_drift_builder_requires_strict_allowlisted_channel_input(channel_id: object) -> None:
    with pytest.raises(ValueError, match=r".+"):
        build_sensor_drift_scenario(seed=1, channel_id=channel_id)  # type: ignore[arg-type]


def test_drift_identity_includes_selected_channel_and_severity() -> None:
    first = build_sensor_drift_scenario(
        seed=1, channel_id="aster-primary-flow-a", severity=SeverityBand.LOW
    )
    alternate_channel = build_sensor_drift_scenario(
        seed=1, channel_id="aster-primary-flow-b", severity=SeverityBand.LOW
    )
    alternate_severity = build_sensor_drift_scenario(
        seed=1, channel_id="aster-primary-flow-a", severity=SeverityBand.MEDIUM
    )

    assert first.scenario_id == "aster-a-drift-1-12-3-low-aster-primary-flow-a"
    assert (
        len({first.scenario_id, alternate_channel.scenario_id, alternate_severity.scenario_id}) == 3
    )


def test_direct_noncanonical_scenario_is_rejected() -> None:
    scenario = build_sensor_drift_scenario(seed=4)
    unsupported = scenario.model_copy(update={"action_sequence": ()})
    with pytest.raises(UnsupportedScenarioError, match="noncanonical"):
        generate_trace(unsupported)


def test_direct_wrong_component_channel_mapping_is_rejected() -> None:
    scenario = build_sensor_drift_scenario(seed=4)
    wrong_injection = scenario.fault_injections[0].model_copy(
        update={"component_id": "aster-train-kestrel"}
    )
    unsupported = scenario.model_copy(update={"fault_injections": (wrong_injection,)})
    with pytest.raises(UnsupportedScenarioError, match="mapping"):
        generate_trace(unsupported)


def test_generate_trace_rejects_unsupported_direct_schema_variants() -> None:
    stable = build_stable_scenario(seed=4)
    drift = build_sensor_drift_scenario(seed=4)
    injection = drift.fault_injections[0]
    scenarios = (
        stable.model_copy(update={"plant_variant_id": PlantVariant.ASTER_B}),
        drift.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"fault_family": FaultFamily.SENSOR_NOISE}),
                )
            }
        ),
        drift.model_copy(update={"fault_injections": (injection, injection)}),
        drift.model_copy(
            update={"fault_injections": (injection.model_copy(update={"duration_ticks": 2}),)}
        ),
    )

    for scenario in scenarios:
        with pytest.raises(UnsupportedScenarioError):
            generate_trace(scenario)


def test_generate_trace_rejects_spoofed_scenario_ids_and_schema_versions() -> None:
    scenarios = (
        build_stable_scenario(seed=4),
        build_load_transient_scenario(seed=4),
        build_sensor_drift_scenario(seed=4),
        build_sensor_stuck_load_scenario(seed=4),
    )

    for scenario in scenarios:
        with pytest.raises(UnsupportedScenarioError, match="id"):
            generate_trace(scenario.model_copy(update={"scenario_id": "spoofed-scenario"}))
        with pytest.raises(UnsupportedScenarioError, match="schema"):
            generate_trace(scenario.model_copy(update={"schema_version": "9.9.9"}))


def test_global_random_state_is_unchanged_and_distinct_seeds_vary() -> None:
    original_state = random.getstate()
    try:
        random.seed(817)
        before = random.getstate()
        first = generate_trace(build_stable_scenario(seed=10))
        assert random.getstate() == before
        drift = generate_trace(build_sensor_drift_scenario(seed=10))
        assert random.getstate() == before
        load = generate_trace(build_load_transient_scenario(seed=10))
        assert random.getstate() == before
        stuck = generate_trace(build_sensor_stuck_load_scenario(seed=10))
        assert random.getstate() == before
        noise = generate_trace(build_sensor_noise_scenario(seed=10))
        assert random.getstate() == before
        transfer = generate_trace(build_transfer_efficiency_loss_scenario(seed=10))
        assert random.getstate() == before
        imbalance = generate_trace(build_flow_imbalance_scenario(seed=10))
        assert random.getstate() == before
        second = generate_trace(build_stable_scenario(seed=11))
        stable_b = generate_trace(
            build_stable_scenario(seed=10, plant_variant=PlantVariant.ASTER_B)
        )
        assert random.getstate() == before
        stable_c = generate_trace(
            build_stable_scenario(seed=10, plant_variant=PlantVariant.ASTER_C)
        )
        assert random.getstate() == before

        assert first.latent_states != second.latent_states
        assert first.visible_payload() != second.visible_payload()
        assert stable_b.visible_payload() != stable_c.visible_payload()
        assert drift.latent_states == first.latent_states
        assert load.latent_states != first.latent_states
        assert stuck.latent_states == load.latent_states
        assert noise.latent_states == first.latent_states
        assert transfer.latent_states != first.latent_states
        assert imbalance.latent_states != first.latent_states
    finally:
        random.setstate(original_state)


def test_maximum_duration_and_aster_spec_channel_cardinality() -> None:
    trace = generate_trace(build_stable_scenario(seed=4, duration_ticks=64))
    drift = generate_trace(build_sensor_drift_scenario(seed=4, duration_ticks=64))
    load = generate_trace(build_load_transient_scenario(seed=4, duration_ticks=64))
    noise = generate_trace(build_sensor_noise_scenario(seed=4, duration_ticks=64))
    pump = generate_trace(build_pump_degradation_scenario(seed=4, duration_ticks=64))

    assert len(trace.observations) == 64
    assert len(drift.observations) == 64
    assert len(load.observations) == 64
    assert len(noise.observations) == 64
    assert len(pump.observations) == 64
    assert len(set(ASTER_A_SPEC.primary_train_ids)) == 2
    assert len(set(ASTER_A_SPEC.support_bus_ids)) == 2
    for variable in StateVariable:
        channel_ids = [
            channel.channel_id for channel in ASTER_A_SPEC.channels if channel.variable is variable
        ]
        assert len(channel_ids) == len(set(channel_ids)) == 2


def test_pump_degradation_visible_trends_preserve_seeded_noise_residuals() -> None:
    for seed in (0, 1, 20, 21, 4_294_967_295):
        pump = generate_trace(build_pump_degradation_scenario(seed=seed, duration_ticks=12))
        stable = generate_trace(build_stable_scenario(seed=seed, duration_ticks=12))

        for pump_frame, stable_frame, latent in zip(
            pump.observations, stable.observations, pump.latent_states, strict=True
        ):
            stable_latent = stable.latent_states[latent.tick]
            for pump_channel, stable_channel in zip(
                pump_frame.channels, stable_frame.channels, strict=True
            ):
                assert pump_channel.value is not None
                assert stable_channel.value is not None
                pump_true = getattr(latent.values, pump_channel.variable.value)
                stable_true = getattr(stable_latent.values, stable_channel.variable.value)
                assert pump_channel.value - pump_true == pytest.approx(
                    stable_channel.value - stable_true
                )

        changed_events = [
            event for event in pump.events if event.event_type is EventType.OBSERVATION_CHANGED
        ]
        directions: list[tuple[float, float]] = []
        for event in changed_events:
            if event.value_before is None or event.value_after is None:
                raise AssertionError("pump observation events must carry values")
            directions.append((event.value_before, event.value_after))
        assert directions[0][1] < directions[0][0]
        assert directions[1][1] > directions[1][0]
        assert directions[2][1] < directions[2][0]
        assert directions[3][1] < directions[3][0]


def test_pump_degradation_fixed_chain_and_prefix_identity() -> None:
    short = generate_trace(build_pump_degradation_scenario(seed=20, duration_ticks=9))
    long = generate_trace(build_pump_degradation_scenario(seed=20, duration_ticks=12))

    assert short.latent_states == long.latent_states[:9]
    assert short.observations == long.observations[:9]
    assert short.events == long.events
    assert tuple(
        decision.model_dump(exclude={"scenario_id"}) for decision in short.targets.decisions
    ) == tuple(decision.model_dump(exclude={"scenario_id"}) for decision in long.targets.decisions)
    assert [event.event_type for event in short.events] == [
        EventType.BENIGN_NOTE,
        EventType.COMPONENT_STATE_CHANGED,
        EventType.OPERATING_MODE_CHANGED,
        EventType.OBSERVATION_CHANGED,
        EventType.OBSERVATION_CHANGED,
        EventType.BENIGN_NOTE,
        EventType.ACTION_APPLIED,
        EventType.OBSERVATION_CHANGED,
        EventType.OBSERVATION_CHANGED,
        EventType.ACTION_APPLIED,
        EventType.BENIGN_NOTE,
        EventType.ACTION_APPLIED,
        EventType.TARGET_CHANGED,
        EventType.OPERATING_MODE_CHANGED,
    ]


def test_pump_degradation_latent_causal_order_and_selected_component_only() -> None:
    seed = 20
    pump = generate_trace(build_pump_degradation_scenario(seed=seed, duration_ticks=64))
    stable = generate_trace(build_stable_scenario(seed=seed, duration_ticks=64))
    selected_id = pump.scenario.fault_injections[0].component_id
    assert selected_id is not None

    def first_change(variable: StateVariable) -> int:
        return next(
            state.tick
            for state, baseline in zip(pump.latent_states, stable.latent_states, strict=True)
            if getattr(state.values, variable.value) != getattr(baseline.values, variable.value)
        )

    assert first_change(StateVariable.PRIMARY_FLOW) == 3
    assert first_change(StateVariable.PRIMARY_THERMAL_STATE) == 4
    assert first_change(StateVariable.STEAM_STATE) == 5
    assert first_change(StateVariable.TURBINE_OUTPUT) == 6
    assert first_change(StateVariable.ELECTRICAL_OUTPUT) == 6
    assert first_change(StateVariable.HEAT_SOURCE_LEVEL) == 8
    assert first_change(StateVariable.LOAD_DEMAND) == 8
    assert all(
        getattr(pump.latent_states[-1].values, variable.value)
        < getattr(stable.latent_states[-1].values, variable.value)
        for variable in (
            StateVariable.PRIMARY_FLOW,
            StateVariable.STEAM_STATE,
            StateVariable.TURBINE_OUTPUT,
            StateVariable.ELECTRICAL_OUTPUT,
        )
    )
    assert (
        pump.latent_states[-1].values.primary_thermal_state
        > stable.latent_states[-1].values.primary_thermal_state
    )
    assert all(
        getattr(state.values, variable.value) == getattr(baseline.values, variable.value)
        for state, baseline in zip(pump.latent_states, stable.latent_states, strict=True)
        for variable in (
            StateVariable.PRIMARY_INVENTORY,
            StateVariable.TRANSFER_EFFICIENCY,
            StateVariable.SECONDARY_FLOW,
            StateVariable.SECONDARY_INVENTORY,
            StateVariable.CONDENSER_FUNCTION,
            StateVariable.HEAT_REJECTION,
            StateVariable.SUPPORT_POWER,
        )
    )
    derived_heat = tuple(
        state.values.heat_source_level
        * state.values.primary_flow
        * state.values.transfer_efficiency
        for state in pump.latent_states
    )
    stable_heat = tuple(
        state.values.heat_source_level
        * state.values.primary_flow
        * state.values.transfer_efficiency
        for state in stable.latent_states
    )
    assert derived_heat[:3] == stable_heat[:3]
    assert derived_heat[3] < stable_heat[3]

    previous_health = 1.0
    for state, baseline in zip(pump.latent_states, stable.latent_states, strict=True):
        current = {component.component_id: component for component in state.components}
        stable_components = {component.component_id: component for component in baseline.components}
        selected = current[selected_id]
        assert selected.health <= previous_health
        previous_health = selected.health
        if state.tick < 2:
            assert selected.state is ComponentState.AVAILABLE
            assert selected.health == 1.0
        else:
            assert selected.state is ComponentState.DEGRADED
            assert selected.health < 1.0
        assert selected.pending_maintenance is (state.tick >= 7)
        for component_id, component in current.items():
            if component_id != selected_id:
                assert component == stable_components[component_id]
                assert component.state is ComponentState.AVAILABLE
                assert component.health == 1.0
                assert component.pending_maintenance is False

    selected_tick_two_health = next(
        component.health
        for component in pump.latent_states[2].components
        if component.component_id == selected_id
    )
    assert 0.010 <= 1.0 - selected_tick_two_health <= 0.014
    assert (
        1.0
        - next(
            component.health
            for component in pump.latent_states[-1].components
            if component.component_id == selected_id
        )
        <= 0.24
    )


def test_pump_degradation_statuses_channels_and_causal_actions() -> None:
    trace = generate_trace(build_pump_degradation_scenario(seed=21, duration_ticks=12))
    statuses: dict[StateVariable, Callable[[int], ObservationStatus]] = {
        StateVariable.PRIMARY_FLOW: lambda tick: (
            ObservationStatus.NORMAL
            if tick < 3
            else ObservationStatus.WATCH
            if tick <= 5
            else ObservationStatus.ABNORMAL
        ),
        StateVariable.PRIMARY_THERMAL_STATE: lambda tick: (
            ObservationStatus.NORMAL
            if tick < 4
            else ObservationStatus.WATCH
            if tick <= 5
            else ObservationStatus.ABNORMAL
        ),
        StateVariable.STEAM_STATE: lambda tick: (
            ObservationStatus.NORMAL
            if tick < 5
            else ObservationStatus.WATCH
            if tick == 5
            else ObservationStatus.ABNORMAL
        ),
        StateVariable.TURBINE_OUTPUT: lambda tick: (
            ObservationStatus.NORMAL if tick < 6 else ObservationStatus.ABNORMAL
        ),
        StateVariable.ELECTRICAL_OUTPUT: lambda tick: (
            ObservationStatus.NORMAL if tick < 6 else ObservationStatus.ABNORMAL
        ),
    }
    for frame in trace.observations:
        assert frame.overall_status is (
            ObservationStatus.NORMAL
            if frame.tick < 2
            else ObservationStatus.WATCH
            if frame.tick <= 5
            else ObservationStatus.ABNORMAL
        )
        for variable in StateVariable:
            pair = [channel for channel in frame.channels if channel.variable is variable]
            assert len(pair) == 2
            assert pair[0].value == pair[1].value
            assert pair[0].quality is pair[1].quality is ChannelQuality.GOOD
            status_fn = statuses.get(variable)
            expected = ObservationStatus.NORMAL if status_fn is None else status_fn(frame.tick)
            assert pair[0].status is pair[1].status is expected

    assert [(event.sim_time, event.event_type, event.action_label) for event in trace.events] == [
        (0, EventType.BENIGN_NOTE, None),
        (2, EventType.COMPONENT_STATE_CHANGED, None),
        (2, EventType.OPERATING_MODE_CHANGED, None),
        (3, EventType.OBSERVATION_CHANGED, None),
        (4, EventType.OBSERVATION_CHANGED, None),
        (4, EventType.BENIGN_NOTE, None),
        (5, EventType.ACTION_APPLIED, ActionLabel.INSUFFICIENT_EVIDENCE),
        (5, EventType.OBSERVATION_CHANGED, None),
        (6, EventType.OBSERVATION_CHANGED, None),
        (7, EventType.ACTION_APPLIED, ActionLabel.REQUEST_COMPONENT_INSPECTION),
        (7, EventType.BENIGN_NOTE, None),
        (8, EventType.ACTION_APPLIED, ActionLabel.REDUCE_SIMULATED_LOAD),
        (8, EventType.TARGET_CHANGED, None),
        (8, EventType.OPERATING_MODE_CHANGED, None),
    ]
    events_by_id = {event.event_id: event for event in trace.events}
    assert all(
        events_by_id[related_id].event_index < event.event_index
        for event in trace.events
        for related_id in event.related_event_ids
    )
    assert [
        (decision.decision_tick, decision.immediate_action, decision.diagnosis_status)
        for decision in trace.targets.decisions
    ] == [
        (4, ActionLabel.INSUFFICIENT_EVIDENCE, DiagnosisStatus.UNRESOLVED),
        (6, ActionLabel.REQUEST_COMPONENT_INSPECTION, DiagnosisStatus.DIAGNOSED),
        (7, ActionLabel.REDUCE_SIMULATED_LOAD, DiagnosisStatus.DIAGNOSED),
    ]
    mature_slots = {
        EvidenceSlot.COMPONENT_HEALTH_DECLINING,
        EvidenceSlot.FLOW_DECLINING,
        EvidenceSlot.MULTIPLE_CHANNELS_AGREE,
        EvidenceSlot.CORRELATED_STATE_CHANGE,
        EvidenceSlot.DEPENDENT_TREND_DELAY,
    }
    assert set(trace.targets.decisions[1].evidence_slots) == mature_slots
    assert set(trace.targets.decisions[2].evidence_slots) == mature_slots
    applied_actions = {
        event.action_label: event.sim_time
        for event in trace.events
        if event.event_type is EventType.ACTION_APPLIED
    }
    assert applied_actions[ActionLabel.INSUFFICIENT_EVIDENCE] == 5
    assert applied_actions[ActionLabel.REQUEST_COMPONENT_INSPECTION] == 7
    assert applied_actions[ActionLabel.REDUCE_SIMULATED_LOAD] == 8
    target = next(event for event in trace.events if event.event_type is EventType.TARGET_CHANGED)
    mode = next(
        event
        for event in trace.events
        if event.event_type is EventType.OPERATING_MODE_CHANGED and event.sim_time == 8
    )
    reduce_event = next(
        event for event in trace.events if event.action_label is ActionLabel.REDUCE_SIMULATED_LOAD
    )
    assert target.related_event_ids == (reduce_event.event_id,)
    assert mode.related_event_ids == (target.event_id,)
    assert reduce_event.event_index < target.event_index < mode.event_index
    assert target.value_after is not None
    assert target.value_before is not None
    assert target.value_after < target.value_before

    filtered_prefix = tuple(
        event
        for event in trace.events
        if event.sim_time <= 4 and event.event_type is not EventType.COMPONENT_STATE_CHANGED
    )
    assert filtered_prefix
    assert all(
        EvidenceSlot.COMPONENT_HEALTH_DECLINING not in event.evidence_slots
        for event in filtered_prefix
    )
    assert trace.targets.decisions[0].diagnosis_status is DiagnosisStatus.UNRESOLVED
    assert trace.targets.decisions[0].immediate_action is ActionLabel.INSUFFICIENT_EVIDENCE


def test_pump_degradation_builder_and_direct_variants_fail_closed() -> None:
    scenario = build_pump_degradation_scenario(seed=4, duration_ticks=12)
    injection = scenario.fault_injections[0]
    malformed = (
        scenario.model_copy(update={"driver": ScenarioDriver.LOAD_TRANSIENT}),
        scenario.model_copy(update={"plant_variant_id": PlantVariant.ASTER_B}),
        scenario.model_copy(update={"duration_ticks": 8}),
        scenario.model_copy(update={"scenario_id": "spoofed"}),
        scenario.model_copy(update={"action_sequence": ()}),
        scenario.model_copy(update={"action_sequence": [*scenario.action_sequence]}),
        scenario.model_copy(update={"fault_injections": [injection]}),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"component_id": "cirrus"}),)}
        ),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"channel_id": "aster-primary-flow-a"}),
                )
            }
        ),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"severity": SeverityBand.MEDIUM}),
                )
            }
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"onset_tick": 3}),)}
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"onset_tick": 2.0}),)}
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"onset_tick": True}),)}
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"duration_ticks": 2}),)}
        ),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"fault_family": "PUMP_DEGRADATION"}),
                )
            }
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"severity": "LOW"}),)}
        ),
        scenario.model_copy(
            update={
                "action_sequence": (
                    scenario.action_sequence[0].model_copy(
                        update={"action": "INSUFFICIENT_EVIDENCE"}
                    ),
                    scenario.action_sequence[1],
                    scenario.action_sequence[2],
                )
            }
        ),
        scenario.model_copy(update={"seed": "4"}),
        scenario.model_copy(update={"seed": 4.0}),
        scenario.model_copy(update={"seed": True}),
    )
    for invalid in malformed:
        with pytest.raises(UnsupportedScenarioError):
            generate_trace(invalid)
    invalid_components: tuple[object, ...] = (
        "cirrus",
        "aster-train-cirrus ",
        1,
        1.0,
        True,
        [],
    )
    for invalid_component in invalid_components:
        with pytest.raises(ValueError, match="component_id"):
            build_pump_degradation_scenario(seed=4, component_id=invalid_component)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 9"):
        build_pump_degradation_scenario(seed=4, duration_ticks=8)


def test_pump_degradation_rng_parity_and_step_variation() -> None:
    original_state = random.getstate()
    try:
        random.seed(817)
        before = random.getstate()
        even = generate_trace(build_pump_degradation_scenario(seed=20))
        assert random.getstate() == before
        odd = generate_trace(build_pump_degradation_scenario(seed=21))
        assert random.getstate() == before
    finally:
        random.setstate(original_state)

    assert even.scenario.fault_injections[0].component_id == ASTER_A_SPEC.primary_train_ids[0]
    assert odd.scenario.fault_injections[0].component_id == ASTER_A_SPEC.primary_train_ids[1]
    assert (
        even.targets.decisions[1].fault_labels
        == odd.targets.decisions[1].fault_labels
        == (FaultFamily.PUMP_DEGRADATION,)
    )
    even_health = next(
        component.health
        for component in even.latent_states[2].components
        if component.component_id == even.scenario.fault_injections[0].component_id
    )
    odd_health = next(
        component.health
        for component in odd.latent_states[2].components
        if component.component_id == odd.scenario.fault_injections[0].component_id
    )
    assert even_health != odd_health


def test_pump_trip_context_pair_changes_only_the_context_driven_branch() -> None:
    available_scenario = build_pump_trip_scenario(
        seed=20,
        component_id=ASTER_A_SPEC.primary_train_ids[0],
        standby_state=ComponentState.AVAILABLE,
    )
    unavailable_scenario = build_pump_trip_scenario(
        seed=20,
        component_id=ASTER_A_SPEC.primary_train_ids[0],
        standby_state=ComponentState.UNAVAILABLE,
    )
    available = generate_trace(available_scenario)
    unavailable = generate_trace(unavailable_scenario)
    available_context = available_scenario.standby_context
    unavailable_context = unavailable_scenario.standby_context
    assert available_context is not None
    assert unavailable_context is not None

    dependency_map = dict(ASTER_A_SPEC.primary_train_support_bus_pairs)
    assert dependency_map == dict(
        zip(ASTER_A_SPEC.primary_train_ids, ASTER_A_SPEC.support_bus_ids, strict=True)
    )
    assert available_context.active_train_id == ASTER_A_SPEC.primary_train_ids[0]
    assert available_context.standby_train_id == ASTER_A_SPEC.primary_train_ids[1]
    assert (
        available_context.standby_support_bus_id
        == dependency_map[available_context.standby_train_id]
    )
    assert available_context.support_bus_state is ComponentState.AVAILABLE
    assert available_context.standby_start_delay_ticks == 1
    assert available_scenario.scenario_id.endswith("aster-train-cirrus-available")
    assert unavailable_scenario.scenario_id.endswith("aster-train-cirrus-unavailable")
    assert available_scenario.action_sequence[0].action is (
        ActionLabel.SELECT_SYNTHETIC_STANDBY_TRAIN
    )
    assert [action.action for action in unavailable_scenario.action_sequence] == [
        ActionLabel.REDUCE_SIMULATED_LOAD,
        ActionLabel.ENTER_SIMULATED_STABLE_STATE,
    ]

    assert tuple(state.values for state in available.latent_states[:6]) == tuple(
        state.values for state in unavailable.latent_states[:6]
    )
    assert available.observations[:6] == unavailable.observations[:6]
    assert available.events[1:7] == unavailable.events[1:7]
    assert available.events[0].model_dump(exclude={"evidence_slots"}) == unavailable.events[
        0
    ].model_dump(exclude={"evidence_slots"})
    assert available.events[0].evidence_slots == (
        EvidenceSlot.STABLE_OPERATION,
        EvidenceSlot.STANDBY_AVAILABLE,
    )
    assert unavailable.events[0].evidence_slots == (
        EvidenceSlot.STABLE_OPERATION,
        EvidenceSlot.COMPONENT_UNAVAILABLE,
    )
    assert (
        available.targets.decisions[0].fault_labels
        == unavailable.targets.decisions[0].fault_labels
        == (FaultFamily.PUMP_TRIP,)
    )
    assert available.targets.decisions[0].immediate_action is (
        ActionLabel.SELECT_SYNTHETIC_STANDBY_TRAIN
    )
    assert unavailable.targets.decisions[0].immediate_action is ActionLabel.REDUCE_SIMULATED_LOAD
    assert [
        (decision.decision_tick, decision.immediate_action)
        for decision in unavailable.targets.decisions
    ] == [
        (5, ActionLabel.REDUCE_SIMULATED_LOAD),
        (6, ActionLabel.ENTER_SIMULATED_STABLE_STATE),
    ]
    assert (
        unavailable.targets.decisions[1].evidence_event_ids[:-1]
        == unavailable.targets.decisions[0].evidence_event_ids
    )
    appended_evidence = next(
        event
        for event in unavailable.events
        if event.event_id == unavailable.targets.decisions[1].evidence_event_ids[-1]
    )
    assert appended_evidence.event_type is EventType.TARGET_CHANGED
    assert appended_evidence.variable is StateVariable.LOAD_DEMAND
    assert appended_evidence.sim_time == unavailable.targets.decisions[1].decision_tick == 6
    assert (
        unavailable.targets.decisions[1].evidence_slots
        == unavailable.targets.decisions[0].evidence_slots
    )
    assert all(
        decision.diagnosis_status is DiagnosisStatus.DIAGNOSED
        and decision.fault_labels == (FaultFamily.PUMP_TRIP,)
        and decision.abstention_reason is None
        for decision in unavailable.targets.decisions
    )
    assert all(
        decision.diagnosis_status is DiagnosisStatus.DIAGNOSED
        and decision.abstention_reason is None
        and decision.decision_tick == 5
        for decision in (
            available.targets.decisions[0],
            unavailable.targets.decisions[0],
        )
    )


def test_pump_trip_latent_causality_recovery_and_stabilization_are_bounded() -> None:
    seed = 21
    stable = generate_trace(build_stable_scenario(seed=seed, duration_ticks=64))
    available = generate_trace(
        build_pump_trip_scenario(
            seed=seed,
            duration_ticks=64,
            standby_state=ComponentState.AVAILABLE,
        )
    )
    unavailable = generate_trace(
        build_pump_trip_scenario(
            seed=seed,
            duration_ticks=64,
            standby_state=ComponentState.UNAVAILABLE,
        )
    )
    context = available.scenario.standby_context
    unavailable_context = unavailable.scenario.standby_context
    assert context is not None
    assert unavailable_context is not None

    def component_state(trace: SimulationTrace, tick: int, component_id: str) -> ComponentState:
        return next(
            component.state
            for component in trace.latent_states[tick].components
            if component.component_id == component_id
        )

    for trace in (available, unavailable):
        active = trace.scenario.standby_context
        assert active is not None
        assert all(
            next(
                component.health
                for component in latent.components
                if component.component_id == active.active_train_id
            )
            == 1.0
            for latent in trace.latent_states
        )
        assert component_state(trace, 1, active.active_train_id) is ComponentState.AVAILABLE
        assert component_state(trace, 2, active.active_train_id) is ComponentState.UNAVAILABLE
        assert all(
            component_state(trace, tick, bus_id) is ComponentState.AVAILABLE
            for tick in range(trace.scenario.duration_ticks)
            for bus_id in ASTER_A_SPEC.support_bus_ids
        )

    assert component_state(available, 5, context.standby_train_id) is ComponentState.AVAILABLE
    assert component_state(available, 6, context.standby_train_id) is ComponentState.STARTING
    assert component_state(available, 7, context.standby_train_id) is ComponentState.RECOVERING
    assert all(
        component_state(unavailable, tick, unavailable_context.standby_train_id)
        is ComponentState.UNAVAILABLE
        for tick in range(unavailable.scenario.duration_ticks)
    )

    def first_change(trace: SimulationTrace, variable: StateVariable) -> int:
        return next(
            state.tick
            for state, baseline in zip(trace.latent_states, stable.latent_states, strict=True)
            if getattr(state.values, variable.value) != getattr(baseline.values, variable.value)
        )

    assert first_change(available, StateVariable.PRIMARY_FLOW) == 3
    assert first_change(available, StateVariable.PRIMARY_THERMAL_STATE) == 4
    assert first_change(available, StateVariable.STEAM_STATE) == 5
    assert first_change(available, StateVariable.TURBINE_OUTPUT) == 5
    assert first_change(available, StateVariable.ELECTRICAL_OUTPUT) == 5
    assert (
        available.latent_states[3].values.primary_flow
        == available.latent_states[6].values.primary_flow
    )
    assert (
        available.latent_states[7].values.primary_flow
        > available.latent_states[6].values.primary_flow
    )
    assert (
        available.latent_states[-1].values.primary_flow
        < stable.latent_states[-1].values.primary_flow
    )
    assert all(
        state.values.primary_flow == unavailable.latent_states[3].values.primary_flow
        for state in unavailable.latent_states[3:]
    )
    assert (
        unavailable.latent_states[5].values.load_demand
        == stable.latent_states[5].values.load_demand
    )
    assert (
        unavailable.latent_states[6].values.load_demand < stable.latent_states[6].values.load_demand
    )
    assert (
        unavailable.latent_states[6].values.heat_source_level
        < stable.latent_states[6].values.heat_source_level
    )
    assert all(
        state.values.transfer_efficiency == baseline.values.transfer_efficiency
        for trace in (available, unavailable)
        for state, baseline in zip(trace.latent_states, stable.latent_states, strict=True)
    )
    assert (
        available.latent_states[3].values.heat_source_level
        * available.latent_states[3].values.primary_flow
        * available.latent_states[3].values.transfer_efficiency
        < stable.latent_states[3].values.heat_source_level
        * stable.latent_states[3].values.primary_flow
        * stable.latent_states[3].values.transfer_efficiency
    )

    for trace in (available, unavailable):
        for before, after in pairwise(trace.latent_states):
            for variable in StateVariable:
                step = abs(
                    getattr(after.values, variable.value) - getattr(before.values, variable.value)
                )
                if before.tick == 2 and variable is StateVariable.PRIMARY_FLOW:
                    assert step > ASTER_A_SPEC.max_per_tick_step
                else:
                    assert step <= ASTER_A_SPEC.max_per_tick_step
    assert [state.operating_mode for state in available.latent_states[:8]] == [
        OperatingMode.STABLE,
        OperatingMode.STABLE,
        OperatingMode.DISTURBED,
        OperatingMode.DISTURBED,
        OperatingMode.DISTURBED,
        OperatingMode.DISTURBED,
        OperatingMode.DISTURBED,
        OperatingMode.RECOVERY,
    ]
    assert unavailable.latent_states[7].operating_mode is OperatingMode.STABILIZED


def test_pump_trip_events_statuses_and_noise_residuals_are_explicit() -> None:
    available = generate_trace(
        build_pump_trip_scenario(seed=20, standby_state=ComponentState.AVAILABLE)
    )
    unavailable = generate_trace(
        build_pump_trip_scenario(seed=20, standby_state=ComponentState.UNAVAILABLE)
    )
    stable = generate_trace(build_stable_scenario(seed=20))

    assert [
        (event.sim_time, event.event_type, event.action_label) for event in available.events
    ] == [
        (0, EventType.BENIGN_NOTE, None),
        (2, EventType.COMPONENT_STATE_CHANGED, None),
        (2, EventType.OPERATING_MODE_CHANGED, None),
        (3, EventType.OBSERVATION_CHANGED, None),
        (4, EventType.OBSERVATION_CHANGED, None),
        (5, EventType.OBSERVATION_CHANGED, None),
        (5, EventType.OBSERVATION_CHANGED, None),
        (6, EventType.ACTION_APPLIED, ActionLabel.SELECT_SYNTHETIC_STANDBY_TRAIN),
        (6, EventType.COMPONENT_STATE_CHANGED, None),
        (7, EventType.COMPONENT_STATE_CHANGED, None),
        (7, EventType.OBSERVATION_CHANGED, None),
        (7, EventType.OPERATING_MODE_CHANGED, None),
    ]
    assert [
        (event.sim_time, event.event_type, event.action_label) for event in unavailable.events
    ] == [
        (0, EventType.BENIGN_NOTE, None),
        (2, EventType.COMPONENT_STATE_CHANGED, None),
        (2, EventType.OPERATING_MODE_CHANGED, None),
        (3, EventType.OBSERVATION_CHANGED, None),
        (4, EventType.OBSERVATION_CHANGED, None),
        (5, EventType.OBSERVATION_CHANGED, None),
        (5, EventType.OBSERVATION_CHANGED, None),
        (6, EventType.ACTION_APPLIED, ActionLabel.REDUCE_SIMULATED_LOAD),
        (6, EventType.TARGET_CHANGED, None),
        (7, EventType.ACTION_APPLIED, ActionLabel.ENTER_SIMULATED_STABLE_STATE),
        (7, EventType.OPERATING_MODE_CHANGED, None),
    ]
    for trace in (available, unavailable):
        events_by_id = {event.event_id: event for event in trace.events}
        assert [event.event_index for event in trace.events] == list(range(len(trace.events)))
        assert [event.sim_time for event in trace.events] == sorted(
            event.sim_time for event in trace.events
        )
        assert all(
            events_by_id[related_id].event_index < event.event_index
            for event in trace.events
            for related_id in event.related_event_ids
        )
        decision = trace.targets.decisions[0]
        assert all(
            events_by_id[event_id].sim_time <= decision.decision_tick
            for event_id in decision.evidence_event_ids
        )
        assert {
            EvidenceSlot.COMPONENT_UNAVAILABLE,
            EvidenceSlot.FLOW_DECLINING,
            EvidenceSlot.MULTIPLE_CHANNELS_AGREE,
            EvidenceSlot.CORRELATED_STATE_CHANGE,
            EvidenceSlot.DEPENDENT_TREND_DELAY,
        }.issubset(decision.evidence_slots)
        assert scan_prohibited_content(trace) == ()

        for frame, stable_frame, latent, stable_latent in zip(
            trace.observations,
            stable.observations,
            trace.latent_states,
            stable.latent_states,
            strict=True,
        ):
            assert frame.overall_status is (
                ObservationStatus.NORMAL if frame.tick < 2 else ObservationStatus.ABNORMAL
            )
            for channel, stable_channel in zip(frame.channels, stable_frame.channels, strict=True):
                assert channel.value is not None
                assert stable_channel.value is not None
                assert channel.quality is ChannelQuality.GOOD
                true_value = getattr(latent.values, channel.variable.value)
                stable_true = getattr(stable_latent.values, stable_channel.variable.value)
                assert channel.value - true_value == pytest.approx(
                    stable_channel.value - stable_true
                )
            for variable in StateVariable:
                pair = [channel for channel in frame.channels if channel.variable is variable]
                assert pair[0].value == pair[1].value
                assert pair[0].status is pair[1].status

    available_flow_status = [
        next(
            channel.status
            for channel in frame.channels
            if channel.variable is StateVariable.PRIMARY_FLOW
        )
        for frame in available.observations
    ]
    unavailable_flow_status = [
        next(
            channel.status
            for channel in frame.channels
            if channel.variable is StateVariable.PRIMARY_FLOW
        )
        for frame in unavailable.observations
    ]
    assert available_flow_status[:7] == [
        ObservationStatus.NORMAL,
        ObservationStatus.NORMAL,
        ObservationStatus.NORMAL,
        ObservationStatus.ABNORMAL,
        ObservationStatus.ABNORMAL,
        ObservationStatus.ABNORMAL,
        ObservationStatus.ABNORMAL,
    ]
    assert all(status is ObservationStatus.WATCH for status in available_flow_status[7:])
    assert all(status is ObservationStatus.ABNORMAL for status in unavailable_flow_status[3:])
    assert (
        next(
            channel.status
            for channel in available.observations[4].channels
            if channel.variable is StateVariable.PRIMARY_THERMAL_STATE
        )
        is ObservationStatus.WATCH
    )
    assert (
        next(
            channel.status
            for channel in available.observations[5].channels
            if channel.variable is StateVariable.STEAM_STATE
        )
        is ObservationStatus.WATCH
    )
    thermal_statuses = [
        next(
            channel.status
            for channel in frame.channels
            if channel.variable is StateVariable.PRIMARY_THERMAL_STATE
        )
        for frame in available.observations
    ]
    steam_statuses = [
        next(
            channel.status
            for channel in frame.channels
            if channel.variable is StateVariable.STEAM_STATE
        )
        for frame in available.observations
    ]
    assert thermal_statuses[:5] == [
        ObservationStatus.NORMAL,
        ObservationStatus.NORMAL,
        ObservationStatus.NORMAL,
        ObservationStatus.NORMAL,
        ObservationStatus.WATCH,
    ]
    assert all(status is ObservationStatus.ABNORMAL for status in thermal_statuses[5:])
    assert steam_statuses[:6] == [
        ObservationStatus.NORMAL,
        ObservationStatus.NORMAL,
        ObservationStatus.NORMAL,
        ObservationStatus.NORMAL,
        ObservationStatus.NORMAL,
        ObservationStatus.WATCH,
    ]
    assert all(status is ObservationStatus.ABNORMAL for status in steam_statuses[6:])
    for unrelated in (
        StateVariable.HEAT_SOURCE_LEVEL,
        StateVariable.PRIMARY_INVENTORY,
        StateVariable.TRANSFER_EFFICIENCY,
        StateVariable.SECONDARY_FLOW,
        StateVariable.SECONDARY_INVENTORY,
        StateVariable.CONDENSER_FUNCTION,
        StateVariable.HEAT_REJECTION,
        StateVariable.LOAD_DEMAND,
        StateVariable.SUPPORT_POWER,
    ):
        assert all(
            channel.status is ObservationStatus.NORMAL
            for frame in available.observations
            for channel in frame.channels
            if channel.variable is unrelated
        )


def test_pump_trip_replay_prefix_aliases_rng_and_max_duration() -> None:
    original_state = random.getstate()
    try:
        random.seed(7331)
        before = random.getstate()
        traces: list[SimulationTrace] = []
        for component_id in ASTER_A_SPEC.primary_train_ids:
            for standby_state in (ComponentState.AVAILABLE, ComponentState.UNAVAILABLE):
                short = generate_trace(
                    build_pump_trip_scenario(
                        seed=20,
                        duration_ticks=8,
                        component_id=component_id,
                        standby_state=standby_state,
                    )
                )
                long = generate_trace(
                    build_pump_trip_scenario(
                        seed=20,
                        duration_ticks=12,
                        component_id=component_id,
                        standby_state=standby_state,
                    )
                )
                assert short == generate_trace(short.scenario)
                context = short.scenario.standby_context
                assert context is not None
                assert context.active_train_id == component_id
                assert context.standby_train_id == next(
                    train_id
                    for train_id in ASTER_A_SPEC.primary_train_ids
                    if train_id != component_id
                )
                assert (
                    context.standby_support_bus_id
                    == dict(ASTER_A_SPEC.primary_train_support_bus_pairs)[context.standby_train_id]
                )
                assert short.latent_states == long.latent_states[:8]
                assert short.observations == long.observations[:8]
                assert short.events == long.events
                assert tuple(
                    decision.model_dump(exclude={"scenario_id"})
                    for decision in short.targets.decisions
                ) == tuple(
                    decision.model_dump(exclude={"scenario_id"})
                    for decision in long.targets.decisions
                )
                traces.append(short)
                assert random.getstate() == before
        maximum = generate_trace(build_pump_trip_scenario(seed=20, duration_ticks=64))
        assert len(maximum.observations) == 64
        assert random.getstate() == before
    finally:
        random.setstate(original_state)

    assert {trace.scenario.fault_injections[0].component_id for trace in traces} == set(
        ASTER_A_SPEC.primary_train_ids
    )
    assert (
        build_pump_trip_scenario(seed=20).fault_injections[0].component_id
        == (ASTER_A_SPEC.primary_train_ids[0])
    )
    assert (
        build_pump_trip_scenario(seed=21).fault_injections[0].component_id
        == (ASTER_A_SPEC.primary_train_ids[1])
    )


def test_valve_lag_and_stuck_are_matched_temporal_counterfactuals() -> None:
    seed = 20
    lag = generate_trace(build_valve_lag_scenario(seed=seed, duration_ticks=12))
    stuck = generate_trace(build_valve_stuck_scenario(seed=seed, duration_ticks=12))
    stable = generate_trace(build_stable_scenario(seed=seed, duration_ticks=12))
    valve_id = ASTER_A_SPEC.primary_flow_valve_ids[0]

    assert lag.scenario.fault_injections[0].duration_ticks == 4
    assert stuck.scenario.fault_injections[0].duration_ticks is None
    assert lag.latent_states[:6] == stuck.latent_states[:6]
    assert lag.observations[:6] == stuck.observations[:6]
    assert tuple(event for event in lag.events if event.sim_time <= 5) == tuple(
        event for event in stuck.events if event.sim_time <= 5
    )
    assert lag.latent_states[6].values.primary_flow != stable.latent_states[6].values.primary_flow
    assert stuck.latent_states[6].values.primary_flow == stable.latent_states[6].values.primary_flow

    for trace in (lag, stuck):
        for state in trace.latent_states:
            valve = next(
                component for component in state.components if component.component_id == valve_id
            )
            assert (valve.commanded_position is None) is (valve.actual_position is None)
            assert valve.commanded_position is not None
            assert valve.actual_position is not None
            assert all(
                component.commanded_position is None and component.actual_position is None
                for component in state.components
                if component.component_id != valve_id
            )

    assert [decision.diagnosis_status for decision in lag.targets.decisions] == [
        DiagnosisStatus.UNRESOLVED,
        DiagnosisStatus.DIAGNOSED,
    ]
    assert all(
        decision.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE
        and decision.immediate_action is ActionLabel.INSUFFICIENT_EVIDENCE
        for decision in (lag.targets.decisions[0], stuck.targets.decisions[0])
    )
    assert lag.targets.decisions[-1].fault_labels == (FaultFamily.VALVE_LAG,)
    assert lag.targets.decisions[-1].immediate_action is ActionLabel.CONTINUE_MONITORING
    assert stuck.targets.decisions[-1].fault_labels == (FaultFamily.VALVE_STUCK,)
    assert stuck.targets.decisions[-1].immediate_action is ActionLabel.REQUEST_COMPONENT_INSPECTION
    assert any(event.event_type is EventType.COMMAND_POSITION_ALIGNED for event in lag.events)
    assert not any(event.event_type is EventType.COMMAND_POSITION_ALIGNED for event in stuck.events)
    assert any(EvidenceSlot.MISMATCH_PERSISTED in event.evidence_slots for event in stuck.events)
    assert all(
        event.sim_time == 7
        for trace in (lag, stuck)
        for event in trace.events
        if event.event_type is EventType.ACTION_APPLIED
        and event.action_label is not ActionLabel.INSUFFICIENT_EVIDENCE
    )
    assert not any(
        component.pending_maintenance
        for state in lag.latent_states
        for component in state.components
    )
    assert not any(
        component.pending_maintenance
        for state in stuck.latent_states[:7]
        for component in state.components
    )
    assert all(
        next(
            component for component in state.components if component.component_id == valve_id
        ).pending_maintenance
        for state in stuck.latent_states[7:]
    )
    assert "VALVE_LAG" not in str(lag.visible_payload())
    assert "VALVE_STUCK" not in str(stuck.visible_payload())


def test_valve_builder_and_model_copy_shapes_fail_closed() -> None:
    scenario = build_valve_lag_scenario(seed=4)
    injection = scenario.fault_injections[0]
    malformed = (
        scenario.model_copy(update={"scenario_id": "spoofed"}),
        scenario.model_copy(update={"duration_ticks": 7}),
        scenario.model_copy(update={"fault_injections": [injection]}),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"duration_ticks": 3}),)}
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"component_id": "lark"}),)}
        ),
        scenario.model_copy(
            update={
                "fault_injections": (injection.model_copy(update={"fault_family": "VALVE_LAG"}),)
            }
        ),
    )
    for candidate in malformed:
        with pytest.raises(UnsupportedScenarioError):
            generate_trace(candidate)

    with pytest.raises(ValueError, match="primary-flow valve"):
        build_valve_stuck_scenario(seed=4, component_id="lark")
    with pytest.raises(ValueError, match="duration_ticks"):
        build_valve_lag_scenario(seed=4, duration_ticks=7)


def test_valve_lag_declared_band_changes_only_its_resolution_schedule() -> None:
    lag_three = generate_trace(build_valve_lag_scenario(seed=20, lag_ticks=3))
    lag_four = generate_trace(build_valve_lag_scenario(seed=20, lag_ticks=4))
    stuck = generate_trace(build_valve_stuck_scenario(seed=20))

    assert lag_three.scenario.fault_injections[0].duration_ticks == 3
    assert lag_four.scenario.fault_injections[0].duration_ticks == 4
    assert lag_three.scenario.scenario_id.endswith("aster-valve-lark-lag-3")
    assert lag_four.scenario.scenario_id.endswith("aster-valve-lark-lag-4")
    assert [decision.decision_tick for decision in lag_three.targets.decisions] == [3, 5]
    assert [decision.decision_tick for decision in lag_four.targets.decisions] == [3, 6]
    assert (
        next(
            event.sim_time
            for event in lag_three.events
            if event.event_type is EventType.COMMAND_POSITION_ALIGNED
        )
        == 5
    )
    assert (
        next(
            event.sim_time
            for event in lag_four.events
            if event.event_type is EventType.COMMAND_POSITION_ALIGNED
        )
        == 6
    )
    assert all(
        event.sim_time == 6
        for event in lag_three.events
        if event.event_type is EventType.ACTION_APPLIED
        and event.action_label is ActionLabel.CONTINUE_MONITORING
    )
    assert all(
        event.sim_time == 7
        for event in lag_four.events
        if event.event_type is EventType.ACTION_APPLIED
        and event.action_label is ActionLabel.CONTINUE_MONITORING
    )
    assert stuck.targets.decisions[-1].decision_tick == 6

    for invalid_lag_ticks in (True, 3.0, 2, 5):
        with pytest.raises(ValueError, match="lag_ticks"):
            build_valve_lag_scenario(seed=4, lag_ticks=invalid_lag_ticks)  # type: ignore[arg-type]


def test_pump_trip_builder_and_model_copy_shapes_fail_closed() -> None:
    scenario = build_pump_trip_scenario(seed=4, standby_state=ComponentState.AVAILABLE)
    injection = scenario.fault_injections[0]
    context = scenario.standby_context
    assert context is not None
    malformed = (
        scenario.model_copy(update={"driver": ScenarioDriver.LOAD_TRANSIENT}),
        scenario.model_copy(update={"plant_variant_id": PlantVariant.ASTER_B}),
        scenario.model_copy(update={"scenario_id": "spoofed"}),
        scenario.model_copy(update={"duration_ticks": 7}),
        scenario.model_copy(update={"duration_ticks": 8.0}),
        scenario.model_copy(update={"duration_ticks": True}),
        scenario.model_copy(update={"seed": "4"}),
        scenario.model_copy(update={"seed": 4.0}),
        scenario.model_copy(update={"seed": True}),
        scenario.model_copy(update={"fault_injections": [injection]}),
        scenario.model_copy(update={"fault_injections": (injection, injection)}),
        scenario.model_copy(update={"action_sequence": [*scenario.action_sequence]}),
        scenario.model_copy(update={"action_sequence": ()}),
        scenario.model_copy(update={"standby_context": None}),
        scenario.model_copy(update={"standby_context": context.model_dump()}),
        scenario.model_copy(update={"standby_context": [context]}),
        scenario.model_copy(
            update={"standby_context": context.model_copy(update={"context_id": "spoofed"})}
        ),
        scenario.model_copy(
            update={
                "standby_context": context.model_copy(
                    update={"active_train_id": context.standby_train_id}
                )
            }
        ),
        scenario.model_copy(
            update={
                "standby_context": context.model_copy(
                    update={"standby_train_id": context.active_train_id}
                )
            }
        ),
        scenario.model_copy(
            update={"standby_context": context.model_copy(update={"standby_state": "AVAILABLE"})}
        ),
        scenario.model_copy(
            update={
                "standby_context": context.model_copy(
                    update={"standby_state": ComponentState.DEGRADED}
                )
            }
        ),
        scenario.model_copy(
            update={
                "standby_context": context.model_copy(
                    update={"standby_support_bus_id": ASTER_A_SPEC.support_bus_ids[0]}
                )
            }
        ),
        scenario.model_copy(
            update={
                "standby_context": context.model_copy(
                    update={"support_bus_state": ComponentState.UNAVAILABLE}
                )
            }
        ),
        scenario.model_copy(
            update={
                "standby_context": context.model_copy(update={"support_bus_state": "AVAILABLE"})
            }
        ),
        scenario.model_copy(
            update={"standby_context": context.model_copy(update={"standby_start_delay_ticks": 2})}
        ),
        scenario.model_copy(
            update={
                "standby_context": context.model_copy(update={"standby_start_delay_ticks": 1.0})
            }
        ),
        scenario.model_copy(
            update={
                "standby_context": context.model_copy(update={"standby_start_delay_ticks": True})
            }
        ),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"component_id": "aster-train-unknown"}),
                )
            }
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"channel_id": "x"}),)}
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"onset_tick": 3}),)}
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"onset_tick": 2.0}),)}
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"onset_tick": True}),)}
        ),
        scenario.model_copy(
            update={
                "fault_injections": (
                    injection.model_copy(update={"severity": SeverityBand.MEDIUM}),
                )
            }
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"severity": "LOW"}),)}
        ),
        scenario.model_copy(
            update={"fault_injections": (injection.model_copy(update={"duration_ticks": 1}),)}
        ),
        scenario.model_copy(
            update={
                "fault_injections": (injection.model_copy(update={"fault_family": "PUMP_TRIP"}),)
            }
        ),
        scenario.model_copy(
            update={
                "action_sequence": (
                    scenario.action_sequence[0].model_copy(
                        update={"action": "SELECT_SYNTHETIC_STANDBY_TRAIN"}
                    ),
                )
            }
        ),
    )
    for invalid in malformed:
        with pytest.raises(UnsupportedScenarioError):
            generate_trace(invalid)

    stable_with_context = build_stable_scenario(seed=4).model_copy(
        update={"standby_context": context}
    )
    with pytest.raises(UnsupportedScenarioError, match="only supported for pump trip"):
        generate_trace(stable_with_context)

    invalid_components: tuple[object, ...] = ("cirrus", 1, 1.0, True, [])
    for invalid_component in invalid_components:
        with pytest.raises(ValueError, match="component_id"):
            build_pump_trip_scenario(seed=4, component_id=invalid_component)  # type: ignore[arg-type]
    invalid_states: tuple[object, ...] = (
        "AVAILABLE",
        ComponentState.DEGRADED,
        ComponentState.STARTING,
        1,
        True,
    )
    for invalid_state in invalid_states:
        with pytest.raises(ValueError, match="standby_state"):
            build_pump_trip_scenario(seed=4, standby_state=invalid_state)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="between 8 and 64"):
        build_pump_trip_scenario(seed=4, duration_ticks=7)


@pytest.mark.parametrize(
    ("plant_variant", "spec"),
    [
        (PlantVariant.ASTER_B, ASTER_B_SPEC),
        (PlantVariant.ASTER_C, ASTER_C_SPEC),
    ],
)
def test_stable_variant_cards_are_replayable_bounded_and_visible(
    plant_variant: PlantVariant, spec: AsterVariantSpec
) -> None:
    """B/C stable traces use only their fixed reviewed component/channel cards."""

    assert plant_variant in VARIANT_REGISTRY
    scenario = build_stable_scenario(seed=23, duration_ticks=12, plant_variant=plant_variant)
    trace = generate_trace(scenario)
    assert trace == generate_trace(scenario)
    assert scenario.scenario_id == f"{plant_variant.value.lower()}-stable-23-12"
    assert {component.component_id for component in trace.latent_states[0].components} == {
        component.component_id for component in spec.components
    }
    expected_channels = tuple(channel.channel_id for channel in spec.channels)
    assert {channel.channel_id for channel in trace.observations[0].channels} == set(
        expected_channels
    )
    assert len(expected_channels) == len(set(expected_channels)) == 2 * len(StateVariable)
    for frame, latent in zip(trace.observations, trace.latent_states, strict=True):
        for channel in frame.channels:
            assert channel.value is not None
            assert abs(channel.value - getattr(latent.values, channel.variable.value)) <= (
                spec.baseline_noise_bound + 0.000001
            )
    payload = trace.visible_payload()
    assert payload["plant_variant_id"] == plant_variant.value
    assert payload["dependency_map_context"] is None
    assert scan_prohibited_content(trace) == ()


def test_variant_selection_and_dependency_context_fail_closed_before_g12() -> None:
    stable_b = build_stable_scenario(seed=23, plant_variant=PlantVariant.ASTER_B)
    canonical_context = dependency_map_context_for(ASTER_B_SPEC)
    assert canonical_context.plant_variant_id is PlantVariant.ASTER_B
    assert tuple(
        (link.support_bus_id, link.dependent_component_id) for link in canonical_context.links
    ) == tuple(
        sorted(
            (link.support_bus_id, link.dependent_component_id) for link in canonical_context.links
        )
    )
    with pytest.raises(UnsupportedScenarioError, match="dependency map context"):
        generate_trace(stable_b.model_copy(update={"dependency_map_context": canonical_context}))
    with pytest.raises(UnsupportedScenarioError, match="dependency map context"):
        generate_trace(
            stable_b.model_copy(
                update={
                    "dependency_map_context": canonical_context.model_dump(mode="json"),
                }
            )
        )
    with pytest.raises(UnsupportedScenarioError, match="canonical enum"):
        generate_trace(stable_b.model_copy(update={"plant_variant_id": "ASTER-B"}))
    with pytest.raises(UnsupportedScenarioError, match="id"):
        generate_trace(stable_b.model_copy(update={"plant_variant_id": PlantVariant.ASTER_A}))
    with pytest.raises(ValueError, match="plant_variant"):
        build_stable_scenario(seed=23, plant_variant="ASTER-B")  # type: ignore[arg-type]


_LEGACY_ASTER_A_COMPONENT_IDS = frozenset(
    {
        "aster-train-cirrus",
        "aster-train-kestrel",
        "aster-valve-lark",
        "aster-bus-rill",
        "aster-bus-quill",
        "aster-instrument-vireo",
    }
)
_ADDED_ASTER_A_COMPONENT_IDS = frozenset(
    {
        "aster-domain-orchid",
        "aster-transfer-wren",
        "aster-feed-brindle",
    }
)
_LEGACY_ASTER_A_PROJECTION_HASHES = {
    "stable": "bb56540974a6c262335c819d245b567b6a5f1172dba19d3936adf30d46efc2be",
    "load": "530cf8bea0c833069b54aec320347981553b269fa8a40c8672cfb709937672db",
    "drift": "e963c7ce13a2697e165cbfc1f698642ce18e6c8a9ea93231685dcf9dff70cb71",
    "stuck_load": "514628a6bebf0bf5ebf3bc22df7b82d709fffa285b62e8e9bb3abc053230b969",
    "noise": "0753c772b1153be893bd000ca4267da7dfc84a78ea1e200345815450556bfe0b",
    "pump_deg": "37194ad9a5b08345cf5bb66ecd5ad22191c0c1e658cdffb489658369a20c8f9d",
    "trip_avail": "7d4ecbed6580866198d5ead29550057d314d8cb3603e36019c8609d07569cf11",
    "trip_unavail": "c5ab0b6b76dd3efb3f7c1b01716a31f55c961b2e70d1faf5cea7ffd09c2d7ec3",
    "valve_lag3": "31b0459d5438d4befe5556df1342bf03c4fdab89b4a6555e1377635195abd3a2",
    "valve_lag4": "3d11a7d2f003625d6a2e31da367dd17e7c0211be6fa65b8d495f264cf88329c3",
    "valve_stuck": "02f5990f06dc158d3baa1bcd3c5652d080ee7b6aa40c4415e3074a7bd4a33a27",
}


def _legacy_aster_a_projection(trace: SimulationTrace) -> dict[str, object]:
    """Normalize a trace exactly as before the registry's three additive roles."""

    latent_states = []
    for state in trace.latent_states:
        projected = state.model_dump(mode="json")
        projected["components"] = [
            component
            for component in projected["components"]
            if component["component_id"] in _LEGACY_ASTER_A_COMPONENT_IDS
        ]
        latent_states.append(projected)
    return {
        "scenario": trace.scenario.model_dump(mode="json", exclude={"dependency_map_context"}),
        "latent_states": latent_states,
        "observations": [frame.model_dump(mode="json") for frame in trace.observations],
        "events": [event.model_dump(mode="json") for event in trace.events],
        "targets": trace.targets.model_dump(mode="json"),
    }


def test_variant_registry_adds_only_audited_aster_a_component_records() -> None:
    """A registry refactor may add roles, but cannot perturb historic A behavior."""

    scenarios = {
        "stable": build_stable_scenario(seed=23),
        "load": build_load_transient_scenario(seed=23),
        "drift": build_sensor_drift_scenario(seed=23),
        "stuck_load": build_sensor_stuck_load_scenario(seed=23),
        "noise": build_sensor_noise_scenario(seed=23),
        "pump_deg": build_pump_degradation_scenario(seed=23),
        "trip_avail": build_pump_trip_scenario(seed=23, standby_state=ComponentState.AVAILABLE),
        "trip_unavail": build_pump_trip_scenario(seed=23, standby_state=ComponentState.UNAVAILABLE),
        "valve_lag3": build_valve_lag_scenario(seed=23, lag_ticks=3),
        "valve_lag4": build_valve_lag_scenario(seed=23, lag_ticks=4),
        "valve_stuck": build_valve_stuck_scenario(seed=23),
    }
    for name, scenario in scenarios.items():
        trace = generate_trace(scenario)
        assert (
            canonical_sha256(_legacy_aster_a_projection(trace))
            == (_LEGACY_ASTER_A_PROJECTION_HASHES[name])
        )
        for state in trace.latent_states:
            added = {
                component.component_id: component
                for component in state.components
                if component.component_id in _ADDED_ASTER_A_COMPONENT_IDS
            }
            assert set(added) == _ADDED_ASTER_A_COMPONENT_IDS
            assert all(
                component.state is ComponentState.AVAILABLE
                and component.health == 1.0
                and component.commanded_position is None
                and component.actual_position is None
                and component.pending_maintenance is False
                for component in added.values()
            )
