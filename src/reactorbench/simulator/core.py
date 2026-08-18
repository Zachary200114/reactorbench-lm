"""Deterministic first-fault Aster-A structured simulator."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from reactorbench.schemas import (
    SCHEMA_VERSION,
    AbstentionReason,
    ActionLabel,
    CanonicalEvent,
    ChannelQuality,
    ComponentLatentState,
    ComponentState,
    DecisionTarget,
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
    StateVariable,
    StructuredTrajectory,
)

from .content_guard import assert_no_prohibited_content

GENERATOR_VERSION = "0.1.0"
_MIN_DURATION = 8
_MAX_DURATION = 64
_MAX_SEED = 4_294_967_295
_NOISE_BOUND = 0.006
_MAX_TICK_STEP = 0.03


class UnsupportedScenarioError(ValueError):
    """Raised when a valid generic schema record is outside this milestone."""


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    subsystem: str


@dataclass(frozen=True)
class ChannelSpec:
    channel_id: str
    variable: StateVariable
    component_id: str


@dataclass(frozen=True)
class AsterVariantSpec:
    version: str
    plant_variant: PlantVariant
    components: tuple[ComponentSpec, ...]
    channels: tuple[ChannelSpec, ...]
    aliases: tuple[str, ...]
    baseline_noise_bound: float
    max_per_tick_step: float
    primary_train_ids: tuple[str, ...]
    support_bus_ids: tuple[str, ...]
    instrumentation_id: str


_PRIMARY_TRAINS = ("aster-train-cirrus", "aster-train-kestrel")
_SUPPORT_BUSES = ("aster-bus-rill", "aster-bus-quill")
_INSTRUMENTATION = "aster-instrument-vireo"
_COMPONENTS = (
    *(ComponentSpec(component_id, "PRIMARY_LOOP") for component_id in _PRIMARY_TRAINS),
    *(ComponentSpec(component_id, "SUPPORT_POWER") for component_id in _SUPPORT_BUSES),
    ComponentSpec(_INSTRUMENTATION, "INSTRUMENTATION"),
)
_CHANNELS = tuple(
    ChannelSpec(
        channel_id=f"aster-{variable.value.replace('_', '-')}-a",
        variable=variable,
        component_id=_PRIMARY_TRAINS[0]
        if variable is StateVariable.PRIMARY_FLOW
        else _INSTRUMENTATION,
    )
    for variable in StateVariable
) + tuple(
    ChannelSpec(
        channel_id=f"aster-{variable.value.replace('_', '-')}-b",
        variable=variable,
        component_id=_PRIMARY_TRAINS[0]
        if variable is StateVariable.PRIMARY_FLOW
        else _INSTRUMENTATION,
    )
    for variable in StateVariable
)
ASTER_A_SPEC = AsterVariantSpec(
    version=GENERATOR_VERSION,
    plant_variant=PlantVariant.ASTER_A,
    components=_COMPONENTS,
    channels=_CHANNELS,
    aliases=("cirrus", "kestrel", "rill", "quill", "vireo"),
    baseline_noise_bound=_NOISE_BOUND,
    max_per_tick_step=_MAX_TICK_STEP,
    primary_train_ids=_PRIMARY_TRAINS,
    support_bus_ids=_SUPPORT_BUSES,
    instrumentation_id=_INSTRUMENTATION,
)


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


def build_stable_scenario(*, seed: int, duration_ticks: int = 12) -> ScenarioDefinition:
    """Build the sole supported no-fault Aster-A scenario shape."""

    _require_uint32(seed, name="seed")
    _require_duration(duration_ticks)
    scenario = ScenarioDefinition(
        scenario_id=f"aster-a-stable-{seed}-{duration_ticks}",
        plant_variant_id=PlantVariant.ASTER_A,
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


def _baseline_values(seed: int) -> PlantValues:
    values: dict[str, float] = {}
    for index, variable in enumerate(StateVariable):
        stream = seed * 1_000_003 + index * 7_919 + 17
        values[variable.value] = round(
            0.47 + Random(stream).uniform(-0.045, 0.045),  # noqa: S311
            6,
        )
    return PlantValues(**values)


def _latent_states(scenario: ScenarioDefinition) -> tuple[LatentPlantState, ...]:
    values = _baseline_values(scenario.seed)
    components = tuple(
        ComponentLatentState(
            component_id=component.component_id,
            state=ComponentState.AVAILABLE,
            health=1.0,
        )
        for component in ASTER_A_SPEC.components
    )
    return tuple(
        LatentPlantState(
            tick=tick,
            operating_mode=OperatingMode.STABLE,
            values=values,
            components=components,
        )
        for tick in range(scenario.duration_ticks)
    )


def _noise(seed: int, tick: int, variable_index: int) -> float:
    stream = seed * 2_000_033 + tick * 4_099 + variable_index * 101 + 29
    return Random(stream).uniform(-_NOISE_BOUND, _NOISE_BOUND)  # noqa: S311


def _clip(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 6)


def _drift_bias(*, scenario: ScenarioDefinition, tick: int, channel_id: str) -> float:
    if not scenario.fault_injections:
        return 0.0
    injection = scenario.fault_injections[0]
    if tick <= injection.onset_tick or channel_id != injection.channel_id:
        return 0.0
    plateau = {SeverityBand.LOW: 0.042, SeverityBand.MEDIUM: 0.056, SeverityBand.HIGH: 0.07}[
        injection.severity
    ]
    magnitude = min(plateau, plateau * (tick - injection.onset_tick) / 3)
    direction = 1.0 if scenario.seed % 2 == 0 else -1.0
    return direction * magnitude


def _selected_channel_status(
    scenario: ScenarioDefinition, tick: int, channel_id: str
) -> ObservationStatus:
    if not scenario.fault_injections or channel_id != scenario.fault_injections[0].channel_id:
        return ObservationStatus.NORMAL
    onset_tick = scenario.fault_injections[0].onset_tick
    if tick <= onset_tick:
        return ObservationStatus.NORMAL
    if tick == onset_tick + 1:
        return ObservationStatus.WATCH
    return ObservationStatus.CONFLICTING


def _selected_channel_quality(
    scenario: ScenarioDefinition, tick: int, channel_id: str
) -> ChannelQuality:
    if (
        scenario.fault_injections
        and channel_id == scenario.fault_injections[0].channel_id
        and tick >= scenario.fault_injections[0].onset_tick + 4
    ):
        return ChannelQuality.SUSPECT
    return ChannelQuality.GOOD


def _observations(
    scenario: ScenarioDefinition, latent_states: tuple[LatentPlantState, ...]
) -> tuple[ObservationFrame, ...]:
    frames: list[ObservationFrame] = []
    for latent in latent_states:
        channels: list[SensorChannelObservation] = []
        for index, variable in enumerate(StateVariable):
            base = getattr(latent.values, variable.value) + _noise(
                scenario.seed, latent.tick, index
            )
            for channel in (
                channel for channel in ASTER_A_SPEC.channels if channel.variable is variable
            ):
                channels.append(
                    SensorChannelObservation(
                        channel_id=channel.channel_id,
                        variable=variable,
                        value=_clip(
                            base
                            + _drift_bias(
                                scenario=scenario, tick=latent.tick, channel_id=channel.channel_id
                            )
                        ),
                        quality=_selected_channel_quality(
                            scenario, latent.tick, channel.channel_id
                        ),
                        status=_selected_channel_status(scenario, latent.tick, channel.channel_id),
                    )
                )
        overall_status = ObservationStatus.NORMAL
        if scenario.fault_injections:
            selected = scenario.fault_injections[0].channel_id or _INSTRUMENTATION
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


def _events_and_targets(
    scenario: ScenarioDefinition, observations: tuple[ObservationFrame, ...]
) -> tuple[tuple[CanonicalEvent, ...], ScenarioTargets]:
    events: list[CanonicalEvent] = []
    stable = _event(
        events,
        sim_time=0,
        event_type=EventType.BENIGN_NOTE,
        subject_id=_INSTRUMENTATION,
        evidence_slots=(EvidenceSlot.STABLE_OPERATION, EvidenceSlot.RELATED_STATE_STABLE),
    )
    if not scenario.fault_injections:
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
    if scenario.plant_variant_id is not PlantVariant.ASTER_A:
        raise UnsupportedScenarioError("only ASTER-A is supported")
    if scenario.driver is not ScenarioDriver.STEADY_OPERATION:
        raise UnsupportedScenarioError("only STEADY_OPERATION is supported")
    try:
        _require_duration(scenario.duration_ticks)
    except ValueError as error:
        raise UnsupportedScenarioError(str(error)) from error
    if len(scenario.fault_injections) > 1:
        raise UnsupportedScenarioError("only zero or one injection is supported")
    if not scenario.fault_injections:
        expected: tuple[ScenarioAction, ...] = (
            ScenarioAction(
                decision_tick=scenario.duration_ticks - 1,
                action=ActionLabel.CONTINUE_MONITORING,
            ),
        )
        if scenario.action_sequence != expected:
            raise UnsupportedScenarioError("stable scenario action sequence is noncanonical")
        return
    injection = scenario.fault_injections[0]
    if injection.fault_family is not FaultFamily.SENSOR_DRIFT:
        raise UnsupportedScenarioError("only SENSOR_DRIFT is supported")
    channel_id = injection.channel_id
    if channel_id is None:
        raise UnsupportedScenarioError("sensor drift requires a primary-flow channel")
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
    assert_no_prohibited_content(ASTER_A_SPEC)
    assert_no_prohibited_content(trace)
    return trace
