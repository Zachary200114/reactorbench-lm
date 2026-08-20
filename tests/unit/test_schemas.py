from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

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
    SplitName,
    StandbyContext,
    StateVariable,
    TaskName,
)

ROOT = Path(__file__).resolve().parents[2]
PLANT_SPEC = ROOT / "research" / "FICTIONAL_PLANT_SPEC.md"


def _values(**overrides: object) -> PlantValues:
    values: dict[str, object] = {variable.value: 0.5 for variable in StateVariable}
    values.update(overrides)
    return PlantValues.model_validate(values)


def _latent_state() -> LatentPlantState:
    return LatentPlantState(
        tick=0,
        operating_mode=OperatingMode.STABLE,
        values=_values(),
        components=(
            ComponentLatentState(
                component_id="train-cedar",
                state=ComponentState.AVAILABLE,
                health=1.0,
            ),
        ),
    )


def _observation() -> ObservationFrame:
    return ObservationFrame(
        tick=0,
        overall_status=ObservationStatus.NORMAL,
        channels=(
            SensorChannelObservation(
                channel_id="pf-primary",
                variable=StateVariable.PRIMARY_FLOW,
                value=0.5,
                quality=ChannelQuality.GOOD,
                status=ObservationStatus.NORMAL,
            ),
        ),
    )


def _event() -> CanonicalEvent:
    return CanonicalEvent(
        event_id="evt-000",
        event_index=0,
        sim_time=0,
        event_type=EventType.OBSERVATION_CHANGED,
        subject_id="pf-primary",
        variable=StateVariable.PRIMARY_FLOW,
        value_after=0.5,
        observation_status=ObservationStatus.NORMAL,
    )


def _provenance() -> ProvenanceRecord:
    return ProvenanceRecord(
        dataset_version="0.1.0",
        generator_commit="abcdef1",
        renderer_version="0.1.0",
        seed=17,
        trajectory_id="trajectory-001",
        scenario_id="scenario-001",
        plant_variant_id=PlantVariant.ASTER_B,
        fault_family_ids=(
            FaultFamily.PUMP_DEGRADATION,
            FaultFamily.SENSOR_DRIFT,
        ),
        template_family_ids=("template-b", "template-a"),
        split_name=SplitName.COMPOSITION_TEST,
        task_name=TaskName.FAULT_FAMILY,
    )


def _spec_section(text: str, start: str, end: str) -> str:
    return text.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]


def test_development_schema_version_is_not_frozen_v1() -> None:
    assert SCHEMA_VERSION == "0.1.0"


def test_enums_match_the_documented_plant_contract() -> None:
    text = PLANT_SPEC.read_text("utf-8")

    operating = _spec_section(text, "### 4.1", "### 4.2")
    documented_operating = set(re.findall(r"^- `([A-Z_]+)`", operating, re.MULTILINE))
    assert {item.value for item in OperatingMode} == documented_operating

    components = _spec_section(text, "### 4.2", "### 4.3")
    documented_components = set(re.findall(r"^- `([A-Z_]+)`", components, re.MULTILINE))
    assert {item.value for item in ComponentState} == documented_components

    observations = _spec_section(text, "### 4.3", "### 4.4")
    documented_observations = set(re.findall(r"^- `([A-Z_]+)`", observations, re.MULTILINE))
    assert {item.value for item in ObservationStatus} == documented_observations

    variables = _spec_section(text, "### 4.4", "## 5.")
    documented_variables = set(
        re.findall(r"^\| `[A-Z]+` \| `([a-z_]+)` \|", variables, re.MULTILINE)
    )
    assert {item.value for item in StateVariable} == documented_variables

    faults = _spec_section(text, "## 7.", "## 8.")
    documented_faults = set(re.findall(r"^\| `([A-Z_]+)`", faults, re.MULTILINE))
    assert ScenarioDriver.LOAD_TRANSIENT.value in documented_faults
    documented_faults.remove(ScenarioDriver.LOAD_TRANSIENT.value)
    assert {item.value for item in FaultFamily} == documented_faults

    actions = _spec_section(text, "## 8.", "## 9.")
    documented_actions = set(re.findall(r"^\| `([A-Z_]+)` \|", actions, re.MULTILINE))
    assert {item.value for item in ActionLabel} == documented_actions


def test_unknown_fields_are_rejected_and_models_are_frozen() -> None:
    payload = _values().model_dump()
    payload["unexpected"] = 0.2
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PlantValues.model_validate(payload)

    state = _latent_state()
    with pytest.raises(ValidationError, match="frozen_instance"):
        state.tick = 1


@pytest.mark.parametrize("bad_tick", [True, "1", 1.5])
def test_integer_fields_do_not_coerce_types(bad_tick: object) -> None:
    payload = _latent_state().model_dump()
    payload["tick"] = bad_tick
    with pytest.raises(ValidationError):
        LatentPlantState.model_validate(payload)


@pytest.mark.parametrize("bad_value", [-0.01, 1.01, math.nan, math.inf, -math.inf, "0.5"])
def test_normalized_values_are_finite_and_bounded(bad_value: object) -> None:
    with pytest.raises(ValidationError):
        _values(primary_flow=bad_value)


def test_observation_status_and_channel_quality_are_distinct() -> None:
    noisy = SensorChannelObservation(
        channel_id="pf-primary",
        variable=StateVariable.PRIMARY_FLOW,
        value=0.4,
        quality=ChannelQuality.NOISY,
        status=ObservationStatus.WATCH,
    )
    assert noisy.quality is ChannelQuality.NOISY
    assert noisy.status is ObservationStatus.WATCH

    with pytest.raises(ValidationError, match="must occur together"):
        SensorChannelObservation(
            channel_id="pf-primary",
            variable=StateVariable.PRIMARY_FLOW,
            value=None,
            quality=ChannelQuality.UNAVAILABLE,
            status=ObservationStatus.NORMAL,
        )


def test_latent_observation_and_event_contracts_cannot_carry_fault_labels() -> None:
    for model, record in (
        (LatentPlantState, _latent_state()),
        (ObservationFrame, _observation()),
        (CanonicalEvent, _event()),
    ):
        assert "fault_labels" not in model.model_fields
        assert "fault_family_ids" not in model.model_fields
        payload = record.model_dump()
        payload["fault_labels"] = (FaultFamily.SENSOR_DRIFT,)
        with pytest.raises(ValidationError, match="extra_forbidden"):
            model.model_validate(payload)


def test_sensor_fault_scope_is_explicit() -> None:
    with pytest.raises(ValidationError, match="sensor faults require channel_id"):
        FaultInjection(
            fault_family=FaultFamily.SENSOR_DRIFT,
            component_id="instrumentation",
            onset_tick=2,
            severity=SeverityBand.LOW,
        )

    with pytest.raises(ValidationError, match="process faults must not set channel_id"):
        FaultInjection(
            fault_family=FaultFamily.PUMP_TRIP,
            component_id="train-cedar",
            channel_id="pf-primary",
            onset_tick=2,
            severity=SeverityBand.HIGH,
        )


def test_fault_labels_behave_as_a_canonical_ordered_set() -> None:
    target = DecisionTarget(
        scenario_id="scenario-001",
        decision_tick=5,
        diagnosis_status=DiagnosisStatus.DIAGNOSED,
        fault_labels=(
            FaultFamily.PUMP_DEGRADATION,
            FaultFamily.SENSOR_DRIFT,
        ),
        evidence_event_ids=("evt-001",),
        immediate_action=ActionLabel.VERIFY_REDUNDANT_CHANNEL,
    )
    assert target.fault_labels == (
        FaultFamily.SENSOR_DRIFT,
        FaultFamily.PUMP_DEGRADATION,
    )

    payload = target.model_dump()
    payload["fault_labels"] = (
        FaultFamily.SENSOR_DRIFT,
        FaultFamily.SENSOR_DRIFT,
    )
    with pytest.raises(ValidationError, match="duplicates"):
        DecisionTarget.model_validate(payload)


def test_diagnosis_status_and_abstention_invariants() -> None:
    no_fault = DecisionTarget(
        scenario_id="scenario-001",
        decision_tick=2,
        diagnosis_status=DiagnosisStatus.NO_FAULT,
        immediate_action=ActionLabel.CONTINUE_MONITORING,
    )
    assert no_fault.fault_labels == ()
    assert no_fault.abstention_reason is None

    unresolved = DecisionTarget(
        scenario_id="scenario-001",
        decision_tick=3,
        diagnosis_status=DiagnosisStatus.UNRESOLVED,
        immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE,
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
    )
    assert unresolved.fault_labels == ()

    invalid = unresolved.model_dump()
    invalid["fault_labels"] = (FaultFamily.SENSOR_DRIFT,)
    with pytest.raises(ValidationError, match="UNRESOLVED requires an empty"):
        DecisionTarget.model_validate(invalid)


def test_scenario_actions_are_ordered_with_one_action_per_tick() -> None:
    scenario = ScenarioDefinition(
        scenario_id="scenario-001",
        plant_variant_id=PlantVariant.ASTER_A,
        seed=7,
        duration_ticks=10,
        driver=ScenarioDriver.LOAD_TRANSIENT,
        action_sequence=(
            ScenarioAction(decision_tick=3, action=ActionLabel.CONTINUE_MONITORING),
            ScenarioAction(decision_tick=6, action=ActionLabel.REDUCE_SIMULATED_LOAD),
        ),
    )
    assert tuple(item.decision_tick for item in scenario.action_sequence) == (3, 6)

    with pytest.raises(ValidationError, match="one scenario action"):
        ScenarioDefinition(
            scenario_id="scenario-001",
            plant_variant_id=PlantVariant.ASTER_A,
            seed=7,
            duration_ticks=10,
            driver=ScenarioDriver.LOAD_TRANSIENT,
            action_sequence=(
                ScenarioAction(decision_tick=3, action=ActionLabel.CONTINUE_MONITORING),
                ScenarioAction(decision_tick=3, action=ActionLabel.COMPARE_RELATED_TRENDS),
            ),
        )


def test_standby_context_is_strict_frozen_and_round_trips_with_scenario() -> None:
    context = StandbyContext(
        context_id="standby-context-001",
        active_train_id="train-cedar",
        standby_train_id="train-hemlock",
        standby_state=ComponentState.AVAILABLE,
        standby_support_bus_id="support-bus-hemlock",
        support_bus_state=ComponentState.AVAILABLE,
        standby_start_delay_ticks=2,
    )
    scenario = ScenarioDefinition(
        scenario_id="scenario-standby-001",
        plant_variant_id=PlantVariant.ASTER_A,
        seed=7,
        duration_ticks=10,
        driver=ScenarioDriver.STEADY_OPERATION,
        standby_context=context,
    )

    assert StandbyContext.model_validate_json(context.model_dump_json()) == context
    assert ScenarioDefinition.model_validate_json(scenario.model_dump_json()) == scenario
    assert scenario.standby_context == context
    with pytest.raises(ValidationError, match="frozen"):
        context.standby_state = ComponentState.UNAVAILABLE


def test_standby_context_rejects_unknown_fields_and_same_train_ids() -> None:
    payload: dict[str, object] = {
        "context_id": "standby-context-001",
        "active_train_id": "train-cedar",
        "standby_train_id": "train-hemlock",
        "standby_state": ComponentState.AVAILABLE,
        "standby_support_bus_id": "support-bus-hemlock",
        "support_bus_state": ComponentState.AVAILABLE,
        "standby_start_delay_ticks": 2,
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StandbyContext.model_validate({**payload, "policy_card": "not-permitted"})

    with pytest.raises(ValidationError, match="must be different"):
        StandbyContext.model_validate({**payload, "standby_train_id": "train-cedar"})


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("standby_state", ComponentState.DEGRADED),
        ("standby_state", ComponentState.STARTING),
        ("support_bus_state", ComponentState.SUSPECT),
        ("support_bus_state", ComponentState.UNKNOWN),
    ],
)
def test_standby_context_rejects_states_outside_dependency_vocabulary(
    field_name: str, bad_value: ComponentState
) -> None:
    context = StandbyContext(
        context_id="standby-context-001",
        active_train_id="train-cedar",
        standby_train_id="train-hemlock",
        standby_state=ComponentState.AVAILABLE,
        standby_support_bus_id="support-bus-hemlock",
        support_bus_state=ComponentState.AVAILABLE,
        standby_start_delay_ticks=2,
    )
    payload = context.model_dump()
    payload[field_name] = bad_value

    with pytest.raises(ValidationError, match=f"{field_name} must be AVAILABLE or UNAVAILABLE"):
        StandbyContext.model_validate(payload)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("context_id", 7),
        ("active_train_id", True),
        ("standby_state", "AVAILABLE"),
        ("standby_support_bus_id", 1.0),
        ("support_bus_state", 0),
        ("standby_start_delay_ticks", True),
        ("standby_start_delay_ticks", "2"),
        ("standby_start_delay_ticks", 2.0),
    ],
)
def test_standby_context_rejects_python_type_coercion(field_name: str, bad_value: object) -> None:
    context = StandbyContext(
        context_id="standby-context-001",
        active_train_id="train-cedar",
        standby_train_id="train-hemlock",
        standby_state=ComponentState.AVAILABLE,
        standby_support_bus_id="support-bus-hemlock",
        support_bus_state=ComponentState.UNAVAILABLE,
        standby_start_delay_ticks=2,
    )
    payload = context.model_dump()
    payload[field_name] = bad_value

    with pytest.raises(ValidationError):
        StandbyContext.model_validate(payload)


def _dependency_map_context() -> DependencyMapContext:
    return DependencyMapContext(
        plant_variant_id=PlantVariant.ASTER_B,
        links=(
            DependencyLink(
                support_bus_id="aster-bus-amber",
                dependent_component_id="aster-train-bravo",
            ),
            DependencyLink(
                support_bus_id="aster-bus-blue",
                dependent_component_id="aster-train-charlie",
            ),
        ),
    )


def test_dependency_map_context_is_strict_frozen_and_round_trips_with_scenario() -> None:
    context = _dependency_map_context()
    scenario = ScenarioDefinition(
        scenario_id="scenario-dependency-map-001",
        plant_variant_id=PlantVariant.ASTER_B,
        seed=7,
        duration_ticks=10,
        driver=ScenarioDriver.STEADY_OPERATION,
        dependency_map_context=context,
    )

    assert DependencyMapContext.model_validate_json(context.model_dump_json()) == context
    assert ScenarioDefinition.model_validate_json(scenario.model_dump_json()) == scenario
    assert scenario.dependency_map_context == context
    with pytest.raises(ValidationError, match="frozen"):
        context.links = ()


def test_scenario_rejects_dependency_map_for_a_different_variant() -> None:
    context = _dependency_map_context()
    with pytest.raises(ValidationError, match="must match plant_variant_id"):
        ScenarioDefinition(
            scenario_id="scenario-dependency-map-mismatch-001",
            plant_variant_id=PlantVariant.ASTER_A,
            seed=7,
            duration_ticks=10,
            driver=ScenarioDriver.STEADY_OPERATION,
            dependency_map_context=context,
        )


def test_scenario_model_copy_lookalike_is_revalidated_for_variant_context() -> None:
    scenario = ScenarioDefinition(
        scenario_id="scenario-dependency-map-copy-001",
        plant_variant_id=PlantVariant.ASTER_B,
        seed=7,
        duration_ticks=10,
        driver=ScenarioDriver.STEADY_OPERATION,
        dependency_map_context=_dependency_map_context(),
    )
    assert scenario.dependency_map_context is not None
    mismatched_context = scenario.dependency_map_context.model_copy(
        update={"plant_variant_id": PlantVariant.ASTER_A}
    )
    lookalike = scenario.model_copy(update={"dependency_map_context": mismatched_context})

    with pytest.raises(ValidationError, match="must match plant_variant_id"):
        ScenarioDefinition.model_validate(lookalike.model_dump(warnings=False))


def test_dependency_link_rejects_unknown_fields_and_identical_endpoints() -> None:
    payload: dict[str, object] = {
        "support_bus_id": "aster-bus-amber",
        "dependent_component_id": "aster-train-bravo",
    }
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DependencyLink.model_validate({**payload, "unexpected": "rejected"})
    with pytest.raises(ValidationError, match="must be different"):
        DependencyLink.model_validate({**payload, "dependent_component_id": "aster-bus-amber"})


@pytest.mark.parametrize(
    ("links", "message"),
    [
        ((), "at least 1 item"),
        (
            (
                DependencyLink(
                    support_bus_id="aster-bus-amber",
                    dependent_component_id="aster-train-bravo",
                ),
                DependencyLink(
                    support_bus_id="aster-bus-amber",
                    dependent_component_id="aster-train-bravo",
                ),
            ),
            "duplicate support-bus/dependent pairs",
        ),
        (
            (
                DependencyLink(
                    support_bus_id="aster-bus-amber",
                    dependent_component_id="aster-train-bravo",
                ),
                DependencyLink(
                    support_bus_id="aster-bus-blue",
                    dependent_component_id="aster-train-bravo",
                ),
            ),
            "exactly one support_bus_id",
        ),
        (
            (
                DependencyLink(
                    support_bus_id="aster-bus-blue",
                    dependent_component_id="aster-train-charlie",
                ),
                DependencyLink(
                    support_bus_id="aster-bus-amber",
                    dependent_component_id="aster-train-bravo",
                ),
            ),
            "canonical",
        ),
    ],
)
def test_dependency_map_context_rejects_empty_duplicate_conflicting_and_noncanonical_links(
    links: tuple[DependencyLink, ...], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        DependencyMapContext(plant_variant_id=PlantVariant.ASTER_B, links=links)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("plant_variant_id", "ASTER_B"),
        ("plant_variant_id", 2),
        ("links", []),
        (
            "links",
            [
                {
                    "support_bus_id": "aster-bus-amber",
                    "dependent_component_id": "aster-train-bravo",
                }
            ],
        ),
    ],
)
def test_dependency_map_context_rejects_coercion_and_noncanonical_containers(
    field_name: str, bad_value: object
) -> None:
    payload = _dependency_map_context().model_dump()
    payload[field_name] = bad_value
    with pytest.raises(ValidationError):
        DependencyMapContext.model_validate(payload)


def test_dependency_map_context_revalidates_model_copy_lookalikes() -> None:
    context = _dependency_map_context()
    lookalike = context.model_copy(update={"links": [*context.links]})
    with pytest.raises(ValidationError):
        DependencyMapContext.model_validate(lookalike.model_dump(warnings=False))


@pytest.mark.parametrize("bad_seed", [-1, 4_294_967_296, True, "7"])
def test_scenario_seed_is_a_strict_uint32(bad_seed: object) -> None:
    with pytest.raises(ValidationError):
        ScenarioDefinition.model_validate(
            {
                "scenario_id": "scenario-001",
                "plant_variant_id": PlantVariant.ASTER_A,
                "seed": bad_seed,
                "duration_ticks": 10,
                "driver": ScenarioDriver.STEADY_OPERATION,
            }
        )


def test_target_sequence_allows_one_immediate_action_per_tick() -> None:
    decision = DecisionTarget(
        scenario_id="scenario-001",
        decision_tick=2,
        diagnosis_status=DiagnosisStatus.NO_FAULT,
        immediate_action=ActionLabel.CONTINUE_MONITORING,
    )
    with pytest.raises(ValidationError, match="one immediate action"):
        ScenarioTargets(
            scenario_id="scenario-001",
            decisions=(decision, decision),
        )


def test_provenance_hash_is_stable_and_content_addressed() -> None:
    provenance = _provenance()
    assert provenance.fault_family_ids == (
        FaultFamily.SENSOR_DRIFT,
        FaultFamily.PUMP_DEGRADATION,
    )
    assert provenance.template_family_ids == ("template-a", "template-b")
    assert (
        provenance.stable_hash()
        == "bdf833bb2d36ce927638a774323089d01deff2ad4c0913b42ab35f80b3fa04b6"
    )
    assert provenance.stable_hash() == _provenance().stable_hash()
    changed = provenance.model_copy(update={"seed": 18})
    assert changed.stable_hash() != provenance.stable_hash()
