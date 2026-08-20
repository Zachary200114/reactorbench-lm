from __future__ import annotations

from collections import Counter
from pathlib import Path

from reactorbench.dataset.config import load_development_dataset_config
from reactorbench.dataset.scenarios import PlannedScenario, build_development_scenario_plan
from reactorbench.schemas.enums import FaultFamily, PlantVariant, SeverityBand, SplitName

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "dataset" / "development-v0.1.0.toml"


def test_development_scenario_plan_is_split_first_complete_and_deterministic() -> None:
    config = load_development_dataset_config(CONFIG)
    first = build_development_scenario_plan(config)
    second = build_development_scenario_plan(config)
    assert first == second
    assert len(first) == 204
    assert len({plan.scenario.scenario_id for plan in first}) == len(first)

    counts = Counter(plan.split_name for plan in first)
    assert counts == {
        SplitName.IID_TRAIN: 70,
        SplitName.IID_VALIDATION: 28,
        SplitName.IID_TEST: 28,
        SplitName.TEMPLATE_TEST: 28,
        SplitName.COMPONENT_TEST: 10,
        SplitName.SEVERITY_TEST: 4,
        SplitName.COMPOSITION_TEST: 8,
        SplitName.COUNTERFACTUAL_TEST: 18,
        SplitName.NOISE_TEST: 10,
    }

    seed_owners: dict[int, SplitName] = {}
    for plan in first:
        previous = seed_owners.setdefault(plan.scenario.seed, plan.split_name)
        assert previous is plan.split_name
        assert plan.scenario.seed > config.dataset.golden_reserved_seed_max


def test_holdouts_are_absent_from_training_and_groups_share_render_style() -> None:
    plans = build_development_scenario_plan(load_development_dataset_config(CONFIG))
    training = [plan for plan in plans if plan.split_name is SplitName.IID_TRAIN]
    assert all(plan.scenario.plant_variant_id is not PlantVariant.ASTER_C for plan in training)
    assert all(len(plan.scenario.fault_injections) <= 1 for plan in training)
    assert all(
        not (plan.scenario.driver.value == "LOAD_TRANSIENT" and plan.scenario.fault_injections)
        for plan in training
    )
    assert all(plan.template_family_id != "research-editorial-v1" for plan in training)
    assert all(plan.alias_family_id != "heldout-v1" for plan in training)

    grouped: dict[tuple[str, int], list[PlannedScenario]] = {}
    for plan in plans:
        if plan.counterfactual_family is not None:
            grouped.setdefault((plan.counterfactual_family, plan.scenario.seed), []).append(plan)
    assert {len(members) for members in grouped.values()} == {2, 3}
    for members in grouped.values():
        assert len({member.split_name for member in members}) == 1
        assert len({member.template_family_id for member in members}) == 1
        assert len({member.alias_family_id for member in members}) == 1

    g14 = [plan for plan in plans if plan.counterfactual_family == "g14-factor"]
    assert {plan.counterfactual_variant for plan in g14} == {
        "pump_only",
        "sensor_only",
        "compound",
    }
    assert all(plan.split_name is SplitName.COMPOSITION_TEST for plan in g14)
    compound = [plan for plan in g14 if plan.counterfactual_variant == "compound"]
    assert all(
        tuple(injection.fault_family for injection in plan.scenario.fault_injections)
        == (FaultFamily.SENSOR_DRIFT, FaultFamily.PUMP_DEGRADATION)
        for plan in compound
    )


def test_severity_and_noise_dimensions_are_independent() -> None:
    plans = build_development_scenario_plan(load_development_dataset_config(CONFIG))
    severity = [plan for plan in plans if plan.split_name is SplitName.SEVERITY_TEST]
    assert {
        injection.severity for plan in severity for injection in plan.scenario.fault_injections
    } == {SeverityBand.MEDIUM, SeverityBand.HIGH}

    noise = [plan for plan in plans if plan.split_name is SplitName.NOISE_TEST]
    assert all(plan.corruption_plan != "none" for plan in noise)
    assert any(not plan.scenario.fault_injections for plan in noise)
    assert any(
        any(
            injection.fault_family is FaultFamily.SENSOR_NOISE
            for injection in plan.scenario.fault_injections
        )
        for plan in noise
    )


def test_renderer_plan_overrides_balance_joint_nuisance_cues_without_target_lookup() -> None:
    plans = build_development_scenario_plan(load_development_dataset_config(CONFIG))
    by_split_seed_case = {
        (plan.split_name, plan.scenario.seed, plan.case_family): plan for plan in plans
    }

    expected_aliases = {
        (SplitName.TEMPLATE_TEST, 1300, "g03-drift"): "short-v1",
        (SplitName.TEMPLATE_TEST, 1300, "g05-noise"): "canonical-v1",
        (SplitName.NOISE_TEST, 1800, "g02-load"): "short-v1",
        (SplitName.NOISE_TEST, 1800, "g03-drift"): "neutral-role-v1",
        (SplitName.NOISE_TEST, 1800, "g10-transfer-b"): "neutral-role-v1",
    }
    for key, alias_family_id in expected_aliases.items():
        plan = by_split_seed_case[key]
        assert plan.alias_family_id == alias_family_id
        if key[0] is SplitName.TEMPLATE_TEST:
            assert plan.template_family_id == "research-editorial-v1"
            assert plan.corruption_plan == "none"
        else:
            assert plan.corruption_plan == "balanced-matrix-v1"
