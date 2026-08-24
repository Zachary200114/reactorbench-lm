"""Deterministic split-first scenario matrix for the Phase 3 development candidate."""

from __future__ import annotations

from dataclasses import dataclass

from reactorbench.dataset.config import DevelopmentDatasetConfig
from reactorbench.schemas.base import canonical_sha256
from reactorbench.schemas.enums import ComponentState, PlantVariant, SeverityBand, SplitName
from reactorbench.schemas.scenario import ScenarioDefinition
from reactorbench.simulator import (
    build_abstract_inventory_loss_scenario,
    build_flow_imbalance_scenario,
    build_load_transient_scenario,
    build_pump_degradation_scenario,
    build_pump_degradation_sensor_drift_scenario,
    build_pump_trip_scenario,
    build_sensor_drift_scenario,
    build_sensor_noise_scenario,
    build_sensor_stuck_load_scenario,
    build_sparse_primary_flow_scenario,
    build_stable_scenario,
    build_support_power_interruption_scenario,
    build_thermal_sensor_drift_scenario,
    build_transfer_efficiency_loss_scenario,
    build_valve_lag_scenario,
    build_valve_stuck_scenario,
)


@dataclass(frozen=True, slots=True)
class PlannedScenario:
    """Audit-only generation plan fixed before any narrative rendering."""

    split_name: SplitName
    case_family: str
    scenario: ScenarioDefinition
    template_family_id: str
    alias_family_id: str
    counterfactual_family: str | None = None
    counterfactual_variant: str | None = None
    corruption_plan: str = "none"


_ALIAS_PLAN_OVERRIDES: dict[tuple[SplitName, int, str], str] = {
    # Explicit pre-render balancing corrections found by the task-scoped joint nuisance
    # audit. They are keyed only by split/seed/case plan and never by a task target.
    (SplitName.TEMPLATE_TEST, 1300, "g03-drift"): "short-v1",
    (SplitName.TEMPLATE_TEST, 1300, "g05-noise"): "canonical-v1",
    (SplitName.NOISE_TEST, 1800, "g02-load"): "short-v1",
    (SplitName.NOISE_TEST, 1800, "g03-drift"): "neutral-role-v1",
    (SplitName.NOISE_TEST, 1800, "g10-transfer-b"): "neutral-role-v1",
}


def _style_for(
    config: DevelopmentDatasetConfig,
    *,
    split_name: SplitName,
    seed: int,
    style_key: str,
) -> tuple[str, str]:
    split = getattr(config.splits, split_name.value)
    digest = canonical_sha256(
        {"split_name": split_name.value, "seed": seed, "style_key": style_key}
    )
    selector = int(digest[:16], 16)
    template = split.template_families[selector % len(split.template_families)]
    alias = _ALIAS_PLAN_OVERRIDES.get(
        (split_name, seed, style_key),
        split.alias_families[
            (selector // len(split.template_families)) % len(split.alias_families)
        ],
    )
    if alias not in split.alias_families:
        raise ValueError("renderer-plan alias override is not permitted in its split")
    return template, alias


def _plan(
    config: DevelopmentDatasetConfig,
    *,
    split_name: SplitName,
    case_family: str,
    scenario: ScenarioDefinition,
    style_key: str | None = None,
    counterfactual_family: str | None = None,
    counterfactual_variant: str | None = None,
    corruption_plan: str = "none",
) -> PlannedScenario:
    template, alias = _style_for(
        config,
        split_name=split_name,
        seed=scenario.seed,
        style_key=style_key or case_family,
    )
    return PlannedScenario(
        split_name=split_name,
        case_family=case_family,
        scenario=scenario,
        template_family_id=template,
        alias_family_id=alias,
        counterfactual_family=counterfactual_family,
        counterfactual_variant=counterfactual_variant,
        corruption_plan=corruption_plan,
    )


def _iid_scenarios(
    config: DevelopmentDatasetConfig, *, split_name: SplitName
) -> list[PlannedScenario]:
    duration = config.dataset.duration_ticks
    plans: list[PlannedScenario] = []
    for seed in getattr(config.splits, split_name.value).seeds:
        definitions: tuple[tuple[str, ScenarioDefinition], ...] = (
            (
                "g01-stable-a",
                build_stable_scenario(
                    seed=seed, duration_ticks=duration, plant_variant=PlantVariant.ASTER_A
                ),
            ),
            (
                "g01-stable-b",
                build_stable_scenario(
                    seed=seed, duration_ticks=duration, plant_variant=PlantVariant.ASTER_B
                ),
            ),
            ("g02-load", build_load_transient_scenario(seed=seed, duration_ticks=duration)),
            ("g03-drift", build_sensor_drift_scenario(seed=seed, duration_ticks=duration)),
            ("g05-noise", build_sensor_noise_scenario(seed=seed, duration_ticks=duration)),
            (
                "g06-pump-degradation",
                build_pump_degradation_scenario(seed=seed, duration_ticks=duration),
            ),
            (
                "g10-transfer-a",
                build_transfer_efficiency_loss_scenario(
                    seed=seed, duration_ticks=duration, plant_variant=PlantVariant.ASTER_A
                ),
            ),
            (
                "g10-transfer-b",
                build_transfer_efficiency_loss_scenario(
                    seed=seed, duration_ticks=duration, plant_variant=PlantVariant.ASTER_B
                ),
            ),
            (
                "g11-flow-a",
                build_flow_imbalance_scenario(
                    seed=seed, duration_ticks=duration, plant_variant=PlantVariant.ASTER_A
                ),
            ),
            (
                "g11-flow-b",
                build_flow_imbalance_scenario(
                    seed=seed, duration_ticks=duration, plant_variant=PlantVariant.ASTER_B
                ),
            ),
            (
                "g13-inventory-a",
                build_abstract_inventory_loss_scenario(
                    seed=seed, duration_ticks=duration, plant_variant=PlantVariant.ASTER_A
                ),
            ),
            (
                "g13-inventory-b",
                build_abstract_inventory_loss_scenario(
                    seed=seed, duration_ticks=duration, plant_variant=PlantVariant.ASTER_B
                ),
            ),
            (
                "g15-sparse-a",
                build_sparse_primary_flow_scenario(
                    seed=seed, duration_ticks=duration, plant_variant=PlantVariant.ASTER_A
                ),
            ),
            (
                "g15-sparse-b",
                build_sparse_primary_flow_scenario(
                    seed=seed, duration_ticks=duration, plant_variant=PlantVariant.ASTER_B
                ),
            ),
        )
        plans.extend(
            _plan(
                config,
                split_name=split_name,
                case_family=family,
                scenario=scenario,
            )
            for family, scenario in definitions
        )
        # Remediation v0.3 adds group-atomic near-neighbor comparisons to the fit and
        # validation inventories.  Keep this version-gated so the historical Phase 3
        # and v0.2 inventories remain byte-for-byte reproducible.  G14 stays reserved
        # for the composition shadow; only the already implemented G07/G08/G09/G12
        # structural families are admitted here.
        if config.dataset.dataset_version == "0.3.0" and split_name in {
            SplitName.IID_TRAIN,
            SplitName.IID_VALIDATION,
        }:
            plans.extend(
                _counterfactual_family_plans(
                    config,
                    split_name=split_name,
                    seed=seed,
                )
            )
    return plans


def _counterfactual_family_plans(
    config: DevelopmentDatasetConfig,
    *,
    split_name: SplitName,
    seed: int,
) -> tuple[PlannedScenario, ...]:
    """Build one complete, target-independent G07/G08/G09/G12 family matrix."""

    duration = config.dataset.duration_ticks
    plans: list[PlannedScenario] = []
    g07 = (
        (
            "g07-standby-available",
            "standby_available",
            build_pump_trip_scenario(
                seed=seed,
                duration_ticks=duration,
                standby_state=ComponentState.AVAILABLE,
            ),
        ),
        (
            "g07-standby-unavailable",
            "standby_unavailable",
            build_pump_trip_scenario(
                seed=seed,
                duration_ticks=duration,
                standby_state=ComponentState.UNAVAILABLE,
            ),
        ),
    )
    plans.extend(
        _plan(
            config,
            split_name=split_name,
            case_family=case_family,
            scenario=scenario,
            style_key="g07-standby",
            counterfactual_family="g07-standby",
            counterfactual_variant=variant,
        )
        for case_family, variant, scenario in g07
    )

    g08_g09 = (
        (
            "g08-lag-3",
            "valve_lag_3",
            build_valve_lag_scenario(seed=seed, duration_ticks=duration, lag_ticks=3),
        ),
        (
            "g08-lag-4",
            "valve_lag_4",
            build_valve_lag_scenario(seed=seed, duration_ticks=duration, lag_ticks=4),
        ),
        (
            "g09-stuck",
            "valve_stuck",
            build_valve_stuck_scenario(seed=seed, duration_ticks=duration),
        ),
    )
    plans.extend(
        _plan(
            config,
            split_name=split_name,
            case_family=case_family,
            scenario=scenario,
            style_key="g08-g09-valve",
            counterfactual_family="g08-g09-valve",
            counterfactual_variant=variant,
        )
        for case_family, variant, scenario in g08_g09
    )

    for plant_variant in (PlantVariant.ASTER_A, PlantVariant.ASTER_B):
        group_family = f"g12-map-{plant_variant.value.lower()}"
        for included in (True, False):
            variant = "map_included" if included else "map_withheld"
            plans.append(
                _plan(
                    config,
                    split_name=split_name,
                    case_family=f"{group_family}-{variant}",
                    scenario=build_support_power_interruption_scenario(
                        seed=seed,
                        duration_ticks=duration,
                        plant_variant=plant_variant,
                        include_dependency_map=included,
                    ),
                    style_key=group_family,
                    counterfactual_family=group_family,
                    counterfactual_variant=variant,
                )
            )
    return tuple(plans)


def _component_holdout(config: DevelopmentDatasetConfig) -> list[PlannedScenario]:
    split_name = SplitName.COMPONENT_TEST
    duration = config.dataset.duration_ticks
    plans: list[PlannedScenario] = []
    for seed in config.splits.component_test.seeds:
        definitions = (
            (
                "g01-stable-c",
                build_stable_scenario(
                    seed=seed, duration_ticks=duration, plant_variant=PlantVariant.ASTER_C
                ),
            ),
            (
                "g10-transfer-c",
                build_transfer_efficiency_loss_scenario(
                    seed=seed, duration_ticks=duration, plant_variant=PlantVariant.ASTER_C
                ),
            ),
            (
                "g11-flow-c",
                build_flow_imbalance_scenario(
                    seed=seed, duration_ticks=duration, plant_variant=PlantVariant.ASTER_C
                ),
            ),
            (
                "g13-inventory-c",
                build_abstract_inventory_loss_scenario(
                    seed=seed, duration_ticks=duration, plant_variant=PlantVariant.ASTER_C
                ),
            ),
            (
                "g15-sparse-c",
                build_sparse_primary_flow_scenario(
                    seed=seed, duration_ticks=duration, plant_variant=PlantVariant.ASTER_C
                ),
            ),
        )
        plans.extend(
            _plan(
                config,
                split_name=split_name,
                case_family=family,
                scenario=scenario,
            )
            for family, scenario in definitions
        )
    return plans


def _severity_holdout(config: DevelopmentDatasetConfig) -> list[PlannedScenario]:
    plans: list[PlannedScenario] = []
    for seed in config.splits.severity_test.seeds:
        for severity in (SeverityBand.MEDIUM, SeverityBand.HIGH):
            plans.append(
                _plan(
                    config,
                    split_name=SplitName.SEVERITY_TEST,
                    case_family=f"g03-drift-{severity.value.lower()}",
                    scenario=build_sensor_drift_scenario(
                        seed=seed,
                        duration_ticks=config.dataset.duration_ticks,
                        severity=severity,
                    ),
                    style_key="g03-drift-severity",
                )
            )
    return plans


def _composition_holdout(config: DevelopmentDatasetConfig) -> list[PlannedScenario]:
    plans: list[PlannedScenario] = []
    duration = config.dataset.duration_ticks
    for seed in config.splits.composition_test.seeds:
        plans.append(
            _plan(
                config,
                split_name=SplitName.COMPOSITION_TEST,
                case_family="g04-stuck-load",
                scenario=build_sensor_stuck_load_scenario(seed=seed, duration_ticks=duration),
            )
        )
        family = "g14-factor"
        scenarios = (
            (
                "g14-pump-only",
                "pump_only",
                build_pump_degradation_scenario(seed=seed, duration_ticks=duration),
            ),
            (
                "g14-thermal-drift-only",
                "sensor_only",
                build_thermal_sensor_drift_scenario(seed=seed, duration_ticks=duration),
            ),
            (
                "g14-compound",
                "compound",
                build_pump_degradation_sensor_drift_scenario(seed=seed, duration_ticks=duration),
            ),
        )
        plans.extend(
            _plan(
                config,
                split_name=SplitName.COMPOSITION_TEST,
                case_family=case_family,
                scenario=scenario,
                style_key=family,
                counterfactual_family=family,
                counterfactual_variant=variant,
            )
            for case_family, variant, scenario in scenarios
        )
    return plans


def _counterfactual_holdout(config: DevelopmentDatasetConfig) -> list[PlannedScenario]:
    plans: list[PlannedScenario] = []
    for seed in config.splits.counterfactual_test.seeds:
        plans.extend(
            _counterfactual_family_plans(
                config,
                split_name=SplitName.COUNTERFACTUAL_TEST,
                seed=seed,
            )
        )
    return plans


def _noise_holdout(config: DevelopmentDatasetConfig) -> list[PlannedScenario]:
    plans: list[PlannedScenario] = []
    duration = config.dataset.duration_ticks
    for seed in config.splits.noise_test.seeds:
        definitions = (
            (
                "g01-stable-a",
                build_stable_scenario(seed=seed, duration_ticks=duration),
            ),
            ("g02-load", build_load_transient_scenario(seed=seed, duration_ticks=duration)),
            ("g03-drift", build_sensor_drift_scenario(seed=seed, duration_ticks=duration)),
            ("g05-noise", build_sensor_noise_scenario(seed=seed, duration_ticks=duration)),
            (
                "g10-transfer-b",
                build_transfer_efficiency_loss_scenario(
                    seed=seed, duration_ticks=duration, plant_variant=PlantVariant.ASTER_B
                ),
            ),
        )
        plans.extend(
            _plan(
                config,
                split_name=SplitName.NOISE_TEST,
                case_family=family,
                scenario=scenario,
                corruption_plan="balanced-matrix-v1",
            )
            for family, scenario in definitions
        )
    return plans


def build_scenario_plan_for_splits(
    config: DevelopmentDatasetConfig,
    *,
    splits: tuple[SplitName, ...],
) -> tuple[PlannedScenario, ...]:
    """Build only the requested split recipes without touching any other split.

    This is the physical isolation boundary used by model remediation.  In
    particular, a development inventory can request ``iid_train`` and
    ``iid_validation`` without constructing held-out scenarios and filtering them
    afterward.
    """

    if type(config) is not DevelopmentDatasetConfig:
        raise TypeError("config must be an exact DevelopmentDatasetConfig")
    if type(splits) is not tuple or not splits:
        raise ValueError("splits must be a non-empty exact tuple")
    if any(type(split) is not SplitName for split in splits):
        raise TypeError("every requested split must be an exact SplitName")
    if len(splits) != len(set(splits)):
        raise ValueError("requested splits must be unique")

    plans: list[PlannedScenario] = []
    for split_name in splits:
        if split_name in {
            SplitName.IID_TRAIN,
            SplitName.IID_VALIDATION,
            SplitName.IID_TEST,
            SplitName.TEMPLATE_TEST,
        }:
            plans.extend(_iid_scenarios(config, split_name=split_name))
        elif split_name is SplitName.COMPONENT_TEST:
            plans.extend(_component_holdout(config))
        elif split_name is SplitName.SEVERITY_TEST:
            plans.extend(_severity_holdout(config))
        elif split_name is SplitName.COMPOSITION_TEST:
            plans.extend(_composition_holdout(config))
        elif split_name is SplitName.COUNTERFACTUAL_TEST:
            plans.extend(_counterfactual_holdout(config))
        elif split_name is SplitName.NOISE_TEST:
            plans.extend(_noise_holdout(config))
        else:  # pragma: no cover - closed enum defense
            raise ValueError(f"unsupported requested split: {split_name.value}")

    result = tuple(plans)
    if not result:
        raise ValueError("requested split plan produced no scenarios")
    scenario_ids = tuple(plan.scenario.scenario_id for plan in result)
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("development scenario IDs must be globally unique")
    seed_splits: dict[int, SplitName] = {}
    for plan in result:
        previous = seed_splits.setdefault(plan.scenario.seed, plan.split_name)
        if previous is not plan.split_name:
            raise ValueError("a development seed cannot cross split boundaries")
    return result


def build_development_scenario_plan(
    config: DevelopmentDatasetConfig,
) -> tuple[PlannedScenario, ...]:
    """Build the complete deterministic, versioned split-first scenario plan."""

    result = build_scenario_plan_for_splits(config, splits=tuple(SplitName))
    if not (
        config.dataset.minimum_trajectories <= len(result) <= config.dataset.maximum_trajectories
    ):
        raise ValueError("development scenario count is outside the reviewed configuration bounds")
    return result


__all__ = [
    "PlannedScenario",
    "build_development_scenario_plan",
    "build_scenario_plan_for_splits",
]
