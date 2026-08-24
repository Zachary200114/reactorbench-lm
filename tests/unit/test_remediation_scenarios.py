"""Regression evidence for the v0.3 remediation scenario inventory.

These tests construct only in-memory, project-authored Aster Station scenarios.
They do not render or write dataset artifacts and do not open held-out or golden
payloads.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from reactorbench.dataset.config import (
    DevelopmentDatasetConfig,
    load_development_dataset_config,
)
from reactorbench.dataset.pipeline import (
    ScopedProjectionInventory,
    build_scoped_projection_inventory,
)
from reactorbench.dataset.scenarios import (
    PlannedScenario,
    build_development_scenario_plan,
    build_scenario_plan_for_splits,
)
from reactorbench.schemas.enums import SplitName

ROOT = Path(__file__).resolve().parents[2]
V03_CONFIG = ROOT / "configs/dataset/remediation-development-v0.3.0.toml"
SOURCE_COMMIT = "abcdef1"
DEVELOPMENT_SPLITS = (SplitName.IID_TRAIN, SplitName.IID_VALIDATION)


@pytest.fixture(scope="module")
def v03_config() -> DevelopmentDatasetConfig:
    return load_development_dataset_config(V03_CONFIG)


@pytest.fixture(scope="module")
def scoped_plans(
    v03_config: DevelopmentDatasetConfig,
) -> tuple[PlannedScenario, ...]:
    return build_scenario_plan_for_splits(v03_config, splits=DEVELOPMENT_SPLITS)


@pytest.fixture(scope="module")
def scoped_inventory_pair(
    v03_config: DevelopmentDatasetConfig,
) -> tuple[ScopedProjectionInventory, ScopedProjectionInventory]:
    def build() -> ScopedProjectionInventory:
        return build_scoped_projection_inventory(
            v03_config,
            generator_commit=SOURCE_COMMIT,
            splits=DEVELOPMENT_SPLITS,
        )

    return build(), build()


def test_v03_authored_scenario_plan_has_the_frozen_inventory(
    v03_config: DevelopmentDatasetConfig,
) -> None:
    first = build_development_scenario_plan(v03_config)
    second = build_development_scenario_plan(v03_config)

    assert v03_config.dataset.dataset_version == "0.3.0"
    assert first == second
    assert len(first) == 412
    assert len({plan.scenario.scenario_id for plan in first}) == 412
    assert Counter(plan.split_name for plan in first) == {
        SplitName.IID_TRAIN: 184,
        SplitName.IID_VALIDATION: 69,
        SplitName.IID_TEST: 42,
        SplitName.TEMPLATE_TEST: 42,
        SplitName.COMPONENT_TEST: 15,
        SplitName.SEVERITY_TEST: 6,
        SplitName.COMPOSITION_TEST: 12,
        SplitName.COUNTERFACTUAL_TEST: 27,
        SplitName.NOISE_TEST: 15,
    }


def test_v03_scoped_scenario_plan_materializes_only_train_and_validation(
    v03_config: DevelopmentDatasetConfig,
    scoped_plans: tuple[PlannedScenario, ...],
) -> None:
    repeated = build_scenario_plan_for_splits(v03_config, splits=DEVELOPMENT_SPLITS)

    assert scoped_plans == repeated
    assert len(scoped_plans) == 253
    assert Counter(plan.split_name for plan in scoped_plans) == {
        SplitName.IID_TRAIN: 184,
        SplitName.IID_VALIDATION: 69,
    }
    assert {plan.split_name for plan in scoped_plans} == set(DEVELOPMENT_SPLITS)
    seed_owners: dict[int, SplitName] = {}
    for plan in scoped_plans:
        previous = seed_owners.setdefault(plan.scenario.seed, plan.split_name)
        assert previous is plan.split_name


def test_v03_scoped_projection_inventory_has_frozen_counts_and_is_repeatable(
    scoped_inventory_pair: tuple[ScopedProjectionInventory, ScopedProjectionInventory],
) -> None:
    first, second = scoped_inventory_pair

    assert first == second
    assert first.requested_splits == DEVELOPMENT_SPLITS
    assert len(first.trajectories) == 253
    assert len(first.projections) == 1_892
    assert len(first.counterfactual_projections) == 55

    trajectory_splits = {
        record.trajectory.trajectory_id: record.split_name for record in first.trajectories
    }
    assert Counter(trajectory_splits.values()) == {
        SplitName.IID_TRAIN: 184,
        SplitName.IID_VALIDATION: 69,
    }
    projection_splits = {
        projection.projection_id: trajectory_splits[projection.lineage.trajectory_id]
        for projection in first.projections
    }
    assert Counter(projection_splits.values()) == {
        SplitName.IID_TRAIN: 1_376,
        SplitName.IID_VALIDATION: 516,
    }
    assert Counter(
        projection_splits[pair.lineage.baseline_projection_id]
        for pair in first.counterfactual_projections
    ) == {
        SplitName.IID_TRAIN: 40,
        SplitName.IID_VALIDATION: 15,
    }


def test_v03_scoped_groups_and_pairs_never_cross_split_boundaries(
    scoped_inventory_pair: tuple[ScopedProjectionInventory, ScopedProjectionInventory],
) -> None:
    inventory = scoped_inventory_pair[0]
    scenario_splits = {
        record.trajectory.scenario_id: record.split_name for record in inventory.trajectories
    }
    trajectory_splits = {
        record.trajectory.trajectory_id: record.split_name for record in inventory.trajectories
    }
    projection_splits = {
        projection.projection_id: trajectory_splits[projection.lineage.trajectory_id]
        for projection in inventory.projections
    }

    assert len(inventory.groups) == 66
    assert sum(group.is_complete for group in inventory.groups) == 44
    assert len({group.counterfactual_group_id for group in inventory.groups}) == 66
    for group in inventory.groups:
        member_splits = {scenario_splits[member.scenario_id] for member in group.members}
        assert len(member_splits) == 1
        assert member_splits <= set(DEVELOPMENT_SPLITS)

    complete_group_ids = {
        group.counterfactual_group_id for group in inventory.groups if group.is_complete
    }
    for pair in inventory.counterfactual_projections:
        baseline_split = projection_splits[pair.lineage.baseline_projection_id]
        counterfactual_split = projection_splits[pair.lineage.counterfactual_projection_id]
        assert baseline_split is counterfactual_split
        assert baseline_split in DEVELOPMENT_SPLITS
        assert pair.lineage.counterfactual_group_id in complete_group_ids
