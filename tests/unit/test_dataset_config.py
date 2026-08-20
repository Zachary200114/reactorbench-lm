from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from reactorbench.dataset.config import (
    DevelopmentDatasetConfig,
    canonical_dataset_config_bytes,
    dataset_config_sha256,
    load_development_dataset_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "dataset" / "development-v0.1.0.toml"


def _raw_config() -> dict[str, object]:
    with CONFIG_PATH.open("rb") as config_file:
        return tomllib.load(config_file)


def test_reviewed_dataset_config_is_strict_and_deterministic() -> None:
    first = load_development_dataset_config(CONFIG_PATH)
    second = load_development_dataset_config(CONFIG_PATH)
    assert first == second
    assert first.dataset.minimum_trajectories == 100
    assert first.dataset.maximum_trajectories == 500
    assert first.quality.ngram_sizes == (3, 4, 5)
    assert first.quality.fail_on_task_scoped_model_input_duplicate is True
    assert canonical_dataset_config_bytes(first) == canonical_dataset_config_bytes(second)
    assert len(dataset_config_sha256(first)) == 64


def test_dataset_config_rejects_unknowns_seed_overlap_and_reserved_seed() -> None:
    raw = _raw_config()
    raw["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        DevelopmentDatasetConfig.model_validate(raw)

    raw = _raw_config()
    quality = raw["quality"]
    assert isinstance(quality, dict)
    quality["fail_on_structured_duplicate"] = quality.pop(
        "fail_on_task_scoped_model_input_duplicate"
    )
    with pytest.raises(ValidationError, match="Extra inputs"):
        DevelopmentDatasetConfig.model_validate(raw)

    raw = _raw_config()
    raw_splits = raw["splits"]
    assert isinstance(raw_splits, dict)
    validation = raw_splits["iid_validation"]
    assert isinstance(validation, dict)
    validation["seeds"] = [1000]
    with pytest.raises(ValidationError, match="appears in both"):
        DevelopmentDatasetConfig.model_validate(raw)

    raw = _raw_config()
    raw_splits = raw["splits"]
    assert isinstance(raw_splits, dict)
    training = raw_splits["iid_train"]
    assert isinstance(training, dict)
    training["seeds"] = [99]
    with pytest.raises(ValidationError, match="golden-reserved"):
        DevelopmentDatasetConfig.model_validate(raw)


@pytest.mark.parametrize(
    "field_name",
    ["renderer_version", "projection_version", "manifest_version", "schema_snapshot_version"],
)
def test_dataset_config_rejects_unimplemented_contract_versions(field_name: str) -> None:
    raw = _raw_config()
    dataset = raw["dataset"]
    assert isinstance(dataset, dict)
    dataset[field_name] = "0.2.0"
    with pytest.raises(ValidationError):
        DevelopmentDatasetConfig.model_validate(raw)


@pytest.mark.parametrize("bad_value", [True, 3.0, "3"])
def test_dataset_config_rejects_non_integer_seed_values(bad_value: object) -> None:
    raw = _raw_config()
    raw_splits = raw["splits"]
    assert isinstance(raw_splits, dict)
    training = raw_splits["iid_train"]
    assert isinstance(training, dict)
    training["seeds"] = [bad_value]
    with pytest.raises(ValidationError):
        DevelopmentDatasetConfig.model_validate(raw)
