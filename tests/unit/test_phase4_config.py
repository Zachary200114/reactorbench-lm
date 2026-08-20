"""Phase 4 configuration boundary tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from reactorbench.model import (
    Phase4Config,
    TransformerConfig,
    load_phase4_config,
    resolve_project_path,
)

CONFIG_PATH = Path("configs/model/phase4-smoke-v0.1.0.toml")


def test_phase4_config_loads_exact_reviewed_tiers() -> None:
    config = load_phase4_config(CONFIG_PATH)

    assert config.tokenizer.vocab_size == 2048
    assert (config.smoke_model.layers, config.pilot_model.layers, config.main_model.layers) == (
        2,
        6,
        8,
    )
    assert config.smoke_training.steps == 300
    assert config.phase4.run_name == "phase4-smoke-v0.1.0"


def test_phase4_config_rejects_unknown_fields_and_scalar_coercion() -> None:
    payload = load_phase4_config(CONFIG_PATH).model_dump(mode="python", round_trip=True)
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Phase4Config.model_validate(payload)

    payload = load_phase4_config(CONFIG_PATH).model_dump(mode="python", round_trip=True)
    smoke_training = cast(dict[str, object], payload["smoke_training"])
    smoke_training["steps"] = "300"
    with pytest.raises(ValidationError, match="int_type"):
        Phase4Config.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("heads", 3), ("context_length", 4), ("dropout", float("nan"))],
)
def test_transformer_config_rejects_invalid_dimensions(field: str, value: object) -> None:
    payload = load_phase4_config(CONFIG_PATH).smoke_model.model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValidationError):
        TransformerConfig.model_validate(payload)


def test_model_tiers_must_increase_monotonically() -> None:
    config = load_phase4_config(CONFIG_PATH)
    payload = config.model_dump(mode="python", round_trip=True)
    payload["pilot_model"] = config.smoke_model.model_dump(mode="python")

    with pytest.raises(ValidationError, match="increase monotonically"):
        Phase4Config.model_validate(payload)


@pytest.mark.parametrize("relative", ["../escape", "/outside", "runs/../escape"])
def test_project_path_resolution_rejects_escape(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError, match="path"):
        resolve_project_path(tmp_path, relative, must_exist=False)


def test_project_path_resolution_rejects_symlink_component(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "redirect").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        resolve_project_path(tmp_path, "redirect/run", must_exist=False)
