"""Tests for deterministic validation-only confidence calibration."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from reactorbench.remediation.calibration import (
    CALIBRATION_GRID,
    CalibrationObservation,
    TemperatureCalibrationReport,
    apply_temperature,
    calibration_inventory_sha256,
    fit_temperature,
)


def _observations() -> tuple[CalibrationObservation, ...]:
    return tuple(
        CalibrationObservation(
            example_id=f"calibration-{index:03d}",
            raw_confidence=confidence,
            exact_match=correct,
        )
        for index, (confidence, correct) in enumerate(
            (
                (0.98, True),
                (0.95, False),
                (0.80, True),
                (0.72, False),
                (0.55, True),
            )
            * 11
            + ((0.51, False),)
        )
    )


def test_temperature_fit_is_deterministic_bounded_and_checksum_bound() -> None:
    observations = _observations()
    first = fit_temperature(
        observations,
        calibration_selection_manifest_sha256="c" * 64,
        calibration_prediction_manifest_sha256="d" * 64,
        calibration_predictions_sha256="e" * 64,
        selected_checkpoint_manifest_sha256="a" * 64,
    )
    replay = fit_temperature(
        tuple(reversed(observations)),
        calibration_selection_manifest_sha256="c" * 64,
        calibration_prediction_manifest_sha256="d" * 64,
        calibration_predictions_sha256="e" * 64,
        selected_checkpoint_manifest_sha256="a" * 64,
    )

    assert replay == first
    assert first.selected_temperature in CALIBRATION_GRID
    assert 0.5 <= first.selected_temperature <= 5.0
    assert math.isfinite(first.selected_nll)
    assert first.calibration_inventory_sha256 == calibration_inventory_sha256(observations)

    payload = first.model_dump(mode="json", round_trip=True)
    payload["selected_nll"] += 0.01
    with pytest.raises(ValidationError, match="checksum"):
        TemperatureCalibrationReport.model_validate(payload)


def test_flat_grid_tie_selects_temperature_one() -> None:
    observations = tuple(
        CalibrationObservation(
            example_id=f"flat-{index}", raw_confidence=0.5, exact_match=bool(index % 2)
        )
        for index in range(56)
    )
    report = fit_temperature(
        observations,
        calibration_selection_manifest_sha256="c" * 64,
        calibration_prediction_manifest_sha256="d" * 64,
        calibration_predictions_sha256="e" * 64,
        selected_checkpoint_manifest_sha256="b" * 64,
    )
    assert report.selected_temperature == 1.0


def test_temperature_transform_is_confidence_only_monotonic_and_finite() -> None:
    raw = (0.0, 0.1, 0.5, 0.9, 1.0)
    calibrated = tuple(apply_temperature(value, 2.0) for value in raw)
    assert calibrated == tuple(sorted(calibrated))
    assert all(0.0 < value < 1.0 and math.isfinite(value) for value in calibrated)
    assert apply_temperature(0.5, 2.0) == 0.5
    with pytest.raises(ValueError, match="grid"):
        apply_temperature(0.8, 1.01)
    with pytest.raises(ValueError, match="finite probability"):
        apply_temperature(float("nan"), 1.0)


def test_calibration_inventory_rejects_duplicates() -> None:
    observations = tuple(
        CalibrationObservation(
            example_id="duplicate" if index == 55 else f"item-{index}",
            raw_confidence=0.8,
            exact_match=True,
        )
        for index in range(56)
    )
    observations = (*observations[:-1], observations[0])
    with pytest.raises(ValueError, match="unique"):
        calibration_inventory_sha256(observations)


def test_incorrect_endpoint_confidence_has_finite_nll() -> None:
    observations = tuple(
        CalibrationObservation(
            example_id=f"endpoint-{index:03d}",
            raw_confidence=1.0 if index == 0 else 0.5,
            exact_match=index != 0,
        )
        for index in range(56)
    )
    report = fit_temperature(
        observations,
        calibration_selection_manifest_sha256="c" * 64,
        calibration_prediction_manifest_sha256="d" * 64,
        calibration_predictions_sha256="e" * 64,
        selected_checkpoint_manifest_sha256="a" * 64,
    )
    assert math.isfinite(report.selected_nll)
