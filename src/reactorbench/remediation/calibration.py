"""Deterministic, validation-only confidence temperature calibration.

The scalar is deliberately fit after raw checkpoint selection.  It changes only
reported confidence used by calibration gates; it never participates in decoding,
token selection, semantic scoring, or the raw prediction artifact.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictFloat, StrictStr, model_validator

from reactorbench.schemas.base import ContractModel, canonical_sha256

CALIBRATION_GRID: tuple[float, ...] = tuple(round(0.5 + index * 0.05, 2) for index in range(91))
PROBABILITY_EPSILON = 1e-12
Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]


class CalibrationObservation(ContractModel):
    """One post-decoding correctness/confidence observation, identity-bound later."""

    example_id: StrictStr = Field(min_length=1, max_length=160)
    raw_confidence: StrictFloat = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    exact_match: StrictBool


class TemperatureCalibrationReport(ContractModel):
    report_version: Literal["0.3.1-targeted"] = "0.3.1-targeted"
    method: Literal["fixed_grid_binary_nll_exact_match"] = "fixed_grid_binary_nll_exact_match"
    grid_start: StrictFloat = 0.5
    grid_stop: StrictFloat = 5.0
    grid_step: StrictFloat = 0.05
    observation_count: Literal[56]
    calibration_selection_manifest_sha256: Sha256
    calibration_prediction_manifest_sha256: Sha256
    calibration_predictions_sha256: Sha256
    calibration_inventory_sha256: Sha256
    selected_checkpoint_manifest_sha256: Sha256
    selected_temperature: StrictFloat = Field(ge=0.5, le=5.0, allow_inf_nan=False)
    selected_nll: StrictFloat = Field(ge=0.0, allow_inf_nan=False)
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def grid_and_checksum_match(self) -> TemperatureCalibrationReport:
        if (self.grid_start, self.grid_stop, self.grid_step) != (0.5, 5.0, 0.05):
            raise ValueError("calibration report grid differs from preregistration")
        if self.selected_temperature not in CALIBRATION_GRID:
            raise ValueError("calibration temperature is outside the preregistered grid")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("calibration report checksum mismatch")
        return self


def _clamp_probability(value: float) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be a finite probability")
    return min(1.0 - PROBABILITY_EPSILON, max(PROBABILITY_EPSILON, value))


def apply_temperature(confidence: float, temperature: float) -> float:
    """Apply the monotonic logit temperature transform with safe probability bounds."""

    if type(temperature) is not float or temperature not in CALIBRATION_GRID:
        raise ValueError("temperature must be a preregistered finite grid value")
    probability = _clamp_probability(confidence)
    logit = math.log(probability / (1.0 - probability))
    return _clamp_probability(1.0 / (1.0 + math.exp(-logit / temperature)))


def calibration_inventory_sha256(observations: tuple[CalibrationObservation, ...]) -> str:
    """Hash canonical calibration observations in identity order."""

    _validated_observations(observations)
    return canonical_sha256(
        tuple(
            (item.example_id, item.raw_confidence, item.exact_match)
            for item in sorted(observations, key=lambda item: item.example_id)
        )
    )


def _validated_observations(
    observations: tuple[CalibrationObservation, ...],
) -> tuple[CalibrationObservation, ...]:
    if type(observations) is not tuple or len(observations) != 56:
        raise ValueError("calibration requires exactly 56 observations")
    if any(type(item) is not CalibrationObservation for item in observations):
        raise TypeError("calibration observations must use exact contracts")
    identifiers = tuple(item.example_id for item in observations)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("calibration observation IDs must be unique")
    return observations


def _binary_nll(observations: tuple[CalibrationObservation, ...], temperature: float) -> float:
    total = 0.0
    for observation in sorted(observations, key=lambda item: item.example_id):
        confidence = apply_temperature(observation.raw_confidence, temperature)
        probability = confidence if observation.exact_match else 1.0 - confidence
        total -= math.log(min(1.0 - PROBABILITY_EPSILON, max(PROBABILITY_EPSILON, probability)))
    return total / len(observations)


def fit_temperature(
    observations: tuple[CalibrationObservation, ...],
    *,
    calibration_selection_manifest_sha256: str,
    calibration_prediction_manifest_sha256: str,
    calibration_predictions_sha256: str,
    selected_checkpoint_manifest_sha256: str,
) -> TemperatureCalibrationReport:
    """Fit one scalar using the frozen grid and deterministic tie breaks."""

    _validated_observations(observations)
    for name, checksum in (
        ("calibration selection", calibration_selection_manifest_sha256),
        ("calibration prediction manifest", calibration_prediction_manifest_sha256),
        ("calibration predictions", calibration_predictions_sha256),
        ("selected checkpoint", selected_checkpoint_manifest_sha256),
    ):
        if (
            type(checksum) is not str
            or len(checksum) != 64
            or any(value not in "0123456789abcdef" for value in checksum)
        ):
            raise ValueError(f"{name} checksum is invalid")
    temperature, nll = min(
        ((value, _binary_nll(observations, value)) for value in CALIBRATION_GRID),
        key=lambda item: (item[1], abs(item[0] - 1.0), item[0]),
    )
    draft = TemperatureCalibrationReport.model_construct(
        observation_count=len(observations),
        calibration_selection_manifest_sha256=calibration_selection_manifest_sha256,
        calibration_prediction_manifest_sha256=calibration_prediction_manifest_sha256,
        calibration_predictions_sha256=calibration_predictions_sha256,
        calibration_inventory_sha256=calibration_inventory_sha256(observations),
        selected_checkpoint_manifest_sha256=selected_checkpoint_manifest_sha256,
        selected_temperature=temperature,
        selected_nll=nll,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return TemperatureCalibrationReport(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}), checksum_sha256=checksum
    )


__all__ = [
    "CALIBRATION_GRID",
    "CalibrationObservation",
    "TemperatureCalibrationReport",
    "apply_temperature",
    "calibration_inventory_sha256",
    "fit_temperature",
]
