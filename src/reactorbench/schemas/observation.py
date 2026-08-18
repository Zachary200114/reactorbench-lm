"""Model-visible observation contracts, separated from latent truth."""

from __future__ import annotations

from pydantic import field_validator, model_validator

from .base import (
    SCHEMA_VERSION,
    ContractId,
    ContractModel,
    NonNegativeInt,
    NormalizedFloat,
    SchemaVersion,
    canonical_string_tuple,
)
from .enums import ChannelQuality, ObservationStatus, StateVariable


class SensorChannelObservation(ContractModel):
    channel_id: ContractId
    variable: StateVariable
    value: NormalizedFloat | None
    quality: ChannelQuality
    status: ObservationStatus

    @model_validator(mode="after")
    def missingness_is_consistent(self) -> SensorChannelObservation:
        unavailable = self.quality is ChannelQuality.UNAVAILABLE
        missing = self.status is ObservationStatus.MISSING
        if unavailable != missing:
            raise ValueError(
                "UNAVAILABLE channel quality and MISSING observation status must occur together"
            )
        if missing != (self.value is None):
            raise ValueError("missing observations must have a null value, and vice versa")
        return self


class ObservationFrame(ContractModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    tick: NonNegativeInt
    overall_status: ObservationStatus
    channels: tuple[SensorChannelObservation, ...]

    @field_validator("channels", mode="after")
    @classmethod
    def channels_are_canonical(
        cls, values: tuple[SensorChannelObservation, ...]
    ) -> tuple[SensorChannelObservation, ...]:
        channel_ids = canonical_string_tuple(
            tuple(item.channel_id for item in values), field_name="channel_ids"
        )
        by_id = {item.channel_id: item for item in values}
        return tuple(by_id[channel_id] for channel_id in channel_ids)
