"""Canonical, model-visible event contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import field_validator, model_validator

from .base import (
    SCHEMA_VERSION,
    ContractId,
    ContractModel,
    NonNegativeInt,
    NormalizedFloat,
    SchemaVersion,
    require_unique,
)
from .enums import (
    ActionLabel,
    ChannelQuality,
    ComponentState,
    EventType,
    EvidenceSlot,
    ObservationStatus,
    OperatingMode,
    StateVariable,
)


@dataclass(frozen=True)
class EventFieldContract:
    """Required and optional typed payload fields for one event category."""

    required: frozenset[str]
    optional: frozenset[str] = frozenset()

    @property
    def allowed(self) -> frozenset[str]:
        return self.required | self.optional


_PAYLOAD_FIELDS = frozenset(
    {
        "operating_mode_before",
        "operating_mode_after",
        "component_state_before",
        "component_state_after",
        "variable",
        "value_before",
        "value_after",
        "observation_status",
        "channel_quality_before",
        "channel_quality",
        "commanded_value",
        "observed_value",
        "action_label",
    }
)

# This is the authoritative runtime matrix. Evidence and related-event references
# are common metadata and therefore intentionally sit outside the typed payload.
EVENT_FIELD_MATRIX: Mapping[EventType, EventFieldContract] = MappingProxyType(
    {
        EventType.OPERATING_MODE_CHANGED: EventFieldContract(
            frozenset({"operating_mode_before", "operating_mode_after"})
        ),
        EventType.TARGET_CHANGED: EventFieldContract(
            frozenset({"variable", "value_before", "value_after"})
        ),
        EventType.COMPONENT_STATE_CHANGED: EventFieldContract(
            frozenset({"component_state_before", "component_state_after"})
        ),
        EventType.OBSERVATION_CHANGED: EventFieldContract(
            frozenset({"variable", "value_after", "observation_status"}),
            frozenset({"value_before"}),
        ),
        EventType.CHANNEL_QUALITY_CHANGED: EventFieldContract(
            frozenset({"channel_quality_before", "channel_quality"})
        ),
        EventType.CHANNEL_DISAGREEMENT: EventFieldContract(
            frozenset({"variable", "observation_status"})
        ),
        EventType.COMMAND_RECORDED: EventFieldContract(frozenset({"variable", "commanded_value"})),
        EventType.COMMAND_POSITION_MISMATCH: EventFieldContract(
            frozenset({"variable", "commanded_value", "observed_value"})
        ),
        EventType.COMMAND_POSITION_ALIGNED: EventFieldContract(
            frozenset({"variable", "commanded_value", "observed_value"})
        ),
        EventType.ACTION_APPLIED: EventFieldContract(frozenset({"action_label"})),
        EventType.BENIGN_NOTE: EventFieldContract(frozenset()),
    }
)


class CanonicalEvent(ContractModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    event_id: ContractId
    event_index: NonNegativeInt
    sim_time: NonNegativeInt
    event_type: EventType
    subject_id: ContractId
    operating_mode_before: OperatingMode | None = None
    operating_mode_after: OperatingMode | None = None
    component_state_before: ComponentState | None = None
    component_state_after: ComponentState | None = None
    variable: StateVariable | None = None
    value_before: NormalizedFloat | None = None
    value_after: NormalizedFloat | None = None
    observation_status: ObservationStatus | None = None
    channel_quality_before: ChannelQuality | None = None
    channel_quality: ChannelQuality | None = None
    commanded_value: NormalizedFloat | None = None
    observed_value: NormalizedFloat | None = None
    action_label: ActionLabel | None = None
    evidence_slots: tuple[EvidenceSlot, ...] = ()
    related_event_ids: tuple[ContractId, ...] = ()

    @field_validator("evidence_slots", "related_event_ids", mode="after")
    @classmethod
    def set_like_fields_are_unique(cls, values: tuple[object, ...]) -> tuple[object, ...]:
        return require_unique(values, field_name="canonical event set-like field")

    @model_validator(mode="after")
    def payload_matches_event_type(self) -> CanonicalEvent:
        contract = EVENT_FIELD_MATRIX[self.event_type]
        missing = contract.required - self.model_fields_set
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"{self.event_type.value} requires fields: {fields}")

        # A nullable value_after is meaningful only for a newly missing observation.
        null_required = {
            field_name for field_name in contract.required if getattr(self, field_name) is None
        }
        if self.event_type is EventType.OBSERVATION_CHANGED:
            null_required.discard("value_after")
        if null_required:
            fields = ", ".join(sorted(null_required))
            raise ValueError(f"{self.event_type.value} requires non-null fields: {fields}")

        populated = {
            field_name for field_name in _PAYLOAD_FIELDS if getattr(self, field_name) is not None
        }
        forbidden = populated - contract.allowed
        if forbidden:
            fields = ", ".join(sorted(forbidden))
            raise ValueError(f"{self.event_type.value} forbids fields: {fields}")

        if self.event_type is EventType.OPERATING_MODE_CHANGED:
            if self.operating_mode_before is self.operating_mode_after:
                raise ValueError("OPERATING_MODE_CHANGED requires distinct before and after modes")
        elif self.event_type is EventType.TARGET_CHANGED:
            if self.value_before == self.value_after:
                raise ValueError("TARGET_CHANGED requires distinct before and after values")
        elif self.event_type is EventType.COMPONENT_STATE_CHANGED:
            if self.component_state_before is self.component_state_after:
                raise ValueError(
                    "COMPONENT_STATE_CHANGED requires distinct before and after states"
                )
        elif self.event_type is EventType.OBSERVATION_CHANGED:
            missing_observation = self.observation_status is ObservationStatus.MISSING
            if missing_observation != (self.value_after is None):
                raise ValueError(
                    "MISSING observation status and null value_after must occur together"
                )
            if self.value_before is not None and self.value_before == self.value_after:
                raise ValueError("OBSERVATION_CHANGED requires a changed value when before is set")
        elif self.event_type is EventType.CHANNEL_QUALITY_CHANGED:
            if self.channel_quality_before is self.channel_quality:
                raise ValueError(
                    "CHANNEL_QUALITY_CHANGED requires distinct before and after qualities"
                )
        elif self.event_type is EventType.CHANNEL_DISAGREEMENT:
            if self.observation_status is not ObservationStatus.CONFLICTING:
                raise ValueError("CHANNEL_DISAGREEMENT requires CONFLICTING observation status")
        elif self.event_type is EventType.COMMAND_POSITION_MISMATCH:
            if self.commanded_value == self.observed_value:
                raise ValueError(
                    "COMMAND_POSITION_MISMATCH requires different commanded and observed values"
                )
        elif self.event_type is EventType.COMMAND_POSITION_ALIGNED:
            if self.commanded_value != self.observed_value:
                raise ValueError(
                    "COMMAND_POSITION_ALIGNED requires equal commanded and observed values"
                )
        return self
