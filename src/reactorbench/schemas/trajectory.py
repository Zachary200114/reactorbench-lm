"""Aggregate audit-source contract for one complete structured trajectory."""

from __future__ import annotations

from pydantic import model_validator

from .base import SCHEMA_VERSION, ContractId, ContractModel, SchemaVersion
from .events import CanonicalEvent
from .latent import LatentPlantState
from .observation import ObservationFrame
from .provenance import ProvenanceRecord
from .scenario import ScenarioDefinition
from .target import ScenarioTargets


class StructuredTrajectory(ContractModel):
    """Join scenario truth, aligned frames, visible events, and bounded decisions.

    This is an audit-source record, not a model-input record. Hidden scenario and
    latent truth remain in their own nested contracts; model-visible observations
    and events cannot accept fault labels or fault-injection fields. Event indices
    are a contiguous zero-based order; multiple events may share one monotonic tick.
    """

    schema_version: SchemaVersion = SCHEMA_VERSION
    trajectory_id: ContractId
    scenario_id: ContractId
    scenario: ScenarioDefinition
    provenance: ProvenanceRecord
    latent_states: tuple[LatentPlantState, ...]
    observations: tuple[ObservationFrame, ...]
    events: tuple[CanonicalEvent, ...]
    targets: ScenarioTargets

    @model_validator(mode="after")
    def records_form_one_bounded_trajectory(self) -> StructuredTrajectory:
        if self.scenario_id != self.scenario.scenario_id:
            raise ValueError("scenario_id must match scenario.scenario_id")
        if self.targets.scenario_id != self.scenario_id:
            raise ValueError("targets must reference the trajectory scenario_id")
        if self.provenance.trajectory_id != self.trajectory_id:
            raise ValueError("provenance trajectory_id must match trajectory_id")
        if self.provenance.scenario_id != self.scenario_id:
            raise ValueError("provenance scenario_id must match scenario_id")
        if self.provenance.plant_variant_id is not self.scenario.plant_variant_id:
            raise ValueError("provenance plant_variant_id must match the scenario")
        if self.provenance.seed != self.scenario.seed:
            raise ValueError("provenance seed must match the scenario")
        injected_faults = {injection.fault_family for injection in self.scenario.fault_injections}
        if set(self.provenance.fault_family_ids) != injected_faults:
            raise ValueError("provenance fault families must match scenario fault injections")

        expected_ticks = tuple(range(self.scenario.duration_ticks))
        latent_ticks = tuple(state.tick for state in self.latent_states)
        observation_ticks = tuple(frame.tick for frame in self.observations)
        if latent_ticks != expected_ticks:
            raise ValueError("latent_states must contain one ordered state per scenario tick")
        if observation_ticks != expected_ticks:
            raise ValueError("observations must contain one ordered frame per scenario tick")
        if latent_ticks != observation_ticks:
            raise ValueError("latent-state and observation ticks must align")

        event_ids = tuple(event.event_id for event in self.events)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_id values must be unique within a trajectory")
        event_indexes = tuple(event.event_index for event in self.events)
        if event_indexes != tuple(range(len(self.events))):
            raise ValueError("event_index values must be contiguous, unique, and start at zero")
        event_times = tuple(event.sim_time for event in self.events)
        if event_times != tuple(sorted(event_times)):
            raise ValueError("event sim_time values must be monotonic")
        if any(event.sim_time >= self.scenario.duration_ticks for event in self.events):
            raise ValueError("event sim_time must be inside the scenario window")

        events_by_id = {event.event_id: event for event in self.events}
        for event in self.events:
            for related_id in event.related_event_ids:
                related = events_by_id.get(related_id)
                if related is None:
                    raise ValueError("related_event_ids must reference events in the trajectory")
                if related.event_index >= event.event_index:
                    raise ValueError("related_event_ids must reference an earlier event")

        for decision in self.targets.decisions:
            if decision.decision_tick >= self.scenario.duration_ticks:
                raise ValueError("decision tick must be inside the scenario window")
            for evidence_id in decision.evidence_event_ids:
                evidence_event = events_by_id.get(evidence_id)
                if evidence_event is None:
                    raise ValueError("evidence_event_ids must reference events in the trajectory")
                if evidence_event.sim_time > decision.decision_tick:
                    raise ValueError("decision evidence cannot reference a future event")
            referenced_slots = {
                slot
                for evidence_id in decision.evidence_event_ids
                for slot in events_by_id[evidence_id].evidence_slots
            }
            if not set(decision.evidence_slots).issubset(referenced_slots):
                raise ValueError("decision evidence slots must occur on referenced events")
        return self
