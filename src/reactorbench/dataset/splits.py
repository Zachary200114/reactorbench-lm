"""Split-first manifest contracts and leakage gates for Phase 3."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Literal

from pydantic import field_validator, model_validator

from reactorbench.dataset.contracts import (
    DATASET_CONTRACT_VERSION,
    ProjectionRecord,
    ProjectionView,
)
from reactorbench.dataset.grouping import (
    CounterfactualFamily,
    CounterfactualVariant,
    GroupAssignment,
)
from reactorbench.schemas.base import (
    SCHEMA_VERSION,
    ContractId,
    ContractModel,
    SchemaVersion,
    SeedInt,
    canonical_enum_tuple,
    canonical_sha256,
    canonical_string_tuple,
    require_unique,
)
from reactorbench.schemas.enums import (
    FaultFamily,
    PlantVariant,
    ScenarioDriver,
    SplitName,
    TaskName,
)
from reactorbench.schemas.trajectory import StructuredTrajectory

MANIFEST_VERSION: Literal["0.1.0"] = "0.1.0"
DEFAULT_GOLDEN_RESERVED_SEED_MAX = 99
_SPLIT_ORDER = {split: index for index, split in enumerate(SplitName)}


class SplitManifestError(ValueError):
    """Raised when a manifest violates an atomicity or holdout contract."""


class SplitRenderPlan(ContractModel):
    """Allowed renderer and alias families declared before any text is produced."""

    split_name: SplitName
    template_family_ids: tuple[ContractId, ...]
    alias_family_ids: tuple[ContractId, ...]

    @field_validator("template_family_ids", "alias_family_ids", mode="after")
    @classmethod
    def families_are_nonempty_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("renderer and alias family plans must be nonempty")
        return canonical_string_tuple(values, field_name="render plan families")


class TemplateAliasPlan(ContractModel):
    """Closed split-to-style plan frozen before rendering."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    dataset_contract_version: Literal["0.1.0"] = DATASET_CONTRACT_VERSION
    splits: tuple[SplitRenderPlan, ...]
    checksum_sha256: str

    @model_validator(mode="after")
    def split_plans_and_holdouts_are_exact(self) -> TemplateAliasPlan:
        if not self.splits:
            raise ValueError("template/alias plan cannot be empty")
        split_names = tuple(plan.split_name for plan in self.splits)
        require_unique(split_names, field_name="render-plan split names")
        if split_names != tuple(sorted(split_names, key=_SPLIT_ORDER.__getitem__)):
            raise ValueError("render plans must use canonical split order")
        by_split = {plan.split_name: plan for plan in self.splits}
        train = by_split.get(SplitName.IID_TRAIN)
        template = by_split.get(SplitName.TEMPLATE_TEST)
        if (
            train is not None
            and template is not None
            and set(train.template_family_ids).intersection(template.template_family_ids)
        ):
            raise ValueError("template_test template families must be absent from training")
        component = by_split.get(SplitName.COMPONENT_TEST)
        if (
            train is not None
            and component is not None
            and set(train.alias_family_ids).intersection(component.alias_family_ids)
        ):
            raise ValueError("component_test alias families must be absent from training")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("template/alias plan checksum does not match content")
        return self


class SplitManifestEntry(ContractModel):
    """Audit mapping from one projection to its pre-render split and style plan."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    manifest_version: Literal["0.1.0"] = MANIFEST_VERSION
    projection_id: ContractId
    projection_checksum_sha256: str
    trajectory_id: ContractId
    scenario_id: ContractId
    source_trajectory_sha256: str
    structured_fingerprint_sha256: str
    seed: SeedInt
    plant_variant_id: PlantVariant
    driver: ScenarioDriver
    fault_family_ids: tuple[FaultFamily, ...]
    task_name: TaskName
    projection_view: ProjectionView
    cut_tick: int
    source_event_index_exclusive: int | None = None
    split_name: SplitName
    template_family_ids: tuple[ContractId, ...]
    alias_family_ids: tuple[ContractId, ...]
    corruption_plan: ContractId = "none"
    counterfactual_group_id: ContractId | None = None
    counterfactual_family: CounterfactualFamily | None = None
    counterfactual_variant_id: CounterfactualVariant | None = None
    counterfactual_expected_variants: tuple[CounterfactualVariant, ...] = ()
    expanded_siblings_supported: bool | None = None
    golden_candidate: bool = False
    entry_checksum_sha256: str

    @field_validator("fault_family_ids", mode="after")
    @classmethod
    def faults_are_canonical(cls, values: tuple[FaultFamily, ...]) -> tuple[FaultFamily, ...]:
        return canonical_enum_tuple(values, enum_type=FaultFamily, field_name="fault_family_ids")

    @field_validator("template_family_ids", "alias_family_ids", mode="after")
    @classmethod
    def style_families_are_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("each manifest entry needs a pre-render style assignment")
        return canonical_string_tuple(values, field_name="entry style families")

    @model_validator(mode="after")
    def grouping_and_hashes_are_consistent(self) -> SplitManifestEntry:
        for name in (
            "projection_checksum_sha256",
            "source_trajectory_sha256",
            "structured_fingerprint_sha256",
            "entry_checksum_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        grouped_values = (
            self.counterfactual_group_id,
            self.counterfactual_family,
            self.counterfactual_variant_id,
            self.expanded_siblings_supported,
        )
        if any(value is not None for value in grouped_values) != all(
            value is not None for value in grouped_values
        ):
            raise ValueError("counterfactual metadata must be wholly present or absent")
        if self.counterfactual_group_id is None and self.counterfactual_expected_variants:
            raise ValueError("ungrouped entries cannot declare expected variants")
        if self.counterfactual_group_id is not None and not self.counterfactual_expected_variants:
            raise ValueError("grouped entries must declare expected variants")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"entry_checksum_sha256"})
        )
        if self.entry_checksum_sha256 != expected:
            raise ValueError("manifest entry checksum does not match content")
        return self


class SplitManifest(ContractModel):
    """Deterministic, split-first projection manifest with strict global gates."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    manifest_version: Literal["0.1.0"] = MANIFEST_VERSION
    golden_reserved_seed_max: SeedInt = DEFAULT_GOLDEN_RESERVED_SEED_MAX
    template_alias_plan: TemplateAliasPlan
    entries: tuple[SplitManifestEntry, ...]
    checksum_sha256: str

    @model_validator(mode="after")
    def global_split_gates_hold(self) -> SplitManifest:
        if not self.entries:
            raise ValueError("split manifest cannot be empty")
        expected_order = tuple(sorted(self.entries, key=_entry_order_key))
        if self.entries != expected_order:
            raise ValueError("manifest entries must use deterministic canonical order")
        projection_ids = tuple(entry.projection_id for entry in self.entries)
        require_unique(projection_ids, field_name="manifest projection IDs")
        task_fingerprints = tuple(
            (entry.structured_fingerprint_sha256, entry.task_name) for entry in self.entries
        )
        require_unique(task_fingerprints, field_name="task-scoped structured fingerprints")
        if any(
            entry.golden_candidate or entry.seed <= self.golden_reserved_seed_max
            for entry in self.entries
        ):
            raise ValueError("golden candidates and reserved seeds cannot enter a manifest")

        plans = {plan.split_name: plan for plan in self.template_alias_plan.splits}
        used_splits = {entry.split_name for entry in self.entries}
        if not used_splits.issubset(plans):
            raise ValueError("every used split must have a pre-render template/alias plan")
        for entry in self.entries:
            plan = plans[entry.split_name]
            if not set(entry.template_family_ids).issubset(plan.template_family_ids):
                raise ValueError("entry template family is outside its declared split plan")
            if not set(entry.alias_family_ids).issubset(plan.alias_family_ids):
                raise ValueError("entry alias family is outside its declared split plan")

        _require_atomic_key(self.entries, "seed", lambda entry: str(entry.seed))
        _require_atomic_key(
            self.entries,
            "scenario_id",
            lambda entry: entry.scenario_id,
        )
        _require_atomic_key(
            self.entries,
            "structured_fingerprint",
            lambda entry: entry.structured_fingerprint_sha256,
        )
        _require_atomic_key(
            tuple(entry for entry in self.entries if entry.counterfactual_group_id is not None),
            "counterfactual_group_id",
            lambda entry: entry.counterfactual_group_id or "",
        )
        _validate_counterfactual_groups(self.entries)
        _validate_composition_holdouts(self.entries)

        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("split manifest checksum does not match content")
        return self


def _entry_order_key(entry: SplitManifestEntry) -> tuple[object, ...]:
    return (
        _SPLIT_ORDER[entry.split_name],
        entry.seed,
        entry.scenario_id,
        entry.cut_tick,
        entry.task_name.value,
        entry.projection_id,
    )


def _require_atomic_key(
    entries: Iterable[SplitManifestEntry],
    key_name: str,
    key: Callable[[SplitManifestEntry], str],
) -> None:
    # ``key`` is intentionally kept callable at runtime while avoiding a public
    # unvalidated callback in any serialized contract.
    owners: dict[str, SplitName] = {}
    for entry in entries:
        value = key(entry)
        previous = owners.setdefault(value, entry.split_name)
        if previous is not entry.split_name:
            raise SplitManifestError(f"{key_name} crosses split boundaries")


def _validate_counterfactual_groups(entries: tuple[SplitManifestEntry, ...]) -> None:
    groups: dict[str, list[SplitManifestEntry]] = defaultdict(list)
    for entry in entries:
        if entry.counterfactual_group_id is not None:
            groups[entry.counterfactual_group_id].append(entry)
    for group_id, members in groups.items():
        families = {entry.counterfactual_family for entry in members}
        expected_sets = {entry.counterfactual_expected_variants for entry in members}
        support = {entry.expanded_siblings_supported for entry in members}
        if len(families) != 1 or len(expected_sets) != 1 or len(support) != 1:
            raise SplitManifestError(f"group {group_id} has inconsistent family metadata")
        family = next(iter(families))
        if family is None:
            raise SplitManifestError("group family cannot be null")
        observed_by_scenario = {
            (entry.scenario_id, entry.counterfactual_variant_id) for entry in members
        }
        variants_by_scenario: dict[str, CounterfactualVariant | None] = {}
        for scenario_id, variant in observed_by_scenario:
            previous = variants_by_scenario.setdefault(scenario_id, variant)
            if previous is not variant:
                raise SplitManifestError("one scenario cannot claim multiple group roles")
        observed = {variant for variant in variants_by_scenario.values() if variant is not None}
        expected = set(next(iter(expected_sets)))
        if family is CounterfactualFamily.G15_EVIDENCE_SUFFICIENCY:
            if next(iter(support)) is not False:
                raise SplitManifestError("G15 must remain generator-incomplete")
            if any(entry.split_name is SplitName.COUNTERFACTUAL_TEST for entry in members):
                raise SplitManifestError("incomplete G15 cannot enter counterfactual_test")
            continue
        if next(iter(support)) is not True or observed != expected:
            raise SplitManifestError(f"group {group_id} is missing required variant roles")
        required_split = (
            SplitName.COMPOSITION_TEST
            if family is CounterfactualFamily.G14_COMPOSITION
            else SplitName.COUNTERFACTUAL_TEST
        )
        if any(entry.split_name is not required_split for entry in members):
            raise SplitManifestError(
                f"{family.value} must be assigned wholly to {required_split.value}"
            )


def _validate_composition_holdouts(entries: tuple[SplitManifestEntry, ...]) -> None:
    training = tuple(entry for entry in entries if entry.split_name is SplitName.IID_TRAIN)
    training_fault_pairs = {
        entry.fault_family_ids for entry in training if len(entry.fault_family_ids) > 1
    }
    training_driver_faults = {
        (entry.driver, entry.fault_family_ids) for entry in training if entry.fault_family_ids
    }
    for entry in entries:
        if entry.split_name is not SplitName.COMPOSITION_TEST:
            continue
        if len(entry.fault_family_ids) > 1 and entry.fault_family_ids in training_fault_pairs:
            raise SplitManifestError("held-out fault composition appears in training")
        if (
            entry.driver is not ScenarioDriver.STEADY_OPERATION
            and entry.fault_family_ids
            and (entry.driver, entry.fault_family_ids) in training_driver_faults
        ):
            raise SplitManifestError("held-out driver-plus-fault composition appears in training")


def make_template_alias_plan(
    split_styles: Mapping[SplitName, tuple[tuple[str, ...], tuple[str, ...]]],
) -> TemplateAliasPlan:
    """Build the canonical checksummed style plan from explicit split mappings."""

    plans = tuple(
        SplitRenderPlan(
            split_name=split,
            template_family_ids=templates,
            alias_family_ids=aliases,
        )
        for split, (templates, aliases) in sorted(
            split_styles.items(), key=lambda item: _SPLIT_ORDER[item[0]]
        )
    )
    draft = TemplateAliasPlan.model_construct(
        schema_version=SCHEMA_VERSION,
        dataset_contract_version=DATASET_CONTRACT_VERSION,
        splits=plans,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return TemplateAliasPlan(
        schema_version=SCHEMA_VERSION,
        dataset_contract_version=DATASET_CONTRACT_VERSION,
        splits=plans,
        checksum_sha256=checksum,
    )


def make_split_manifest_entry(
    record: ProjectionRecord,
    trajectory: StructuredTrajectory,
    *,
    split_name: SplitName,
    template_family_ids: tuple[str, ...],
    alias_family_ids: tuple[str, ...],
    group_assignment: GroupAssignment | None = None,
    corruption_plan: str = "none",
    golden_candidate: bool = False,
) -> SplitManifestEntry:
    """Build and cross-check one projection-to-split mapping."""

    if type(record) is not ProjectionRecord:
        raise TypeError("record must be a ProjectionRecord")
    if type(trajectory) is not StructuredTrajectory:
        raise TypeError("trajectory must be a StructuredTrajectory")
    if record.lineage.trajectory_id != trajectory.trajectory_id:
        raise SplitManifestError("projection and trajectory IDs do not match")
    if record.lineage.scenario_id != trajectory.scenario_id:
        raise SplitManifestError("projection and scenario IDs do not match")
    if record.lineage.seed != trajectory.scenario.seed:
        raise SplitManifestError("projection and trajectory seeds do not match")
    if record.lineage.source_trajectory_sha256 != canonical_sha256(
        trajectory.model_dump(mode="json", round_trip=True)
    ):
        raise SplitManifestError("projection source hash does not match trajectory")
    if group_assignment is not None and group_assignment.scenario_id != trajectory.scenario_id:
        raise SplitManifestError("group assignment must reference the source scenario")

    scenario = trajectory.scenario
    fault_family_ids = tuple(injection.fault_family for injection in scenario.fault_injections)
    group_id = group_assignment.counterfactual_group_id if group_assignment else None
    group_family = group_assignment.family if group_assignment else None
    group_variant = group_assignment.variant_id if group_assignment else None
    expected_variants = group_assignment.expected_variants if group_assignment else ()
    siblings_supported = group_assignment.expanded_siblings_supported if group_assignment else None
    draft = SplitManifestEntry.model_construct(
        schema_version=SCHEMA_VERSION,
        manifest_version=MANIFEST_VERSION,
        projection_id=record.projection_id,
        projection_checksum_sha256=record.checksum_sha256,
        trajectory_id=trajectory.trajectory_id,
        scenario_id=trajectory.scenario_id,
        source_trajectory_sha256=record.lineage.source_trajectory_sha256,
        structured_fingerprint_sha256=record.lineage.structured_fingerprint_sha256,
        seed=scenario.seed,
        plant_variant_id=scenario.plant_variant_id,
        driver=scenario.driver,
        fault_family_ids=fault_family_ids,
        task_name=record.task_target.task_name,
        projection_view=record.projection_view,
        cut_tick=record.model_input.cut_tick,
        source_event_index_exclusive=record.model_input.source_event_index_exclusive,
        split_name=split_name,
        template_family_ids=template_family_ids,
        alias_family_ids=alias_family_ids,
        corruption_plan=corruption_plan,
        counterfactual_group_id=group_id,
        counterfactual_family=group_family,
        counterfactual_variant_id=group_variant,
        counterfactual_expected_variants=expected_variants,
        expanded_siblings_supported=siblings_supported,
        golden_candidate=golden_candidate,
        entry_checksum_sha256="0" * 64,
    )
    checksum_payload = draft.model_dump(
        mode="json",
        round_trip=True,
        exclude={"entry_checksum_sha256"},
    )
    final_payload = draft.model_dump(mode="python", round_trip=True)
    final_payload["entry_checksum_sha256"] = canonical_sha256(checksum_payload)
    return SplitManifestEntry.model_validate(final_payload)


def build_split_manifest(
    entries: Iterable[SplitManifestEntry],
    *,
    template_alias_plan: TemplateAliasPlan,
    golden_reserved_seed_max: int = DEFAULT_GOLDEN_RESERVED_SEED_MAX,
) -> SplitManifest:
    """Canonicalize, checksum, and validate a complete split-first manifest."""

    ordered = tuple(sorted(entries, key=_entry_order_key))
    draft = SplitManifest.model_construct(
        schema_version=SCHEMA_VERSION,
        manifest_version=MANIFEST_VERSION,
        golden_reserved_seed_max=golden_reserved_seed_max,
        template_alias_plan=template_alias_plan,
        entries=ordered,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return SplitManifest(
        schema_version=SCHEMA_VERSION,
        manifest_version=MANIFEST_VERSION,
        golden_reserved_seed_max=golden_reserved_seed_max,
        template_alias_plan=template_alias_plan,
        entries=ordered,
        checksum_sha256=checksum,
    )
