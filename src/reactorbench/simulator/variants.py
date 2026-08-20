"""Immutable, synthetic Aster variant registry for Phase 2 scenarios.

The registry is deliberately data-only: it contains fictional identifiers and
bounded normalized values, but no transition or inference logic.  Keeping this
topology separate from :mod:`reactorbench.simulator.core` lets later scenario
families select a variant without accepting arbitrary caller-provided maps.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from reactorbench.schemas import AsterSubsystem, PlantVariant, StateVariable

GENERATOR_VERSION = "0.1.0"
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9-]{2,79}$")
_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
_PROHIBITED_TERMS = frozenset(
    {
        "agency",
        "docket",
        "government",
        "military",
        "naval",
        "nuclear",
        "procedure",
        "security",
        "setpoint",
    }
)


class ComponentRole(StrEnum):
    """Closed semantic roles available in every synthetic Aster variant."""

    PRIMARY_LOOP_DOMAIN = "PRIMARY_LOOP_DOMAIN"
    PRIMARY_TRAIN_ONE = "PRIMARY_TRAIN_ONE"
    PRIMARY_TRAIN_TWO = "PRIMARY_TRAIN_TWO"
    PRIMARY_FLOW_VALVE = "PRIMARY_FLOW_VALVE"
    TRANSFER_UNIT = "TRANSFER_UNIT"
    SECONDARY_FEED = "SECONDARY_FEED"
    SUPPORT_BUS_ONE = "SUPPORT_BUS_ONE"
    SUPPORT_BUS_TWO = "SUPPORT_BUS_TWO"
    INSTRUMENTATION = "INSTRUMENTATION"


class ChannelRole(StrEnum):
    """The two non-interchangeable observation channels per state variable."""

    PRIMARY = "PRIMARY"
    REDUNDANT = "REDUNDANT"


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """One fictional component and its stable semantic role."""

    component_id: str
    role: ComponentRole
    subsystem: AsterSubsystem
    alias: str | None = None


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """One model-visible observation channel in a variant's fixed vocabulary."""

    channel_id: str
    variable: StateVariable
    role: ChannelRole
    component_id: str
    sensor_alias: str | None = None


@dataclass(frozen=True, slots=True)
class DependencyLinkSpec:
    """A declared support dependency from a supplier to one dependent."""

    supplier_component_id: str
    dependent_component_id: str


@dataclass(frozen=True, slots=True)
class AsterVariantSpec:
    """Strict, immutable topology and channel card for one Aster variant."""

    version: str
    plant_variant: PlantVariant
    components: tuple[ComponentSpec, ...]
    channels: tuple[ChannelSpec, ...]
    dependency_links: tuple[DependencyLinkSpec, ...]
    baseline_noise_bound: float
    max_per_tick_step: float
    standby_start_delay_ticks: int
    _components_by_role: Mapping[ComponentRole, ComponentSpec] = field(init=False, repr=False)
    _channels_by_key: Mapping[tuple[StateVariable, ChannelRole], ChannelSpec] = field(
        init=False, repr=False
    )
    _alias_targets: Mapping[str, str] = field(init=False, repr=False)
    _dependents_by_supplier: Mapping[str, tuple[str, ...]] = field(init=False, repr=False)
    _supplier_by_dependent: Mapping[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_variant_shape(self)

        components_by_role = {component.role: component for component in self.components}
        channels_by_key = {(channel.variable, channel.role): channel for channel in self.channels}
        alias_targets = {
            alias: target
            for alias, target in (
                *((component.alias, component.component_id) for component in self.components),
                *((channel.sensor_alias, channel.channel_id) for channel in self.channels),
            )
            if alias is not None
        }
        dependents_by_supplier: dict[str, list[str]] = defaultdict(list)
        supplier_by_dependent: dict[str, str] = {}
        for link in self.dependency_links:
            dependents_by_supplier[link.supplier_component_id].append(link.dependent_component_id)
            supplier_by_dependent[link.dependent_component_id] = link.supplier_component_id

        object.__setattr__(self, "_components_by_role", MappingProxyType(components_by_role))
        object.__setattr__(self, "_channels_by_key", MappingProxyType(channels_by_key))
        object.__setattr__(self, "_alias_targets", MappingProxyType(alias_targets))
        object.__setattr__(
            self,
            "_dependents_by_supplier",
            MappingProxyType(
                {
                    supplier_id: tuple(dependent_ids)
                    for supplier_id, dependent_ids in dependents_by_supplier.items()
                }
            ),
        )
        object.__setattr__(self, "_supplier_by_dependent", MappingProxyType(supplier_by_dependent))

    @property
    def aliases(self) -> tuple[str, ...]:
        """Stable component aliases, retaining Aster-A's existing public order."""

        return tuple(
            component.alias for component in self.components if component.alias is not None
        )

    @property
    def all_aliases(self) -> Mapping[str, str]:
        """Read-only mapping of every component or sensor alias to its identifier."""

        return self._alias_targets

    @property
    def primary_train_ids(self) -> tuple[str, str]:
        return (
            self.component_for_role(ComponentRole.PRIMARY_TRAIN_ONE).component_id,
            self.component_for_role(ComponentRole.PRIMARY_TRAIN_TWO).component_id,
        )

    @property
    def primary_flow_valve_ids(self) -> tuple[str, ...]:
        return (self.component_for_role(ComponentRole.PRIMARY_FLOW_VALVE).component_id,)

    @property
    def support_bus_ids(self) -> tuple[str, str]:
        return (
            self.component_for_role(ComponentRole.SUPPORT_BUS_ONE).component_id,
            self.component_for_role(ComponentRole.SUPPORT_BUS_TWO).component_id,
        )

    @property
    def primary_train_support_bus_pairs(self) -> tuple[tuple[str, str], tuple[str, str]]:
        return tuple((train_id, self.support_for(train_id)) for train_id in self.primary_train_ids)  # type: ignore[return-value]

    @property
    def instrumentation_id(self) -> str:
        return self.component_for_role(ComponentRole.INSTRUMENTATION).component_id

    @property
    def primary_loop_domain_id(self) -> str:
        return self.component_for_role(ComponentRole.PRIMARY_LOOP_DOMAIN).component_id

    @property
    def transfer_unit_id(self) -> str:
        return self.component_for_role(ComponentRole.TRANSFER_UNIT).component_id

    @property
    def secondary_feed_id(self) -> str:
        return self.component_for_role(ComponentRole.SECONDARY_FEED).component_id

    def component_for_role(self, role: ComponentRole) -> ComponentSpec:
        """Resolve an exact component role; raw strings are intentionally rejected."""

        if type(role) is not ComponentRole:
            raise TypeError("role must be a ComponentRole")
        return self._components_by_role[role]

    def channel_for(self, variable: StateVariable, role: ChannelRole) -> ChannelSpec:
        """Resolve an exact variable/channel-role pair; raw strings are rejected."""

        if type(variable) is not StateVariable:
            raise TypeError("variable must be a StateVariable")
        if type(role) is not ChannelRole:
            raise TypeError("role must be a ChannelRole")
        return self._channels_by_key[(variable, role)]

    def channels_for(self, variable: StateVariable) -> tuple[ChannelSpec, ChannelSpec]:
        """Return channels in fixed PRIMARY then REDUNDANT order."""

        return (
            self.channel_for(variable, ChannelRole.PRIMARY),
            self.channel_for(variable, ChannelRole.REDUNDANT),
        )

    def support_for(self, component_id: str) -> str:
        """Resolve a declared supplier for an exact component identifier."""

        if type(component_id) is not str:
            raise TypeError("component_id must be a string")
        return self._supplier_by_dependent[component_id]

    def dependents_for(self, component_id: str) -> tuple[str, ...]:
        """Return the fixed dependent set for an exact supplier identifier."""

        if type(component_id) is not str:
            raise TypeError("component_id must be a string")
        return self._dependents_by_supplier.get(component_id, ())


def _require_identifier(value: str, *, field_name: str) -> None:
    if type(value) is not str or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase fictional identifier")
    lowered = value.lower()
    if "://" in lowered or any(term in lowered for term in _PROHIBITED_TERMS):
        raise ValueError(f"{field_name} contains prohibited content")


def _validate_variant_shape(spec: AsterVariantSpec) -> None:
    if type(spec.version) is not str or _VERSION_PATTERN.fullmatch(spec.version) is None:
        raise ValueError("version must be a semantic-version string")
    if type(spec.plant_variant) is not PlantVariant:
        raise TypeError("plant_variant must be a PlantVariant")
    if not 0 < spec.baseline_noise_bound <= 0.02:
        raise ValueError("baseline_noise_bound must be normalized to (0, 0.02]")
    if not 0 < spec.max_per_tick_step <= 0.05:
        raise ValueError("max_per_tick_step must be normalized to (0, 0.05]")
    if (
        type(spec.standby_start_delay_ticks) is not int
        or not 1 <= spec.standby_start_delay_ticks <= 6
    ):
        raise ValueError("standby_start_delay_ticks must be an integer in [1, 6]")

    component_ids: set[str] = set()
    component_roles: set[ComponentRole] = set()
    aliases: set[str] = set()
    for component in spec.components:
        if type(component) is not ComponentSpec:
            raise TypeError("components must contain ComponentSpec instances")
        _require_identifier(component.component_id, field_name="component_id")
        if (
            type(component.role) is not ComponentRole
            or type(component.subsystem) is not AsterSubsystem
        ):
            raise TypeError("component roles and subsystems must use closed enums")
        if component.component_id in component_ids or component.role in component_roles:
            raise ValueError("component IDs and roles must be unique")
        component_ids.add(component.component_id)
        component_roles.add(component.role)
        if component.alias is not None:
            _require_identifier(component.alias, field_name="component alias")
            if component.alias in aliases:
                raise ValueError("component and sensor aliases must be unique")
            aliases.add(component.alias)
    if component_roles != set(ComponentRole):
        raise ValueError("every ComponentRole must resolve exactly once")

    channel_ids: set[str] = set()
    channel_keys: set[tuple[StateVariable, ChannelRole]] = set()
    channel_roles_by_variable: dict[StateVariable, set[ChannelRole]] = defaultdict(set)
    for channel in spec.channels:
        if type(channel) is not ChannelSpec:
            raise TypeError("channels must contain ChannelSpec instances")
        _require_identifier(channel.channel_id, field_name="channel_id")
        if type(channel.variable) is not StateVariable or type(channel.role) is not ChannelRole:
            raise TypeError("channels must use closed variable and role enums")
        if channel.channel_id in channel_ids or channel.channel_id in component_ids:
            raise ValueError("component and channel IDs must be globally unique")
        key = (channel.variable, channel.role)
        if key in channel_keys or channel.component_id not in component_ids:
            raise ValueError("channel keys must be unique and reference a component")
        channel_ids.add(channel.channel_id)
        channel_keys.add(key)
        channel_roles_by_variable[channel.variable].add(channel.role)
        if channel.sensor_alias is not None:
            _require_identifier(channel.sensor_alias, field_name="sensor alias")
            if channel.sensor_alias in aliases:
                raise ValueError("component and sensor aliases must be unique")
            aliases.add(channel.sensor_alias)
    if set(channel_roles_by_variable) != set(StateVariable) or any(
        roles != set(ChannelRole) for roles in channel_roles_by_variable.values()
    ):
        raise ValueError("every StateVariable requires PRIMARY and REDUNDANT channels")

    supplied_dependents: set[str] = set()
    support_bus_ids = {
        component.component_id
        for component in spec.components
        if component.role in {ComponentRole.SUPPORT_BUS_ONE, ComponentRole.SUPPORT_BUS_TWO}
    }
    supported_dependent_roles = {
        ComponentRole.PRIMARY_TRAIN_ONE,
        ComponentRole.PRIMARY_TRAIN_TWO,
        ComponentRole.PRIMARY_FLOW_VALVE,
        ComponentRole.TRANSFER_UNIT,
        ComponentRole.SECONDARY_FEED,
    }
    dependent_ids = {
        component.component_id
        for component in spec.components
        if component.role in supported_dependent_roles
    }
    for link in spec.dependency_links:
        if type(link) is not DependencyLinkSpec:
            raise TypeError("dependency_links must contain DependencyLinkSpec instances")
        if (
            link.supplier_component_id not in support_bus_ids
            or link.dependent_component_id not in dependent_ids
            or link.dependent_component_id in supplied_dependents
        ):
            raise ValueError("every supported component must have one valid support-bus supplier")
        supplied_dependents.add(link.dependent_component_id)
    if supplied_dependents != dependent_ids:
        raise ValueError("every supported component requires exactly one support-bus supplier")


def _channels(
    *,
    prefix: str,
    primary_component_id: str,
    instrumentation_id: str,
    sensor_alias_prefix: str | None,
) -> tuple[ChannelSpec, ...]:
    """Create the fixed two-channel vocabulary without any random process."""

    channels: list[ChannelSpec] = []
    for variable in StateVariable:
        token = variable.value.replace("_", "-")
        component_id = (
            primary_component_id if variable is StateVariable.PRIMARY_FLOW else instrumentation_id
        )
        for role, suffix in ((ChannelRole.PRIMARY, "a"), (ChannelRole.REDUNDANT, "b")):
            sensor_alias = (
                None if sensor_alias_prefix is None else f"{sensor_alias_prefix}-{token}-{suffix}"
            )
            channels.append(
                ChannelSpec(
                    channel_id=f"{prefix}-{token}-{suffix}",
                    variable=variable,
                    role=role,
                    component_id=component_id,
                    sensor_alias=sensor_alias,
                )
            )
    return tuple(channels)


def _spec(
    *,
    plant_variant: PlantVariant,
    components: tuple[ComponentSpec, ...],
    prefix: str,
    noise_bound: float,
    max_tick_step: float,
    standby_delay: int,
    sensor_alias_prefix: str | None,
    dependency_links: tuple[DependencyLinkSpec, ...],
) -> AsterVariantSpec:
    roles = {component.role: component.component_id for component in components}
    return AsterVariantSpec(
        version=GENERATOR_VERSION,
        plant_variant=plant_variant,
        components=components,
        channels=_channels(
            prefix=prefix,
            primary_component_id=roles[ComponentRole.PRIMARY_TRAIN_ONE],
            instrumentation_id=roles[ComponentRole.INSTRUMENTATION],
            sensor_alias_prefix=sensor_alias_prefix,
        ),
        dependency_links=dependency_links,
        baseline_noise_bound=noise_bound,
        max_per_tick_step=max_tick_step,
        standby_start_delay_ticks=standby_delay,
    )


_A_COMPONENTS = (
    ComponentSpec(
        "aster-domain-orchid", ComponentRole.PRIMARY_LOOP_DOMAIN, AsterSubsystem.PRIMARY_LOOP
    ),
    ComponentSpec(
        "aster-train-cirrus", ComponentRole.PRIMARY_TRAIN_ONE, AsterSubsystem.PRIMARY_LOOP, "cirrus"
    ),
    ComponentSpec(
        "aster-train-kestrel",
        ComponentRole.PRIMARY_TRAIN_TWO,
        AsterSubsystem.PRIMARY_LOOP,
        "kestrel",
    ),
    ComponentSpec(
        "aster-valve-lark", ComponentRole.PRIMARY_FLOW_VALVE, AsterSubsystem.PRIMARY_LOOP, "lark"
    ),
    ComponentSpec("aster-transfer-wren", ComponentRole.TRANSFER_UNIT, AsterSubsystem.TRANSFER_UNIT),
    ComponentSpec(
        "aster-feed-brindle", ComponentRole.SECONDARY_FEED, AsterSubsystem.SECONDARY_LOOP
    ),
    ComponentSpec(
        "aster-bus-rill", ComponentRole.SUPPORT_BUS_ONE, AsterSubsystem.SUPPORT_POWER, "rill"
    ),
    ComponentSpec(
        "aster-bus-quill", ComponentRole.SUPPORT_BUS_TWO, AsterSubsystem.SUPPORT_POWER, "quill"
    ),
    ComponentSpec(
        "aster-instrument-vireo",
        ComponentRole.INSTRUMENTATION,
        AsterSubsystem.INSTRUMENTATION,
        "vireo",
    ),
)
ASTER_A_SPEC = _spec(
    plant_variant=PlantVariant.ASTER_A,
    components=_A_COMPONENTS,
    prefix="aster",
    noise_bound=0.006,
    max_tick_step=0.03,
    standby_delay=1,
    sensor_alias_prefix=None,
    dependency_links=(
        DependencyLinkSpec("aster-bus-rill", "aster-train-cirrus"),
        DependencyLinkSpec("aster-bus-quill", "aster-train-kestrel"),
        DependencyLinkSpec("aster-bus-rill", "aster-valve-lark"),
        DependencyLinkSpec("aster-bus-quill", "aster-transfer-wren"),
        DependencyLinkSpec("aster-bus-rill", "aster-feed-brindle"),
    ),
)

_B_COMPONENTS = (
    ComponentSpec(
        "aster-b-domain-mosaic", ComponentRole.PRIMARY_LOOP_DOMAIN, AsterSubsystem.PRIMARY_LOOP
    ),
    ComponentSpec(
        "aster-b-train-nomad", ComponentRole.PRIMARY_TRAIN_ONE, AsterSubsystem.PRIMARY_LOOP, "nomad"
    ),
    ComponentSpec(
        "aster-b-train-saffron",
        ComponentRole.PRIMARY_TRAIN_TWO,
        AsterSubsystem.PRIMARY_LOOP,
        "saffron",
    ),
    ComponentSpec(
        "aster-b-valve-umbra",
        ComponentRole.PRIMARY_FLOW_VALVE,
        AsterSubsystem.PRIMARY_LOOP,
        "umbra",
    ),
    ComponentSpec(
        "aster-b-transfer-gale", ComponentRole.TRANSFER_UNIT, AsterSubsystem.TRANSFER_UNIT, "gale"
    ),
    ComponentSpec(
        "aster-b-feed-cascade",
        ComponentRole.SECONDARY_FEED,
        AsterSubsystem.SECONDARY_LOOP,
        "cascade",
    ),
    ComponentSpec(
        "aster-b-bus-lyric", ComponentRole.SUPPORT_BUS_ONE, AsterSubsystem.SUPPORT_POWER, "lyric"
    ),
    ComponentSpec(
        "aster-b-bus-sonnet", ComponentRole.SUPPORT_BUS_TWO, AsterSubsystem.SUPPORT_POWER, "sonnet"
    ),
    ComponentSpec(
        "aster-b-instrument-oriol",
        ComponentRole.INSTRUMENTATION,
        AsterSubsystem.INSTRUMENTATION,
        "oriol",
    ),
)
ASTER_B_SPEC = _spec(
    plant_variant=PlantVariant.ASTER_B,
    components=_B_COMPONENTS,
    prefix="boreal",
    noise_bound=0.008,
    max_tick_step=0.028,
    standby_delay=2,
    sensor_alias_prefix="boreal-sensor",
    dependency_links=(
        DependencyLinkSpec("aster-b-bus-sonnet", "aster-b-train-nomad"),
        DependencyLinkSpec("aster-b-bus-lyric", "aster-b-train-saffron"),
        DependencyLinkSpec("aster-b-bus-sonnet", "aster-b-valve-umbra"),
        DependencyLinkSpec("aster-b-bus-lyric", "aster-b-transfer-gale"),
        DependencyLinkSpec("aster-b-bus-sonnet", "aster-b-feed-cascade"),
    ),
)

_C_COMPONENTS = (
    ComponentSpec(
        "aster-c-domain-velvet", ComponentRole.PRIMARY_LOOP_DOMAIN, AsterSubsystem.PRIMARY_LOOP
    ),
    ComponentSpec(
        "aster-c-train-peregrine",
        ComponentRole.PRIMARY_TRAIN_ONE,
        AsterSubsystem.PRIMARY_LOOP,
        "peregrine",
    ),
    ComponentSpec(
        "aster-c-train-fable", ComponentRole.PRIMARY_TRAIN_TWO, AsterSubsystem.PRIMARY_LOOP, "fable"
    ),
    ComponentSpec(
        "aster-c-valve-helix",
        ComponentRole.PRIMARY_FLOW_VALVE,
        AsterSubsystem.PRIMARY_LOOP,
        "helix",
    ),
    ComponentSpec(
        "aster-c-transfer-twilight",
        ComponentRole.TRANSFER_UNIT,
        AsterSubsystem.TRANSFER_UNIT,
        "twilight",
    ),
    ComponentSpec(
        "aster-c-feed-kinetic",
        ComponentRole.SECONDARY_FEED,
        AsterSubsystem.SECONDARY_LOOP,
        "kinetic",
    ),
    ComponentSpec(
        "aster-c-bus-cobalt", ComponentRole.SUPPORT_BUS_ONE, AsterSubsystem.SUPPORT_POWER, "cobalt"
    ),
    ComponentSpec(
        "aster-c-bus-lantern",
        ComponentRole.SUPPORT_BUS_TWO,
        AsterSubsystem.SUPPORT_POWER,
        "lantern",
    ),
    ComponentSpec(
        "aster-c-instrument-marrow",
        ComponentRole.INSTRUMENTATION,
        AsterSubsystem.INSTRUMENTATION,
        "marrow",
    ),
)
ASTER_C_SPEC = _spec(
    plant_variant=PlantVariant.ASTER_C,
    components=_C_COMPONENTS,
    prefix="cinder",
    noise_bound=0.004,
    max_tick_step=0.025,
    standby_delay=3,
    sensor_alias_prefix="cinder-sensor",
    dependency_links=(
        DependencyLinkSpec("aster-c-bus-cobalt", "aster-c-train-peregrine"),
        DependencyLinkSpec("aster-c-bus-lantern", "aster-c-train-fable"),
        DependencyLinkSpec("aster-c-bus-lantern", "aster-c-valve-helix"),
        DependencyLinkSpec("aster-c-bus-lantern", "aster-c-transfer-twilight"),
        DependencyLinkSpec("aster-c-bus-cobalt", "aster-c-feed-kinetic"),
    ),
)

VARIANT_REGISTRY: Mapping[PlantVariant, AsterVariantSpec] = MappingProxyType(
    {
        PlantVariant.ASTER_A: ASTER_A_SPEC,
        PlantVariant.ASTER_B: ASTER_B_SPEC,
        PlantVariant.ASTER_C: ASTER_C_SPEC,
    }
)


def _validate_registry() -> None:
    if set(VARIANT_REGISTRY) != set(PlantVariant):
        raise ValueError("registry must define exactly Aster-A, Aster-B, and Aster-C")

    all_ids: set[str] = set()
    all_aliases: set[str] = set()
    topology_signatures: set[tuple[tuple[ComponentRole, ComponentRole], ...]] = set()
    numeric_signatures: set[tuple[float, float, int]] = set()
    for plant_variant, spec in VARIANT_REGISTRY.items():
        if plant_variant is not spec.plant_variant:
            raise ValueError("registry key must match each variant card")
        identifiers = {
            *(component.component_id for component in spec.components),
            *(channel.channel_id for channel in spec.channels),
        }
        if identifiers & all_ids:
            raise ValueError("component and channel identifiers must be unique across variants")
        all_ids.update(identifiers)
        aliases = set(spec.all_aliases)
        if aliases & all_aliases:
            raise ValueError("aliases must be unique across variants")
        all_aliases.update(aliases)

        role_by_id = {component.component_id: component.role for component in spec.components}
        topology_signatures.add(
            tuple(
                sorted(
                    (
                        (
                            role_by_id[link.dependent_component_id],
                            role_by_id[link.supplier_component_id],
                        )
                        for link in spec.dependency_links
                    ),
                    key=lambda pair: (pair[0].value, pair[1].value),
                )
            )
        )
        numeric_signatures.add(
            (spec.baseline_noise_bound, spec.max_per_tick_step, spec.standby_start_delay_ticks)
        )
    if len(numeric_signatures) != len(VARIANT_REGISTRY):
        raise ValueError("variant numeric bounds and standby delays must differ")
    if len(topology_signatures) < 2:
        raise ValueError("at least one variant dependency role map must differ")


_validate_registry()


def get_variant_spec(plant_variant: PlantVariant) -> AsterVariantSpec:
    """Return one registered variant card, rejecting raw strings and unknown values."""

    if type(plant_variant) is not PlantVariant:
        raise TypeError("plant_variant must be a PlantVariant")
    return VARIANT_REGISTRY[plant_variant]


__all__ = [
    "ASTER_A_SPEC",
    "ASTER_B_SPEC",
    "ASTER_C_SPEC",
    "GENERATOR_VERSION",
    "VARIANT_REGISTRY",
    "AsterVariantSpec",
    "ChannelRole",
    "ChannelSpec",
    "ComponentRole",
    "ComponentSpec",
    "DependencyLinkSpec",
    "get_variant_spec",
]
