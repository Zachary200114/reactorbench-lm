from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from reactorbench.schemas import PlantVariant, StateVariable
from reactorbench.simulator import variants as variant_module
from reactorbench.simulator.content_guard import assert_no_prohibited_content
from reactorbench.simulator.variants import (
    ASTER_A_SPEC,
    ASTER_B_SPEC,
    ASTER_C_SPEC,
    VARIANT_REGISTRY,
    AsterVariantSpec,
    ChannelRole,
    ComponentRole,
    get_variant_spec,
)

_G12_DEPENDENT_ROLES = {
    ComponentRole.PRIMARY_TRAIN_ONE,
    ComponentRole.PRIMARY_TRAIN_TWO,
    ComponentRole.PRIMARY_FLOW_VALVE,
    ComponentRole.TRANSFER_UNIT,
    ComponentRole.SECONDARY_FEED,
}


def _role_dependency_image(spec: AsterVariantSpec) -> dict[ComponentRole, ComponentRole]:
    role_by_id = {component.component_id: component.role for component in spec.components}
    return {
        role_by_id[link.dependent_component_id]: role_by_id[link.supplier_component_id]
        for link in spec.dependency_links
    }


def test_registry_is_exact_and_lookup_rejects_raw_strings() -> None:
    assert set(VARIANT_REGISTRY) == set(PlantVariant)
    assert get_variant_spec(PlantVariant.ASTER_A) is ASTER_A_SPEC
    assert get_variant_spec(PlantVariant.ASTER_B) is ASTER_B_SPEC
    assert get_variant_spec(PlantVariant.ASTER_C) is ASTER_C_SPEC

    with pytest.raises(TypeError, match="PlantVariant"):
        get_variant_spec("ASTER-A")  # type: ignore[arg-type]


def test_aster_a_preserves_existing_phase_two_identifiers_and_bounds() -> None:
    assert ASTER_A_SPEC.primary_train_ids == ("aster-train-cirrus", "aster-train-kestrel")
    assert ASTER_A_SPEC.primary_flow_valve_ids == ("aster-valve-lark",)
    assert ASTER_A_SPEC.support_bus_ids == ("aster-bus-rill", "aster-bus-quill")
    assert ASTER_A_SPEC.instrumentation_id == "aster-instrument-vireo"
    assert ASTER_A_SPEC.aliases == ("cirrus", "kestrel", "lark", "rill", "quill", "vireo")
    assert ASTER_A_SPEC.baseline_noise_bound == 0.006
    assert ASTER_A_SPEC.max_per_tick_step == 0.03
    assert ASTER_A_SPEC.standby_start_delay_ticks == 1
    assert ASTER_A_SPEC.primary_train_support_bus_pairs == (
        ("aster-train-cirrus", "aster-bus-rill"),
        ("aster-train-kestrel", "aster-bus-quill"),
    )
    for variable in StateVariable:
        assert ASTER_A_SPEC.channels_for(variable)[0].channel_id == (
            f"aster-{variable.value.replace('_', '-')}-a"
        )
        assert ASTER_A_SPEC.channels_for(variable)[1].channel_id == (
            f"aster-{variable.value.replace('_', '-')}-b"
        )


def test_every_semantic_role_and_channel_pair_resolves() -> None:
    for spec in VARIANT_REGISTRY.values():
        assert isinstance(spec, AsterVariantSpec)
        for role in ComponentRole:
            assert spec.component_for_role(role).role is role
        for variable in StateVariable:
            primary, redundant = spec.channels_for(variable)
            assert primary.variable is redundant.variable is variable
            assert primary.role is ChannelRole.PRIMARY
            assert redundant.role is ChannelRole.REDUNDANT
            assert primary.channel_id != redundant.channel_id

        for train_id, bus_id in spec.primary_train_support_bus_pairs:
            assert spec.support_for(train_id) == bus_id
            assert train_id in spec.dependents_for(bus_id)


def test_registry_identifiers_aliases_and_references_are_unique_and_valid() -> None:
    identifiers: set[str] = set()
    aliases: set[str] = set()
    for spec in VARIANT_REGISTRY.values():
        component_ids = {component.component_id for component in spec.components}
        channel_ids = {channel.channel_id for channel in spec.channels}
        assert not component_ids & channel_ids
        assert not identifiers & (component_ids | channel_ids)
        identifiers.update(component_ids | channel_ids)

        assert not aliases & set(spec.all_aliases)
        aliases.update(spec.all_aliases)
        assert all(channel.component_id in component_ids for channel in spec.channels)
        assert all(
            link.supplier_component_id in component_ids
            and link.dependent_component_id in component_ids
            for link in spec.dependency_links
        )
        role_by_id = {component.component_id: component.role for component in spec.components}
        assert {role_by_id[link.dependent_component_id] for link in spec.dependency_links} == (
            _G12_DEPENDENT_ROLES
        )
        assert all(
            role_by_id[link.supplier_component_id]
            in {ComponentRole.SUPPORT_BUS_ONE, ComponentRole.SUPPORT_BUS_TWO}
            for link in spec.dependency_links
        )
        assert len({link.dependent_component_id for link in spec.dependency_links}) == 5


def test_variant_cards_are_meaningfully_distinct() -> None:
    assert (
        ASTER_A_SPEC.primary_train_support_bus_pairs != ASTER_B_SPEC.primary_train_support_bus_pairs
    )
    assert ASTER_B_SPEC.standby_start_delay_ticks > ASTER_A_SPEC.standby_start_delay_ticks
    assert ASTER_C_SPEC.aliases != ASTER_A_SPEC.aliases
    assert ASTER_C_SPEC.baseline_noise_bound != ASTER_A_SPEC.baseline_noise_bound
    c_sensor_aliases = tuple(
        channel.sensor_alias
        for channel in ASTER_C_SPEC.channels
        if channel.sensor_alias is not None
    )
    assert c_sensor_aliases
    assert len(c_sensor_aliases) == len(set(c_sensor_aliases))
    assert all(alias.startswith("cinder-sensor-") for alias in c_sensor_aliases)

    role_images = {
        _variant: _role_dependency_image(spec) for _variant, spec in VARIANT_REGISTRY.items()
    }
    assert len({tuple(sorted(image.items())) for image in role_images.values()}) == 3
    assert role_images[PlantVariant.ASTER_B] == {
        ComponentRole.PRIMARY_TRAIN_ONE: ComponentRole.SUPPORT_BUS_TWO,
        ComponentRole.PRIMARY_TRAIN_TWO: ComponentRole.SUPPORT_BUS_ONE,
        ComponentRole.PRIMARY_FLOW_VALVE: ComponentRole.SUPPORT_BUS_TWO,
        ComponentRole.TRANSFER_UNIT: ComponentRole.SUPPORT_BUS_ONE,
        ComponentRole.SECONDARY_FEED: ComponentRole.SUPPORT_BUS_TWO,
    }


def test_specs_are_frozen_and_internal_mappings_are_read_only() -> None:
    with pytest.raises(FrozenInstanceError):
        ASTER_A_SPEC.baseline_noise_bound = 0.01  # type: ignore[misc]
    with pytest.raises(TypeError):
        ASTER_A_SPEC.all_aliases["other"] = "aster-train-cirrus"  # type: ignore[index]
    with pytest.raises(TypeError):
        VARIANT_REGISTRY[PlantVariant.ASTER_A] = ASTER_B_SPEC  # type: ignore[index]


def _component_alias_identifier_collision(spec: AsterVariantSpec) -> AsterVariantSpec:
    return replace(
        spec,
        components=(
            spec.components[0],
            replace(spec.components[1], alias=spec.components[0].component_id),
            *spec.components[2:],
        ),
    )


def _sensor_alias_identifier_collision(spec: AsterVariantSpec) -> AsterVariantSpec:
    return replace(
        spec,
        channels=(
            replace(spec.channels[0], sensor_alias=spec.channels[0].channel_id),
            *spec.channels[1:],
        ),
    )


@pytest.mark.parametrize(
    "mutator", [_component_alias_identifier_collision, _sensor_alias_identifier_collision]
)
def test_variant_constructor_rejects_alias_identifier_collisions(
    mutator: Callable[[AsterVariantSpec], AsterVariantSpec],
) -> None:
    with pytest.raises(ValueError, match="aliases must not collide with identifiers"):
        mutator(ASTER_A_SPEC)


def test_registry_rejects_alias_matching_identifier_in_another_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias_collision = replace(
        ASTER_B_SPEC,
        components=(
            ASTER_B_SPEC.components[0],
            replace(ASTER_B_SPEC.components[1], alias=ASTER_A_SPEC.components[0].component_id),
            *ASTER_B_SPEC.components[2:],
        ),
    )
    monkeypatch.setattr(
        variant_module,
        "VARIANT_REGISTRY",
        {
            PlantVariant.ASTER_A: ASTER_A_SPEC,
            PlantVariant.ASTER_B: alias_collision,
            PlantVariant.ASTER_C: ASTER_C_SPEC,
        },
    )
    with pytest.raises(ValueError, match="aliases and identifiers"):
        variant_module._validate_registry()


def test_registry_rejects_identifier_matching_alias_in_another_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier_collision = replace(
        ASTER_B_SPEC,
        channels=(
            replace(ASTER_B_SPEC.channels[0], channel_id=ASTER_A_SPEC.aliases[0]),
            *ASTER_B_SPEC.channels[1:],
        ),
    )
    monkeypatch.setattr(
        variant_module,
        "VARIANT_REGISTRY",
        {
            PlantVariant.ASTER_A: ASTER_A_SPEC,
            PlantVariant.ASTER_B: identifier_collision,
            PlantVariant.ASTER_C: ASTER_C_SPEC,
        },
    )
    with pytest.raises(ValueError, match="aliases and identifiers"):
        variant_module._validate_registry()


def test_variant_cards_pass_the_existing_prohibited_content_guard() -> None:
    for spec in VARIANT_REGISTRY.values():
        assert_no_prohibited_content(spec)


def test_registry_module_is_static_and_deterministic() -> None:
    module_source = Path(__file__).parents[2] / "src/reactorbench/simulator/variants.py"
    source = module_source.read_text(encoding="utf-8")
    assert "from random" not in source
    assert "Random(" not in source
    assert get_variant_spec(PlantVariant.ASTER_B) is get_variant_spec(PlantVariant.ASTER_B)
