"""Shared primitives for the developmental Aster Station contracts.

The contracts in this package deliberately prefer immutable tuples and nested
frozen models over open-ended dictionaries.  That keeps validation strict and
makes serialized records deterministic enough to hash and compare.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

SCHEMA_VERSION = "0.1.0"

SchemaVersion = Annotated[
    StrictStr,
    Field(pattern=r"^0\.1\.0$", description="Developmental Aster schema version."),
]
ContractId = Annotated[
    StrictStr,
    Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
]
SemanticVersion = Annotated[
    StrictStr,
    Field(pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"),
]
NormalizedFloat = Annotated[
    StrictFloat,
    Field(ge=0.0, le=1.0, allow_inf_nan=False),
]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
SeedInt = Annotated[StrictInt, Field(ge=0, le=4_294_967_295)]


class ContractModel(BaseModel):
    """Base class shared by every externally serialized contract."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


def require_unique[T](values: tuple[T, ...], *, field_name: str) -> tuple[T, ...]:
    """Reject duplicates without requiring values to be naturally orderable."""

    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


def canonical_enum_tuple[T: Enum](
    values: tuple[T, ...],
    *,
    enum_type: type[T],
    field_name: str,
) -> tuple[T, ...]:
    """Treat an enum tuple as a set and serialize it in declaration order."""

    require_unique(values, field_name=field_name)
    order = {member: index for index, member in enumerate(enum_type)}
    return tuple(sorted(values, key=lambda value: order[value]))


def canonical_string_tuple(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    """Treat a string tuple as a set and serialize it lexicographically."""

    require_unique(values, field_name=field_name)
    return tuple(sorted(values))


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical JSON representation used for snapshots."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash a value after canonical JSON serialization."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
