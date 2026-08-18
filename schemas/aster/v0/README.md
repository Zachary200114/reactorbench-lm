# Aster schema development snapshot

This directory describes the developmental `0.1.0` contract set. It is not a
frozen version 1 interface. Validated model instances are immutable, but that
runtime property does not freeze the developmental interface against reviewed
changes.

`snapshot-contract.json` fixes the root model-to-filename mapping. The
`reactorbench.schemas.export.export_json_schemas` helper deterministically emits
the seven JSON Schema documents, including the aggregate
`structured-trajectory.schema.json`, plus a SHA-256 `manifest.json`. Contract tests
verify repeatable bytes, self-consistent hashes, and the present local root mapping.

Generated schema files are local review artifacts at this checkpoint, not committed
or released assets while local Git remains uninitialized. They should be
regenerated and reviewed after any contract change; `frozen` must remain `false`
until the pre-build schema review is complete.
