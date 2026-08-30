# Aster schema development snapshot

I use this directory for the developmental `0.1.0` Aster contract set. It is not a
frozen version 1 interface. Validated model instances are immutable at runtime, but I
can still make reviewed changes to the developmental interface.

`snapshot-contract.json` records the root model-to-filename mapping. I use
`reactorbench.schemas.export.export_json_schemas` to emit the seven JSON Schema
documents deterministically, including `structured-trajectory.schema.json`, along
with a SHA-256 `manifest.json`. Contract tests check repeatable bytes,
self-consistent hashes, and the current local root mapping.

I keep the generated files under version control so schema changes are reviewable.
They are still developmental artifacts, not a stable released interface. After any
contract change, I regenerate and review them; `frozen` stays `false` until the schema
freeze gate is complete.
