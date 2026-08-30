# Dataset contract development snapshot

I use this directory for the developmental `0.1.0` JSON Schema snapshot at the Phase 3
dataset boundary. It covers single-input projections, paired counterfactual
projections, and the split-first manifest. These are audit and interchange contracts;
only a projection's nested `ModelInput` may enter the narrative renderer.

`snapshot-contract.json` records the exact model-to-filename mapping and marks the
interface as developmental. `manifest.json` binds that descriptor and every schema
document with SHA-256 checksums. I use the
`reactorbench.dataset.schema_export` helpers to produce canonical JSON bytes and
validate the descriptor, root documents, paths, and checksums when loading.

After any public dataset-contract change, I regenerate and review the snapshot. While
it remains developmental, I do not promise backward compatibility.
