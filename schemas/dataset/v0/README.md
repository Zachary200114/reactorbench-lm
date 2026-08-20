# Dataset contract development snapshot

This directory contains the developmental `0.1.0` JSON Schema snapshot for the
public Phase 3 dataset boundary. It covers the single-input projection record,
the paired counterfactual projection record, and the split-first manifest. These
are audit and interchange contracts; only each projection's nested `ModelInput`
may cross into the narrative renderer.

`snapshot-contract.json` fixes the exact model-to-filename mapping and explicitly
records that the interface is developmental rather than frozen. `manifest.json`
binds that descriptor and every schema document with SHA-256 checksums. The
`reactorbench.dataset.schema_export` helpers produce canonical JSON bytes and
validate the descriptor, root documents, paths, and checksums when loading.

The snapshot must be regenerated and reviewed after any public dataset contract
change. A developmental snapshot is not a promise of backward compatibility.
