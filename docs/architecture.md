# ReactorBench-LM architecture

Status: Phase 1 foundation locally integrated and verified at the 2026-08-18 checkpoint; later phases are not complete

## Design objective

ReactorBench-LM separates a wholly fictional source of truth from its observable narratives and from every model trained on those narratives. This makes causal labels auditable and prevents model output from becoming ground truth.

```text
project-authored scenario definition
                 |
                 v
        latent Aster state machine
                 |
                 v
        bounded observation model
                 |
                 v
        canonical event extraction
                 |
                 v
        deterministic text renderers
                 |
                 v
 split-first manifests and dataset views
                 |
                 v
 tokenizer -> baselines -> Transformer
                 |
                 v
 versioned evaluation and result artifacts
                 |
                 v
 narrow inference service -> Research Editorial UI
```

Only the left-to-right data exposed by a task contract reaches a model. Latent fault injection and withheld ground truth must never leak into visible observations or model inputs unless a specific supervised target explicitly requires them.

## Layers and contracts

| Layer | Responsibility | Must not do |
|---|---|---|
| Scenario definition | Select fictional plant variant, driver, faults, onset, severity, seed, and ordered `{decision_tick, action}` entries | Contain real facility data or depend on rendered prose |
| Latent state | Apply bounded deterministic transitions and injected faults to the invented state graph | Copy real physics, setpoints, procedures, or layouts |
| Observation | Convert latent variables into bounded channels with explicit quality and status | Alter latent process state or reveal hidden fault labels |
| Canonical event | Record exact fictional events, evidence slots, and versioned identifiers | Infer truth from natural language |
| Renderer | Produce traceable project-authored narratives from canonical events | Use a hosted LLM or introduce unlabeled facts |
| Dataset | Materialize structured, narrative, and task views from frozen split manifests | Randomly split rendered text or edit generated records by hand |
| Tokenizer and models | Learn only from permitted training artifacts | Load pretrained core-model weights or test data |
| Evaluation | Compare fixed predictions with structured truth on separately reported splits | Select checkpoints using test results or hide failures |
| Inference service | Offer a narrow, bounded operation over curated fictional scenarios | Accept files, URLs, checkpoints, paths, or real logs |
| Web application | Present model evidence, ground truth, versions, failures, and limitations | Become an unrestricted chat box or realistic control room |

## Phase 1 package boundary

The initial package uses `src/reactorbench/` and is deliberately small:

```text
src/reactorbench/
  config.py
  schemas/
    base.py
    enums.py
    latent.py
    observation.py
    events.py
    scenario.py
    target.py
    provenance.py
    export.py
```

The semantic separation is required even if package organization evolves later. The schema version begins at development version `0.1.0`.

The shared model base is strict, rejects unknown fields, validates defaults, and
makes each validated model instance immutable. Normalized continuous fields require
finite values in `[0, 1]`; identifiers and nonnegative integers have explicit bounds.
The development interface is separately marked `frozen=false`, so reviewed contract
changes remain allowed before version 1. Instance immutability must not be confused
with interface freeze. Canonical JSON and a SHA-256 manifest provide deterministic
snapshots for schema review. These are contract choices, not claims that the
generator or dataset already exists.

## Artifact lineage

Every serious artifact must form a verifiable chain:

```text
source revision
  -> resolved generator configuration and schema version
  -> trajectory and split manifests
  -> rendered dataset manifest
  -> tokenizer configuration and checksum
  -> model/training configuration
  -> checkpoint and checksum
  -> evaluation configuration and predictions
  -> figures, report, and local UI release identifier
```

Generated corpora, checkpoints, and run directories must not silently overwrite earlier artifacts. Estimates and interface mock values never enter the results chain.

## Future local service boundary

After model and evaluation gates pass, the local product may add:

```text
untrusted browser
      |
      v
server-side gateway: request schema, body/method limits, rate policy, safe errors
      |
      v
inference service: fixed tokenizer/checkpoint, bounded work, structured response
      |
      v
checksummed read-only release artifacts
```

The browser never selects paths, models, token limits, execution devices, or arbitrary content. The inference service revalidates security-critical fields and returns allowlisted structured data. Model output remains untrusted display text.

## Current implementation boundary

The Phase 1 research specifications, package, configuration boundary, developmental
contracts, local schema snapshots, and focused tests are present. The recorded
2026-08-18 Python 3.12 checkpoint passed `make check` and `make build`; a separate
coverage run measured 91.41%. Phase 2 is the structured generator. Dataset
generation, tokenizer training, model training, measured evaluation, inference
serving, and UI construction remain later work. Refer to [the implementation
status](IMPLEMENTATION_STATUS.md) for the exact evolving integration state rather
than inferring completion from this architecture.

The current task prepares everything locally but excludes GitHub push and Vercel or inference-host deployment.
