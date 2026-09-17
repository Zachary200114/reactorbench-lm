# ReactorBench-LM research documentation

This directory contains the research contracts, specifications, and decisions behind
ReactorBench-LM. It explains what the project is allowed to claim, how Aster Station
and its dataset are constructed, how experiments are evaluated, and what must happen
before a model or public interface can be released.

> **Current state:** Phase 6 model-quality remediation is in progress. Targeted-06 is
> implemented and source-verified but has not been trained. Phase 7 remains blocked.
> See the [implementation status](../docs/IMPLEMENTATION_STATUS.md) for the exact live
> checkpoint.

## Research question

> How well can a small decoder-only Transformer, trained from random initialization on
> project-authored synthetic event language, learn causal sequence structure and
> generalize to unseen wording, component roles, and fault combinations?

ReactorBench-LM is a model-development and evaluation project. It is not a real plant
simulator, digital twin, operator assistant, safety system, emergency tool, or source
of engineering guidance.

## How to read this directory

The documents serve different purposes:

- **Requirements and specifications** define the boundaries the implementation must
  satisfy.
- **Preregistrations and acceptance plans** record decisions before results are seen.
- **Decision logs** explain why a version or policy changed.
- **Implementation reports** live under [`../docs/`](../docs/) and record what was
  actually built or measured.

Some research documents intentionally retain the status language from the phase in
which they were written. They are not silently rewritten after an experiment. For the
current run identity, measured results, test state, and immediate next step, use
[`../docs/IMPLEMENTATION_STATUS.md`](../docs/IMPLEMENTATION_STATUS.md).

## Authority and precedence

When two documents differ, use this order:

1. [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md)
2. [`DECISION_LOG.md`](DECISION_LOG.md)
3. [`PREBUILD_CHECKLIST.md`](PREBUILD_CHECKLIST.md)
4. [`FICTIONAL_PLANT_SPEC.md`](FICTIONAL_PLANT_SPEC.md)
5. [`DATASET_SPEC.md`](DATASET_SPEC.md)
6. [`EXPERIMENT_ACCEPTANCE_PLAN.md`](EXPERIMENT_ACCEPTANCE_PLAN.md)
7. [`GOLDEN_SCENARIOS.md`](GOLDEN_SCENARIOS.md)
8. [`SECURE_ENGINEERING_PLAN.md`](SECURE_ENGINEERING_PLAN.md)
9. [`REPRODUCIBILITY_RELEASE_PLAN.md`](REPRODUCIBILITY_RELEASE_PLAN.md)
10. [`UI_PRODUCT_REQUIREMENTS.md`](UI_PRODUCT_REQUIREMENTS.md)
11. [`LIVE_DEMO_PLAN.md`](LIVE_DEMO_PLAN.md)
12. [`RESEARCH_BLUEPRINT.md`](RESEARCH_BLUEPRINT.md)
13. Remaining research notes

Later versioned entries in the decision log may amend an older plan without deleting
the original evidence.

## Core research package

### Scope and causal world

| Document | Purpose |
| --- | --- |
| [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md) | Canonical completion criteria, capabilities, and non-goals |
| [`RESEARCH_BLUEPRINT.md`](RESEARCH_BLUEPRINT.md) | Original framing, architecture, experiment strategy, risks, and roadmap |
| [`FICTIONAL_PLANT_SPEC.md`](FICTIONAL_PLANT_SPEC.md) | Aster Station topology, state, observations, events, faults, and invariants |
| [`VOCABULARY_SEED.md`](VOCABULARY_SEED.md) | Safe terminology boundary for synthetic authoring |
| [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) | Project positioning and reviewed external context; not training-corpus approval |
| [`SOURCE_MANIFEST.csv`](SOURCE_MANIFEST.csv) | Reviewed sources, licenses, and ingestion decisions |

### Data and evaluation

| Document | Purpose |
| --- | --- |
| [`DATASET_SPEC.md`](DATASET_SPEC.md) | Data views, task contracts, grouping, splitting, rendering, and provenance |
| [`EXPERIMENT_ACCEPTANCE_PLAN.md`](EXPERIMENT_ACCEPTANCE_PLAN.md) | Hypotheses, baselines, metrics, acceptance gates, and error taxonomy |
| [`GOLDEN_SCENARIOS.md`](GOLDEN_SCENARIOS.md) | Approved withheld G01–G15 behavioral suite and access boundary |
| [`PHASE5_EXPERIMENT_PLAN.md`](PHASE5_EXPERIMENT_PLAN.md) | Preregistered baseline and pilot experiment |
| [`PREBUILD_CHECKLIST.md`](PREBUILD_CHECKLIST.md) | Phase gates and prerequisite status |
| [`DECISION_LOG.md`](DECISION_LOG.md) | Settled decisions and versioned amendments |

### Security, reproducibility, and product

| Document | Purpose |
| --- | --- |
| [`SECURE_ENGINEERING_PLAN.md`](SECURE_ENGINEERING_PLAN.md) | Trust boundaries, secure implementation controls, tests, and CI expectations |
| [`REPRODUCIBILITY_RELEASE_PLAN.md`](REPRODUCIBILITY_RELEASE_PLAN.md) | Artifact lineage, clean reproduction, release contents, and evidence levels |
| [`UI_PRODUCT_REQUIREMENTS.md`](UI_PRODUCT_REQUIREMENTS.md) | Approved Research Editorial interface direction |
| [`LIVE_DEMO_PLAN.md`](LIVE_DEMO_PLAN.md) | Public GitHub, inference, and deployment plan |

## Implemented research stack

- A deterministic fictional causal generator with separate latent state,
  observations, events, and prose rendering.
- An approved Phase 3 development candidate with grouped leakage controls and typed
  provenance.
- A 2,048-token SentencePiece BPE trained only on approved `iid_train` prose.
- A decoder-only Transformer implemented from PyTorch primitives and initialized from
  random weights.
- Majority, rule, n-gram, bag-of-words, GRU, and smaller-Transformer baselines.
- Smoke, pilot, main, remediation, calibration, robustness, and diagnostic workflows.
- Checksum-bound safetensors checkpoints, manifests, non-overwriting experiments, and
  independent evidence reconstruction.

The main tier contains 15,179,520 parameters. The original Phase 6 held-out experiment
was a verified negative result. The strongest recent development attempts passed nine
of ten checks but still missed the frozen fault-comparator margin. Targeted-06 is the
next preregistered attempt; its exact intervention is documented in
[`../docs/model/PHASE6_TARGETED06_PLAN.md`](../docs/model/PHASE6_TARGETED06_PLAN.md).

## Research boundaries

- No pretrained model weights or hosted LLM perform the core data generation,
  labeling, model behavior, evaluation, or judging.
- All model scenarios and narratives are project-authored and synthetic.
- No real plant records, procedures, manuals, setpoints, facility data, security
  details, or Navy nuclear information enter the corpus.
- Values are normalized or use explicitly fictional units.
- Public sources support research framing and terminology; their prose is not scraped
  into the model.
- The project does not provide real operating, maintenance, emergency, or safety
  instructions.
- A diagnostic sweep may collect failures, but it cannot certify a model, unlock final
  evaluation, or authorize Phase 7.

## Evidence and implementation reports

- [Current implementation status](../docs/IMPLEMENTATION_STATUS.md)
- [Architecture](../docs/architecture.md)
- [Dataset card](../docs/data/DATASET_CARD.md)
- [Phase 4 tokenizer and smoke report](../docs/model/PHASE4_SMOKE.md)
- [Phase 5 baseline and pilot report](../docs/model/PHASE5_PILOT.md)
- [Phase 6 main result](../docs/model/PHASE6_MAIN.md)
- [Phase 6 remediation plan](../docs/model/PHASE6_REMEDIATION_PLAN.md)
- [Targeted-06 plan](../docs/model/PHASE6_TARGETED06_PLAN.md)
- [Diagnostic full-sweep contract](../docs/model/PHASE6_DIAGNOSTIC_SWEEP.md)
- [Local remediation runbook](../docs/model/PHASE6_REMEDIATION_RUNBOOK.md)
- [Threat model](../docs/threat-model.md)
- [Security control map](../docs/security-controls.md)

## Updating the documentation

Keep current, visitor-facing status in the root README and the implementation-status
handoff. Keep detailed experimental decisions here or in versioned reports. Do not
replace historical measurements with the newest result, and never present a planned
or partial value as measured evidence.

The reusable section order, update checklist, and copy-ready blocks are in the
[README maintenance template](../docs/README_TEMPLATE.md).
