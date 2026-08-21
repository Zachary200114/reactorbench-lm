# ReactorBench-LM research dossier

Status: **research complete; Phases 0–6 implemented and verified locally; Phase 6
closed as a negative experiment and Phase 7 blocked**
Prepared: 2026-08-18; implementation status reconciled 2026-08-20.

## Recommended project

**ReactorBench-LM: Training a Small Transformer from Scratch on Synthetic
Nuclear-Plant Event Sequences**

ReactorBench-LM is a small decoder-only Transformer trained from random initialization
on a wholly project-authored dataset representing a fictional civilian,
pressurized-water-inspired energy facility. The model learns causal language modeling
and will later be evaluated on structured tasks over synthetic event narratives:
next-event prediction, fault-family identification, evidence extraction, incident
summarization, and selection of a fictional diagnostic label.

The research question is:

> How well can a small causal Transformer trained from scratch learn system-event
> language and generalize to unseen combinations of faults in a controlled,
> simulator-grounded synthetic domain?

The project is a model-development and evaluation experiment. It is **not** a real
plant simulator, digital twin, operator assistant, safety system, emergency tool, or
source of engineering guidance.

## Settled boundaries

- The model is trained from scratch; no pretrained weights and no hosted LLM API
  perform the core modeling work.
- PyTorch provides tensor operations, automatic differentiation, optimization, and an
  optional MPS backend. Project code defines the architecture, data, initialization,
  training, and evaluation.
- All scenarios, training/evaluation narratives, identifiers, state variables, fault
  rules, policies, and outputs are project-authored and synthetic.
- No real event notifications, operator logs, procedures, plant manuals, setpoints,
  facility names, security details, or Navy nuclear information enter the corpus.
- Public sources support broad research, terminology, governance, and experiment
  design; their prose is not scraped into the model.
- Telemetry uses normalized values or explicitly fictional synthetic units.
- Any eventual interface will accept only generated scenarios from the fictional world.

## What this folder contains

- `RESEARCH_BLUEPRINT.md` — framing, system abstraction, model plan, experiments,
  compute plan, risks, and roadmap.
- `DATASET_SPEC.md` — corpus views, contracts, generation, splitting, validation,
  provenance, and target scale.
- `SOURCE_MANIFEST.csv` — reviewed sources with licensing and ingestion decisions.
- `VOCABULARY_SEED.md` — safe terminology boundary for synthetic authoring.
- `FICTIONAL_PLANT_SPEC.md` — Aster Station topology, variables, causal rules, fault
  contracts, invariants, and safety boundary.
- `GOLDEN_SCENARIOS.md` — 15 draft withheld behavior families pending final human review
  and freeze.
- `LITERATURE_REVIEW.md` — simulator scope, small-model training, compositional
  evaluation, nuclear-AI precedent, and governance.
- `LIVE_DEMO_PLAN.md` — later GitHub/Vercel-style presentation architecture and gates.
- `SECURE_ENGINEERING_PLAN.md` — trust boundaries, controls, tests, CI safeguards, and
  public evidence requirements.
- `PROJECT_REQUIREMENTS.md` — canonical finished-project scope and non-goals.
- `EXPERIMENT_ACCEPTANCE_PLAN.md` — hypotheses, baselines, metrics, robustness,
  acceptance, golden review, and error taxonomy.
- `REPRODUCIBILITY_RELEASE_PLAN.md` — smoke/full reproduction, artifact lineage,
  performance/cost records, and release contents.
- `UI_PRODUCT_REQUIREMENTS.md` — approved Research Editorial product direction.
- `DECISION_LOG.md` — settled and provisional decisions.
- `PREBUILD_CHECKLIST.md` — completed and remaining phase gates.

## Implemented checkpoint

The Phase 3 project-owner-approved local candidate contains 204 audit trajectories,
1,762 single-input projections, 14 counterfactual pairs, 553 rendered candidates,
1,776 task examples, and 18 bounded corruption records. Its split, evidence, grouping,
duplicate, shortcut, review, and typed-artifact gates passed. It is not a public
release.

Phase 4 trains a deterministic 2,048-token SentencePiece BPE only on the 195 approved
`iid_train` documents and implements the decoder-only causal Transformer from PyTorch
primitives. Exact model tiers are 675,328 smoke, 5,328,896 pilot, and 15,179,520 main
parameters. The 300-step CPU smoke run passed causal masking, target shifting,
padding, deterministic evaluation, tiny-shard overfit, and checksum-bound safetensors
reload equality. The repository gate passed 677 tests with 85.37% branch coverage.

Phase 5 completed all preregistered baselines plus 300-step smaller and 500-step pilot
Transformers. The pilot used MPS, selected validation target NLL 0.1593, and produced a
checksum-bound 23,682,552-byte safetensors checkpoint. Only train/validation views were
used. The 15,179,520-parameter Phase 6 main configuration and numerical gates are now
frozen before test access.

Exact implementation state, measurements, limitations, and hashes are in
`../docs/IMPLEMENTATION_STATUS.md`, `../docs/model/PHASE4_SMOKE.md`, and
`../docs/model/PHASE5_PILOT.md`.

## Hardware evidence

The target computer is an Apple M3 MacBook Air with 16 GB unified memory. Torch 2.13.0
ran the Phase 5 pilot on MPS. The pilot measured 1,909.38 target tokens/second and
3,401,547,776 peak driver-allocated bytes. Thermal throttling is not directly
observable and remains an explicit limitation.

## Immediate next step

Begin Phase 6 with a read-only freeze audit. Complete the human golden-suite review,
freeze test manifests, implement the E0–E7 evaluator/calibration/ablation contracts,
and verify the approved Phase 5 artifact before main training or any test access. Do
not build inference or UI work yet.

GitHub pushing and Vercel deployment remain reserved for Zachary. Code/data licensing
must be resolved before distribution.
