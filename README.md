# ReactorBench-LM

**A small decoder-only Transformer trained from random initialization on a wholly
synthetic causal world.**

ReactorBench-LM is an NLP and machine-learning research project built around
**Aster Station**, a fictional reactor-inspired environment. I generate structured
event sequences, render them into narrative tasks, train a project-specific tokenizer
and Transformer, and test whether the model learns causal structure rather than
memorizing familiar wording.

> **Work in progress — Phase 6.** The generator, dataset, tokenizer, model, baselines,
> training pipeline, and development evaluation are implemented. The current model is
> not deployment-ready, and the targeted-06 quality-remediation experiment has not
> started. I preserve failed experiments and do not lower thresholds to manufacture a
> passing result.

> **Fictional and non-operational.** ReactorBench-LM uses no real plant logs,
> procedures, manuals, setpoints, facility data, restricted information, or Navy
> nuclear material. It must not be used for operating, engineering, maintenance,
> emergency, licensing, security, or safety decisions.

## At a glance

| Area | Implementation |
| --- | --- |
| Core model | Decoder-only causal Transformer written from PyTorch primitives |
| Initialization | Random weights; no pretrained checkpoint |
| Tokenizer | Project-trained 2,048-token SentencePiece BPE |
| Main model tier | 15,179,520 parameters |
| Data | Project-authored synthetic Aster Station scenarios and narratives |
| Tasks | Continuation, fault family, evidence, action, summary, and counterfactual comparison |
| Evaluation | Baselines, IID, lexical, structural, compositional, robustness, calibration, and abstention |
| Artifacts | Checksum-bound configurations, manifests, safetensors checkpoints, and reports |
| Current stage | Targeted-06 development remediation prepared; run not started |

## Research question

> Can a small causal Transformer trained entirely from random initialization learn the
> temporal structure of a controlled synthetic event world and generalize to unseen
> wording, component roles, and fault combinations better than simple baselines?

The project covers the complete path from an invented causal environment to a trained
and evaluated language model:

```text
latent Aster state
    -> bounded observations
    -> canonical events
    -> synthetic narratives
    -> leakage-resistant splits
    -> project-trained tokenizer
    -> baselines and from-scratch Transformer
    -> robustness and behavioral evaluation
    -> narrow research interface (planned)
```

Ground truth comes from the structured generator. It never depends on rendered prose,
model output, or a hosted language model.

## What I built

### Synthetic causal environment

- Deterministic seeded state transitions for fictional Aster-A, Aster-B, and Aster-C
  plant variants.
- Separate latent state, observations, canonical events, and narrative rendering.
- Stable operation, benign changes, faults, compound conditions, counterfactuals, and
  insufficient-evidence cases.
- Runtime schemas, invariants, prohibited-content checks, and versioned provenance.

### Dataset and tokenizer

- Project-authored structured and narrative examples with grouped,
  leakage-resistant splits.
- Duplicate, shortcut, overlap, and holdout audits.
- A deterministic SentencePiece BPE trained only on approved `iid_train` prose.
- No external corpus, real operational record, or model-generated label source.

### Model and training

- Token and position embeddings, causal multi-head attention, pre-norm residual
  blocks, feed-forward layers, and a tied language-model head.
- Shifted targets, causal masking, padding exclusion, deterministic evaluation, and
  exact parameter counts.
- Random initialization, training loops, safe checkpoint reload, and MPS support.
- Smoke, smaller, pilot, and main model configurations.

### Evaluation and research engineering

- Majority, deterministic/rule, n-gram, bag-of-words logistic regression, GRU, and
  smaller-Transformer baselines.
- IID and unseen-template evaluation, component and structural holdouts,
  counterfactuals, robustness cases, calibration, selective risk, and abstention.
- Predefined scientific gates, non-overwriting experiment identities, resumable runs,
  artifact checksums, and preserved negative results.
- A local macOS monitor with progress, cooperative safe stop/resume, diagnostic sweep,
  and terminal-failure alarm controls.

PyTorch supplies tensor operations, automatic differentiation, optimization, and the
optional MPS backend. SentencePiece supplies the BPE trainer, and safetensors supplies
the data-only weight format. They do not supply the architecture, weights, scenarios,
labels, or model behavior.

## Current evidence

This project currently has **real but negative model-quality evidence**. Low
teacher-forced validation loss has not yet translated into reliable free-running
structured output.

| Milestone | Measured result |
| --- | --- |
| Smoke correctness | Four-example shard overfit in 300 CPU steps; loss 7.6617 → 0.01160; deterministic reload logits matched |
| Phase 5 pilot | Validation target NLL 0.1593 on MPS; no held-out split opened |
| Original Phase 6 held-out run | IID exact/schema rate 5.16%; golden exact match 3.33%; 23 acceptance checks failed |
| Targeted-03 | 9 of 10 development checks passed after independent gate replay; fault margin missed |
| Targeted-04 | 8 of 10 checks passed; diagnosed-fault oversampling regressed fault margin and continuation F1 |
| Targeted-05 diagnostic | v0.3 passed 9 of 10 checks; fault margin was `-0.003280` against `>= 0.02`; later exposed a shadow generation-cap defect |
| Targeted-06 | Implemented and source-verified; no training or evaluation run started |

Targeted-06 keeps all ten thresholds unchanged. It adds bounded fault emphasis,
preserves hierarchical six-task training in v0.4, validates its six-row batch through
a 1/2/4/6 MPS pilot, applies validation-only temperature scaling, and audits a
shadow-only 256-token counterfactual cap before training. The diagnostic path can
continue across isolated shadow-view boundary failures to collect more evidence, but
it cannot certify a model or authorize final evaluation.

See the [implementation status](docs/IMPLEMENTATION_STATUS.md) for the exact current
checkpoint and the [targeted-06 plan](docs/model/PHASE6_TARGETED06_PLAN.md) for the
frozen intervention.

## Repository layout

```text
configs/              Versioned dataset, model, and experiment configurations
docs/                 Architecture, evidence, runbooks, status, and security notes
research/             Authoritative research requirements and design decisions
schemas/              Versioned Aster, dataset, and compact-output contracts
scripts/              Reproduction, verification, monitoring, and run controls
src/reactorbench/     Generator, dataset, tokenizer, model, training, and evaluation code
tests/                Unit, property, integration, and contract tests
```

Generated datasets, checkpoints, and run evidence are local artifacts unless a
release explicitly includes a reviewed subset.

## Getting started

### Requirements

- Python 3.12 (reproducible baseline; Python 3.13 is also declared compatible)
- [`uv`](https://docs.astral.sh/uv/)
- macOS with Apple MPS for the documented long local experiments; correctness tests
  and smoke work can run on CPU

### Install and inspect

```bash
git clone https://github.com/Zachary200114/reactorbench-lm.git
cd reactorbench-lm
uv sync --frozen --all-groups
make help
```

### Run the quality gates

```bash
make check
```

This runs formatting checks, linting, strict type checking, tests, coverage, and
artifact verification. Some reproduction commands require reviewed local artifacts
that are intentionally not treated as public training data.

### Verify existing milestones

```bash
make phase4-verify
make phase5-verify
```

### Inspect the Phase 6 development runner

Do not start a long run until the working tree is clean and the dry run succeeds.

```bash
./scripts/run_phase6_pipeline.sh --dry-run
./scripts/open_phase6_progress_gui.sh
```

The GUI can select either the official fail-fast run or the non-certifying diagnostic
full sweep. Opening the GUI does not start training. Exact run, status, stop, resume,
and safety instructions are in the
[Phase 6 runbook](docs/model/PHASE6_REMEDIATION_RUNBOOK.md).

## Security and scientific integrity

ReactorBench-LM treats model training and evaluation as an artifact-controlled
research workflow:

- strict schemas reject unknown fields and out-of-range values;
- browser and model output are treated as untrusted data;
- checkpoints use safetensors rather than unsafe pickle loading;
- runs are non-overwriting and bound to source, configuration, and artifact hashes;
- evaluation partitions are isolated from tokenizer and model training;
- diagnostic sweeps cannot silently convert a failed gate into a pass;
- final and historical golden access remain explicitly controlled; and
- failures and limitations are retained rather than rewritten.

These controls reduce risk; they do not make the project “fully secure.” See the
[threat model](docs/threat-model.md), [security control map](docs/security-controls.md),
and [security policy](SECURITY.md).

## Documentation

Start here:

- [Current implementation status](docs/IMPLEMENTATION_STATUS.md)
- [Architecture](docs/architecture.md)
- [Dataset card](docs/data/DATASET_CARD.md)
- [Phase 4 tokenizer and smoke evidence](docs/model/PHASE4_SMOKE.md)
- [Phase 5 baselines and pilot](docs/model/PHASE5_PILOT.md)
- [Phase 6 main result](docs/model/PHASE6_MAIN.md)
- [Targeted-06 remediation plan](docs/model/PHASE6_TARGETED06_PLAN.md)
- [Phase 6 diagnostic sweep](docs/model/PHASE6_DIAGNOSTIC_SWEEP.md)
- [Phase 6 local runbook](docs/model/PHASE6_REMEDIATION_RUNBOOK.md)
- [Research documentation index](research/README.md)
- [Canonical project requirements](research/PROJECT_REQUIREMENTS.md)
- [README maintenance template](docs/README_TEMPLATE.md)

## Roadmap

- [x] Fictional Aster Station generator and invariants
- [x] Structured/narrative dataset and leakage-resistant splits
- [x] Project-trained tokenizer
- [x] Decoder-only Transformer from random initialization
- [x] Baselines, smoke, pilot, and original main experiment
- [ ] Pass or conclusively close the frozen Phase 6 development gates
- [ ] Run the separately authorized final evaluation, if prerequisites pass
- [ ] Build the narrow inference service
- [ ] Build the Research Editorial public interface
- [ ] Prepare the reproducible release, model card, and public evidence bundle

Phase 7 remains blocked until a model passes the required development gates and the
separate final-evaluation prerequisites are satisfied.

## License

Unless a file says otherwise, I release the original code, documentation,
project-authored synthetic data, and other original material in this repository under
the [0BSD License](LICENSE).

Use it, copy it, change it, share it, or build on it however you want. Credit is
appreciated, but it is not required. Third-party dependencies and referenced material
keep their own licenses and terms.

## Publication boundary

I handle GitHub pushes and any eventual deployment myself. Project scripts do not
publish the repository or deploy a service. [`robots.txt`](robots.txt) allows crawler
access; legal reuse terms come from [LICENSE](LICENSE), not crawler policy.
