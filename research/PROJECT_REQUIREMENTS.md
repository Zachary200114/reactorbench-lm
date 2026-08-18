# ReactorBench-LM project requirements

Status: canonical pre-implementation scope
Purpose: define what the finished project must contain, what may be added if evidence supports it, and what is explicitly excluded

## 1. Finished-project definition

ReactorBench-LM is complete only when it demonstrates the full path from an invented causal world to a trained and evaluated AI system:

1. versioned Aster Station synthetic state generator;
2. project-authored dataset with leakage-resistant splits;
3. project-specific tokenizer;
4. decoder-only Transformer trained from random initialization;
5. meaningful non-Transformer and smaller-model baselines;
6. preregistered IID, structural, lexical, and compositional evaluation;
7. abstention, robustness, counterfactual, and failure analysis;
8. reproducible configurations, manifests, checksums, and release records;
9. secure inference service and public Research Editorial web experience;
10. GitHub repository containing evidence for the research, engineering, and security claims.

The model, data, results, and deployment must remain inside the wholly fictional Aster Station world.

## 2. Required capabilities

### Synthetic environment

- Deterministic seeded scenario generation.
- Separate latent state, observation, canonical event, and language-rendering layers.
- Normal operation, benign load changes, single faults, compound faults, and insufficient-evidence cases.
- Versioned plant variants, aliases, renderer families, and dependency maps.
- Generator invariants and prohibited-content scanning.
- Structured ground truth that never depends on generated prose or model output.

### Model construction

- Decoder-only causal Transformer defined by the project.
- Random initialization with no pretrained model weights.
- Project-trained tokenizer.
- Smoke, pilot, main, and optional stretch configurations with exact parameter counts.
- Checkpoint save/reload equivalence and deterministic evaluation tests.
- Training curves, configuration snapshots, and measured hardware performance.

### Model tasks

- Causal language modeling and next-event prediction.
- Fault-family identification.
- Evidence-event selection or extraction.
- Structured incident summarization.
- Fictional action-label selection.
- Required abstention when the visible evidence is insufficient.

### Evaluation

- Baseline comparisons.
- IID and strict compositional holdouts.
- Golden-scenario behavioral testing.
- Out-of-distribution and malformed-input testing.
- Calibration and coverage-risk analysis.
- Error taxonomy and failure gallery.
- Confidence intervals where appropriate.
- Negative results reported rather than hidden.

### Publication

- GitHub repository with documentation, safe sample artifacts, reproducibility commands, and tagged releases.
- Public Research Editorial website hosted through Vercel or an equivalent presentation platform.
- Versioned inference service running the project's own trained model.
- Dataset card, model card, experiment report, security evidence, and limitations.
- A short paper-style report when the main results are stable.

## 3. Required comparisons

The project must compare the main Transformer with:

- most-common-class baseline;
- deterministic or keyword/rule baseline;
- n-gram language-model baseline for sequence prediction;
- bag-of-words logistic regression for classification;
- small LSTM or GRU if pilot compute permits;
- smaller Transformer configurations.

The LSTM/GRU comparison may be omitted only if the reason is documented before test evaluation. Simple baselines may outperform the model on some tasks; that result must be preserved.

## 4. Required robustness cases

- unknown but schema-valid component aliases;
- missing observations;
- conflicting observations;
- shortened context and truncated sequences;
- harmless event insertions;
- unusual but valid wording and event ordering;
- values near invented classification boundaries;
- no-fault and no-recognizable-fault cases;
- multiple plausible explanations;
- incompatible schema or model versions;
- compound faults not seen together during training;
- sparse evidence requiring abstention.

## 5. Required public UI

Use the Research Editorial direction: light, calm, evidence-centered, and technical without resembling a real control room.

The finished site requires:

- landing page and research question;
- guided scenario laboratory;
- normalized telemetry and simplified topology;
- canonical event timeline;
- prediction, confidence, abstention, evidence, and fictional action output;
- hidden-then-revealed simulator ground truth;
- challenge mode;
- research/technical mode;
- model and baseline comparison;
- results dashboard;
- failure-case gallery;
- architecture, model card, dataset card, security, and limitations pages;
- responsive, accessible, keyboard-operable behavior;
- loading, timeout, offline, rate-limit, and inference-unavailable states;
- deterministic shareable scenario links;
- visible model, schema, and scenario versions.

Every displayed metric must come from a recorded evaluation artifact. Concept-design placeholder values must never reach production.

## 6. Required secure engineering

- Threat model and risk register.
- Strict server-side runtime schemas.
- Request, token, output, duration, rate, and concurrency limits.
- Server-only credentials and environment separation.
- Safe output encoding and production error handling.
- CSP and verified response security headers.
- Trusted, checksummed, safely loaded checkpoint artifacts.
- Static analysis, dependency review, secret scanning, tests, and least-privilege CI.
- Release provenance, checksums, and SBOM.
- `SECURITY.md` and private vulnerability-reporting instructions.
- Honest residual-risk documentation.

## 7. Required reproducibility

- Fixed seeds and configuration files.
- Dataset split and provenance manifests.
- Tokenizer, dataset, model, and results checksums.
- Clean-environment dependency installation.
- A fast smoke reproduction command.
- A documented path for reproducing the main experiment.
- Source commit, generator, dataset, tokenizer, checkpoint, and evaluation linkage.
- Training and inference compute, latency, memory, and cost record.

## 8. High-value additions after the core passes

- Counterfactual “why this prediction?” view.
- Evidence-removal experiments.
- In-browser quantized inference if equivalence and performance are acceptable.
- Public smaller demonstration checkpoint.
- Interactive comparison among rule, sequence, and Transformer models.
- Short demonstration video or animated walkthrough.
- Paper-style PDF or technical poster.

These cannot delay the core experiment or replace missing evidence.

## 9. Explicit non-goals

- General-purpose chatbot.
- Retrieval-augmented generation over real nuclear documents.
- Real plant logs, procedures, event reports, setpoints, or facility data.
- Navy nuclear or service-derived non-public information.
- Real operator, engineering, safety, emergency, or maintenance advice.
- Realistic control-room replication.
- Arbitrary file upload, URL ingestion, or model upload.
- User accounts without a demonstrated requirement.
- Complex 3D plant graphics.
- Security attack or vulnerability scenarios.
- Numerous architectures without a preregistered experimental reason.
- Administrative dashboards created only for visual effect.

## 10. Completion rule

No single accuracy number makes the project complete. Completion requires traceable evidence that the generator, dataset, model, baselines, evaluation, UI, deployment, security controls, and documentation correspond to the same versioned release.
