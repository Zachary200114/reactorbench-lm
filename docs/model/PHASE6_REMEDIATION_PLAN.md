# Phase 6 model-remediation plan

Status: **preregistered design only; no implementation, training, or new evaluation**
Date: 2026-08-20 America/New_York

## Objective

ReactorBench-LM v0.1 trained successfully but failed its behavioral gate. This plan
defines three bounded improvement iterations before any inference service or Research
Editorial application is built:

1. v0.2 — output reliability;
2. v0.3 — semantic task learning and abstention; and
3. v0.4 — strict generalization and final go/no-go evaluation.

These are experimental iterations inside model remediation, not three new product
phases. Phase 7 remains blocked until a final checkpoint passes a fresh, frozen,
owner-reviewed behavioral evaluation.

## Evidence motivating the plan

The plan responds to separate measured failure classes rather than assuming one larger
model will solve everything:

- Only 16 of 894 v0.1 held-out generations were schema-valid even though IID
  teacher-forced NLL was 0.04254. This is primarily an output/exposure problem.
- IID fault/action/continuation macro-F1 was 0.0000 / 0.0290 / 0.2029, evidence F1 was
  0.0238, and required abstention was 4/60. Valid syntax alone will not solve the
  semantic problem.
- Strict holdout exact match was 0% for template, component, severity, composition,
  and counterfactual splits. This is a generalization problem distinct from syntax.
- The 512-token window truncated 491/630 training prompts, 198/252 validation prompts,
  and 788/894 held-out prompts. Four composition and ten counterfactual targets could
  not fit by design. Target length and context allocation are material constraints.
- Simple baselines substantially outperformed the Transformer on several tasks.
  Complexity is not accepted as value without measured improvement.

The complete v0.1 result and hashes remain in `docs/model/PHASE6_MAIN.md`.

## Scientific boundary

- Preserve every v0.1 config, checkpoint, prediction, report, correction record, and
  held-out-access record unchanged.
- Do not use v0.1 held-out examples for architecture choice, prompt/target design,
  hyperparameter selection, stopping, or threshold changes. The aggregate v0.1 result
  motivates this program but is not an optimization set.
- Use only project-authored synthetic data, `iid_train`, `iid_validation`, and newly
  designated development-only views for iteration decisions.
- Before any v0.4 training or final-model selection, freeze the policy that generates
  a fresh v0.4 final holdout:
  seed ranges, scenario groups, aliases, renderer families, corruption rules, task
  inventory, checksums, and one-access policy. Do not render or inspect its labels in
  routine iteration commands.
- The v0.1 golden suite is historical evidence, not a clean v0.4 gate. Prepare a
  versioned golden extension using new scenario seeds/renderings, keep it out of all
  training and development views, and require project-owner approval before final
  access.
- Never weaken the v0.1 report. v0.2–v0.4 use a new, preregistered contract and must be
  compared transparently with v0.1.
- Report both unconstrained model output and constrained serving output where
  applicable. A grammar may guarantee syntax; it cannot be credited with semantic
  correctness.

## Shared implementation rules

- Keep the decoder-only Transformer random-initialized and project-trained. No
  pretrained weights, hosted LLMs, model hubs, or external corpora.
- Keep strict Pydantic target models as the source of truth. Any compact model language
  must compile bijectively to and from those targets.
- Make every iteration non-overwriting and checksum-bound to source commit, dataset,
  tokenizer, output-contract version, config, checkpoint, predictions, and report.
- Select checkpoints using validation data only. Teacher-forced NLL cannot be the sole
  selection metric after v0.1; free-running structured behavior must participate.
- Permit at most one control and two preregistered variants per iteration. Do not grow
  an open-ended search after seeing validation results.
- Run smoke correctness before pilot or main training. A failed phase gate stops the
  iteration for diagnosis and documentation.
- Keep the original 85% branch-coverage floor and the existing Ruff, mypy, build,
  artifact, safe-checkpoint, determinism, and provenance gates.

## Iteration 1 — v0.2 output reliability

### Question

Can a shorter task-specific output language plus a truth-independent constrained
decoder eliminate structural failures without changing the underlying semantic task?

### Planned changes

- Define a compact, versioned target language containing only task name, allowlisted
  enum values, ordered prompt-local fact references, and bounded task-specific fields.
- Implement a deterministic compiler between that language and the existing strict
  `ProjectionTaskTargetValue` contracts. Round trips must be canonical and one-to-one.
- Remove redundant JSON field names from the learned sequence; retain canonical JSON
  as the audit/API representation produced by the compiler.
- Build a finite-state or trie-constrained decoder from the task contract, enum
  allowlists, and prompt-visible fact references only. It must never inspect latent
  truth, targets, scenario IDs, or future events.
- Keep the v0.1 tokenizer, 8×384 architecture, and 512-token context for the primary
  comparison so output representation is the main changed factor.
- Replace the single 256-token reserve with a measured, task-specific generation cap
  frozen from training/validation target-length inventories.
- Record unconstrained greedy results alongside constrained results. The constrained
  path is the serving candidate; the unconstrained path reveals how much the model
  itself learned about syntax.

### Tests before training

- Compact target round-trip equality for every task contract and enum value.
- Unknown token, invalid enum, duplicated/misordered fact reference, wrong task,
  premature EOS, extra trailing token, and maximum-length rejection.
- Decoder allowlist proof: every accepted sequence compiles to the strict task schema;
  every canonical training target is reachable.
- No truth leakage from targets, latent state, provenance, scenario ID, or source event
  IDs into decoder constraints.
- Shifted loss, causal mask, prompt/target boundary, cache parity, save/reload,
  deterministic evaluation, request bounds, and no-global-RNG regressions.
- Exact target-length and prompt-retention report before any fit.

### Advancement gate

- 100% of train/validation targets compile, fit, and round-trip.
- Constrained validation parse and schema validity are exactly 100%.
- No constrained output requires repair, fallback JSON, or truth-dependent completion.
- Generation-cap exhaustion is at most 1% on validation and is reported per task.
- Prompt truncation is materially lower than v0.1 and contains no target truncation;
  the exact percentage is frozen and reported before v0.2 main training.
- Unconstrained parse/schema rates, exact semantic match, latency, and memory are
  reported even when poor; they do not silently inherit the constrained score.

If structural correctness does not pass after the control plus two preregistered
variants, stop. Do not proceed to semantic tuning on an unstable output contract.

## Iteration 2 — v0.3 semantic learning and abstention

### Question

Once syntax is reliable, can the model beat simple comparators on diagnosis/action and
learn evidence selection and required abstention using development-only evidence?

### Planned changes

- Freeze the v0.2 compact output contract and constrained decoder; do not change both
  representation and semantic training simultaneously.
- Add deterministic task-balanced batching so frequent or short targets cannot
  dominate optimization.
- Expand only project-authored training/development scenarios using group-atomic seeds,
  paraphrases, component aliases, evidence-removal views, and insufficient-evidence
  counterfactuals. Preserve leakage-resistant split grouping.
- Increase abstention and near-neighbor contrast coverage without changing the task
  truth rules. Every augmentation must have generator/projection provenance.
- Compare at most three candidates: the v0.2 control, task-balanced training, and one
  preregistered sequence-level exposure candidate. The exposure candidate may use a
  bounded scheduled/self-conditioned loss only after correctness tests prove it cannot
  leak targets or cross the causal boundary.
- Select using a frozen validation composite led by semantic metrics, not NLL alone.

### Tests before training

- Class/task balance and group-atomic split tests.
- Duplicate, renderer/template shortcut, alias shortcut, target-text leakage, and
  provenance audits over every added record.
- Evidence-removal metamorphic tests: removing decisive evidence must change diagnosis
  or cause abstention according to the generator contract.
- Counterfactual factor-isolation tests and exact prompt-local evidence resolution.
- Scheduled/self-conditioning tests, if used, proving only earlier generated target
  positions are visible and global RNG state remains isolated.
- Calibration and selective-risk computations fail closed on invalid or empty support.

### Advancement gate

On the frozen v0.3 development suite, the selected checkpoint must satisfy the
behavioral thresholds already expected of a useful model:

- fault-family and next-action macro-F1 each at least 0.02 above the strongest
  preregistered simple comparator on the same view;
- continuation macro-F1 at least 0.90;
- evidence F1 at least 0.70;
- required-abstention accuracy at least 0.80;
- no-fault false-positive rate at most 0.10;
- expected calibration error at most 0.15 and selective risk at 80% coverage at most
  0.20; and
- constrained parse/schema validity exactly 100%, with unconstrained behavior reported.

If no candidate passes, preserve the negative result and stop before generalization or
fresh-test access. A larger model is not the automatic fallback.

## Iteration 3 — v0.4 strict generalization

### Question

Can the validation-passing semantic model generalize across unseen renderers,
components, severities, compositions, counterfactuals, and narrative noise?

### Planned changes

- Freeze v0.3 output and semantic-training contracts.
- Add development-only shadow holdouts for renderer, component-role, severity,
  composition, counterfactual, and noise generalization. Their seeds/groups must be
  separate from the future final holdout.
- Compare context strategies only if measured truncation remains material: compact
  512-token control versus one preregistered longer-context candidate supported by a
  new MPS memory/throughput pilot. Do not assume 1,024 tokens are feasible or better.
- Compare the 15,179,520-parameter control with at most one justified capacity variant
  only if v0.3 learning curves show underfitting. Scaling is not a substitute for data
  or output-contract correctness.
- Select on a frozen worst-split-aware validation rule so IID strength cannot conceal a
  failed structural holdout.

### Tests before training

- Exact separation of fit, IID validation, shadow holdouts, fresh final holdouts, and
  golden records by group and content checksum.
- New alias/renderer/seed inventories and leakage/duplicate/shortcut audits.
- Context-bound tests proving complete targets fit, prompt truncation is measured, and
  no task silently receives a different information boundary.
- Per-variant parameter, memory, throughput, checkpoint, cache, and deterministic
  evaluation measurements.
- Separate metrics and intervals for every split; no merged headline substitutes for
  split evidence.

### Advancement gate

- All v0.3 behavioral gates continue to pass on IID validation.
- Every shadow holdout meets its preregistered task gate; composition remains a
  mandatory reported result with an interval rather than a post-hoc threshold.
- No split has schema validity below 100% on the constrained path.
- No unresolved leakage, shortcut, context-fit, artifact, or provenance finding.
- The selected configuration, thresholds, final manifests, golden extension, decoder,
  bootstrap seed, and checkpoint-selection rule are frozen before final access.

Only then may one fresh v0.4 held-out evaluation run. Phase 7 unlocks only if that
report passes the frozen deployment-relevant gates. Otherwise v0.4 closes as another
honest negative result and the project presents its research findings without a live
model endpoint.

## Experiment budget and stop policy

Per iteration:

- one smoke correctness run;
- one small/pilot feasibility run per candidate;
- no more than three preregistered main-tier candidates total;
- no final held-out decoding until all development gates pass;
- one fresh final held-out access for v0.4 only; and
- no overwrite, threshold relaxation, or test-driven retry.

The measured v0.1 E3 fit took 776.66 seconds and its held-out decoding/evaluation took
about 88 minutes on the local machine. Those are historical measurements, not promises
for the new design. Each new run must record its own training time, target tokens per
second, peak MPS memory, checkpoint size, decoding latency, and interruption history.

At or below the user-reported 5% account-usage boundary, begin checkpoint preparation
and do not start a new iteration or training run. At or below 1%, stop all
non-checkpoint work. Exact account usage is not visible to the implementation process,
so the user must supply the visible percentage when precise enforcement matters.

## Planned artifact/version layout

No listed file exists merely because it is planned. Implementation should use:

- `configs/experiments/phase6-remediation-v0.2.0.toml` — shared program and R1 freeze;
- later versioned configs for v0.3 and v0.4 rather than mutating v0.2;
- a versioned compact-output schema/manifest under `schemas/`;
- non-overwriting `runs/phase6-remediation-v0.2.0-*`, `v0.3.0-*`, and `v0.4.0-*`;
- a dedicated fresh-test access ledger distinct from v0.1;
- per-iteration reports and a final model card amendment; and
- exact source/config/data/tokenizer/output-contract/checkpoint/report hashes.

## Immediate implementation milestone after usage reset

Implement only the v0.2 compact target contract, compiler, constrained decoder, and
their unit/property/contract tests. Do not generate new data or train a model in the
same atomic milestone. The milestone passes when every existing train/validation
target round-trips and the decoder can emit only schema-compilable task outputs without
using hidden truth.

## Explicitly deferred

- Exact compact token grammar and task generation caps, pending a train/validation
  inventory report during v0.2 implementation.
- Longer context, pending measured truncation after compact targets.
- Any larger model, pending v0.3 underfitting evidence.
- Final v0.4 thresholds/manifests and golden-extension checksums, which must be frozen
  before their one permitted access.
- Phase 7 inference/UI, Phase 8 release preparation, GitHub push, and Vercel deployment.
