# Pre-build checklist

This began as a pre-implementation gate. It now records which research and Phase 1
foundation items are complete and which later gates remain open. The local Phase 2
structured generator is complete through developmental G01–G15; Phase 3 dataset work
has not started and no pilot data exists.

## Workspace

- [x] Rename `AI-transformer ` to `AI-transformer` and verify the corrected path.
- [x] Confirm `ReactorBench-LM` as the final project name.
- [x] Initialize Git only after confirming the correct folder. Local repository initialized on `codex/foundation`; no remote is configured.
- [x] Add a root README with the visible synthetic/non-operational disclaimer.
- [ ] Choose initial code and data licenses. Deferred by D-041; nonblocking for local foundation and generator work, but required before distribution.

## Scope

- [x] Confirm the primary research question.
- [x] Confirm the pressurized-water-inspired but fictional abstraction.
- [x] Confirm 11 fault families, one benign load-transient driver, and the fictional action-label set.
- [x] Confirm prohibited sources and Navy-information exclusion.
- [x] Confirm that the eventual demo will not accept real-facility data.

## Data contracts

- [ ] Freeze version 1 of the structured trajectory schema.
- [ ] Freeze the task names and structured target formats.
- [x] Define generator invariants before generating the pilot dataset.
- [ ] Define split manifests before narrative rendering.
- [ ] Implement and snapshot-review the decision-tick/channel task-projection contract. A truth-filtered audit payload is not itself a prompt.
- [ ] Group and audit G07, G08/G09, G12, G14, and G15 counterfactual relatives before assigning any split.
- [ ] Create template-family IDs and component-alias families.
- [ ] Create prohibited-content patterns and the real-facility denylist. A redacting, non-exhaustive Phase 2 pattern guard and fixtures now exist; complete the reviewed denylist and manual sample procedure before the dataset pilot.
- [ ] Define duplicate and cross-split leakage tests.
- [x] Define provenance fields and dataset-versioning rules.

## Model contracts

- [ ] Calculate exact parameter counts for smoke, pilot, and main configs.
- [ ] Define causal-mask and next-token target tests.
- [ ] Define checkpoint save/reload equivalence tests.
- [ ] Define the tokenizer corpus and special tokens.
- [ ] Fix baseline implementations and metrics before test evaluation.
- [ ] Define validation-only checkpoint selection.

## Compute

- [ ] Verify the installed PyTorch build supports MPS.
- [ ] Run smoke overfit on a tiny generated shard.
- [ ] Run a 500–1,000-step pilot microbenchmark.
- [ ] Record tokens/sec, peak memory, checkpoint size, and throttling behavior.
- [ ] Estimate local main-run duration from measurements.
- [ ] Set a cloud budget only if local training is impractical.

## Evaluation preregistration

- [ ] Lock IID and strict holdout definitions.
- [ ] Lock primary metrics for every task.
- [ ] Lock E0–E7 experiment comparisons.
- [ ] Define bootstrap confidence-interval procedure.
- [ ] Freeze test splits before the main training run.
- [ ] Commit to reporting negative results and IID-to-composition gaps.
- [ ] Freeze pilot-informed numerical acceptance thresholds before main test evaluation.
- [ ] Freeze majority, rule, n-gram, bag-of-words, recurrent, and smaller-Transformer baseline definitions.
- [ ] Freeze the robustness/OOD suite and error taxonomy.
- [ ] Complete and record human review for every golden scenario.
- [ ] Define the public failure-gallery selection procedure.

## Release documentation

- [ ] Dataset card template.
- [ ] Model card template.
- [ ] Experiment/results report template.
- [x] Source and attribution manifest.
- [x] Known limitations and prohibited-use section.
- [ ] Reproduction instructions and artifact checksums.
- [ ] Implement and verify a fast smoke-reproduction command in a clean environment.
- [ ] Link commit, generator, dataset, tokenizer, checkpoint, evaluation, report, and deployment identifiers.
- [ ] Record training/inference time, throughput, memory, artifact sizes, and cost.
- [ ] Prepare the paper-style experiment report after results stabilize.

## Live demonstration

- [x] Document the intended GitHub and Vercel-style publication outcome.
- [ ] Freeze the versioned inference request and response schemas.
- [ ] Benchmark checkpoint size, cold start, memory, and latency before selecting an inference host.
- [ ] Choose the frontend framework and visualization approach.
- [ ] Implement curated-scenario and constrained-input controls only.
- [ ] Add prohibited-input, request-size, timeout, and rate-limit tests.
- [ ] Verify the live demo against the golden scenario suite.
- [ ] Perform responsive-layout, accessibility, and browser checks.
- [ ] Publish model/dataset cards, checksums, limitations, and GitHub links with the site.
- [x] Select Research Editorial with restrained technical detail as the UI direction.
- [ ] Implement challenge, technical, comparison, and failure-case views after the core lab works.
- [ ] Add deterministic share links and an honestly labeled cached fallback.
- [ ] Replace every concept placeholder with a measured, traceable release value.

## Secure engineering

- [x] Establish the initial security architecture and verification plan.
- [x] Write the implementation threat model and risk register before exposing an API.
- [ ] Freeze a strict runtime request schema with unknown-field rejection and explicit bounds.
- [ ] Define request-byte, token, output, duration, rate, and concurrency limits from benchmarks.
- [ ] Keep inference credentials server-side and separate development, preview, and production secrets.
- [ ] Define CSP and security-header policy, then verify actual deployed responses.
- [ ] Select a safe checkpoint format and verify artifact checksums at startup and release.
- [ ] Add input-boundary, output-encoding, authentication, timeout, and safe-error tests.
- [ ] Enable dependency review, static analysis, secret scanning, and least-privilege CI permissions.
- [ ] Generate release checksums, provenance metadata, and an SBOM.
- [x] Add `SECURITY.md`, threat model, control-to-test mapping, and private reporting instructions.
- [x] Document residual risks without making unqualified security claims.

## Research-phase completion criteria

- [x] Project framing selected.
- [x] External source inventory completed.
- [x] Corpus-ingestion policy established.
- [x] Synthetic dataset schema specified.
- [x] Model tiers and training stages outlined.
- [x] Evaluation splits, baselines, metrics, and ablations outlined.
- [x] Compute plan tailored to the current machine.
- [x] Safety and portfolio boundaries documented.
- [x] Fictional plant topology, state variables, transition contracts, and invariants specified.
- [x] Fifteen golden scenario families and their metamorphic variants drafted.
- [x] Focused literature review and defensible differentiation documented.
- [x] Behavioral-test framework selected.
- [x] Complete the research-only package before local implementation; no model training was performed in that research phase.

The final line above is historical: local implementation was authorized and began on
2026-08-18. Current integration and verification status is tracked in
`../docs/IMPLEMENTATION_STATUS.md`. The unchecked license choices remain deliberately
deferred and do not block local work; they do block distribution.
