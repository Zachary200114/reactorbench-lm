# ReactorBench-LM implementation status

Last updated: 2026-08-23 America/New_York

Current phase: **Phase 6 remediation, Iteration 1 source-correctness milestone complete;
Phase 7 remains blocked**

Current objective: preserve the verified v0.2 compact-output contract, bijective
compiler, and truth-independent constrained token decoder as a safe local checkpoint.
Do not begin data conversion, target inventories, training, held-out evaluation, UI,
publication, or deployment in this checkpoint.

Checkpoint reason: the user authorized only the bounded Iteration 1 implementation
slice and explicitly prohibited new data generation, training, held-out access, changes
to Phase 6 v0.1 artifacts, GitHub push, and deployment.

Intended project path: `/Users/zachary/Documents/Personal-Projects/AI-transformer`

## Completed work

- Phases 0-5 remain complete locally. Phase 6 v0.1 remains closed as an immutable,
  honestly reported negative experiment; Phase 7 is still gated off.
- Added developmental compact-output contract `0.2.0` with wire prefix `RB2`, bounded
  one-line syntax, fixed task-specific fields, closed enum vocabularies, `-` for empty
  or absent values, canonical set ordering, and prompt-local fact-reference checks.
- Added a deterministic compiler in both directions between compact text and the
  existing strict `ProjectionTaskTargetValue` Pydantic contracts.
- Canonical JSON remains the audit/API representation; compact text is only the
  learned/decoded representation.
- Added a constrained greedy token selector driven only by task name, contract enums,
  prompt-visible fact references, tokenizer state, and a generated-token bound.
  Decoder context has no target, fault truth, latent state, scenario ID, provenance,
  source event ID, or future-event field.
- EOS is legal only when the complete decoded text compiles without repair. Unknown,
  BOS, PAD, and premature EOS tokens fail closed. One tokenizer-leading zero-width
  SentencePiece boundary marker is permitted, but repeated zero-width loops are not.
- Added a checksum-bound developmental schema snapshot and packaged-resource API.
  The contract remains `frozen: false` until the development-only target-length
  inventory and task generation caps are measured and preregistered.
- Added unit, Hypothesis property, contract, packaging, maximum-length, reference-order,
  deterministic-selection, no-global-RNG, and no-truth-context tests.
- Verified reachability with the existing project SentencePiece tokenizer without
  changing it: one representative `next_action` target used 31 tokens with the
  2,048-token vocabulary; the full per-prefix allowlist/EOS check took 0.263 seconds.
- No model weights, dataset artifacts, v0.1 reports, predictions, checkpoints, or
  access ledgers were changed. No data was rendered and no model was trained.
- No push, release, Vercel deployment, hosted model/API, or external publication was
  performed.

## Files created or changed

- Core implementation:
  `src/reactorbench/evaluation/compact.py`,
  `src/reactorbench/evaluation/__init__.py`, and
  `src/reactorbench/resources.py`.
- Versioned contract:
  `schemas/compact-output/v0/README.md`,
  `schemas/compact-output/v0/contract.json`, and
  `schemas/compact-output/v0/manifest.json`.
- Packaging: `pyproject.toml`.
- Tests:
  `tests/unit/test_compact_targets.py`,
  `tests/property/test_compact_target_properties.py`,
  `tests/contract/test_compact_output_contract.py`,
  `tests/contract/test_package_resources.py`, and
  `tests/contract/verify_distribution_artifacts.py`.
- Documentation:
  this file, `docs/model/PHASE6_REMEDIATION_PLAN.md`, and
  `research/DECISION_LOG.md`.

## Tests and checks run

- Runtime: CPython 3.12.14 after repairing the stale local `.venv` interpreter link.
- Final actual-repository compact/resource suite: **30 passed in 1.70 seconds**.
- Focused compact module branch coverage: **92%**.
- Full isolated staging suite: **739 passed, 1 skipped in 326.18 seconds**.
  The skip is the expected missing local project-owner review record.
- Full repository branch coverage: **85.20%**; the 85% gate was not weakened.
- Final Ruff format: **151 files already formatted**.
- Final Ruff lint: passed.
- Final strict mypy: **117 source files**, no issues.
- Wheel and sdist: built successfully.
- Distribution verifier: passed exact wheel/sdist resource inventory and an isolated,
  no-dependency/no-index wheel installation.
- Real tokenizer reachability: passed for the representative compact target, including
  the leading SentencePiece boundary marker and terminal EOS.

## Decisions made

- Use the exact wire shape `RB2|<task>|<task fields>` with `,` for lists and `~` for
  nested counterfactual conclusions. Delimiters cannot occur in allowlisted atoms.
- Preserve enum order only where the strict target treats a tuple as a set. Preserve
  meaningful evidence-slot ordering. Require fact references to be unique, visible,
  and increasing within each `o`, `e`, or `c` namespace.
- Keep the existing v0.1 tokenizer, 8×384 primary architecture, and 512-token context
  unchanged for the later controlled comparison.
- Keep the output contract developmental rather than falsely frozen. Task-specific
  generation caps require a permitted train/validation-only inventory first.
- Treat constrained syntax as structural reliability only. It cannot receive credit
  for semantic correctness, and later reports must retain unconstrained results.
- Do not integrate this decoder into a training/evaluation run in the same milestone.

## Assumptions

- All future inputs remain project-authored, synthetic, fictional, normalized, and
  non-operational.
- The strict existing Pydantic target models remain the semantic source of truth.
- Exact account-usage percentage is not observable. No claim is made that the 5% or
  1% cutoff was measured.
- The project owner still controls GitHub push and Vercel publication.

## Known failures and residual risks

- Two broad read-only inspection commands accidentally traversed historical Phase 6
  v0.1 held-out report/prediction or generated-data paths while locating/comparing the
  workspace. No fresh holdout was accessed, no historical artifact was modified, and
  none of the observed historical content was used for design, selection, or tuning.
  Subsequent synchronization and review use an explicit source/test/schema/docs
  allowlist. This process breach cannot be undone and must remain disclosed.
- The compact contract has not yet been run over a safely isolated train/validation
  target inventory. The mixed historical artifact was deliberately not used after the
  boundary incident. Therefore the full Iteration 1 advancement gate is not claimed.
- Task-specific generation caps, target-length inventory, prompt-retention report,
  constrained validation behavior, and cap-exhaustion rates remain unmeasured.
- The token allowlist is correctness-first. Only one representative real-tokenizer
  path has a latency measurement; task-wide latency and memory remain to be measured.
- Free-running semantic quality is unchanged. No v0.2 model exists, and Phase 7 must
  not use the failed v0.1 checkpoint as if this syntax work improved its behavior.
- Code and data licenses remain `TBD`, blocking public distribution.

## Open blockers

- A development-only, holdout-safe target inventory boundary is required before
  measuring all train/validation round trips and freezing generation caps.
- Iteration 1 advancement still requires 100% development compilation/fit/round-trip,
  100% constrained validation parse/schema validity, cap-exhaustion and prompt-
  retention reports, and unconstrained comparison metrics.
- Iterations 2 and 3, Phase 7, and Phase 8 remain unstarted.
- License decisions, SBOM/security/release evidence, and accessibility/deployment
  gates remain open.
- GitHub push and Vercel deployment remain owner-managed and unauthorized here.

## Uncommitted work

- No source or documentation edit is intentionally left unfinished. At checkpoint
  commit time, the exact reviewed Iteration 1 allowlist is included in one local
  commit; confirm a clean worktree on resume.
- The Homebrew Python 3.12 runtime was installed and the previously broken `.venv`
  interpreter symlink was repaired. `.venv` is ignored environment state, not a source
  change.
- Staging-only coverage, bytecode, build, and distribution outputs were not copied into
  the repository checkpoint.

## Repository state

- Checkpoint branch: `main`.
- Remote:
  `origin https://github.com/Zachary200114/reactorbench-lm.git`.
- Last known commit before this checkpoint:
  `81d79e5` (`docs: plan bounded model remediation`).
- The worktree was clean before Iteration 1 began.
- No remote push was performed.

## Immediate next step

After a new explicit user authorization, design a development-only target inventory
that cannot open or materialize historical/fresh held-out rows. Use it to measure
train/validation compact lengths and round trips, then preregister task-specific
generation caps. Do not train a v0.2 model in that same atomic step.

## Exact recommended next command

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
git status --short --branch
```

## Relevant artifact and configuration paths

- Compact compiler/decoder: `src/reactorbench/evaluation/compact.py`.
- Compact contract: `schemas/compact-output/v0/`.
- Remediation program: `docs/model/PHASE6_REMEDIATION_PLAN.md`.
- Historical v0.1 config (immutable):
  `configs/experiments/phase6-main-v0.1.0.toml`.
- Historical v0.1 runs (immutable, ignored): `runs/phase6-main-v0.1.0*`.
- No v0.2 data, config, checkpoint, prediction, or training run exists.

## Exact resume prompt

Resume ReactorBench-LM from the safe checkpoint in
/Users/zachary/Documents/Personal-Projects/AI-transformer.
Read research/PROJECT_REQUIREMENTS.md and docs/IMPLEMENTATION_STATUS.md first.
Inspect Git status and verify the recorded tests before making changes.
Continue from the documented immediate next step without repeating completed work.
