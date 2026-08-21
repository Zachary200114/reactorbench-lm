# ReactorBench-LM implementation status

Last updated: 2026-08-20 America/New_York
Current phase: **Phase 6 remediation planning complete; implementation not started; Phase 7 blocked**
Current objective: after the user's usage reset, implement only the v0.2 compact target
and constrained-decoding correctness milestone described in the remediation plan.
Checkpoint reason: the user requested planning only at a self-reported 27% remaining
usage. The three-iteration remediation program is now preregistered without changing
model code, generating data, training, or accessing any held-out record.
Intended project path: `/Users/zachary/Documents/Personal-Projects/AI-transformer`

## Completed work

- Phases 0-3 are complete locally: the strict Aster Station G01-G15 generator,
  projection/split/renderer pipeline, owner-approved synthetic dataset, and provenance
  chain are implemented and verified.
- Phase 4 is complete locally: a 2,048-token project SentencePiece BPE and a
  from-scratch decoder-only Transformer passed masking, shifted-target, padding,
  deterministic evaluation, tiny-shard overfit, safetensors, and reload gates.
- Phase 5 is complete locally: all preregistered majority, rule, n-gram, bag-of-words,
  GRU, smaller-Transformer, and pilot comparisons ran using only `iid_train` for fit
  and `iid_validation` for selection.
- The project owner approved the checksum-bound G01-G15 packet with all seven required
  confirmations. The review record is local, strict, and independently verified.
- Phase 6 validation-only selection trained E3 main, E5 renderer-diversity, and E6
  abstention-data models. The 15,179,520-parameter E3 checkpoint selected step 1,400
  with validation NLL 0.073041; every preregistered selection threshold passed.
- Exactly one held-out access evaluated 894 frozen examples across seven splits plus
  60 approved golden task examples. All E0-E7 results, simple baselines, learned
  comparisons, raw predictions, uncertainty metrics, and negative results were saved.
- An evaluator defect was diagnosed after the first report: generated supervised
  targets correctly ended with `\n<|sep|>`, but the parser treated that frozen transport
  delimiter as JSON. The original report/predictions were preserved unchanged.
- Commit `e6504695583403f7e31b118f99ce67c6873bd8e8` fixes future parsing and adds a
  versioned one-time rescore. The corrected graph reuses every generated string byte
  for byte, generates no new token, recomputes valid-output confidence from stored
  tokens, preserves the one-access ledger, and passed independent reconstruction.
- Phase 6 is closed honestly as a negative experiment. No acceptance threshold was
  changed, no test result selected a checkpoint, and no post-test retraining occurred.
- The planning-only remediation program is documented as three gated iterations:
  v0.2 output reliability, v0.3 semantic learning/abstention, and v0.4 strict
  generalization/final evaluation. Each iteration is capped at one control plus two
  preregistered variants, and v0.1 held-outs are report-only.
- No GitHub push, remote publication, Vercel deployment, or external service was used.

## Measured Phase 6 results

- Selection report semantic SHA-256:
  `29a1c07864e24b47e7928df05d5738834ea10879ec46ff91f2834fae7113379d`.
- Corrected evaluation report semantic SHA-256:
  `fb1ee4e13ba8ca44116641e5892ff3a7eef523846cd907fe07f368a76e09a0ce`.
- Main IID teacher-forced target NLL/perplexity: 0.042540 / 1.043458.
- Main IID parse/schema/exact: 21.03% / 5.16% / 5.16% (13/252 exact; deterministic
  95% bootstrap interval 2.38%-7.94%).
- Template/component/severity/composition/counterfactual exact match: all 0%.
- Noise exact match: 4.17% (2/48; interval 0.00%-10.42%).
- Golden exact match: 3.33% (2/60), both from G15 sparse-evidence tasks.
- IID fault/action/continuation macro-F1: 0.0000 / 0.0290 / 0.2029.
- IID required-abstention accuracy: 6.67% (4/60); component, severity, composition,
  counterfactual, and template required-abstention accuracy: 0%.
- IID evidence F1: 0.0238. Selective risk at approximately 80% coverage: 0.9356 IID
  and at least 0.9487 on every other split.
- Four composition and ten counterfactual targets are
  `INSUFFICIENT_CONTEXT_BY_DESIGN` rather than silently truncated.
- Ten acceptance checks pass numerically; 23 fail. Passed calibration/no-fault numeric
  checks are vacuous because invalid outputs receive confidence zero and diagnosed
  schema-valid predictions are absent.
- Strong simple comparators include IID rule fault macro-F1 0.6227, template GRU fault
  macro-F1 0.6414, severity rule fault macro-F1 1.0000, and several continuation
  macro-F1 scores of 1.0000. The main model fails every required simple-margin gate.

## Files created or changed

- Permanent correction:
  `src/reactorbench/evaluation/decoding.py`,
  `src/reactorbench/training/main.py`,
  `tests/unit/test_phase6_evaluation.py`, and `Makefile`.
- One-time, source-controlled rescore:
  `scripts/phase6_rescore_v0_1_1.py`.
- Phase 6 evidence documentation:
  `docs/model/PHASE6_MAIN.md`, `docs/model/PHASE6_PRETEST.md`, this status file,
  README, acceptance plan, checklist, decision log, and reproducibility plan.
- Remediation planning evidence:
  `docs/model/PHASE6_REMEDIATION_PLAN.md` and decision D-071.
- Original ignored evidence:
  `runs/phase6-main-v0.1.0-selection/`, `runs/phase6-main-v0.1.0/`, and
  `runs/phase6-main-v0.1.0-heldout-access.json`.
- Corrected ignored evidence:
  `runs/phase6-main-v0.1.0-rescore-v0.1.1/`.

## Tests and checks run

- Python: CPython 3.12.11.
- Final repository suite after the parser correction: **713 passed in 300.74 seconds**
  with **85.10% branch coverage**; the 85% floor was not weakened.
- Focused Phase 6 parser/orchestration suite: **20 passed**.
- Ruff format check: **144 files formatted**. Ruff lint passed.
- Strict mypy: **113 source files**, no issues. The standalone rescore script also
  passes strict mypy with the local source tree selected.
- `git diff --check`: passed before the correction commit.
- Phase 6 selection independently verified.
- Original Phase 6 report and all three original prediction artifacts independently
  verified at report checksum
  `2009afdeae9247125a30b4afcc24b41409ffb6ad3afd64d2772c2a491fc55967`.
- Corrected rescore independently reconstructed exactly at report checksum
  `fb1ee4e13ba8ca44116641e5892ff3a7eef523846cd907fe07f368a76e09a0ce`.
- The final intended repository built the wheel and sdist successfully and passed the
  isolated no-network artifact verifier after documentation closeout. The Make target
  could not locate `uv` on the shell PATH, so the exact underlying locked-venv Python
  commands were run and passed.
- Planning-only checkpoint: `git diff --check`, trailing-whitespace scan, documentation
  link/path inspection, and docs-only diff review passed. No model/code test was rerun
  because this checkpoint changes documentation only.

## Decisions made

- Preserve the frozen E3 architecture, 1,500-step schedule, selected step, decoder,
  metrics, thresholds, and all negative results after held-out access.
- Classify the delimiter issue as an evaluator defect, preserve the faulty original
  report, and issue a versioned mechanical rescore rather than rerun generation or
  hide the incident.
- Treat low teacher-forced NLL and poor free-running structured generation as a real
  exposure/sequence-level gap. Token likelihood alone is not a deployability result.
- Close Phase 6 as executed but failed. Do not start Phase 7 inference/UI around this
  checkpoint, because the prompt explicitly gates Phase 7 on a stable trained
  checkpoint and inference contract.
- Any improvement cycle must be preregistered as v0.2 and use training/validation
  evidence for design choices. These held-out results may be reported and analyzed but
  not repeatedly optimized against.
- Separate remediation into structural output reliability, semantic behavior, and
  strict generalization so a later improvement can be attributed to a bounded change.
- Require a fresh, checksum-frozen final holdout and versioned golden extension for
  v0.4. The already accessed v0.1 tests cannot become a clean selection gate again.

## Assumptions

- All inputs remain project-authored, synthetic, fictional, normalized, and
  non-operational.
- Original and corrected local artifacts remain immutable and available at their
  recorded paths.
- Exact account-usage percentage is not observable. No claim is made that a 1% account
  cutoff was measured.

## Known failures and residual risks

- Free-running JSON generation is the dominant failure: only 16/894 corrected main
  outputs are schema-valid.
- Fault-family identification produces no correct schema-valid held-out prediction.
- Robustness, component, severity, composition, and counterfactual exact match are 0%.
- Golden exact match is only 2/60 and does not support a behavioral capability claim.
- The 512-token window truncated prompt prefixes in training/validation and 788/894
  held-out generations; ten counterfactual and four composition targets cannot fit by
  design.
- MPS is not guaranteed bitwise deterministic, and thermal throttling was not directly
  observable.
- The content guard remains bounded and non-exhaustive; human review reduces but does
  not eliminate content risk.
- Code/data licenses remain `TBD`, blocking distribution.

## Open blockers

- Phase 7 inference/UI is blocked by the failed Phase 6 model gate.
- The user authorized remediation planning only. Implementation/training remains
  intentionally unstarted until a later resume after usage reset.
- Before any distribution: choose code/data licenses and complete release, SBOM,
  security, accessibility, clean-environment, and deployment gates.
- GitHub push and Vercel deployment remain owner-managed and unauthorized here.

## Uncommitted work

- No source or documentation work is intentionally left uncommitted at this
  checkpoint. Confirm with `git status --short` on resume.
- All datasets, checkpoints, review records, and run artifacts remain ignored local
  evidence. No artifact was deleted or overwritten.

## Repository state

- Branch: `codex/foundation`.
- Phase 6 pretest implementation commit:
  `56bce54c00bf7d16e62f34b49387b88fa5e3906b`.
- Phase 6 main runner/source commit:
  `d8dc2936b1f9355d4246a7be2a297ad210f1cbfe`.
- Phase 6 parser correction/rescore commit:
  `e6504695583403f7e31b118f99ce67c6873bd8e8`.
- Phase 6 documentation closeout commit:
  `c67bc20db4bba46970366e71dfd567d714826fb1`.
- The planning-only checkpoint is committed locally at turn completion; use
  `git rev-parse HEAD` for its non-self-referential identifier.
- No Git remote, push, publication, or deployment exists.

## Immediate next step

After usage reset, implement only the v0.2 compact target contract, strict compiler,
truth-independent constrained decoder, and their correctness tests. Do not generate
new data, train a model, or access held-outs in that first atomic milestone.

## Exact recommended next command

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
sed -n '1,320p' docs/model/PHASE6_REMEDIATION_PLAN.md
```

## Relevant artifacts and configuration

- Phase 6 config canonical/raw SHA-256:
  `be1df0cee9752912b5c317b62fb618b896598906252da25801161e344071b784` /
  `e01a3ac57277f198826476d3bea0d431eb521a3989fec3ab85f42a1139ea1439`.
- Golden packet semantic/raw SHA-256:
  `c2e966564dadfab7e8b944ca9b6f8ef59d8545d1da1cc4ea75f8b27a9c44077c` /
  `118720638aeb9d082a6ddc7efd367f3d972c5831a12c6b76f171a10076cc64ea`.
- Golden review record semantic/raw SHA-256:
  `1f5307889d259cfb0fa39e86e33ed9c2ce0922742e59af1d5ff5e0c904337288` /
  `9105c6e7e76979fdbc8b4a73d42f323acb636d36affbee13f8147dc23e3f06be`.
- Selection report semantic/raw SHA-256:
  `29a1c07864e24b47e7928df05d5738834ea10879ec46ff91f2834fae7113379d` /
  `059504a13748c2c9131441782217b980fcee19fbebaf4d8157ef17b532d09552`.
- Original report semantic/raw SHA-256:
  `2009afdeae9247125a30b4afcc24b41409ffb6ad3afd64d2772c2a491fc55967` /
  `2b0a78a703c17810c87e5908e1034c72056315e40a9147e5139a8b2499034b44`.
- Corrected report semantic/raw SHA-256:
  `fb1ee4e13ba8ca44116641e5892ff3a7eef523846cd907fe07f368a76e09a0ce` /
  `11306d82324b190bb06cb96d96137ef399bebccc0f65591e15e9ff2782d0aa22`.
- Correction record semantic/raw SHA-256:
  `04077738a5fb915ea818ee2aadbda320769c6721476e9bc304fdcace96b4cd81` /
  `75834c089400959f8a258b786dee1c6c12067fe7f1c7fdb39934b518a67c2aeb`.
- Corrected main/comparison/golden prediction raw SHA-256:
  `debcb28432ed15b407d159bc966d01d642b00b01c9c22a39db7b219d019df854` /
  `4e4d851e8f253a1c8fe4dc88b40b34b8ed95b61a2c630e1fba1e1f15ee6d8cf3` /
  `92bbb2efbb681462808902c5b0e33e2227fccfd922c164fcb3b3ce2ed1e646ab`.
- Held-out access record raw SHA-256:
  `410ab34416d860b5cc3b6067cfc599f691ad199a986d56f019e81b75246ae74f`.
- Planning contract:
  `docs/model/PHASE6_REMEDIATION_PLAN.md` and D-071 in
  `research/DECISION_LOG.md`.

## Exact resume prompt

Resume ReactorBench-LM from the safe checkpoint in
/Users/zachary/Documents/Personal-Projects/AI-transformer.
Read research/PROJECT_REQUIREMENTS.md and docs/IMPLEMENTATION_STATUS.md first.
Inspect Git status and verify the recorded tests before making changes.
Continue from the documented immediate next step without repeating completed work.
