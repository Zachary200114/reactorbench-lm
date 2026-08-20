# ReactorBench-LM implementation status

Last updated: 2026-08-20 America/New_York
Current phase: **Phase 6 in progress — pre-test implementation ready; owner golden review pending**
Current objective: obtain one explicit project-owner decision on the exact G01-G15
packet, then run the already-frozen main, ablation, and held-out evaluation sequence.
Checkpoint reason: the test inventories, golden cases, decoder, cache, uncertainty
metrics, and fail-closed review contracts are implemented without reading a model
prediction from any held-out split.
Intended project path: `/Users/zachary/Documents/Personal-Projects/AI-transformer`

## Completed work

- Reverified the complete Phase 5 report and both safetensors checkpoints before Phase
  6 changes; pilot validation NLL remains `0.15928723867008093`.
- Enumerated all 1,776 approved task examples without model scoring and froze all 894
  held-out records by per-split checksum.
- Prepared the exact 15-case G01-G15 owner-review packet, bound it to generator commit
  `4473718`, and recorded zero findings from the bounded automated prohibited-content
  scanner. The packet is not yet human-approved.
- Added strict golden packet/record contracts, non-overwriting generation, duplicate-key
  rejection, staleness regeneration, project-owner-only approval, and CLI verification.
- Added full-split experiment materialization, paired/corrupted prompt reconstruction,
  cached autoregressive Transformer decoding, strict JSON/task-schema validation,
  deterministic bootstrap intervals, evidence-set F1, ECE, and selective-risk metrics.
- Preregistered the exact decoder and ablation semantics plus the honest E7
  not-applicable result before held-out access.

- Phases 0–2 and the developmental G01–G15 Aster Station generator are complete locally.
- Phase 3 is complete locally. Its owner-approved candidate contains 204 audit
  trajectories, 1,762 single-input projections, 14 counterfactual pairs, 553 rendered
  candidates, 1,776 task examples, and 18 bounded corruption records.
- Phase 4 is complete locally: a project-trained 2,048-token SentencePiece BPE and a
  from-scratch decoder-only causal Transformer passed tokenizer isolation, causal mask,
  shifted-target, padding, deterministic evaluation, tiny-shard overfit, safetensors,
  and independent smoke-verification gates.
- Phase 5 behavior, serialization, validation-only selection, training schedules,
  resource limits, and learning thresholds were preregistered before fitting. Only 630
  `iid_train` examples fit models; only 252 `iid_validation` examples selected
  checkpoints or informed the Phase 6 freeze. Prohibited-split count is zero.
- Implemented majority/frequency, deterministic rules, word/token trigram,
  project-defined bag-of-words softmax regression, project-defined GRU, smaller
  Transformer, and pilot Transformer comparisons.
- The first pilot invocation stopped atomically before fitting because complete targets
  exceeded the Phase 4 128/256 contexts. It wrote no output. Before any result existed,
  both Phase 5 contexts were fixed at 512 to preserve every target; width, depth, seeds,
  optimizer, and step counts remained unchanged.
- The real run completed 10 baseline result rows, a 300-step 724,480-parameter smaller
  Transformer, and a 500-step 5,394,432-parameter pilot Transformer on Apple MPS. An
  independent invocation re-parsed, re-hashed, and reloaded both checkpoints.
- A strict packaged `phase6-main-v0.1.0.toml` now freezes the 15,179,520-parameter
  8×384 model, context 512, batch four, 1,500 MPS steps, E0–E7 matrix, seeded bootstrap,
  validation selection, and numerical capability gates before test evaluation.

## Measured Phase 5 results

- Majority fault/action accuracy: 0.3448 / 0.3448; macro-F1: 0.0641 / 0.0641.
- Deterministic-rule fault/action accuracy: 0.6379 / 0.4828; macro-F1: 0.6626 / 0.4132.
- Next-event majority and word-trigram macro-F1: 0.4444 and 0.8667.
- Token trigram validation NLL/perplexity: 1.0955 / 2.9906.
- Bag-of-words fault accuracy/macro-F1: 0.5000 / 0.2610.
- GRU fault accuracy/macro-F1: 0.6034 / 0.5050; GRU next-event accuracy/macro-F1:
  1.0000 / 1.0000 on only 20 validation examples.
- Smaller Transformer initial/selected validation NLL: 7.6093 / 0.5343, a 92.98%
  reduction. It selected step 300, ran 20.27 seconds, measured 15,335.62 target
  tokens/second, and wrote 3,949,312-byte weights.
- Pilot Transformer initial/selected validation NLL: 7.7257 / 0.1593, a 97.94%
  reduction. It selected step 500, ran 268.70 seconds, measured 1,909.38 target
  tokens/second, and wrote 23,682,552-byte weights.
- Pilot peak current/driver MPS allocation: 143,629,312 / 3,401,547,776 bytes. Recorded
  process peak RSS: 994,131,968 bytes. Thermal throttling was not observable.
- Context 512 preserved every complete target but truncated the oldest prompt prefix in
  491/630 training and 198/252 validation examples. This is a material limitation.

## Files created or changed

- Phase 5 evaluation: `src/reactorbench/evaluation/`.
- Pilot runner/verifier and CLI: `src/reactorbench/training/pilot.py`, training package,
  and `Makefile`.
- Reviewed contracts: `configs/experiments/phase5-pilot-v0.1.0.toml` and
  `configs/experiments/phase6-main-v0.1.0.toml`.
- Phase 5 tests: `tests/unit/test_phase5_*.py` plus package/artifact contract updates.
- Evidence: `docs/model/PHASE5_PILOT.md` and reconciled README, architecture, security,
  research, checklist, acceptance, decision, and reproducibility documents.
- Local ignored evidence: `runs/phase5-pilot-v0.1.0/`.

## Tests and checks run

- Python: CPython 3.12.11.
- Phase 6 pre-test fresh suite: **698 passed in 299.73 seconds** with **85.06%
  branch coverage**. The required 85% floor was not weakened.
- Phase 6 pre-test Ruff format check: **141 files formatted**. Ruff lint passed.
  Strict mypy: **111 source files**, no issues.
- Phase 6 wheel and sdist build passed. Isolated no-network installation verified the
  packaged Phase 6 config, golden packet, schemas, and guard resources byte for byte.
- Intended-repository final suite: **687 passed in 299.71 seconds** with **85.07%
  branch coverage** at the completed Phase 5 checkpoint.
- Ruff format check: **136 files formatted**. Ruff lint: passed.
- Strict mypy: **107 source files**, no issues.
- `git diff --check`: passed.
- Independent `verify-pilot`: `phase5_pilot_passed`, 10 baseline rows, selected pilot
  validation NLL 0.15928723867008093, exact report checksum reproduced.
- Wheel and sdist build passed. Isolated no-network wheel install and byte-for-byte
  packaged config/schema/guard verification passed, including both experiment configs.

## Decisions made

- Preserve complete targets and expand only Phase 5 learned position embeddings to
  context 512 after the first pre-fit sizing failure.
- Report simple-baseline wins honestly; added model complexity is not automatic value.
- Keep the main architecture at 8 layers, width 384, 8 heads, context 512, exactly
  15,179,520 parameters. Use batch four, fixed seed 6601, 1,500 MPS steps, no silent
  CPU fallback, and validation-only checkpoint selection.
- Freeze the numerical and E0–E7 experiment contract from validation evidence before
  test access. Composition has no pass threshold and must be reported even when poor.
- Keep checkpoints safetensors-only and bind reports/configs/checkpoints to data,
  tokenizer, dependency lock, and source checksums.

## Assumptions

- All model inputs remain project-authored, synthetic, fictional, normalized, and
  non-operational.
- The approved Phase 3 candidate/reviews, Phase 4 tokenizer, and Phase 5 run remain
  immutable local inputs.
- Exact account-usage percentage is not observable. No claim is made that a 1% account
  cutoff was measured; conservative phase checkpoints were used.

## Known failures and residual risks

- The 512-token window truncates substantial prompt prefixes, although never targets.
  Phase 6 must classify insufficient-context failures and cannot conceal this rate.
- Both Transformer validation curves selected the final scheduled step. The pilot was
  nearly flat at steps 450–500, but longer-run behavior is still unknown.
- The perfect GRU next-event score uses only 20 validation examples and is not a test or
  generalization result. Rules outperform the GRU on current fault-family macro-F1.
- Apple MPS execution is not guaranteed bitwise deterministic, and thermal throttling
  is not directly observable.
- No IID test, strict holdout, robustness, compositional, calibration, abstention,
  behavioral, or golden-suite model metric has been accessed.
- Golden scenarios remain developmental pending final human review. The Phase 3 content
  guard is non-exhaustive; human approvals reduce but do not eliminate risk.
- Code and data licenses remain `TBD`, blocking distribution but not local work.

## Open blockers

- One blocker remains before Phase 6 main/test work: the project owner must review and
  approve, revise, or reject the exact checksum-bound G01-G15 packet. No approval record
  has been fabricated.
- Main/ablation orchestration and report assembly continue after that decision. No test
  prediction has been accessed.
- Before distribution: choose code/data licenses and complete release/SBOM/security
  gates. GitHub push and Vercel deployment remain owner-managed and unauthorized here.

## Uncommitted work

- Expected handoff state after the documentation closeout commit: no tracked
  modifications. Phase 3–5 data/run artifacts remain ignored local evidence by design.
- No artifact was deleted, pushed, published, or deployed.

## Repository state

- Phase 5 preregistration commit: `566346d`.
- Phase 5 implementation commit: `04652092859a3dce75a10eb9e68d8bc9431c667d`.
- Complete-target sizing correction/source commit:
  `a2c180ec35e1c916f4e5068a7c932c5d6d2fec18`.
- Phase 5 evidence and Phase 6 freeze commit:
  `a58abf4e8c4c3ef0f629fae3a6045f03fb49ae94`.
- Branch: `codex/foundation`.
- No Git remote, push, publication, or deployment exists.

## Immediate next step

Review the compact G01-G15 decision summary and provide an explicit approve, revise, or
reject decision. If approved, write the bound local record, verify it, finish the
remaining Phase 6 orchestration, train the main and ablation models, then evaluate all
held-out splits exactly once.

## Exact recommended next command

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
.venv/bin/python -m reactorbench.training verify-golden-review \
  --packet golden/golden-suite-v0.1.0.json \
  --record artifacts/review/golden-review-record-v0.1.0.json \
  --expected-packet-sha256 c2e966564dadfab7e8b944ca9b6f8ef59d8545d1da1cc4ea75f8b27a9c44077c
```

## Relevant artifacts and configuration

- Phase 5 config canonical/raw SHA-256:
  `5e75ccc0c5dfab1ba1a34c37ba490ff55b53c49d6596d956efd037369bee7bc3` /
  `84a4dfef27bdb30f6ab36cf974dd7b059bc30dc36eb93f6d5db1cc6db8a8868f`
- Phase 5 report internal/raw SHA-256:
  `5c21a4ff93701cdaa73e59e5b9a488cc171009dd1c35ac1f2a234dc7db029ffc` /
  `67d89bf24ea6b3d739b29c4253d7f9cd53843096750961e43d0c2033682eb28a`
- Experiment inventory SHA-256:
  `94b8ea55d804c7a48726a3a135b89aea5f15a471d0b84bb1d19b947d09015fe0`
- Smaller checkpoint manifest/weights SHA-256:
  `917425dad7c6995cb99cc7bc19fc821bbd921dbd98931ee2efd8b502748495ce` /
  `d2a0bd783d82dce684742e40de88d3c8f325900cc65d086f3764867ff0355318`
- Pilot checkpoint manifest/weights SHA-256:
  `8d4b93d29cd1c4becf2062cff8e3eb810259601d1685eb565ced5f5382da2daa` /
  `2b2b60205e77ddf8cedf8c41bfa0761814eb82eeef51db2bb8d586fe5afd319f`
- Phase 6 config canonical/raw SHA-256:
  `be1df0cee9752912b5c317b62fb618b896598906252da25801161e344071b784` /
  `e01a3ac57277f198826476d3bea0d431eb521a3989fec3ab85f42a1139ea1439`
- Golden packet semantic/raw SHA-256:
  `c2e966564dadfab7e8b944ca9b6f8ef59d8545d1da1cc4ea75f8b27a9c44077c` /
  `118720638aeb9d082a6ddc7efd367f3d972c5831a12c6b76f171a10076cc64ea`
- Split-manifest/task-example raw SHA-256:
  `ee01aea896831c90c04e7be324eb05a40341bbc7d752bcf34f9280f7003c8abb` /
  `b45e3466a390b31031a3a39b82046cfef17fd0fb159fa85b97405cbe2ff02cc1`
- Dataset candidate SHA-256:
  `3bba04bdb2030425ef67845332540fa2d148d0a318ab1d9e658f52bb890bf10c`
- Tokenizer manifest SHA-256:
  `ef80afa52030c764598663b0f51b90e7b753b91377b47b4a5648d729e0011ef8`
- Dependency lock SHA-256:
  `dc09fe5e44a7a08f314558b92776f2ca77153842770524ea1818c6a41c6269e6`
- Final wheel SHA-256 / size:
  `f42affa4fb4d2bed236a2125a33df5630c0e6fb05b94fd40d2924d7df0aa7fea` /
  250,440 bytes.
- Final sdist SHA-256 / size:
  `532eecd9b6e514386b3ea683691f66f6f05df4c86b9fb23a18a663df730dc522` /
  197,080 bytes.

## Exact resume prompt

Resume ReactorBench-LM from the safe checkpoint in
/Users/zachary/Documents/Personal-Projects/AI-transformer.
Read research/PROJECT_REQUIREMENTS.md and docs/IMPLEMENTATION_STATUS.md first.
Inspect Git status and verify the recorded tests before making changes.
Continue from the documented immediate next step without repeating completed work.
