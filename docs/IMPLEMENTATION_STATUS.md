# ReactorBench-LM implementation status

Last updated: 2026-08-20 America/New_York
Current phase: **Phase 4 complete locally — tokenizer and model-correctness gate passed**
Current objective: preserve the verified Phase 4 evidence and begin Phase 5 only with a
read-only baseline/pilot design audit.
Checkpoint reason: the approved Phase 3 candidate produced a project-trained tokenizer
and a from-scratch smoke Transformer that passed the preregistered overfit, causal-mask,
deterministic-evaluation, and safe checkpoint reload gates.
Intended project path: `/Users/zachary/Documents/Personal-Projects/AI-transformer`

## Completed work

- Phases 0–2 and the developmental G01–G15 Aster Station generator are complete locally.
- Phase 3 is complete locally. Its project-owner-approved candidate contains 204 audit
  trajectories, 1,762 single-input projections, 14 counterfactual pairs, 553 distinct
  rendered candidates, 1,776 task examples, and 18 bounded corruption records. It is
  approved for local research use, not public release.
- The tokenizer corpus is derived only from the approved `iid_train` render split: 195
  documents, 685,978 UTF-8 bytes, corpus SHA-256
  `e8433ec549df79d274ebee6ffa32f1fe7810df3db256a7db5bf817dac4ccdc6e`.
  Validation, IID test, template, component, severity, composition, counterfactual, and
  noise holdouts never enter tokenizer training.
- A deterministic project SentencePiece BPE tokenizer was trained with vocabulary 2,048,
  identity normalization, byte fallback, and fixed IDs `UNK=0`, `BOS=1`, `EOS=2`,
  `PAD=3`. Project symbols are `<|prompt|>`, `<|target|>`, and `<|sep|>`.
- The decoder-only causal Transformer is project-defined from PyTorch primitives: token
  and learned position embeddings, explicit multi-head masked self-attention, pre-norm
  residual blocks, GELU feed-forward layers, tied output embeddings, and random normal
  initialization. No pretrained weight, hosted LLM, model hub, or remote data source is
  used.
- Reviewed model tiers calculate to exactly 675,328 smoke parameters, 5,328,896 pilot
  parameters, and 15,179,520 main parameters with the 2,048-token vocabulary.
- The real Phase 4 smoke run used four approved training documents, batch size four,
  context 128, 508 scored target tokens per step, seed 4404, and 300 CPU steps. Loss
  changed from `7.6617279052734375` to `0.011601113714277744`, a measured reduction of
  `0.998485835850906`. The timed loop took `3.8152835840010084` seconds and measured
  `39,944.60612025628` target tokens/second.
- The smoke run passed future-token isolation, exact one-token target shifting,
  padding-mask loss exclusion, deterministic repeated evaluation, tiny-shard overfit,
  safetensors save/reload logit equality, checksum verification, and non-overwriting
  artifact publication.
- The checkpoint loader accepts only the fixed `manifest.json` plus
  `model.safetensors` inventory. It rejects symlinks, unknown files, oversized data,
  wrong tokenizer bindings, manifest mismatch, and weight checksum/size mismatch. It
  never accepts pickle or a user-supplied checkpoint path through the Phase 4 CLI.
- `make phase4-smoke`, `make phase4-verify`, and `make reproduce-smoke` are implemented.
  The config-selected run is local and ignored by Git; it does not publish or deploy.

## Files created or changed

- Model code: `src/reactorbench/model/`.
- Tokenizer/corpus boundary: `src/reactorbench/tokenizer/`.
- Smoke orchestration and narrow CLI: `src/reactorbench/training/`.
- Reviewed config: `configs/model/phase4-smoke-v0.1.0.toml`.
- Locked dependencies: `pyproject.toml` and `uv.lock` now include NumPy, PyTorch,
  SentencePiece, and safetensors.
- Package resources/build verification: `src/reactorbench/resources.py`,
  `tests/contract/test_package_resources.py`, and
  `tests/contract/verify_distribution_artifacts.py`.
- Phase 4 unit tests: `tests/unit/test_phase4_*.py` and
  `tests/unit/test_transformer.py`.
- Phase 4 evidence: `docs/model/PHASE4_SMOKE.md` plus the reconciled README,
  architecture, security, checklist, decision, and reproducibility documents.

## Tests and checks run

- Python baseline: CPython 3.12.11.
- Intended-repository final suite: **677 passed in 300.44 seconds** with **85.37%
  branch coverage**; the required floor remains 85% and was not weakened.
- Ruff format check: **123 files formatted**. Ruff lint: passed.
- Strict mypy: **96 source files**, no issues.
- `git diff --check`: passed.
- Compile check passed with `PYTHONPYCACHEPREFIX` redirected to `/private/tmp`; the
  initial compile-only attempt was blocked only because the sandbox disallowed writing
  external-repository `__pycache__` files.
- Wheel and sdist build passed. Isolated local wheel install and byte-for-byte packaged
  config/schema/guard resource verification passed.
- Real smoke run status: `phase4_smoke_passed`. A separate read-only
  `verify-smoke` invocation re-parsed, re-hashed, reloaded, and reproduced the exact
  evaluation-logit hash.
- PyTorch 2.13.0 reports MPS support compiled in, but MPS was unavailable to the current
  sandboxed process. No MPS throughput or memory result is claimed; Phase 5 must measure
  the actual local backend before selecting pilot settings.

## Decisions made

- Use deterministic project-trained SentencePiece BPE with vocabulary 2,048 for the
  Phase 4 path; an optional custom BPE comparison remains deferred.
- Keep smoke training on CPU for deterministic correctness evidence. MPS is a measured
  Phase 5 benchmark, not a Phase 4 assumption.
- Keep the smoke/pilot/main configurations at 2×128, 6×256, and 8×384 respectively,
  with contexts 128/256/512. Pilot evidence may still change provisional main settings
  before the experiment freeze.
- Store model weights only as safetensors and bind the checkpoint to its tokenizer,
  approved corpus/candidate, source commit, configuration, seed, and checksums.
- Bind the smoke report to the exact `uv.lock` hash and record NumPy, Pydantic, PyTorch,
  SentencePiece, and safetensors versions.
- Tiny-shard overfit is a correctness proof, not evidence of generalization or model
  usefulness. No Phase 5 baseline, validation, holdout, or benchmark claim has been
  made.

## Assumptions

- All model inputs remain project-authored, synthetic, fictional, normalized, and
  non-operational.
- The Phase 3 candidate and both owner approvals remain immutable local inputs.
- Exact account-usage percentage is not observable. No claim is made that a 1% account
  cutoff was measured; conservative phase checkpointing was used.

## Known failures and residual risks

- MPS is compiled into the installed Torch build but unavailable inside this process;
  local MPS operation, memory, thermal throttling, and throughput are unmeasured.
- The smoke model deliberately memorizes four short training prefixes. It has not been
  evaluated on validation, IID test, compositional, robustness, behavioral, abstention,
  or golden cases.
- The golden suite, dataset/task schemas, and experiment thresholds remain
  developmental and unfrozen.
- The Phase 3 content guard remains non-exhaustive. Human approvals reduce risk but do
  not prove the absence of all prohibited or source-derived material.
- Code and data licenses remain `TBD`, blocking distribution but not local work.
- The superseded first smoke run was preserved under
  `runs/phase4-smoke-v0.1.0-pre-lock-binding-882c8d9`; it is not the config-selected
  final artifact and must not be cited.

## Open blockers

- No blocker remains for beginning a read-only Phase 5 baseline/pilot audit.
- Before pilot training, preregister exact baseline behavior, task serialization,
  validation-only checkpoint selection, throughput/memory measurement, stopping rules,
  and pilot acceptance thresholds.
- Before main experiments, freeze the approved manifests, golden cases, split/task
  contracts, acceptance thresholds, and test-use policy.
- Before distribution, choose code/data licenses and complete release/SBOM/security
  gates. GitHub push and Vercel deployment remain owner-managed and unauthorized here.

## Uncommitted work

- No implementation code is uncommitted. This documentation reconciliation is the
  final atomic Phase 4 closeout edit; the expected handoff state after its commit is a
  clean tracked worktree.
- Approved Phase 3 data/review artifacts and Phase 4 run artifacts remain ignored local
  evidence by design. They are not deleted, published, or added to Git.

## Repository state

- Phase 3 closeout commit: `1d2401b8a56ae06b151b70d4cd912547af77f2c0`.
- Phase 4 implementation commit: `882c8d9c13a62373f45080864f595748e2bc1db9`.
- Phase 4 dependency-provenance hardening commit:
  `f636f2b2b9f7f5915bef78903611aa047aaecc30`.
- Branch: `codex/foundation`.
- No Git remote, push, publication, or deployment exists.

## Immediate next step

Begin Phase 5 with a read-only audit. Verify the Phase 4 run, then preregister baseline
contracts and a bounded pilot benchmark before implementing or training anything. Do
not run test splits, main training, inference, or UI work.

## Exact recommended next command

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
git status --short --branch
.venv/bin/python -m reactorbench.training verify-smoke \
  --config configs/model/phase4-smoke-v0.1.0.toml
```

## Relevant artifacts and configuration

- Phase 4 config: `configs/model/phase4-smoke-v0.1.0.toml`
- Phase 4 config canonical SHA-256:
  `af55961930cf856e9953d9b27f2d3f270307bc3bb55e60e73bd31b53cae4bee9`
- Phase 4 config raw SHA-256:
  `05b73499995d13073942f6396d37401fe34c8fab100d9ce9406d94fb6bef439c`
- Dependency lock raw SHA-256:
  `dc09fe5e44a7a08f314558b92776f2ca77153842770524ea1818c6a41c6269e6`
- Approved Phase 3 candidate bundle SHA-256:
  `3bba04bdb2030425ef67845332540fa2d148d0a318ab1d9e658f52bb890bf10c`
- Approved post-render review record internal SHA-256:
  `e066d5944839423fdd6e49491dfa5b57867b0c753ac465afb4ac196e7a87958d`
- Training-corpus SHA-256:
  `e8433ec549df79d274ebee6ffa32f1fe7810df3db256a7db5bf817dac4ccdc6e`
- Final run directory (ignored): `runs/phase4-smoke-v0.1.0`
- Tokenizer manifest internal SHA-256:
  `ef80afa52030c764598663b0f51b90e7b753b91377b47b4a5648d729e0011ef8`
- Tokenizer model SHA-256 / size:
  `b2ced4e9699f019a053516c9ff4a6c698d1bd17f9b070632ac3e03b565a2af6c`
  / 33,586 bytes.
- Tokenizer vocabulary SHA-256 / size:
  `a1eb163398d042f2d5741d2a96fa2520a879a3bb887c6ab028889ea112a02e91`
  / 26,693 bytes.
- Checkpoint manifest internal SHA-256:
  `0d969c046e95f76c270b6f6d01a8331ef540a2bf52ab2d15ac57b566c697260e`
- Safetensors weights SHA-256 / size:
  `f085fff20bfe14df785b23a43630516228890aa38868cafb3ef478ac30cac241`
  / 3,752,704 bytes.
- Smoke input SHA-256:
  `52cfcc1b5e288c85d0aac6f075f14e216397f20d2082cc77f10f2d940a1f0548`
- Evaluation-logit SHA-256:
  `e1f726f6a89e6dd987c30f496008a49d2c3d74af672444097f92c91975fc2d5a`
- Smoke report internal SHA-256:
  `b2b28c6b4d95661ef67698f5a2eb4752c2c5f8d26c7541a503f3e71ddecd4683`
- Smoke report raw SHA-256:
  `9be65be363bcc8b02bd04b49ca1dd08615f22fd5eac615fa1d29239674fa095b`
- Built wheel SHA-256 / size:
  `74507469fc60404245c29ca20642a572bdb5d3c7a7f8f4a0f393385b127723b4`
  / 207,751 bytes.
- Built source distribution SHA-256 / size:
  `b7cd43432ada840cccfdc4772d88de295029bf4bffb18a15466822c957fbc024`
  / 164,019 bytes.

## Exact resume prompt

Resume ReactorBench-LM from the safe checkpoint in
/Users/zachary/Documents/Personal-Projects/AI-transformer.
Read research/PROJECT_REQUIREMENTS.md and docs/IMPLEMENTATION_STATUS.md first.
Inspect Git status and verify the recorded tests before making changes.
Continue from the documented immediate next step without repeating completed work.
