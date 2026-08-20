# Threat model

Status: Phase 3 data-pipeline and Phase 4 local tokenizer/checkpoint controls are
implemented and verified. Future evaluation, browser, inference, and deployment
boundaries remain planned.

## Scope and assumptions

The project produces only project-authored, wholly fictional Aster Station trajectories.
No real plant, Navy, restricted, operational, proprietary, user-supplied, or hosted-LLM
corpus material is permitted. External publication and deployment remain owner-managed.

```text
reviewed config/source -> strict generator -> split-first ModelInput/projection manifests
    -> hash-bound human reviews -> approved local candidate -> IID-train tokenizer corpus
    -> project tokenizer -> random-init model -> checksummed safetensors/report
untrusted browser -> future gateway -> future inference -> checksummed artifacts
```

The generator, developmental dataset, both human reviews, tokenizer boundary, model
correctness proof, and local checkpoint boundary are implemented. Browser, gateway,
and inference boundaries do not exist, and this document does not claim their controls
are live.

## Current risks and gates

| ID | Risk | Current mitigation | Remaining gate |
|---|---|---|---|
| TM-01 | Prohibited or real-world material enters the corpus | Synthetic-only policy, disclaimer, project-authored catalogs, non-exhaustive scans, and both completed hash-bound owner reviews | Preserve review/scan gates for any changed corpus; scanners and reviews cannot prove universal absence |
| TM-02 | Truth, IDs, or later outcomes leak into model input | Strict `ModelInput` projection strips IDs, evidence annotations, targets, latent/provenance truth, and later effects; dedicated rejection tests | Preserve the same contract in tokenizer/model/service boundaries |
| TM-03 | Invalid/unbounded contracts bypass validation | Strict immutable Pydantic projection/manifest/review/artifact models, duplicate-key/non-finite JSON rejection, and packaged developmental snapshots | Human review and later version-1 freeze; snapshot review is not freeze |
| TM-04 | Unsafe local path/artifact input | Config-selected contained paths; symlink/nonfile checks; no overwrite; explicit size/count limits; Phase 4 accepts no user URL/checkpoint/output root; safe data-only checkpoint loader | Preserve allowlists and add service-side limits before inference |
| TM-05 | Dataset split contamination or nondeterminism | Split-first manifest, disjoint seeds, template/alias/composition holdouts, atomic counterfactual groups, exact/skeleton/n-gram audits | Freeze manifests after pilot evidence and rerun the complete audit on every candidate |
| TM-06 | Model/rendered text causes unsafe display behavior | No UI exists | Encode as text, forbid raw HTML, add CSP and browser tests in Phase 7 |
| TM-07 | Secrets/logs/errors expose sensitive material | No service credentials or routes exist | Server-only secrets, minimal logs, safe errors, secret scanning before deployment |
| TM-08 | Artifact/model replacement changes results | Candidate lineage plus tokenizer, corpus, config, source, dependency-lock, safetensors, input, logit, and report hashes; strict independent verification | Add release allowlist/signature and startup tamper checks before serving |
| TM-09 | Release/dependency compromise | Local lockfile and package build verification | CI hardening, dependency review, SBOM, release allowlist |
| TM-10 | Public output is mistaken for real advice | Prominent fictional/non-operational disclaimer | Curated UI, limitation review, and release evidence |
| TM-11 | A renderer corruption becomes a label shortcut | Corruptions are audit-only lineage, label invariant, applicability checked, and balanced across multiple target outcomes | Human review of the exact corrupted prose and ongoing shortcut tests |
| TM-12 | A test fixture is mistaken for owner approval | Production generation verifies a `project-owner` record bound to the exact packet and structured bundle; docs label fixture output as test-only | Project owner creates both real review records; never reuse a test record |
| TM-13 | A feature or feature interaction is a label shortcut only within one task and is hidden by aggregate/marginal counts | Per-task marginal contingencies plus pairwise/full renderer-plan `renderer_nuisance` interactions; six held-out-alias continuation examples removed and no component-test continuation claim retained; `semantic_context` reported separately; five plan overrides keyed only by split/seed/case, never runtime target | Rerun whenever task projection, split, alias, corruption, or renderer plans change |
| TM-14 | Evidence targets cite absent facts or silently erase supported evidence | Every one of 405 evidence targets is nonempty and resolves to visible prompt-local facts; four map-withheld omissions are explicit and tested | Reaudit any new evidence kind or context policy |

## Implemented data and model-artifact controls

The audit payload is truth-filtered, but is not itself a model prompt. The implemented
projection exposes only allowed event/channel/context facts through the task cut. G07,
G08/G09, G12, and G14 relatives are grouped atomically. G15 sparse groups are explicitly
incomplete because the proposed expanded siblings are unsupported; the system does not
fabricate them or promote them to counterfactual comparisons.

The pre-render packet contains the 176 catalog combinations plus every actual authored
renderer and corruption language surface. It binds those surfaces, catalog, and guard
to the exact resolved configuration, generator commit, structured bundle, split
manifest, and complete target inventory. A `project-owner` record approving that exact
binding is required before generation, and the resulting candidate inventory must
receive a separate post-render human review. Both exact reviews completed on
2026-08-20, and the local candidate passed typed reconstruction. The non-exhaustive
scanners may reduce accidental prohibited strings but cannot prove safety.

The tokenizer re-verifies that chain and selects only `iid_train`. The checkpoint is a
fixed checksum-bound safetensors inventory, never pickle, and the CLI accepts no
arbitrary checkpoint path. The smoke report binds its source commit, dependency lock,
data/corpus/tokenizer/checkpoint, input tensors, and evaluation logits. A separate
read-only command verifies and reloads the complete chain.

## Verified local evidence

At Phase 4 provenance commit
`f636f2b2b9f7f5915bef78903611aa047aaecc30`, the Python 3.12.11 gate passed Ruff,
strict mypy for 96 source files, 677 tests in 300.44 seconds with 85.37% branch
coverage, package build, and isolated artifact verification. The config-selected smoke
run and independent verifier passed. This does not verify model generalization,
holdout performance, a service, UI, deployment, or public release.

The fresh independent SHIP review passed 58 focused tests in 173.95 seconds. Its
placeholder-commit fixture
reports 402 task-scoped contingencies (358 `renderer_nuisance`, 44
`semantic_context`) and zero shortcut, exact-text, forbidden-skeleton, target-text, or
provenance findings over all 1,776 tasks. This does not replace human review.
