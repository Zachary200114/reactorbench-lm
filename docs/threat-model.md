# Threat model

Status: Phase 3 local data-pipeline controls are technically complete through the
mandatory owner-review checkpoint on 2026-08-20, and the intended-repository
verification gate passed. Future browser, inference, and deployment boundaries are
planned only.

## Scope and assumptions

The project produces only project-authored, wholly fictional Aster Station trajectories.
No real plant, Navy, restricted, operational, proprietary, user-supplied, or hosted-LLM
corpus material is permitted. External publication and deployment remain owner-managed.

```text
reviewed config/source -> strict generator -> split-first ModelInput/projection manifests
    -> hash-bound human catalog review -> pending candidate -> human candidate review
    -> future tokenizer/model/artifacts
untrusted browser -> future gateway -> future inference -> checksummed artifacts
```

The generator and developmental dataset boundaries are implemented locally. The two
human data reviews are not complete. Browser/gateway/inference boundaries do not exist
and this document does not claim their controls are live.

## Current risks and gates

| ID | Risk | Current mitigation | Remaining gate |
|---|---|---|---|
| TM-01 | Prohibited or real-world material enters the corpus | Synthetic-only policy, disclaimer, project-authored catalogs, non-exhaustive denylist/pattern/fingerprint scans | Hash-bound pre-render and complete post-render human review; scanners cannot prove safety |
| TM-02 | Truth, IDs, or later outcomes leak into model input | Strict `ModelInput` projection strips IDs, evidence annotations, targets, latent/provenance truth, and later effects; dedicated rejection tests | Preserve the same contract in tokenizer/model/service boundaries |
| TM-03 | Invalid/unbounded contracts bypass validation | Strict immutable Pydantic projection/manifest/review/artifact models, duplicate-key/non-finite JSON rejection, and packaged developmental snapshots | Human review and later version-1 freeze; snapshot review is not freeze |
| TM-04 | Unsafe local path/artifact input | Config must resolve inside a validated checkout; sole write command targets fixed `data/generated/<artifact_name>`; no arbitrary output root; bounded local inputs, symlink/nonfile checks, no overwrite, explicit file/record/byte limits, no user URL/checkpoint input | Safe data-only checkpoint loading and checksum tests before models/services |
| TM-05 | Dataset split contamination or nondeterminism | Split-first manifest, disjoint seeds, template/alias/composition holdouts, atomic counterfactual groups, exact/skeleton/n-gram audits | Freeze manifests after pilot evidence and rerun the complete audit on every candidate |
| TM-06 | Model/rendered text causes unsafe display behavior | No UI exists | Encode as text, forbid raw HTML, add CSP and browser tests in Phase 7 |
| TM-07 | Secrets/logs/errors expose sensitive material | No service credentials or routes exist | Server-only secrets, minimal logs, safe errors, secret scanning before deployment |
| TM-08 | Artifact/model replacement changes results | Candidate manifest binds config, structured/split/schema/catalog/guard/review/report roots; verifier strictly re-parses every typed file and checks cross-file links | Create candidate/post-render hashes only after mandatory owner review; add pinned data-only model artifacts and startup tamper tests before serving |
| TM-09 | Release/dependency compromise | Local lockfile and package build verification | CI hardening, dependency review, SBOM, release allowlist |
| TM-10 | Public output is mistaken for real advice | Prominent fictional/non-operational disclaimer | Curated UI, limitation review, and release evidence |
| TM-11 | A renderer corruption becomes a label shortcut | Corruptions are audit-only lineage, label invariant, applicability checked, and balanced across multiple target outcomes | Human review of the exact corrupted prose and ongoing shortcut tests |
| TM-12 | A test fixture is mistaken for owner approval | Production generation verifies a `project-owner` record bound to the exact packet and structured bundle; docs label fixture output as test-only | Project owner creates both real review records; never reuse a test record |
| TM-13 | A feature or feature interaction is a label shortcut only within one task and is hidden by aggregate/marginal counts | Per-task marginal contingencies plus pairwise/full renderer-plan `renderer_nuisance` interactions; six held-out-alias continuation examples removed and no component-test continuation claim retained; `semantic_context` reported separately; five plan overrides keyed only by split/seed/case, never runtime target | Rerun whenever task projection, split, alias, corruption, or renderer plans change |
| TM-14 | Evidence targets cite absent facts or silently erase supported evidence | Every one of 405 evidence targets is nonempty and resolves to visible prompt-local facts; four map-withheld omissions are explicit and tested | Reaudit any new evidence kind or context policy |

## Implemented Phase 3 controls and remaining review

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
receive a separate post-render human review. The non-exhaustive scanners may reduce
accidental prohibited strings but cannot prove safety. No real owner review is recorded
and no narrative candidate or approved corpus exists.

## Verified local evidence

At verified implementation commit
`d3d22b7f9b2888d281c1c92cd283e10b4f0e3af1`, the intended-repository Python 3.12.11
gate passed Ruff formatting for 106 files, Ruff lint, strict mypy for 80 source files,
and 649 tests in 297.78 seconds with 87.24% branch coverage. The same final suite passed
in staging in 307.07 seconds. The sdist/wheel build and isolated no-network typed-
artifact verification passed. This checkpoint does not verify a human-approved
dataset, tokenizer, model, service, UI, deployment, or public release.

The fresh independent SHIP review passed 58 focused tests in 173.95 seconds. Its
placeholder-commit fixture
reports 402 task-scoped contingencies (358 `renderer_nuisance`, 44
`semantic_context`) and zero shortcut, exact-text, forbidden-skeleton, target-text, or
provenance findings over all 1,776 tasks. This does not replace human review.
