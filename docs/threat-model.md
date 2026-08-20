# Threat model

Status: Phase 2 local generator review, 2026-08-20. Future browser, inference, and deployment boundaries are planned only.

## Scope and assumptions

The project produces only project-authored, wholly fictional Aster Station trajectories.
No real plant, Navy, restricted, operational, proprietary, user-supplied, or hosted-LLM
corpus material is permitted. External publication and deployment remain owner-managed.

```text
reviewed config/source -> strict generator -> audit trajectory -> Phase 3 manifests
                                                        -> future tokenizer/model/artifacts
untrusted browser -> future gateway -> future inference -> checksummed artifacts
```

The first boundary is implemented locally. Browser/gateway/inference boundaries do not
exist yet and this document does not claim their controls are live.

## Current risks and gates

| ID | Risk | Current mitigation | Remaining gate |
|---|---|---|---|
| TM-01 | Prohibited or real-world material enters the corpus | Synthetic-only policy, disclaimer, narrow redacting scanner | Reviewed full denylist and human sample review before pilot rendering |
| TM-02 | Truth, IDs, or later outcomes leak into model input | Separate contracts and audit-payload allowlist | Phase 3 decision-tick/channel projection and shortcut/leakage tests |
| TM-03 | Invalid/unbounded contracts bypass validation | Strict immutable developmental Pydantic models and generator validation | Snapshot-review projection/manifest schemas before pilot use |
| TM-04 | Unsafe local path/artifact input | Project-relative config boundaries; no user file/URL/checkpoint input | Safe data-only artifact loading and checksum tests before models/services |
| TM-05 | Dataset split contamination or nondeterminism | Deterministic generators, snapshots, canonical IDs | Split-first manifests; duplicate/skeleton/group leakage checks |
| TM-06 | Model/rendered text causes unsafe display behavior | No UI exists | Encode as text, forbid raw HTML, add CSP and browser tests in Phase 7 |
| TM-07 | Secrets/logs/errors expose sensitive material | No service credentials or routes exist | Server-only secrets, minimal logs, safe errors, secret scanning before deployment |
| TM-08 | Artifact/model replacement changes results | Build/artifact verification only | Pinned data-only artifact manifests and tamper tests before serving |
| TM-09 | Release/dependency compromise | Local lockfile and package build verification | CI hardening, dependency review, SBOM, release allowlist |
| TM-10 | Public output is mistaken for real advice | Prominent fictional/non-operational disclaimer | Curated UI, limitation review, and release evidence |

## Required Phase 3 controls

The audit payload is truth-filtered, but is not itself a model prompt. The first Phase 3
milestone must project only allowed event/channel evidence through a decision tick and
reject truth, injections, targets, provenance/scenario IDs, audit-only context IDs, and
post-decision action consequences. It must group and audit G07, G08/G09, G12, G14, and
G15 counterfactual relationships before rendering.

The Phase 2 scanner remains intentionally non-exhaustive. It may reduce accidental
prohibited strings but cannot prove safety; the full denylist and human sample review
are deferred requirements, not completed controls.

## Verified local evidence

After local code commit `b63f6d8`, the intended-project full gate passed 434 tests at
92.20% branch coverage with Ruff format/lint, strict mypy for 41 source files,
sdist/wheel, and isolated artifact verification green. This evidence does not verify a
dataset, tokenizer, model, service, UI, deployment, or public release.
