# ReactorBench-LM implementation status

Last updated: 2026-08-20 America/New_York
Current phase: **Phase 3 complete locally — development candidate approved**
Current objective: preserve the immutable Phase 3 evidence and hand off cleanly to a
future Phase 4 tokenizer/model-correctness audit.
Checkpoint reason: the project owner approved both exact hash-bound review packets;
generation, quality audit, typed reconstruction, separate read-only verification, and
strict post-render approval-record verification all passed. Phase 4 has not started.
Intended project path: `/Users/zachary/Documents/Personal-Projects/AI-transformer`

## Completed work

- Phase 0 audit, Phase 1 foundation, and Phase 2 developmental G01–G15 structured
  generator are complete locally.
- Phase 3 defines strict unknown-field-rejecting contracts for `ModelInput`, structured
  task targets, lineage, grouping, split manifests, counterfactual pairs, renderer
  output, bounded corruption, quality reports, human review, and artifact manifests.
- The deterministic development configuration produces 204 audit trajectories, 1,762
  single-input projections, and 14 paired counterfactual comparisons. Split counts are
  70/28/28/28/10/4/8/18/10 for train, validation, IID test, template, component,
  severity, composition, counterfactual, and noise respectively.
- Projection task counts are 148 continuation, 399 fault-family, and 405 each for
  evidence extraction, next action, and incident summary. The 14 paired comparison
  tasks are assembled separately for 1,776 total task examples in the generated
  candidate.
- All 405 evidence targets are nonempty and resolve only to visible prompt-local facts.
  Four map-withheld G12 targets intentionally omit `MAPPED_COMPONENT_CHANGE` because
  the dependency fact is absent. Six `component_test` continuation projections are
  intentionally excluded because their held-out alias would identify the next-event
  target within that task. `component_test` therefore has no `continue_log` coverage,
  and no component-generalization claim is made for that task.
- Projection ends at the task decision/event cut. `ModelInput` strips source event IDs
  and relationships, evidence annotations, latent state, fault injections, targets,
  scenario/trajectory IDs, seeds, provenance, and later `ACTION_APPLIED` consequences.
  G07, G12, and G15 use explicit minimal context policies.
- Seed, renderer-family, alias-family, severity, composition, and counterfactual
  assignments occur before rendering. G07, G08/G09, G12, and G14 groups are complete
  and atomic. G14 uses the exact primary-thermal sensor-only comparator. All 24 G15
  sparse-only groups are marked incomplete because evidence-expanded siblings are not
  supported; no G15 pair is fabricated.
- The deterministic renderer is entirely project-authored. Four template families by
  four alias families by eleven event types produce 176 mandatory catalog-review
  entries.
- Exact text and structured-input duplicates fail;
  `single_input_structured_duplicate_count=0` and
  `counterfactual_input_structured_duplicate_count=0`. All normalized skeleton and
  normalized-token 3/4/5-gram overlap is reported. Skeleton overlap fails when it
  violates the explicit template or alias holdout. Shortcut contingencies are scoped by
  task and feature class, including `semantic_context` and explicit `corruption:none`;
  marginal plus pairwise and full renderer-plan `renderer_nuisance`
  template/alias/corruption exclusivities become shortcut findings.
  `task_record_count=1,776`, and
  `audited_task_records` binds every exact
  record ID and hash, including both render foreign keys for each of the 14 paired
  tasks. No unregistered maximum-share or n-gram pass threshold was chosen after
  inspecting results.
- `noise_test` uses an applicability-checked balanced matrix of benign insertion,
  duplicate line, noncritical omission, and safe timestamp-respecting reorder. Its
  provenance remains separate from simulator truth.
- The pre-render packet presents every actual authored renderer/corruption language
  surface and the complete structured-target inventory, bound to the exact resolved
  configuration, generator commit, structured bundle, split manifest, catalog, and
  guard. The project owner approved that exact packet on 2026-08-20. The gated path
  generated 553 distinct render candidates, 1,776 task examples, and 18 corruption
  records. The project owner approved the complete post-render packet on 2026-08-20;
  the candidate is approved for local Phase 4 tokenizer/model work, not public release.
- Candidate artifacts use canonical JSONL. The sole write-capable command resolves the
  validated project checkout from the config and targets
  `data/generated/<artifact_name>`; it accepts no arbitrary output root or directory and
  will not overwrite. Explicit file/record/per-file-byte/total-byte limits and
  complete candidate/config/structured/split/schema/catalog/guard/review/quality
  provenance. Verification strictly re-parses every typed record and checks cross-file
  links as well as hashes. `task-shortcut-records.jsonl` preserves the exact quality-
  audit input. Developmental dataset schema snapshots are packaged and validated.
- No tokenizer, model, checkpoint, training run, inference service, UI, measured model
  result, GitHub push, Vercel deployment, or public release exists.

## Files created or changed

- Dataset package: `src/reactorbench/dataset/` and its packaged guard resources.
- Development configuration: `configs/dataset/development-v0.1.0.toml`.
- Dataset contract snapshots: `schemas/dataset/v0/`; the Aster snapshot was updated for
  the empty-safe incident-summary shape and primary-thermal G14 comparator.
- Dataset unit/property/contract tests under `tests/` and package/build integration in
  `pyproject.toml`.
- Phase 3 documentation: `README.md`, `docs/architecture.md`,
  `docs/security-controls.md`, `docs/threat-model.md`, `docs/data/`, and the reconciled
  research documents.

## Tests and checks run

- Last committed Phase 2 evidence: 434 tests passed on Python 3.12.11 with 92.20%
  branch coverage; Ruff format/lint, strict mypy for 41 source files, sdist/wheel build,
  and isolated artifact verification passed.
- The reconciled Phase 3 test-only inventory is 204 trajectories, 1,762 projections, 14
  pairs, 553 distinct render candidates, 1,776 task examples, and 18 corruption records.
  It has zero empty evidence targets, four intentional map-withheld
  `MAPPED_COMPONENT_CHANGE` omissions,
  `single_input_structured_duplicate_count=0`, and
  `counterfactual_input_structured_duplicate_count=0`. With placeholder test commit
  `abcdef0`, the corrected fixture audits all 1,776 task records and 1,977,422 rendered
  UTF-8 bytes. It reports 402 contingencies (358 `renderer_nuisance`, 44
  `semantic_context`), 120 normalized-skeleton groups, and zero exact-text,
  forbidden-skeleton, shortcut, target-text, or provenance findings; `passed=true`.
  These are deterministic fixture measurements, not a human-approved corpus or final
  integrated gate. For that exact `generator_commit=abcdef0` synthetic-approval fixture
  only, `quality_report_sha256` is
  `10a8ddde5213868156419fa35b762f5ba96dbcdf2f2fed90db4b870eb294f435`.
  It is not an artifact, release, or candidate-approval hash and is not promoted into
  the verified artifact fields below.
- A fresh independent SHIP review passed **58 focused tests in 173.95 seconds**, covering
  the corrected marginal/joint shortcut, evidence-grounding, inventory, review, and
  artifact boundaries.
- Intended-repository static gate on Python 3.12.11: Ruff format passed for 106 files;
  Ruff lint passed; strict mypy passed for 80 source files.
- Intended-repository full gate on Python 3.12.11: **649 tests passed in 297.78 seconds**
  with **87.24% branch coverage**. The same final 649-test suite passed in staging in
  307.07 seconds.
- The sdist/wheel build and isolated no-network artifact verification passed. The exact
  package hashes are recorded below.
- The project-owner pre-render approval record verified against the exact structured
  bundle. Local generation completed without overwrite, and both the built-in typed
  reconstruction and separate read-only `verify` command passed. The real quality
  report has 553 candidate records, 1,776 task records, 402 contingencies, and zero
  exact-duplicate, forbidden-skeleton, shortcut, target-text, or provenance findings;
  `passed=true`.
- The project-owner post-render approval record strict-parsed and verified against the
  unchanged 553-entry packet with `require_approved=True`. Its internal checksum is
  `e066d5944839423fdd6e49491dfa5b57867b0c753ac465afb4ac196e7a87958d`;
  its raw-file SHA-256 is
  `001eca9925d9d6c5b8a50c9bd524cbe990b17b6a675b53200f16cf379d9b7af7`.
- Documentation stale-value and trailing-whitespace checks passed for the 12 reconciled
  Phase 3 documents on 2026-08-20. Rerun them after any subsequent documentation
  change.

## Decisions made

- `ModelInput` is the renderer's only source contract; audit and target contracts never
  enter the renderer.
- Split assignment and atomic grouping precede narrative rendering.
- G14's sensor-only comparator is exact primary-thermal sensor drift; G15 remains
  explicitly incomplete until real generator-supported expanded siblings exist.
- Counterfactual pairs vary one preregistered causal factor or intervention, which may
  produce multiple visible consequences; they are not described as literal one-fact
  edits.
- Evidence targets form a closed visible-fact set. The four G12 map-withheld omissions
  are intentional, and the six component continuation exclusions remove a measured
  heldout-alias shortcut rather than fabricating counterexamples.
- Exact duplicates always fail. Skeleton overlap is fully reported and currently fails
  only when it defeats an explicit renderer/alias holdout; any global share or n-gram
  threshold must be preregistered after pilot evidence and before freeze.
- Shortcut evidence is task-scoped and path-aware. `semantic_context` contingencies are
  reported separately; marginal plus pairwise and full renderer-plan
  `renderer_nuisance` exclusivity raises findings. Path-aware categorical target labels
  drive within-task contingencies, while separate leak-oriented labels drive input-text
  leakage scanning.
- Five explicit pre-freeze alias-plan overrides balance the measured joint nuisance
  cues. They are keyed only by split, seed, and case, never by a runtime target lookup;
  the exact plan is recorded in D-059.
- Corruptions are balanced across target outcomes and never change ground truth.
- Human pre-render and post-render approval are mandatory and cannot be replaced by
  scanners or automated approval fixtures. Both owner approvals were completed on
  2026-08-20 and are bound to their exact packets. The pre-render record also binds the
  structured bundle, target inventory, and complete authored language surface.
- Canonical JSONL is the developmental format; a binary table dependency requires a
  later measured need.
- Artifact verification is bounded, typed, cross-file, and config-selected, not
  checksum-only. Only `generate-full-development` writes; `verify` is read-only.
- Developmental schema snapshots are reviewed implementation evidence, not a version-1
  freeze. Code and dataset licenses remain `TBD` and block distribution, not local
  review.

## Assumptions

- All source scenarios, text, values, aliases, policies, and actions remain
  project-authored, wholly fictional, normalized, and non-operational.
- Exact account usage percentage is not observable in this environment. Conservative
  phase checkpoints are used; no claimed 1% account cutoff has been measured.
- The project owner explicitly approved the exact catalog and candidate packets on
  2026-08-20; the agent only serialized and verified the supplied decisions.

## Known failures and residual risks

- No known technical or human-review Phase 3 gate remains after the intended-repository
  gate, independent SHIP review, candidate generation, separate typed verification,
  and both project-owner approvals.
- The denylist, pattern suite, and copied-span fingerprint registry are non-exhaustive.
  Zero automated findings cannot prove the absence of prohibited or source-derived
  content.
- Repeated synthetic grammar skeletons remain visible in the report; no global
  maximum-share threshold is frozen.
- G15 evidence-expanded siblings are unsupported, so the 24 sparse-only groups remain
  incomplete and are excluded from paired comparisons.
- The golden suite is not human reviewed or frozen. Dataset/task schemas, manifests,
  and thresholds are developmental, not version 1.

## Open blockers

- No blocker remains for beginning local Phase 4 tokenizer/model-correctness work.
- Code and dataset licenses must be selected before any distribution.
- Developmental dataset/task schemas and the golden suite remain unfrozen; they must be
  reviewed and frozen at their documented later gates before main evaluation/release.

## Uncommitted work

- No tracked source or documentation work should remain after the local closeout
  commit. The approved review packets/records and generated candidate remain ignored
  local artifacts by design; their exact paths and hashes are recorded below.

## Repository state

- Prior Phase 2 HEAD: `d59a94d` (`docs: checkpoint completed phase 2`) on
  `codex/foundation`.
- Verified Phase 3 implementation commit:
  `d3d22b7f9b2888d281c1c92cd283e10b4f0e3af1`.
- Last known pre-closeout documentation checkpoint:
  `ff23e88593a87ddf647e28c0902d6e9280497639`.
- This handoff is part of the subsequent local Phase 3 closeout commit. The tracked
  worktree is expected to be clean after that commit; verify it with `git status` and
  obtain the exact closeout hash with `git rev-parse --verify HEAD`. Both approved
  review packets/records and the generated candidate remain intentionally ignored local
  artifacts. No remote, push, publication, or deployment exists.

## Immediate next step

On a future turn, begin Phase 4 with a read-only tokenizer/model-correctness design
audit. Re-read `research/PROJECT_REQUIREMENTS.md`, the Phase 4 sections of
`research/RESEARCH_BLUEPRINT.md`, and this handoff; verify the clean Git checkpoint and
both local approval records before changing code. Do not repeat Phase 3 generation or
rewrite the approved candidate.

## Exact recommended next command

At the start of Phase 4, verify the local checkpoint and approval artifacts before the
read-only design audit:

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
git status --short --branch
shasum -a 256 artifacts/review/postrender-review-record-v0.1.0.json
```

The expected approval-record raw-file hash is
`001eca9925d9d6c5b8a50c9bd524cbe990b17b6a675b53200f16cf379d9b7af7`.

## Relevant artifacts and configuration

- Schema/generator version: `0.1.0`, `frozen=false`
- Verified generator implementation commit:
  `d3d22b7f9b2888d281c1c92cd283e10b4f0e3af1`
- Aster schema snapshot SHA-256:
  `060a1ee1b85c0333936fd14ded2975df95b0234907c82bf31f2510897bb39794`
- Dataset schema snapshot SHA-256:
  `56efabaa2f9bd0c51371a1f34854f959361ab62a8880d782d9d5026711c2fc92`
- Resolved development-config SHA-256:
  `340f9185049e5e3760a77f63c2b52186770507eb976e76dfe6536f8487dafcb9`
- Structured-bundle SHA-256:
  `fc74f0b15cbbcaba45c164bdfab979214c8dae25c903210faff9b45e7ac35004`
- Split-manifest SHA-256:
  `1f9bcb95f667f6ea1a3bf29343b37195a2265d38ae8634af250d8ab0e89affa1`
- Renderer-catalog SHA-256:
  `18ac9eae5e2e02ca781a3afb382524486b51439b82fbc881a5e58892ccc87b90`
- Authored-language-surface SHA-256:
  `e35f3507c19788396326421d131dd9bc14e7ac9e42727507a7219bac7b8c6210`
- Guard-manifest SHA-256:
  `000a4e7c09eed0cc20c45101afcbde452b14f91d28e2bb151e2d7d2d8c4c2347`
- Structured-target-inventory SHA-256:
  `8b4b5a576516d9963b3008274b805f151eeb20a414622669f812566444393951`
- Approved pre-render packet internal SHA-256:
  `faa50900db2890b3bc167a44aabcb416b0a3eaa756cb578978f8e58fc3a24b8a`
- Approved pre-render packet raw-file SHA-256:
  `2bc3e226e202a4c5c9baddaef512cf195e6086db7194b158856a782bb880dfce`
  (896,151 bytes; strict parse passed; ignored local review artifact)
- Built wheel SHA-256:
  `c9cfdb87b71a44e1b5bc6dbc6cda69104d740d8c6fe115df0198f1faa4fe3470`
- Built source-distribution SHA-256:
  `c6154e00cb9f592f43cd8a7aca013963c9d6549e7397f244422ca833c9484572`
- Approved pre-render review-record internal SHA-256:
  `528b46e378e83da25e0a6c92c8ea24824d01cf1a00d6aeb764b6203bf4c26bb4`
- Generated candidate-bundle SHA-256:
  `3bba04bdb2030425ef67845332540fa2d148d0a318ab1d9e658f52bb890bf10c`
- Generated artifact-manifest internal SHA-256:
  `222141b3ab7c9e77c4eac544f1433da067e3725057039f3bb1603be56f98bf55`
- Generated quality-report internal SHA-256:
  `2549e0b0d4512424959f687834208c5572ceae98dfb2ae2edc2274268fac26e6`
- Generated post-render packet internal SHA-256:
  `b0d4c3cf11a2877e030d062efed0bebe1e53c5c87d7218402beb9bfe19f86684`
- Generated post-render packet raw-file SHA-256:
  `1ad8a07e01d748046c8eadff1a953dafecf56d36e0e787e9e707fc9aad70c390`
  (3,041,014 bytes; strict parse passed)
- Approved post-render review-record internal SHA-256:
  `e066d5944839423fdd6e49491dfa5b57867b0c753ac465afb4ac196e7a87958d`
- Approved post-render review-record raw-file SHA-256:
  `001eca9925d9d6c5b8a50c9bd524cbe990b17b6a675b53200f16cf379d9b7af7`
  (strict parse and exact-packet verification passed; ignored local review artifact)
- The `abcdef0` fixture quality hash above remains test-only evidence and is not a
  candidate, approval, artifact, or release hash.
- Development configuration: `configs/dataset/development-v0.1.0.toml`
- Review procedure: `docs/data/PHASE3_REVIEW.md`
- Approved pre-render packet path (ignored, not committed):
  `artifacts/review/catalog-review-v0.1.0.json`
- Approved pre-render owner-record path (ignored, not committed):
  `artifacts/review/catalog-review-record-v0.1.0.json`
- Generated config-selected candidate path (ignored, not committed):
  `data/generated/phase3-development-v0.1.0-candidate`
- Post-render review packet path:
  `data/generated/phase3-development-v0.1.0-candidate/postrender-review.jsonl`
- Approved post-render owner-record path (ignored, not committed):
  `artifacts/review/postrender-review-record-v0.1.0.json`
- The candidate's embedded `candidate_pending_postrender_review` value remains unchanged
  as generation-time state; the separate verified record approves it for local Phase 4
  work without making it a public release.

## Exact resume prompt

Resume ReactorBench-LM from the safe checkpoint in
/Users/zachary/Documents/Personal-Projects/AI-transformer.
Read research/PROJECT_REQUIREMENTS.md and docs/IMPLEMENTATION_STATUS.md first.
Inspect Git status and verify the recorded tests before making changes.
Continue from the documented immediate next step without repeating completed work.
