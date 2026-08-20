# Security controls and verification map

Status: Phase 3 dataset controls are technically complete through the mandatory local
owner-review checkpoint on 2026-08-20. The intended-repository verification gate
passed. No human-approved corpus or real narrative candidate exists; no service, UI,
deployment, or release control is verified.

Controls are **documented**, **planned**, **implemented**, or **verified** only when the
recorded evidence supports that exact scope. Verification does not make the project
“fully secure.”

| Control | Current local evidence | Next required gate |
|---|---|---|
| Synthetic-only boundary | Visible disclaimer; project-authored catalogs; non-exhaustive denylist/pattern/fingerprint scans | Owner reviews the complete hash-bound authored-language/structured-target packet and every post-render candidate before use |
| Strict contracts | Immutable unknown-field-rejecting generator, projection, grouping, manifest, review, and artifact contracts; duplicate-key/non-finite JSON rejection; packaged schema snapshots | Human review and later version-1 freeze; snapshots are developmental, not a freeze |
| Truth isolation | Renderer accepts only strict `ModelInput`; source IDs, evidence annotations, latent/injection/target/provenance fields, and later action effects are rejected | Preserve this boundary in tokenizer/model loaders and any future API |
| Determinism/integrity | Seeded local RNG; canonical snapshots; split-first build; content checksums; contained, non-overwriting JSONL bundles | Record a real candidate manifest only after both human gates; freeze before pilot/model use |
| Split integrity | Disjoint seed cohorts; template/alias/composition holdouts; atomic G07/G08-G09/G12/G14 groups; G15 explicitly incomplete | Owner review, pilot-informed freeze, and continuing cross-split audit on each candidate |
| Evidence integrity | All 405 evidence targets resolve to visible prompt-local facts; zero empty targets; four declared G12 map-withheld omissions exclude the non-visible `MAPPED_COMPONENT_CHANGE` fact | Preserve the closed-world check when task contracts or generator coverage change |
| Shortcut and duplicate integrity | Zero single-input and paired structured duplicates; task-scoped marginal contingencies plus pairwise/full-plan `renderer_nuisance` interactions; `semantic_context` reported separately while only nuisance features/interactions may raise findings; exact 1,776 task IDs/hashes and paired render keys bound; six alias-revealing component continuation examples excluded, leaving no component-test continuation claim; five pre-freeze alias overrides keyed only by split/seed/case | Rerun the complete audit whenever task views, aliases, or split plans change |
| Text integrity | Exact text duplicates fail; all skeleton and 3/4/5-gram overlap reported; explicit renderer/alias holdout skeleton violations fail; target-text checks | Preregister any maximum-share/overlap threshold after pilot evidence and before freeze |
| Controlled corruption | Four bounded corruption types assigned as a balanced evaluation matrix; corruption lineage remains separate from simulator truth | Review every rendered corruption and retain label-invariance tests |
| Review authority | The pre-render packet binds the exact structured bundle and complete target inventory to all authored renderer/corruption wording; only `project-owner` may satisfy either human gate | Owner must personally inspect and approve both exact packets; no automated fixture counts |
| Candidate artifact safety | Sole write command; fixed `data/generated/<artifact_name>` path under the config-validated checkout; no arbitrary output root; non-overwriting write; strict typed reparse; cross-file linkage; per-file/total byte, file-count, and record-count limits; complete config/schema/bundle/review/report provenance | Create and record candidate/post-render hashes only after mandatory owner review; add data-only checkpoint controls later |
| Browser/service safety | No browser, service, credentials, or routes exist | Strict request/response schemas, limits, safe output rendering/errors, CSP, gateway authorization if required, and tests in Phase 7; no user accounts |
| Supply-chain/release | Locked local dependencies and package builds verified | CI review/secret scanning/SBOM and release provenance before owner-managed publication |

## Local evidence and limits

At verified implementation commit
`d3d22b7f9b2888d281c1c92cd283e10b4f0e3af1`, the intended-repository Python 3.12.11
gate passed Ruff formatting for 106 files, Ruff lint, strict mypy for 80 source files,
and 649 tests in 297.78 seconds with 87.24% branch coverage. The same 649-test suite
passed in staging in 307.07 seconds. The sdist/wheel build and isolated no-network
typed-artifact verification passed. Exact schema, config, lineage, packet, and package
hashes are recorded in `docs/IMPLEMENTATION_STATUS.md`.

The fresh independent SHIP review passed 58 focused tests in 173.95 seconds. Its
placeholder-commit fixture
audits 402 task-scoped contingencies (358 `renderer_nuisance`, 44
`semantic_context`) over all 1,776 tasks and reports zero shortcut, exact-text,
forbidden-skeleton, target-text, or provenance findings. This is focused contract
evidence, not either human review or a generated candidate.

The scanners are deliberately non-exhaustive. They match selected denylist, pattern,
and copied-span fingerprints without reflecting matched content into safe errors. They
cannot prove absence of prohibited content and cannot self-attest human approval.

Artifact format `0.1.0` caps bundles at 32 JSONL payload files, 10,000 records per file,
50,000 records total, 64 MiB per payload file, 256 MiB of payloads total, 8 MiB per
record, and a 1 MiB manifest. The development config applies the tighter 5,000-task and
5,000,000-rendered-byte limits. These are local safety ceilings, not measured scale
targets or release guarantees.

## Remaining Phase 3 human gate

The technical projection and split controls are implemented. Before any real candidate
is generated, the project owner must approve the packet containing the 176-entry
catalog and every authored renderer/corruption language surface, bound to the exact
resolved configuration, generator commit, structured bundle, split manifest, target
inventory, catalog, and guard. After generation, the owner must separately review the
complete 553-candidate packet and quality report. Automated test fixtures that set the
review decision to approved are contract tests only and are never review evidence.

The grouped relationships are:

- G07 matched standby contexts;
- G08/G09 resolution-versus-persistence counterfactuals;
- G12 dependency-map-included versus withheld contexts;
- G14 compound/pump-only/drift-only and component/channel role relations; and
- G15 sparse-only groups, marked incomplete because expanded-evidence relatives do not
  exist in the generator.

No approved dataset, tokenizer, model, service, or UI exists. The project owner alone
will push or deploy after later release gates; no external publication happened here.
