# ReactorBench-LM Phase 3 development candidate

Status: the exact pre-render packet was approved by the project owner, and the real
local Phase 3 development candidate was generated and independently typed-verified.
It remains `candidate_pending_postrender_review`; this document is not an approved
dataset release or a measured model result.

## Purpose

The candidate exercises the deterministic path from wholly fictional Aster Station
audit trajectories to leakage-resistant task projections, split-first manifests,
project-authored narrative rendering, and reproducible local artifacts. It is intended
for schema, renderer, data-integrity, and future tokenizer smoke tests.

It is not suitable for operational, engineering, licensing, emergency, maintenance,
security, safety, or real-facility decisions.

## Composition

The deterministic structured audit contains 204 trajectories, 1,762 single-input task
projections, and 14 atomic counterfactual pairs. Its trajectory distribution is 70
`iid_train`, 28 `iid_validation`, 28 `iid_test`, 28 `template_test`, 10
`component_test`, 4 `severity_test`, 8 `composition_test`, 18
`counterfactual_test`, and 10 `noise_test`. Projection counts are 148
`continue_log`, 399 `fault_family`, and 405 each for `extract_evidence`,
`next_action`, and `incident_summary`; the 14 paired `counterfactual_compare`
records are assembled separately.

The approved local candidate contains 553 distinct render candidates, 1,776 task
examples, and 18 bounded corruption records. Its automated quality report passed, but
the complete rendered inventory still requires project-owner review. The formal
5K–10K experimental pilot is a later phase and is not represented here.

The development matrix uses disjoint seed cohorts, split-first renderer and alias
assignments, held-out composition/template/component cohorts, and atomic
counterfactual groups. `SENSOR_NOISE` is simulator truth; `noise_test` is a separate
renderer-corruption dimension and never creates a fault label. The corruption matrix
uses bounded benign insertion, safe duplication, noncritical omission, and timestamp-
respecting reorder cases across more than one target outcome to reduce plan-label
shortcuts.

Six otherwise eligible `component_test` continuation records are intentionally
excluded: within that task, the held-out component alias would uniquely identify the
next-event target. `component_test` therefore has no `continue_log` coverage, and this
candidate makes no component-generalization claim for continuation. The task-scoped
audit reports marginal contingencies plus pairwise and full renderer-plan
`renderer_nuisance` interactions, with `semantic_context` reported separately. Only
exclusive nuisance template, alias, or corruption cues/interactions can become an
automated shortcut finding.
`task_record_count=1,776`, and
`audited_task_records` binds every record ID and hash, including both render foreign
keys for all 14 paired tasks. The report also records zero duplicate single-input and
zero duplicate paired-input structured fingerprints
(`single_input_structured_duplicate_count=0` and
`counterfactual_input_structured_duplicate_count=0`). Categorical target keys are
path-aware, including evidence, summary, and counterfactual roles, so unlike fields are
not collapsed into one misleading label bucket; separate leak-oriented labels remain
available for input-text leakage scanning.

Five explicit alias-plan overrides rebalance the measured joint nuisance cues before
freeze. They are derived only from split, seed, and case; no runtime task target is
consulted. This preserves deterministic split-first planning while removing the
observed exclusive renderer-plan combinations.

Seeds 0–99 remain reserved for the unfrozen golden suite; all development-matrix seeds
start at 1000. Golden scenarios are not rendered into this candidate and remain outside
training pending their separate human review and freeze.

G07 standby siblings, G08/G09 lag/stuck siblings, G12 map included/withheld siblings,
and G14 compound/pump-only/sensor-only siblings are assigned atomically. G14's
sensor-only member uses the primary-thermal sensor comparator. The 24 G15 groups contain
only the supported sparse fixture and are marked incomplete; missing evidence-expanded
siblings are neither invented nor rendered as counterfactual pairs.

Each of the 14 pairs varies one preregistered causal factor or intervention. A varied
factor may produce multiple bounded visible consequences, all of which remain available
for review; the candidate does not claim that these are literal one-line edits.

## Provenance and views

Every record is reproducible from the generator commit, scenario schema, dataset,
projection, renderer, catalog, split-manifest, and seed metadata. Audit trajectories,
model inputs, and targets are separate contracts. The renderer cannot accept a raw
simulator trace. Model input excludes latent state, injections, targets, provenance,
source event IDs, evidence annotations, later ticks, and `ACTION_APPLIED` events.

Task-specific context policies are explicit. G07 omits standby context from
fault-family and continuation views and exposes its semantic relationship only for
evidence, action, and summary views. G12 exposes dependency links only for the
map-included sibling and emits no withheld-map marker. G15 exposes one sparse
primary-flow fact and no expanded evidence. Prompt-local fact references cannot be
joined back to source event IDs.

All 405 `extract_evidence` targets are nonempty, and every target reference resolves to
a visible prompt-local observation, event, or context fact. Four G12 map-withheld
targets intentionally omit `MAPPED_COMPONENT_CHANGE` because their prompts do not
expose the dependency map. This omission is explicit, counted, and tested.

JSONL is used for the small development candidate to avoid adding an otherwise
unnecessary binary-table dependency. The sole write-capable command resolves the
validated checkout from the config and targets `data/generated/<artifact_name>`; it
accepts no arbitrary output root or directory and will not overwrite. Explicit file-
count, record-count, per-file-byte, and total-byte limits apply. Verification strictly
re-parses each JSONL record through its typed
contract and checks cross-file identities, counts, provenance, and hashes. A later
measured need may justify Parquet.

Artifact format `0.1.0` permits at most 32 JSONL payload files, 10,000 records per file,
50,000 records total, 64 MiB per payload file, 256 MiB of payloads total, 8 MiB per
record, and a 1 MiB manifest. The development config is tighter: at most 5,000 task
records and 5,000,000 rendered UTF-8 bytes.

The intended-repository gate established these exact implementation and review roots:

- generator implementation commit:
  `d3d22b7f9b2888d281c1c92cd283e10b4f0e3af1`;
- Aster schema snapshot:
  `060a1ee1b85c0333936fd14ded2975df95b0234907c82bf31f2510897bb39794`;
- dataset schema snapshot:
  `56efabaa2f9bd0c51371a1f34854f959361ab62a8880d782d9d5026711c2fc92`;
- resolved development configuration:
  `340f9185049e5e3760a77f63c2b52186770507eb976e76dfe6536f8487dafcb9`;
- structured bundle:
  `fc74f0b15cbbcaba45c164bdfab979214c8dae25c903210faff9b45e7ac35004`;
- split manifest:
  `1f9bcb95f667f6ea1a3bf29343b37195a2265d38ae8634af250d8ab0e89affa1`;
- renderer catalog:
  `18ac9eae5e2e02ca781a3afb382524486b51439b82fbc881a5e58892ccc87b90`;
- authored-language surface:
  `e35f3507c19788396326421d131dd9bc14e7ac9e42727507a7219bac7b8c6210`;
- guard manifest:
  `000a4e7c09eed0cc20c45101afcbde452b14f91d28e2bb151e2d7d2d8c4c2347`;
- structured-target inventory:
  `8b4b5a576516d9963b3008274b805f151eeb20a414622669f812566444393951`;
- pre-render packet internal checksum:
  `faa50900db2890b3bc167a44aabcb416b0a3eaa756cb578978f8e58fc3a24b8a`;
  and
- pre-render packet raw-file SHA-256:
  `2bc3e226e202a4c5c9baddaef512cf195e6086db7194b158856a782bb880dfce`.

The approved strict-parsed packet is the ignored 896,151-byte local file
`artifacts/review/catalog-review-v0.1.0.json`; it contains 176 catalog entries and binds
all 1,776 targets. The verified wheel SHA-256 is
`c9cfdb87b71a44e1b5bc6dbc6cda69104d740d8c6fe115df0198f1faa4fe3470`,
and the source-distribution SHA-256 is
`c6154e00cb9f592f43cd8a7aca013963c9d6549e7397f244422ca833c9484572`.

The pre-render packet binds all of those applicable roots, the complete structured
target inventory, and every authored renderer/corruption language surface. Its approved
record has internal checksum
`528b46e378e83da25e0a6c92c8ea24824d01cf1a00d6aeb764b6203bf4c26bb4`.
The generated candidate has bundle checksum
`3bba04bdb2030425ef67845332540fa2d148d0a318ab1d9e658f52bb890bf10c`,
artifact-manifest checksum
`222141b3ab7c9e77c4eac544f1433da067e3725057039f3bb1603be56f98bf55`,
quality-report checksum
`2549e0b0d4512424959f687834208c5572ceae98dfb2ae2edc2274268fac26e6`,
and post-render packet checksum
`b0d4c3cf11a2877e030d062efed0bebe1e53c5c87d7218402beb9bfe19f86684`.
Any bound change invalidates the approval. The candidate is not training-approved until
the project owner completes the separate post-render review.

With placeholder test commit `abcdef0`, the corrected fixture audits all 1,776 task
records and 1,977,422 rendered UTF-8 bytes. It reports 402 contingencies (358
`renderer_nuisance`, 44 `semantic_context`), 120 normalized-skeleton groups, and zero
exact-text, forbidden-skeleton, shortcut, target-text, or provenance findings;
`passed=true`. These are deterministic fixture measurements, not real artifact or
release hashes, human approval, or a generated corpus. For this exact synthetic-
approval fixture only,
`quality_report_sha256` is
`10a8ddde5213868156419fa35b762f5ba96dbcdf2f2fed90db4b870eb294f435`;
it is not a release or candidate-approval hash.

## Sources and ownership

No external source is approved as narrative training data. Text is emitted only by the
repository's deterministic grammar and fictional alias catalogs. External documents in
the research dossier are citations and design references, not corpus inputs. Code and
dataset licenses remain intentionally undecided; no distribution permission is implied.

## Quality and review

Generation is non-overwriting, limit-bounded, provenance-complete, and typed-
verification-gated. The pipeline rejects schema, split, group, structured/text
duplicate, forbidden-skeleton, provenance, evidence-grounding, and prohibited-content
failures and reports n-gram overlap without inventing a post-hoc threshold. It prepares
complete hash-bound human-review packets and cannot self-attest that a human review
occurred.

Every normalized skeleton overlap is reported. Exact text duplicates always fail.
Normalized skeleton overlap fails this developmental gate when it defeats the explicit
template-family or alias-family holdout; a global maximum skeleton share has not been
preregistered. Normalized-token 3/4/5-gram overlap is descriptive until a threshold
is frozen from pilot evidence rather than chosen after test inspection.

The denylist, pattern scan, and copied-span fingerprints are deliberately
non-exhaustive. A clean report is evidence for a reviewer, not proof that the text is
safe or source-free. The `project-owner` must review the complete bound renderer,
corruption, target, catalog, and guard packet before rendering, then separately review
the full post-render packet. The first review is approved; the generated candidate is
still awaiting the second review.

Known limitations include synthetic grammar artifacts, a small developmental sample,
low-severity-dominant simulator coverage, no training-compound examples, an unfrozen
golden suite, and simplified fictional causal dynamics. These limitations are expected
to be measured rather than hidden. Code and dataset licenses remain `TBD`, so no
distribution permission is implied.
