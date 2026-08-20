# Phase 3 data-review procedure

ReactorBench-LM data is synthetic, but automated validation cannot prove that every
rendered phrase is appropriate for the project's fictional, non-operational boundary.
This procedure is a required human gate, not a ceremonial checkbox.

## Pre-render review

Open the existing checksum-verified packet recorded in
`docs/IMPLEMENTATION_STATUS.md`. Review the complete renderer-template catalog, alias
catalog, policy/context phrases, event
clauses, observation/status/quality wording, state/mode wording, corruption language,
content-rule metadata, and one deterministic preview for every supported
event/template/alias combination. Also inspect the complete structured-target inventory
to which that language surface is bound. The reviewer must verify:

- the text is wholly fictional and contains no real facility, procedure, setpoint,
  incident, personal, Navy, restricted, or proprietary material;
- every sentence describes only normalized Aster Station facts;
- no template states or implies a target label, hidden fault injection, or action result;
- the visible disclaimer and prohibited-use boundary remain accurate;
- the recorded resolved configuration, generator commit, structured bundle, split
  manifest, target inventory, authored-language, catalog, and content-rule SHA-256
  values match the material reviewed.

Record only the bounded reviewer role `project-owner`, the UTC review date, the exact
packet hash, one serialized decision (`approved`, `revise`, or `rejected`), all seven
required strict-boolean confirmations, and non-sensitive notes. A `revise` or
`rejected` record may and should preserve `false` for any confirmation the owner could
not make; it is valid review evidence but never authorizes generation. An `approved`
record is valid only when every confirmation is literally `true`. String and integer
substitutes for booleans are rejected. Do not add personal details or a signature to
the repository. Generation must fail closed when the record is absent, unapproved,
stale, uses any role other than `project-owner`, does not bind the exact structured
bundle, or contains unknown fields.

The catalog portion has 176 entries: four template families by four alias families by
eleven event types. It is only one part of the packet; catalog coverage does not replace
review of the actual authored renderer/corruption surfaces or the 1,776-target binding.
The verified implementation and review roots are:

- generator commit:
  `d3d22b7f9b2888d281c1c92cd283e10b4f0e3af1`;
- resolved configuration:
  `340f9185049e5e3760a77f63c2b52186770507eb976e76dfe6536f8487dafcb9`;
- structured bundle:
  `fc74f0b15cbbcaba45c164bdfab979214c8dae25c903210faff9b45e7ac35004`;
- split manifest:
  `1f9bcb95f667f6ea1a3bf29343b37195a2265d38ae8634af250d8ab0e89affa1`;
- target inventory:
  `8b4b5a576516d9963b3008274b805f151eeb20a414622669f812566444393951`;
- renderer catalog:
  `18ac9eae5e2e02ca781a3afb382524486b51439b82fbc881a5e58892ccc87b90`;
- authored-language surface:
  `e35f3507c19788396326421d131dd9bc14e7ac9e42727507a7219bac7b8c6210`;
  and
- guard manifest:
  `000a4e7c09eed0cc20c45101afcbde452b14f91d28e2bb151e2d7d2d8c4c2347`.

The deterministic packet was prepared as the ignored local file
`artifacts/review/catalog-review-v0.1.0.json`. Its strict parse passed; it is 896,151
bytes, its raw-file SHA-256 is
`2bc3e226e202a4c5c9baddaef512cf195e6086db7194b158856a782bb880dfce`,
and its internal packet checksum is
`faa50900db2890b3bc167a44aabcb416b0a3eaa756cb578978f8e58fc3a24b8a`.
It contains all 176 catalog entries and binds all 1,776 targets. Review that file and
compare every bound field; do not approve from counts or documentation alone. Any
difference requires regeneration and a complete new review. The project owner approved
this exact packet on 2026-08-20. The strict approval record is
`artifacts/review/catalog-review-record-v0.1.0.json`; its internal checksum is
`528b46e378e83da25e0a6c92c8ea24824d01cf1a00d6aeb764b6203bf4c26bb4`.

If deterministic regeneration is needed for comparison, pin the implementation commit
rather than the later documentation `HEAD` and avoid overwriting the reviewed packet:

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
uv run --frozen python -m reactorbench.dataset prepare-review \
  --config configs/dataset/development-v0.1.0.toml \
  --generator-commit d3d22b7f9b2888d281c1c92cd283e10b4f0e3af1 \
  > /private/tmp/reactorbench-catalog-review-v0.1.0.json
shasum -a 256 /private/tmp/reactorbench-catalog-review-v0.1.0.json
cmp artifacts/review/catalog-review-v0.1.0.json \
  /private/tmp/reactorbench-catalog-review-v0.1.0.json
```

The existing packet is ignored and not committed. Preparing or regenerating a packet
does not record a decision.

### Owner-review-record workflow

1. Open the existing JSON and verify `review_stage`, the exact resolved-configuration
   and generator-commit binding, structured-bundle and split-manifest hashes, the full
   1,776-record target inventory and its hash, `catalog_preview.entry_count=176`, all
   authored-language surfaces, the catalog, guard, inner preview, and outer packet
   hashes. Review every entry, alias, context phrase, event/observation/state clause,
   corruption phrase, policy wording, target, denylist/pattern metadata, and copied-span
   fingerprint registry. Do not approve from counts or checksums alone.
2. If any item is questionable, record `ReviewDecision.REVISE` or
   `ReviewDecision.REJECTED` with the seven literal boolean confirmation results,
   including `false` wherever the owner could not confirm the statement. Preserve that
   unapproved record as review evidence, change the authored source, regenerate the
   entire packet, and restart review. Never edit a preview entry or hash by hand.
3. Only after the project owner has personally completed every confirmation, call the
   repository's `create_review_record(...)` helper with reviewer role `project-owner`,
   the actual UTC date, `ReviewDecision.APPROVED`, all seven explicit
   `ReviewConfirmations` set to literal `true`, and any non-sensitive notes. Serialize
   the returned strict object canonically to the reserved path
   `artifacts/review/catalog-review-record-v0.1.0.json`.
4. Immediately call
   `verify_catalog_review_gate(packet, record, structured_bundle=bundle)`. It must fail
   if the packet, authored surface, guard, configuration, commit, structured bundle,
   split manifest, or target inventory changed; if the stage/hash or `project-owner`
   role does not match; if the record has unknown fields; or if the decision is not
   approved. Do not copy a record from tests or edit its checksum.
5. Preserve both files. Supply those exact paths to the gated local candidate command.
   The CLI accepts no output-root or arbitrary-directory option. The reviewed config
   selects the fixed, project-contained
   `data/generated/phase3-development-v0.1.0-candidate` path, which must not already
   exist.

The following generation and verification commands completed successfully for this
candidate. Do not rerun them in the occupied non-overwriting output path; they are
recorded for clean-checkout reproduction:

```bash
uv run --frozen python -m reactorbench.dataset generate-full-development \
  --config configs/dataset/development-v0.1.0.toml \
  --review-packet artifacts/review/catalog-review-v0.1.0.json \
  --review-record artifacts/review/catalog-review-record-v0.1.0.json \
  --generator-commit d3d22b7f9b2888d281c1c92cd283e10b4f0e3af1
uv run --frozen python -m reactorbench.dataset verify \
  --config configs/dataset/development-v0.1.0.toml
```

The only write-capable command is `generate-full-development`. It writes once to the
config-selected path and immediately typed-verifies the full bundle. `verify` is
read-only and selects that same path from the config.

The helper intentionally requires explicit human-supplied values; there is no command
that auto-approves a packet. If an agent assists with serialization after the owner
reviews it, the owner must explicitly state the decision and date. Neither this
document nor the expected hashes authorizes creating an `APPROVED` record.

## Post-render review

Generation produces a deterministic full distinct inventory spanning every split,
task, diagnosis status, plant variant, renderer family, alias family, and
counterfactual role present in the development candidate. Review the full inventory
against the same boundary and record its checksum in a separate status record. The
candidate artifact remains
`candidate_pending_postrender_review` until this step is complete.

The generated candidate presents the full distinct render inventory, not a hidden
subset: 553 candidates, 1,776 task records, and 18 corruption records. Its packet
checksum is
`b0d4c3cf11a2877e030d062efed0bebe1e53c5c87d7218402beb9bfe19f86684`;
the raw packet file SHA-256 is
`1ad8a07e01d748046c8eadff1a953dafecf56d36e0e787e9e707fc9aad70c390`.
These measured inventory values and a passing automated report are not permission to
train. The project owner must inspect the complete packet and record a separate
post-render decision.

Promotion creates a new status record; it never rewrites generated data. The project
owner explicitly approved this exact packet on 2026-08-20 and instructed Phase 3
closeout. The canonical record is the ignored local file
`artifacts/review/postrender-review-record-v0.1.0.json`; its internal checksum is
`e066d5944839423fdd6e49491dfa5b57867b0c753ac465afb4ac196e7a87958d`
and its raw-file SHA-256 is
`001eca9925d9d6c5b8a50c9bd524cbe990b17b6a675b53200f16cf379d9b7af7`.
Strict re-parsing and `verify_review_record(..., require_approved=True)` passed against
the unchanged post-render packet. The candidate's embedded
`candidate_pending_postrender_review` value is intentionally retained as historical
generation state; the separate approval record closes the gate. An approved
development candidate is still not a public release, training result, operational
dataset, or the later 5K–10K experimental pilot tier.

Candidate output is fixed to `data/generated/<artifact_name>` inside the validated
project checkout; the CLI accepts no arbitrary output root or directory. The selected
path must not exist. File count, record count, per-file bytes, and total bytes are
bounded. Before review, the verifier must strictly re-parse every typed JSONL file,
check cross-file IDs/counts/hashes, and confirm the manifest binds the resolved config,
structured/split bundles, both schema snapshots, catalog, guard, both review packets,
pre-render review record, and quality report. Typed generation and a separate read-only
verification both passed. The candidate bundle checksum is
`3bba04bdb2030425ef67845332540fa2d148d0a318ab1d9e658f52bb890bf10c`;
the artifact-manifest checksum is
`222141b3ab7c9e77c4eac544f1433da067e3725057039f3bb1603be56f98bf55`;
and the quality-report checksum is
`2549e0b0d4512424959f687834208c5572ceae98dfb2ae2edc2274268fac26e6`.

## Automated evidence

Automation must report schema validation, evidence-grounding coverage, provenance
coverage, single/pair structured duplicates, exact text duplicates, normalized skeleton
overlap, normalized-token 3/4/5-gram overlap, split leakage, complete task-scoped
marginal contingencies, pairwise/full-plan `renderer_nuisance` interactions,
separately reported `semantic_context`, prohibited-content findings, and checksums.
Categorical target keys are path-aware, including evidence, summary, and
counterfactual roles; separate leak-oriented labels are retained for input-text leakage
scanning. The nuisance audit includes `corruption:none`, so its support accounts
for uncorrupted as well as corrupted task records. Require `task_record_count=1,776`;
`audited_task_records` must bind every task-record ID/hash and both render foreign keys
for each paired task. The artifact preserves those inputs in
`task-shortcut-records.jsonl`. Only exclusive nuisance template/alias/corruption cues
and their bounded interactions are automated shortcut findings. Confirm that
`component_test` has no `continue_log` coverage and do not infer component-
generalization evidence for that task. A
zero-finding scanner result is evidence for review, not proof that no prohibited
material exists.

With placeholder test commit `abcdef0`, the corrected fixture audits all 1,776 task
records and 1,977,422 rendered UTF-8 bytes. It reports 402 contingencies (358
`renderer_nuisance`, 44 `semantic_context`), 120 normalized-skeleton groups, and zero
exact-text, forbidden-skeleton, shortcut, target-text, or provenance findings;
`passed=true`. This remains automated contract evidence, not a substitute for either
human review or a real artifact/release hash.

Exact duplicates are an unconditional failure. All normalized skeleton matches are
reported; the developmental gate rejects those that defeat the explicit template or
alias holdout. No maximum skeleton-share or n-gram threshold was preregistered, so the
reviewer must not invent one after seeing this candidate. Freeze pilot-informed
thresholds before the main data/test manifests are frozen.
