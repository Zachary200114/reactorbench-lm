# ReactorBench-LM dataset specification

Status: Phase 3 is complete locally, the intended-repository verification gate passed,
and the exact pre-render and post-render packets were approved by the project owner.
The approved development candidate is not a public release. Phase 4 has trained a
project tokenizer only on its `iid_train` prose and produced a local smoke checkpoint;
no Phase 5 baseline, pilot, holdout evaluation, or released model exists.

## 1. Dataset purpose

Create a wholly synthetic, simulator-grounded language dataset for training and evaluating small decoder-only Transformers on fictional system-event sequences.

The dataset must support:

- causal next-token modeling;
- next-event label prediction;
- fault-family identification;
- evidence extraction;
- concise incident summarization;
- fictional next-action label selection;
- abstention under incomplete or conflicting evidence;
- controlled in-distribution and compositional generalization studies.

It must not support or imply real nuclear operations, emergency response, engineering design, maintenance, licensing, security analysis, or operator training.

## 2. Provenance model

Every released example must be reproducible from:

- `dataset_version`
- `generator_commit`
- `scenario_schema_version`
- `renderer_version`
- `seed`
- `scenario_id`
- `plant_variant_id`
- `fault_family_ids`
- `template_family_ids`
- `split_name`
- `task_name`

The developmental trajectory-index and split-manifest contracts add two explicit
matched-context fields:

- `counterfactual_group_id`: one stable identifier derived only from the pair's
  preregistered shared factors. Its derivation explicitly excludes every decisive
  varied factor, target label, rendered string, and post-decision outcome;
- `counterfactual_variant_id`: the bounded semantic role that differs inside that group,
  such as `standby_available` or `standby_unavailable`.

For G07 specifically, group siblings share plant variant, seed, active component role,
duration, `PUMP_TRIP` family, onset, severity, and pre-branch causal schedule; standby
availability is the varied factor. Other counterfactual families may vary the fault
itself—for example, the implemented G08/G09 lag-versus-stuck comparison—so a generic group
key must not assume that fault family, onset, or severity are always shared.

These fields are grouping metadata, not prompt text or target labels. Phase 3 implements
them in separate strict grouping and split-manifest contracts, which are included in the
developmental dataset snapshot. A raw simulator `context_id` is audit metadata and must
never substitute for these fields because its value can encode the availability word.
The snapshot is reviewed implementation evidence, not a version-1 schema freeze.

## 2.1 Implemented projection and group audit

The completed G01–G15 generator produces truth-filtered audit trajectories, not model
prompts. Phase 3 implements a strict, read-only projection that (1) ends each decision
task at its exact decision/event cut, (2) selects only task-allowed event/channel/context
facts, and (3) rejects latent state, injection, targets, source IDs and evidence
annotations, scenario/provenance IDs, audit-only context identifiers, and later action
effects. Full audit trajectories remain available only for provenance and review.

The split manifest assigns every supported related family as a group before rendering:

- G07 matched standby-availability contexts;
- G08/G09 lag-versus-stuck resolution/persistence counterfactuals;
- G12 included versus withheld dependency-map contexts;
- G14 compound, its single-factor comparators, and affected component/channel roles; and
- G15 sparse evidence only. The 24 sparse groups are explicitly incomplete because
  evidence-expanded relatives are not generator-supported.

The current deterministic structured audit contains 204 trajectories, 1,762
single-input projections, and 14 atomic counterfactual pairs. It reports and rejects
group separation, target/context or renderer-plan shortcuts, post-decision leakage,
duplicate structured records, and prohibited cross-split
scenario/template/component/fault-composition overlap. G07 omits standby context from
fault-family and continuation views and exposes its semantic relationship only for
evidence, action, and summary views. G12 exposes dependency links only for the included
sibling and emits no withheld marker. G15 exposes one sparse primary-flow fact. It
never claims that a truth-filtered payload is automatically safe prompt text.

Task counts are 148 `continue_log`, 399 `fault_family`, and 405 each for
`extract_evidence`, `next_action`, and `incident_summary`. Six otherwise eligible
`component_test` continuation projections are intentionally excluded because the
held-out component alias identified their next-event target inside that task. The
split therefore has no `continue_log` coverage and supports no component-generalization
claim for that task. The quality report is task-scoped and contains every
marginal feature/target contingency plus pairwise and full renderer-plan
`renderer_nuisance` interactions, rather than aggregating incompatible labels across
tasks. `semantic_context` remains in the report under its own feature class, while only
exclusive nuisance features or interactions may raise shortcut findings.

Five explicit alias-plan overrides rebalance the measured joint nuisance cues before
freeze. Their keys contain only split, seed, and case; generation never looks up a
runtime target to choose a renderer plan. D-059 records the exact five entries.

All 405 evidence targets are nonempty, and every target fact reference resolves to a
fact actually visible in that projection. Four G12 map-withheld targets intentionally
omit `MAPPED_COMPONENT_CHANGE`: those prompts withhold the dependency map, so including
that semantic slot would create an ungrounded target. This bounded omission is counted
and tested rather than silently discarded. The structured gate separately requires
`single_input_structured_duplicate_count=0` and
`counterfactual_input_structured_duplicate_count=0`.

G14's single-factor set uses the same-role pump comparator plus an exact primary-
thermal sensor-drift comparator for the sensor-only member. This prevents a component
or channel-role mismatch from masquerading as compound-composition generalization.

No manually edited generated record should enter a release. If a template or rule is corrected, regenerate the affected shard and bump the appropriate version.

## 3. Three canonical data views

### A. Structured trajectory view

Recommended format: Parquet.

One row per event/time step, containing exact simulator state and ground truth. This is the audit source and should not necessarily be exposed verbatim to the model.

Suggested fields:

| Field | Type | Meaning |
|---|---|---|
| `trajectory_id` | string | Stable unique trajectory identity |
| `scenario_id` | string | Structured scenario definition identity |
| `event_index` | integer | Monotonic event order |
| `sim_time` | integer | Fictional elapsed time tick |
| `plant_variant_id` | string | Invented topology/configuration variant |
| `operating_mode` | enum | Synthetic mode such as steady, transition, recovery |
| `fault_family_ids` | list[string] | Injected ground-truth fault families; excludes benign drivers such as `LOAD_TRANSIENT` |
| `fault_onset_ticks` | list[int] | Fictional onset positions |
| `severity_band` | enum | low, medium, high synthetic severity |
| `component_states` | struct/map | Discrete invented component states |
| `sensor_values` | struct/map | Normalized values only |
| `observation_status` | struct/map | normal, watch, abnormal, missing, conflicting |
| `channel_quality` | struct/map | good, suspect, unavailable, noisy |
| `event_type` | enum | Ground-truth event category |
| `evidence_slots` | list[string] | Facts supporting the ground-truth target |
| `action_label` | enum | One immediate fictional next-action label at this decision tick |
| `action_sequence` | list[`{decision_tick: int, action: enum}`] | Ordered trajectory-level sequence; exactly one immediate fictional action per listed decision tick |
| `standby_context` | struct/null | Bounded fictional context for context-aware trajectories; audit source, not automatically model input |
| `counterfactual_group_id` | string/null | Split-group key shared by matched context variants |
| `counterfactual_variant_id` | enum/null | Semantic role within a matched counterfactual group |
| `policy_state` | enum | Fictional policy-card state |
| `is_ambiguous` | boolean | Whether abstention is expected |

`action_sequence` entries are ordered by increasing `decision_tick`, ticks are unique
within a trajectory, and each `action` is an exact member of the fictional action enum.
For example:

```json
[
  {"decision_tick": 3, "action": "VERIFY_REDUNDANT_CHANNEL"},
  {"decision_tick": 6, "action": "FLAG_SENSOR_SUSPECT"}
]
```

### B. Narrative corpus view

Recommended format: text shards plus a JSONL index.

Contains rendered event logs, shift summaries, fictional policy cards, and incident narratives for causal language-model training. Text must be generated from project-authored grammars and phrase banks with traceable template IDs.

Example style—not final training content:

```text
[T+018] Primary Train B flow trend changed from stable to falling.
[T+020] Channel P-2 disagreed with its redundant channel.
[T+023] Support power remained available. Pump state was unchanged.
```

All tags, component names, values, and relationships are fictional.

The current catalog is project-authored and deterministic: four template families by
four alias families by eleven event types create 176 full-preview combinations. No
production narrative view exists until the exact hash-bound pre-render catalog review
is approved by the project owner. A second human review is required for the exact
post-render inventory before candidate use.

### C. Task view

Recommended format: JSONL.

One prompt/target record per supervised task:

```json
{
  "example_id": "...",
  "task": "fault_family",
  "context": "...synthetic narrative...",
  "target": "SENSOR_DRIFT",
  "evidence": ["CHANNEL_DISAGREEMENT", "RELATED_STATE_STABLE"],
  "split": "composition_test"
}
```

The target format should be easy to parse and score. Natural-language explanations can accompany structured targets, but structured labels remain authoritative.

Every decision-task record must be built from a prefix ending at its exact
`decision_tick`. Events and state consequences after that tick—including a later
`ACTION_APPLIED` event for the target action—are excluded. An applied action from an
earlier decision may be included for a later decision only when it occurred on or
before the later decision tick and the task contract explicitly permits that causal
history. Audit-only fields such as `context_id`, scenario identity, fault injection,
latent truth, and provenance are never copied into a rendered prompt.

## 4. Corpus composition targets

These are planning ranges, to be finalized after pilot throughput and learning curves:

| Release | Trajectories | Approximate tokens | Purpose |
|---|---:|---:|---|
| Development | 100–500 | under 500K | Schema, invariants, and smoke tests |
| Pilot | 5K–10K | 2M–5M | Establish learnability and template quality |
| Version 1 | 50K–100K | 25M–50M | Main approximately 15M-parameter model |
| Stretch | 200K+ | 100M–200M | Scaling experiment only if compute justifies it |

Use observed validation curves rather than assuming that more synthetic repetition always helps.

## 5. Scenario balance

Recommended training mixture:

- 25–35% normal and benign transient trajectories.
- 35–45% single-fault trajectories.
- 15–25% compound-fault trajectories.
- 5–10% ambiguous/incomplete-evidence trajectories.
- 5–10% corrupted/noisy narrative variants.

Avoid perfect class balance if it creates an implausibly easy dataset, but cap extreme imbalance and report every class count.

## 6. Narrative-generation strategy

### Hand-authored grammar

Build a compositional renderer from:

- event templates;
- synonyms and controlled paraphrases;
- optional clauses;
- temporal connectors;
- active/passive phrasing variants;
- component alias families;
- benign distractor events;
- different shift-note styles.

Do not use a frontier LLM to write training examples. That would blur ownership, introduce unknown source patterns, and weaken the “built from scratch” story.

### Diversity controls

- Maintain at least several template families per event type.
- Partition template families across train and template-holdout test sets.
- Track lexical overlap and duplicate n-grams across splits.
- Report every repeated rendered string and normalized template skeleton. Exact text
  duplicates fail. During development, skeleton overlap fails when it violates the
  explicit template-family or alias-family holdout. Preregister a maximum skeleton
  share only after pilot evidence and before freezing the dataset; do not select it
  after inspecting a held-out result.
- Generate matched counterfactual pairs that differ in one preregistered causal factor
  or intervention. That factor may produce multiple visible evidence facts; report all
  decisive prompt-local facts rather than pretending the rendered traces differ in a
  single line.
- Balance and filter structural context cues. A non-null `standby_context`, a G07-only
  tick-0 note, or one template family must not by itself identify `PUMP_TRIP` or its
  action label; add suitable negative controls or exclude the structural field from a
  task view until that shortcut test passes.
- Compute shortcut contingencies separately for each task. Report the full table for
  every audited marginal feature value and the pairwise/full-plan
  `renderer_nuisance` interactions against structured targets, even when no finding is
  raised.
  Report `semantic_context` under its own feature class rather than treating it as a
  `renderer_nuisance` feature. The current matrix excludes six
  `component_test` continuation examples because the held-out alias would otherwise be
  a sole-target cue for that task. Report that this leaves no component-test
  continuation coverage or component-generalization claim for `continue_log`.
- Include both explicit and indirect evidence, but never require outside nuclear knowledge.

### Controlled noise

- Omit a noncritical log line.
- Duplicate an event line.
- Insert a benign unrelated event.
- Use a suspect sensor channel.
- Reorder entries only when timestamps preserve the real synthetic order.
- Contradict channels to create an abstention case.

Noise must be applied before target computation when it affects evidence sufficiency.

For the current renderer-only `noise_test`, the four bounded corruption plans are
assigned across multiple task outcomes as a balanced matrix so the plan itself is not a
sole-target cue. Corruption provenance remains separate from simulator fault truth.

`SENSOR_NOISE` and `noise_test` are separate dimensions. `SENSOR_NOISE` is structured
simulator ground truth for an injected observation-layer fault. `noise_test` is a later
evaluation split for controlled corruption of otherwise generated examples. A scenario
may have either, both, or neither; assigning an example to `noise_test` must not itself
add a `SENSOR_NOISE` label, and narrative corruption must never become ground truth.

## 7. Split policy

### Rule

Assign splits from structured scenario definitions **before** narrative rendering.

### Recommended split families

- `iid_train`, `iid_validation`, `iid_test`
- `template_test`
- `component_test`
- `severity_test`
- `composition_test`
- `counterfactual_test`
- `noise_test`

### Leakage controls

- No shared `scenario_id` or random seed across splits.
- No template-family overlap in `template_test`.
- No held-out component aliases in training.
- No held-out fault pair in training, even if order is reversed.
- No driver-plus-fault composition assigned to `composition_test` may occur in
  training on another seed, channel, alias, or component role. Training may contain
  that benign driver alone and that fault alone where each is otherwise valid, but
  never their held-out composition.
- Compute `counterfactual_group_id` before split assignment and assign the entire group
  atomically. Matched standby-available and standby-unavailable siblings must never be
  placed in different splits; both members of a `counterfactual_test` pair must remain
  together so comparison is possible without train/test sibling leakage.
- Do not derive a split from `context_id`, rendered availability words, a target label,
  or a post-decision outcome. Grouping uses the preregistered shared-factor definition
  and is computed independently of model input text and targets.
- Preserve and report counts by fault family, context presence, and
  `counterfactual_variant_id`. For G07-derived records, require paired 1:1 availability
  roles before rendering; any later filtering must remove or retain the pair together.
- Reject duplicate structured model inputs separately for single-input task records and
  two-input counterfactual task records; report both duplicate counts even when zero.
- Reject exact text duplicates globally. Report every normalized skeleton overlap and
  reject overlap that defeats the explicit template-family or alias-family holdout.
- Measure normalized-token 3/4/5-gram overlap across train/test and publish it. The
  development report is descriptive because no threshold was preregistered.
- Freeze test manifests before the main run.

## 8. Task definitions

### `continue_log`

Input: prefix of a synthetic event narrative.
Target: next text tokens and separately the next structured event type.

The structured next-event view uses an event-index-exclusive prefix and never selects
`ACTION_APPLIED` as its target. This prevents the applied form of a policy label from
turning continuation into target disclosure.

### `fault_family`

Input: a complete or partial synthetic narrative.
Target: diagnostic status plus zero or more fault-family labels. An abstaining target uses status `UNRESOLVED`, an empty fault-label list, and reason `INSUFFICIENT_EVIDENCE`.

### `extract_evidence`

Input: synthetic narrative.
Target: canonical evidence slots present in the input.

Every target slot must carry one or more prompt-local fact references, and each
reference must resolve to a visible observation, event, or context fact. The current
development matrix has 405 nonempty evidence targets and zero unresolved references.
For four G12 map-withheld projections, the target deliberately omits
`MAPPED_COMPONENT_CHANGE` because no dependency-map fact is visible.

### `next_action`

Input: narrative plus optional fictional policy card.
Target: one fictional action label. It is never a real procedure.

The input ends at the action's `decision_tick`; its own later application and effects
are forbidden from the prompt. Context-aware examples expose only the semantic facts
needed by the fictional task, never audit identifiers whose spelling encodes the
answer.

### `incident_summary`

Input: event narrative.
Target: compact structured summary with affected invented subsystem, observed trend, fault label if supported, system state, and fictional action label.

### `counterfactual_compare`

Input: two matched narratives differing in one preregistered causal factor or
intervention. Its bounded causal consequences may change multiple visible facts.
Target: identify which structured conclusion changes and why, using canonical evidence
labels and prompt-local decisive facts. The benchmark reports the number of differing
visible facts for each pair; it does not claim a literal one-line edit.

## 9. Validation and quality gates

### Generator invariants

- Values remain within their normalized domains.
- State transitions follow the invented transition graph.
- Fault onsets precede their modeled effects.
- Ground-truth evidence is present unless ambiguity is deliberately labeled.
- Action labels are valid for the fictional policy state.
- Normal trajectories contain no injected fault label.

### Text checks

- No exact duplicates within or across released splits.
- Every narrative maps back to a structured trajectory.
- Every mentioned value/status matches the structured record.
- No target appears accidentally in the input unless the task intends it.
- Every decision-task input ends at the declared `decision_tick`; no later event,
  `ACTION_APPLIED` record, mode change, or recovery/stabilization effect is present.
- `context_id`, scenario identity, counterfactual group identifiers, and provenance do
  not appear in narrative or task prompts.
- Matched context pairs share one `counterfactual_group_id`, have distinct bounded
  variant roles, remain in one split, and survive filtering together.
- Context presence, tick-0 context-note templates, and availability wording are tested
  for fault-family and action-label shortcuts; contingency counts are reported.
- Shortcut checks are task-scoped and publish full feature/target contingency counts;
  `semantic_context` is reported separately, and only `renderer_nuisance` exclusivity
  in marginal or pairwise/full-plan features raises a finding. Use path-aware
  categorical target keys for within-task
  contingencies and distinct leak-oriented labels for input-text leakage scans.
- The quality report binds `task_record_count=1,776` and every audited task-record
  ID/hash, including both render foreign keys for all 14 paired examples; preserve the
  exact inputs as `task-shortcut-records.jsonl`.
- Duplicate single-input and paired counterfactual structured fingerprints both equal
  zero before rendering.
- No malformed timestamps, component aliases, or unfinished templates.
- No real facility names, people, contact information, addresses, emails, phone numbers, or real incident identifiers.

### Prohibited-content scan

Maintain a versioned denylist and pattern suite covering:

- Navy and naval nuclear terms;
- real U.S. and international nuclear facility names;
- NRC event and docket identifier formats;
- real procedure/checklist identifiers;
- common patterns for engineering setpoints and real operating units;
- security/safeguards terminology outside the project's purpose;
- URLs, emails, phone numbers, and postal addresses;
- copied spans from any reviewed reference document.

Automated scanning is a gate, not proof. The implemented denylist/pattern suite and
copied-span fingerprint registry are intentionally non-exhaustive. Before any local
candidate is generated, the project owner must review the exact hash-bound packet
containing all authored renderer and corruption language surfaces, the catalog and
guard, and the complete structured-target inventory. The packet is bound to the
resolved configuration, generator commit, structured bundle, and split manifest. After
generation, a separate owner review must cover the full distinct render inventory and
its quality report. Automated test-only approval fixtures are never human evidence.

## 9.1 Development checkpoint and artifact policy

The current configuration fixes 204 structured trajectories, 1,762 single-input
projections, and 14 counterfactual comparisons. The approved local path creates 553
distinct render candidates, 1,776 task examples, and 18 bounded corruption records.
The candidate and both review records are ignored local artifacts rather than committed
or distributed data.

With placeholder test commit `abcdef0`, the corrected fixture audits all 1,776 task
records and 1,977,422 rendered UTF-8 bytes. It reports 402 contingencies (358
`renderer_nuisance`, 44 `semantic_context`), 120 normalized-skeleton groups, and zero
exact-text, forbidden-skeleton, shortcut, target-text, or provenance findings;
`passed=true`. These measurements are contract evidence only, not real artifact or
release hashes, a human-approved candidate, or permission to train.

Canonical JSONL is the developmental storage format. At this scale it keeps every
record human-inspectable without adding a binary table dependency. The sole write-
capable command resolves a validated checkout from the config and targets
`data/generated/<artifact_name>`; arbitrary output roots/directories are rejected and
existing candidates are not overwritten. Output is canonical and bounded by explicit
file, record, per-file-byte, and total-byte limits. Its manifest binds the exact candidate,
resolved configuration, structured bundle, split manifest, Aster and dataset schema
snapshots, catalog, guard, pre-render packet and record, post-render packet, and quality
report. Verification re-parses every file through its strict typed contract and checks
all cross-file links and hashes. Dataset JSON Schema snapshots are packaged and
validated alongside the Aster snapshots. Parquet remains a later option only if
measured scale or analysis requirements justify it.

At generator implementation commit
`d3d22b7f9b2888d281c1c92cd283e10b4f0e3af1`, the intended-repository gate passed
Ruff formatting for 106 files, Ruff lint, strict mypy for 80 source files, and 649 tests
on Python 3.12.11 in 297.78 seconds with 87.24% branch coverage; build and isolated
no-network artifact verification also passed. The exact config, structured, split,
schema, catalog, authored-surface, guard, target-inventory, packet, and package hashes
are recorded in `docs/IMPLEMENTATION_STATUS.md`. The strict-parsed pre-render packet is
the ignored 896,151-byte local file
`artifacts/review/catalog-review-v0.1.0.json`, with raw-file SHA-256
`2bc3e226e202a4c5c9baddaef512cf195e6086db7194b158856a782bb880dfce`
and internal checksum
`faa50900db2890b3bc167a44aabcb416b0a3eaa756cb578978f8e58fc3a24b8a`.
The real candidate, passing quality report, post-render packet, and candidate artifact
manifest were created and typed-verified without overwrite. The project owner approved
the exact post-render packet on 2026-08-20. The separate approval record has internal
checksum
`e066d5944839423fdd6e49491dfa5b57867b0c753ac465afb4ac196e7a87958d`
and raw-file SHA-256
`001eca9925d9d6c5b8a50c9bd524cbe990b17b6a675b53200f16cf379d9b7af7`.
It closes Phase 3 for local tokenizer/model work without rewriting the candidate or
granting public distribution permission.

## 10. Data documentation

The released dataset card should cover:

- motivation and intended research uses;
- composition and class distribution;
- simulator and renderer versions;
- generation process and human-authored templates;
- split logic and leakage tests;
- known simplifications and biases;
- excluded sources and prohibited uses;
- licensing and attribution;
- checksums and reproducibility commands;
- change history between dataset versions.

This follows the transparency goal of “Datasheets for Datasets”: https://www.microsoft.com/en-us/research/publication/datasheets-for-datasets/

## 11. External-source decision

No external source is approved as narrative training data.

Permitted external use is limited to:

- public high-level reference reading;
- a manually screened vocabulary seed of isolated, non-procedural terms;
- architecture, evaluation, governance, and synthetic-dataset design research;
- citations in documentation.

The source-level decisions are recorded in `SOURCE_MANIFEST.csv`.

## 12. Ownership and licensing

The corpus is generated from project-authored rules and text templates rather than a
scraped corpus. D-092 selects 0BSD for original code, documentation, project-authored
synthetic data, and other original repository material unless a file explicitly says
otherwise. Third-party dependencies and referenced material keep their own licenses
and terms.

Public documentation must continue to identify NRC/DOE/IAEA sources as references,
not training inputs, and must not imply endorsement by any agency. Obtain legal
guidance if the project later ingests or redistributes third-party text.

This dossier is not legal advice.
