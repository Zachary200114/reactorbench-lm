# ReactorBench-LM dataset specification

Status: design specification; no dataset has been generated yet.

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

Before the Phase 3 pilot, trajectory-index and split-manifest provenance must also add
two explicit matched-context fields:

- `counterfactual_group_id`: one stable identifier derived only from the pair's
  preregistered shared factors. Its derivation explicitly excludes every decisive
  varied factor, target label, rendered string, and post-decision outcome;
- `counterfactual_variant_id`: the bounded semantic role that differs inside that group,
  such as `standby_available` or `standby_unavailable`.

For G07 specifically, group siblings share plant variant, seed, active component role,
duration, `PUMP_TRIP` family, onset, severity, and pre-branch causal schedule; standby
availability is the varied factor. Other counterfactual families may vary the fault
itself—for example, the future G08/G09 lag-versus-stuck comparison—so a generic group
key must not assume that fault family, onset, or severity are always shared.

These fields are grouping metadata, not prompt text or target labels. The current
developmental `ProvenanceRecord` does not yet contain them, so either that schema or a
separate strict split-manifest contract must be extended and snapshot-reviewed before a
dataset pilot. A raw simulator `context_id` is audit metadata and must never substitute
for these fields because its current value contains the availability word.

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
- Limit any single rendered string or template skeleton to a declared maximum share.
- Generate matched counterfactual pairs where one evidence fact changes the target.
- Balance and filter structural context cues. A non-null `standby_context`, a G07-only
  tick-0 note, or one template family must not by itself identify `PUMP_TRIP` or its
  action label; add suitable negative controls or exclude the structural field from a
  task view until that shortcut test passes.
- Include both explicit and indirect evidence, but never require outside nuclear knowledge.

### Controlled noise

- Omit a noncritical log line.
- Duplicate an event line.
- Insert a benign unrelated event.
- Use a suspect sensor channel.
- Reorder entries only when timestamps preserve the real synthetic order.
- Contradict channels to create an abstention case.

Noise must be applied before target computation when it affects evidence sufficiency.

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
- Deduplicate exact text and normalized template skeletons globally.
- Measure n-gram overlap across train/test and publish it.
- Freeze test manifests before the main run.

## 8. Task definitions

### `continue_log`

Input: prefix of a synthetic event narrative.
Target: next text tokens and separately the next structured event type.

### `fault_family`

Input: a complete or partial synthetic narrative.
Target: diagnostic status plus zero or more fault-family labels. An abstaining target uses status `UNRESOLVED`, an empty fault-label list, and reason `INSUFFICIENT_EVIDENCE`.

### `extract_evidence`

Input: synthetic narrative.
Target: canonical evidence slots present in the input.

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

Input: two matched narratives differing in one evidence fact.
Target: identify which structured conclusion changes and why, using canonical evidence labels.

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

Automated scanning is a gate, not proof; manually review stratified samples from every dataset release.

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

## 12. Ownership and license planning

Because the released corpus will be generated entirely from project-authored rules and text templates, licensing will be simpler than a scraped corpus. Before public release:

- choose a dataset license, provisionally CC BY 4.0 or CC0;
- choose a code license, provisionally Apache-2.0 or MIT;
- document NRC/DOE/IAEA sources as references, not training inputs;
- do not imply endorsement by any agency;
- obtain legal guidance if the project later changes to ingest third-party text.

This dossier is not legal advice.

License selection is explicitly deferred under D-041. That deferral does not block
local schema, generator, or evaluation work, but no code or dataset distribution may
imply reuse permission until the applicable licenses are selected.
