# ReactorBench-LM architecture

Status: Phase 3 technical pipeline implemented through the owner-review checkpoint on
2026-08-20; final repository verification and both human reviews remain open.

## Design objective

ReactorBench-LM keeps a wholly fictional source of truth separate from observations,
canonical events, rendered text, and model behavior. The state generator—not prose or
a learned model—owns every label.

```text
scenario definition -> latent Aster state -> observations -> canonical events
                    -> audit trajectory -> split-first task projection -> renderer
                    -> tokenizer/models -> evaluation -> future narrow service/UI
```

## Contracts and boundaries

| Layer | Responsibility | Must not do |
|---|---|---|
| Scenario | Select validated fictional variant, driver, fault, context, and actions | Carry real data or infer truth from prose |
| Latent state | Apply bounded deterministic fictional transitions | Model real physics, procedures, or setpoints |
| Observation | Create bounded channels/status/quality from latent state | Reveal latent fault truth |
| Event/target | Record canonical evidence and structured truth | Depend on language rendering |
| Audit payload | Carry allowlisted review data | Serve directly as a model prompt |
| Phase 3 projection | Produce split-first, decision-tick/channel-limited task inputs | Include audit IDs, truth, later outcomes, or target shortcuts |
| Renderer | Convert only strict `ModelInput` records with a reviewed project-authored catalog | Accept a raw trajectory, target, evidence annotation, or provenance record |
| Candidate artifacts | Preserve canonical JSONL, checksums, lineage, and review state | Overwrite an existing bundle or imply pending data is approved |
| Model/UI | Later learn or display only permitted artifacts | Ingest real-world material or claim operational usefulness |

The development schema and generator are `0.1.0` with `frozen=false`. Model instances
are immutable and strict, but the interface is intentionally not a frozen v1 release.

## Completed Phase 2 boundary

`src/reactorbench/simulator/` contains developmental G01–G15 fixtures: stable/no-fault
and benign load behavior; sensor drift, stuck, and noise; pump degradation and matched
standby-context trip; valve lag/stuck contrast; transfer-efficiency loss, flow
imbalance, support-power interruption, inventory loss; the narrow G14 compound; and
the sparse G15 abstention audit fixture. Aster-A, Aster-B, and Aster-C use immutable
project-authored cards/maps and all supported builders fail closed outside their exact
contracts.

The generator uses deterministic local RNG streams, normalized bounded values, two
fictional channels per variable, and explicit latent/observation/event/target
separation. G14 labels are a semantic set and serialize under D-039 as
`(SENSOR_DRIFT, PUMP_DEGRADATION)`. No Phase 2 fixture is a frozen golden example or a
training record.

## Phase 3 implemented boundary

Phase 3 now implements the strict boundary that Phase 2 left open. A deterministic
development configuration assigns structured scenarios, seeds, renderer families,
alias families, compositions, and counterfactual relatives to splits before prose can
exist. Its measured pre-render inventory is 204 trajectories, 1,762 single-input
projections, and 14 paired counterfactual comparisons.

`ModelInput` is the renderer's only accepted source contract. Each projection stops at
its task-specific decision/event cut and exposes only prompt-local observation, event,
and bounded context facts. Source event IDs and relationships, evidence annotations,
scenario/trajectory IDs, seeds, latent state, injections, targets, provenance, and
later `ACTION_APPLIED` effects are held in separate audit contracts. G07 omits standby
context from fault-family and continuation views and permits its bounded semantic
relationship only for evidence, action, and summary views. G12 exposes dependency
edges only for the map-included sibling and emits no withheld-map marker. G15 exposes
one sparse primary-flow observation and no invented expanded sibling. Continuation
targets use an event-index-exclusive prefix and never select `ACTION_APPLIED` as the
next-event target. The six otherwise eligible continuation projections in
`component_test` are deliberately absent: their held-out alias would identify the
next-event label within that task. `component_test` therefore has no continuation
coverage and supports no component-generalization claim for `continue_log`. The
remaining task inventory is 148 continuation, 399 fault-family, and 405 each for
evidence extraction, next action, and incident summary.

Evidence projection closes the visible-target loop. All 405 evidence targets contain
at least one prompt-local fact reference, and every reference resolves to an included
observation, event, or context fact. Four G12 map-withheld targets intentionally omit
`MAPPED_COMPONENT_CHANGE` because their input does not expose the dependency map; this
is a declared semantic omission, not missing target data.

Groups are derived and assigned atomically before rendering. Implemented complete
groups cover G07, G08/G09, G12, and G14. G14 includes the compound plus pump-only and
sensor-only comparators; the sensor-only comparator is the exact primary-thermal
sensor-drift fixture required to keep component/channel roles aligned. G15 has 24
explicitly incomplete sparse-only groups because the proposed evidence-expanded
siblings are not generator-supported. Incomplete G15 groups cannot be promoted to
paired comparisons.

Each of the 14 comparison pairs isolates one preregistered causal factor or
intervention. That factor may create several bounded visible consequences, and the
pair preserves every prompt-local decisive fact; the benchmark does not mislabel these
as literal one-fact text edits.

The renderer is deterministic and entirely project-authored: four template families
by four alias families by eleven event types create 176 mandatory catalog entries. The
pre-render packet additionally presents the actual observation/status/quality,
state/mode, context, event-clause, guard, and corruption wording surfaces. It binds
those authored surfaces to the exact resolved configuration, generator commit,
structured bundle, split manifest, and complete structured-target inventory. Full
development-candidate assembly is blocked unless a `project-owner` record approves
that exact packet and binding. After rendering, a separate full candidate packet must
receive human review. A test-only approval fixture exercises 553 distinct rendered
contexts, 1,776 task records, and 18 bounded corruptions, but it is not human evidence
and no real narrative candidate or approved artifact exists.

Exact text and structured-input duplicates are unconditional failures. The structured
gate separately reports zero duplicate single-input fingerprints and zero duplicate
paired counterfactual fingerprints. The text audit reports all normalized skeleton and
normalized-token 3/4/5-gram overlap. Its shortcut analysis is scoped by task and emits
the full task-scoped marginal table plus pairwise and full renderer-plan
`renderer_nuisance` interactions. `semantic_context` features are reported in their own
class, while only exclusive nuisance features/interactions can raise a shortcut
finding. Skeleton overlap fails the current developmental gate when it violates the
explicit template-family or alias-family holdout; no maximum skeleton-share or n-gram
threshold is retrofitted
after seeing the developmental output. Those thresholds remain subject to
preregistration after pilot evidence and before a frozen dataset.

Five explicit alias-plan overrides rebalance the measured joint nuisance cues before
freeze. The plan keys use only split, seed, and case, never the runtime task target, so
style assignment remains split-first and deterministic rather than answer-selected.

The content guard and copied-span fingerprints are deterministic, non-exhaustive
review aids. A zero-finding scan cannot prove the absence of prohibited material. The
mandatory human reviews remain the safety boundary before candidate use.

## Artifact lineage

```text
source revision -> generator/config/schema -> structured projection audit
-> hash-bound authored-language + structured-target review
-> pending rendered candidate + quality report
-> post-render review -> tokenizer -> model/checkpoint -> evaluation -> UI release
```

Development artifacts use canonical JSONL because the current scale does not justify a
binary-table dependency. The sole write-capable command resolves a validated project
checkout from the config and writes once to `data/generated/<artifact_name>`; arbitrary
output roots and directories are not accepted. The path is contained and bounded by
explicit file-count, record-count, per-file-byte, and total-byte limits. The manifest
binds the candidate, resolved configuration, structured bundle, split manifest, both
schema snapshots, catalog, guard, pre-render packet and
record, post-render packet, and quality report. Verification re-parses every JSONL file
through its strict typed contract and checks all cross-file links and hashes, rather
than validating bytes alone. Dataset contract snapshots are packaged with the wheel.
At verified implementation commit
`d3d22b7f9b2888d281c1c92cd283e10b4f0e3af1`, the full Python 3.12.11, Ruff, strict
mypy, 649-test/87.24%-branch-coverage, build, and isolated no-network artifact gates
passed. The implementation/schema/config and package hashes are recorded in
`docs/IMPLEMENTATION_STATUS.md`. The real candidate, quality report, post-render
packet, and candidate artifact manifest have intentionally not been created because
the mandatory project-owner pre-render review is not complete. No approved dataset,
tokenizer, model, checkpoint, service, UI, measured model result, GitHub push, or
deployment exists in this checkpoint.

## Future service boundary

After data/model gates pass, the browser will call a server-side gateway with bounded
schemas; the gateway will call a narrow inference service; and that service will load
only fixed, checksummed, data-only artifacts. Model text remains untrusted display data.
These are planned boundaries, not an implemented deployment claim.
