# ReactorBench-LM architecture

Status: Phase 2 structured generator complete locally on 2026-08-20; Phase 3 has not started.

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
| Renderer/model/UI | Learn or display only permitted artifacts | Ingest real-world material or claim operational usefulness |

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

## Phase 3 handoff boundary

The truth-filtered audit payload is not a prompt. Phase 3 must define a strict
projection that ends at each decision tick, selects only allowed event/channel facts,
excludes scenario IDs, injection/truth/provenance fields and post-decision effects, and
then assigns split manifests before any renderer runs. It must preserve groups and test
shortcuts for G07 matched standby contexts, G08/G09 resolution-versus-persistence
counterfactuals, G12 included/withheld map contexts, G14 factor/role relationships, and
G15 sparse/evidence-expanded relatives.

The Phase 2 prohibited-content scanner is a deterministic redacting gate only. It is
narrow by design and does not replace the reviewed full denylist or stratified human
sample review required before a pilot dataset is rendered.

## Artifact lineage

```text
source revision -> generator/config/schema -> trajectory and split manifests
-> rendered dataset -> tokenizer -> model/checkpoint -> evaluation -> UI release
```

No stage may silently overwrite artifacts. There is no dataset, tokenizer, model,
checkpoint, service, UI, measured result, GitHub push, or deployment in this checkpoint.

## Future service boundary

After data/model gates pass, the browser will call a server-side gateway with bounded
schemas; the gateway will call a narrow inference service; and that service will load
only fixed, checksummed, data-only artifacts. Model text remains untrusted display data.
These are planned boundaries, not an implemented deployment claim.
