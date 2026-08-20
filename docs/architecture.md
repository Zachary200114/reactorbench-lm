# ReactorBench-LM architecture

Status: Phase 1 complete; selected Phase 2 generator milestones locally integrated and verified on 2026-08-20

## Design objective

ReactorBench-LM separates a wholly fictional source of truth from its observable narratives and from every model trained on those narratives. This makes causal labels auditable and prevents model output from becoming ground truth.

```text
project-authored scenario definition
                 |
                 v
        latent Aster state machine
                 |
                 v
        bounded observation model
                 |
                 v
        canonical event extraction
                 |
                 v
        deterministic text renderers
                 |
                 v
 split-first manifests and dataset views
                 |
                 v
 tokenizer -> baselines -> Transformer
                 |
                 v
 versioned evaluation and result artifacts
                 |
                 v
 narrow inference service -> Research Editorial UI
```

Only the left-to-right data exposed by a task contract reaches a model. Latent fault injection and withheld ground truth must never leak into visible observations or model inputs unless a specific supervised target explicitly requires them.

## Layers and contracts

| Layer | Responsibility | Must not do |
|---|---|---|
| Scenario definition | Select fictional plant variant, driver, faults, onset, severity, seed, and ordered `{decision_tick, action}` entries | Contain real facility data or depend on rendered prose |
| Latent state | Apply bounded deterministic transitions and injected faults to the invented state graph | Copy real physics, setpoints, procedures, or layouts |
| Observation | Convert latent variables into bounded channels with explicit quality and status | Alter latent process state or reveal hidden fault labels |
| Canonical event | Record exact fictional events, evidence slots, and versioned identifiers | Infer truth from natural language |
| Renderer | Produce traceable project-authored narratives from canonical events | Use a hosted LLM or introduce unlabeled facts |
| Dataset | Materialize structured, narrative, and task views from frozen split manifests | Randomly split rendered text or edit generated records by hand |
| Tokenizer and models | Learn only from permitted training artifacts | Load pretrained core-model weights or test data |
| Evaluation | Compare fixed predictions with structured truth on separately reported splits | Select checkpoints using test results or hide failures |
| Inference service | Offer a narrow, bounded operation over curated fictional scenarios | Accept files, URLs, checkpoints, paths, or real logs |
| Web application | Present model evidence, ground truth, versions, failures, and limitations | Become an unrestricted chat box or realistic control room |

## Phase 1 package boundary

The initial package uses `src/reactorbench/` and is deliberately small:

```text
src/reactorbench/
  config.py
  schemas/
    base.py
    enums.py
    latent.py
    observation.py
    events.py
    scenario.py
    target.py
    provenance.py
    export.py
```

The semantic separation is required even if package organization evolves later. The schema version begins at development version `0.1.0`.

The shared model base is strict, rejects unknown fields, validates defaults, and
makes each validated model instance immutable. Normalized continuous fields require
finite values in `[0, 1]`; identifiers and nonnegative integers have explicit bounds.
The development interface is separately marked `frozen=false`, so reviewed contract
changes remain allowed before version 1. Instance immutability must not be confused
with interface freeze. Canonical JSON and a SHA-256 manifest provide deterministic
snapshots for schema review. These contracts now compose the first structured-generator
milestone, but they are not yet a frozen version 1 interface and no dataset exists.

## Phase 2 simulator boundary

`src/reactorbench/simulator/` currently implements Aster-A stable operation, one
observation-only `SENSOR_DRIFT` case, a benign `LOAD_TRANSIENT`, and one constrained
`SENSOR_STUCK`-during-`LOAD_TRANSIENT` composition. It also implements one
observation-only `SENSOR_NOISE` case and the bounded `PUMP_DEGRADATION` and
context-aware `PUMP_TRIP` process-fault cases. It uses
deterministic local random streams, two fictional channels per normalized variable,
bounded latent updates, and explicit observation/event/target separation. A same-seed
stable trace and drift trace have identical latent states; only the selected channel
and aggregate observation status may diverge after the declared onset.

The load transient changes demand, heat, flow, steam, and output through explicit
fictional stage lags, then returns to `STABLE`. Transfer efficiency remains unchanged,
both observation channels agree, and structured truth is `NO_FAULT`. Its behavior is
fully derived from the validated driver, seed, duration, and generator version rather
than from unrepresented caller input.

The stuck-load case is limited to Aster-A, exactly one indefinite low-severity
`SENSOR_STUCK` injection, and either electrical-output channel. At tick 2 the selected
channel freezes at its tick-1 observed value while the load transition begins. The
latent trace remains exactly equal to the same-seed benign load trace, and every
nonselected observation remains unchanged. A genuine redundant-channel response and
channel disagreement support `VERIFY_REDUNDANT_CHANNEL` at decision tick 5; that action
is applied at tick 6, when coordinated load evidence is also recorded. The second
decision, `FLAG_SENSOR_SUSPECT`, occurs at tick 6 and is applied at tick 7, after which
the selected channel quality is `SUSPECT`. Unsupported variants, channels, severities,
durations, action sequences, identifiers, container shapes, drivers, and extra faults
fail closed. This developmental fixture informs G04 but does not freeze the golden
scenario; the required human review has not occurred.

The noise case is limited to Aster-A steady operation and exactly one indefinite
low-severity `SENSOR_NOISE` injection on either primary-thermal-state channel. Its
selected channel matches the same-seed stable trace through onset tick 2, then receives
deterministic alternating offsets whose paired magnitudes are sampled in the normalized
`[0.018, 0.024]` interval. Each adjacent pair has equal magnitude and opposite signs;
the next pair may have a different seeded magnitude. Latent state and every nonselected
observation remain exactly equal to the stable trace. Decisions at ticks 3 and 4 are
`UNRESOLVED` with `INSUFFICIENT_EVIDENCE`; the tick-5 diagnosis selects
`COMPARE_RELATED_TRENDS`, and stable-related evidence at tick 6 supports
`FLAG_SENSOR_SUSPECT`. Applied actions occur one tick later. The selected quality
changes from `GOOD` to `SUSPECT` at tick 7. The model-visible trace never uses the
lexical `NOISY` quality label as a shortcut before diagnosis—or afterward. This
developmental fixture informs G05 but does not freeze it; human review remains pending.

The first process-fault case is limited to one indefinite low-severity
`PUMP_DEGRADATION` injection on either fictional Aster-A primary train during steady
operation. The selected component becomes `DEGRADED` at tick 2, followed by primary
flow at tick 3, primary thermal state at tick 4, steam at tick 5, and turbine/electrical
output at tick 6. Normal same-seed observation-noise residuals remain unchanged, both
channels for each process variable agree and remain `GOOD`, and transfer efficiency is
invariant. A tick-4 short-evidence decision abstains; correlated evidence supports a
tick-6 inspection request and a tick-7 simulated-load reduction decision. Inspection
and reduction apply one tick later, with only the selected component marked pending
maintenance and the operating mode entering `RECOVERY` at tick 8 while degradation
persists. This developmental fixture informs G06 but does not freeze it; human review
remains pending.

The pump-trip fixture is limited to one indefinite low-severity `PUMP_TRIP` on either
fictional Aster-A primary train during steady operation. It uses a strict immutable
`StandbyContext` containing exactly one context identifier, the distinct active and
standby train identifiers, standby state, mapped support-bus identifier and state, and
a positive start delay. Aster-A's wholly project-authored dependency card maps Cirrus
to Rill and Kestrel to Quill one-to-one and gives standby start a one-tick delay. These
names and relationships are design constants in the invented world, not a model of a
real facility.

Matched `AVAILABLE` and `UNAVAILABLE` contexts share the same causal trip prefix. At
tick 2 the active train changes from `AVAILABLE` to `UNAVAILABLE` and the mode changes
from `STABLE` to `DISTURBED`; tick 3 contains the seed-derived abrupt primary-flow loss,
which is the sole normal per-tick step-bound exception; tick 4 contains a bounded
primary-thermal rise; and tick 5 contains delayed steam, turbine, and electrical-output
declines and the `PUMP_TRIP` diagnosis. In the available branch, tick 5 selects the
standby, tick 6 applies the selection and changes the standby to `STARTING`, and tick 7
changes it to `RECOVERING`, begins bounded partial flow recovery, and enters `RECOVERY`
only after the visible recovery. In the unavailable branch, tick 5 selects simulated
load reduction; tick 6 applies it, lowers fictional heat/load targets, and selects the
simulated stable-state action; tick 7 applies that action and enters `STABILIZED` while
the standby remains unavailable and flow does not recover. The active train remains
unavailable in both branches. This developmental fixture informs G07 but does not freeze
it; human review remains pending.

The public structured payload is an exact allowlist containing schema version, safe
standby context or `null`, observations, and canonical events. Scenario injections,
action sequences, latent truth, fault labels, targets, and provenance remain outside
that payload. Before Phase 3 narrative/task generation, the availability-bearing
`context_id` must be filtered from prompts, matched context pairs must be grouped before
split assignment, and context presence/templates must be balanced so the G07-only
object or tick-0 note cannot become a fault-label shortcut.
Decision labels are recorded at their decision tick; any corresponding applied-action
event occurs on the next tick so an action cannot precede its supporting evidence.
Task inputs must be constructed from prefixes ending at the decision tick; later
`ACTION_APPLIED` events and their consequences cannot be allowed to reveal that target.

The current prohibited-content guard is a deterministic, redacting structural gate.
It is intentionally non-exhaustive and does not replace the reviewed denylist and
human sample procedure required before dataset work.

## Artifact lineage

Every serious artifact must form a verifiable chain:

```text
source revision
  -> resolved generator configuration and schema version
  -> trajectory and split manifests
  -> rendered dataset manifest
  -> tokenizer configuration and checksum
  -> model/training configuration
  -> checkpoint and checksum
  -> evaluation configuration and predictions
  -> figures, report, and local UI release identifier
```

Generated corpora, checkpoints, and run directories must not silently overwrite earlier artifacts. Estimates and interface mock values never enter the results chain.

## Future local service boundary

After model and evaluation gates pass, the local product may add:

```text
untrusted browser
      |
      v
server-side gateway: request schema, body/method limits, rate policy, safe errors
      |
      v
inference service: fixed tokenizer/checkpoint, bounded work, structured response
      |
      v
checksummed read-only release artifacts
```

The browser never selects paths, models, token limits, execution devices, or arbitrary content. The inference service revalidates security-critical fields and returns allowlisted structured data. Model output remains untrusted display text.

## Current implementation boundary

The Phase 1 foundation and selected Phase 2 generator milestones are present. The recorded
2026-08-20 Python 3.12 gate passed formatting across 52 files, lint, strict typing, 206 tests, and
92.57% total coverage with branch measurement enabled, plus distribution builds and
isolated wheel verification. Phase 2 is not
complete: additional process faults, compositions, variants, and broader generator gates
still need implementation. Dataset generation, tokenizer training, model training,
measured evaluation, inference serving, and UI construction remain later work. Refer to
[the implementation status](IMPLEMENTATION_STATUS.md) for the exact evolving integration
state rather than inferring completion from this architecture.

The current task prepares everything locally but excludes GitHub push and Vercel or inference-host deployment.
