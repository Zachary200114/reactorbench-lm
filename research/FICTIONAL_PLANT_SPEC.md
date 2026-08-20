# Fictional plant specification

Status: developmental design contract; G01–G15 fixtures and Aster-A/B/C cards are implemented locally, but the golden suite is not frozen
Plant type: wholly fictional, civilian, pressurized-water-inspired teaching abstraction
Working plant name: **Aster Station**

## 1. Purpose and boundary

Aster Station is an invented discrete-time state machine for generating internally consistent event narratives. It is not a digital twin, engineering model, thermohydraulic code, operator trainer, or approximation of any named reactor.

Public NRC and IAEA education materials support only the broad conceptual chain used here: an abstract heat source transfers heat through a closed primary side to a separate secondary side, which drives an electrical generation and heat-rejection chain. Every variable, topology choice, coefficient, threshold, fault rule, action, and identifier below is project-authored.

The specification intentionally excludes:

- neutron kinetics, fuel geometry, chemistry, radiation, dose, containment, and severe-accident modeling;
- protection-system logic, emergency procedures, real alarm priorities, and licensing limits;
- real dimensions, units, setpoints, capacities, component counts, and time constants;
- security, safeguards, physical access, cyberattack, and vulnerability scenarios;
- plant-specific and Navy-derived knowledge.

## 2. Modeling level

IAEA guidance distinguishes educational simulators by purpose and scope. ReactorBench-LM adopts an even narrower **fictional concept model**: enough causal structure to create testable language sequences, but deliberately unsuitable for real education or operations.

The simulator will use:

- discrete time ticks;
- normalized continuous state in `[0, 1]`;
- categorical equipment states;
- deterministic transition rules plus seeded bounded noise;
- explicit fault injection;
- a separate sensor-observation layer;
- a separate narrative-rendering layer.

This separation matters:

```text
latent fictional state
        ↓
component transition rules
        ↓
sensor observation model
        ↓
canonical event labels
        ↓
language renderer
        ↓
model prompt and target
```

The language model never defines ground truth. Ground truth comes from the structured state machine.

## 3. Topology

```text
HEAT_SOURCE
    │ synthetic heat
    ▼
PRIMARY_LOOP ───► TRANSFER_UNIT ───► SECONDARY_LOOP
    ▲                    │                    │
    │                    │                    ▼
PRIMARY_PUMPS            │               TURBINE_UNIT
                         │                    │
                         │                    ▼
                         └────────────── GENERATOR_UNIT
                                              │
                                              ▼
                                        ELECTRICAL_BUS

SECONDARY_LOOP ───► CONDENSER_UNIT ───► FEED_SYSTEM ───► SECONDARY_LOOP
                           │
                           ▼
                    HEAT_REJECTION

SUPPORT_POWER supplies pumps, valves, sensors, and controllers.
INSTRUMENTATION observes every subsystem through fictional channels.
```

The final topology may be rendered with different fictional component names, but the semantic graph remains versioned.

## 4. State domains

### 4.1 Global operating mode

- `STABLE`: conditions near current synthetic targets.
- `LOAD_CHANGE`: planned fictional demand transition.
- `DISTURBED`: one or more abnormal trends exist.
- `RECOVERY`: a valid fictional action has begun restoring state.
- `STABILIZED`: abnormal propagation has stopped at a reduced synthetic target.
- `UNKNOWN`: available observations cannot establish the mode.

These names describe the invented state machine only.

### 4.2 Component state

- `AVAILABLE`
- `DEGRADED`
- `UNAVAILABLE`
- `STARTING`
- `RECOVERING`
- `SUSPECT`
- `UNKNOWN`

### 4.3 Observation status

- `NORMAL`
- `WATCH`
- `ABNORMAL`
- `MISSING`
- `CONFLICTING`

### 4.4 Normalized variables

Every continuous variable is clipped to `[0, 1]`. Values have no physical unit.

| Symbol | Canonical field | Meaning inside the fictional world |
|---|---|---|
| `H` | `heat_source_level` | Abstract produced heat fraction |
| `PF` | `primary_flow` | Abstract primary circulation fraction |
| `PT` | `primary_thermal_state` | Abstract primary thermal-energy proxy |
| `PI` | `primary_inventory` | Abstract closed-loop inventory proxy |
| `TE` | `transfer_efficiency` | Heat-transfer effectiveness proxy |
| `SF` | `secondary_flow` | Abstract secondary circulation fraction |
| `SI` | `secondary_inventory` | Abstract secondary inventory proxy |
| `ST` | `steam_state` | Abstract steam-production proxy |
| `CF` | `condenser_function` | Abstract condenser availability/capacity |
| `HR` | `heat_rejection` | Abstract external heat-removal fraction |
| `TO` | `turbine_output` | Abstract turbine mechanical output |
| `EO` | `electrical_output` | Abstract electrical output fraction |
| `LD` | `load_demand` | Fictional requested electrical output |
| `SP` | `support_power` | Support-power availability fraction |

## 5. Qualitative transition model

Equations below are dimensionless software rules, not physics claims. Coefficients will be invented, documented, and chosen for stable narrative generation.

At each tick `t`:

```text
effective_primary_flow = PF × primary_pump_health × valve_flow_factor × SP_factor

transferred_heat = H × effective_primary_flow × TE

primary_thermal_change = H - transferred_heat - synthetic_damping

steam_generation = transferred_heat × SI × secondary_availability

secondary_inventory_change = feed_flow - steam_generation - synthetic_loss

turbine_output = steam_generation × turbine_health

electrical_output = turbine_output × generator_health × bus_availability

condenser_stress = steam_generation - (CF × HR)
```

Implementation must use bounded updates and prevent a single tick from jumping across multiple status bands unless the scenario explicitly defines an abrupt event.

### Directional relationships

The following relationships are the core causal contract:

- Increasing `H` without matching transfer increases `PT`.
- Decreasing effective primary flow reduces transferred heat and tends to increase `PT`.
- Decreasing `TE` creates a transfer mismatch: primary thermal state rises while steam state and output tend to fall.
- Feed flow below steam generation decreases `SI`.
- Reduced `CF` or `HR` increases condenser stress and eventually constrains turbine output.
- Reduced `SP` can make dependent components unavailable, depending on the fictional plant variant.
- A load-demand change alone is a benign transient unless other evidence indicates a fault.
- Sensor faults change observations without directly changing latent process state.

No other real-world inference is authorized from these rules.

## 6. Subsystem contracts

### 6.1 `HEAT_SOURCE`

State:

- `target_heat`
- `heat_source_level`
- `heat_response_rate`
- `controller_state`

Rules:

- Moves gradually toward `target_heat` when support conditions are available.
- `REDUCE_SIMULATED_LOAD` lowers the target within the fictional world.
- It has no fuel, reactivity, control-rod, or protection-system representation.

### 6.2 `PRIMARY_LOOP`

State:

- `primary_flow`
- `primary_thermal_state`
- `primary_inventory`
- two fictional pump-train states;
- one fictional flow-valve state.

Rules:

- Flow is produced by the selected available pump train.
- Degraded pump health causes gradual flow decline.
- A synthetic trip causes abrupt loss of that train's contribution.
- Inventory changes only through the invented inventory-loss fault or recovery rule.

### 6.3 `TRANSFER_UNIT`

State:

- `transfer_efficiency`
- `primary_side_availability`
- `secondary_side_availability`

Rules:

- Transfers synthetic heat between separated loop abstractions.
- Efficiency loss is gradual unless the scenario marks an abrupt state change.
- The unit never models tube geometry, material, pressure boundary, or leakage between sides.

### 6.4 `SECONDARY_LOOP`

State:

- `secondary_flow`
- `secondary_inventory`
- `steam_state`
- feed-pump state;
- feed-valve state.

Rules:

- Feed system restores inventory toward its synthetic target.
- Steam state follows transferred heat and available inventory with delay.
- Flow imbalance is visible through matched trend evidence, not real water-level behavior.

### 6.5 `TURBINE_GENERATOR`

State:

- `turbine_health`
- `generator_health`
- `turbine_output`
- `electrical_output`
- `load_demand`

Rules:

- Electrical output follows available steam and component health.
- Load changes are normal scenario inputs and should not automatically imply a fault.

### 6.6 `CONDENSER_AND_REJECTION`

State:

- `condenser_function`
- `heat_rejection`
- `condenser_stress`

Rules:

- Reduced rejection capability increases the abstract stress proxy.
- Sustained stress constrains output in the fictional state graph.
- No vacuum, cooling-water temperature, weather, or environmental release model is included.

### 6.7 `SUPPORT_POWER`

State:

- `bus_alpha`
- `bus_beta`
- `support_power`
- dependency map for the current fictional plant variant.

Rules:

- A bus interruption changes only dependencies assigned in the invented variant.
- The dependency map must not resemble or be sourced from a real facility.
- The current Aster-A card uses an invented one-to-one mapping: primary train Cirrus
  maps to support bus Rill, and primary train Kestrel maps to support bus Quill.
- A valid Aster-A standby start uses a one-tick fictional delay. Both this delay and the
  mapping are project-authored software constants with no claimed real-plant analogue.

### 6.8 `INSTRUMENTATION`

Each observed variable has:

- latent true value;
- primary observed channel;
- redundant observed channel;
- quality flag;
- bias term;
- bounded noise term;
- availability flag.

Observation equation:

```text
observed = clip(true_value + channel_bias + seeded_noise, 0, 1)
```

Sensor faults alter bias, noise, or availability. They never alter the latent true value.

## 7. Fault and benign-driver contracts

| Fault | Injection | Required latent effect | Required observation signature | Exclusion |
|---|---|---|---|---|
| `SENSOR_DRIFT` | Increasing channel bias | None | One channel separates gradually while related state remains stable | No copied instrument behavior |
| `SENSOR_STUCK` | Freeze one channel | None | Frozen channel fails to follow a genuine state change | No real failure threshold |
| `SENSOR_NOISE` | Increase bounded variance | None | Rapid inconsistent readings without matching related trends | No real signal frequencies |
| `PUMP_DEGRADATION` | Decrease fictional health gradually | Flow declines gradually | Related thermal/transfer trends follow after delay | No bearing or mechanical diagnosis |
| `PUMP_TRIP` | Set selected train unavailable | Abrupt flow contribution loss | Component state change precedes dependent trends | No real trip logic |
| `VALVE_LAG` | Delay commanded state change | Flow response occurs late | Command/position mismatch resolves after bounded delay | No real actuator timing |
| `VALVE_STUCK` | Hold position despite command | Flow remains inconsistent with command | Persistent command/position mismatch | No real valve or system identification |
| `TRANSFER_EFFICIENCY_LOSS` | Reduce `TE` | Transfer decreases; upstream/downstream trends diverge | Correlated mismatch across loop proxies | No geometry or fouling mechanism |
| `FLOW_IMBALANCE` | Offset feed-flow target | `SI` trends away from target | Feed and steam trends disagree persistently | No real level-control logic |
| `SUPPORT_POWER_INTERRUPTION` | Reduce one fictional bus | Assigned dependencies change state | Bus event precedes only mapped component changes | No real electrical topology |
| `LOAD_TRANSIENT` *(benign driver, not a fault)* | Change `LD` | Coordinated normal transition | Related variables move consistently | Ground truth is `NO_FAULT`; never include in `fault_family_ids` |
| `ABSTRACT_INVENTORY_LOSS` | Apply synthetic loss to `PI` | Inventory falls; related fictional trends follow | Multiple independent observations agree | No break size/location or emergency progression |

### 7.1 Developmental G07 pump-trip contract

The implemented Aster-A G07 fixture is a matched counterfactual pair, not a generalized
plant policy. Both members use steady operation, one indefinite low-severity
`PUMP_TRIP` on either active primary train, the other primary train as standby, its
invented mapped support bus, and a one-tick standby-start delay. A strict
`StandbyContext` records only the context identifier, distinct active/standby train
identifiers, standby state, support-bus identifier and state, and positive start delay.
Only exact `AVAILABLE` or `UNAVAILABLE` standby contexts are supported. The current
matched pair holds the mapped support bus `AVAILABLE` and changes only the standby
state; other bus-state combinations remain outside this fixture and fail closed.

The fixed fictional schedule is:

- Tick 2: the active train changes from `AVAILABLE` to `UNAVAILABLE`, and operating
  mode changes from `STABLE` to `DISTURBED`; active-train numeric health remains `1.0`
  because this case models an abrupt state change rather than gradual degradation.
- Tick 3: primary flow loses the selected train's contribution. This seed-derived
  abrupt loss is the sole exception to the normal per-tick step bound; transfer
  efficiency remains unchanged.
- Tick 4: primary thermal state rises by a bounded fictional increment.
- Tick 5: steam, turbine output, and electrical output decline after the dependent
  delay. Both contexts diagnose `PUMP_TRIP` using the same process evidence plus the
  visible standby fact.
- Available branch: tick 5 selects `SELECT_SYNTHETIC_STANDBY_TRAIN`; tick 6 applies the
  action and moves the standby from `AVAILABLE` to `STARTING`; tick 7 moves it to
  `RECOVERING`, begins bounded partial flow recovery, and changes the mode to `RECOVERY`
  only after recovery is visible.
- Unavailable branch: tick 5 selects `REDUCE_SIMULATED_LOAD`; tick 6 applies it, lowers
  fictional heat/load targets, and selects `ENTER_SIMULATED_STABLE_STATE`; tick 7
  applies that action and changes the mode to `STABILIZED`. The standby remains
  unavailable and primary flow does not recover.

The tripped active train stays unavailable in both branches. G07 remains a
developmental fixture outside training and is not frozen until the required human
golden-scenario review occurs.

### 7.2 Completed developmental G08–G15 fixtures

Phase 2 also implements the remaining planned fixture families on the same fictional,
normalized state machine. G08/G09 use the same command/position surface and differ only
through bounded resolution (`VALVE_LAG`) versus persistent mismatch (`VALVE_STUCK`).
G10 preserves primary flow while an invented transfer proxy falls; G11 makes an
invented feed/steam mismatch drive secondary inventory; G12 uses only the current
variant's project-authored dependency map; and G13 models abstract inventory decline
with agreeing channels and delayed fictional effects. Their values, timing, aliases,
and maps are software design constants, not operating guidance.

G14 is intentionally the sole developmental compound: pump degradation plus one
independent primary-thermal sensor drift. The semantic pair may be described in that
human order, but D-039 requires canonical serialized fault order
`(SENSOR_DRIFT, PUMP_DEGRADATION)`. G15 is a sparse primary-flow audit fixture: its
single isolated reading must resolve to `UNRESOLVED` and `INSUFFICIENT_EVIDENCE`.
Neither the truth-filtered audit payload nor its full trace is a model prompt; Phase 3
must construct decision-tick/channel projections and split groups before rendering.

All G01–G15 fixtures are developmental, outside training, and unfrozen pending the
documented human golden review. The current scanner is a narrow redacting check only;
the full denylist and human sample review remain Phase 3 gates.

## 8. Action-label semantics

Actions are task labels and optional fictional state transitions—not human instructions.

| Label | State-machine effect |
|---|---|
| `VERIFY_REDUNDANT_CHANNEL` | Reveals or confirms the redundant synthetic observation |
| `COMPARE_RELATED_TRENDS` | Adds canonical evidence slots; no process effect |
| `FLAG_SENSOR_SUSPECT` | Changes channel quality to `SUSPECT`; no process effect |
| `REQUEST_COMPONENT_INSPECTION` | Adds a pending fictional maintenance flag; no immediate process effect |
| `SELECT_SYNTHETIC_STANDBY_TRAIN` | Starts a valid invented standby train after a delay if available |
| `REDUCE_SIMULATED_LOAD` | Lowers `LD` and `target_heat` within the abstract model |
| `ENTER_SIMULATED_STABLE_STATE` | Moves targets toward a reduced, bounded synthetic state |
| `CONTINUE_MONITORING` | Advances time without changing targets |
| `INSUFFICIENT_EVIDENCE` | Makes no state change and records abstention |

The renderer must never convert these labels into numbered real-world procedures.

## 9. Plant variants

Create at least three invented variants to prevent the model from memorizing one naming/topology surface:

- `ASTER-A`: two interchangeable primary trains and two support buses.
- `ASTER-B`: different fictional component names and delayed standby availability.
- `ASTER-C`: different dependency mapping and sensor alias families.

Variants share the semantic schema but differ in:

- invented asset names;
- channel aliases;
- which fictional bus supplies which synthetic component;
- response delays;
- baseline noise bands;
- renderer style.

No variant may be tuned to resemble a real vendor or facility.

## 10. Event-generation order

Each tick follows one fixed order:

1. Apply scheduled target changes.
2. Inject or advance fault state.
3. Apply previously selected fictional actions.
4. Update component states.
5. Update bounded latent process variables.
6. Produce sensor observations and quality flags.
7. Derive canonical evidence slots.
8. Derive event labels and correct task targets.
9. Render zero or more narrative lines.
10. Validate invariants and record provenance.

This ordering must be versioned because changing it can change the dataset distribution.

## 11. Required invariants

- Every value remains within its declared domain.
- Sensor-only faults never change latent process variables.
- Every process fault has at least one latent effect before its diagnosis becomes labelable.
- Every non-abstaining diagnosis has required evidence in the model-visible context.
- Removing decisive evidence changes the target to `INSUFFICIENT_EVIDENCE` when no alternative proof remains.
- Benign load changes remain `NO_FAULT` unless a separate fault is injected.
- A command/position mismatch caused by lag eventually resolves; one caused by stuck state persists.
- Standby selection cannot recover flow if the invented standby component or its assigned bus is unavailable.
- Renderer wording cannot alter structured ground truth.
- No state path can produce a real facility name, real unit, procedure identifier, or engineering value.

## 12. Behavioral and metamorphic expectations

Inspired by CheckList-style behavioral testing, define three categories:

### Minimum-functionality tests

- Distinguish sensor drift from genuine process change.
- Distinguish valve lag from valve stuck using persistence.
- Distinguish benign load change from fault.
- Abstain when decisive evidence is absent.
- Identify a held-out combination when individual evidence remains present.

### Invariance tests

The correct structured target should not change when:

- component aliases are replaced consistently;
- irrelevant benign events are inserted;
- equivalent narrative template families are used;
- non-causal notes are reordered while timestamps remain clear;
- normalized values change within the same status band.

### Directional-expectation tests

- Increasing channel bias should increase confidence in sensor drift only when redundant evidence is available.
- Extending a command/position mismatch beyond the lag limit should shift the target from lag toward stuck.
- Removing the only supporting evidence should increase abstention.
- Adding an independent agreeing channel should reduce abstention.
- Increasing compound-fault severity should not make structured output less parseable.

## 13. Source basis

- NRC's public PWR overview provides the high-level separated-loop and electricity-generation concept: https://www.nrc.gov/reactors/power/pwrs
- NRC's Reactor Concepts Manual was reviewed only for broad component relationships; no values, layouts, procedures, or passages are adopted: https://ww2.nrc.gov/sites/default/files/doc_library/cdn/legacy/reading-rm/basic-ref/students/for-educators/04.pdf
- IAEA TECDOC-1887 describes classifications and objective-driven use of educational simulators: https://www-pub.iaea.org/MTCD/Publications/PDF/TE-1887_web.pdf
- IAEA TCS-70 explicitly frames basic-principle material as “big picture” education: https://www-pub.iaea.org/MTCD/Publications/PDF/TCS-70web.pdf
- CheckList motivates minimum-functionality, invariance, and directional behavioral tests beyond aggregate accuracy: https://aclanthology.org/2020.acl-main.442/

These are research references, not corpus sources.
