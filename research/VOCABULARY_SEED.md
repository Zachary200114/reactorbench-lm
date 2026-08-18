# Initial vocabulary and content boundary

Status: seed for later human review; not a finished tokenizer vocabulary.

## Purpose

Define a safe, high-level vocabulary for authoring the fictional corpus. These terms may name broad concepts, but their relationships, values, rules, events, and narratives must be invented by the project.

## Candidate high-level system nouns

- plant
- unit
- system
- subsystem
- component
- train
- heat source
- reactor
- core state
- primary loop
- secondary loop
- heat-transfer loop
- heat exchanger
- steam generator
- condenser
- turbine
- generator
- pump
- valve
- line
- tank
- coolant
- feed flow
- steam flow
- support power
- electrical bus
- sensor
- channel
- redundant channel
- controller
- status flag

## Candidate observation language

- stable
- rising
- falling
- oscillating
- delayed
- unavailable
- degraded
- recovering
- normal
- watch
- abnormal
- unknown
- inconsistent
- correlated
- uncorrelated
- expected range
- synthetic limit
- synthetic unit
- trend
- reading
- state
- transition
- timestamp
- event
- log
- note
- summary

## Candidate fault-family labels

- `SENSOR_DRIFT`
- `SENSOR_STUCK`
- `SENSOR_NOISE`
- `PUMP_DEGRADATION`
- `PUMP_TRIP`
- `VALVE_LAG`
- `VALVE_STUCK`
- `TRANSFER_EFFICIENCY_LOSS`
- `FLOW_IMBALANCE`
- `SUPPORT_POWER_INTERRUPTION`
- `ABSTRACT_INVENTORY_LOSS`

## Candidate benign scenario-driver labels

- `LOAD_TRANSIENT`

## Candidate diagnostic-status and abstention labels

- `DIAGNOSED`
- `NO_FAULT`
- `UNRESOLVED`
- `INSUFFICIENT_EVIDENCE`

`NO_FAULT` is a diagnostic status with an empty fault-label set; it is not a fault
family.

## Candidate evidence labels

- `CHANNEL_DISAGREEMENT`
- `RELATED_CHANNEL_AGREEMENT`
- `TREND_CHANGE`
- `STATE_CHANGE`
- `FLOW_MISMATCH`
- `TRANSFER_MISMATCH`
- `POWER_AVAILABLE`
- `POWER_UNAVAILABLE`
- `COMPONENT_STATE_UNCHANGED`
- `COMPONENT_STATE_CHANGED`
- `TEMPORAL_ORDER_SUPPORTS`
- `TEMPORAL_ORDER_CONFLICTS`
- `EVIDENCE_MISSING`

## Candidate fictional action labels

- `VERIFY_REDUNDANT_CHANNEL`
- `COMPARE_RELATED_TRENDS`
- `FLAG_SENSOR_SUSPECT`
- `REQUEST_COMPONENT_INSPECTION`
- `SELECT_SYNTHETIC_STANDBY_TRAIN`
- `REDUCE_SIMULATED_LOAD`
- `ENTER_SIMULATED_STABLE_STATE`
- `CONTINUE_MONITORING`
- `INSUFFICIENT_EVIDENCE`

These are classification labels within the invented world, not steps for a person to follow.

## Prohibited content

- Real facility, ship, unit, operator, licensee, vendor, or individual names.
- Real equipment model numbers, component identifiers, docket numbers, or event numbers.
- Real values, units, capacities, thresholds, setpoints, trip points, or time constants.
- Real procedures, checklists, alarm-response steps, emergency classifications, corrective actions, or licensing requirements.
- Real incident/event narratives or paraphrases of them.
- Security architecture, safeguards, access-control, vulnerability, adversarial, or attack content.
- Navy nuclear propulsion terminology, training content, qualification knowledge, personal recollections, or service-derived system details.
- Classified, controlled, export-controlled, proprietary, access-restricted, or personally identifying information.
- Agency logos or wording that implies NRC, DOE, IAEA, ORNL, INL, Navy, or employer endorsement.

## Numeric rule

Use normalized values (`0.00`–`1.00`), standardized scores, categorical bands, or values explicitly marked `SU` for synthetic units. Never select numbers by adapting real operating values.

## Source note

General terms may be checked against the NRC public glossary and educational PWR overview, but definitions and prose must not be copied into the corpus:

- https://www.nrc.gov/reading-rm/basic-ref/glossary
- https://www.nrc.gov/reactors/power/pwrs
