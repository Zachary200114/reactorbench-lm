# Golden scenario suite

Status: human-authored pre-implementation acceptance contract
Scope: Aster Station, a wholly fictional normalized state machine

## Purpose

These cases define behavior that the generator, labels, baselines, and trained model must eventually be tested against. They are not training templates. Their exact combinations, event order, aliases, and surface wording must be withheld from training and development data.

All values are dimensionless and normalized to `[0, 1]`. Approximate bands are `LOW < 0.35`, `NOMINAL 0.35–0.75`, and `HIGH > 0.75`; final implementation thresholds remain invented and versioned. Actions are fictional classification labels, not instructions.

Each scenario must be rendered through multiple unseen wording and component-alias families. Passing requires correct structured labels and evidence, not merely plausible prose.

## Review rules

- Ground truth is determined from latent state and injected faults, never from generated text.
- The expected sequence specifies precedence, not real elapsed time.
- Exact action enums and their stated order are authoritative where listed. Concrete integer `decision_tick` values must be supplied by generator fixtures during human golden freeze; they must not be inferred from prose arrows. Each frozen fixture then serializes its action sequence as a list of `{decision_tick, action}` objects.
- Evidence must identify observations that distinguish the target from credible alternatives.
- A model must abstain when observations are deliberately insufficient or contradictory.
- Metamorphic variants change irrelevant presentation details while preserving, reversing, or weakening the expected answer in a declared way.
- These cases remain outside all training shards. Any change requires a version bump and recorded review.

## G01 — Stable baseline

- **Test type:** minimum-functionality / false-positive control
- **Start state:** `STABLE`; `H`, `PF`, `PT`, `TE`, `SF`, `SI`, `ST`, `CF`, `HR`, `TO`, `EO`, and `LD` are mutually consistent in the nominal band; redundant sensors agree.
- **Injection:** none; only seeded bounded observation noise.
- **Expected sequence:** stable observations → small bounded variation → stable observations.
- **Visible evidence:** no persistent trend, no component-state change, no channel disagreement.
- **Diagnostic status:** `NO_FAULT`.
- **Fault labels:** none.
- **Target action:** `CONTINUE_MONITORING`.
- **Abstention:** no.
- **Metamorphic checks:** rename components, reorder simultaneous normal observations, and perturb values within the same bands; classification must not change.
- **Why it matters:** prevents a benchmark optimized only for detecting abnormalities.

## G02 — Benign load transition

- **Test type:** directional expectation / confounder control
- **Start state:** `STABLE` at a midrange synthetic demand.
- **Injection:** raise `LD` by a bounded amount with all equipment available.
- **Expected sequence:** demand change → coordinated controller response → `H`, transfer, `ST`, `TO`, and `EO` move in compatible directions → new stable state.
- **Visible evidence:** related variables respond coherently; no persistent command mismatch or sensor conflict.
- **Diagnostic status:** `NO_FAULT`.
- **Fault labels:** none; `LOAD_TRANSIENT` is the benign scenario driver, not a fault family.
- **Target action:** `CONTINUE_MONITORING`.
- **Abstention:** no.
- **Metamorphic checks:** reverse the demand direction and corresponding trends; the case remains `NO_FAULT`. Add an irrelevant healthy-component status message; answer remains invariant.
- **Why it matters:** distinguishes a planned transient from a failure.

## G03 — Single-channel sensor drift

- **Test type:** minimum-functionality / observation-layer diagnosis
- **Start state:** stable latent `PF`; two flow channels initially agree.
- **Injection:** gradually increase bias on one `PF` channel only.
- **Expected sequence:** channels agree → one channel separates monotonically → redundant channel and related transfer trends remain stable → channel quality becomes conflicting.
- **Visible evidence:** disagreement is isolated to one channel; latent-correlated observations do not support a real flow change.
- **Target:** `SENSOR_DRIFT`.
- **Target action:** `VERIFY_REDUNDANT_CHANNEL`, then `FLAG_SENSOR_SUSPECT`.
- **Abstention:** no after sufficient trend history; yes during the earliest ambiguous tick.
- **Metamorphic checks:** drift upward or downward and swap which channel drifts; diagnosis remains the same.
- **Why it matters:** tests separation of observed data from latent process state.

## G04 — Stuck sensor during a real load change

- **Test type:** compositional / temporal evidence
- **Start state:** stable with agreeing `EO` channels.
- **Injection:** initiate a benign load decrease while freezing one `EO` channel at its pre-change value.
- **Expected sequence:** demand decreases → related latent and redundant observations fall → frozen channel remains constant → conflict is declared.
- **Visible evidence:** the channel is implausibly flat while independent related measurements change coherently.
- **Target:** `SENSOR_STUCK` plus benign `LOAD_TRANSIENT` context.
- **Target action:** `VERIFY_REDUNDANT_CHANNEL`, then `FLAG_SENSOR_SUSPECT`.
- **Abstention:** no after the independent trend appears.
- **Metamorphic checks:** freeze a different channel or reverse the load change; the stuck-channel label follows the frozen observation.
- **Why it matters:** requires reasoning over change, not a single abnormal value.

## G05 — Noisy sensor

- **Test type:** robustness / temporal-pattern discrimination
- **Start state:** stable latent `PT` with agreeing channels.
- **Injection:** increase bounded, zero-centered variance on one channel.
- **Expected sequence:** agreement → rapid alternating deviations → no matching latent-correlated trend → continued unstable channel readings.
- **Visible evidence:** high-frequency inconsistency localized to one observation channel.
- **Target:** `SENSOR_NOISE`.
- **Target action:** `COMPARE_RELATED_TRENDS`, then `FLAG_SENSOR_SUSPECT`.
- **Abstention:** yes until enough samples distinguish noise from a one-tick disturbance.
- **Metamorphic checks:** vary noise realization and narrative order while retaining the temporal signature; diagnosis remains stable.
- **Why it matters:** prevents collapsing all instrument faults into drift.

## G06 — Gradual primary-pump degradation

- **Test type:** minimum-functionality / causal chain
- **Start state:** stable with one selected pump train carrying fictional flow.
- **Injection:** gradually decrease that train's synthetic health.
- **Expected sequence:** health decline → `PF` declines → transferred heat falls → `PT` trends upward and downstream production trends downward after delay.
- **Visible evidence:** component degradation precedes several mutually consistent process trends; redundant flow channels agree.
- **Target:** `PUMP_DEGRADATION`.
- **Target action:** `REQUEST_COMPONENT_INSPECTION`, with `REDUCE_SIMULATED_LOAD` if the trend persists.
- **Abstention:** no once correlated trends are present.
- **Metamorphic checks:** change train alias and exact slope; classification remains invariant. Remove component-health evidence and shorten the trace; expected output becomes `INSUFFICIENT_EVIDENCE`.
- **Why it matters:** tests causal ordering and graded deterioration.

## G07 — Pump trip with fictional standby available

- **Test type:** abrupt event / action selection
- **Start state:** stable; active train available and a project-authored standby train available.
- **Injection:** set the active train to `UNAVAILABLE` at one tick.
- **Expected sequence:** component-state event → abrupt `PF` contribution loss → dependent thermal/transfer trends → standby selection label → recovery of flow → `RECOVERY`.
- **Visible evidence:** the discrete trip event comes before process changes; the variant dependency map declares standby availability.
- **Target:** `PUMP_TRIP`.
- **Target action:** `SELECT_SYNTHETIC_STANDBY_TRAIN`.
- **Abstention:** no.
- **Metamorphic checks:** in a paired variant where standby is unavailable, the target action changes to `REDUCE_SIMULATED_LOAD` and `ENTER_SIMULATED_STABLE_STATE`; fault label stays fixed.
- **Why it matters:** separates fault identification from context-dependent action classification.

## G08 — Valve lag that resolves

- **Test type:** temporal boundary / near-neighbor contrast
- **Start state:** stable flow; a fictional valve receives a bounded position command.
- **Injection:** delay observed and effective position change for a finite permitted interval, then allow completion.
- **Expected sequence:** command changes → temporary command/position mismatch → delayed flow response → position reaches target → mismatch clears.
- **Visible evidence:** mismatch is bounded and self-resolving.
- **Target:** `VALVE_LAG`.
- **Target action:** `CONTINUE_MONITORING` or `COMPARE_RELATED_TRENDS` according to the scenario's declared duration band.
- **Abstention:** yes before the lag-duration evidence is sufficient.
- **Metamorphic checks:** modestly change the invented delay within the lag band; label remains fixed.
- **Why it matters:** establishes a temporal decision boundary against a stuck valve.

## G09 — Valve stuck

- **Test type:** near-neighbor contrast with G08
- **Start state:** identical to G08.
- **Injection:** hold the valve position despite the command for the full observation window.
- **Expected sequence:** command changes → persistent command/position mismatch → expected flow response never occurs → dependent discrepancy persists.
- **Visible evidence:** non-resolution beyond the fictional lag window.
- **Target:** `VALVE_STUCK`.
- **Target action:** `REQUEST_COMPONENT_INSPECTION`; optionally `REDUCE_SIMULATED_LOAD` when the declared scenario policy requires it.
- **Abstention:** yes early; no after persistence is established.
- **Metamorphic checks:** the minimum counterfactual is G08's eventual position response; adding only that response must flip the label from stuck to lag.
- **Why it matters:** tests whether the model uses decisive later evidence.

## G10 — Transfer-efficiency loss

- **Test type:** multivariable causal diagnosis
- **Start state:** stable transfer with normal flow and inventory.
- **Injection:** gradually reduce `TE` while leaving pump health and sensor quality normal.
- **Expected sequence:** `TE` falls → transferred heat decreases → `PT` rises → `ST`, `TO`, and `EO` fall after delay despite preserved `PF`.
- **Visible evidence:** upstream/downstream divergence with normal primary flow and agreeing channels.
- **Target:** `TRANSFER_EFFICIENCY_LOSS`.
- **Target action:** `REDUCE_SIMULATED_LOAD`, then `ENTER_SIMULATED_STABLE_STATE` if needed by the fictional policy.
- **Abstention:** no after the cross-loop divergence is visible.
- **Metamorphic checks:** replace `TE` observation with missing data; if only divergence remains and alternatives cannot be excluded, target becomes `INSUFFICIENT_EVIDENCE`.
- **Why it matters:** tests diagnosis from relationships rather than keyword cues.

## G11 — Secondary flow imbalance

- **Test type:** trend consistency / fault isolation
- **Start state:** stable feed and steam balance.
- **Injection:** offset the fictional feed-flow target below synthetic steam generation.
- **Expected sequence:** feed/steam mismatch → `SI` trends downward → related secondary observations agree → downstream production becomes constrained if sustained.
- **Visible evidence:** persistent mismatch confined initially to the secondary-side abstraction; primary pump and transfer evidence do not lead the event.
- **Target:** `FLOW_IMBALANCE`.
- **Target action:** `COMPARE_RELATED_TRENDS`, followed by `ENTER_SIMULATED_STABLE_STATE` for sustained cases.
- **Abstention:** no once persistence is established.
- **Metamorphic checks:** change component aliases and initial inventory within the nominal band; causal label remains fixed.
- **Why it matters:** verifies subsystem localization.

## G12 — Fictional support-power interruption

- **Test type:** plant-variant dependency reasoning
- **Start state:** stable `ASTER-B`; its invented dependency map assigns selected components to `bus_beta`.
- **Injection:** reduce `bus_beta` availability.
- **Expected sequence:** bus event → only mapped components change availability → their dependent flows/observations change → unmapped components remain available.
- **Visible evidence:** the bus event temporally precedes a dependency-map-consistent cluster.
- **Target:** `SUPPORT_POWER_INTERRUPTION`.
- **Target action:** `ENTER_SIMULATED_STABLE_STATE`.
- **Abstention:** no when the variant map is included; yes if the prompt withholds the map and multiple causes remain possible.
- **Metamorphic checks:** run the same bus event on `ASTER-A` with a different invented map; affected components and evidence must change, but the fault family remains fixed.
- **Why it matters:** tests conditional reasoning without encoding real electrical topology.

## G13 — Abstract inventory loss

- **Test type:** persistent latent-process fault
- **Start state:** stable `PI` with two agreeing observation channels.
- **Injection:** apply a small sustained synthetic loss to `PI`.
- **Expected sequence:** `PI` declines across ticks → independent channels agree → related fictional flow/thermal effects appear after delay → decline persists absent recovery action.
- **Visible evidence:** multiple observations support a true latent change, unlike a single-channel sensor fault.
- **Target:** `ABSTRACT_INVENTORY_LOSS`.
- **Target action:** `REDUCE_SIMULATED_LOAD`, then `ENTER_SIMULATED_STABLE_STATE`.
- **Abstention:** no after correlated evidence; yes if only one channel is shown.
- **Metamorphic checks:** replace one agreeing channel with a biased outlier while retaining the redundant and related trends; process-fault label must remain primary and the sensor issue may be secondary.
- **Why it matters:** provides a deliberate contrast with observation-layer faults while remaining non-engineering and abstract.

## G14 — Held-out compound: pump degradation plus sensor drift

- **Test type:** strict compositional generalization
- **Start state:** stable; no training scenario may contain this exact fault pair on these component roles.
- **Injection:** gradually degrade a pump while independently biasing one `PT` observation channel.
- **Expected sequence:** pump health and agreeing flow observations establish real degradation → downstream latent-correlated trends appear → one thermal channel separates beyond the true trend → two fault labels become supported.
- **Visible evidence:** redundant and cross-variable evidence separates a process fault from a simultaneous observation fault.
- **Target:** ordered multi-label result `PUMP_DEGRADATION` + `SENSOR_DRIFT`.
- **Target action:** `VERIFY_REDUNDANT_CHANNEL`, `FLAG_SENSOR_SUSPECT`, and `REQUEST_COMPONENT_INSPECTION`, with `REDUCE_SIMULATED_LOAD` when the declared persistent-degradation policy requires it.
- **Abstention:** no after both evidence chains mature; partial output is expected earlier.
- **Metamorphic checks:** swap the drifting channel and component aliases; preserve the pair. Remove the channel disagreement; the secondary sensor label must disappear.
- **Why it matters:** this is a central proof that the model learned reusable factors rather than memorized scenario templates.

## G15 — Insufficient-evidence counterfactual pair

- **Test type:** abstention calibration / information sufficiency
- **Start state:** stable but the prompt exposes only a short observation window.
- **Injection:** one `PF` channel reports a single low value; latent cause is deliberately withheld from the model-facing view.
- **Expected sequence:** isolated low reading → no redundant channel, component state, or related trend is supplied → trace ends.
- **Visible evidence:** compatible with sensor noise, drift onset, genuine flow reduction, or a transient observation artifact.
- **Diagnostic status:** `UNRESOLVED`.
- **Fault labels:** none.
- **Abstention reason:** `INSUFFICIENT_EVIDENCE`.
- **Target action:** `INSUFFICIENT_EVIDENCE`.
- **Abstention:** required.
- **Metamorphic checks:**
  - **G15-A:** add an agreeing redundant low channel and related delayed trends; abstention should resolve toward a process fault only if the added evidence identifies one.
  - **G15-B:** add stable related trends plus a normal redundant channel; abstention should resolve toward an observation-layer fault.
  - Paraphrase or reorder the original sparse facts; abstention must remain unchanged.
- **Why it matters:** makes calibrated non-answering a first-class measured behavior.

## Acceptance matrix

| Capability | Required cases | Primary check |
|---|---|---|
| Normal-state specificity | G01–G02 | No-fault precision |
| Sensor/process separation | G03–G05, G13 | Fault-family accuracy and evidence F1 |
| Temporal reasoning | G04, G06, G08–G09 | Event-order and final-label accuracy |
| Causal/subsystem reasoning | G06–G13 | Evidence sufficiency and fault accuracy |
| Context-sensitive action labels | G07, G12 | Action accuracy conditioned on variant facts |
| Composition | G14 | Exact multi-label match and per-label F1 |
| Abstention | G03, G05, G08–G10, G12–G13, G15 | Coverage-risk curve and required-abstention accuracy |
| Wording invariance | all | Prediction consistency across held-out renderers |

## Human review record required before implementation

The repository must later record, for each scenario version:

- reviewer name or role;
- review date;
- schema and generator version;
- confirmation that no real setpoint, procedure, plant topology, or service-derived information is present;
- expected structured answer checksum;
- approval or requested revision.

The suite is not considered frozen until all scenarios receive this review.
