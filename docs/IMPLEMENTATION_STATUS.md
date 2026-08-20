# ReactorBench-LM implementation status

Last updated: 2026-08-20 05:18 EDT
Current phase: Phase 2 in progress — developmental G06 `PUMP_DEGRADATION` milestone complete
Checkpoint reason: the first constrained process-fault case passed focused and full gates, fresh independent review, and local code integration; documentation is being preserved as a separate handoff checkpoint
Intended project path: `/Users/zachary/Documents/Personal-Projects/AI-transformer`

## Current objective

Audit the authoritative G07 `PUMP_TRIP` contract, Aster-A standby/dependency representation, abrupt component-to-process causal ordering, and context-dependent action requirement before changing code. Determine whether the smallest fictional case can be represented without prematurely adding a second plant variant or changing schemas. Do not add another fault, generalize composition, render a dataset, train a tokenizer/model, build the UI, or expose a network service until the relevant gates pass.

## Completed work

- Audited all 16 authoritative research files, the corrected project path, initial repository state, applicable instructions, and installed tooling.
- Reconciled the research terminology and phase map without weakening requirements: `LOAD_TRANSIENT` is a benign driver; diagnosis uses `DIAGNOSED`, `NO_FAULT`, or `UNRESOLVED`; action sequences carry concrete decision ticks; `ReactorBench-LM` is the final name.
- Recorded the local-only boundary: prepare the whole project, but do not push to GitHub or deploy/publish through Vercel.
- Added a Python 3.12 PEP 621/Hatchling project, a locked `uv` environment, safe Make targets, strict TOML configuration, project-relative output containment, canonical configuration hashing, and non-overwriting run reservation.
- Added strict Pydantic v2 contracts with immutable validated instances for latent state, observations, canonical events, scenarios, aggregate structured trajectories, provenance, task targets, and the supporting closed enums.
- Added an exhaustive per-`EventType` payload matrix, six discriminated task-specific targets, trajectory-wide provenance/ordering/evidence invariants, uint32 seeds, finite normalized values, one action per decision tick, and fault-duration bounds.
- Added seven deterministic JSON Schema snapshots. The loader rejects partial, empty, unexpected, non-canonical, duplicate-key, non-finite, checksum-mismatched, traversal-bearing, and symlink-escaping snapshots.
- Added package-resource access for the reviewed default configuration and schema snapshots. The wheel and sdist contents are checked, and a wheel-only no-network install/import check passes.
- Added the root disclaimer, architecture, Phase 0 audit, threat model, security control map, residual-risk language, and private vulnerability-reporting policy.
- Added unit, contract, and Hypothesis property tests. No dataset, tokenizer, model, checkpoint, measured model result, inference service, or web interface exists yet.
- Reproduced the complete Phase 1 gate from the intended project path and preserved the verified foundation in local commit `0cfc98c`; no remote was added and nothing was pushed or published.
- Added generator version `0.1.0` with one immutable Aster-A variant card, two fictional channels for every normalized variable, deterministic stable traces, and one single-channel `SENSOR_DRIFT` behavior using only local `random.Random` streams.
- Added strict scenario builders and fail-closed rejection for unsupported variants, drivers, faults, mappings, action sequences, durations, loose channel inputs, and out-of-range scalar inputs.
- Added early required abstention, later evidence-backed diagnosis, next-tick action application, causal event ordering, stable/drift latent equality, model-visible truth isolation, and conversion into the existing validated `StructuredTrajectory` contract.
- Added a recursive Phase 2 prohibited-content guard with bounded redacted findings and tests for URL, contact, identifier, operating-value, real-plant, military/agency, and security-related pattern classes. It is explicitly non-exhaustive and does not replace later human sample review.
- Received an independent `ship` review with no findings. The reviewer host exposed workspace-write rather than hard read-only isolation, so exact before/after hashes and Git state were compared and remained unchanged.
- Added the canonical benign Aster-A `LOAD_TRANSIENT`, fully derived from driver, seed, duration, and generator version: demand/heat/flow begin at tick 2, steam at tick 3, output at tick 4, coordinated evidence at tick 6, and return to `STABLE` at tick 7.
- Kept `transfer_efficiency` invariant as a capability proxy, preserved available components and agreeing `GOOD`/`NORMAL` redundant channels, supported rising and falling seed-derived cases, and resolved every benign case to empty fault labels with `NO_FAULT` and `CONTINUE_MONITORING`.
- The first load draft failed independent review because downstream response began too early. It was corrected to explicit causal stage lags, retested, and independently re-reviewed against exact SHA-256 file hashes before integration; the final verdict was `ship` with no findings.
- Added one deliberately narrow Aster-A composition: `LOAD_TRANSIENT` plus exactly one indefinite low-severity `SENSOR_STUCK` injection on either electrical-output channel, beginning at tick 2 and supported for both seed-derived load directions.
- Froze the selected channel at its tick-1 observed value from tick 2 onward while proving the latent trajectory exactly equals the same-seed benign load trace and every nonselected observation is unchanged.
- Added a causal evidence/status/action sequence: selected status becomes `WATCH` at tick 4 and `CONFLICTING` at tick 5; redundant electrical output changes and disagreement become evidence at tick 5; verification is decided at tick 5 and applied at tick 6; flagging is decided at tick 6 and applied at tick 7; selected quality becomes `SUSPECT` at tick 7.
- Required the mature diagnosis to use only `SENSOR_STUCK` with `CHANNEL_FROZEN`, `CORRELATED_STATE_CHANGE`, and `CHANNEL_DISAGREEMENT`; benign `LOAD_TRANSIENT` remains driver context and never enters `fault_family_ids`.
- Added fail-closed validation for unsupported variants, versions, drivers, fault/container shapes, channels, channel/component mappings, severities, onsets, finite durations, action sequences, extra faults, short traces, and noncanonical scenario identifiers.
- Verified that visible structured payloads omit fault, driver, scenario, severity, onset, and provenance truth. Golden G04 remains outside training and is not frozen because its required human review has not occurred.
- The first stuck-load review identified settlement lineage tied to disagreement rather than coordinated load evidence, non-tuple injection acceptance through unchecked model copying, and an incomplete trajectory contract test. All three were corrected, focused tests were strengthened, and the final independent verdict was `ship`.
- Added one narrow observation-only Aster-A `SENSOR_NOISE` case over `STEADY_OPERATION`: exactly one indefinite low-severity injection on either primary-thermal-state channel, with stable/noise equality through onset tick 2.
- Added deterministic, prefix-preserving alternating offsets from tick 3 onward. Adjacent offsets share a seeded magnitude in the normalized `[0.018, 0.024]` interval and use opposite signs; each later pair may draw a different deterministic magnitude and seed parity controls the initial phase.
- Proved that the noise trace has exactly the same latent states as its same-seed stable trace, that every nonselected observation is identical, and that the selected channel alone differs after onset while all values remain bounded.
- Added two explicit `UNRESOLVED` decisions with `INSUFFICIENT_EVIDENCE` at ticks 3 and 4. Rapid inconsistent readings and disagreement support a tick-5 `SENSOR_NOISE` diagnosis with `COMPARE_RELATED_TRENDS`; related-state stability supports `FLAG_SENSOR_SUSPECT` at tick 6. Applied actions occur at ticks 4, 5, 6, and 7, and selected quality becomes `SUSPECT` at tick 7.
- Prevented target leakage by using `NORMAL`, `WATCH`, `CONFLICTING`, and generic evidence in the visible trace; the lexical `NOISY` quality label appears neither before diagnosis nor anywhere in the visible payload. Hidden structured targets and provenance retain `SENSOR_NOISE` truth.
- Hardened scenario dispatch against unchecked `model_copy` lookalikes: driver, variant, fault, severity, and action values must be canonical enum instances; ticks must be exact integers rather than floats or booleans; action/fault containers and members must use their canonical tuple/model forms.
- Independent review initially returned `fix-first` because a string-valued driver and float-valued tick could survive unchecked model copying and reach supported dispatch. The validation was generalized across the supported fault cases, tests were expanded, and the final independent verdict was `ship`.
- Kept structured `SENSOR_NOISE` fault truth distinct from the later dataset `noise_test` corruption split. Golden G05 remains outside training and is not frozen because its required human review has not occurred.
- Added the first bounded process-fault case: one indefinite low-severity Aster-A `PUMP_DEGRADATION` injection on either fictional primary train during `STEADY_OPERATION`, with a canonical seed-derived default alias and explicit alias support.
- Added a deterministic normalized selected-component health decline beginning at tick 2, followed by primary-flow decline at tick 3, primary-thermal rise at tick 4, steam decline at tick 5, and turbine/electrical-output decline at tick 6. Transfer efficiency and unrelated latent variables remain invariant; derived transferred heat falls from tick 3.
- Preserved the existing same-seed observation-noise residuals instead of freezing or altering them. Both channels for every process variable agree and remain `GOOD`; signal magnitudes are bounded while keeping the canonical visible trend directions stable across tested seeds.
- Added a tick-4 `UNRESOLVED` decision with `INSUFFICIENT_EVIDENCE`, a tick-6 diagnosis with `REQUEST_COMPONENT_INSPECTION`, and a tick-7 persistent diagnosis with `REDUCE_SIMULATED_LOAD`. Each action applies on the next tick.
- The inspection application marks only the selected train pending maintenance at tick 7 without repairing its degradation. The load-reduction application lowers normalized load demand and heat source at tick 8 and changes the mode to `RECOVERY` while the selected train remains degraded.
- Added a contiguous, monotonic, backward-linked causal event chain and mature evidence spanning component-health decline, agreeing flow decline, correlated state change, and dependent delay. Visible payloads omit fault truth, scenario identifiers, severity, onset, numeric health, maintenance state, provenance, latent state, and targets.
- Added strict pump-specific fail-closed checks, deterministic replay and prefix tests, both train aliases, global RNG isolation, bounded per-tick changes, stable-noise residual parity, and `StructuredTrajectory` provenance validation. Golden G06 remains outside training and is not frozen because human review has not occurred.
- A fresh Sol reviewer returned `ship` with no material findings and changed no files. Exact hashes and Git state were identical before and after review. The reviewer noted one acceptable developmental residual: the component-evidence removal check filters an existing prefix rather than constructing a separately regenerated ablation trajectory; strengthen that test before golden-suite freeze.

## Files created or changed

- Foundation: `.editorconfig`, `.gitignore`, `.python-version`, `Makefile`, `pyproject.toml`, `uv.lock`, `configs/default.toml`.
- Public documentation: `README.md`, `SECURITY.md`.
- Project documentation: `docs/IMPLEMENTATION_STATUS.md`, `docs/PHASE0_AUDIT.md`, `docs/architecture.md`, `docs/threat-model.md`, `docs/security-controls.md`.
- Reconciled research: `research/DATASET_SPEC.md`, `research/DECISION_LOG.md`, `research/FICTIONAL_PLANT_SPEC.md`, `research/GOLDEN_SCENARIOS.md`, `research/PREBUILD_CHECKLIST.md`, `research/README.md`, `research/RESEARCH_BLUEPRINT.md`, `research/VOCABULARY_SEED.md`.
- Package: `src/reactorbench/__init__.py`, `src/reactorbench/config.py`, `src/reactorbench/resources.py`.
- Schema package: `src/reactorbench/schemas/__init__.py`, `base.py`, `enums.py`, `events.py`, `export.py`, `latent.py`, `observation.py`, `provenance.py`, `scenario.py`, `target.py`, `trajectory.py`.
- Simulator package: `src/reactorbench/simulator/__init__.py`, `content_guard.py`, `core.py`; the latest code milestone changed `__init__.py` and `core.py`.
- Reviewed contracts: `schemas/aster/v0/README.md`, `snapshot-contract.json`, seven `*.schema.json` files, and `manifest.json`.
- Tests: existing foundation tests plus `tests/unit/test_simulator.py`, `test_content_guard.py`; `tests/property/test_simulator_properties.py`; and `tests/contract/test_simulator_contract.py`. The latest milestone changed the simulator unit, property, and contract files.

Generated `dist/`, caches, run directories, corpora, checkpoints, and artifacts are ignored and are not release evidence.

## Tests and checks run

Environment: isolated CPython 3.12.11 managed under `/private/tmp`; lock resolved by `uv 0.8.24`. No global Python package installation was performed.

- `make sync` using the project lockfile: exit 0; 22 locked project/development packages installed into the isolated Python 3.12 environment.
- Focused final pump-degradation gate in the shared assembly tree: 59 tests passed.
- `make check`: exit 0 on the final pump-degradation milestone under CPython 3.12.11.
  - Ruff format: passed.
  - Ruff lint: all checks passed.
  - Mypy strict mode: success across 29 source files reported by mypy.
  - Pytest: 181 passed.
  - Branch coverage: 92.29%; required threshold 85%.
  - Build: `reactorbench_lm-0.1.0.tar.gz` and `reactorbench_lm-0.1.0-py3-none-any.whl` built successfully, with the wheel built from the sdist.
  - Artifact verifier: passed; wheel resources match reviewed roots, sdist omits incomplete tests, and the wheel installs/imports with `--no-deps --no-index` in an isolated target.
- `git diff --cached --check`: exit 0 after whitespace-only repository-hygiene corrections.
- `uv pip check` against the isolated Python 3.12 environment: exit 0; all 22 installed packages compatible.
- Current schema snapshot hash: `50a6b8ce8a4118d7598ef0131b050475844a21a00529047fbdcb5995ba2bccbc`.

## Decisions made

- Project name: `ReactorBench-LM`.
- Runtime baseline: Python 3.12; declared compatibility `>=3.12,<3.14`.
- Environment/build: `uv`, PEP 621, Hatchling, locked-project workflow.
- Initial runtime dependency: Pydantic v2 only; PyTorch remains deferred to the model phase.
- Internal schema interface: developmental `0.1.0`, explicitly not frozen as v1; validated model instances are immutable.
- Structured trajectory events use contiguous zero-based indices and monotonic fictional simulation ticks.
- `TaskTarget` is the public structured task root; legacy scenario decisions remain nested in `StructuredTrajectory`.
- Code and dataset licenses remain `TBD` and must be selected before distribution, not before local generator work.
- Browser and Computer Use are reserved for runnable local UI verification in Phase 7. Visualize is available for requested in-conversation design exploration; it does not replace repository-native UI work.
- Generator `0.1.0` currently supports Aster-A stable operation, a single primary-flow-channel `SENSOR_DRIFT`, a benign staged `LOAD_TRANSIENT`, that load driver combined with exactly one indefinite low-severity `SENSOR_STUCK` on either electrical-output channel, exactly one indefinite low-severity `SENSOR_NOISE` on either primary-thermal-state channel during steady operation, and exactly one indefinite low-severity `PUMP_DEGRADATION` on either primary train during steady operation. Supported trajectories are 8–64 fictional ticks; stuck/noise milestones default to 12 and require at least 8, while pump degradation requires at least 9.
- A decision is recorded at its decision tick, while any `ACTION_APPLIED` event occurs on the next tick. The suspect quality change follows the applied `FLAG_SENSOR_SUSPECT` action rather than preceding its evidence.
- `LOAD_TRANSIENT` direction and magnitude are deterministic functions of its seed; caller-selectable driver parameters remain deferred until they can be represented explicitly in validated scenario truth.
- The developmental stuck-load schedule uses onset tick 2, evidence/first decision tick 5, applied verification/second decision tick 6, and applied flag plus suspect quality tick 7. Its final settlement is linked to coordinated benign-load evidence, not to the sensor disagreement.
- Driver-plus-fault combinations are explicit compositional split keys. A held-out composition may expose its separate factors in training, but not the combination under another seed, channel, alias, or component role.
- Developmental sensor noise begins producing paired alternating offsets after onset tick 2. Decisions at ticks 3 and 4 abstain, tick 5 diagnoses and selects `COMPARE_RELATED_TRENDS`, tick 6 selects `FLAG_SENSOR_SUSPECT`, and all corresponding applied-action events occur on the next tick.
- `SENSOR_NOISE` structured fault truth and the later narrative-corruption `noise_test` split are independent dimensions; renderer corruption must not create a fault label.
- Developmental pump degradation begins selected-train health loss at tick 2, then stages flow/thermal/steam/output effects at ticks 3/4/5/6. Tick 4 abstains, tick 6 requests fictional component inspection, tick 7 selects simulated-load reduction, and their effects apply at ticks 5/7/8 without repairing the degraded train.
- Exact canonical types are revalidated at simulator entry even when immutable Pydantic models were bypassed with unchecked `model_copy` updates.
- The next likely process-fault milestone is `PUMP_TRIP`, but only after a read-only G07 contract and capability audit resolves its standby/dependency and paired-variant action semantics; this checkpoint does not silently settle that design.

## Assumptions

- All permitted scenarios remain project-authored and synthetic; no real facility, Navy-derived, operational, or proprietary material may enter code, fixtures, data, prompts, or outputs.
- The system Python 3.13.2 may be compatible, but reproducible gates use the isolated Python 3.12 baseline.
- Golden scenarios remain outside training. Concrete golden `decision_tick` values require generator fixtures and human review before freeze; implemented developmental behavior does not by itself freeze G04, G05, or G06.
- Exact account-level remaining usage is not observable in this environment. The requested 1% cutoff was not measured; durable phase checkpoints are the conservative fallback unless the user supplies the visible percentage.

## Known failures

- No known Phase 1 or completed Phase 2 milestone test, type, lint, formatting, build, snapshot, or artifact failure remains.
- The generator intentionally rejects ASTER-B/C, finite-duration drift/stuck/noise/pump cases, every unsupported fault family, multiple simultaneous faults, and every driver/fault composition except the single reviewed `LOAD_TRANSIENT` plus `SENSOR_STUCK` contract.
- The G06 component-evidence removal check is a filtered-prefix counterfactual rather than a separately regenerated ablation trajectory. This is acceptable for the developmental fixture but must be strengthened before the golden suite is frozen.
- The prohibited-content guard is deliberately non-exhaustive. A reviewed real-facility denylist and stratified human sample-review procedure remain required before any dataset pilot or release.
- A private public-reporting route cannot be configured until the owner creates the eventual public repository or names another private channel.
- Production headers, rate limits, safe service errors, artifact-loader startup checks, browser behavior, and deployment isolation are later-phase controls and are not claimed as verified.

## Open blockers

- No blocker prevents Phase 2 local generator work.
- Code/data license selection blocks distribution only.
- External publication, credentials, spending, hosted infrastructure, GitHub push, and Vercel deployment remain outside authorization.
- `uv` is available through the isolated temporary toolchain used for this checkpoint, not as a globally installed command. A future clean-machine reproduction guide must include an approved `uv` installation step.

## Uncommitted work and Git state

- The verified source/test assembly was synchronized to the intended project path for the pump-degradation integration.
- The complete `make check` gate was rerun successfully from the intended project immediately before the pump-degradation code commit.
- Local Git is initialized on `codex/foundation`; Phase 1 and the completed Phase 2 code milestones are committed locally.
- Last known Git branch: `codex/foundation`.
- Last known code commit: `193d195` (pump-degradation simulator milestone).
- Remote state: none; no remote will be created and no push is authorized.
- The expected worktree state immediately after this separate documentation checkpoint is clean; verify rather than assume it on resume. The status file records the preceding implementation commit because a commit cannot contain its own final Git hash.

## Relevant paths

- Intended repository: `/Users/zachary/Documents/Personal-Projects/AI-transformer`
- Shared assembly tree: `/Users/zachary/Documents/ChatGPT/Projects/.reactorbench-worktree`
- Authoritative requirements: `research/PROJECT_REQUIREMENTS.md`
- Current handoff: `docs/IMPLEMENTATION_STATUS.md`
- Configuration: `configs/default.toml`
- Package: `src/reactorbench/`
- Simulator: `src/reactorbench/simulator/`
- Schema snapshots: `schemas/aster/v0/`
- Tests: `tests/unit/`, `tests/property/`, `tests/contract/`
- Generated data, tokenizers, checkpoints, measured results, and UI artifacts: none

## Immediate next step

Perform a read-only audit of G07, the fictional `PUMP_TRIP` contract, Aster-A component/dependency map, standby representation, abrupt causal ordering, and context-dependent action labels. Define acceptance tests for the smallest deterministic milestone and identify any schema or variant prerequisite before making source changes. Do not implement the fault, add a plant variant, generalize composition, or alter the golden suite during this audit.

Exact recommended next command before implementation:

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
git status --short --branch
rg -n "G07|PUMP_TRIP|standby|dependency" research src tests
```

## Exact resume prompt

> Resume ReactorBench-LM from the safe checkpoint in
> /Users/zachary/Documents/Personal-Projects/AI-transformer.
> Read research/PROJECT_REQUIREMENTS.md and docs/IMPLEMENTATION_STATUS.md first.
> Inspect Git status and verify the recorded tests before making changes.
> Continue from the documented immediate next step without repeating completed work.
