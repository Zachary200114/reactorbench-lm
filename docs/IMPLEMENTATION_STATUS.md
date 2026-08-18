# ReactorBench-LM implementation status

Last updated: 2026-08-18 17:47 EDT
Current phase: Phase 2 in progress — benign load-transition prerequisite complete
Checkpoint reason: staged `LOAD_TRANSIENT` behavior passed full integration and corrected independent review, then was preserved in a local commit
Intended project path: `/Users/zachary/Documents/Personal-Projects/AI-transformer`

## Current objective

Implement the smallest coherent `SENSOR_STUCK` milestone as one Aster-A `LOAD_TRANSIENT` plus one frozen electrical-output channel. Prove that latent state exactly matches the benign transient, only the selected observation freezes, diagnosis follows genuine related movement, and the two documented actions occur in causal order. Do not add other fault families, render a dataset, train a tokenizer/model, build the UI, or expose a network service until their later gates pass.

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

## Files created or changed

- Foundation: `.editorconfig`, `.gitignore`, `.python-version`, `Makefile`, `pyproject.toml`, `uv.lock`, `configs/default.toml`.
- Public documentation: `README.md`, `SECURITY.md`.
- Project documentation: `docs/IMPLEMENTATION_STATUS.md`, `docs/PHASE0_AUDIT.md`, `docs/architecture.md`, `docs/threat-model.md`, `docs/security-controls.md`.
- Reconciled research: `research/DATASET_SPEC.md`, `research/DECISION_LOG.md`, `research/FICTIONAL_PLANT_SPEC.md`, `research/GOLDEN_SCENARIOS.md`, `research/PREBUILD_CHECKLIST.md`, `research/README.md`, `research/RESEARCH_BLUEPRINT.md`, `research/VOCABULARY_SEED.md`.
- Package: `src/reactorbench/__init__.py`, `src/reactorbench/config.py`, `src/reactorbench/resources.py`.
- Schema package: `src/reactorbench/schemas/__init__.py`, `base.py`, `enums.py`, `events.py`, `export.py`, `latent.py`, `observation.py`, `provenance.py`, `scenario.py`, `target.py`, `trajectory.py`.
- Simulator package: `src/reactorbench/simulator/__init__.py`, `content_guard.py`, `core.py`.
- Reviewed contracts: `schemas/aster/v0/README.md`, `snapshot-contract.json`, seven `*.schema.json` files, and `manifest.json`.
- Tests: existing foundation tests plus `tests/unit/test_simulator.py`, `test_content_guard.py`; `tests/property/test_simulator_properties.py`; and `tests/contract/test_simulator_contract.py`.

Generated `dist/`, caches, run directories, corpora, checkpoints, and artifacts are ignored and are not release evidence.

## Tests and checks run

Environment: isolated CPython 3.12.11 managed under `/private/tmp`; lock resolved by `uv 0.8.24`. No global Python package installation was performed.

- `make sync` using the project lockfile: exit 0; 22 locked project/development packages installed into the isolated Python 3.12 environment.
- `make check`: exit 0.
  - Ruff format: 52 files already formatted.
  - Ruff lint: all checks passed.
  - Mypy strict mode: success across 29 source/test files reported by mypy.
  - Pytest: 152 passed in 3.89 seconds on the final benign-load milestone rerun.
  - Branch coverage: 91.37%; required threshold 85%.
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
- Generator `0.1.0` currently supports Aster-A stable operation, a single primary-flow-channel `SENSOR_DRIFT`, and a benign staged `LOAD_TRANSIENT`; supported trajectories are 8–64 fictional ticks.
- A decision is recorded at its decision tick, while any `ACTION_APPLIED` event occurs on the next tick. The suspect quality change follows the applied `FLAG_SENSOR_SUSPECT` action rather than preceding its evidence.
- `LOAD_TRANSIENT` direction and magnitude are deterministic functions of its seed; caller-selectable driver parameters remain deferred until they can be represented explicitly in validated scenario truth.
- The next causal milestone is `SENSOR_STUCK` composed with the verified load transient, because a frozen channel is meaningful only against a genuinely changing latent signal.

## Assumptions

- All permitted scenarios remain project-authored and synthetic; no real facility, Navy-derived, operational, or proprietary material may enter code, fixtures, data, prompts, or outputs.
- The system Python 3.13.2 may be compatible, but reproducible gates use the isolated Python 3.12 baseline.
- Golden scenarios remain outside training. Concrete golden `decision_tick` values require later generator fixtures and human review before freeze.
- Exact account-level remaining usage is not observable in this environment. The requested 1% cutoff was not measured; durable phase checkpoints are the conservative fallback unless the user supplies the visible percentage.

## Known failures

- No known Phase 1 or completed Phase 2 milestone test, type, lint, formatting, build, snapshot, or artifact failure remains.
- The generator intentionally rejects ASTER-B/C, finite-duration drift, every other fault family, and every driver/fault composition—including `LOAD_TRANSIENT` plus `SENSOR_STUCK`—until its own acceptance gate is implemented.
- The prohibited-content guard is deliberately non-exhaustive. A reviewed real-facility denylist and stratified human sample-review procedure remain required before any dataset pilot or release.
- A private public-reporting route cannot be configured until the owner creates the eventual public repository or names another private channel.
- Production headers, rate limits, safe service errors, artifact-loader startup checks, browser behavior, and deployment isolation are later-phase controls and are not claimed as verified.

## Open blockers

- No blocker prevents Phase 2 local generator work.
- Code/data license selection blocks distribution only.
- External publication, credentials, spending, hosted infrastructure, GitHub push, and Vercel deployment remain outside authorization.
- `uv` is available through the isolated temporary toolchain used for this checkpoint, not as a globally installed command. A future clean-machine reproduction guide must include an approved `uv` installation step.

## Uncommitted work and Git state

- The verified assembly tree is synchronized to the intended project path.
- The complete `make check` gate was rerun successfully from the intended project immediately before the benign-load commit.
- Local Git is initialized on `codex/foundation`; Phase 1 and both completed Phase 2 generator milestones are committed.
- Last known Git branch: `codex/foundation`.
- Last known Git commit: `ce74ebab1f0f26759a05678fe77fa4712d8d8f1b` (`feat: add benign Aster-A load transients`).
- Remote state: none; no remote will be created and no push is authorized.
- Uncommitted work after the local documentation checkpoint: none; `SENSOR_STUCK` source work has not begun.

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

Freeze acceptance tests for one Aster-A `LOAD_TRANSIENT` plus one indefinite, low-severity `SENSOR_STUCK` on either electrical-output channel. Implement its observation-only freeze and two-action causal sequence without adding another fault or a generalized composition engine.

Exact recommended next command before implementation:

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
git status --short --branch
```

## Exact resume prompt

> Resume ReactorBench-LM from the safe checkpoint in
> /Users/zachary/Documents/Personal-Projects/AI-transformer.
> Read research/PROJECT_REQUIREMENTS.md and docs/IMPLEMENTATION_STATUS.md first.
> Inspect Git status and verify the recorded tests before making changes.
> Continue from the documented immediate next step without repeating completed work.
