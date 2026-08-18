# ReactorBench-LM implementation status

Last updated: 2026-08-18 16:46 EDT
Current phase: Phase 2 starting — deterministic Aster Station structured generator
Checkpoint reason: Phase 1 was reverified and preserved in a local checkpoint commit before generator work
Intended project path: `/Users/zachary/Documents/Personal-Projects/AI-transformer`

## Current objective

Implement Phase 2 with acceptance-test-first deterministic stable operation and one `SENSOR_DRIFT` scenario only. Prove seeded replay, latent/observation separation, bounded state evolution, causal event ordering, leakage resistance, and evidence-gated abstention before adding any other fault. Do not render a dataset, train a tokenizer/model, build the UI, or expose a network service until the corresponding later phase gates pass.

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

## Files created or changed

- Foundation: `.editorconfig`, `.gitignore`, `.python-version`, `Makefile`, `pyproject.toml`, `uv.lock`, `configs/default.toml`.
- Public documentation: `README.md`, `SECURITY.md`.
- Project documentation: `docs/IMPLEMENTATION_STATUS.md`, `docs/PHASE0_AUDIT.md`, `docs/architecture.md`, `docs/threat-model.md`, `docs/security-controls.md`.
- Reconciled research: `research/DATASET_SPEC.md`, `research/DECISION_LOG.md`, `research/FICTIONAL_PLANT_SPEC.md`, `research/GOLDEN_SCENARIOS.md`, `research/PREBUILD_CHECKLIST.md`, `research/README.md`, `research/RESEARCH_BLUEPRINT.md`, `research/VOCABULARY_SEED.md`.
- Package: `src/reactorbench/__init__.py`, `src/reactorbench/config.py`, `src/reactorbench/resources.py`.
- Schema package: `src/reactorbench/schemas/__init__.py`, `base.py`, `enums.py`, `events.py`, `export.py`, `latent.py`, `observation.py`, `provenance.py`, `scenario.py`, `target.py`, `trajectory.py`.
- Reviewed contracts: `schemas/aster/v0/README.md`, `snapshot-contract.json`, seven `*.schema.json` files, and `manifest.json`.
- Tests: `tests/unit/test_config.py`, `test_schemas.py`, `test_schema_gates.py`; `tests/property/test_schema_gate_properties.py`; `tests/contract/test_package_resources.py`, `test_schema_snapshots.py`, `test_snapshot_validation.py`, `verify_distribution_artifacts.py`.

Generated `dist/`, caches, run directories, corpora, checkpoints, and artifacts are ignored and are not release evidence.

## Tests and checks run

Environment: isolated CPython 3.12.11 managed under `/private/tmp`; lock resolved by `uv 0.8.24`. No global Python package installation was performed.

- `make sync` using the project lockfile: exit 0; 22 locked project/development packages installed into the isolated Python 3.12 environment.
- `make check`: exit 0.
  - Ruff format: 45 files already formatted.
  - Ruff lint: all checks passed.
  - Mypy strict mode: success across 22 source/test files reported by mypy.
  - Pytest: 104 passed in 1.15 seconds on the final pre-commit rerun.
  - Branch coverage: 89.58%; required threshold 85%.
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

## Assumptions

- All permitted scenarios remain project-authored and synthetic; no real facility, Navy-derived, operational, or proprietary material may enter code, fixtures, data, prompts, or outputs.
- The system Python 3.13.2 may be compatible, but reproducible gates use the isolated Python 3.12 baseline.
- Golden scenarios remain outside training. Concrete golden `decision_tick` values require later generator fixtures and human review before freeze.
- Exact account-level remaining usage is not observable in this environment. The requested 1% cutoff was not measured; durable phase checkpoints are the conservative fallback unless the user supplies the visible percentage.

## Known failures

- No known Phase 1 test, type, lint, formatting, build, snapshot, or artifact failure remains.
- A private public-reporting route cannot be configured until the owner creates the eventual public repository or names another private channel.
- Production headers, rate limits, safe service errors, artifact-loader startup checks, browser behavior, and deployment isolation are later-phase controls and are not claimed as verified.

## Open blockers

- No blocker prevents Phase 2 local generator work.
- Code/data license selection blocks distribution only.
- External publication, credentials, spending, hosted infrastructure, GitHub push, and Vercel deployment remain outside authorization.
- `uv` is available through the isolated temporary toolchain used for this checkpoint, not as a globally installed command. A future clean-machine reproduction guide must include an approved `uv` installation step.

## Uncommitted work and Git state

- The verified Phase 1 tree is assembled in `/Users/zachary/Documents/ChatGPT/Projects/.reactorbench-worktree` and fully synchronized to the intended project path.
- The complete `make check` gate was rerun successfully from the intended project immediately before the foundation commit.
- Local Git is initialized on `codex/foundation`; the Phase 1 source, contracts, tests, and documentation are committed.
- Last known Git branch: `codex/foundation`.
- Last known Git commit: `0cfc98cf6a30f27ce22e802e6cd0cf4dfb264ab0` (`chore: establish ReactorBench-LM foundation`).
- Remote state: none; no remote will be created and no push is authorized.
- Uncommitted work after the local documentation checkpoint: none; Phase 2 source changes have not begun.

## Relevant paths

- Intended repository: `/Users/zachary/Documents/Personal-Projects/AI-transformer`
- Shared assembly tree: `/Users/zachary/Documents/ChatGPT/Projects/.reactorbench-worktree`
- Authoritative requirements: `research/PROJECT_REQUIREMENTS.md`
- Current handoff: `docs/IMPLEMENTATION_STATUS.md`
- Configuration: `configs/default.toml`
- Package: `src/reactorbench/`
- Schema snapshots: `schemas/aster/v0/`
- Tests: `tests/unit/`, `tests/property/`, `tests/contract/`
- Generated data, tokenizers, checkpoints, measured results, and UI artifacts: none

## Immediate next step

Begin Phase 2 by freezing focused acceptance tests and the minimal Aster-A configuration for stable operation plus `SENSOR_DRIFT`. Implement no additional fault family until this gate passes.

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
