# Phase 0 workspace and decision audit

Status: complete as a dated pre-implementation audit
Audit date: 2026-08-18
Intended project path: `/Users/zachary/Documents/Personal-Projects/AI-transformer`

## Audit facts

- The intended project path exists with the corrected name `AI-transformer`; the former trailing-space issue is resolved.
- At the audit snapshot, the intended project contained `.DS_Store` and the 16-file `research/` package only.
- The research package includes the canonical requirements, Aster Station and dataset specifications, experiment plan, golden scenarios, security plan, UI plan, release plan, source manifest, and related design material.
- No `AGENTS.md` applied at the intended project root or the shared staging root.
- Neither the intended project nor the staging tree had Git metadata at the start of the audit. There was therefore no branch, commit, remote, or prior tracked work to preserve.
- The installed system `python3` reported Python 3.13.2.
- `uv` was not installed or visible on `PATH` at the audit snapshot.
- The implementation is being assembled in `/Users/zachary/Documents/ChatGPT/Projects/.reactorbench-worktree` because that directory is writable in the current environment. Integration into the intended path is a separate, explicit local copy step.

No dependency installation, Git initialization, remote creation, publication, deployment, dataset generation, or training occurred as part of this audit.

Post-audit update: the staging tree was copied into the intended project path on
2026-08-18. Local integration is complete, but local Git is still not initialized;
no remote, push, publication, or deployment was created.

## Authoritative document resolution

The implementation follows the precedence supplied in the starting request, led by:

1. `research/PROJECT_REQUIREMENTS.md`;
2. `research/DECISION_LOG.md`;
3. `research/PREBUILD_CHECKLIST.md`;
4. the plant, dataset, experiment, golden-suite, security, reproducibility, UI, live-demo, and blueprint specifications in that order; and
5. the remaining research documents.

No conflict was found that prevents the local Phase 1 foundation. When a provisional value differs from a measured outcome, the measured pilot result may change it only through an explicit recorded decision.

## Safe decisions for Phase 1

### Identity and runtime

- Use `ReactorBench-LM` as the final project name.
- Use Python 3.12 as the reproducibility baseline with the declared compatibility window `>=3.12,<3.14`.
- Use `uv` dependency groups and a locked dependency workflow. A `uv.lock` is present at the local integration checkpoint; its presence is not a commit claim while local Git remains uninitialized.
- Use a `src/` Python package layout and Hatchling build backend.
- Keep the initial runtime dependency surface to Pydantic v2. PyTorch enters only when the model phase requires it.

### Versioned schema contract

The Phase 1 development schema is version `0.1.0` and records `frozen=false`; it is
not falsely labeled as a frozen v1 interface. Individual Pydantic model instances
are immutable after validation. Instance immutability does not freeze the
developmental interface against reviewed schema changes.

Selected safety properties are:

- a shared strict, immutable Pydantic v2 base with unknown-field rejection and default validation;
- finite normalized floating-point values bounded to `[0, 1]`;
- strict, bounded identifiers, nonnegative integer ticks and indices, and unsigned 32-bit seeds;
- separate latent-state, observation, canonical-event, scenario, target, and provenance modules;
- `LOAD_TRANSIENT` represented as a scenario driver rather than a fault;
- explicit `DIAGNOSED`, `NO_FAULT`, and `UNRESOLVED` diagnosis states;
- an unresolved diagnosis requiring no asserted fault and the `INSUFFICIENT_EVIDENCE` abstention reason;
- deterministic fault ordering and deterministic canonical JSON hashing;
- one immediate fictional action per target decision tick, distinct from an ordered scenario action sequence; and
- distinct channel-quality and observation-status concepts.

These resolutions implement the documented separation between ground truth and visible evidence while avoiding silent type coercion and ambiguous target states. They do not freeze the later public inference schema.

### Configuration and filesystem behavior

- Configuration must reject unknown fields and unsafe coercion.
- Output locations must be project-relative and reviewed rather than arbitrary user-controlled paths.
- A run directory must not be silently overwritten.
- Resolved configuration and its deterministic hash must accompany generated artifacts.
- The default seed is a reproducibility convenience, not evidence of complete determinism across all future hardware backends.

## Material decisions still unresolved

| Decision | Current handling | Blocking effect |
|---|---|---|
| Code license: Apache-2.0 or MIT | Do not create a license or claim reuse rights | Blocks public distribution decision, not local work |
| Dataset license: CC BY 4.0 or CC0 | Do not label unreleased data | Blocks dataset publication, not generator work |
| Exact model, tokenizer, corpus, optimizer, and context sizes | Defer to smoke and pilot evidence | Blocks main-run freeze, not schemas |
| LSTM/GRU baseline | Decide before test evaluation from measured compute | Does not block foundation |
| Model-weight publication | Defer until artifact, license, size, and safety review | Blocks weight release only |
| Inference host and production limits | Select from measured checkpoint and load benchmarks | Blocks deployment, not local service design |

## Explicitly excluded actions

The project owner asked this implementation effort to complete the local project while leaving publication to them. Accordingly:

- do not push to GitHub;
- do not create or modify a hosted remote;
- do not deploy to Vercel or another inference host;
- do not publish packages, datasets, checkpoints, releases, or websites; and
- do not spend money or provision external infrastructure.

Local Git initialization and local commits may be used later for safe versioning if the integration owner determines that all changes are understood. They do not authorize a remote push.

## Phase plan and first milestone

- **Phase 0 — Audit:** workspace facts, authoritative-document resolution, safety boundaries, and deferred decisions.
- **Phase 1 — Foundation:** packaging, strict configuration, developmental versioned schemas, documentation, and focused tests.
- **Phase 2 — Structured generator:** stable operation plus initial faults, deterministic transitions, observation separation, invariants, and property tests.
- **Phase 3 — Dataset and renderer:** split-first manifests, deterministic rendering, leakage checks, provenance, and a tiny pilot only.
- **Phase 4 — Model correctness:** project tokenizer, from-scratch causal Transformer, masking and checkpoint tests, and tiny-shard overfit.
- **Phase 5 — Baselines and pilot:** preregistered comparisons and measured Apple MPS feasibility.
- **Phase 6 — Main experiments:** frozen gates, separate split reporting, calibration, robustness, abstention, and failure analysis.
- **Phase 7 — Inference and UI:** narrow local inference service and Research Editorial UI driven only by recorded artifacts; no deployment.
- **Phase 8 — Release readiness:** cards, documentation, checksums, provenance, SBOM, and local release verification; no publication or deployment.

The Phase 1 milestone is a strict, importable contract whose configuration and
schemas reject malformed or ambiguous values, remain deterministic under
serialization, and have focused tests. At the 2026-08-18 Python 3.12 checkpoint,
`make check` and `make build` passed; a separate coverage run measured 91.41%.
Exact evolving integration state belongs in `IMPLEMENTATION_STATUS.md`.

## Usage visibility

The coding environment does not expose the project owner's account-level remaining-usage percentage to this audit. The implementation must not claim it measured the requested 1% threshold. The durable status file and conservative phase-boundary checkpoints are the available fallback; a precise cutoff requires the user to provide the visible percentage.
