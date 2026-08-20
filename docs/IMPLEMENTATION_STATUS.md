# ReactorBench-LM implementation status

Last updated: 2026-08-20 America/New_York
Current phase: **Phase 2 complete locally — structured generator**
Current objective: begin Phase 3 with a read-only dataset/task projection and split-first manifest audit; do not render a pilot dataset yet.
Checkpoint reason: expanded Phase 2 completed through the final compound and sparse-evidence generator gates. This handoff records the verified local code state before any dataset work.
Intended project path: `/Users/zachary/Documents/Personal-Projects/AI-transformer`

## Completed work

- Phase 0 audit and Phase 1 repository/contracts foundation are complete.
- Phase 2 implements deterministic, fail-closed Aster Station development fixtures G01–G15: stable, observation-only sensor faults, benign load transient, process faults, valve lag/stuck contrast, transfer and flow faults, variant-aware support-power interruption, abstract inventory loss, a G14 compound, and a G15 sparse audit fixture.
- Immutable Aster-A, Aster-B, and Aster-C cards/maps and strict alias/component/channel validation are present. The schema and generator remain `0.1.0` developmental with `frozen=false`.
- Single-family behavior has exact causal, counterfactual, prefix, deterministic replay, bounded-value, global-RNG isolation, and fail-closed contract coverage. G14 is the deliberately narrow compound `SENSOR_DRIFT` plus `PUMP_DEGRADATION`; its semantic labels are unordered but serialize canonically as `(SENSOR_DRIFT, PUMP_DEGRADATION)` under D-039. G15 is a sparse, truth-filtered audit fixture that must abstain.
- Visible structured payloads exclude latent truth, fault injection, scenario/provenance identifiers, targets, and action sequences. A truth-filtered audit payload is **not** a model prompt: Phase 3 must produce decision-tick/channel-projected task inputs.
- No dataset, renderer, tokenizer, model, checkpoint, measured model result, inference service, or UI exists. No GitHub push, Vercel deployment, or public release has occurred.

## Files created or changed in the verified code checkpoint

- Local code commits: `f61327d` (`G13`/`G15`) and `b63f6d8` (final Phase 2 compound generator).
- Generator, schemas, tests, and the reconciled Phase 2 documentation are in the intended project path.
- The documentation checkpoint changes: `README.md`; `docs/IMPLEMENTATION_STATUS.md`, `docs/architecture.md`, `docs/security-controls.md`, `docs/threat-model.md`; and the listed `research/` planning documents.

## Tests and checks run

- Fresh intended-project full gate after documentation synchronization: **434 tests passed** on Python **3.12.11**, with **92.20% branch coverage**; Ruff format and lint passed; strict mypy passed for **41 source files**; sdist/wheel build and isolated artifact verification passed.
- Schema snapshot remains unchanged: `2a7f8b120658c3414dbc2d4b1935f20274b5eb416cd91e16a8edc2411d01add4`.
- Documentation stale-text, trailing-whitespace, and actual-project diff-whitespace checks passed before the local checkpoint commit.

## Decisions made

- Phase 2 is locally complete but developmental. It does not freeze schemas, golden scenarios, datasets, or acceptance thresholds.
- Keep G14’s human-facing semantic order as “pump degradation plus sensor drift” where useful, while treating compound labels as sets and using D-039’s canonical serialized order `(SENSOR_DRIFT, PUMP_DEGRADATION)` for data, hashes, and exact-match checks.
- Phase 3 must assign split groups before any narrative rendering. Required grouping/shortcut controls include G07 matched standby contexts; G08/G09 lag-versus-stuck counterfactuals; G12 included/withheld dependency-map contexts; G14 compound/pump-only/drift-only factor and role relationships; and G15 sparse/expanded evidence relatives. Audit-only IDs and post-decision action effects cannot enter prompts.
- The Phase 2 prohibited-content scanner is intentionally narrow and redacting. A reviewed full denylist and stratified human sample review remain mandatory before a dataset pilot.
- License remains `TBD`; it blocks distribution only, not local Phase 3 preparation.

## Assumptions

- All scenarios remain project-authored, wholly fictional, normalized, and non-operational.
- Exact account usage percentage is not observable in this environment. Conservative phase checkpoints are used; no claimed 1% account cutoff has been measured.

## Known failures and residual risks

- No known failure remains in the recorded full Phase 2 gate.
- The golden suite has not received the required human review and is not frozen. No golden checksum or final evaluation claim exists.
- The scanner cannot prove absence of prohibited material; full denylist review and human sampling are deferred to Phase 3.
- Prompt projection, split manifest schemas, duplicate/leakage checks, renderer contamination tests, pilot rendering, and all model/service/UI controls are not implemented.

## Open blockers

- None for the next read-only Phase 3 projection and split-manifest audit.
- License selection is still required before any external distribution.

## Uncommitted work

- None at this completed Phase 2 handoff. The intended repository is expected to be clean after the local documentation checkpoint commit. No remote, push, publication, or deployment exists.

## Immediate next step

Implement no renderer or dataset yet. First define and test a strict, read-only Phase 3 task-projection and split-manifest contract: source audit trajectories; derive prompts only from allowed channel/event prefixes through a decision tick; capture grouping metadata; reject shortcut-bearing fields; and audit required split groups before allowing pilot rendering.

## Exact recommended next command

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer && git status --short && sed -n '1,240p' research/DATASET_SPEC.md && sed -n '1,260p' docs/IMPLEMENTATION_STATUS.md
```

## Relevant artifacts and configuration

- Schema/generator version: `0.1.0`, `frozen=false`
- Schema snapshot SHA-256: `2a7f8b120658c3414dbc2d4b1935f20274b5eb416cd91e16a8edc2411d01add4`
- Latest local code commit: `b63f6d8`
- Prior Phase 2 code commit: `f61327d`
- Project configuration: `configs/default.toml`
- Generator package: `src/reactorbench/simulator/`
- Phase 3 contracts to consult: `research/DATASET_SPEC.md`, `research/GOLDEN_SCENARIOS.md`, and `docs/security-controls.md`

## Exact resume prompt

Resume ReactorBench-LM from the safe checkpoint in
/Users/zachary/Documents/Personal-Projects/AI-transformer.
Read research/PROJECT_REQUIREMENTS.md and docs/IMPLEMENTATION_STATUS.md first.
Inspect Git status and verify the recorded tests before making changes.
Continue from the documented immediate next step without repeating completed work.
