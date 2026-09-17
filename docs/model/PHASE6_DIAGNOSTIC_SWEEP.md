# Phase 6 diagnostic full sweep

Status: **targeted-05 diagnostic preserved as failed engineering evidence; targeted-06
diagnostic implemented and not started**

## Why I added it

The official Phase 6 run is deliberately fail-fast: if a scientific acceptance gate
does not pass, later stages do not run. That is the right behavior for certification,
but it made development unnecessarily slow because I had to spend another full run to
discover the next model-quality miss.

The diagnostic full sweep solves that workflow problem without weakening an acceptance
threshold. The first diagnostic run used targeted-05 data, model, training,
evaluation, and the frozen ten-check policy. It reached v0.4 shadow evaluation and
proved the workflow useful, but an undersized historical counterfactual cap stopped
the stage after three shadow views. That run remains unchanged. The new targeted-06
diagnostic adds a pre-training cap audit and can collect isolated shadow-view boundary
failures before stopping the stage safely.

## Exact identity

- Run: `phase6-remediation-v0.4.1-targeted-06-diagnostic-02`
- Config:
  `configs/experiments/phase6-remediation-pipeline-v0.4.1-targeted-06-diagnostic-02.toml`
- Config SHA-256:
  `3c81b6197c6b971a9d8e57e63e5fb18db5b73f71557a1c845dbb1dbfa75e6bab`
- Mode: `collect_scientific_failures`
- Official targeted-06 pipeline SHA-256:
  `e7539a66c4df658807fa6d0c0fa0a8d12bf0b47939a5931d69c84befff35dc48`
- Preserved failed diagnostic:
  `runs/phase6-remediation-v0.4.0-targeted-05-diagnostic-01/`

## What may continue

Only these two completed scientific outcomes may be recorded as
`scientific_failed` while the next stage begins:

1. `v03_gate`
2. `v04_gate_and_final_policy_freeze`

This is a code-level allowlist, not a configurable list. Missing thresholds, changed
thresholds, malformed results, bad checksums, unsafe paths, provenance drift,
resource-limit failures, pilot infeasibility, stop requests, and other non-allowlisted
denials still stop the run. During v0.4 shadow evaluation, an isolated
contract/boundary `ValueError` is recorded without its raw message while the remaining
independent views run. The stage still ends failed and cannot advance; this exists
only to collect more diagnostic evidence in one attempt.

## What it can never do

The diagnostic sweep cannot:

- turn a failed check into a pass;
- produce an official Phase 6 acceptance result;
- authorize Phase 7;
- create final-evaluation readiness or owner-approval files;
- access fresh final, historical final, or golden payloads; or
- replace the official fail-fast run.

The v0.4 gate writes `diagnostic-final-evaluation-lock.json` instead of an official
final-access policy. That lock always says final evaluation is unauthorized, even if
all development checks happen to pass.

## End-of-run evidence

If engineering and integrity checks stay healthy through all 16 stages, the pipeline
finishes with `diagnostic_completed`. The review stage writes:

- `diagnostic-sweep-report.json`, a strict checksum-bound stage and failure inventory;
- `DIAGNOSTIC_SWEEP_REPORT.md`, a readable summary; and
- the permanent diagnostic final-access lock from stage 15.

This status means “the diagnostic sweep finished,” not “the model passed.”

## Using the GUI

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
./scripts/open_phase6_progress_gui.sh
```

Use the **Official fail-fast** / **Diagnostic full sweep** selector at the top of the
window. Select **Diagnostic full sweep**, run **Readiness check**, then press
**Start diagnostic full sweep**. The progress bars, status, safe stop, Resume, Finder,
and alarm controls follow the selected run only.

The official selection retains the original fail-fast behavior. Switching the monitor
view does not start, stop, resume, delete, or modify either run.

## Terminal equivalents

```bash
./scripts/run_phase6_diagnostic_pipeline.sh --dry-run
caffeinate -i ./scripts/run_phase6_diagnostic_pipeline.sh
./scripts/check_phase6_diagnostic_status.sh
./scripts/stop_phase6_diagnostic_pipeline.sh
caffeinate -i ./scripts/resume_phase6_diagnostic_pipeline.sh
```

Every wrapper is fixed to this exact config and accepts no caller-supplied run name,
model path, checkpoint, or arbitrary command.
