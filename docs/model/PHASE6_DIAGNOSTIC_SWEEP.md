# Phase 6 diagnostic full sweep

Status: **implemented as a separate non-certifying development workflow; not started**

## Why I added it

The official Phase 6 run is deliberately fail-fast: if a scientific acceptance gate
does not pass, later stages do not run. That is the right behavior for certification,
but it made development unnecessarily slow because I had to spend another full run to
discover the next model-quality miss.

The diagnostic full sweep solves that workflow problem without weakening an acceptance
threshold. It uses the same targeted-05 data, model, training, evaluation, and frozen
ten-check policy, but it records allowlisted scientific gate misses and continues far
enough to collect the remaining development evidence in one run.

## Exact identity

- Run: `phase6-remediation-v0.4.0-targeted-05-diagnostic-01`
- Config:
  `configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-05-diagnostic-01.toml`
- Config SHA-256:
  `f3eaeb84ee20834571d9b20caf8a0c946622327125f77b5e691c67db9c4c5137`
- Mode: `collect_scientific_failures`
- Existing targeted-05 model/data/config hashes: unchanged
- Existing official targeted-05 config SHA-256: unchanged at
  `bfdb3833fce80fd31e018126b1ae225e04cf85d7fcbd9ef0fcada69a69f4a354`

## What may continue

Only these two completed scientific outcomes may be recorded as
`scientific_failed` while the next stage begins:

1. `v03_gate`
2. `v04_gate_and_final_policy_freeze`

This is a code-level allowlist, not a configurable list. Missing thresholds, changed
thresholds, malformed results, exceptions, bad checksums, unsafe paths, provenance
drift, resource-limit failures, pilot infeasibility, stop requests, and any other
non-allowlisted denial still stop the run.

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
