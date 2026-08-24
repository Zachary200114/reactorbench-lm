# ReactorBench-LM implementation status

Last updated: 2026-08-24 America/New_York

Current phase: **Phase 6 remediation rerun ready; local macOS progress monitor
implemented and verified; rerun not started**

Current objective: hand the project owner a simple local window for readiness, Start,
verified progress, cooperative safe stop, and stopped-run Resume while preserving the
failed run and every scientific boundary. After the rerun completes, blocks, stops,
or fails, Codex should review its immutable evidence without retraining or opening
final data.

Checkpoint reason: the optional local Phase 6 monitor is complete and verified at a
safe pre-run boundary. Opening and visually checking the exact launcher did not create
the rerun, start training, evaluate data, or change the preserved failed-run evidence.

Project path:
/Users/zachary/Documents/Personal-Projects/AI-transformer

Exact account usage is not observable in this environment. No claim is made that an
account-level percentage was measured.

## Completed work

- Preserved Phase 6 v0.1 as immutable negative evidence.
- Completed v0.2 compact-output reliability: the frozen RB2 contract, strict
  bidirectional compiler, task-footer-preserving serialization, tokenizer
  reachability proof, truth-independent constrained decoding, task-balanced sampling,
  safe checkpoints, and development-only acceptance contracts.
- Completed the v0.3 scoped extension without changing historical v0.1/v0.2
  inventories. The frozen-source build expands 1,892 source projections into 5,859
  raw IID task examples used only in memory for cap reproduction, then removes 24
  same-task exact-prompt duplicates whose targets are identical. Only the resulting
  5,835 examples are written, audited, selected, or trained. All 55 group-atomic
  counterfactual rows remain bit-exact: 40 train and 15 validation.
- Embedded a bounded 24-row removal inventory and exact retained-counterpart bindings
  in the v0.3 compatibility report. Added target-leak scans, report-only task/class
  inventories, and task-scoped visible-structure separation.
- Completed conditional v0.4 engineering. The 1,024-token candidate activates only
  when both the frozen v0.2 and measured deduplicated-v0.3 IID-train prompt truncation
  rates are at least 0.10. Otherwise the pilot is a passing no-op and the 512-token
  control is reused.
- For an active v0.4 path, require native Apple MPS without fallback for ten-step
  batches 1, 2, and 4. The pilot profiles complete 1,024-token IID train/validation
  inventories, chooses the longest row per task, and proves the global train maximum
  appears in every batch schedule before longer-context training can start.
- Made scientific consumers reconstruct evidence instead of trusting self-checksummed
  summaries. They rebuild the deterministic 48-row selection, raw/deduplicated data
  bridge, tokenizer and length inventories, pilot schedules, training/checkpoint
  bindings, exact IID and six-shadow scopes, semantic composites, ranking tie-breaks,
  and v0.3/v0.4 acceptance results.
- Implemented the immutable 16-stage, non-overwriting pipeline with strict Git/config/
  artifact provenance, 30-second heartbeats, bounded progress, safe public errors,
  active-runtime and storage limits, and terminal review bundles.
- Added cooperative polling inside BOW optimizer steps, GRU train/evaluation batches,
  comparator boundaries, and each decoded example. No partial baseline, prediction,
  or semantic-view artifact is published when work stops.
- Made stop-request archival crash-durable and idempotent after strict equality.
  Cross-attempt resume copies and verifies a successor before symlink-safe retirement
  of superseded durable states.
- Added separate-process lifecycle coverage for start, status, stop, exit code 8,
  checkpoint resume, no completed-stage rerun, all 16 simulated stages, and the final
  access lock.
- Kept the ordinary runner physically separated from fresh final evaluation. The
  final executor remains intentionally unimplemented and fails closed.
- Added and updated the owner runbook, README, decision log, and executable wrappers.
- Preserved the first owner-operated run at
  `runs/phase6-remediation-v0.4.0-local/`. It completed four stage-prefix entries,
  including all 1,500 v0.2 development-training steps, then failed safely before its
  first gate decode. Final evaluation was not accessed.
- Diagnosed the exact boundary mismatch: the result requested `mps`, the loaded model
  parameter device was the equivalent `mps:0`, and strict decoder equality rejected
  the spelling difference. Checkpoint consumers now return the actual parameter
  device while rejecting a different device type or explicitly requested index.
- Added `phase6-remediation-v0.4.0-local-rerun-01` as the new default run identity.
  The failed run is neither deleted nor reused.
- Added a local-only native macOS Phase 6 monitor with overall and current-work
  progress bars, stage position, verified status details, a bounded activity log, and
  fixed confirmation-gated owner controls. It is explicitly not the Phase 7 UI.
- Added a standard-library controller that parses only the existing strictly
  validated status command, rejects unknown fields and mismatched run evidence, and
  routes only the frozen readiness, Start, status, safe-stop, Resume, and Finder
  wrappers. It accepts no arbitrary path, run name, configuration, or command.
- Start and Resume detach under `caffeinate`; closing the window does not kill an
  active run. Safe stop remains cooperative. There is deliberately no delete,
  overwrite, force-kill, or automatic start-over control.
- Added `Latest verified update UTC` to the validated status display and packaged the
  launcher, controller bridge, AppKit source/metadata, runbook, and rerun configuration
  in the source and wheel distributions.
- Visually verified the exact launcher in the pre-run state: `Not started`, stage 0 of
  16, correct source commit/run identity, Start enabled, and Stop/Resume/Finder
  disabled. Only Close was pressed; no lifecycle action was exercised.

## Frozen development evidence

- Frozen data source commit:
  992d86823a32813b226b73bc495d2ae6723d47ab
- Raw 5,859-example cap-only manifest:
  87420933f3e9549f8ef6785994a55e845e957c2942e0c5db0c4df5931075790a
- Sole written/audited/trained 5,835-example manifest:
  02ab7b7e29de7c74df5d308683b8c3d9f5d6204db0649a02d8288489d3be0af3
- Tokenizer manifest:
  ef80afa52030c764598663b0f51b90e7b753b91377b47b4a5648d729e0011ef8
- v0.2 inventory: 882 IID train/validation examples (630/252), with 882/882
  compile, target-fit, round-trip, constrained reachability, and footer retention;
  zero cap exhaustion; 668/882 prompt truncations retained honestly.
- v0.2 report checksum:
  c4f9739d67503714ca83b9ecbd4e65d288592161be1ffe5f2c038d5e3485295b
- Frozen raw v0.3 cap inventory: 55 counterfactual pairs (40/15), with 55/55
  compile, target-fit, round-trip, reachability, and footer retention; zero cap
  exhaustion; cap 108; every control prompt truncates at 512 tokens.
- v0.3 frozen cap report checksum:
  19612c9784612b2cbf5feb7c97a6bb2b351a510e28a853706ba212cfbfdf113f
- Preserved failed-run configuration checksum:
  4de973e2e009dccea7fc2ea430b4946c85b3066bde7b79673786479517ae666a
- New rerun configuration checksum:
  96850668232781faa9d14319ce40e136aa1ada0c85317ed88b08ef795fcd6a13
- Preserved v0.2 training result checksum:
  bc0332e69dd01aa9d2b48f5ca5c130c2e3c944a9435e5863399c0340bad68cac
- Preserved v0.2 checkpoint-manifest checksum:
  b27547e10fc0dfd08ea08337368dd3011c1c3bcb4e98747259ff49486ef9a44e
- Preserved v0.2 selected validation NLL: 0.1484147810350759 at the completed
  1,500-step boundary.

The failed v0.2 checkpoint is engineering evidence, not an accepted model-quality
result: the behavioral gate never decoded its first example. No remediation acceptance
result and no v0.3/v0.4 training result exists.

## Tests and checks run

- Final permitted repository gate after the GUI implementation:
  - Ruff format: 190 files passed.
  - Ruff lint: passed.
  - Strict mypy: 155 source files passed.
  - Pytest: 1,094 passed and 2 deliberately deselected in 548.54 seconds.
  - Branch coverage: 85.72%, above the required 85%.
- The two deselections are the documented historical final/golden asset readers:
  test_resource_api_reads_the_root_reviewed_assets_without_drift and
  test_approved_golden_packet_projects_sixty_examples. No held-out or final evaluation
  was run.
- Final GUI-focused check after the runbook correction: 44 passed and 1 protected
  resource-reader test was deselected in 1.37 seconds.
- Native Swift/AppKit type checking: passed. Property-list lint: passed. The GUI
  launcher and closed controller bridge both pass Bash syntax.
- Real non-window launcher smoke check: passed and reported `Not started`; its strict
  JSON snapshot reported the fixed rerun, stage total 16, and the expected conservative
  button policy. The rerun directory remained absent.
- Exact native launcher visual check: passed. The temporary app closed cleanly, and
  its temporary bundle was removed. Start, readiness, stop, resume, and Finder were not
  invoked.
- Source and wheel distributions rebuilt successfully. Archive listings confirm the
  AppKit source/metadata, local controller, both launcher scripts, corrected runbook,
  and `phase6-remediation-pipeline-v0.4.0-rerun-01.toml` are present.
- Preserved failed-run checksums remain unchanged:
  - run manifest:
    `ac1e0b5d1b5731d08c061fbceffd0fca115ee2e45003f986ee1d5c90091cd6c2`
  - pipeline state:
    `b2f5e2c8896bed6efef53c622c9bbf0234e84d295b5e8ab0c2d5f78892955eae`
  - status:
    `9292ac5256c0a633877f80a27db0e0a89dfe3cebd7efb24797a6dedf7789d786`
  - progress:
    `a7ae3b74c5ba66f864cd3cb69bae331ba6d2089bf61265975971f4495e9ccec0`
  - terminal review bundle:
    `aced75cf17027e5a8bdb4ee2a31b21996cd278fa06ccf2c59600afc4b92718ee`
- Git diff whitespace check: passed.

The module-name form of pytest-cov previously triggered a PyTorch import segmentation
fault in this local Python environment. The path-based coverage target above completed
successfully.

## Decisions and assumptions

- D-077 selects v0.2 checkpoints using six free-running structural indicators, with
  validation NLL and earlier step as frozen tie-breaks.
- D-078 defines task-scoped visible-structure separation and safe exact-duplicate
  removal.
- D-079 binds raw cap reproduction to the deduplicated training inventory.
- D-080 defines two-condition v0.4 activation and longest-sequence native-MPS proof.
- D-081 bounds cross-attempt storage and charges only active progress time.
- D-082 defines cooperative evaluation stopping and view-atomic artifact publication.
- D-083 requires downstream reconstruction of scientific evidence and ranking.
- D-084 makes stop archival crash-durable and conflict-intolerant.
- D-085 preserves the failed run, normalizes actual checkpoint device identity without
  weakening real mismatch rejection, and creates a new non-overwriting rerun name.
- D-086 defines the local-only native macOS monitor, strict status-only parser,
  closed wrapper allowlist, detached Start/Resume ownership, cooperative stop, and
  prohibition on delete, overwrite, or automatic restart.
- The project owner controls GitHub pushes, external publication, credentials, and
  deployment. Local checkpoint commits are permitted.

## Residual risks, known failures, and blockers

- The first long run failed after approximately 3,627.6 active seconds at the v0.2
  development gate. It is terminal and must not be resumed or edited.
- The replacement development rerun is pending. Approximately 6–24 hours on Apple MPS
  remains only a planning estimate.
- The monitor is a macOS owner utility, not a cross-platform product surface. It
  compiles a temporary AppKit bundle on each open, so the window can take roughly 20
  to 30 seconds to appear on this Mac.
- The final GUI check intentionally did not press readiness, Start, stop, Resume, or
  Finder. Lifecycle correctness is covered by unit/contract tests and the existing
  wrapper integration suite; the first real Start remains an owner action.
- CPU fallback is permitted for earlier stages but cannot satisfy an activated v0.4
  pilot. There is no successful activated-CPU estimate. An inactive control-only path
  may complete but provides no 1,024-token evidence.
- Recognized lock/temporary crash remnants and abandoned final-checkpoint directories
  are retained for review and count toward the 8 GiB limit. Repeated hard kills may
  eventually require reviewed manual intervention; unknown or unsafe entries fail
  closed.
- Phase 7 inference/UI remains blocked until a development candidate passes the frozen
  gates and later fresh-final prerequisites are independently implemented and
  approved.
- Fresh final evaluation remains locked and unimplemented. Do not create readiness or
  approval markers by hand.
- Code and dataset license placeholders remain deferred until release preparation.
- Exact account usage cannot be monitored here; the user must supply the visible
  percentage for a precise account-level cutoff.

## Files and artifact paths

- Pipeline configuration:
  configs/experiments/phase6-remediation-pipeline-v0.4.0-rerun-01.toml
- Preserved failed-run configuration:
  configs/experiments/phase6-remediation-pipeline-v0.4.0.toml
- Iteration configurations:
  configs/experiments/phase6-remediation-v0.2.0.toml,
  configs/experiments/phase6-remediation-v0.3.0.toml,
  configs/experiments/phase6-remediation-v0.4.0.toml
- Frozen inventory reports:
  docs/model/PHASE6_V02_INVENTORY.json,
  docs/model/PHASE6_V03_COUNTERFACTUAL_CAP.json
- Owner instructions:
  docs/model/PHASE6_REMEDIATION_RUNBOOK.md
- Local GUI launcher:
  scripts/open_phase6_progress_gui.sh
- Closed native-controller bridge:
  scripts/phase6_monitor_controller.sh
- Monitor implementation:
  src/reactorbench/remediation/Phase6RunMonitor.swift,
  src/reactorbench/remediation/Phase6RunMonitor-Info.plist,
  src/reactorbench/remediation/local_monitor.py
- Monitor tests:
  tests/unit/test_phase6_local_monitor.py
- Preserved failed run:
  runs/phase6-remediation-v0.4.0-local/
- New rerun root:
  runs/phase6-remediation-v0.4.0-local-rerun-01/
- Canonical live status:
  runs/phase6-remediation-v0.4.0-local-rerun-01/status.json
- Append-only progress:
  runs/phase6-remediation-v0.4.0-local-rerun-01/progress.jsonl

## Repository state

- Branch: main.
- Failed-run source/handoff commit:
  2aafcd1661ec7c3640a385621db171041532e547
- Complete device-fix/rerun implementation commit:
  f6f2369a050c9bf50d6c04351da603969a1f1273
- Local monitor implementation commit:
  295f8e894f16e9d49f84e8dbe29e36697824a340
- This status update is intended as the next local handoff commit. After it is created,
  main is ten commits ahead of origin/main.
- Uncommitted work after the status handoff commit: none; verify the clean tree on
  resume.
- The failed run directory is preserved; the rerun directory does not exist yet.
- No fresh-final ledger/result was created or accessed.
- No push or deployment was performed during this work.

## Immediate next step

Open the local monitor from the final clean local commit. Wait for the native window
to appear, confirm it reports `Not started` and the fixed rerun name, optionally run
the non-mutating readiness check, and press `Start new rerun` only when ready for the
long owner-operated run. Do not resume or modify the preserved failed run.

## Exact recommended next command after final handoff

    cd /Users/zachary/Documents/Personal-Projects/AI-transformer
    git status --short --branch
    ./scripts/open_phase6_progress_gui.sh

Opening the monitor does not start training. The readiness, Start, cooperative stop,
and stopped-run Resume controls are inside the window. The locked evaluation wrapper
is not a research command in this release.

## Resume prompt

Resume ReactorBench-LM from the safe checkpoint in
/Users/zachary/Documents/Personal-Projects/AI-transformer.
Read research/PROJECT_REQUIREMENTS.md and docs/IMPLEMENTATION_STATUS.md first.
Inspect Git status and verify the recorded tests before making changes.
Continue from the documented immediate next step without repeating completed work.
