# ReactorBench-LM implementation status

Last updated: 2026-08-24 America/New_York

Current phase: **Phase 6 remediation rerun 02 preserved after a checksum-contract
failure; the canonical fix and non-overwriting rerun 03 are prepared but not run**

Current objective: verify and hand off the single-source tokenized-inventory checksum
fix, preserve all three completed engineering attempts unchanged, and leave rerun 03
ready for an owner-operated run. No claim is made that the corrected rerun will pass
the frozen scientific gate.

Checkpoint reason: rerun 02 completed the first nine pipeline stages, both 2,000-step
v0.3 candidate trainings, and the 531-example development evaluation before stage 10
failed. The gate reconstructed tokenized inventories as dictionaries while training
hashed the same inventory as tuples. Only those two bindings differed. Delegating the
gate to the training contract exactly reproduces both recorded hashes. A separate
read-only reconstruction shows that the preserved model passes seven of ten v0.3
criteria and legitimately misses fault margin, continuation F1, and calibration.

Project path:
/Users/zachary/Documents/Personal-Projects/AI-transformer

Exact account usage is not observable in this environment. No account-level
percentage was measured.

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
- Added `phase6-remediation-v0.4.0-local-rerun-01` as the first non-overwriting retry
  identity. It is now also preserved as failed engineering evidence; neither earlier
  run was deleted or reused.
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
- Preserved rerun 01 at
  `runs/phase6-remediation-v0.4.0-local-rerun-01/`. It completed the first seven
  pipeline stages, including the v0.2 gate and v0.3 smoke stage, then failed safely
  during v0.3 candidate training at step 200 and work item 32 of 48. Final evaluation
  was not accessed.
- Reproduced the failure twice from a private copy of the preserved step-100 state.
  Development example `rbexample:c64d2ab90f7c0a7437993c08` reached the prefix
  `RB2|counterfactual_compare|0~9,A~-~6~-|0~9,A~-~6~-`: both conclusions were
  individually valid but identical, so the required changed-fields suffix had no
  legal completion.
- Added a relation-aware, truth-independent counterfactual conclusion constraint. It
  uses only the already generated baseline conclusion, not target truth, and retains
  at least one reachable difference while decoding the second conclusion.
- Replayed steps 101–200 from the copied state on Apple MPS after the fix. The exact
  formerly failing example completed schema-valid with EOS in 58 generated tokens.
  The durable disposable replay reached step 200 with validation NLL
  1.3842025559550057, training NLL 0.4324597716331482, and no MPS fallback. The
  verification harness then made an incorrect diagnostic-only attribute lookup after
  those assertions and exited nonzero; project code and the successful decode were
  unaffected.
- Added a bounded, checksum-bound local `failure-diagnostic.json` for future internal
  stage callback failures. It records only safe exception class names and package-
  relative code sites; exception messages, raw traceback text, and absolute paths are
  excluded. Diagnostic publication can never mask or delay durable failure state.
- Added `phase6-remediation-v0.4.0-local-rerun-02` as the next non-overwriting run
  identity. It is now also preserved as failed engineering evidence; none of the
  three existing run roots was deleted, reused, resumed, or edited.
- Rerun 02 completed stages 1 through 9, both 2,000-step v0.3 candidate trainings,
  and the 531-example development evaluation. Stage 10 failed safely because its
  duplicated reconstruction helper encoded tokenized inventories as dictionaries
  while training encoded the same canonical values as tuples.
- Made `tokenized_inventory_sha256` the one public, validated, tuple-based contract
  used by both the training producer and pipeline reconstruction gate. Unit tests
  freeze its canonical digest and prove the two call sites cannot drift silently.
- Verified read-only that the corrected contract exactly reproduces rerun 02's two
  recorded tokenized hashes. Every other candidate-ranking binding already passed.
- Reconstructed the preserved v0.3 acceptance evidence without using held-out or
  final data. Seven of ten criteria pass; fault margin, continuation macro F1, and
  expected calibration error miss their frozen thresholds.
- Added `phase6-remediation-v0.4.0-local-rerun-03` as the new default identity in the
  CLI, GUI monitor, packaged resources, tests, README, and runbook. It has not been
  created or started.
- Refreshed the public README, owner runbook, decision log, and this handoff record to
  distinguish the engineering fix from the model's still-unmet scientific criteria.
- Added a distinct, visually prominent `ENTIRE RERUN` bar for all 16 stages to the
  local macOS testing GUI. A second bar now reports progress within Setup, v0.2, v0.3,
  v0.4, or Finalization, while the third retains exact current-task work. The entire-
  run value is explicitly documented as stage-based, not a wall-clock estimate.

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
- Rerun 01 configuration checksum:
  96850668232781faa9d14319ce40e136aa1ada0c85317ed88b08ef795fcd6a13
- Rerun 02 canonical configuration checksum:
  6515bd0f2ae78ef566b4e322630a8759f49068f4577a98c83cdaf6acdf308710
- Rerun 02 configuration file SHA-256:
  edf6effe587eb25e5f0f3bd8103368b26a066158d749a8740ec9ba1f7ed1e538
- Preserved v0.2 training result checksum:
  bc0332e69dd01aa9d2b48f5ca5c130c2e3c944a9435e5863399c0340bad68cac
- Preserved v0.2 checkpoint-manifest checksum:
  b27547e10fc0dfd08ea08337368dd3011c1c3bcb4e98747259ff49486ef9a44e
- Preserved v0.2 selected validation NLL: 0.1484147810350759 at the completed
  1,500-step boundary.

The first failed v0.2 checkpoint and rerun 01's partial v0.3 checkpoints are
engineering evidence, not accepted model-quality results. Rerun 01's step-100
validation NLL was 1.6342359354; its disposable corrected step-200 replay measured
1.3842025559550057.

Rerun 02 is completed development evidence but not an accepted model result:

- Source commit: `cf732307d1d1f756772af7a87214ffde8e9bf8b0`
- Terminal stage: 10 of 16, `v03_gate`, after 10,772.9 active seconds
- Selected candidate/checkpoint: `v02-uniform-control`, step 1,900
- Selected validation NLL: 0.3456375
- Selected semantic composite: 0.9513827957041954
- Full IID constrained exact match: 0.749529
- Full IID constrained parse/schema validity: 1.0 / 1.0
- Full IID unconstrained parse/schema validity: 0.873823 / 0.762712
- Reconstructed v0.3 acceptance: seven of ten criteria pass
  - fault comparator margin: 0.00975186975; required at least 0.02
  - continuation macro F1: 0.718230958; required at least 0.90
  - expected calibration error: 0.185504519; required at most 0.15
- Recorded tokenized-inventory hashes now reproduced exactly by both call sites:
  - train: `0de3e814e74f3960b17918e0086b9b3c9870c7416d1adaf04c7a37985930f560`
  - validation: `bc4583c06412784dd6726fe33dea8c0ebc9b46453848a40cb65164a32c48290c`
- Preserved run-manifest checksum:
  `4ed02f09278cf985bd17c152359ee80dae3bcf154cabaeb243efb245447655c0`
- Preserved pipeline-state checksum:
  `fab17fd1b31c2e230be439ceec0d86a6846063817410f89ecc913f1b97fc05d3`
- Preserved status checksum:
  `14c800385882d3655af606e5f24614d2aa9fd3c60a9e2aece70e25122b1e8916`
- Preserved progress checksum:
  `4af616a1d43f3acbdc84e28a4e29df860560d8bd5e753fd0229aaeb4957324ff`

An unchanged deterministic rerun 03 is expected to reach a legitimate v0.3
scientific block after the checksum fix unless training variation changes the three
missed metrics. Repeating it verifies the corrected contract; it does not guarantee
scientific acceptance.

## Tests and checks run

- Final permitted repository gate after the counterfactual fix and rerun-02 update:
  - Ruff format: 190 files passed.
  - Ruff lint: passed.
  - Strict mypy: 155 source files passed.
  - Pytest: 1,095 passed and 2 deliberately deselected in 565.60 seconds.
  - Branch coverage: 85.73%, above the required 85%.
- The two deselections are the documented historical final/golden asset readers:
  test_resource_api_reads_the_root_reviewed_assets_without_drift and
  test_approved_golden_packet_projects_sixty_examples. No held-out or final evaluation
  was run.
- Focused decoder, diagnostic, packaging, CLI, and GUI checks: 135 passed and 1
  protected resource-reader test was deselected in 15.20 seconds.
- Native Swift/AppKit type checking: passed. Property-list lint: passed. The GUI
  launcher and closed controller bridge both pass Bash syntax.
- Three-level GUI-focused verification: Ruff formatting/lint passed, strict mypy
  passed, and 30 controller tests passed in 1.10 seconds. Native Swift type checking,
  property-list lint, and both GUI-wrapper Bash syntax checks passed. The exact local
  window visually displayed all three bars in the verified `Not started` state; only
  Close was pressed, and no lifecycle operation ran.
- Post-commit dry run passed at source commit
  `3e70032d1767b4bee1a0e357cbbaca3b07b96eb3`: config checksum
  `6515bd0f2ae78ef566b4e322630a8759f49068f4577a98c83cdaf6acdf308710`,
  16 frozen stages, and no run creation.
- Real non-window launcher smoke check passed and reported `Not started`; its strict
  JSON snapshot reported rerun 02, stage total 16, source commit `3e70032d...`, Start
  enabled, and Stop/Resume/Finder disabled. The rerun-02 directory remained absent.
- Exact MPS regression replay from a private copy of rerun 01's step-100 state passed
  the formerly failing step-200 decode: constrained schema true, EOS true, 58 tokens,
  and no fallback. The three disposable replay directories were then removed; no
  preserved run artifact changed.
- Source and wheel distributions rebuilt successfully. The first isolated build was
  blocked by sandboxed dependency resolution; after explicit approval, the pinned
  Hatchling dependency was installed only in the temporary build environment. The
  first isolated-install verification exposed a whitespace-dependent assertion in
  the verifier itself; it now parses the canonical JSON and the verification passes.
  Archive listings confirm both rerun configurations, the corrected runbook, AppKit
  monitor source/metadata, controller, and all fixed wrapper scripts are present.
- Exact native launcher visual check: passed. The temporary app closed cleanly, and
  its temporary bundle was removed. Start, readiness, stop, resume, and Finder were not
  invoked.
- Rerun 01 preserved checksums:
  - run manifest:
    `c1ee26805021be3218de26493faebb9e59f7c13692d959ffcaacdd333aa6a959`
  - pipeline state:
    `5219d0a96f99ed0bcb0d5c4408b0ab32757a25d3b7c7a68ad0865bdc2d234e61`
  - status:
    `007d53385902f80d5f8641f48b6b7a03365f61d904055ba60016a55de3a8abae`
  - progress:
    `3b8bea26d7dfea07c074fc8cc75031e20bbb431b8c1489fa876578e586ccb83d`
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
- D-087 preserves rerun 01, requires a truth-independent reachable difference between
  counterfactual conclusions, adds bounded safe local diagnostics, and assigns the
  corrected attempt the new non-overwriting rerun-02 identity.
- D-088 preserves rerun 02, makes the training checksum contract canonical, assigns
  rerun 03, and requires honest reporting of the reconstructed seven-of-ten gate.
- The project owner controls GitHub pushes, external publication, credentials, and
  deployment. Local checkpoint commits are permitted.

## Residual risks, known failures, and blockers

- The first long run failed after approximately 3,627.6 active seconds at the v0.2
  development gate. It is terminal and must not be resumed or edited.
- Rerun 01 failed after approximately 4,802.7 active seconds during v0.3 candidate
  training at step 200 and work item 32 of 48. It is terminal and must not be resumed
  or edited. Its preserved engineering evidence proves the device fix and identifies
  the now-corrected counterfactual dead end.
- Rerun 02 failed safely after approximately 10,772.9 active seconds at `v03_gate`.
  It is terminal and must not be resumed or edited. The false checksum mismatch is
  corrected, but its preserved model independently misses three frozen thresholds.
- Rerun 03 is pending. A new run may take hours on Apple MPS. Because the inputs and
  training contract remain deterministic, it is expected to reach a legitimate v0.3
  scientific block if its results reproduce rerun 02; it may also expose a different
  engineering failure. The checksum fix is not a promise of model acceptance.
- The monitor is a macOS owner utility, not a cross-platform product surface. It
  compiles a temporary AppKit bundle on each open, so the window can take roughly 20
  to 30 seconds to appear on this Mac.
- The final rerun-03 GUI check must not press readiness, Start, stop, Resume, or
  Finder. Lifecycle correctness is covered by unit/contract tests; a real Start
  remains an owner action.
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
  configs/experiments/phase6-remediation-pipeline-v0.4.0-rerun-03.toml
- Preserved rerun-02 configuration:
  configs/experiments/phase6-remediation-pipeline-v0.4.0-rerun-02.toml
- Preserved rerun-01 configuration:
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
- Preserved failed rerun 01:
  runs/phase6-remediation-v0.4.0-local-rerun-01/
- Preserved failed rerun 02:
  runs/phase6-remediation-v0.4.0-local-rerun-02/
- New rerun-03 root (currently absent):
  runs/phase6-remediation-v0.4.0-local-rerun-03/
- Canonical live status after start:
  runs/phase6-remediation-v0.4.0-local-rerun-03/status.json
- Append-only progress after start:
  runs/phase6-remediation-v0.4.0-local-rerun-03/progress.jsonl

## Repository state

- Branch: main.
- Failed-run source/handoff commit:
  2aafcd1661ec7c3640a385621db171041532e547
- Complete device-fix/rerun implementation commit:
  f6f2369a050c9bf50d6c04351da603969a1f1273
- Local monitor implementation commit:
  295f8e894f16e9d49f84e8dbe29e36697824a340
- Rerun-01 source commit:
  034b41cca07b999f701850986a67a692b40d8c30
- Counterfactual decoder fix, safe diagnostic, and rerun-02 implementation commit:
  3e70032d1767b4bee1a0e357cbbaca3b07b96eb3
- Rerun-02 source commit:
  cf732307d1d1f756772af7a87214ffde8e9bf8b0
- Canonical checksum/rerun-03 implementation commit: pending local checkpoint.
- Status handoff commit: pending local checkpoint.
- Uncommitted work: checksum fix, rerun-03 identity, tests, README, runbook, decision
  log, packaging inventory, and this status update.
- All three failed run directories are preserved; the rerun-03 directory does not
  exist.
- No fresh-final ledger/result was created or accessed.
- No push or deployment was performed during this work.

## Immediate next step

Complete the permitted repository and package gates, make local checkpoint commits,
and confirm the tree is clean. The owner may then push. Tomorrow, open the local
monitor, confirm it reports `Not started` and
`phase6-remediation-v0.4.0-local-rerun-03`, optionally run the non-mutating readiness
check, and press `Start new rerun` only if a clean engineering replay is worth the
compute despite the preserved three-metric scientific shortfall. Do not resume or
modify any preserved failed run.

## Exact recommended next command after final handoff

    cd /Users/zachary/Documents/Personal-Projects/AI-transformer
    git status
    git push origin main
    ./scripts/open_phase6_progress_gui.sh

Pushing does not start training. Opening the monitor does not start training. The
readiness, Start, cooperative stop, and stopped-run Resume controls are inside the
window. The locked evaluation wrapper is not a research command in this release.

## Resume prompt

Resume ReactorBench-LM from the safe checkpoint in
/Users/zachary/Documents/Personal-Projects/AI-transformer.
Read research/PROJECT_REQUIREMENTS.md and docs/IMPLEMENTATION_STATUS.md first.
Inspect Git status and verify the recorded tests before making changes.
Continue from the documented immediate next step without repeating completed work.
