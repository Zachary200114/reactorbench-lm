# Phase 6 remediation local runbook

Status: **targeted-03 training/evaluation preserved; no-training gate replay certified
9 of 10 checks and a scientific block**
Scope: v0.2 output reliability, v0.3 semantic learning, and v0.4 development-only
generalization gates

This runbook operates the local, checksum-bound remediation pipeline. It does not push
to GitHub, deploy a service, call a hosted model, or automatically open the frozen
final evaluation. A completed development pipeline is engineering evidence, not proof
that the model passed its scientific gates.

## Safety boundary

The ordinary pipeline may use only training, IID validation, and designated
development shadow views. Final held-out data and golden answers are isolated from all
training, checkpoint selection, threshold selection, and routine status commands.

The historical G01–G15 packet is prohibited for the v0.4 final gate. A future final
evaluation requires all of the following before its one permitted access:

- a newly generated and frozen v0.4 final manifest;
- a fresh golden extension that was never used for development;
- explicit project-owner review and approval of that extension;
- the checksum-bound final-ready, owner-review, and fresh-extension markers;
- a development pipeline whose frozen gates completed; and
- the explicit `--confirm-final-evaluation` flag.

Those prerequisites are necessary but are not sufficient in this release. The
separate fresh final-evaluation executor is intentionally unimplemented and locked;
the command is only a fail-closed guard and returns exit code `10`. Files whose names
look like readiness, owner-review, extension, or access records cannot authorize
access or substitute for the missing independently reviewed executor. Do not create
such files by hand or use the command to probe whether a historical packet works.

## Preserved failed attempts and current rerun

The first owner-operated run remains preserved at:

```text
runs/phase6-remediation-v0.4.0-local/
```

It completed preflight, v0.2 inventory/caps, smoke, and all 1,500 v0.2 development
training steps. It then failed safely at `0/252` before the first behavioral decode
because the checkpoint loader requested generic device `mps`, PyTorch placed the model
on the equivalent concrete device `mps:0`, and the decoder compared those spellings
strictly. This is failed engineering evidence, not a model-quality gate result. Do not
delete, rename, resume, or edit it.

The first non-overwriting rerun remains preserved at:

```text
runs/phase6-remediation-v0.4.0-local-rerun-01/
```

It proved that the v0.2 device correction works, completed the first seven stages,
and entered v0.3 candidate training. At step 200, development-only checkpoint
selection failed on example `rbexample:c64d2ab90f7c0a7437993c08`. The
truth-independent grammar allowed the second counterfactual conclusion to become
identical to the first even though the target contract requires them to differ; no
valid changed-fields suffix could then exist. A read-only-source replay from the
preserved step-100 state reproduced the exact dead end. The corrected constraint now
requires a still-reachable difference while constructing the second conclusion, and
the exact step-200 example completes with schema-valid output and EOS. Do not delete,
rename, resume, or edit this failed run.

The second non-overwriting rerun remains preserved at:

```text
runs/phase6-remediation-v0.4.0-local-rerun-02/
```

It completed the first nine stages, both 2,000-step v0.3 candidates, and all 531
selected-candidate full-IID development evaluations in 2h 59m 32s. The v0.3 gate then
failed before acceptance because training hashed each tokenized inventory as tuples,
while the gate independently hashed the same fields as dictionaries. Field-by-field
reconstruction passed every other candidate-ranking binding; only the training and
selection tokenized-inventory hashes differed. This was a duplicate canonicalization
bug, not artifact corruption. Training and every gate consumer now delegate to the one
training-contract checksum function. Read-only reconstruction exactly matches rerun
02's recorded train and validation hashes. Do not delete, rename, resume, or edit this
failed run.

The completed development evidence also shows that fixing the checksum does not make
the model pass. The reconstructed v0.3 gate passes seven of ten checks and misses:

- fault comparator margin: 0.0097519, required at least 0.02;
- continuation macro-F1: 0.718231, required at least 0.90; and
- expected calibration error: 0.185505, required at most 0.15.

Constrained parse and schema validity are both 1.0, constrained exact match is
0.749529, and the other five supported semantic checks pass. These are development
measurements, not a final result. An unchanged rerun is expected to reach a valid
scientific `blocked` state at v0.3 rather than advance to v0.4.

The first targeted scientific attempt remains preserved at:

```text
runs/phase6-remediation-v0.4.0-targeted-01/
```

It completed the first nine stages, two 2,000-step candidates, calibration, and all
427 IID gate predictions in 2h 32m 20s. The v0.3 gate failed before acceptance because
canonical example-ID-ordered predictions were paired with calibration examples in
selection order. The run is not corrupt: an identity-bound read-only reconstruction
reproduces all 56 observations, temperature 2.35, and the saved calibration checksum
bit-for-bit. Its calibrated development evidence would pass eight of ten checks:
expected calibration error is 0.087461, continuation macro-F1 is 0.739619, and fault
comparator margin is -0.045440. The latter two miss their unchanged 0.90 and +0.02
thresholds. Do not delete, rename, resume, or edit this failed run.

The focused targeted-02 attempt remains preserved at:

```text
runs/phase6-remediation-v0.4.0-targeted-02/
```

It completed ten stages in 1h 12m 30s and stopped correctly at the v0.3 scientific
gate. It passed constrained parse, constrained schema validity, continuation macro-F1,
and required abstention, but failed the other six unchanged checks. Continuation
macro-F1 improved to 0.9418; next-action macro-F1 fell to 0.2421, evidence F1 was
0.6778, and the no-fault false-positive rate was 0.5556. The focused sampler starved
the other task families, and its 48-row selector chose step 1,200 even though validation
NLL continued improving through step 2,000. This is a valid negative scientific result,
not an implementation failure. Do not delete, rename, resume, or edit this blocked run.
See `docs/model/PHASE6_TARGETED02_DIAGNOSIS.md` for the exact ten-check table.

The current corrected default configuration is:

```text
configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-03.toml
runs/phase6-remediation-v0.4.0-targeted-03/
```

Targeted-03 completed its 2,500 training steps and 427 development-gate evaluations.
Its original stage-10 attempt failed because the independent gate consumer applied the
historical score to hierarchical checkpoint evidence. The source run remains
unchanged. The policy-aware correction was certified through the separate one-time
identity below, with no training or final access:

```bash
./scripts/replay_phase6_targeted03_gate.sh
```

The completed replay is under
`runs/phase6-remediation-v0.4.0-targeted-03-gate-replay-01/`. It exits `9` because the
unchanged scientific gate passes nine of ten checks and misses only fault-comparator
margin (`-0.0344200`, required `>= 0.02`). Do not delete that identity to rerun it.
See `docs/model/PHASE6_TARGETED03_GATE_REPLAY.md` for exact metrics and checksums.

Checkpoint consumers now return the model's actual parameter device after verifying
that its device type and any explicitly requested index match. A real CPU/MPS or
explicit-index mismatch still fails closed. The exact preserved checkpoint operation
that failed was repeated read-only after the fix and decoded one validation example
on `mps:0` without changing the checkpoint.

The ordinary wrappers below still select the now-preserved targeted-03 identity.
Targeted-01, targeted-02, and targeted-03 remain historical evidence and are not valid
restart targets. The
original run is bound to source
commit `2aafcd1661ec7c3640a385621db171041532e547`, rerun 01 is bound to
`034b41cca07b999f701850986a67a692b40d8c30`, and rerun 02 is bound to
`cf732307d1d1f756772af7a87214ffde8e9bf8b0`. Their live status commands therefore
fail closed from the corrected checkout. Inspect immutable terminal summaries when
present; never edit a preserved run to make it match the current source.

```text
runs/phase6-remediation-v0.4.0-local/terminal-reviews/state-b5d0053842367b2175837e6e647cce3b359beda90648eb12b254091ab427013a/TERMINAL_REVIEW.md
runs/phase6-remediation-v0.4.0-local/terminal-reviews/state-b5d0053842367b2175837e6e647cce3b359beda90648eb12b254091ab427013a/terminal-review-bundle.json
runs/phase6-remediation-v0.4.0-local-rerun-01/terminal-reviews/state-5c8a51da7e9642d7b9900b32975a019bede81872ac86f3e413ad71e5d1e9a411/TERMINAL_REVIEW.md
runs/phase6-remediation-v0.4.0-local-rerun-01/terminal-reviews/state-5c8a51da7e9642d7b9900b32975a019bede81872ac86f3e413ad71e5d1e9a411/terminal-review-bundle.json
```

## One-time setup

Run these commands from Terminal:

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
uv sync --frozen --all-groups
./scripts/run_phase6_pipeline.sh --dry-run
```

The wrappers resolve the repository and invoke only its `.venv/bin/python`. Preflight
checks reject a missing or unsafe virtual environment, changed source commit,
incompatible configuration or contract, incorrect tokenizer or artifact checksum,
unsafe existing output, and unavailable pipeline component.

Do not start the long run until the dry run exits with code `0` and reports the frozen
configuration checksum, source commit, and stage count. The primary run requests Apple
MPS. Earlier development work may use the documented CPU fallback, but a fallback does
not satisfy the mandatory v0.4 MPS feasibility gate. Every pilot measurement records
its requested and resolved device; if any mandatory measurement resolves to CPU, the
pilot is a non-passing result and the 1,024-token candidate does not train.

Finish and review the intended source commit before starting. From the start of the
run through the later Codex evidence review, do not create a new commit, check out
another commit or branch, or pull changes. The run remains bound to its original
commit. Pushing that exact already-bound commit is safe because it does not change the
local checkout; pushing remains an owner action, not a pipeline action.

## Start the development pipeline

The exact primary command is:

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
./scripts/run_phase6_pipeline.sh
```

Keep that Terminal window open. On a Mac laptop, connect power and prevent idle sleep
if necessary; an optional foreground form is:

```bash
caffeinate -i ./scripts/run_phase6_pipeline.sh
```

The runner retains the frozen 16-stage interface. For targeted-03, preflight and the
four v0.2 stages verify the preserved rerun-02 manifest, completion prefix, outcomes,
and all referenced artifacts by canonical contract, size, SHA-256, source, config,
path containment, and non-symlink checks. The targeted configuration externally pins
the exact canonical path and SHA-256 of all 21 permitted files plus their aggregate
inventory hash, so a coherently re-signed addition or substitution is rejected. The
stages write only small new reuse reports; they do not regenerate v0.2 data, train
v0.2, or decode its gate. The runner then
advances to v0.3 data audit, candidate training, semantic evaluation, and acceptance;
when v0.3 passes, it runs the permitted v0.4 pilot, candidate, shadow evaluation,
policy freeze, and review-bundle stages. A failed scientific gate stops later work
cleanly instead of weakening a threshold.

Hierarchical v0.3 trains exactly one candidate from random initialization. Each
six-row batch contains one row from every task. Continuation and next-action labels
rotate uniformly. Fault draws preserve a 50% unresolved, 10% no-fault, and 40%
diagnosed hierarchy, with diagnosed families rotating uniformly. This restores the
task exposure lost in targeted-02 without flattening the heterogeneous fault labels.
Model size, teacher-forced exposure, and every acceptance threshold are unchanged;
training increases from 2,000 to 2,500 steps. Checkpoint selection uses the same
target-independent 48-example subset as an eligibility check: semantic composite must
reach 0.75, then lower validation NLL and earlier step choose the checkpoint. A separate
target-independent 56-example validation subset is decoded only after selection to
fit a scalar temperature on the fixed 0.50–5.00 grid. The remaining 427 IID rows form
the gate. Calibration changes only confidence used for ECE and selective risk; raw
tokens, canonical outputs, prediction checksums, semantic metrics, and candidate
selection remain unchanged. Confidence endpoints are clamped before NLL evaluation.
The exact 56 calibration predictions and 427 raw gate predictions are immutable
JSONL artifacts. The gate reopens those artifacts and independently recomputes the
temperature, raw report, calibrated report, and acceptance result before binding them.
Targeted cross-attempt recovery writes its checksum-bound sampling sidecar before
publishing a copied model state, so an interruption cannot leave an unbindable state.

The v0.3 audit regenerates IID material once at the frozen source commit. It first
reproduces the reviewed counterfactual-cap evidence against the raw 5,859-example
inventory, then deterministically removes 24 same-task exact prompt duplicates whose
targets are identical. The 5,835-example development inventory must contain all 55
counterfactual rows unchanged. The stage records the raw/deduplicated manifest bridge,
removal inventory, task-scoped visible-structure separation, target-leak scan, and
report-only task/class inventories. A conflicting duplicate, cap drift, missing
counterfactual, or cross-view structured collision blocks advancement.

The longer-context pilot activates only when both the frozen v0.2 rate and the newly
measured deduplicated v0.3 IID-train prompt-truncation rate meet the frozen 0.10
materiality threshold. If v0.3 is below that threshold, the pilot records a passing
no-op and the 512-token control is reused; that is not a 1,024-token feasibility claim.
When activated, the mandatory v0.4 context pilot must actually resolve to Apple MPS
while checking batch sizes **1, 2, and 4**; batch 4 must also pass finite-loss and
checkpoint-reload checks. The pilot profiles the full 1,024-token IID train and
validation inventories, selects the longest sequence for each task, and proves each
ten-step batch schedule actually samples the global training maximum. CPU fallback is
recorded as a negative feasibility result, not an MPS pass.
Before longer-context main training, the consumer independently reopens all three
pilot training-result contracts and verifies the exact batch-specific training
configuration, model configuration, tokenizer, tokenized train/validation inventories,
example counts, device resolution, checkpoint hash, and report checksum. It also
regenerates the frozen v0.3 IID material, retokenizes the complete train and validation
views at 1,024 tokens, reselects the longest row per task, recomputes every length
profile and pilot inventory hash, and proves each batch schedule exercises the global
maximum. A self-consistent summary that substitutes shorter rows cannot authorize the
candidate. Before either the active or no-op branch, it also retokenizes the exact
deduplicated IID-train view at 512 tokens and rederives the measured truncation rate
and activation decision; a self-bound false activation or false no-op fails closed.
The final policy consumer independently repeats this proof for an activated main
candidate: it regenerates the exact deduplicated IID material, resolves the frozen
48-row selection, retokenizes both inputs at 1,024 tokens, and cross-binds the training
configuration, tokenizer, source, selected step/loss, and checkpoint weights and size.
Both checkpoints are selected using only the frozen IID development selection set;
the six shadow views are not read during fitting or checkpoint selection. The
512-token control and eligible 1024-token candidate are then compared on full IID
validation and all six development shadow views. Final candidate selection follows
the preregistered order: all required gates must pass, then the highest worst-view
semantic composite wins, then the highest full-IID composite, with the shorter
512-token context as the deterministic final tie-break. No result may change that rule
after the run begins. The ranking score is reconstructed from each immutable semantic
report: schema validity is a hard prerequisite, followed by equal weighting of exact
match, every supported classification/evidence/abstention/specificity metric,
calibration quality, and selective-risk quality. The v0.3 gate reopens candidate
training results, checkpoint manifests, and selection reports to verify the composite,
training selection score, validation-NLL, step, and checkpoint tie-break fields. It
independently rebuilds the
deterministic 48-row selection, then checks every candidate's raw/tokenized validation
inventory and exact scoped evaluation manifest against those rows. The v0.4 policy
gate likewise reloads exact IID and shadow datasets; each reopened report must cover
the full named view and match its scoped dataset, source commit, configuration,
tokenizer, compact contract, checkpoint, predictions, and comparator bindings before
IID/worst-view composites are rederived and selection repeats. It separately
reopens the selected v0.3 control and the optional v0.4 candidate-training evidence,
loads their bound checkpoints/configurations, and requires exactly one 512-token
candidate when inactive or exactly that control plus the 1,024-token variant when
active; relabelled or context-rebound index rows fail closed.

The run is non-overwriting. If the run directory already exists, the start command
refuses and tells you to resume it. It also remains bound to the Git commit at which it
started; do not change commits, check out or pull other code, or edit tracked
scientific inputs before the Codex evidence review is complete.

## Progress and periodic updates

The runner records a heartbeat every **30 seconds**, plus stage, bounded decode-
progress, evaluation, checkpoint, stop, failure, and completion events. Stop/resource
polling is finer grained than durable progress: it occurs inside each BOW optimizer
step, GRU batch, comparator boundary, and decoded example, while decode progress is
persisted only at a bounded interval plus completion to avoid excessive disk syncs.
Before singleton decoding begins, the complete view inventory must also provide
non-empty, globally unique example IDs; a duplicate fails before any model call.
In another Terminal window, inspect the newest verified snapshot at any time:

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
./scripts/check_phase6_status.sh
```

For a continuously refreshed view, use:

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
while true; do
  ./scripts/check_phase6_status.sh
  sleep 30
done
```

Press Control-C only in the monitoring window to stop that display. The status command
does not train, evaluate, mutate checkpoints, or open final data.

The status command strictly validates canonical `status.json` and requires it to equal
the final complete event in `progress.jsonl`; it refuses mismatched, truncated, or
unsafe progress evidence. The verified display reports the overall state, current or
next stage, interruption count, latest event and message, stage position, completed
work units, latest metric, estimated time remaining, and latest durable checkpoint
when those fields are available. The machine-readable heartbeat and append-only event
log are:

```text
runs/phase6-remediation-v0.4.0-targeted-03/status.json
runs/phase6-remediation-v0.4.0-targeted-03/progress.jsonl
```

### Optional local progress window

The owner may use the small local-only macOS monitor instead of keeping several
Terminal windows open:

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
./scripts/open_phase6_progress_gui.sh
```

Opening the monitor does not start training. It identifies the fixed, preserved
`phase6-remediation-v0.4.0-targeted-03` run, refreshes through the existing
strictly validated status command, and exposes only controls allowed by verified run
state. Start and Resume remain disabled for this terminal failed source run. The
activity log
is bounded and non-scientific; the monitor never reads `status.json` or
`progress.jsonl` directly and never changes their artifact contracts.

The window separates progress into three explicit levels:

1. **ENTIRE RERUN** — stage-based completion across setup, every v0.2–v0.4 stage,
   and final review (all 16 pipeline stages);
2. **Current version** — completion within Setup, v0.2, v0.3, v0.4, or Finalization;
3. **Current work** — exact units reported by the active task when available.

The entire-rerun percentage is a completion indicator, not a wall-clock prediction:
the 16 stages do not have equal runtimes. Use the separately displayed estimated
remaining time only when the validated status reporter supplies it.

Keep the window open or minimized if an audible failure notification is required.
When strictly validated status first enters `Failed` or `Blocked`, the monitor requests
critical macOS attention and loops the system `Sosumi` sound at application volume 1.0
for 45 seconds. If the sound file is unavailable, it emits 23 system beeps over about
45 seconds. The alert is issued only once per monitor session and is not repeated by
later refreshes. While it is active, **Stop alarm** stops the loaded sound and cancels
pending fallback beeps without changing the pipeline or its alert guard. It cannot
override muted or low system output, an incorrect output
device, disconnected headphones or speakers, or a sleeping/powered-off computer; test
those macOS settings before leaving a run unattended. The pipeline itself does not
depend on notification delivery.

The Homebrew project Python currently lacks `_tkinter`, and the importable macOS
system Tk 8.5 was visually probed and rendered even a minimal window blank on this
Mac. The launcher therefore uses the already-installed `/usr/bin/swiftc` AppKit toolchain
to build a native app in a private temporary directory and removes that directory when
the window closes. The native app invokes only the fixed
`scripts/phase6_monitor_controller.sh` bridge, which keeps its shell process as the
AppKit child and runs the standard-library controller with the project
`.venv/bin/python`. A visual host test showed that launching framework Python directly
from AppKit can route through `Python.app` and stall. Every project operation therefore
remains inside the project `.venv` and frozen wrappers. No dependency or permanent
application is installed. Building the temporary native app can make the first window
take roughly 20 to 30 seconds to appear on this Mac.

Start and Resume launch under `caffeinate` in a detached process session. Closing the
window does not signal or terminate an active pipeline and displays a warning that
the run will continue. `Request safe stop` remains cooperative rather than an
immediate cancel. The monitor has no delete, overwrite, or automatic start-over
operation: after this run exists, a scientifically justified fresh attempt requires
a separately reviewed, versioned run-name/configuration change so existing evidence
remains preserved.

A non-window smoke check is available for maintainers and does not create or start a
run:

```bash
./scripts/open_phase6_progress_gui.sh --smoke
```

## Stop safely

From a second Terminal window, request a cooperative stop:

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
./scripts/stop_phase6_pipeline.sh
```

The request is idempotent. The active stage stops at its next safe boundary. An
ordinary training-loop stop writes its next applicable durable checkpoint. If the
request arrives inside a checkpoint-selection decode, that incomplete evaluation
aborts without publishing a partial artifact; resume uses the newest earlier verified
training state, or restarts that candidate when no durable step exists yet. At most the
work since that recorded state is repeated. Baseline/evaluation work polls
cooperatively within its inner loops and commits no partial scientific view artifact.
Continue checking status until it says `stopped`. A stop request is not an immediate
process kill, so allow the current atomic operation to finish.

Stop-request archival is crash-durable across its source and archive directories. If
a prior crash left both an identical source request and its checksum-named archive,
resume verifies strict canonical equality and retires the duplicate source request;
a conflicting existing archive fails closed.

Use the stop command instead of deleting files or terminating the process. If an
unavoidable Control-C or system interruption occurs, inspect status before resuming;
the orchestrator recovers only checksum-valid completion markers and starts a new,
non-overwriting attempt when required.

## Resume

After status reports `stopped`, or after diagnosing a safely recorded interruption,
run:

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
./scripts/resume_phase6_pipeline.sh
```

To prevent idle sleep while the resumed process remains in the foreground, use:

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
caffeinate -i ./scripts/resume_phase6_pipeline.sh
```

Resume verifies the original source/configuration binding, audits committed stage
boundaries, archives the previous stop request, and continues from the last valid
stage or durable training state. It does not silently skip a failed integrity check.

Across repeated attempts, resume selects the highest verified durable step, copies it
atomically into the new attempt, verifies the successor, records its checkpoint, and
only then retires superseded committed states. Recognized
`.state-step-########.lock` and `.state-step-########.tmp-*` crash remnants are
preserved for forensic review; unknown, symbolic-link, or otherwise unsafe entries
fail closed. Repeated hard kills can accumulate those recognized remnants and
abandoned final-checkpoint directories. The actual run size includes them, and a
projected write that could meet or exceed the 8 GiB run limit is refused before model
work begins. They are not auto-deleted and may eventually require reviewed manual
intervention.

Do not use resume to override `blocked`. A blocked state means a frozen scientific gate
did not pass and later stages were intentionally not run; preserve it for analysis.

## Runtime and storage expectations

These are planning estimates, not measured v0.3/v0.4 results:

- **Apple MPS development pipeline estimate:** approximately 6–24 hours.
- **CPU fallback:** permitted for earlier development stages, but it cannot satisfy an
  activated v0.4 MPS pilot. No successful activated CPU runtime estimate is claimed.
  If measured v0.3 materiality makes the pilot a no-op, a control-only completion may
  still be valid but provides no 1,024-token feasibility evidence.
- **Storage estimate:** approximately 2–6 GiB, with a hard configured run-directory
  limit of 8 GiB.
- **Process-memory boundary:** 16 GiB RSS as configured for this pipeline.
- **Overall active development-run time boundary:** 72 hours. Time spent in a durable
  stopped state does not consume this active-runtime budget.

Actual time depends on the Mac, MPS availability, selected candidate, decoding length,
and whether optional v0.4 work is activated by measured development evidence. The run
records duration, throughput, checkpoint size, device/runtime details, and failures;
replace these estimates with those measurements when reporting results.

No final-evaluation runtime estimate is offered because that separate executor is not
implemented in this release.

## Outputs and interpretation

All ordinary pipeline outputs stay under:

```text
runs/phase6-remediation-v0.4.0-targeted-03/
```

Important top-level evidence includes:

- `run-manifest.json` — command, Git commit, runtime, and frozen config bindings;
- `pipeline-state.json` — checksum-bound stage state and completion prefix;
- `status.json` — latest bounded human/machine-readable progress snapshot;
- `progress.jsonl` — ordered progress, heartbeat, and checkpoint evidence;
- `stages/` — immutable numbered stage attempts, outcomes, reports, and completion
  markers; and
- `failure-diagnostic.json` inside a failed attempt, when local diagnostic publication
  succeeds — checksum-bound exception classes and project-relative code sites with no
  exception message, traceback text, or absolute path; and
- the final `review_bundle` stage artifacts — compact machine-readable and human-
  readable result indexes with paths to detailed evidence.

Within the v0.3 audit attempt, the raw cap reproduction,
`v03-counterfactual-cap-compatibility.json`, deduplicated cap report, duplicate-removal
inventory embedded in the compatibility report, visible-structure separation report,
and class inventories provide the
trace from the frozen raw material to the actual training inventory. Class inventories
are descriptive evidence, not acceptance thresholds. Within the v0.4 pilot attempt,
the pilot report records full-view length hashes, maximum/mean sequence lengths,
native-device resolution, and proof that the maximum training row was exercised.

After a completed development run, the human and machine review indexes are under the
committed review-bundle attempt:

```text
runs/phase6-remediation-v0.4.0-targeted-03/stages/15-review_bundle/attempt-*/REVIEW_BUNDLE.md
runs/phase6-remediation-v0.4.0-targeted-03/stages/15-review_bundle/attempt-*/review-bundle.json
```

For `blocked`, `stopped`, or `failed` runs, the command prints the exact paths to an
idempotent terminal-prefix bundle. If you return later, find it at:

```text
runs/phase6-remediation-v0.4.0-targeted-03/terminal-reviews/state-<pipeline-state-sha256>/TERMINAL_REVIEW.md
runs/phase6-remediation-v0.4.0-targeted-03/terminal-reviews/state-<pipeline-state-sha256>/terminal-review-bundle.json
```

Use the exact attempt or state-checksum path reported by the command; do not choose a
file merely because its name looks current.

Interpret terminal states as follows:

- `completed`: every permitted development stage and review-bundle stage committed.
  This still does **not** mean the fresh final evaluation ran or that Phase 7 unlocked.
- `blocked`: a scientific advancement gate failed; later stages were not run. This is
  a valid negative research result, not corrupted execution.
- `stopped`: work ended at a safe resumable boundary.
- `failed`: implementation, environment, resource, or integrity validation failed;
  inspect the recorded failure and do not edit around it.

Shell exit codes are meaningful: `0` success, `2` invalid usage, `3` missing input,
`4` configuration refusal, `5` unsafe/incompatible state, `6` another active runner,
`7` stage failure, `8` a managed stage stop or interrupt preserved at a safe boundary,
`9` scientific block, `10` locked final access, and `130` only a keyboard interrupt
that reached the outer command layer rather than being handled inside a managed stage.

## Safe failure recovery

Always run `./scripts/check_phase6_status.sh` first and preserve the entire run
directory. Public errors are deliberately bounded and may report a category without a
raw traceback or private internal detail.

- Exit `4` means the source, configuration, contract, tokenizer, or other frozen input
  binding was refused. Restore the exact bound checkout; do not edit evidence to make
  it match.
- Exit `5` means local state or integrity validation failed. Stop and ask Codex to
  inspect the verified state and terminal bundle before retrying.
- Exit `6` means another runner holds the lock. Check the existing process and status;
  do not start a competing runner.
- Exit `7` means a managed stage failed safely. Preserve its attempt and terminal
  bundle so Codex can distinguish implementation, environment, resource, and
  integrity categories before deciding whether resume is valid.
- Exit `8` means the managed pipeline reached a resumable stop boundary. Resume only
  after status verifies `stopped`.
- Exit `9` is a frozen scientific block, not a runtime failure. Do not resume past it
  or weaken the gate.
- Exit `10` is the expected final-access lock in this release. Do not create or edit
  marker files.
- Exit `130` means the outer command was interrupted. Verify status before deciding
  whether the managed pipeline recorded a resumable state.

## Files that must not be edited during or after a run

Do not manually edit, rename, replace, or delete:

- any file under `runs/phase6-remediation-v0.4.0-targeted-03/` or any preserved
  failed/blocked run: `runs/phase6-remediation-v0.4.0-targeted-02/`,
  `runs/phase6-remediation-v0.4.0-targeted-01/`,
  `runs/phase6-remediation-v0.4.0-local-rerun-02/`,
  `runs/phase6-remediation-v0.4.0-local-rerun-01/`, and
  `runs/phase6-remediation-v0.4.0-local/`;
- the v0.2, v0.3, v0.4, or pipeline TOML configurations;
- the compact-output contract or its manifest;
- the v0.2 inventory or v0.3 counterfactual-cap reports;
- tokenizer files, checkpoints, safetensors, manifests, checksums, stage markers,
  progress records, review bundles, final-ready records, or access ledgers; or
- dataset split manifests, final manifests, fresh golden-extension records, or owner-
  review records.

Do not remove `STOP_REQUESTED` yourself; the resume command archives it safely. Do not
copy another run over this one. If a clean restart is scientifically justified, ask
for a reviewed, versioned run-name change rather than deleting evidence.

## Review handoff after the development run

First record one final verified snapshot:

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
./scripts/check_phase6_status.sh
```

Then ask Codex to inspect, without rerunning the long job:

> Review the completed ReactorBench-LM Phase 6 remediation run in
> `/Users/zachary/Documents/Personal-Projects/AI-transformer`. Read
> `docs/IMPLEMENTATION_STATUS.md` and
> `docs/model/PHASE6_REMEDIATION_RUNBOOK.md` first. Verify Git status and inspect the
> checksum-bound run manifest, pipeline state, status, progress log, completion
> markers, review bundle, detailed v0.2/v0.3/v0.4 reports, baselines, checkpoints, and
> acceptance outcomes under
> `runs/phase6-remediation-v0.4.0-targeted-03/`. Do not retrain,
> open frozen final data, use historical G01–G15 as the new golden gate, push, or
> deploy. Tell me whether development completed, blocked, stopped, or failed; what the
> measured results show; and the exact reviewed prerequisite needed next.

Codex should verify the run manifest, `pipeline-state.json`, `status.json`,
`progress.jsonl`, every committed `stages/*/completed.json`, the review-bundle indexes,
and only the detailed artifacts referenced by those verified indexes. For a non-
completed run, provide the printed terminal-review manifest and summary paths as the
starting point.

## Future frozen final evaluation

The separate fresh final-evaluation executor remains intentionally unimplemented in
this release. The development runner, a `completed` state, a passing policy, a
confirmation flag, or readiness-looking files cannot unlock it. The existing
`run_phase6_evaluation.sh` wrapper is a fail-closed boundary test, not an executable
research step; it returns exit code `10` and does not open final data or create an
access record.

Its exact boundary-test form is shown only so its refusal remains auditable; do not use
it as a research command:

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
./scripts/run_phase6_evaluation.sh --confirm-final-evaluation
```

A later reviewed implementation must first define the distinct executor, freeze a
fresh final manifest and fresh golden extension, obtain explicit owner approval, and
bind every prerequisite before one-access evaluation can be considered. Historical
G01–G15 remains prohibited. Never fabricate, hand-edit, or bypass readiness,
extension, access, or owner-review records.
