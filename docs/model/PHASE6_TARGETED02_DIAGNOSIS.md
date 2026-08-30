# Phase 6 targeted-02 diagnosis and targeted-03 remediation

Status: **targeted-02 preserved as a scientific block; targeted-03 prepared but not run**

## Outcome

`phase6-remediation-v0.4.0-targeted-02` completed through the v0.3 development gate
and stopped correctly. This was not a runner, decoder, checksum, or GUI failure. The
gate reconstructed the evidence and rejected a model that passed four of the ten
unchanged acceptance checks.

| Check | Observed | Required | Result |
|---|---:|---:|---|
| constrained parse | 1.0000 | = 1.0000 | pass |
| constrained schema validity | 1.0000 | = 1.0000 | pass |
| fault comparator margin | -0.1400 | >= 0.0200 | fail |
| action comparator margin | -0.0680 | >= 0.0200 | fail |
| continuation macro-F1 | 0.9418 | >= 0.9000 | pass |
| evidence F1 | 0.6778 | >= 0.7000 | fail |
| required abstention accuracy | 0.9885 | >= 0.8000 | pass |
| no-fault false-positive rate | 0.5556 | <= 0.1000 | fail |
| expected calibration error | 0.2843 | <= 0.1500 | fail |
| selective risk at 80% coverage | 0.4678 | <= 0.2000 | fail |

The acceptance artifact is checksum
`c1a80d3703f411cb5e1e066648e8540ba4d42cae9ea0534c168636a413293175`.
The terminal pipeline-state checksum is
`71de4360001d6851dce54a5720d97b0a30137fa6d6038b0f334263339589289c`.
No final or golden evaluation was opened.

## Diagnosis

The focused sampler achieved its intended continuation improvement: continuation
macro-F1 rose from targeted-01's 0.7396 to 0.9418. Its six-row batch, however, assigned
four rows to fault and continuation and alternated only two rows across the other four
tasks. That reduced exposure for action, evidence, and the remaining tasks. The gate
then measured next-action macro-F1 0.2421 and evidence F1 0.6778. Fault predictions
also over-selected `UNRESOLVED`; five of nine `NO_FAULT` examples became false
positives.

Checkpoint selection amplified the problem. The 48-row semantic selector chose step
1,200 with composite 0.8480 and validation NLL 0.1080, even though validation NLL
continued down to 0.0688 at step 2,000. The small selector's semantic score fluctuated
enough to prefer an early checkpoint that did not generalize to the independent
427-row gate.

This evidence supports a narrower correction. It does not prove that the next run
will pass.

## Frozen targeted-03 correction

The new non-overwriting identity is
`phase6-remediation-v0.4.0-targeted-03`. It uses:

- one example from every task in every six-row training batch;
- uniform label rotation for continuation and next-action tasks;
- a fault hierarchy that allocates 50% of fault draws to `UNRESOLVED`, 10% to
  `NO_FAULT`, and 40% uniformly across diagnosed fault families;
- 2,500 training steps at the same 15,179,520-parameter architecture;
- a frozen 0.75 semantic checkpoint floor, followed by lower validation NLL and then
  earlier step as tie-breaks; and
- the same 48/56/427 development partitions, calibration grid, externally pinned
  21-file v0.2 prefix, and all ten existing acceptance thresholds.

The v0.3 config checksum is
`77ab1698ec37bc65e319fcc55b3d6921860bdd317d7600b5a2da1bd5f71fa158`.
The pipeline config checksum is
`46672289436a789e21e017c822f0dc831da6ef38b308abf6d271e223d1e782a7`.
Targeted-02 and every earlier run remain immutable.

## Local failure alarm

The macOS monitor now plays the system `Sosumi` sound in a loop at the maximum volume
available to the application for 45 seconds when a run first enters `Blocked` or
`Failed`. If that sound cannot be loaded, it emits 23 system beeps over approximately
45 seconds. It also requests critical user attention and retains the once-per-session
guard.

This is intentionally not described as guaranteed to wake someone. Application volume
cannot override a muted Mac, a low system output level, Focus settings, disconnected or
routed headphones, powered-off speakers, or a sleeping/powered-off computer. Verify
the Mac's output device and system volume before leaving a run unattended.
