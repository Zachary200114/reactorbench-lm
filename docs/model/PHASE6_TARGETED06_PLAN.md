# Phase 6 targeted-06 remediation plan

Status: **implemented and source-verified; no targeted-06 training run started**
Date: 2026-09-17 America/Chicago

## Why I am running another version

The preserved targeted-05 diagnostic run reached v0.4 shadow evaluation after all
2,500 v0.3 and 2,500 v0.4 training steps. Its v0.3 gate passed nine of ten unchanged
checks. The only miss was fault-comparator margin: `-0.0032795487`, with `>= 0.02`
required.

The partial v0.4 IID result passed eight of ten checks. It missed fault-comparator
margin (`0.004935`, required `>= 0.02`) and expected calibration error (`0.173229`,
required `<= 0.15`). Continuation macro-F1 was `0.91`, so I am retaining the
successful continuation weighting rather than broadening the experiment.

The diagnostic run then failed for an engineering reason before completing v0.4:
six `shadow_composition` counterfactual targets exceeded the historical 108-token
counterfactual generation cap. Their measured lengths were 122, 130, and 230 tokens.
This was not a threshold result and does not make the partial shadow metrics official.
The failed run remains unchanged at
`runs/phase6-remediation-v0.4.0-targeted-05-diagnostic-01/`.

## Frozen targeted-06 changes

Targeted-06 is a new non-overwriting experiment. I did not lower an acceptance
threshold, alter a historical run, or open final/golden evaluation.

- v0.3 keeps the targeted-05 six-task hierarchical sampler and class mix.
- Fault-family target tokens receive weight `3.0`, continuation target tokens retain
  weight `2.0`, and all other target tokens retain weight `1.0`.
- The existing semantic, fault-F1, and continuation-F1 checkpoint floors remain
  mandatory before validation NLL can break a tie.
- v0.4 now carries the same hierarchical weighted objective and task-aware checkpoint
  policy forward instead of reverting to the historical task-balanced objective.
- Because that hierarchy consumes one row from each of the six tasks, v0.4 main
  training uses batch size `6`. Its bounded MPS pilot now measures batches `1`, `2`,
  `4`, and `6`; batch `6` is the mandatory feasibility result.
- Each v0.4 candidate fits temperature on the existing disjoint 56-row
  validation-only calibration partition. The 48-row checkpoint-selection partition
  remains excluded, and acceptance uses the existing 427-row IID gate partition.
- The shadow-only counterfactual cap is `256`. IID training caps are unchanged.
- Before v0.4 pilot or main training, a checksum-bound audit measures every shadow
  target and fails if any target exceeds its task cap.
- The gate independently reopens calibration predictions, baseline artifacts, and
  candidate predictions to reconstruct calibrated IID and shadow reports.

## Diagnostic continuation policy

The diagnostic run still stops on training failures, resource limits, unsafe paths,
checksum/provenance errors, stop requests, and shared-stage contract failures. During
v0.4 shadow evaluation only, an isolated `ValueError` for one candidate/view is
recorded without its raw message, and the sweep continues through the remaining
independent shadow views. It then fails the stage safely with a checksum-bound
`v04-diagnostic-view-failures.json` report. This collects more engineering evidence;
it does not convert a failed stage into a passing result.

## Identities and hashes

- v0.3 config:
  `configs/experiments/phase6-remediation-v0.3.6-fault-emphasis.toml`
- v0.3 canonical SHA-256:
  `f15c9f77450fb038c86243890d25a7eb426d35dd7fb47470cbfebeb911773e8b`
- v0.4 config: `configs/experiments/phase6-remediation-v0.4.1.toml`
- v0.4 canonical SHA-256:
  `db3501a4d533b912a669cc6d7cefc6ef76a455a6fb8c1d24f3866b38671de84f`
- official run: `phase6-remediation-v0.4.1-targeted-06`
- official pipeline SHA-256:
  `e7539a66c4df658807fa6d0c0fa0a8d12bf0b47939a5931d69c84befff35dc48`
- diagnostic run: `phase6-remediation-v0.4.1-targeted-06-diagnostic-02`
- diagnostic pipeline SHA-256:
  `3c81b6197c6b971a9d8e57e63e5fb18db5b73f71557a1c845dbb1dbfa75e6bab`

## Stop condition

The official path remains fail-fast. The diagnostic path may continue past the two
allowlisted scientific gates, but it never certifies the model or authorizes Phase 7.
If targeted-06 still misses a model-quality threshold, I will preserve that negative
result and diagnose it from the complete development evidence rather than lower the
threshold after seeing the result.
