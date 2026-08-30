# Phase 6 targeted-03 gate replay

Status: **checksum-bound replay complete; scientific block at 9 of 10 checks**

## Outcome

`phase6-remediation-v0.4.0-targeted-03` completed all 2,500 candidate-training steps
and all 427 development-gate evaluations. Its original stage-10 gate attempt failed
safely before publishing acceptance because the independent reconstruction compared
the hierarchical training score against the historical
`1 - semantic_composite` formula. Training, checkpoint selection, calibration, and
evaluation had completed successfully. The preserved source run was not modified.

The gate now delegates to the same policy-aware minimized score used during training:

- historical selection minimizes `1 - semantic_composite`;
- hierarchical selection requires the frozen 0.75 semantic floor, then uses validation
  NLL and the earlier step; and
- candidate reconstruction uses the corresponding frozen policy and fails closed on
  an incompatible inventory.

A separate replay at implementation commit
`6b8a89c10aeec45518246d6b1ac082480cec4af7` verified the clean checkout, run
manifest, pipeline state, all nine completed-stage markers, stage outcomes, and every
referenced artifact by canonical contract, size, SHA-256, source binding, path
containment, and non-symlink checks. It then independently reconstructed the gate from
the completed artifacts. It did not train, decode new examples, open final data, or
access the historical golden packet.

## Reconstructed acceptance

| Check | Observed | Required | Result |
|---|---:|---:|---|
| constrained parse | 1.0000000 | = 1.0000000 | pass |
| constrained schema validity | 1.0000000 | = 1.0000000 | pass |
| fault comparator margin | -0.0344200 | >= 0.0200000 | **fail** |
| action comparator margin | 0.5242703 | >= 0.0200000 | pass |
| continuation macro-F1 | 0.9418182 | >= 0.9000000 | pass |
| evidence F1 | 0.9330784 | >= 0.7000000 | pass |
| required abstention accuracy | 0.9540230 | >= 0.8000000 | pass |
| no-fault false-positive rate | 0.0000000 | <= 0.1000000 | pass |
| expected calibration error | 0.0931758 | <= 0.1500000 | pass |
| selective risk at 80% coverage | 0.1695906 | <= 0.2000000 | pass |

The unchanged gate passes nine of ten checks and denies advancement. This is now a
valid model-quality result. The remaining work is fault-comparator remediation; the
threshold must not be lowered merely to advance.

## Immutable identities and checksums

- preserved source run: `phase6-remediation-v0.4.0-targeted-03`
- preserved source commit: `240d3955e141432d2d24cf567f0117701e634037`
- replay identity: `phase6-remediation-v0.4.0-targeted-03-gate-replay-01`
- replay implementation commit: `6b8a89c10aeec45518246d6b1ac082480cec4af7`
- pipeline config SHA-256:
  `46672289436a789e21e017c822f0dc831da6ef38b308abf6d271e223d1e782a7`
- source run-manifest contract SHA-256:
  `1d85f61b5199b5328621850f8df554fe632023eb861aad98037a69e64162ef8d`
- source pipeline-state contract SHA-256:
  `25a7116ccbdd7215e0283930043e9cf6f8d4a996ad88c20ad6573135d4f76f12`
- training completion-marker file SHA-256:
  `ddec981c58d605590e7a17c29ac694e5d079652c414e4d15f3250ba1f47de86a`
- evaluation completion-marker file SHA-256:
  `3bcc122ca7b25bc85694ca0b7de9f4f777d689fed706acddb4adf1a05bb254c8`
- reconstructed acceptance contract SHA-256:
  `c86624943c607c99836d2c05e8d5e727655b5d8472fa2f18269321b337506c61`
- targeted gate-binding contract SHA-256:
  `14afcc2adc9f78371c2068e8614c94e85f3d9c90a31e4f5f099a8438309576c5`
- replay-certification contract SHA-256:
  `59ae5bf70e538c61b99181e27ce2de7a90e48ced0d332c1c9ce5537d7b80ee21`

The local replay files are under:

```text
runs/phase6-remediation-v0.4.0-targeted-03-gate-replay-01/
```

The `runs/` tree is intentionally excluded from Git. The checked-in report and code
record the result; local JSON remains the authoritative machine-readable evidence.

## Reproduction boundary

The fixed one-time command is:

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
./scripts/replay_phase6_targeted03_gate.sh
```

It refuses a dirty source checkout, a changed targeted-03 source run, changed config,
missing or mismatched completion evidence, symbolic-link traversal, an existing replay
identity, or any checksum/contract disagreement. The successful scientific block exits
with code `9`. Because the identity is non-overwriting, the existing certificate must
be reviewed rather than rerun or deleted.

## Local monitor alarm control

The Phase 6 development monitor retains the 45-second looping failure alarm and
fallback system beeps. While an alarm is active, **Stop alarm** is enabled. Pressing it
stops the loaded sound and cancels all pending fallback work items; it does not alter
pipeline evidence, status, or the once-per-session alert guard.
