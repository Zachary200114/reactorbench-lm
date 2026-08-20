# Phase 4 tokenizer and smoke-model evidence

Status: local correctness artifact, not a released model or generalization result.

> All systems, events, values, policies, and narratives in ReactorBench-LM are
> synthetic and fictional. This model is not suitable for operational, engineering,
> licensing, emergency, maintenance, security, or safety decisions.

## What this proves

Phase 4 proves that the project can train its own tokenizer, allocate a decoder-only
causal Transformer from random initialization, optimize next-token loss, preserve
causal masking, and save/reload a safe data-only checkpoint without changing logits.

It does not prove validation performance, generalization, diagnostic accuracy,
calibration, abstention quality, robustness, or usefulness. Those claims require the
preregistered Phase 5–6 baselines and evaluations.

## Data boundary

The tokenizer reads only the project-owner-approved Phase 3 render candidates assigned
to `iid_train`. The extracted corpus contains 195 documents and 685,978 UTF-8 bytes.
Its SHA-256 is
`e8433ec549df79d274ebee6ffa32f1fe7810df3db256a7db5bf817dac4ccdc6e`.
No validation, IID-test, template, component, severity, composition,
counterfactual, or noise-test prose enters tokenizer training.

## Tokenizer

| Property | Value |
|---|---:|
| Algorithm | SentencePiece BPE, trained by this project |
| Vocabulary | 2,048 |
| Normalization | identity |
| Byte fallback | enabled |
| IDs | UNK 0, BOS 1, EOS 2, PAD 3 |
| Project symbols | `<|prompt|>`, `<|target|>`, `<|sep|>` |
| Manifest checksum | `ef80afa52030c764598663b0f51b90e7b753b91377b47b4a5648d729e0011ef8` |
| Model checksum | `b2ced4e9699f019a053516c9ff4a6c698d1bd17f9b070632ac3e03b565a2af6c` |
| Model size | 33,586 bytes |

The training implementation fixes sentence ordering, disables shuffling, uses one
trainer thread, and trains under a constant local model prefix. Two independent test
runs produce byte-identical model and vocabulary files.

## Model architecture

The implementation in `src/reactorbench/model/transformer.py` uses PyTorch for tensor
operations, automatic differentiation, and optimization only. Project code defines the
architecture and training behavior:

- token and learned positional embeddings;
- explicit query/key/value projections and lower-triangular causal masking;
- pre-layer-normalization residual blocks;
- multi-head self-attention;
- GELU feed-forward networks;
- tied token/output embeddings;
- cross-entropy next-token loss; and
- random normal initialization with standard deviation 0.02.

No pretrained weights or hosted model API is loaded.

| Tier | Layers | Width | Heads | Context | Exact parameters |
|---|---:|---:|---:|---:|---:|
| Smoke | 2 | 128 | 4 | 128 | 675,328 |
| Pilot | 6 | 256 | 8 | 256 | 5,328,896 |
| Main | 8 | 384 | 8 | 512 | 15,179,520 |

Pilot and main settings remain provisional until Phase 5 measurements.

## Measured smoke run

The final run is bound to source commit
`f636f2b2b9f7f5915bef78903611aa047aaecc30`, canonical Phase 4 config checksum
`af55961930cf856e9953d9b27f2d3f270307bc3bb55e60e73bd31b53cae4bee9`, and dependency
lock checksum
`dc09fe5e44a7a08f314558b92776f2ca77153842770524ea1818c6a41c6269e6`.

| Measurement | Result |
|---|---:|
| Device | CPU |
| Documents / batch | 4 / 4 |
| Sequence length | 128 |
| Scored targets per step | 508 |
| Steps | 300 |
| Initial loss | 7.6617279052734375 |
| Final loss | 0.011601113714277744 |
| Loss reduction | 99.8485835850906% |
| Timed optimization loop | 3.8152835840010084 s |
| Timed target throughput | 39,944.60612025628 tokens/s |

The time and throughput cover the deterministic optimization loop, not Phase 3
artifact verification, tokenizer training, checkpoint I/O, or independent verification.
Peak memory and MPS throughput are unmeasured. Torch reports MPS compiled in, but the
backend was unavailable to the sandboxed process.

## Integrity evidence

- Causal-mask probe: passed.
- Exact shifted-target and padding-boundary tests: passed.
- Tiny-shard overfit thresholds: final loss ≤ 0.75 and reduction ≥ 80%; passed without
  changing either threshold.
- Deterministic repeated evaluation: passed.
- Safetensors save/reload exact-logit equality: passed.
- Independent config-selected verification: passed.
- Evaluation-logit SHA-256:
  `e1f726f6a89e6dd987c30f496008a49d2c3d74af672444097f92c91975fc2d5a`.
- Checkpoint-manifest checksum:
  `0d969c046e95f76c270b6f6d01a8331ef540a2bf52ab2d15ac57b566c697260e`.
- Safetensors checksum:
  `f085fff20bfe14df785b23a43630516228890aa38868cafb3ef478ac30cac241`.
- Smoke-report checksum:
  `b2b28c6b4d95661ef67698f5a2eb4752c2c5f8d26c7541a503f3e71ddecd4683`.

The config-selected artifacts remain ignored under `runs/phase4-smoke-v0.1.0`.
They are local research evidence, not public release assets.

## Reproduction

In a clean checkout that also contains the approved ignored Phase 3 candidate and
post-render approval record:

```bash
make sync
make reproduce-smoke
make phase4-verify
```

The write is non-overwriting. To verify the current local artifact without creating a
new run:

```bash
.venv/bin/python -m reactorbench.training verify-smoke \
  --config configs/model/phase4-smoke-v0.1.0.toml
```

Do not cite the archived pre-lock-binding run. The config-selected final run is the
only Phase 4 smoke artifact recorded by the implementation handoff.
