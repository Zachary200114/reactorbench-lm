# Phase 5 baseline and pilot evidence

Status: complete local development evidence, 2026-08-20 America/New_York

## Claim boundary

This report covers only the approved `iid_train` and `iid_validation` task examples.
No IID test, template, component, severity, composition, counterfactual, noise, golden,
or robustness result was accessed. The models were initialized from random weights;
the tokenizer and all text are project-authored local artifacts. These measurements do
not establish test generalization, operational usefulness, or real-world safety.

## Inputs and correction before measurement

The run used 630 training examples and 252 validation examples. A first invocation
stopped before baseline fitting or optimization because some complete canonical targets
did not fit the Phase 4 128/256-token tier contexts. No output directory was produced.
The amended, committed experiment retained every target and fixed both Phase 5 contexts
at 512 while leaving width, depth, seeds, optimizers, and step counts unchanged. The
amendment is source commit `a2c180ec35e1c916f4e5068a7c932c5d6d2fec18`.

The 512-token contract retained the complete target for every example and truncated
only the oldest prompt prefix where necessary: 491/630 training prompts and 198/252
validation prompts. That high prompt-truncation rate is a measured limitation and is
one reason Phase 6 must report insufficient-context errors explicitly.

## Baseline measurements

All values below are validation measurements. “Accuracy / macro-F1” is shown for
classification rows; NLL/perplexity is shown for the token language model.

| Baseline | Task | Parameters | Result |
|---|---|---:|---:|
| Majority/frequency | Fault family | 0 | 0.3448 / 0.0641 |
| Majority/frequency | Next action | 0 | 0.3448 / 0.0641 |
| Majority/frequency | Next event | 0 | 0.8000 / 0.4444 |
| Deterministic keyword rules | Fault family | 0 | 0.6379 / 0.6626 |
| Deterministic keyword rules | Next action | 0 | 0.4828 / 0.4132 |
| Word suffix trigram | Next event | 9 | 0.9000 / 0.8667 |
| Add-0.1 token trigram | Target language model | 865 | NLL 1.0955 / PPL 2.9906 |
| Bag-of-words softmax | Fault family | 8,848 | 0.5000 / 0.2610 |
| One-layer GRU | Fault family | 178,504 | 0.6034 / 0.5050 |
| One-layer GRU | Next event | 177,922 | 1.0000 / 1.0000 |

The validation subsets for fault/action and next-event classification contain only 58
and 20 examples respectively. The perfect GRU next-event result is therefore recorded
but not treated as a generalization claim. The rules exceeding the GRU on fault-family
macro-F1 is a useful negative result: sequence-model complexity is not automatically
valuable on every current task view.

## Transformer measurements

| Tier | Parameters | Steps | Initial NLL | Selected NLL | Reduction | Time | Target tok/s | Checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Smaller | 724,480 | 300 | 7.6093 | 0.5343 | 92.98% | 20.27 s | 15,335.62 | 3,949,312 B |
| Pilot | 5,394,432 | 500 | 7.7257 | 0.1593 | 97.94% | 268.70 s | 1,909.38 | 23,682,552 B |

Both checkpoints selected the final scheduled step using validation target NLL only.
The pilot curve was `2.1402`, `1.1738`, `0.8057`, `0.5702`, `0.3895`, `0.3420`,
`0.2576`, `0.1919`, `0.1594`, and `0.1593` at steps 50–500. Training was not stopped
early. The smaller and pilot final training NLL values were 0.5529 and 0.08385.

Both runs used Apple MPS. Peak current/driver allocations were 18,547,200 / 1,178,468,352
bytes for the smaller tier and 143,629,312 / 3,401,547,776 bytes for the pilot. The
process peak RSS recorded by both rows was 994,131,968 bytes. Thermal throttling was not
observable and is not claimed. The 15,179,520-parameter main tier remains feasible on
the measured machine, but Phase 6 uses batch four and no silent CPU fallback to preserve
memory margin and make a backend change explicit.

## Pilot-informed Phase 6 freeze

`configs/experiments/phase6-main-v0.1.0.toml` freezes the next experiment before any
test access. The main model remains 8 layers × width 384 × 8 heads, context 512, and
15,179,520 parameters. Training is fixed at 1,500 MPS steps, batch four, seed 6601,
learning rate 0.00025, weight decay 0.01, and validation every 100 steps.

The selected main checkpoint must reduce validation NLL by at least 90%, reach NLL at
most 0.50, and improve at least 10% relatively over the smaller tier. Structured test
gates are frozen as margins or capability thresholds, not predictions: +0.02 macro-F1
over the strongest preregistered simple comparator for fault and action, next-event
macro-F1 at least 0.90, target NLL at most 75% of trigram NLL, evidence F1 at least
0.70, parse/schema validity at least 0.99, no-fault false-positive rate at most 0.10,
required-abstention accuracy at least 0.80, expected calibration error at most 0.15,
and selective risk at 80% coverage at most 0.20. Composition has no pass threshold and
must be reported with uncertainty even if poor. A failed gate remains a reportable
negative result.

The config also freezes 2,000 seeded bootstrap resamples at 95% confidence and the E0–E7
comparison matrix. It does not authorize Phase 6 execution: the golden-suite human
review, test-manifest freeze, and Phase 6 evaluator implementation remain prerequisites.

## Artifact lineage

- Run directory (ignored local evidence): `runs/phase5-pilot-v0.1.0`
- Phase 5 source commit: `a2c180ec35e1c916f4e5068a7c932c5d6d2fec18`
- Report internal SHA-256: `5c21a4ff93701cdaa73e59e5b9a488cc171009dd1c35ac1f2a234dc7db029ffc`
- Report raw SHA-256: `67d89bf24ea6b3d739b29c4253d7f9cd53843096750961e43d0c2033682eb28a`
- Phase 5 config canonical SHA-256: `5e75ccc0c5dfab1ba1a34c37ba490ff55b53c49d6596d956efd037369bee7bc3`
- Phase 5 config raw SHA-256: `84a4dfef27bdb30f6ab36cf974dd7b059bc30dc36eb93f6d5db1cc6db8a8868f`
- Experiment inventory SHA-256: `94b8ea55d804c7a48726a3a135b89aea5f15a471d0b84bb1d19b947d09015fe0`
- Dataset candidate SHA-256: `3bba04bdb2030425ef67845332540fa2d148d0a318ab1d9e658f52bb890bf10c`
- Tokenizer manifest SHA-256: `ef80afa52030c764598663b0f51b90e7b753b91377b47b4a5648d729e0011ef8`
- Dependency lock SHA-256: `dc09fe5e44a7a08f314558b92776f2ca77153842770524ea1818c6a41c6269e6`
- Smaller manifest / weights SHA-256: `917425dad7c6995cb99cc7bc19fc821bbd921dbd98931ee2efd8b502748495ce` /
  `d2a0bd783d82dce684742e40de88d3c8f325900cc65d086f3764867ff0355318`
- Pilot manifest / weights SHA-256: `8d4b93d29cd1c4becf2062cff8e3eb810259601d1685eb565ced5f5382da2daa` /
  `2b2b60205e77ddf8cedf8c41bfa0761814eb82eeef51db2bb8d586fe5afd319f`
- Phase 6 config canonical / raw SHA-256: `d50089cd85863a411e93417df6448110e2d021bf96547f7e6ae471e880c4ab1c` /
  `0dbbab59962fac4f4a90c981a613e6b223eacd7c21f77550339c64f78ed989cc`
