# Phase 6 main experiment and held-out evaluation

Status: **executed and independently verified; behavioral acceptance failed**  
Date: 2026-08-20 America/New_York

Phase 6 is complete as an experiment, not as a successful model-capability gate. The
frozen main run, E0-E7 matrix, all 894 held-out examples, and the owner-approved G01-G15
suite were evaluated. The result is a useful negative finding: low teacher-forced
target NLL did not translate into reliable free-running structured generation.

The failed gates prohibit presenting this checkpoint as ready for inference or the
Phase 7 live application. No threshold was changed after test access, and no model was
retrained on a held-out result.

## Integrity boundary

- Golden review: project-owner `APPROVED`, all seven confirmations true.
- Golden packet semantic SHA-256:
  `c2e966564dadfab7e8b944ca9b6f8ef59d8545d1da1cc4ea75f8b27a9c44077c`.
- Golden review record semantic/raw SHA-256:
  `1f5307889d259cfb0fa39e86e33ed9c2ce0922742e59af1d5ff5e0c904337288` /
  `9105c6e7e76979fdbc8b4a73d42f323acb636d36affbee13f8147dc23e3f06be`.
- Selection and original evaluation source commit:
  `d8dc2936b1f9355d4246a7be2a297ad210f1cbfe`.
- Held-out access count: exactly one, after validation-only selection.
- Held-out access record raw SHA-256:
  `410ab34416d860b5cc3b6067cfc599f691ad199a986d56f019e81b75246ae74f`.
- Test inventory: 894 examples across IID, template, component, severity,
  composition, counterfactual, and narrative-noise splits.
- E7 remains `not_applicable_no_compound_iid_train_rows`; it was not fabricated.

## Selection results

| Experiment | Parameters | Selected step | Validation NLL | Fit time |
|---|---:|---:|---:|---:|
| E3 main Transformer | 15,179,520 | 1,400 | 0.073041 | 776.66 s |
| E5 renderer-diversity ablation | 5,394,432 | 500 | 0.320094 | 154.31 s |
| E6 abstention-data ablation | 5,394,432 | 500 | 0.676041 | 163.30 s |

The E3 selection thresholds passed. Its checkpoint is a 63,873,952-byte safetensors
artifact; manifest/weights SHA-256 values are
`4a7750a64da19530b1965c1b40860ae4c8ddd3bdac6c76f62f7f6f0c6c916421` /
`3d1ba00431fb95695d4b17baf4271e574f777832c7ee7f7a3097901744205ab5`.
Selection report semantic/raw SHA-256 values are
`29a1c07864e24b47e7928df05d5738834ea10879ec46ff91f2834fae7113379d` /
`059504a13748c2c9131441782217b980fcee19fbebaf4d8157ef17b532d09552`.

## Evaluator incident and correction

The original evaluator generated complete serialized targets ending in the frozen
training delimiter `\n<|sep|>`, then attempted to parse the entire serialized string as
JSON. That made every output appear unparseable. This was an evaluator defect because
the Phase 5 serialization contract explicitly trains on the separator.

The original report and predictions were preserved unchanged. Commit
`e6504695583403f7e31b118f99ce67c6873bd8e8` fixes future parsing and adds the one-time
`scripts/phase6_rescore_v0_1_1.py` audit. The correction:

- strips only the exact terminal configured delimiter before strict JSON parsing;
- reuses every original generated string byte for byte;
- generates no new model token;
- recomputes confidence only for outputs that become schema-valid, using the exact
  stored greedy token sequence and frozen checkpoint;
- leaves the one-time held-out access record unchanged; and
- independently reconstructs the complete corrected report and prediction graph.

Original report semantic/raw SHA-256 values are
`2009afdeae9247125a30b4afcc24b41409ffb6ad3afd64d2772c2a491fc55967` /
`2b0a78a703c17810c87e5908e1034c72056315e40a9147e5139a8b2499034b44`.
The corrected report semantic/raw SHA-256 values are
`fb1ee4e13ba8ca44116641e5892ff3a7eef523846cd907fe07f368a76e09a0ce` /
`11306d82324b190bb06cb96d96137ef399bebccc0f65591e15e9ff2782d0aa22`.
Correction-record semantic/raw SHA-256 values are
`04077738a5fb915ea818ee2aadbda320769c6721476e9bc304fdcace96b4cd81` /
`75834c089400959f8a258b786dee1c6c12067fe7f1c7fdb39934b518a67c2aeb`.

## Corrected held-out behavior

| Split | N | Parse | Schema valid | Exact match | Evidence F1 | Required abstention | Selective risk @ 80% |
|---|---:|---:|---:|---:|---:|---:|---:|
| IID | 252 | 21.03% | 5.16% | 5.16% | 0.0238 | 6.67% (4/60) | 0.9356 |
| Template | 252 | 23.02% | 0.40% | 0.00% | 0.0120 | 0.00% (0/60) | 1.0000 |
| Component | 72 | 15.28% | 0.00% | 0.00% | 0.0000 | 0.00% (0/18) | 1.0000 |
| Severity | 52 | 7.69% | 0.00% | 0.00% | 0.0000 | 0.00% (0/12) | 1.0000 |
| Composition | 116 | 12.07% | 0.00% | 0.00% | 0.0000 | 0.00% (0/18) | 1.0000 |
| Counterfactual | 102 | 3.92% | 0.00% | 0.00% | 0.0000 | 0.00% (0/12) | 1.0000 |
| Noise | 48 | 25.00% | 4.17% | 4.17% | 0.0000 | N/A (0 required) | 0.9487 |

IID exact match has a deterministic 95% bootstrap interval of 2.38%-7.94%. Noise
exact match has a 0.00%-10.42% interval. All other split exact-match intervals are
0.00%-0.00%.

Golden exact match is **2/60 (3.33%)**. Both exact cases are G15 sparse-evidence
outputs; this is not broad golden-suite success.

The main checkpoint's teacher-forced target NLL/perplexity remains strong on IID
(0.04254 / 1.04346), template (0.14332 / 1.15406), and severity
(0.04621 / 1.04730). It degrades on composition (0.62170 / 1.86210) and especially
counterfactual (1.19179 / 3.29300). Four composition and ten counterfactual records are
`INSUFFICIENT_CONTEXT_BY_DESIGN` because their complete targets exceed the model
context.

This likelihood/decoding gap is the central result. Teacher forcing rewards the next
token when every prior target token is correct. Free-running greedy generation must
recover from its own errors; it usually produced malformed or contract-invalid JSON.

## Structured task results and baselines

The main model's IID macro-F1 is 0.0000 for fault family, 0.0290 for next action, and
0.2029 for continuation. On noise it reaches 0.2222 continuation macro-F1; other
reported strict-holdout task macro-F1 values are zero.

Simple comparators remain materially stronger. Examples include:

- IID deterministic-rule fault macro-F1: 0.6227;
- template GRU fault macro-F1: 0.6414;
- severity deterministic-rule fault macro-F1: 1.0000;
- IID/template/noise best continuation macro-F1: 1.0000; and
- composition word n-gram continuation macro-F1: 0.7333.

The main model passes the preregistered target-NLL fraction gate on IID, template,
component, severity, and noise, but fails it on counterfactual. It does not beat the
simple baselines on any required fault/action margin.

## Acceptance verdict

Ten checks pass: golden approval, held-out ordering, main selection, calibration's
numeric bound, no-fault false-positive numeric bound, and five target-NLL fraction
checks. Twenty-three checks fail, including:

- parse success and schema validity;
- evidence F1;
- required abstention and selective risk;
- all required simple-baseline fault/action margins;
- all required continuation macro-F1 gates; and
- counterfactual target-NLL fraction.

The calibration and no-fault checks must not be overinterpreted. Invalid outputs carry
confidence zero, only 16/894 main outputs are schema-valid, and no valid prediction is
a diagnosed false positive. These numerical passes are therefore vacuous rather than
evidence of a deployable uncertainty model.

The failure gallery contains delimiter-corrected `invalid_json`, `schema_invalid`, and
`evidence_error` examples. Negative results are preserved in the report rather than
hidden by a post-hoc threshold change.

## Reproduction and verification

```bash
.venv/bin/python -m reactorbench.training verify-phase6-selection \
  --config configs/experiments/phase6-main-v0.1.0.toml

.venv/bin/python -m reactorbench.training verify-phase6-evaluation \
  --config configs/experiments/phase6-main-v0.1.0.toml

.venv/bin/python scripts/phase6_rescore_v0_1_1.py verify \
  --config configs/experiments/phase6-main-v0.1.0.toml
```

The final code gate passed 713 tests with 85.10% branch coverage on CPython 3.12.11,
Ruff formatting/lint, strict mypy for 113 source files, and strict typing of the
one-time rescore script.

## Consequence

Phase 6 is closed as a fully executed negative experiment. The current checkpoint is
not eligible for the Phase 7 inference/UI gate. The next work must be a separately
preregistered v0.2 remediation cycle using training/validation evidence—not repeated
tuning against these held-out results. Plausible candidates include grammar-constrained
decoding, shorter task-specific target encodings, stronger sequence-level training,
and a context strategy that does not discard so much prompt history. Those are future
hypotheses, not retrospective changes to this result.
