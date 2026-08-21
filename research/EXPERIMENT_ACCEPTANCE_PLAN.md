# Experiment acceptance and error-analysis plan

Status: Phase 6 executed once; corrected mechanical rescore verified; behavioral acceptance failed
Rule: changing a frozen threshold requires a versioned amendment and cannot use test results

## 1. Purpose

This plan prevents ReactorBench-LM from declaring success based on a favorable aggregate score. It defines the comparisons, capability measurements, failure categories, and acceptance gates required for the main result.

## 2. Hypotheses

- **H1 — sequence learning:** the main Transformer improves next-event modeling over frequency and n-gram baselines.
- **H2 — structured diagnosis:** learned sequence models improve macro-averaged fault identification over bag-of-words and deterministic baselines on strict holdouts.
- **H3 — temporal evidence:** access to event order improves evidence selection and near-neighbor distinctions such as lag versus stuck.
- **H4 — composition gap:** compositional performance will be lower than IID performance; the size and causes of that gap are a primary result.
- **H5 — abstention:** confidence-based abstention reduces error as coverage decreases and outperforms forced classification on insufficient-evidence cases.
- **H6 — scaling:** model or data growth provides measurable benefit only up to a point within the local compute budget.

Failure to support a hypothesis is a valid result if the experiment and reporting remain sound.

## 3. Baselines

| Baseline | Purpose | Tasks |
|---|---|---|
| Majority/frequency | Establish class and event-frequency floor | fault, action, next event |
| Deterministic rule set | Test whether generator signatures make learning unnecessary | fault, action, abstention |
| N-gram model | Establish local language-pattern performance | next token, next event |
| Bag-of-words logistic regression | Test whether ordering is unnecessary | fault family |
| LSTM or GRU | Compare a non-Transformer sequence learner | next event, fault family |
| Smaller Transformers | Measure parameter and data scaling | all supported tasks |

All baselines must use the same train/development/test manifests and model-visible information.

## 4. Main metrics

- Language modeling: validation/test negative log-likelihood and perplexity.
- Next event: top-1 accuracy, top-k accuracy, and macro-F1 where events are imbalanced.
- Fault identification: macro-F1, per-family precision/recall/F1, exact match for compound labels.
- Evidence: evidence-event precision, recall, and F1.
- Action label: accuracy and macro-F1.
- Structured output: parse success and schema-validity rate.
- Abstention: required-abstention accuracy, selective risk, coverage-risk curve, and calibration error.
- Generalization: absolute and relative IID-to-holdout gaps.
- Efficiency: parameters, tokens, training time, peak memory, inference latency, and checkpoint size.

Open-ended prose scores are secondary and cannot override incorrect structured ground truth.

## 5. Split reporting

Report every primary metric separately for:

- training-distribution development set;
- IID test;
- unseen renderer/template families;
- unseen component aliases;
- structural role holdout;
- held-out fault combinations;
- robustness/OOD suite;
- golden behavioral suite.

Never merge these into one headline number without also showing the individual results.

## 6. Acceptance gates

The pilot-informed values below are frozen in
`configs/experiments/phase6-main-v0.1.0.toml`. They are acceptance gates, not
predictions. A failure remains a negative result and may not be repaired by changing a
threshold after test access.

| Gate | Required condition |
|---|---|
| Data integrity | zero prohibited-source, schema, invariant, duplicate, and cross-split leakage failures |
| Training correctness | tiny-shard overfit, causal-mask, save/reload, seed, and evaluation-isolation tests pass |
| Baseline evidence | every required baseline completed or omission documented before test access |
| Learned value | main Transformer beats the preregistered simple baseline on designated primary tasks |
| Normal specificity | no-fault false-positive rate is below the frozen threshold |
| Abstention | required-abstention and coverage-risk thresholds pass |
| Composition | strict holdout result is reported with uncertainty, regardless of whether a target threshold passes |
| Output validity | structured responses meet the frozen parse/schema-validity threshold |
| Reproducibility | smoke reproduction and release checksum verification pass in a clean environment |
| Deployment parity | deployed inference matches offline checkpoint outputs within the declared deterministic tolerance |

Numerical Phase 6 gates are:

- selected validation NLL reduction at least 90%, selected NLL at most 0.50, and at
  least 10% relative improvement over the smaller Transformer;
- fault-family and next-action macro-F1 each at least 0.02 above the strongest
  preregistered simple comparator on the same split;
- next-event macro-F1 at least 0.90 and target-token NLL at most 75% of trigram NLL;
- evidence F1 at least 0.70, parse success and schema validity each at least 0.99;
- no-fault false-positive rate at most 0.10 and required-abstention accuracy at least
  0.80;
- expected calibration error at most 0.15 and selective risk at 80% coverage at most
  0.20; and
- no pass threshold for composition: the result and 95% interval must be reported even
  when poor.

Every proportion/score comparison uses 2,000 deterministic bootstrap resamples with
seed 6602 and a 95% interval. Small supports and interval width must be shown; a point
estimate cannot conceal uncertainty.

If a gate fails, the result may still be published as a research finding, but the site and README must not imply that the capability passed.

## 7. Ablations

- Remove explicit component-state events.
- Remove redundant sensor evidence.
- Remove event ordering or shuffle within declared windows.
- Reduce renderer diversity.
- Remove abstention examples.
- Compare tokenizer choices if budget permits.
- Compare one versus multiple plant variants.
- Compare data and model-size tiers.
- Remove compound-fault examples while preserving single faults.

Ablations must answer a research question; do not run combinations simply to enlarge the experiment table.

## 8. Robustness and OOD suite

### Schema-valid but unfamiliar

- held-out aliases;
- unseen template families;
- harmless notes inserted;
- simultaneous events reordered without altering causality;
- values perturbed within the same state bands;
- scenarios lengthened or shortened within supported bounds.

### Information quality

- missing channel;
- conflicting redundant channels;
- one-tick outlier;
- truncated history;
- delayed decisive evidence;
- multiple compatible fault explanations;
- no recognizable fault.

### Contract and deployment

- unknown field;
- wrong enum;
- unsupported schema version;
- oversized sequence;
- invalid event reference;
- model/scenario version mismatch;
- safely encoded HTML-like output.

Robustness test inputs remain fictional and are separate from security attack scenarios.

## 9. Human review and golden-suite freeze

The prepared local `0.1.0` packet contains exact G01-G15 scenarios and structured
targets bound to generator commit `4473718`. Its semantic SHA-256 is
`c2e966564dadfab7e8b944ca9b6f8ef59d8545d1da1cc4ea75f8b27a9c44077c`.
The project owner approved the packet and all seven confirmations on 2026-08-20 before
held-out access. The checksum-bound record has semantic SHA-256
`1f5307889d259cfb0fa39e86e33ed9c2ce0922742e59af1d5ff5e0c904337288`.

Each golden scenario requires:

- reviewer identity or role;
- review date;
- generator/schema version;
- fault and evidence logic approval;
- expected action and abstention approval;
- confirmation that no real procedure, setpoint, topology, unit, facility, or Navy information appears;
- expected structured-answer checksum;
- final status: approved, revise, or retired.

Golden cases cannot be used as training examples. Revisions after test evaluation require a version bump and explanation.

## 9.1 Measured Phase 6 verdict

Validation-only selection passed, choosing E3 step 1,400 at validation NLL 0.073041.
The one held-out access evaluated 894 frozen examples plus 60 golden task examples.
After mechanically stripping the exact supervised `\n<|sep|>` transport suffix before
strict JSON parsing, IID parse/schema/exact rates were 21.03% / 5.16% / 5.16%; all
strict holdout exact-match rates except narrative noise were 0%; noise exact match was
4.17%; and golden exact match was 3.33%. Ten checks pass numerically and 23 fail.

The original predictions and report remain immutable. The correction generated no new
tokens and passed independent reconstruction. These results close v0.1 as a negative
experiment and block Phase 7. Any remediation must be preregistered as a separate
version and may not repeatedly tune against the frozen v0.1 held-outs.

## 10. Error taxonomy

Every reviewed failure receives one primary category and optional contributing categories:

- `FAULT_WRONG_FAMILY`
- `FAULT_MISSING_SECONDARY`
- `FAULT_SPURIOUS_SECONDARY`
- `EVIDENCE_MISSING`
- `EVIDENCE_SPURIOUS`
- `ACTION_WRONG`
- `FAILED_TO_ABSTAIN`
- `UNNECESSARY_ABSTENTION`
- `TEMPORAL_ORDER_FAILURE`
- `COMPOSITION_FAILURE`
- `RENDERER_SENSITIVITY`
- `BOUNDARY_SENSITIVITY`
- `INVALID_STRUCTURED_OUTPUT`
- `INSUFFICIENT_CONTEXT_BY_DESIGN`
- `GENERATOR_OR_LABEL_DEFECT`

Generator or label defects are dataset issues, not model failures, and must trigger correction plus affected-experiment review.

## 11. Failure gallery

The public report should include a balanced selection of:

- correct confident prediction;
- correct abstention;
- incorrect confident prediction;
- unnecessary abstention;
- counterfactual flip that works;
- counterfactual flip that fails;
- compound-fault success or failure;
- example where a simple baseline beats the Transformer.

Each example should show input evidence, model output, ground truth, error category, and interpretation without implying real-world relevance.
