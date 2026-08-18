# Experiment acceptance and error-analysis plan

Status: pre-implementation measurement contract
Rule: numerical thresholds will be frozen after pilot measurements and before final test evaluation

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

Exact numerical values remain `TBD-PILOT` until the pilot is complete. Before main training, replace them with justified frozen thresholds.

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
