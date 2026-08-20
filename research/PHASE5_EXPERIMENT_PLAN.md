# Phase 5 baseline and pilot experiment plan

Status: preregistered developmental contract, 2026-08-20 America/New_York

## 1. Scope and claim boundary

Phase 5 measures whether the approved local data and Phase 4 model stack support a
bounded baseline comparison and pilot training run. It does not access test metrics,
freeze the golden suite, run the main model, claim generalization, or authorize
publication, inference, or deployment.

Only `iid_train` may fit a model, tokenizer-dependent baseline, vocabulary, rule
fallback, or class prior. Only `iid_validation` may choose a checkpoint or inform the
Phase 6 acceptance thresholds. `iid_test`, template, component, severity, composition,
counterfactual, and noise splits are prohibited experiment inputs during this phase.
The complete approved candidate may be structurally checksum-verified, but prohibited
records must be discarded before experiment examples are materialized and must never
enter a metric or selection function.

## 2. Fixed task serialization

Each supervised record uses exactly one approved rendered prompt and its typed target.
The record is serialized as:

```text
<|prompt|>
TASK=<task_name>
<approved rendered prompt>
<|target|>
<canonical compact JSON target>
<|sep|>
```

The target JSON is generated from the strict typed target, never reconstructed from
prose. Transformer loss applies only to target and separator tokens. If a record
exceeds model context, retain the beginning-of-sequence marker and the most recent
prompt suffix needed to fit the complete target. Never truncate the target. Record the
number of truncated prompts. Pair/counterfactual tasks are absent from the permitted
train/validation inventory in this developmental candidate and are not fabricated.

## 3. Required baselines

All baselines use the same permitted train/validation examples and prompt text.

1. **Majority/frequency:** most frequent exact training label independently for fault
   family, next action, and next event.
2. **Deterministic keyword/rule:** a fixed, source-controlled ordered phrase rule set
   for fault family and next action with a training-majority fallback. Rules cannot be
   learned from validation data or inspect target fields at inference.
3. **N-gram:** an order-3 add-0.1 token language model for target-token NLL/perplexity,
   plus a suffix n-gram next-event classifier with a training-frequency fallback.
4. **Bag-of-words logistic regression:** project-implemented deterministic multiclass
   softmax regression for fault-family identification, at most 2,048 training-derived
   features, 400 full-batch steps, learning rate 0.25, and L2 0.001.
5. **GRU:** project-defined one-layer 64-dimensional token embedding and 96-dimensional
   hidden state, trained separately for fault family and next event for 60 epochs with
   batch 16 and learning rate 0.005. The Phase 5 compute probe makes this comparison
   feasible, so it may not be omitted.
6. **Smaller Transformer:** the Phase 4 smoke width/depth architecture, extended only
   to the Phase 5 fixed context of 512 tokens (724,480 parameters), trained on the same
   serialized multitask records for 300 steps.
7. **Pilot Transformer:** the Phase 4 pilot width/depth architecture, extended only to
   the Phase 5 fixed context of 512 tokens (5,394,432 parameters), trained from random
   initialization for 500 steps.

Simple baselines are allowed to outperform learned models. Every measured result is
preserved.

## 4. Metrics and model selection

Classification reports exact accuracy, macro-F1, per-label support, and the complete
confusion matrix. Language modeling reports target-token NLL and perplexity. Efficiency
reports exact parameter count, selected step, optimization time, scored tokens/second,
checkpoint size, and observed process/MPS memory.

Transformer checkpoints are evaluated every 50 steps. The selected checkpoint is the
lowest finite `iid_validation` target-token NLL; ties select the earlier step. No test
record or test metric may influence training length, hyperparameters, checkpoint
selection, or thresholds. Training always runs the fixed step count so the benchmark
does not use validation as an implicit stopping-time search.

## 5. Compute contract

The pre-implementation MPS probe used the exact 5,328,896-parameter architecture,
batch four, context 256, and ten timed optimization steps. It measured approximately
15,012 target tokens/second, 150,871,040 current allocated bytes, and 498,958,336 driver
allocated bytes against a 12,713,115,648-byte recommended maximum. This is a sizing
probe, not a reportable pilot result.

### Pre-run sizing correction

The first local Phase 5 invocation stopped before baseline fitting or optimization
because complete canonical targets did not fit the preregistered Phase 4 contexts. The
approved train inventory has a maximum complete target length of 299 tokens; 220 of
630 targets are at least 128 tokens and 10 are at least 256. The validation inventory
has the same 299-token maximum; 88 of 252 targets are at least 128 and 4 are at least
256. No run directory was produced.

To preserve the non-negotiable complete-target rule, this plan fixes both Phase 5
Transformer contexts at 512 tokens while leaving width, depth, optimizer, seeds, and
step counts unchanged. This changes only learned positional-embedding parameters:
the smaller tier is 724,480 parameters and the pilot tier is 5,394,432 parameters.
The correction was recorded before any Phase 5 baseline or training result existed.

The experiment requests MPS and may fall back to CPU only if MPS becomes unavailable or
an explicitly recorded backend error occurs. The report must name the actual backend.
MPS timing synchronizes before and after the measured loop. Observed current/driver MPS
memory is sampled during training; CPU peak RSS is recorded for process context. Thermal
throttling is not directly observable and must remain a limitation rather than an
invented measurement.

## 6. Phase 5 acceptance gate

Phase 5 closes locally only when:

- the Phase 4 smoke artifact and approved Phase 3 candidate reverify;
- split isolation and canonical task serialization tests pass;
- all seven registered comparison rows complete, including both GRU tasks;
- both Transformer runs have finite losses and improve validation target NLL by at
  least 2% from initialization;
- checkpoint selection is demonstrably validation-only and save/reload verification
  passes;
- runtime, throughput, memory, parameter count, and checkpoint size are measured;
- report/config/dependency/data/tokenizer/checkpoint relationships are checksummed;
- the full repository quality, build, and isolated-artifact gates pass; and
- the measured pilot evidence is used to recommend and freeze Phase 6 thresholds before
  any main or test evaluation begins.

A failed row or threshold remains a negative result; it cannot be deleted or repaired by
accessing a test split. Exact Phase 6 thresholds are intentionally not set in this
preregistration because the authoritative acceptance plan requires pilot evidence
first.

## 7. Security and reproducibility

The Phase 5 CLI accepts only the reviewed project-relative configuration in the current
checkout, writes a single non-overwriting run below `runs/`, and loads only the
config-selected approved candidate and Phase 4 tokenizer. It accepts no arbitrary
checkpoint, dataset, URL, or model path. Model weights remain safetensors; JSON is
strict, duplicate-key rejecting, finite, bounded, and checksum verified. No credentials,
network service, push, publication, or deployment is part of Phase 5.
