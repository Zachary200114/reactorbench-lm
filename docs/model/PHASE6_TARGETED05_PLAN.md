# Phase 6 targeted-05 task-weighted remediation

Status: **preregistered implementation in verification; training not started**

## Diagnosis

Targeted-04 completed 2,500 training steps and all 427 IID development-gate
predictions, then stopped correctly at stage 10. It passed eight of ten unchanged
checks. Fault-comparator margin was `-0.0723480` against `>= 0.02`, and continuation
macro-F1 was `0.8933333` against `>= 0.90`.

The diagnosed-fault-only extra row changed the fault-task training mix. Targeted-03's
hierarchical fault draw was 50% unresolved, 10% no-fault, and 40% diagnosed. With the
targeted-04 extra diagnosed row, the effective two-row fault mix became 25% unresolved,
5% no-fault, and 70% diagnosed. On the 427-row gate, targeted-04 recovered all five
support-power cases and all ten flow-imbalance cases, but it introduced seven false
diagnoses on 29 unresolved cases and new pump/sensor confusions. Its fault macro-F1
fell from `0.8707354` to `0.8328074`.

The checkpoint rule exposed a second problem. The selected targeted-04 checkpoint
reached the aggregate semantic floor, but its fault macro-F1 on the separate 48-row
checkpoint-selection subset was only `0.6875`. Aggregate eligibility therefore hid a
task-level regression before lower validation NLL chose step 2,200.

These are development-only findings. Targeted-04 and its artifacts remain immutable.

## Targeted-05 hypothesis

Restore targeted-03's exact hierarchical class mix and six-task batch, then strengthen
fault and continuation gradients through a checksum-bound objective instead of
changing which classes are sampled. Require both tasks to reach their preregistered
selection floors on the disjoint 48-row checkpoint subset before validation NLL can
break a tie. This should protect unresolved behavior and continuation while giving
the two historically weak tasks additional learning weight.

This is a prospective hypothesis, not a claim that targeted-05 will pass.

## Frozen change

- New run identity: `phase6-remediation-v0.4.0-targeted-05`.
- Policy: `0.3.5-task-weighted`.
- One random-initialized 15,179,520-parameter Transformer candidate.
- Exactly 2,500 optimizer updates and batch size six.
- The exact targeted-03 hierarchical sampler:
  - one row for every task per update;
  - fault rows remain 50% unresolved, 10% no-fault, and 40% diagnosed;
  - diagnosed fault labels, continuation labels, and action labels rotate uniformly.
- Target-token NLL weight `2.0` for `fault_family` and `continue_log`; weight `1.0`
  for every other task. Prompt and padding tokens remain excluded. Loss is normalized
  by active weighted-target-token mass.
- Checkpoint eligibility on the existing 48-row selection partition requires:
  - semantic composite at least `0.75`;
  - fault-family macro-F1 at least `0.90`; and
  - continuation macro-F1 at least `0.90`.
- Eligible checkpoints are still ranked by lower validation NLL, then earlier step.
- The 56-row calibration partition, 427-row gate partition, tokenizer, dataset,
  architecture, optimizer, learning rate, baseline definitions, and all ten scientific
  thresholds are unchanged.
- The checksum-pinned 21-file v0.2 prefix is verified rather than regenerated.

The objective uses only the task identity already attached to IID-training rows. It
does not consume validation labels during gradient updates, targeted-04 predictions,
latent simulator state, final data, golden answers, or real-world plant information.

## Required gates before owner training

- task-weighted loss excludes prompts/padding and binds exact weights;
- the sampler remains bit-equivalent to targeted-03's six-row hierarchical schedule;
- task-level checkpoint floors fail closed when metrics are absent or malformed;
- historical selection policies and artifacts remain loadable and unchanged;
- targeted-05 configuration and package resources are checksum-bound;
- monitor and wrappers select only targeted-05;
- focused/full tests, Ruff, strict mypy, Swift type-check, shell syntax, and a clean
  pipeline dry-run pass.

## Advancement rule

All ten existing v0.3 checks remain mandatory. No threshold may be lowered. If
targeted-05 misses any check, preserve it as negative development evidence and keep
later stages blocked. No final or golden evaluation is permitted.

## Owner command after verification

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
./scripts/open_phase6_progress_gui.sh
```

The monitor must show **Not started** for targeted-05. Run **Readiness check** before
pressing **Start new rerun**. Opening the window alone does not start training.
