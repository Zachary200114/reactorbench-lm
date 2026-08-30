# Phase 6 targeted-04 fault-margin remediation

Status: **preregistered implementation complete; training not started**

## Motivation

The checksum-bound targeted-03 gate replay passed nine of ten unchanged development
checks. Its only miss was fault-comparator margin: the Transformer reached fault-family
macro-F1 0.8707354 while the strongest preregistered simple comparator reached
0.9051555, producing a margin of -0.0344200 against the required +0.0200000.
Targeted-03, its predictions, and its replay certificate remain immutable.

## Hypothesis

Targeted-03 gave fault-family learning one of six rows per update. Its hierarchical
schedule correctly protected unresolved and no-fault behavior but did not provide
enough diagnosed-family exposure to exceed the unusually strong fault comparator.
Adding one label-balanced diagnosed-fault row while retaining the complete six-task
hierarchical anchor may improve diagnosed-family discrimination without starving any
of the nine already-passing behaviors.

This is a prospective training hypothesis, not a claim that targeted-04 will pass.

## Frozen targeted-04 change

- New run identity: `phase6-remediation-v0.4.0-targeted-04`.
- New policy: `0.3.4-fault-boosted`.
- One random-initialized 15,179,520-parameter Transformer candidate.
- Exactly 2,500 optimizer updates, unchanged from targeted-03.
- Seven rows per batch:
  - the exact v0.3.3 six-task hierarchical anchor, including its 50% unresolved,
    10% no-fault, and 40% diagnosed fault-tier schedule; and
  - one additional, distinct fault-family row rotating uniformly across diagnosed
    labels and then examples.
- Every non-fault task retains one row in every update.
- The same training corpus, tokenizer, output contract, architecture, optimizer,
  learning rate, checkpoint interval, 48/56/427 validation partitions, calibration
  grid, semantic floor, comparator definitions, and all ten acceptance thresholds.
- The checksum-pinned 21-file v0.2 prefix is verified rather than regenerated.

The sampler consumes only IID-train task and supervised-label metadata. It does not use
validation outcomes, individual targeted-03 errors, latent simulator state, final
data, golden answers, model predictions, or target text.

## Required gates before owner training

- deterministic sampler replay and exact seven-row shape;
- all six tasks retained in every batch;
- exactly two distinct fault-family rows per batch;
- the added row is always diagnosed and rotates across every diagnosed label;
- malformed inventories, wrong batch size, and missing diagnosed strata fail closed;
- sampling policy/config/binding versions cannot drift;
- targeted-04 config references and package resources are checksum-bound;
- monitor and wrappers select only targeted-04;
- full focused tests, Ruff, strict mypy, Swift type-check, shell syntax, and pipeline
  dry-run pass from a clean committed checkout.

## Advancement rule

All ten existing v0.3 checks remain mandatory. No threshold may be lowered and no
historical result may be overwritten. If targeted-04 still misses fault margin or
regresses another check, it is preserved as another honest development result and
later stages remain blocked.

No final or golden evaluation is permitted by this experiment.

## Owner commands

After the implementation is committed and the worktree is clean:

```bash
cd /Users/zachary/Documents/Personal-Projects/AI-transformer
./scripts/open_phase6_progress_gui.sh
```

The monitor should display **Not started** for
`phase6-remediation-v0.4.0-targeted-04`. Run **Readiness check** first, then
use **Start new rerun**. Opening the monitor alone does not start training.
