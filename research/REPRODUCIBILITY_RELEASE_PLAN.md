# Reproducibility, artifacts, and release plan

Status: pre-implementation contract

## 1. Reproduction levels

### Level 1 — verification

Fast checks that require no training:

- validate schemas and manifests;
- verify published checksums;
- load tokenizer and checkpoint;
- run fixed golden inputs;
- compare expected structured outputs.

### Level 2 — smoke reproduction

One documented command, such as `make reproduce-smoke`, should:

1. generate a small deterministic dataset;
2. train a tiny model for a short run;
3. prove tiny-shard overfitting;
4. run representative evaluation;
5. write a miniature report and checksums.

This command must run on an ordinary development machine in a practical amount of time measured during implementation.

### Level 3 — main experiment

Document exact data generation, tokenizer training, model training, checkpoint selection, and evaluation commands. The full run may require substantial time, but every input configuration and artifact relationship must be available.

## 2. Artifact chain

```text
Git commit
  → generator version and configuration
  → trajectory and split manifests
  → rendered dataset manifest
  → tokenizer configuration and checksum
  → model/training configuration
  → checkpoint and checksum
  → evaluation configuration
  → metrics and prediction artifacts
  → figures and report
  → deployed release identifier
```

Every arrow must be machine-verifiable through identifiers or checksums.

## 3. Required metadata

- repository commit and release tag;
- environment and dependency lockfile hash;
- random seeds;
- generator, schema, renderer, and split versions;
- dataset counts, token counts, and checksums;
- tokenizer algorithm, vocabulary size, special tokens, and checksum;
- model architecture, exact parameter count, and initialization settings;
- optimizer, schedule, batch/token accumulation, and stopping rule;
- hardware, backend, runtime, peak memory, and interruptions;
- checkpoint-selection criterion based on development data only;
- evaluation code/configuration version;
- deployed model, scenario, and API schema versions.

## 4. Configuration policy

- Store experiment settings in reviewed configuration files rather than scattered command-line defaults.
- Save the resolved configuration with every run.
- Prevent test-set metrics from controlling checkpoint selection.
- Record intentional deviations from the preregistered plan.
- Use fixed seeds where determinism is supported and document unavoidable nondeterminism.
- Do not overwrite prior run directories or release artifacts.

## 5. Storage policy

- Commit schemas, source, small safe samples, configurations, manifests, and reports to Git.
- Keep large generated corpora, optimizer states, and large checkpoints outside normal Git history.
- Publish only artifacts whose size, license, safety boundary, and checksum have been reviewed.
- Use immutable or versioned release locations.
- Never load a release artifact without matching its recorded checksum.

## 6. Performance and cost record

For every serious training tier, record:

- parameter and token count;
- steps and effective tokens per update;
- wall-clock duration;
- tokens per second;
- peak memory;
- checkpoint and tokenizer size;
- local or hosted compute type;
- estimated or actual compute cost;
- inference cold start, warm latency, throughput, and memory;
- hosting cost assumptions and observed usage where available.

Do not report estimated values as measurements.

## 7. Release contents

Each tagged public release should include or link to:

- source revision;
- changelog;
- generator, schema, and dataset version;
- tokenizer and permitted checkpoint artifacts;
- model card and dataset card;
- experiment/results report;
- checksums and provenance manifest;
- software bill of materials;
- known limitations and security notes;
- reproduction commands;
- deployment version when applicable.

## 8. Repository presentation

The root README should provide:

- one-sentence contribution;
- live-demo and documentation links;
- concise architecture diagram;
- measured headline results with split names;
- explanation of what was trained from scratch;
- safe smoke-reproduction instructions;
- hardware and runtime summary;
- visible synthetic/non-operational limitation;
- links to dataset, model, results, security, and citation documents.

Add a short demonstration video or animation only after the UI and results are stable.

## 9. Paper-style report

Prepare a concise technical report after main evaluation:

1. abstract;
2. research question and contribution;
3. related work;
4. Aster Station generator and dataset;
5. model and baselines;
6. experimental protocol;
7. results;
8. error and robustness analysis;
9. limitations, ethics, and safety boundary;
10. reproducibility statement.

No publication claim should precede completed experiments.
