# ReactorBench-LM architecture

Status: Phases 0–5 implemented and verified locally on 2026-08-20. Phase 6 main
evaluation has not started.

## Design objective

ReactorBench-LM keeps a wholly fictional source of truth separate from observations,
canonical events, rendered text, learned weights, and future model behavior. The state
generator—not prose or a learned model—owns every label.

```text
scenario definition -> latent Aster state -> observations -> canonical events
                    -> audit trajectory -> split-first task projection -> renderer
                    -> approved candidate -> IID-train tokenizer corpus
                    -> project tokenizer -> random-init causal Transformer
                    -> baselines/pilot -> future main evaluation -> future narrow service/UI
```

## Contracts and boundaries

| Layer | Responsibility | Must not do |
|---|---|---|
| Scenario | Select validated fictional variant, driver, fault, context, and actions | Carry real data or infer truth from prose |
| Latent state | Apply bounded deterministic fictional transitions | Model real physics, procedures, or setpoints |
| Observation | Create bounded channels/status/quality from latent state | Reveal latent fault truth |
| Event/target | Record canonical evidence and structured truth | Depend on language rendering |
| Phase 3 projection | Produce decision-tick/channel-limited task inputs | Include audit IDs, truth, later outcomes, or target shortcuts |
| Renderer | Convert only strict `ModelInput` through reviewed authored catalogs | Accept raw trajectories, targets, or provenance |
| Candidate artifact | Preserve canonical data, lineage, reviews, and split assignment | Overwrite an existing bundle or imply public release |
| Tokenizer corpus | Select only approved `iid_train` render text | Read validation/test/holdout prose |
| Tokenizer | Train deterministic BPE from the selected project corpus | Load pretrained vocabulary/model assets |
| Transformer | Learn next-token behavior from random initialization | Load pretrained weights or hosted-model behavior |
| Checkpoint | Preserve data-only tensors and strict provenance | Deserialize pickle or accept arbitrary user paths |
| Evaluation | Train-only fitting and validation-only selection with frozen test gates | Use test results for training, checkpoint choice, or threshold changes |
| UI | Future measured presentation | Claim operational utility or render untrusted output as HTML |

The generator, dataset, tokenizer, and checkpoint formats remain developmental
`0.1.0` artifacts, not frozen version-1 releases.

## Generator and data boundary

`src/reactorbench/simulator/` contains developmental G01–G15 fixtures across fictional
Aster-A/B/C cards. It uses local deterministic RNG streams, normalized bounded values,
two fictional channels per variable, and explicit latent/observation/event/target
separation.

Phase 3 assigns scenarios, seeds, renderer families, aliases, compositions, and
counterfactual relatives to splits before prose exists. `ModelInput` is the renderer's
only source contract and excludes source event relationships, evidence annotations,
latent state, injections, targets, provenance, and later action consequences.

The approved local candidate contains 204 trajectories, 1,762 single-input
projections, 14 counterfactual pairs, 553 rendered candidates, 1,776 task examples,
and 18 corruption records. Both project-owner review records are hash-bound. Exact and
structured duplicates fail, evidence targets resolve only to visible prompt-local
facts, group members remain atomic across splits, and nuisance shortcut contingencies
are task scoped. The candidate is approved for local modeling, not public release.

## Tokenizer boundary

`src/reactorbench/tokenizer/` verifies the complete candidate and its post-render owner
approval, then selects only `SplitName.IID_TRAIN`. It records the render-ID/text-hash
inventory, document count, byte count, candidate/manifest/review bindings, and corpus
checksum before training.

SentencePiece trains under a constant local prefix with fixed input order, no shuffle,
one thread, identity normalization, byte fallback, vocabulary 2,048, fixed special IDs,
and three project task separators. Tokenizer output is an atomic, non-overwriting
three-file directory. Loading rejects symlinks, extra files, size/checksum drift,
manifest mismatch, and runtime vocabulary mismatch.

## Transformer boundary

`src/reactorbench/model/transformer.py` defines the decoder-only model directly from
PyTorch primitives:

```text
token embedding + learned position embedding
  -> [pre-norm -> masked multi-head attention -> residual
      -> pre-norm -> GELU feed-forward -> residual] × N
  -> final norm -> tied language-model head
```

The lower-triangular causal mask is explicit. Padding masks apply to attention keys and
loss targets. `shift_next_token_targets` performs exactly one shift; the loss scores
only visible target positions. Initialization uses a forked RNG stream, preserving the
caller's global CPU RNG state.

Exact reviewed configurations are:

| Tier | Layers | Width | Heads | Context | Parameters |
|---|---:|---:|---:|---:|---:|
| Smoke | 2 | 128 | 4 | 128 | 675,328 |
| Phase 4 pilot definition | 6 | 256 | 8 | 256 | 5,328,896 |
| Phase 5 measured pilot | 6 | 256 | 8 | 512 | 5,394,432 |
| Main | 8 | 384 | 8 | 512 | 15,179,520 |

The Phase 6 main tier is frozen by the measured Phase 5 contract.

## Checkpoint and run boundary

`src/reactorbench/model/checkpoint.py` saves cloned CPU tensors through safetensors.
The manifest binds architecture, exact parameter count, tokenizer/corpus/candidate,
source commit, seed, step/loss metadata, weight checksum, and size. Loading accepts only
`manifest.json` and `model.safetensors`, verifies both before allocation, reconstructs
the declared architecture, and uses strict state loading.

`src/reactorbench/training/smoke.py` verifies the approved data chain, trains the
tokenizer, calculates all tier counts, creates a bounded four-document batch, proves
causal masking, performs deterministic CPU optimization, verifies repeated evaluation,
saves/reloads the checkpoint, and requires identical logits. Its report additionally
binds the reviewed config, `uv.lock`, dependency versions, smoke inputs, and evaluation
logits. Output is config-selected, atomic, and non-overwriting under
`runs/phase4-smoke-v0.1.0`.

`src/reactorbench/evaluation/` and `src/reactorbench/training/pilot.py` add the Phase 5
boundary. They materialize only train/validation task records, run fixed baselines,
mask Transformer loss to complete canonical targets, select checkpoints only by
validation NLL, measure MPS resources, and publish two safetensors checkpoints plus a
strict checksum-bound report under `runs/phase5-pilot-v0.1.0`. The verifier reconstructs
the data/tokenizer/config/lock relationships before loading either checkpoint.

## Artifact lineage

```text
Git commit
  -> generator/config/schema
  -> approved split-first candidate and review records
  -> IID-training render inventory and corpus hash
  -> tokenizer config/model/manifest hashes
  -> Transformer/training config and exact parameter count
  -> safetensors checkpoint manifest and weight hash
  -> smoke inputs, logits, report, dependency lock, and source commit
  -> Phase 5 baseline metrics, validation curves, MPS measurements, and checkpoints
  -> frozen Phase 6 configuration -> future test evaluation artifacts
```

The Phase 4/5 reports and hashes are in `docs/model/PHASE4_SMOKE.md`,
`docs/model/PHASE5_PILOT.md`, and `docs/IMPLEMENTATION_STATUS.md`.

## Future service boundary

After model/evaluation gates pass, the browser will call a server-side gateway with
bounded schemas; the gateway will call a narrow inference service; and that service
will load only fixed, checksummed, data-only artifacts. Model text remains untrusted
display data. These are planned Phase 7 boundaries, not implemented deployment claims.
