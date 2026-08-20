# Security controls and verification map

Status: dataset and Phase 4 local model-artifact controls are implemented and verified.
No service, browser route, credential, deployment, or public release exists.

Controls are **documented**, **planned**, **implemented**, or **verified** only when the
recorded evidence supports that exact scope. Verification does not make the project
“fully secure.”

| Control | Current local evidence | Next required gate |
|---|---|---|
| Synthetic-only boundary | Visible disclaimer; project-authored generator/renderer; owner-approved candidate; non-exhaustive content scans | Preserve scope during Phase 5 task serialization and human-review final golden cases |
| Strict contracts | Frozen unknown-field-rejecting generator, dataset, tokenizer-config, model-config, checkpoint, and report models; duplicate-key/non-finite JSON rejection | Freeze only after pilot evidence and human review |
| Truth isolation | Renderer sees only `ModelInput`; tokenizer sees only approved `iid_train` prose; model sees token IDs, never latent/injection/target provenance | Retest every new training/task view |
| Split integrity | Split-first assignment; atomic groups; tokenizer corpus explicitly filters to `iid_train`; unit test proves holdout exclusion | Validation-only checkpoint selection and frozen test policy before pilot/main evaluation |
| Determinism | Local seeded RNG; fixed tokenizer ordering/prefix/threading; explicit dropout/config; CPU deterministic smoke algorithms | Measure MPS nondeterminism and preserve seeds/configs during Phase 5 |
| Model initialization | Project-defined architecture and random initialization; no pretrained/model-hub/API code path | Audit Phase 5 training entrypoints for the same boundary |
| Checkpoint safety | Fixed manifest+safetensors inventory; checksum/size/tokenizer/config/source bindings; no pickle; no user checkpoint input | Add release allowlist/signature policy before public inference |
| Path safety | Strict canonical project-relative paths; symlink rejection; config-selected candidate/run directories; atomic non-overwriting writes | Keep future evaluation/service paths allowlisted and server controlled |
| Resource limits | Tokenizer/model/checkpoint/input/report byte ceilings; bounded vocab, context, generation length, steps, and run inventory | Add measured CPU/MPS memory and inference concurrency/time limits |
| Error safety | Narrow Phase 4 CLI catches expected failures and emits bounded messages without tracebacks | Apply equivalent errors at the Phase 7 HTTP boundary |
| Dependency provenance | Frozen `uv.lock`; smoke report binds lock hash and records NumPy/Pydantic/Torch/SentencePiece/safetensors versions | Dependency review, SBOM, CodeQL, and secret scanning before release |
| Artifact verification | Full suite, coverage floor, package build, isolated wheel install, tokenizer/checkpoint tamper tests, independent smoke verifier | Verify pilot/main artifacts and release checksums in clean environments |
| Browser/service safety | No browser, account, service, routes, arbitrary input, or credentials exist | Strict request schemas, limits, CSP, safe rendering, timeout/rate tests in Phase 7 |

## Phase 4 artifact controls

The tokenizer writer publishes only `tokenizer.model`, `tokenizer.vocab`, and
`manifest.json` after successful training and checksum construction. The loader rejects
unexpected inventory, symlinks, oversize, checksum/size mismatch, vocabulary mismatch,
and unknown-token output despite byte fallback.

The checkpoint writer clones tensors to contiguous CPU storage and writes
`model.safetensors`; no optimizer/Python object or pickle is serialized. The loader
requires the expected manifest and tokenizer checksums, verifies sizes and file hashes,
allocates only the manifest-declared bounded architecture, and uses strict state loading.
The public Phase 4 CLI cannot accept a checkpoint, tokenizer, output root, URL, or model
path from the caller.

The smoke run is bound to the approved Phase 3 candidate, review record, `iid_train`
corpus hash, tokenizer manifest, source commit, reviewed config, exact `uv.lock`, input
tensors, checkpoint, and evaluation logits. A separate read-only command reconstructs
and verifies that chain.

## Local evidence and limits

- Phase 4 implementation commits:
  `882c8d9c13a62373f45080864f595748e2bc1db9` and
  `f636f2b2b9f7f5915bef78903611aa047aaecc30`.
- Python 3.12.11; 677 tests passed in 300.44 seconds with 85.37% branch coverage.
- Ruff format/lint and strict mypy for 96 source files passed.
- Wheel/sdist and isolated local artifact verification passed.
- Independent final smoke verification reproduced the evaluation-logit checksum.
- Safetensors size limit: 512 MiB; final smoke weights: 3,752,704 bytes.
- Tokenizer model limit: 64 MiB; final tokenizer model: 33,586 bytes.
- Smoke input limit: 8 MiB; final input artifact: 4,824 bytes.
- Context/config maximums are strict; the smoke run uses context 128.

PyTorch reports MPS support compiled in but unavailable inside this process. CPU smoke
throughput is measured; MPS memory, speed, and throttling are not. The content guard is
non-exhaustive, and tiny-shard overfit is not a safety or generalization result.

## Remaining gates

- Phase 5: preregister baselines, validation-only selection, pilot limits, backend
  measurements, and stop conditions before training.
- Phase 6: freeze golden/test manifests and acceptance thresholds before main runs.
- Phase 7: implement and test request/response schemas, authentication choice, rate/
  duration/concurrency/memory limits, CSP, safe rendering, and failure behavior.
- Phase 8: choose licenses, produce SBOM/provenance/security evidence, and verify release
  artifacts. The owner alone may push or deploy.
