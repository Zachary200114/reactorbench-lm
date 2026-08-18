# ReactorBench-LM decision log

## Settled decisions

| ID | Decision | Rationale |
|---|---|---|
| D-001 | The project will center on a trained AI model, not an API wrapper. | The goal is to demonstrate actual model construction and learning. |
| D-002 | Use a decoder-only causal Transformer initialized from random weights. | It directly teaches the architecture underlying GPT-style language models while remaining feasible at small scale. |
| D-003 | Use a fictional civilian nuclear-plant setting. | It connects to Zachary's operational background without producing another cybersecurity project. |
| D-004 | Use a generic pressurized-water-inspired abstraction, not a real design. | Familiar high-level structure makes the data coherent while preventing real-plant claims. |
| D-005 | The narrative corpus will be entirely project-authored and synthetic. | Known ground truth, clear ownership, rigorous splits, and a safer public portfolio. |
| D-006 | Real event reports and real operator logs are excluded from all model inputs and targets. | They contain real facilities, sequences, operational details, corrective actions, third-party content, and sometimes personal information. |
| D-007 | All Navy nuclear information is categorically excluded. | The public project must not rely on service-derived non-public knowledge or invite ambiguity about it. |
| D-008 | Use normalized values or synthetic units only. | Prevents accidental correspondence with real thresholds and keeps claims scoped to the invented world. |
| D-009 | The model may output only structured fictional action labels, not real instructions. | Retains a measurable task without presenting the system as operational guidance. |
| D-010 | PyTorch is permitted for tensors and automatic differentiation. | Building a model does not require reimplementing GPU kernels or gradient engines. The architecture, data, training, and evaluation remain project work. |
| D-011 | Train a project-specific SentencePiece tokenizer for the main path. | Keeps attention on the Transformer and dataset. Custom BPE can be a later comparison. |
| D-012 | The public interface is deferred until after model and dataset gates pass. | Prevents the UI from overshadowing or concealing weak model evidence. |
| D-013 | Frame the artifact as a benchmark/research demonstration. | “Assistant,” “operator,” “digital twin,” and “safety system” would overstate the project. |
| D-014 | Use split-first generation with compositional holdouts. | Random rendered-text splits would leak templates and scenario logic. |
| D-015 | Primary scoring will use structured ground-truth metrics. | Open-ended prose similarity alone cannot establish factual correctness. |
| D-016 | Name the fictional state-machine world `Aster Station` and version its topology, transition rules, and plant variants. | A stable invented world makes labels reproducible while separating the work from real facilities. |
| D-017 | Use the state, observation, rendering, and model layers as separate contracts. | The generator—not generated language or the trained model—must own ground truth. |
| D-018 | Maintain a withheld, human-reviewed suite of 15 golden scenario families. | Explicit expected evidence, actions, counterfactuals, and abstention make behavioral claims auditable. |
| D-019 | Adapt CheckList-style minimum-functionality, invariance, and directional-expectation tests. | Aggregate accuracy alone can conceal shortcuts and failures on decisive evidence changes. |
| D-020 | Position novelty around the disclosed fictional generator, from-scratch model, and compositional benchmark—not “AI for nuclear plants.” | Nuclear diagnostic AI has decades of precedent; a narrower, evidence-based contribution is more credible. |
| D-021 | Publish both a reproducible GitHub repository and a polished live portfolio demonstration. | The repository proves the work while the live application makes the model and results accessible. |
| D-022 | Target Vercel for the web presentation layer and select inference hosting only after model benchmarks. | Frontend deployment and model-serving constraints should remain independently changeable. |
| D-023 | Restrict the public demo to curated or schema-constrained fictional inputs. | Visitors can explore the trained model without turning the site into a real-facility analysis or advice interface. |
| D-024 | Treat security as a documented and tested engineering property across the web, API, artifacts, and CI/CD pipeline. | Concrete controls and evidence are more credible than a general claim that the application is secure. |
| D-025 | Keep browser, Vercel gateway, inference service, and model artifacts as explicit trust boundaries. | Server-side enforcement and narrow interfaces reduce exposure and prevent client-side restrictions from being mistaken for security controls. |
| D-026 | Publish a threat model, security-control mapping, verification results, release checksums, and vulnerability-reporting policy. | Recruiters and reviewers should be able to inspect how security claims were established. |
| D-027 | Keep secure engineering secondary to the AI research narrative. | This demonstrates transferable security discipline without turning the flagship project back into a cybersecurity product. |
| D-028 | Make `PROJECT_REQUIREMENTS.md` the canonical statement of finished-project scope. | The expanded research package needs one place that separates required work, conditional additions, and non-goals. |
| D-029 | Require simple, sequence, and smaller-Transformer baselines before interpreting the main model. | The project must prove what the Transformer adds rather than relying on an isolated score. |
| D-030 | Add a dedicated robustness/OOD suite, formal error taxonomy, and public failure gallery. | Generalization and failure behavior are central evidence, not optional polish. |
| D-031 | Require smoke reproduction and end-to-end artifact lineage for releases. | Reviewers must be able to verify the pipeline and connect deployed results to exact artifacts. |
| D-032 | Adopt Research Editorial with restrained technical detail as the approved UI direction. | It communicates serious AI research without resembling a control room or generic cybersecurity dashboard. |
| D-033 | Treat all current UI metrics as illustrative until replaced by measured release artifacts. | Portfolio presentation must never create unsupported model or performance claims. |
| D-034 | Begin local implementation under the final project name `ReactorBench-LM`; do not push to GitHub or deploy to Vercel. | Zachary explicitly authorized all local project work and reserved external publication/deployment for himself. |
| D-035 | Treat `LOAD_TRANSIENT` as a benign scenario driver/event family, not an injected fault. | The canonical plant contract and G02 both require `NO_FAULT` ground truth for a coordinated load change. |
| D-036 | Represent diagnosis as `DIAGNOSED`, `NO_FAULT`, or `UNRESOLVED`; an unresolved result has no fault labels and uses abstention reason `INSUFFICIENT_EVIDENCE`. | This separates fault identity, diagnostic status, abstention, and action semantics without inventing a new fault. |
| D-037 | Store one immediate action label per decision tick and serialize each trajectory-level action as `{decision_tick, action}` in an ordered sequence. | Task/API scoring remains singular while golden scenarios can express later fictional actions without losing their decision points. |
| D-038 | Keep observation status and channel quality as separate enums. | `NORMAL/WATCH/ABNORMAL/MISSING/CONFLICTING` describes an observation; `GOOD/SUSPECT/UNAVAILABLE/NOISY` describes its channel. |
| D-039 | Treat compound fault labels semantically as a set and serialize them in documented enum order. | Deterministic ordering enables stable hashes and exact-match scoring without making label order meaningful. |
| D-040 | Target Python 3.12 compatibility using `uv` and a PEP 621 package; allow local verification on compatible Python 3.13. | Python 3.12 matches the research compute plan while a bounded compatibility range permits the current machine to run foundation checks. |
| D-041 | Defer code and dataset license selection until release preparation. | Licensing does not block local implementation, and no external publication is authorized in this phase. |

## Provisional decisions

| ID | Current default | What could change it |
|---|---|---|
| P-002 | Main model: about 15M parameters, context 512. | Pilot throughput, memory, learning curves, or task complexity. |
| P-003 | Version 1 corpus: 25M–50M tokens. | Pilot learning curves and measured data diversity. |
| P-004 | Dataset license: CC BY 4.0 or CC0; currently `TBD`. | Desired attribution and downstream-use policy before public release. |
| P-005 | Code license: Apache-2.0 or MIT; currently `TBD`. | Dependency review and desired patent language before public release. |
| P-006 | Optional cloud GPU only after local benchmark. | Measured runtime on the M3/16 GB machine. |
| P-007 | Public demo uses scenario selection only. | A later safety design may permit constrained fictional-world free text. |

## Decisions explicitly deferred until evidence exists

- Exact optimizer hyperparameters and learning-rate schedule.
- Final tokenizer vocabulary size.
- Final context length.
- Main/stretched model size.
- Dataset class proportions and exact token count.
- Whether an LSTM baseline is worth the compute.
- Whether the results justify a poster or paper-style submission.
- Whether to publish model weights.

## Resolved workspace issue

The original directory name had a trailing space. Zachary renamed it to `AI-transformer` before implementation; the research dossier now targets that corrected path.
