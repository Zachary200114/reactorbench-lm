# Live demo and GitHub publication plan

Status: pre-implementation deployment contract
Target experience: a polished public application comparable in presentation to the referenced Threat Sequence project

## 1. Publication outcome

The finished project should have two connected public artifacts:

1. **GitHub repository** — source code, model architecture, generator, evaluation, configuration, documentation, selected safe artifacts, and reproducibility instructions.
2. **Live web demonstration** — a Vercel-hosted portfolio interface that lets visitors run curated Aster Station scenarios and inspect how the trained model behaves.

The website is a presentation and inference layer for the project's own trained model. It must not substitute an external LLM API for the core result.

## 2. Recommended deployment architecture

```text
Visitor
   |
   v
Vercel web application
   |-- project explanation
   |-- scenario selector
   |-- event timeline and telemetry visualization
   |-- prediction/evidence/confidence display
   |-- benchmark and model-card pages
   |
   v
Versioned inference API
   |-- loads the frozen ReactorBench-LM checkpoint
   |-- accepts only schema-valid fictional scenarios
   |-- returns structured predictions
   |-- rate limits and logs non-sensitive operational metrics
   |
   v
Project-trained Transformer checkpoint
```

The frontend can be deployed through Vercel's Git integration. The initial inference deployment should remain separate from the presentation layer so model loading, memory, startup time, and Python/PyTorch dependencies can be measured and changed without redesigning the site.

The browser, Vercel gateway, inference service, and model artifacts are separate trust boundaries. The browser never receives the inference credential; the server-side gateway validates and bounds the public request before making an authenticated service-to-service call. Detailed controls and verification requirements are defined in `SECURE_ENGINEERING_PLAN.md`.

Potential inference paths must be selected only after benchmarking:

- **Containerized CPU inference:** recommended first production path for predictable Python and model packaging.
- **Vercel function inference:** consider only if the quantized checkpoint, dependencies, memory, startup time, and execution duration fit the then-current platform limits.
- **In-browser inference:** valuable stretch goal using an exported format and browser runtime, but only after output-equivalence and browser-performance testing.

No hosting vendor for the inference service is fixed during research because pricing and platform limits change.

## 3. Public user experience

### Landing section

- One-sentence research question.
- Clear statement: “Small Transformer trained from scratch on wholly synthetic fictional plant-event sequences.”
- Visible non-operational disclaimer.
- Links to GitHub, model card, dataset card, and results.

### Interactive laboratory

- Select an approved plant variant and curated scenario.
- Play, pause, reset, or step through synthetic event ticks.
- Display normalized telemetry and component states.
- Show the model's next-event prediction, fault-family prediction, supporting evidence, fictional action label, and confidence/abstention result.
- Reveal ground truth only after prediction or through an explicit comparison control.
- Offer a deterministic replay seed so visitors can reproduce a run.

### Research results

- IID versus compositional-holdout performance.
- Baseline versus Transformer comparison.
- Confusion matrix and per-fault metrics.
- Calibration and abstention/coverage plot.
- Model-size and training-token ablations.
- Examples of correct predictions, failures, and counterfactual flips.

### How it was built

- Compact architecture diagram.
- Parameter count, tokenizer size, context length, training tokens, hardware, and runtime.
- Explanation of random initialization and what PyTorch supplied.
- Links to exact configuration and checkpoint provenance.

## 4. Input and output contract

The public demo must accept only:

- curated scenario identifiers bundled with the release; or
- values generated through constrained controls that map directly to the fictional schema.

It must reject free-form real-facility logs, uploads, real plant names, real units, procedures, and attempts to obtain operational advice.

The inference response should be structured and versioned:

```json
{
  "model_version": "reactorbench-lm-v1",
  "scenario_version": "aster-v1",
  "fault_labels": ["SENSOR_DRIFT"],
  "evidence_event_ids": ["evt_004", "evt_006"],
  "action_label": "VERIFY_REDUNDANT_CHANNEL",
  "abstained": false,
  "confidence": 0.82
}
```

The UI converts labels into explanatory fictional-world text. It does not generate or display real instructions.

## 5. Repository and release structure

The future repository should clearly separate:

- `simulator/` — Aster Station structured generator;
- `data/` — schemas, manifests, small sample data, and generation instructions;
- `model/` — Transformer and tokenizer implementation;
- `training/` — configurations and training entrypoints;
- `evaluation/` — metrics, baselines, golden tests, and reports;
- `inference/` — stable prediction interface;
- `web/` — Vercel application;
- `docs/` — research dossier, cards, limitations, and figures.

Large generated corpora, optimizer states, and oversized checkpoints should not be committed directly to normal Git history. Release storage and download integrity must be designed after artifact sizes are known.

## 6. Deployment gates

The site may be developed locally after model interfaces stabilize, but public deployment requires:

- frozen model, tokenizer, schema, and scenario versions;
- successful golden-suite and prohibited-input tests;
- measured inference latency and memory on the chosen host;
- rate limiting, request-size limits, timeouts, and graceful error handling;
- dependency and license review;
- dataset card, model card, limitations, and prohibited-use language;
- no secrets or private paths in the client bundle or repository;
- accessibility, responsive-layout, and basic browser testing;
- a pinned demonstration checkpoint with checksum;
- monitoring that records service health without collecting unnecessary visitor content.
- completed threat model and security control-to-test mapping;
- server-enforced schema, size, token, duration, rate, and concurrency boundaries;
- deployed security-header, CORS, secret-exposure, and safe-error verification;
- checkpoint checksum validation and reviewed release provenance;
- passing static analysis, dependency review, secret scanning, and abuse-focused tests.

## 7. Portfolio standard

The live site should prove the model rather than merely decorate it. A visitor should be able to answer:

- What was trained from scratch?
- What data did it learn from?
- What is the ground truth?
- How does it perform on unseen combinations?
- When does it abstain?
- Where does it fail?
- How can the result be reproduced?

This makes the demo materially stronger than a generic chatbot interface.

## 8. Deferred decisions

- Frontend framework and visualization library.
- Inference host and monthly budget.
- Whether the browser can run a quantized model locally.
- Whether full weights or only a smaller demo checkpoint will be public.
- Domain name and visual identity.
- Authentication, if abuse measurements show it is needed.

These decisions should follow measured model size, latency, artifact size, and traffic expectations.

## 9. Reference

- Vercel deployment documentation: https://vercel.com/docs/deployments
- Vercel Git integration documentation: https://vercel.com/docs/git
- Referenced presentation target: https://threat-sequence.vercel.app/
