# Research Editorial UI product requirements

Status: approved visual direction; implementation deferred
Design direction: Research Editorial with restrained technical detail

## 1. Product character

The interface should feel like an accessible research publication connected to a live experiment. It should be calm, precise, and evidence-centered—not cinematic, game-like, or modeled after a control room.

Visual principles:

- warm light editorial surfaces;
- serif research headings with readable modern interface text;
- thin rules and deliberate spacing rather than decorative cards;
- restrained teal, rust, blue, and amber for stable series mappings;
- monospaced text only for event IDs, versions, configuration, and structured output;
- technical details available without overwhelming the first view.

## 2. Information architecture

### Home

- one-sentence contribution;
- research question;
- trained-from-scratch statement;
- synthetic/non-operational boundary;
- launch-lab action;
- measured result preview;
- GitHub and report links.

### Experiment

- guided scenario laboratory;
- plant/scenario selection;
- playback and single-step controls;
- normalized topology and telemetry;
- canonical event stream;
- model output, evidence, confidence, abstention, and fictional action;
- hidden simulator ground truth;
- deterministic seed and version identifiers.

### Challenge

- hidden scenario label;
- visitor prediction before reveal;
- visitor, model, and simulator comparison;
- no leaderboard or gamified nuclear-emergency language.

### Results

- baseline comparison;
- IID versus lexical, structural, and compositional holdouts;
- per-fault metrics and confusion matrix;
- calibration and coverage-risk view;
- model/data-size ablations;
- performance and cost measurements.

### Failure cases

- selected correct and incorrect cases;
- event evidence and ground truth;
- error taxonomy;
- counterfactual comparison;
- concise interpretation.

### Method

- generator architecture;
- data schema and split strategy;
- Transformer architecture;
- tokenizer and training method;
- baseline definitions;
- experiment protocol.

### Documentation

- model card;
- dataset card;
- experiment report;
- secure-engineering evidence;
- limitations and non-claims;
- reproducibility and citation.

## 3. Experiment screen layout

Recommended desktop structure:

```text
Header: project, navigation, model status
Research title and measured model configuration
Controls: plant, scenario, replay, step/play

Main column                         Analysis column
  simplified system state            structured prediction
  normalized telemetry               calibrated confidence
  canonical event stream             evidence IDs
                                      fictional action label
                                      JSON/technical detail
                                      ground-truth reveal

Footer: versions, inference measurement, safety boundary
```

On narrow screens, controls wrap and the analysis follows the telemetry/event content. No essential result may depend on hovering.

## 4. Interaction rules

- First render must show a useful curated scenario.
- Play, pause, previous, next, and reset must be deterministic.
- Changing scenario resets evidence and hides ground truth.
- Ground truth remains separate from model input and hidden until requested.
- Confidence must identify its calibration method or be labeled as an uncalibrated score.
- Evidence IDs link visibly to highlighted events.
- Model comparison uses the same scenario and visible context for every model.
- Share links contain only safe scenario identifiers, version, and seed.
- Controls never permit arbitrary files, URLs, checkpoints, model paths, or unrestricted real-world text.

## 5. Technical mode

Technical mode may reveal:

- structured model input;
- tokenizer output and token count;
- model, tokenizer, schema, and scenario versions;
- top class probabilities;
- evidence-event scores;
- structured JSON response;
- measured inference latency;
- checkpoint checksum prefix;
- link to the exact evaluation/configuration artifact.

It must not expose credentials, internal service paths, private host details, stack traces, or user-identifying logs.

## 6. Required states

- initial curated scenario;
- loading model or scenario;
- ready;
- playing;
- paused;
- correctly completed;
- model abstained;
- invalid constrained selection;
- request rate limited;
- inference timeout;
- inference service unavailable;
- offline/static explanation fallback;
- incompatible model/scenario version;
- ground truth revealed.

The fallback should allow visitors to explore a cached, clearly labeled recorded scenario when live inference is unavailable. It must not pretend that recorded output is live.

## 7. Accessibility and responsive behavior

- Keyboard-operable controls in logical source order.
- Visible focus treatment.
- Semantic headings, navigation, tables, buttons, forms, and status regions.
- Screen-reader summaries for charts and topology.
- Color paired with labels, shapes, or line styles.
- Sufficient contrast for text and status meanings.
- Reduced-motion support.
- No autoplay by default.
- No essential hover-only content.
- Mobile layouts down to approximately 320 CSS pixels without clipped controls or results.
- Charts simplify labels rather than shrink text into unreadability.

## 8. Content integrity

The UI concept currently contains illustrative values. Production must replace every placeholder—including parameter count, training-token count, accuracy, confidence, inference latency, tokenizer size, and checkpoint version—with measured, traceable values.

Every result should identify:

- model version;
- dataset/split version;
- evaluation split;
- sample count where appropriate;
- whether the value is measured, estimated, or unavailable.

Never use fabricated metrics to complete the visual design.

## 9. Security presentation

Add an **Engineering → Security** area only after controls exist. It should link to:

- threat model;
- trust-boundary diagram;
- security control-to-test mapping;
- CI security checks;
- artifact checksums and SBOM;
- vulnerability-reporting policy;
- residual risks.

Avoid vague phrases such as “military-grade,” “unhackable,” or “fully secure.”

## 10. Performance requirements

Freeze numerical targets after the inference pilot. Measure:

- page performance and bundle size;
- cold and warm inference latency;
- model-service memory;
- timeout rate;
- cached-fallback availability;
- accessibility and browser compatibility.

The interface should prioritize fast explanation and scenario use over animation or decorative complexity.
