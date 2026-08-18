# Focused literature review and project positioning

Status: research synthesis; no external source is approved as training-corpus text
Reviewed: 2026-08-18

## 1. Research question

ReactorBench-LM asks whether a small decoder-only Transformer, trained from random initialization on project-authored fictional plant-event language, can learn causal sequence structure and generalize to deliberately unseen combinations of faults.

This combines four lines of work: educational simulation, small language-model training, compositional generalization and behavioral evaluation, and AI research in nuclear contexts. The combination—and the controlled benchmark built around it—is the proposed contribution.

## 2. Simulator scope and abstraction

IAEA TECDOC-1887 surveys nuclear-reactor simulation and modeling for education and training, while IAEA Training Course Series No. 70 describes simulator exercises used in educational contexts. These sources show why simulator purpose, modeled scope, assumptions, and learning objectives must be explicit.

ReactorBench-LM deliberately occupies a narrower category than those systems. Aster Station is a software state machine for producing controlled language experiments. It omits real component parameters, reactor physics, protection logic, operating procedures, and facility-specific behavior. Its normalized transition rules are internally consistent inventions rather than low-fidelity engineering claims.

Design consequence: external simulator materials inform the boundary and documentation approach, not coefficients, scenarios, prose, or data.

## 3. Training a small Transformer from scratch

The original Transformer work establishes attention-based sequence modeling. ReactorBench-LM uses its decoder-only causal form and implements the model with randomly initialized project-defined layers. SentencePiece provides a reproducible way to train a tokenizer on the synthetic corpus without relying on a pretrained tokenizer.

TinyStories demonstrates that deliberately constrained synthetic text can support coherent behavior in small language models. ReactorBench-LM borrows the experimental lesson—not its data-generation pipeline—that small models benefit from a bounded world, controlled vocabulary, and measurable concepts. Unlike TinyStories, the core corpus here will not be generated or judged by a larger hosted language model.

Chinchilla-style scaling work establishes that model size, token budget, and compute must be considered jointly. Its precise large-scale optimum should not be transplanted to a small Apple-silicon experiment. The relevant lesson is to use pilot learning curves and measured throughput before freezing the main model or corpus size.

Design consequences:

- train from random initialization;
- build a project-specific tokenizer;
- start with smoke and pilot tiers before the approximately 15M-parameter target;
- publish parameter counts, token counts, training curves, and compute measurements;
- compare against simple non-neural and smaller sequence baselines.

## 4. Synthetic data and compositional generalization

Synthetic sequence tasks are valuable when the generator exposes latent factors and exact ground truth. Recent work on compositional generalization in Transformers reinforces that performance on familiar combinations does not demonstrate systematic recombination. Random example splits are therefore inadequate for this project: shared templates, component aliases, and fault combinations can leak the answer.

ReactorBench-LM will generate split manifests before narrative rendering and reserve:

- fault-pair combinations;
- subsystem/fault role assignments;
- component-alias families;
- narrative template families;
- selected plant-variant dependency patterns.

The primary comparison is the gap between ordinary IID performance and strict compositional holdouts. A negative result remains useful if the experiment shows where generalization fails.

## 5. Behavioral evaluation

CheckList argues for testing NLP systems through capabilities and test types rather than relying only on aggregate held-out accuracy. Its minimum-functionality, invariance, and directional-expectation ideas directly inform the golden suite.

ReactorBench-LM adapts that approach as follows:

- **minimum functionality:** stable operation, sensor faults, component faults, and required abstention;
- **invariance:** aliases, harmless value perturbations, reordered simultaneous observations, and unseen paraphrase families should preserve valid answers;
- **directional expectation:** adding decisive evidence should resolve uncertainty or flip a near-neighbor label in a declared direction;
- **counterfactual pairs:** valve lag versus valve stuck, sensor fault versus latent process fault, standby available versus unavailable;
- **coverage-risk analysis:** measure whether confidence-based abstention reduces error rather than merely reducing coverage.

Structured ground-truth metrics remain primary: fault macro-F1, exact match for compound labels, evidence-span or evidence-event F1, action-label accuracy, next-event accuracy, sequence validity, calibration, and the IID-to-compositional performance gap. Text fluency is secondary.

## 6. Prior AI work in nuclear contexts

The field is not new. A 1991 DOE/OSTI record describes artificial-neural-network diagnostics coupled with a nuclear power plant simulator, establishing long-standing precedent for learned status diagnosis. More recently, Idaho National Laboratory researchers reported NLP methods for processing real operator logs. IAEA's 2025 publication on artificial intelligence for nuclear applications discusses opportunities, lifecycle considerations, data, validation, transparency, human factors, and governance.

These sources mean that “AI for a nuclear plant” alone is not a distinctive claim. ReactorBench-LM should not present itself as the first nuclear-language or diagnostic model. Its defensible differentiation is the combination of:

1. a fully disclosed, wholly fictional causal state generator;
2. a small decoder-only Transformer trained from scratch rather than an API application;
3. paired structured and natural-language dataset views with known latent ground truth;
4. strict held-out fault composition and renderer-family evaluation;
5. explicit abstention and behavioral counterfactual tests;
6. a public, reproducible portfolio scope designed to avoid real operational data and claims.

The project is closer to a controlled ML benchmark in a fictional industrial world than to an operator-log product, diagnostic tool, or plant simulator.

## 7. Documentation and governance

The NIST AI Risk Management Framework emphasizes documented measurement, uncertainty, limitations, and evaluation in relevant conditions. Model Cards and Datasheets for Datasets provide practical reporting patterns for intended use, performance breakdowns, data provenance, collection or generation processes, and known limitations.

The eventual release should therefore include:

- a dataset card describing generator versions, schemas, split logic, provenance, licenses, exclusions, and known artifacts;
- a model card stating intended and prohibited uses, hardware, training data, metrics by split and capability, calibration, and limitations;
- an experiment report with preregistered comparisons, negative results, random seeds, and confidence intervals;
- a source manifest that distinguishes conceptual references from corpus content;
- conspicuous wording that outputs are valid only inside the Aster Station fictional world.

## 8. What the evidence does not establish

The reviewed literature does not establish that:

- performance in Aster Station transfers to any real facility;
- a small language model can perform licensed operator or engineering work;
- synthetic prose faithfully represents real operational communication;
- normalized rules represent real thermohydraulics or safety response;
- high IID accuracy demonstrates causal or compositional reasoning;
- an internally consistent generated answer is safe or useful outside the benchmark.

Those are explicit non-claims, not future marketing language.

## 9. Resulting pre-implementation commitments

- Keep every external source `REFERENCE_ONLY` or governance-only unless an isolated term is explicitly approved.
- Freeze structured state, causal rules, and split manifests before generating prose.
- Keep the golden scenarios outside training and use human-approved expected answers.
- Measure false positives on normal transients and required abstention, not only fault recall.
- Report IID, lexical holdout, structural holdout, and compositional holdout results separately.
- Compare model results with deterministic, frequency, and lightweight learned baselines.
- Treat success as evidence about this benchmark only.

## References

- Vaswani et al., *Attention Is All You Need*: https://arxiv.org/abs/1706.03762
- Eldan and Li, *TinyStories*: https://arxiv.org/abs/2305.07759
- Hoffmann et al., *Training Compute-Optimal Large Language Models*: https://arxiv.org/abs/2203.15556
- Kudo and Richardson, *SentencePiece*: https://aclanthology.org/D18-2012/
- Ramesh et al., *Compositional Capabilities of Autoregressive Transformers*: https://arxiv.org/abs/2311.12997
- Ribeiro et al., *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*: https://aclanthology.org/2020.acl-main.442/
- IAEA, *Use of Nuclear Reactor Simulators for Education and Training* (TECDOC-1887): https://www-pub.iaea.org/MTCD/Publications/PDF/TE-1887_web.pdf
- IAEA, *Nuclear Reactor Simulators for Education and Training* (Training Course Series No. 70): https://www-pub.iaea.org/MTCD/Publications/PDF/TCS-70web.pdf
- DOE/OSTI, *Artificial neural networks for nuclear power plant status diagnostics*: https://www.osti.gov/biblio/10104399
- INL/OSTI, *Natural Language Processing on Nuclear Power Plant Operator Logs* record: https://www.osti.gov/biblio/2583138
- IAEA, *Artificial Intelligence for Nuclear Applications* (2025): https://www-pub.iaea.org/MTCD/publications/PDF/p15866-PUB2119_web.pdf
- NIST, AI RMF Core: https://airc.nist.gov/airmf-resources/airmf/5-sec-core/
- Mitchell et al., *Model Cards for Model Reporting*: https://research.google/pubs/model-cards-for-model-reporting/
- Gebru et al., *Datasheets for Datasets*: https://www.microsoft.com/en-us/research/publication/datasheets-for-datasets/
