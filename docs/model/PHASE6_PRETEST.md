# Phase 6 pre-test freeze and golden review

Status: historical pre-test gate completed; owner approval and one held-out access recorded

Phase 6 may not read model predictions on any held-out split until the exact G01-G15
packet in `golden/golden-suite-v0.1.0.json` has an approved, checksum-bound
`project-owner` record. Automated generation and prohibited-content scanning are not
human approval.

## Frozen inputs

- Golden packet semantic SHA-256:
  `c2e966564dadfab7e8b944ca9b6f8ef59d8545d1da1cc4ea75f8b27a9c44077c`.
- Golden packet raw-file SHA-256:
  `118720638aeb9d082a6ddc7efd367f3d972c5831a12c6b76f171a10076cc64ea`.
- Generator commit bound by the packet: `4473718`.
- Split manifest raw SHA-256:
  `ee01aea896831c90c04e7be324eb05a40341bbc7d752bcf34f9280f7003c8abb`.
- Task-example inventory raw SHA-256:
  `b45e3466a390b31031a3a39b82046cfef17fd0fb159fa85b97405cbe2ff02cc1`.
- Frozen held-out inventory: 894 examples across IID, renderer/template, component,
  severity, composition, counterfactual, and narrative-noise splits.

Every per-split example checksum is recorded in the strict Phase 6 TOML. The evaluator
reconstructs corrupted and paired prompts from the approved Phase 3 graph and rejects
any count, checksum, or foreign-key drift before model scoring.

## Preregistered evaluation behavior

- Main model: 8 layers, width 384, 8 heads, 15,179,520 exact parameters.
- Training: Apple MPS, seed 6601, batch four, 1,500 steps, validation-only selection.
- Decoding: cached greedy decoding, at most 256 generated target tokens, invalid output
  confidence zero, 10-bin ECE, and selective risk at 80% coverage.
- Uncertainty: 2,000 deterministic bootstrap resamples, seed 6602, 95% intervals.
- E0-E7 remain the frozen comparison matrix. E7 is explicitly not applicable because
  the approved IID training split contains no compound rows; it will be reported rather
  than silently replaced with another experiment.
- Composition has no pass threshold. Every split and negative result must be reported.

The tokenizer inspection found counterfactual targets as long as 1,036 tokens, beyond
the 512-token model context. Those records remain in the benchmark. They must be
reported as `INSUFFICIENT_CONTEXT_BY_DESIGN`; the evaluator may not truncate their
ground-truth target and pretend the task was scored normally.

## Owner review contract

For each G01-G15 case, review the scenario, exact decision ticks, diagnosis status,
fault-label set, action sequence, and abstention behavior. Approval confirms that all
cases are synthetic and fictional and contain no real setpoints, units, procedures,
facility topology, service-derived non-public information, or operational guidance.

An approval authorizes local Phase 6 training and held-out evaluation only. It does not
authorize publication, GitHub push, Vercel deployment, or a claim of real-world use.

## Recorded approval and outcome

On 2026-08-20, the project owner approved the exact G01–G15 packet and all seven
confirmations. The strict review record has semantic SHA-256
`1f5307889d259cfb0fa39e86e33ed9c2ce0922742e59af1d5ff5e0c904337288` and raw-file
SHA-256 `9105c6e7e76979fdbc8b4a73d42f323acb636d36affbee13f8147dc23e3f06be`.

Phase 6 then completed one held-out access. The selected checkpoint passed validation
selection but failed 23 behavioral acceptance checks; Phase 7 is therefore blocked.
The original evaluator output, versioned delimiter correction, corrected metrics, and
artifact checksums are reported in [the Phase 6 main report](PHASE6_MAIN.md).
