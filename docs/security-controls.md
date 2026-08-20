# Security controls and verification map

Status: local Phase 2 generator controls verified on 2026-08-20. No service, UI, deployment, or release control is verified.

Controls are **documented**, **planned**, **implemented**, or **verified** only when the
recorded evidence supports that exact scope. Verification does not make the project
“fully secure.”

| Control | Current Phase 2 evidence | Next required gate |
|---|---|---|
| Synthetic-only boundary | Visible disclaimer and a fixture-tested, redacting prohibited-content scanner | Reviewed full denylist and stratified human samples before pilot rendering |
| Strict contracts | Immutable, unknown-field-rejecting developmental schemas; bounded values; strict enum/container/tick validation | Review/freeze any Phase 3 projection/manifest schema before pilot data |
| Truth isolation | Latent injection, observations, events, targets, and audit payload separate; visible-payload allowlist tested | Prove task prompts exclude audit truth/IDs and post-decision action effects |
| Determinism/integrity | Seeded local RNG, canonical snapshots, non-overwriting configuration/run handling, sdist/wheel verification | Frozen split manifests, checksums, duplicate/skeleton/leakage reports |
| Split integrity | Requirement documented only | Group and test G07, G08/G09, G12, G14, and G15 relatives before renderer/data generation |
| Artifact safety | No checkpoint or user artifact interface exists | Data-only checkpoint selection, hashes, tamper/mismatch tests before model/service use |
| Browser/service safety | No browser, service, credentials, or routes exist | Strict request/response schemas, limits, safe output rendering/errors, CSP, authentication, and tests in Phase 7 |
| Supply-chain/release | Locked local dependencies and package builds verified | CI review/secret scanning/SBOM and release provenance before owner-managed publication |

## Phase 2 evidence

The intended-project full gate after `b63f6d8` passed **434 tests** at **92.20% branch
coverage**. Ruff format/lint and strict mypy for 41 source files passed; sdist/wheel
build plus isolated artifact verification passed. The developmental schema snapshot is
`2a7f8b120658c3414dbc2d4b1935f20274b5eb416cd91e16a8edc2411d01add4`.

The scanner is deliberately narrow: it redacts bounded findings for selected prohibited
patterns. It is not a full real-facility denylist, cannot prove absence of prohibited
content, and is not a substitute for human sample review. Those residual risks remain
open until Phase 3's reviewed denylist and sample-review gate.

## Phase 3 security gate

Before any pilot rendering, implement a read-only task-projection and split-manifest
contract. It must build a prompt only from an allowed prefix ending at the decision tick;
never expose latent truth, injection, scenario/provenance identity, target, audit-only
context IDs, or later `ACTION_APPLIED` consequences; and reject shortcut-bearing fields.
It must group and audit at least these relationships before split assignment:

- G07 matched standby contexts;
- G08/G09 resolution-versus-persistence counterfactuals;
- G12 dependency-map-included versus withheld contexts;
- G14 compound/pump-only/drift-only and component/channel role relations; and
- G15 sparse versus expanded-evidence relatives.

No dataset, tokenizer, model, service, or UI exists. The project owner alone will push
or deploy after later release gates; no external publication happened here.
