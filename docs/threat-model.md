# Threat model

Status: Phase 1 foundation review, version 0.1
Verification status: selected local foundation controls are verified at the 2026-08-18 checkpoint; no production or deployment control has been verified

## Scope and assumptions

This model covers the local research pipeline and the intended future narrow inference and Research Editorial application. It treats browsers, request payloads, rendered/model text, dependency inputs, generated artifacts, and externally stored secrets as untrusted until a specific boundary validates them.

Assumptions:

- all permitted domain content is project-authored and fictional;
- no public service exists during Phase 1;
- the future public interface exposes curated identifiers and bounded controls, not arbitrary prose, files, URLs, or artifact paths;
- ground truth is created only by the versioned state generator;
- model checkpoints originate only from reviewed project runs; and
- the project owner performs external publication and deployment separately.

This threat model does not authorize testing third-party infrastructure, real facilities, or public denial-of-service behavior.

## Assets

- integrity of the Aster Station state, observation, event, scenario, and target contracts;
- confidentiality of future service credentials and deployment configuration;
- integrity and availability of datasets, tokenizers, checkpoints, manifests, and evaluation results;
- isolation of test data and latent ground truth from training/model inputs;
- availability of the future bounded inference service;
- safety and privacy of site visitors; and
- credibility of claims linking displayed results to measured artifacts.

## Trust boundaries

```text
[Research source and reviewed configuration]
                 |
      B1: schema/config validation
                 v
[Generator and local artifact pipeline]
                 |
      B2: manifest/checksum boundary
                 v
[Fixed tokenizer/model/evaluation artifacts]

[Untrusted browser]
                 |
      B3: HTTPS + server-side request validation
                 v
[Gateway]
                 |
      B4: authenticated, allowlisted service request
                 v
[Inference service]
                 |
      B5: pinned checksum and safe artifact loader
                 v
[Read-only released artifacts]
```

Boundaries B3 through B5 are future design boundaries, not implemented deployment claims.

## Risk register

Status values mean **documented**, **planned**, **implemented**, or **verified**. Nothing is verified without recorded evidence.

| ID | Threat and impact | Primary mitigation | Current status | Residual risk or gate |
|---|---|---|---|---|
| TM-01 | Real, restricted, Navy, or facility-specific content enters data and creates safety, legal, or provenance harm | Synthetic-only policy, no ingestion endpoint, denylist/pattern scan, provenance, stratified human review | Documented; automation planned | Denylists cannot prove absence; release review required |
| TM-02 | Hidden target or scenario identifiers leak into model input, inflating results | Separate latent/observation/event contracts, split-first manifests, leakage and contamination tests | Foundation contract isolation implemented and verified; renderer/model-input contamination tests remain Phase 3 work | Semantic leakage may survive string deduplication |
| TM-03 | Unknown fields, coercion, NaN/infinity, invalid bounds, or ambiguous targets bypass a data contract | Strict immutable model instances under an explicitly unfrozen developmental interface, `extra='forbid'`, finite bounded values, explicit target invariants | Internal foundation contracts implemented and verified; public/cross-language API parity remains later work | Cross-language API schemas require later parity testing |
| TM-04 | User-controlled paths, files, URLs, or serialized objects lead to traversal, SSRF, code execution, or unsafe deserialization | No such input fields; project-relative outputs; fixed artifacts; data-only serialization where feasible | Local configuration path/overwrite boundary implemented and verified; artifact loading and service boundary remain planned | Local developer privileges remain outside sandbox guarantees |
| TM-05 | Oversized, deeply nested, repeated, or expensive requests exhaust service resources | Pre-parse byte cap, depth/schema limits, token/output/time/rate/concurrency caps | Planned after benchmarks | Distributed abuse and platform limits remain |
| TM-06 | Malicious or malformed model/rendered text causes script injection or unsafe links | Structured allowlisted output, framework text escaping, no raw HTML, CSP | Planned with UI | Browser/framework defects and unsafe future component usage |
| TM-07 | Secrets appear in source, client bundles, logs, fixtures, screenshots, or CI | Server-only variables, environment separation, redaction, secret scanning, rotation procedure | Planned; no service secrets currently required | Platform/operator mistakes remain possible |
| TM-08 | Checkpoint replacement, corruption, or unsafe loading executes code or changes results | Trusted source, canonical manifest, SHA-256 at startup/release, read-only artifact, safe format | Planned for model phase | Hashes protect integrity only when reference hashes are trusted |
| TM-09 | Dependency or CI compromise changes source or artifacts | Minimal pinned dependencies, lockfile, immutable CI actions, review, CodeQL/dependency/secret scanning, SBOM | Foundation dependency policy selected; CI planned | Upstream and maintainer compromise cannot be eliminated |
| TM-10 | Dataset split contamination or nondeterministic regeneration invalidates evaluation | Frozen manifests, seeds, canonical serialization, versioning, cross-split checks, immutable run directories | Canonical foundation serialization, hashing, schema snapshots, and non-overwriting run directories implemented and verified; split and dataset checks remain planned | Hardware/library nondeterminism must be measured and documented |
| TM-11 | UI metrics refer to a different model/schema or are fabricated placeholders | Artifact lineage, result manifests, explicit versions/checksums, production rejection of placeholders | Documented; implementation planned | Human transcription and stale caching require release checks |
| TM-12 | Logs collect visitor identifiers, request bodies, or credentials unnecessarily | Curated inputs, minimal aggregate reliability fields, redaction, retention limits | Planned with service | Hosting providers may retain metadata under their own policies |
| TM-13 | Verbose exceptions disclose paths, source, configuration, or secrets | Safe error mapping, production debug disabled, negative tests | Planned with service | Operational diagnostics require protected access |
| TM-14 | Project output is mistaken for real advice or used outside the fictional world | Prominent exact disclaimer, structured fictional labels, curated interface, no real input path | Disclaimer implemented in documentation; UI gate planned | Public misunderstanding cannot be fully prevented |
| TM-15 | Local or public release accidentally publishes large/private/unreviewed artifacts | Explicit allowlist, license/safety/checksum review, `.gitignore`, release checklist | Local policy documented; release automation planned | Maintainer can override safeguards; manual review remains required |

Foundation evidence recorded on 2026-08-18 under Python 3.12: `make check`
passed, a separate coverage run measured 91.41%, and `make build` passed. This
checkpoint supports only the scoped local statuses above; it does not verify the
generator, dataset, model, service, public interface, or deployment controls.

## Abuse cases to test later

- unknown and duplicate JSON keys;
- boolean or string coercion into numeric fields;
- NaN, infinity, out-of-range normalized values, extreme seeds, and invalid enum values;
- malformed, truncated, deeply nested, oversized, or wrong-content-type requests;
- attempts to choose a URL, file, checkpoint, tokenizer, model, device, or execution limit;
- HTML-like text, control characters, and bidirectional text in display fields;
- version or checksum mismatch;
- authentication failure between future gateway and inference service;
- timeouts, concurrency exhaustion, and safe rate-limit behavior; and
- missing evidence that requires abstention rather than a diagnosis.

Tests must use bounded local or authorized preview environments. They must not perform uncontrolled load testing against shared public systems.

## Review gates

- Review schema threats before the structured generator gate.
- Review leakage and provenance threats before any pilot dataset is rendered.
- Review safe artifact loading before checkpoint persistence becomes a supported workflow.
- Re-run the service and web threat model before exposing a network listener.
- Verify actual headers, secrets, error behavior, limits, CORS, and client bundles before the owner deploys anything.
- Review the risk register and residual risks for every release-ready checkpoint.

See [security-controls.md](security-controls.md) for the control-to-evidence map.
