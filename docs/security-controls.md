# Security controls and verification map

Status: Phase 1 complete; selected Phase 2 generator controls verified at the 2026-08-20 checkpoint

This file tracks security work as evidence, not aspiration. A control is:

- **Documented** when its requirement and threat are written down.
- **Planned** when an implementation location and verification method are identified.
- **Implemented** when code or configuration exists.
- **Verified** only when a recorded test or inspection has passed against that implementation.

No production or deployment control is verified by the current local milestones. A plan or green tool badge is not proof by itself.

| Control | Threats | Requirement | Intended implementation | Verification evidence | Current status |
|---|---|---|---|---|---|
| SC-001 Synthetic-only boundary | TM-01, TM-14 | Exclude real plant, Navy, restricted, operational, and proprietary content | Generator-only inputs, prohibited-content scanner, visible exact disclaimer | Scanner fixtures, sampled-release review record, UI inspection | Disclaimer and a redacting non-exhaustive Phase 2 scanner are implemented and fixture-tested; reviewed full denylist and human sampling remain pending |
| SC-002 Strict internal schemas | TM-02, TM-03 | Reject coercion, unknown fields, ambiguous targets, non-finite and out-of-range values | Strict Pydantic v2 contracts with immutable instances under the unfrozen `0.1.0` developmental interface | Foundation unit tests plus generator unit/property tests for strictness, bounds, staged ordering, mappings, replay, canonical IDs and enum instances, exact integer ticks, tuple containers, and unsupported driver/fault compositions; G07 tests additionally cover the strict `StandbyContext`, distinct train roles, exact `AVAILABLE`/`UNAVAILABLE` states, the invented train/bus map and one-tick delay, and unchecked `model_copy` lookalikes | Implemented and verified for the current internal-contract and structured-generator scope |
| SC-003 Configuration boundary | TM-03, TM-04, TM-10 | Reject unknown config and unsafe output locations; never overwrite runs | `src/reactorbench/config.py`, reviewed project-relative roots, immutable run snapshots | Unit tests for extra fields, traversal and symlink escape, collision, canonical serialization, and hash | Implemented and verified for the local Phase 1 configuration scope |
| SC-004 Ground-truth isolation | TM-02, TM-10 | Keep latent injection separate from visible observation and task input | Separate state, observation, event, scenario, and target models; an exact structured-payload allowlist | Contract tests reject hidden fault, driver, scenario, severity, onset, action sequence, provenance, numeric health, maintenance state, and `NOISY` shortcut fields from visible payloads; G07 exposes only the strict safe standby context or `null`, observations, and events while preserving audit truth separately; matched trip tests prove identical same-seed pre-branch values/observations and context-driven action changes | Implemented and verified for the current structured simulator boundary; prompt filtering, decision-tick slicing, and renderer contamination gates remain pending |
| SC-005 Split and duplicate integrity | TM-02, TM-10 | Assign structured splits before rendering and detect leakage | Frozen split manifests, scenario/fault/template constraints, matched-context group keys, text/skeleton hashes | Cross-split unit/property tests must keep both members of a context counterfactual group in one assigned split, reject availability-bearing audit identifiers in prompts, exclude post-decision applied actions, and publish overlap/class-context reports | Planned for Phase 3 |
| SC-006 Narrow request schema | TM-03, TM-04, TM-05 | Accept only versioned curated identifiers and bounded controls | Server-side gateway schema plus inference-side critical revalidation | Valid/invalid API contract suite and schema parity test | Planned for Phase 7 |
| SC-007 Request resource limits | TM-05 | Bound bytes, structure, tokens, output, duration, rate, batch, and concurrency | Gateway/inference configuration selected from benchmarks | Boundary, timeout, rate, and concurrency tests | Planned; numeric limits require measurements |
| SC-008 No arbitrary ingestion | TM-01, TM-04 | Expose no file, URL, path, checkpoint, tokenizer, or unrestricted-log input | Such fields absent from public schemas and routes | Negative API route/schema tests | Documented; service not yet present |
| SC-009 Safe artifact loading | TM-04, TM-08 | Load only fixed, trusted, checksummed project artifacts without unsafe user objects | Data-only format where feasible, read-only paths, pinned SHA-256 manifest | Tamper and mismatch startup tests; artifact review | Planned for model/service phases |
| SC-010 Safe output rendering | TM-06 | Treat renderer and model output as untrusted text; no raw HTML | Structured response and framework-default text escaping | HTML/control-character rendering tests and DOM review | Planned for Phase 7 |
| SC-011 Safe errors | TM-07, TM-13 | No stack trace, secret, internal path, or request dump in production responses | Central exception mapping, debug disabled, structured error codes | Negative tests and response/log inspection | Planned for Phase 7 |
| SC-012 Secret isolation | TM-07 | Keep credentials server-side and separate dev/preview/production values | Platform secret store, non-public variable names, local ignored files | Client-bundle scan, log scan, secret scanning, rotation drill | Planned; no service credential currently required |
| SC-013 Minimal logging | TM-07, TM-12 | Collect only justified reliability data and redact credentials/content | Aggregate status/latency/version fields, retention policy | Log-schema test and retention/access review | Planned for Phase 7 |
| SC-014 Browser headers | TM-06, TM-14 | Enforce CSP, framing, MIME, referrer, and permissions policies | Framework/hosting response configuration | Inspection of actual deployed responses | Planned; cannot be verified before owner deployment |
| SC-015 Origin and service authentication | TM-05, TM-07 | Restrict origins deliberately and authenticate gateway-to-inference calls | Allowlisted CORS, server-only scoped credential | Permitted/denied origin and authentication tests | Planned for Phase 7 |
| SC-016 Dependency controls | TM-09 | Minimize and pin dependencies; use reproducible clean installs | Present local `uv.lock`, bounded dependency ranges, review policy | Frozen sync/build in clean environment and dependency audit | Local lockfile present and `make build` verified; clean-environment sync and dependency audit pending |
| SC-017 Static and secret analysis | TM-07, TM-09 | Run supported static analysis, dependency review, and secret scanning | Least-privilege CI workflows prepared for eventual repository | Recorded tool outputs and triage notes | Planned; no remote CI configured |
| SC-018 Release integrity and SBOM | TM-08, TM-09, TM-11, TM-15 | Link source, data, tokenizer, model, results, checksum, and software inventory | Provenance manifest, checksums, SBOM, release allowlist | Clean-environment smoke reproduction and manifest verification | Planned for Phase 8 release readiness |
| SC-019 Metric provenance | TM-10, TM-11 | Display only recorded metrics for matching versions and splits | Immutable evaluation result schema consumed by reports/UI | Fixture mismatch tests and release review | Documented; evaluation artifacts do not yet exist |
| SC-020 Private vulnerability reporting | TM-07, TM-09 | Provide a tested private route without inventing contact data | GitHub private reporting or named private channel selected by owner | Pre-publication reporting drill | Policy documented; route not configured |

Current local evidence recorded on 2026-08-20 under CPython 3.12.11: `make check`
passed with 206 tests and 92.57% total coverage while branch measurement was enabled;
Ruff formatting across 52 files and
lint, strict typing across 29 source files, distribution builds,
and isolated wheel verification also passed. These results do not verify any
production, service, UI, deployment, full-denylist, or human-review control.

## Phase-gate evidence

### Before generator pilot

- Strict-schema tests pass.
- Latent state cannot carry observation-only fields and observations cannot carry hidden fault labels.
- Configuration traversal, unknown-field, collision, and deterministic-snapshot tests pass.
- The fictional/non-operational disclaimer is visible in the root documentation.

### Before dataset pilot

- Prohibited-content fixtures and manual sample-review procedure exist.
- Split manifests are frozen before narrative rendering.
- Duplicate, skeleton-overlap, scenario-identity, template-family, component,
  fault-pair, and driver-plus-fault composition leakage tests pass.

### Before model artifacts

- Checkpoint format and loader threat review is complete.
- Save/reload equivalence and checksum mismatch behavior pass.
- Training and evaluation manifests cannot overwrite prior run artifacts.

### Before a local network demo

- Strict request and response schemas are frozen and versioned.
- Input-size, unsupported-method/content-type, encoding, timeout, safe-error, authentication, and version-mismatch tests pass.
- No endpoint accepts arbitrary prose, files, URLs, paths, or artifacts.

### Phase 8, before owner-managed publication or deployment

- The private reporting route is configured and tested.
- Production response headers, CORS, rate/concurrency behavior, source maps, client bundles, debug routes, and secrets are inspected on the actual target.
- Dependency/static/secret findings are triaged.
- Release checksums, provenance, SBOM, cards, limitations, and smoke reproduction are complete.
- The project owner—not this implementation task—authorizes and performs GitHub push and hosting deployment.

## Residual-risk rule

Verification narrows uncertainty; it does not prove that the project is fully secure. Release notes must name which controls are implemented, which are tested, which are inherited from a platform, and which remain deferred.

The current G07 structured allowlist is not yet a safe rendered-task contract. A
non-null standby context or the G07-only tick-0 context note could become a fault-label
shortcut; `context_id` itself contains an availability word; and a record built after
the decision could expose its later `ACTION_APPLIED` event. Phase 3 must filter audit-only
identifiers, balance semantic context roles and templates, group matched pairs before
splitting, slice inputs at the decision tick, and test these properties before any
dataset pilot.
