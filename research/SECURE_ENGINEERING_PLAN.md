# Secure engineering and verification plan

Status: pre-implementation security contract
Scope: public GitHub repository, Vercel presentation layer, inference API, model artifacts, and release pipeline

## 1. Objective

ReactorBench-LM will be secure by design and will publish evidence for its security claims. Security remains a supporting engineering dimension—the project's primary contribution is still the from-scratch Transformer, synthetic dataset, and evaluation benchmark.

The public portfolio should demonstrate:

- a written threat model;
- narrow trust boundaries and least privilege;
- validated and size-limited inputs;
- safe model-artifact handling;
- protected secrets and deployment environments;
- automated security checks in continuous integration;
- abuse resistance and privacy-conscious logging;
- documented limitations and vulnerability reporting.

The project must never claim to be “secure” merely because tools ran successfully. Claims must name the control, threat, test, and current limitation.

## 2. Security architecture

```text
Untrusted browser
    |
    | HTTPS, constrained requests
    v
Vercel web application / server-side API gateway
    |-- schema validation
    |-- request-size and method restrictions
    |-- rate and concurrency controls
    |-- safe error mapping
    |
    | authenticated server-to-server request
    v
Inference service
    |-- no arbitrary URL or file inputs
    |-- fixed tokenizer and checkpoint
    |-- bounded token count and generation length
    |-- timeout and resource limits
    v
Versioned project model artifacts
    |-- pinned checksum
    |-- trusted release source only
    |-- no user-supplied checkpoint loading
```

The browser is always untrusted. It must never contain inference-service credentials, signing secrets, administrative tokens, or private storage credentials. Public UI restrictions are usability controls; the server independently enforces every boundary.

## 3. Assets and threats

| Asset | Primary threats | Required controls |
|---|---|---|
| Inference availability | request floods, expensive inputs, concurrency exhaustion | rate limit, request/body/token ceilings, timeout, queue/concurrency cap, cached curated scenarios |
| Inference boundary | arbitrary payloads, real-facility content, schema bypass | server-side allowlist schema, enum validation, prohibited-pattern checks, no arbitrary files or URLs |
| Model checkpoint | replacement, corruption, unsafe deserialization | release checksum, read-only deployment, trusted artifact source, safe serialization format where feasible |
| Secrets | repository exposure, client-bundle leakage, verbose logs | platform secret store, server-only variables, secret scanning, rotation procedure, log redaction |
| Web visitors | cross-site scripting, malicious links, unwanted tracking | framework escaping, no raw HTML rendering, CSP, security headers, minimal telemetry, output encoding |
| Deployment pipeline | dependency compromise, unauthorized release, mutable actions | protected branch, reviewed changes, pinned actions/dependencies, least-privilege tokens, provenance record |
| Research credibility | fabricated metrics, mismatched model/UI versions | immutable result manifests, model/schema version response, checksums, reproducible evaluation |

The implementation threat model must use a lightweight data-flow review and record assumptions, trust boundaries, abuse cases, mitigations, residual risk, and verification status.

## 4. Input contract

The public endpoint should accept a small versioned JSON object containing a curated scenario ID, plant variant, replay seed, and permitted controls. It must not accept:

- file uploads;
- remote URLs;
- serialized Python objects;
- arbitrary model or tokenizer paths;
- shell fragments or executable code;
- unrestricted prose or real operating logs;
- client-selected generation limits;
- undocumented fields.

Server-side validation requirements:

- reject unknown fields;
- use explicit enum allowlists;
- enforce numeric ranges and integer bounds;
- cap request bytes before JSON parsing where supported;
- cap tokens, scenario ticks, output length, and batch size;
- accept only required HTTP methods and content types;
- return generic validation errors without stack traces or internal paths;
- normalize once and validate once against the canonical schema.

TypeScript types alone do not validate hostile runtime input. A runtime schema must enforce the contract at the gateway, and the inference service must revalidate the security-critical subset.

## 5. Web application controls

- Use framework-default escaping and never render model or visitor content as raw HTML.
- Avoid dangerous DOM sinks and dynamic code execution.
- Set an explicit Content Security Policy compatible with the final application.
- Set relevant security headers, including MIME sniffing protection, clickjacking protection through CSP `frame-ancestors`, a deliberate referrer policy, and a permissions policy.
- Keep cross-origin access restricted to the production site and required preview environments. Treat CORS as a browser policy, not authentication.
- Do not use cookie-based sessions for the public demo unless a later requirement justifies them. If cookies are introduced, reassess CSRF and session security.
- Permit outbound requests only to fixed services controlled by the project; never create a server-side URL-fetch feature.
- Ensure source maps, error pages, and client bundles do not reveal secrets, private paths, or internal service details.
- Maintain accessible error states that fail closed without exposing debug information.

Headers must be verified against the deployed response, not merely present in configuration files.

## 6. Inference-service controls

- Expose one narrow inference operation rather than a general model-execution interface.
- Keep model selection, checkpoint path, device, batch size, token limits, and decoding limits server controlled.
- Load only the project's pinned checkpoint from a trusted release location.
- Prefer data-only model artifacts such as `safetensors` when compatible with the implementation; never load user-supplied pickle-based artifacts.
- Run as a non-root user in a minimal container with a read-only filesystem where practical.
- Remove compilers, shells, development servers, notebooks, and unused packages from the production image where feasible.
- Apply CPU, memory, duration, request, and concurrency limits.
- Disable verbose framework and exception output in production.
- Return structured allowlisted fields rather than arbitrary rendered markup.
- Bind administrative or health-detail endpoints privately; expose only a minimal public liveness response if needed.

Model output is untrusted display data even though it comes from a project model. It must be encoded and rendered as text.

## 7. Secrets and environment separation

- Store secrets only in approved deployment secret managers and local untracked environment files.
- Commit an example environment file containing names and safe placeholders only.
- Separate development, preview, and production credentials.
- Scope server-to-server credentials to the single required service and operation.
- Never expose server-only variables through public environment-variable prefixes.
- Prevent secrets from appearing in error messages, analytics, test fixtures, screenshots, or CI output.
- Define revocation and rotation steps before the public launch.

If the public frontend calls a Vercel server-side gateway, that gateway—not browser JavaScript—will attach the inference credential.

## 8. Dependency and build-chain controls

- Commit lockfiles and use reproducible clean installs in CI.
- Pin production dependencies and review major upgrades.
- Enable automated dependency update alerts and dependency review for pull requests.
- Run CodeQL or equivalent static analysis for supported languages.
- Enable secret scanning and push protection when available for the repository.
- Generate a software bill of materials for tagged releases.
- Pin third-party CI actions to immutable revisions under the selected policy.
- Grant workflow tokens only the permissions required by each job.
- Build production artifacts in CI rather than on an untracked workstation path.
- Record source revision, model checksum, schema version, and build identifier in each release.

Automated findings must be triaged; a green badge is not a substitute for review.

## 9. Tests required before deployment

### Unit and integration tests

- schema accepts every intended request and rejects unknown fields;
- boundary values for seed, tick count, scenario selection, and token count;
- unsupported content type and method rejection;
- malformed, truncated, deeply nested, duplicate-key, and oversized JSON handling;
- output encoding for HTML-like and control-character content;
- inference timeout and resource-exhaustion behavior;
- authentication failure between gateway and inference service;
- production error responses contain no stack trace, secret, or filesystem path;
- checkpoint checksum mismatch prevents startup;
- model/schema version mismatch fails closed;
- prohibited real-facility and free-form input paths remain unavailable.

### Deployment verification

- HTTPS and redirect behavior;
- CSP and other security headers on actual production responses;
- CORS behavior from permitted and unpermitted origins;
- rate-limit and concurrency behavior;
- no secrets in client bundles or source maps;
- no directory listing, development route, debug endpoint, or unintended API documentation;
- dependency, container, and static-analysis scans evaluated before release;
- golden model scenarios still pass in the deployed build.

### Abuse testing

Use safe automated cases for oversized requests, rapid requests, enum manipulation, JSON structure abuse, output rendering, and attempts to choose files, URLs, checkpoints, or arbitrary generation settings. Do not perform uncontrolled denial-of-service testing against public shared infrastructure.

## 10. Logging and privacy

The curated-input design means the service does not need to collect visitor prose.

Record only what is necessary for reliability and abuse response, such as:

- timestamp bucket;
- scenario identifier and model version;
- coarse latency, status, and rate-limit outcome;
- non-reversible or short-lived request correlation identifier if required.

Do not log request credentials, IP addresses beyond a justified short-lived platform need, browser fingerprints, full headers, or unnecessary visitor identifiers. Document retention, access, and deletion behavior. Error logging must redact secrets and avoid serializing entire request objects.

## 11. Public evidence package

The GitHub repository should eventually include:

- `SECURITY.md` with supported versions and a private reporting route;
- `docs/threat-model.md` with the current data-flow diagram and risk register;
- `docs/security-controls.md` mapping threats to implementation and tests;
- CI workflows for tests, static analysis, dependency review, and secret scanning;
- release checksums and an SBOM;
- a short deployed-header verification result;
- a security limitations section that distinguishes implemented, tested, inherited, and deferred controls.

The website can summarize this under **Engineering → Security**, linking to the evidence. It should show concrete statements such as “requests reject unknown fields and are capped at X bytes,” once verified, rather than broad claims such as “military-grade security.”

## 12. Portfolio balance

The recommended presentation order is:

1. from-scratch Transformer and research question;
2. synthetic causal dataset and generalization results;
3. live interactive model demonstration;
4. reproducibility and secure engineering evidence.

This shows that Zachary can build a serious AI system securely without making the project appear to be another cybersecurity application.

## 13. Standards and primary references

- OWASP Application Security Verification Standard: https://owasp.org/www-project-application-security-verification-standard/
- OWASP API Security Top 10 (2023): https://owasp.org/API-Security/editions/2023/en/0x11-t10/
- OWASP Secure Headers Project: https://owasp.org/www-project-secure-headers/
- GitHub CodeQL documentation: https://docs.github.com/en/code-security/code-scanning/introduction-to-code-scanning/about-code-scanning-with-codeql
- GitHub secret scanning documentation: https://docs.github.com/en/code-security/secret-scanning/introduction/about-secret-scanning
- GitHub dependency review documentation: https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/about-dependency-review
- Vercel secure backend access: https://vercel.com/docs/security/secure-backend-access
- Vercel security headers: https://vercel.com/docs/headers/security-headers

Exact framework, platform, and standard versions must be recorded when implementation begins because their controls and limits can change.
