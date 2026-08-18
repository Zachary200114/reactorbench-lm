# Security policy

ReactorBench-LM is in local pre-release development. This policy describes how to handle potential vulnerabilities without overstating the maturity of controls that are still being implemented.

## Supported versions

There is no public release, hosted service, or supported production version yet.

| Version | Status |
|---|---|
| Local development snapshot | Under active development; reports are welcome |
| Public or hosted release | Not available |

Before any release, this table must be replaced with concrete version ranges and support dates.

## Reporting a vulnerability

Do not disclose suspected vulnerabilities, secrets, sensitive logs, or exploit details in a public issue or discussion.

Until a repository-private reporting feature or a named private address is configured, contact the project owner through an already-established private channel and include only enough initial detail to establish contact. A concrete private reporting route must be configured and tested before publication. This document intentionally does not invent an email address or imply that an unconfigured channel exists.

When a private channel is available, a useful report includes:

- the affected version, commit, or artifact identifier;
- the component and preconditions;
- reproducible steps using only synthetic ReactorBench-LM inputs;
- observed and expected behavior;
- likely impact;
- any suggested mitigation; and
- whether the issue may expose a credential or another person's data.

Do not send real nuclear-facility information, Navy material, operational records, third-party secrets, personal information, or destructive proof-of-concept payloads. Use the smallest safe synthetic reproduction.

## Response expectations

No formal service-level agreement exists during local development. The project owner should privately acknowledge, triage, remediate, test, and coordinate disclosure before describing a report as resolved. Target timelines will be published only when there is a maintained public release.

## Security boundary

Security work covers the software and artifacts created by this project, including:

- strict schema and configuration boundaries;
- synthetic-data provenance and split integrity;
- model and tokenizer artifact integrity;
- the future narrow inference service;
- the future Research Editorial web application;
- dependency and build-chain hygiene; and
- safe errors, logging, secrets, and release evidence.

The following are outside scope and must not be introduced as test inputs:

- real plant systems, records, procedures, event reports, setpoints, or facility data;
- Navy nuclear or service-derived non-public information;
- real operational, emergency, maintenance, licensing, security, or safety advice;
- arbitrary file, URL, checkpoint, tokenizer, or model-path ingestion;
- attacks against third-party or public infrastructure; and
- denial-of-service testing beyond bounded local test fixtures.

## Current limitations

- No security control is represented as deployment-verified because no service is deployed.
- Planned controls are not implemented controls, and implemented controls are not verified controls until a recorded test establishes their behavior.
- Dependency scanning, static analysis, secret scanning, artifact checksums, an SBOM, production headers, rate limiting, and deployment isolation remain release gates rather than current claims unless [the control map](docs/security-controls.md) records evidence otherwise.
- This project cannot guarantee that code is free of vulnerabilities.

See [the threat model](docs/threat-model.md) and [security control map](docs/security-controls.md) for tracked risks, ownership, status, and future verification evidence.
