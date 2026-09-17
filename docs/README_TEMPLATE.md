# README maintenance template

Use this file when the project status changes or the public README needs a new section.
The goal is to keep the repository easy to understand without turning the front page
into an experiment log.

## Source-of-truth map

| README content | Update from |
| --- | --- |
| Current phase and next run | `docs/IMPLEMENTATION_STATUS.md` |
| Canonical project scope | `research/PROJECT_REQUIREMENTS.md` |
| Latest versioned decision | `research/DECISION_LOG.md` |
| Measured model results | Versioned reports under `docs/model/` |
| Dataset facts | `docs/data/DATASET_CARD.md` and approved manifests |
| Security claims | `docs/threat-model.md`, `docs/security-controls.md`, and `SECURITY.md` |
| License terms | `LICENSE` and `pyproject.toml` |

Never copy a planned, concept-art, partial, or unverified value into a measured-results
table. Link to the evidence report instead of reproducing long checksum inventories in
the README.

## Stable root README structure

Keep these sections in this order:

1. Project name and one-sentence description
2. Work-in-progress and fictional/non-operational notices
3. At-a-glance technical table
4. Research question and system flow
5. What was built
6. Current measured evidence
7. Repository layout
8. Getting started and verification commands
9. Security and scientific-integrity summary
10. Documentation links
11. Roadmap
12. License and publication boundary

Sections 1–4 should remain understandable to a technical reader who has not studied
machine learning. Detailed run history, hashes, failure traces, and operator steps
belong in the implementation status, versioned reports, or runbook.

## Copy-ready status block

```markdown
> **Work in progress — Phase X.** [One sentence describing what is implemented.]
> [One sentence describing what is not complete.] [One sentence preventing an
> unsupported success or deployment claim.]
```

Example update rules:

- Change only the phase and the smallest factual summary.
- Do not call a configuration “trained” until a completed run proves it.
- Do not call a development pass a final benchmark result.
- If a run fails, state the failure plainly and link to its report.

## Copy-ready evidence row

```markdown
| Experiment name | Measured result; include the split and limitation |
```

Every row should answer three questions:

1. What was evaluated?
2. What was measured?
3. What prevents overinterpreting the number?

Prefer a small table of decision-relevant milestones. Move superseded run-by-run
history to `docs/IMPLEMENTATION_STATUS.md` or a versioned model report.

## Copy-ready feature section

```markdown
### Feature name

- What it does.
- What was implemented directly in this repository.
- What external library primitive is used, if relevant.
- Where the detailed evidence or contract lives.
```

Remove the entire block when a feature is no longer important to the public story.
Do not leave empty headings or “coming soon” sections without a concrete roadmap item.

## Research README structure

The research README is an index, not a second project homepage. Keep it limited to:

1. Current research state
2. Research question
3. How to interpret document types
4. Authority and precedence
5. Grouped research-document index
6. Implemented research stack
7. Research boundaries
8. Links to evidence reports
9. Documentation-update rules

Detailed measurements should remain in `docs/model/`; detailed live state should
remain in `docs/IMPLEMENTATION_STATUS.md`.

## Update checklist

Before committing a README change:

- [ ] Confirm the worktree and current run identity.
- [ ] Read the top checkpoint in `docs/IMPLEMENTATION_STATUS.md`.
- [ ] Verify every new metric against a versioned artifact or report.
- [ ] Keep planned work visibly separate from completed work.
- [ ] Preserve the fictional and non-operational boundary.
- [ ] Preserve the from-random-initialization and no-hosted-LLM claim accurately.
- [ ] Check every relative link and command.
- [ ] Keep the main README concise; link to detailed history.
- [ ] Update the roadmap checkboxes only after the milestone is actually complete.
- [ ] Run the README/license contract tests and `git diff --check`.

Recommended verification:

```bash
.venv/bin/pytest -q tests/contract/test_license_policy.py
git diff --check
```

## Writing style

- Use direct first-person language for work completed by the project owner.
- Lead with the result, then explain the mechanism.
- Prefer concrete nouns and measured values over promotional adjectives.
- Say “implemented,” “measured,” “failed,” or “planned” precisely.
- Avoid claiming that extensive documentation alone proves model quality.
- Keep negative results visible; they are part of the research evidence.
