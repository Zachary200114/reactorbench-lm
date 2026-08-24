UV ?= uv

.DEFAULT_GOAL := help

.PHONY: help sync format format-check lint typecheck test check build artifact-test phase3-audit phase3-prepare-review phase4-smoke phase4-verify reproduce-smoke phase5-pilot phase5-verify phase6-verify-golden phase6-selection phase6-verify-selection phase6-evaluate phase6-verify phase6-rescore phase6-verify-rescore phase6-remediation-dry-run phase6-remediation-run phase6-remediation-status phase6-remediation-stop phase6-remediation-resume phase6-remediation-evaluate-final

help: ## Show the available development commands.
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "%-36s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## Install all project and development dependencies from the project lockfile.
	$(UV) sync --frozen --all-groups

format: ## Apply deterministic Python formatting and safe lint fixes.
	$(UV) run --frozen ruff format .
	$(UV) run --frozen ruff check --fix .

format-check: ## Verify formatting without changing files.
	$(UV) run --frozen ruff format --check .

lint: ## Run the configured Python linter.
	$(UV) run --frozen ruff check .

typecheck: ## Run strict static type checking.
	$(UV) run --frozen mypy

test: ## Run the test suite.
	$(UV) run --frozen pytest --cov=reactorbench --cov-report=term-missing --cov-fail-under=85

check: format-check lint typecheck test artifact-test ## Run all local quality gates.

build: ## Build source and wheel distributions from the locked environment.
	$(UV) run --frozen python -m build

artifact-test: build ## Verify distribution contents and an isolated local wheel install.
	$(UV) run --frozen python tests/contract/verify_distribution_artifacts.py

phase3-audit: ## Build the structured Phase 3 graph in memory and print its audit summary.
	$(UV) run --frozen python -m reactorbench.dataset audit-development --config configs/dataset/development-v0.1.0.toml --generator-commit "$$(git rev-parse --verify HEAD)"

phase3-prepare-review: ## Print the hash-bound pre-render catalog review packet; does not approve it.
	$(UV) run --frozen python -m reactorbench.dataset prepare-review --config configs/dataset/development-v0.1.0.toml --generator-commit "$$(git rev-parse --verify HEAD)"

phase4-smoke: ## Train the project tokenizer and prove the smoke model can overfit a tiny shard.
	$(UV) run --frozen python -m reactorbench.training run-smoke --config configs/model/phase4-smoke-v0.1.0.toml --source-commit "$$(git rev-parse --verify HEAD)"

phase4-verify: ## Independently verify the tokenizer, safetensors checkpoint, and smoke report.
	$(UV) run --frozen python -m reactorbench.training verify-smoke --config configs/model/phase4-smoke-v0.1.0.toml

reproduce-smoke: phase4-smoke ## Reproduce the Phase 4 smoke milestone in a clean checkout.

phase5-pilot: ## Run all preregistered baselines and the validation-selected pilot.
	$(UV) run --frozen python -m reactorbench.training run-pilot --config configs/experiments/phase5-pilot-v0.1.0.toml --source-commit "$$(git rev-parse --verify HEAD)"

phase5-verify: ## Verify the complete Phase 5 report and safe checkpoints.
	$(UV) run --frozen python -m reactorbench.training verify-pilot --config configs/experiments/phase5-pilot-v0.1.0.toml

phase6-verify-golden: ## Verify the owner-approved golden packet before held-out access.
	$(UV) run --frozen python -m reactorbench.training verify-golden-review --packet golden/golden-suite-v0.1.0.json --record artifacts/review/golden-review-record-v0.1.0.json --expected-packet-sha256 c2e966564dadfab7e8b944ca9b6f8ef59d8545d1da1cc4ea75f8b27a9c44077c

phase6-selection: phase6-verify-golden ## Train and validation-select E3, E5, and E6 without test access.
	$(UV) run --frozen python -m reactorbench.training run-phase6-selection --config configs/experiments/phase6-main-v0.1.0.toml --source-commit "$$(git rev-parse --verify HEAD)"

phase6-verify-selection: ## Verify the validation-only Phase 6 selection artifact.
	$(UV) run --frozen python -m reactorbench.training verify-phase6-selection --config configs/experiments/phase6-main-v0.1.0.toml

phase6-evaluate: phase6-verify-golden phase6-verify-selection ## Run the single authorized held-out and golden evaluation.
	$(UV) run --frozen python -m reactorbench.training run-phase6-evaluation --config configs/experiments/phase6-main-v0.1.0.toml --source-commit "$$(git rev-parse --verify HEAD)"

phase6-verify: ## Verify Phase 6 reports, predictions, access record, and checkpoints.
	$(UV) run --frozen python -m reactorbench.training verify-phase6-evaluation --config configs/experiments/phase6-main-v0.1.0.toml

phase6-rescore: phase6-verify ## Mechanically reparse stored generations after the v0.1.0 delimiter defect.
	$(UV) run --frozen python scripts/phase6_rescore_v0_1_1.py run --config configs/experiments/phase6-main-v0.1.0.toml --correction-source-commit "$$(git rev-parse --verify HEAD)"

phase6-verify-rescore: ## Reconstruct and verify the delimiter-aware Phase 6 rescore.
	$(UV) run --frozen python scripts/phase6_rescore_v0_1_1.py verify --config configs/experiments/phase6-main-v0.1.0.toml

phase6-remediation-dry-run: ## Verify the frozen remediation runner without starting work.
	./scripts/run_phase6_pipeline.sh --dry-run

phase6-remediation-run: ## Start the non-overwriting v0.2-v0.4 development pipeline.
	./scripts/run_phase6_pipeline.sh

phase6-remediation-status: ## Print the latest verified remediation progress snapshot.
	./scripts/check_phase6_status.sh

phase6-remediation-stop: ## Request a cooperative stop at the next safe boundary.
	./scripts/stop_phase6_pipeline.sh

phase6-remediation-resume: ## Resume the existing checksum-bound remediation run.
	./scripts/resume_phase6_pipeline.sh

phase6-remediation-evaluate-final: ## Run one future frozen final access after owner review.
	./scripts/run_phase6_evaluation.sh --confirm-final-evaluation
