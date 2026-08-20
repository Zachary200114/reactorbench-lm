UV ?= uv

.DEFAULT_GOAL := help

.PHONY: help sync format format-check lint typecheck test check build artifact-test phase3-audit phase3-prepare-review phase4-smoke phase4-verify reproduce-smoke

help: ## Show the available development commands.
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "%-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

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
