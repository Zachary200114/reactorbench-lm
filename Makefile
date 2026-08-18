UV ?= uv

.DEFAULT_GOAL := help

.PHONY: help sync format format-check lint typecheck test check build artifact-test

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
