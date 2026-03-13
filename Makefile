# TASFT Build System
# Task-Aware Sparse Fine-Tuning — development automation
#
# Usage:
#   make install      — editable install with all extras + pre-commit hooks
#   make lint         — ruff check + format verification
#   make typecheck    — mypy strict type checking
#   make test         — unit + integration tests (excludes slow markers)
#   make test-unit    — unit tests only
#   make test-integration — integration tests only
#   make test-all     — full suite including slow tests
#   make bench        — pytest-benchmark suite
#   make coverage     — pytest with HTML coverage report
#   make chaos        — stress/fault-injection tests
#   make clean        — remove all build/cache artifacts
#   make ci           — lint + typecheck + test (CI pipeline)
#   make all          — install + ci

.DEFAULT_GOAL := all

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# Source directories for linting/checking
SRC_DIRS := tasft/ tests/
LINT_DIRS := tasft/ tests/ scripts/ axolotl_plugin/

# Timeout configuration (seconds)
UNIT_TIMEOUT := 30
INTEGRATION_TIMEOUT := 120
CHAOS_TIMEOUT := 120

.PHONY: all install lint typecheck test test-unit test-integration test-all \
        bench coverage chaos clean ci help

##@ Setup

install: ## Editable install with all extras + pre-commit hooks
	pip install -e ".[dev,train,eval]"
	pre-commit install

##@ Quality

lint: ## Run ruff linter and format checker
	ruff check $(LINT_DIRS)
	ruff format --check $(LINT_DIRS)

typecheck: ## Run mypy strict type checking
	mypy --strict tasft/

##@ Testing

test: ## Run unit + integration tests (excludes slow)
	pytest tests/unit/ tests/integration/ -x --timeout=60 -v --tb=short -m "not slow"

test-unit: ## Run unit tests only
	pytest tests/unit/ -x --timeout=$(UNIT_TIMEOUT) -v --tb=short

test-integration: ## Run integration tests only
	pytest tests/integration/ -x --timeout=$(INTEGRATION_TIMEOUT) -v --tb=short -m "not slow"

test-all: ## Run full test suite including slow tests
	pytest tests/unit/ tests/integration/ tests/chaos/ -v --tb=short --timeout=$(INTEGRATION_TIMEOUT)

bench: ## Run benchmark suite with JSON output
	pytest tests/benchmarks/ --benchmark-only -v --benchmark-json=bench_results.json

coverage: ## Run tests with HTML coverage report
	pytest tests/unit/ tests/integration/ \
		--cov=tasft \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-fail-under=80 \
		-v --tb=short -m "not slow"

chaos: ## Run stress/fault-injection tests
	pytest tests/chaos/ -v --timeout=$(CHAOS_TIMEOUT)

##@ CI Pipeline

ci: lint typecheck test ## Run full CI pipeline (lint + typecheck + test)

all: install ci ## Install dependencies and run CI pipeline

##@ Maintenance

clean: ## Remove all build/cache artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf dist build
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage coverage.xml
	rm -f bench_results.json

##@ Help

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} \
		/^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
