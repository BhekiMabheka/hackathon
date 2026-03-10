.PHONY: help install install-dev lint test train predict submit dvc-repro clean setup-mlflow

PYTHON := python
PIP    := pip
SRC    := src
CONF   := conf/config.yaml

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
install:  ## Install runtime dependencies
	$(PIP) install -e .

install-dev:  ## Install dev + runtime dependencies
	$(PIP) install -e ".[dev]"
	pre-commit install

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------
lint:  ## Run ruff linter
	ruff check $(SRC) scripts tests

format:  ## Auto-fix linting issues
	ruff check --fix $(SRC) scripts tests

type-check:  ## Run mypy
	mypy $(SRC)

test:  ## Run test suite with coverage
	pytest --cov=$(SRC) --cov-report=term-missing --cov-report=html:outputs/reports/coverage

# ---------------------------------------------------------------------------
# Data pipeline (DVC)
# ---------------------------------------------------------------------------
dvc-pull:  ## Pull data from remote DVC store
	dvc pull

dvc-repro:  ## Reproduce full DVC pipeline
	dvc repro

dvc-dag:  ## Show DVC pipeline DAG
	dvc dag

# ---------------------------------------------------------------------------
# ML pipeline
# ---------------------------------------------------------------------------
train:  ## Run full training pipeline
	$(PYTHON) scripts/train.py --config-path ../conf --config-name config

train-quick:  ## Quick training run (debug mode, small data)
	$(PYTHON) scripts/train.py --config-path ../conf --config-name config training.debug=true

predict:  ## Generate predictions on test set
	$(PYTHON) scripts/predict.py --config-path ../conf --config-name config

submit:  ## Validate + package submission CSV
	$(PYTHON) scripts/validate_submission.py

# ---------------------------------------------------------------------------
# DE pipeline
# ---------------------------------------------------------------------------
de-validate:  ## Run schema drift + data quality checks
	$(PYTHON) -m src.de.quality_checks

de-backfill:  ## Run backfill pipeline for a date range
	$(PYTHON) -m src.de.backfill --start-date $(START) --end-date $(END)

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
setup-mlflow:  ## Start local MLflow tracking server
	mlflow server \
		--backend-store-uri sqlite:///outputs/mlflow.db \
		--default-artifact-root outputs/artifacts \
		--host 0.0.0.0 --port 5000

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:  ## Remove generated artefacts (keep data)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf outputs/models/* outputs/submissions/* outputs/reports/coverage
	rm -rf .ruff_cache .mypy_cache .pytest_cache
