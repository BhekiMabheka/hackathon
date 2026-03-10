# Nedbank Data & Analytics Masters 2026 — MLOps Skeleton

Production-grade MLOps skeleton for the Nedbank Masters Challenge.
Covers both the **ML/DS track** (Zindi) and the **Data Engineering track** (Otinga).

---

## Architecture Overview

```
Raw Data (data/raw/)
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Ingestion   │────▶│  Validation  │────▶│ Quality Chks │  ← DE track
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────┐
                                    │  Schema Registry   │  ← DE track
                                    └────────────────────┘
       │
       ▼
┌──────────────┐
│ Preprocessing│  Fit on train only — no leakage
└──────────────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│   Feature    │────▶│  Feature     │
│ Engineering  │     │  Selection   │
└──────────────┘     └──────────────┘
       │
       ▼
┌────────────────────────────────────────────┐
│            CV TRAINING LOOP                │
│  stratified_time split (anti-leakage)      │
│  Fold1 │ Fold2 │ Fold3 │ ...               │
│        └───────┴───────┴── OOF Preds       │
└────────────────────────────────────────────┘
       │
       ├──▶ Retrain on full train → Final model (outputs/models/model.pkl)
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Evaluation  │────▶│   MLflow     │────▶│  Submission  │
│ AUC/Gini/KS  │     │  Tracking   │     │   CSV out    │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## Project Structure

```
hackathon/
├── conf/                      # Hydra configuration (composable)
│   ├── config.yaml
│   ├── data/banking.yaml      # Column roles — update after brief released
│   ├── model/lgbm.yaml
│   ├── model/xgb.yaml
│   └── training/cv.yaml
│
├── src/
│   ├── data/
│   │   ├── ingestion.py       # Load CSV/Parquet/Excel
│   │   ├── validation.py      # Schema + leakage checks
│   │   ├── preprocessing.py   # Stateful preprocessor
│   │   └── splits.py          # Temporal / stratified CV splits
│   ├── features/
│   │   ├── engineering.py     # Rolling, lag, temporal, velocity features
│   │   └── selection.py       # Variance, correlation, importance filters
│   ├── models/
│   │   ├── base.py
│   │   ├── lgbm_model.py
│   │   ├── xgb_model.py
│   │   └── ensemble.py        # Weighted blend + model factory
│   ├── evaluation/
│   │   ├── metrics.py         # AUC, Gini, KS, PR-AUC, capture rate, ECE
│   │   └── reports.py         # CV aggregation, OOF save, feature importance
│   ├── pipelines/
│   │   ├── training_pipeline.py
│   │   ├── inference_pipeline.py
│   │   └── submission_pipeline.py
│   ├── de/                    # Data Engineering track
│   │   ├── schema_registry.py # Schema definition + drift detection
│   │   ├── quality_checks.py  # Duplicates, nulls, future dates, ranges
│   │   ├── backfill.py        # Idempotent partition-by-date backfill
│   │   └── observability.py   # Prometheus metrics + stage timing
│   └── utils/
│       ├── logging.py         # Structured logging (structlog, JSON/text)
│       ├── reproducibility.py # seed_everything()
│       └── io.py              # Format-detecting file reader
│
├── scripts/
│   ├── train.py               # Full training pipeline entry point
│   ├── predict.py             # Inference + submission builder
│   └── validate_submission.py # Pre-upload submission validator
│
├── tests/                     # pytest suite with shared fixtures
├── data/{raw,interim,processed}/
├── outputs/{models,submissions,reports}/
├── dvc.yaml                   # Reproducible pipeline DAG
├── params.yaml                # DVC-tracked hyperparameters
├── pyproject.toml
└── Makefile
```

---

## Quick Start

```bash
# Install dependencies
make install-dev

# Set environment variables
cp .env.example .env

# Place competition data in data/raw/ (available from Zindi, 25 March 2026)
# Then update column names in params.yaml and conf/data/banking.yaml

# Run the full reproducible pipeline
make dvc-repro

# Or run directly
make train        # trains + logs to MLflow
make predict      # generates submission CSV
make submit       # validates CSV before upload

# Review experiments
make setup-mlflow  # opens at http://localhost:5000

# Run tests
make test
```

---

## Key Design Decisions

**Temporal leakage prevention** — default CV strategy is `stratified_time`:
validation folds are always from a strictly later period than training folds.
Standard K-Fold on transaction data leaks future information.

**Evaluation discipline** — beyond AUC, the skeleton tracks Gini (2·AUC−1),
KS statistic (SA regulatory benchmark), PR-AUC (imbalanced targets),
capture rate at top decile (business metric), and Expected Calibration Error.

**Reproducibility** — `seed_everything(42)`, all params tracked in `params.yaml`,
every run logged to MLflow with full params + metrics + artefact.

**DE observability** — schema drift detection on every data load, quality
checks for duplicates/future dates/range violations, Prometheus metrics
per pipeline stage, idempotent date-partitioned backfill.

---

## Adapting to the Actual Brief (from 25 March 2026)

1. Update `conf/data/banking.yaml`: set `id_col`, `date_col`, `target_col`.
2. Run EDA in `notebooks/`.
3. Extend `src/features/engineering.py` with domain-specific features.
4. Run `make dvc-repro` — only invalidated stages rebuild.

## Model Switching

```bash
python scripts/train.py model=xgb           # XGBoost
python scripts/train.py training.n_folds=3  # fewer folds
python scripts/train.py training.debug=true # subsample for speed
```