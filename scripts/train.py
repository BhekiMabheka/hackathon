"""
Training entry point.

Usage
-----
    python scripts/train.py                           # defaults from conf/config.yaml
    python scripts/train.py training.debug=true       # quick debug run
    python scripts/train.py model=xgb                 # switch to XGBoost
    python scripts/train.py training.n_folds=3        # fewer folds
    make train                                        # via Makefile shortcut
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data.ingestion import load_train_test
from src.data.preprocessing import Preprocessor
from src.data.validation import validate_no_leakage, validate_schema, validate_statistics
from src.de.quality_checks import DataQualityChecker
from src.features.engineering import FeatureEngineer
from src.features.selection import apply_selection
from src.pipelines.training_pipeline import TrainingPipeline
from src.utils.logging import get_logger
from src.utils.reproducibility import seed_everything

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    log.info("Starting training", config=cfg_dict.get("project"))

    data_cfg = cfg_dict["data"]
    seed_everything(data_cfg.get("random_state", 42))

    # ── 1. Load raw data ──────────────────────────────────────────────
    nrows = cfg_dict["training"].get("debug_nrows") if cfg_dict["training"].get("debug") else None
    train, test = load_train_test(
        raw_dir=data_cfg["raw_path"],
        date_col=data_cfg["date_col"],
        id_col=data_cfg["id_col"],
        nrows=nrows,
    )

    # ── 2. Validate ───────────────────────────────────────────────────
    required_cols = [data_cfg["id_col"], data_cfg["target_col"]]
    validate_schema(train, required_cols=required_cols)
    validate_statistics(train, target_col=data_cfg["target_col"], id_col=data_cfg["id_col"])
    validate_no_leakage(train, test, id_col=data_cfg["id_col"], date_col=data_cfg["date_col"])

    checker = DataQualityChecker(dataset_name="train")
    report = checker.run(train, date_col=data_cfg["date_col"], id_col=data_cfg["id_col"])
    report.save("outputs/reports/quality_report.json")
    # Log warnings but don't abort — quality issues are expected in real data
    if not report.passed:
        log.warning("Quality checks flagged issues — review outputs/reports/quality_report.json")

    # ── 3. Preprocess ─────────────────────────────────────────────────
    pp = Preprocessor(
        target_col=data_cfg["target_col"],
        id_col=data_cfg["id_col"],
        date_col=data_cfg["date_col"],
        max_null_ratio=cfg_dict["features"].get("drop_high_null_thresh", 0.8),
    )
    train = pp.fit_transform(train)
    test = pp.transform(test)
    pp.save_meta("data/interim/preprocessing_meta.json")

    # ── 4. Feature engineering ────────────────────────────────────────
    fe_config = {**data_cfg, **cfg_dict["features"]}
    fe = FeatureEngineer(config=fe_config)
    train = fe.fit_transform(train)
    test = fe.transform(test)
    fe.save_meta("data/processed/feature_meta.json")

    # ── 5. Feature selection ──────────────────────────────────────────
    train, dropped = apply_selection(
        train,
        target_col=data_cfg["target_col"],
        id_col=data_cfg["id_col"],
        date_col=data_cfg["date_col"],
    )
    log.info("Feature selection dropped", n=len(dropped))

    # Persist processed data (for DVC tracking)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    train.to_parquet("data/processed/train_features.parquet", index=False)
    test.to_parquet("data/processed/test_features.parquet", index=False)

    # ── 6. Define feature columns ─────────────────────────────────────
    exclude = {data_cfg["id_col"], data_cfg["target_col"], data_cfg["date_col"]}
    feature_cols = [c for c in train.columns if c not in exclude]
    log.info("Feature set size", n_features=len(feature_cols))

    # ── 7. Train with CV ──────────────────────────────────────────────
    pipeline = TrainingPipeline(
        cfg=cfg_dict,
        mlflow_uri=cfg_dict.get("mlflow", {}).get("tracking_uri", "sqlite:///outputs/mlflow.db"),
        experiment=cfg_dict.get("mlflow", {}).get("experiment_name", "nedbank-masters-2026"),
    )
    report = pipeline.run(
        train_df=train,
        feature_cols=feature_cols,
        target_col=data_cfg["target_col"],
        id_col=data_cfg["id_col"],
        date_col=data_cfg.get("date_col"),
    )

    log.info(
        "Training complete",
        roc_auc=report.get("oof_roc_auc"),
        gini=report.get("oof_gini"),
        ks=report.get("oof_ks"),
    )


if __name__ == "__main__":
    main()
