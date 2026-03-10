"""
Training entry point.

Usage
-----
    python scripts/train.py                           # defaults from conf/config.yaml
    python scripts/train.py training.debug=true       # quick debug run (2000 rows)
    python scripts/train.py model=xgb                 # switch to XGBoost
    python scripts/train.py training.n_folds=3        # fewer folds
    make train                                        # via Makefile shortcut
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data.ingestion import load_single_file
from src.data.preprocessing import Preprocessor
from src.data.splits import train_val_holdout
from src.data.validation import validate_schema, validate_statistics
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
    feat_cfg = cfg_dict["features"]
    train_cfg = cfg_dict["training"]

    seed_everything(data_cfg.get("random_state", 42))

    # ── 1. Load single-file loan dataset ─────────────────────────────
    nrows = train_cfg.get("debug_nrows") if train_cfg.get("debug") else None
    df = load_single_file(
        raw_dir=data_cfg["raw_path"],
        filename=data_cfg["train_file"],
        id_col=data_cfg["id_col"],
        nrows=nrows,
    )

    # ── 2. Validate raw data ──────────────────────────────────────────
    # No id_col in raw file — validation checks target and numeric cols
    validate_schema(df, required_cols=[data_cfg["target_col"]])
    validate_statistics(df, target_col=data_cfg["target_col"], id_col=data_cfg["id_col"])

    checker = DataQualityChecker(dataset_name="loan_dataset")
    qreport = checker.run(df, id_col=data_cfg["id_col"])
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    qreport.save("outputs/reports/quality_report.json")
    if not qreport.passed:
        log.warning("Quality checks flagged issues — review outputs/reports/quality_report.json")

    # ── 3. Train/val/holdout split ────────────────────────────────────
    # No date column — random stratified split
    train, val, holdout = train_val_holdout(
        df,
        target_col=data_cfg["target_col"],
        date_col=None,
        test_size=data_cfg["test_size"],
        val_size=data_cfg["val_size"],
        random_state=data_cfg["random_state"],
    )
    log.info(
        "Dataset split",
        train=len(train),
        val=len(val),
        holdout=len(holdout),
        train_pos_rate=round(float(train[data_cfg["target_col"]].mean()), 4),
    )

    # ── 4. Preprocess ─────────────────────────────────────────────────
    pp = Preprocessor(
        target_col=data_cfg["target_col"],
        id_col=data_cfg["id_col"],
        date_col=None,
        max_null_ratio=feat_cfg.get("drop_high_null_thresh", 0.8),
        drop_redundant=feat_cfg.get("drop_redundant", True),
        drop_leaky=feat_cfg.get("drop_leaky", False),
    )
    train = pp.fit_transform(train)
    val = pp.transform(val)
    holdout = pp.transform(holdout)
    pp.save_meta("data/interim/preprocessing_meta.json")

    # ── 5. Feature engineering ────────────────────────────────────────
    fe_config = {**data_cfg, **feat_cfg}
    fe = FeatureEngineer(config=fe_config)
    train = fe.fit_transform(train)
    val = fe.transform(val)
    holdout = fe.transform(holdout)
    fe.save_meta("data/processed/feature_meta.json")

    # ── 6. Feature selection ──────────────────────────────────────────
    train, dropped = apply_selection(
        train,
        target_col=data_cfg["target_col"],
        id_col=data_cfg["id_col"],
        date_col=None,
    )
    log.info("Feature selection dropped", n=len(dropped))

    # Drop same columns from val/holdout
    val = val.drop(columns=[c for c in dropped if c in val.columns])
    holdout = holdout.drop(columns=[c for c in dropped if c in holdout.columns])

    # Persist processed data for DVC tracking
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    train.to_parquet("data/processed/train_features.parquet", index=False)
    holdout.to_parquet("data/processed/test_features.parquet", index=False)

    # ── 7. Feature columns ────────────────────────────────────────────
    exclude = {data_cfg["id_col"], data_cfg["target_col"]}
    feature_cols = [c for c in train.columns if c not in exclude]
    log.info("Feature set size", n_features=len(feature_cols))

    # ── 8. CV training ────────────────────────────────────────────────
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
        date_col=None,
    )

    log.info(
        "Training complete",
        oof_roc_auc=report.get("oof_roc_auc"),
        oof_gini=report.get("oof_gini"),
        oof_ks=report.get("oof_ks"),
        oof_capture_top10=report.get("oof_capture_top10pct"),
    )


if __name__ == "__main__":
    main()
