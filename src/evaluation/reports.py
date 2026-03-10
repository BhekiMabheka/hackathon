"""
Evaluation reporting — structured summaries of CV runs and feature importance.

Outputs:
  - cv_metrics.json    (picked up by DVC metrics)
  - feature_importance.parquet
  - oof_predictions.parquet  (for post-hoc analysis and stacking)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_all, save_metrics
from src.utils.logging import get_logger

log = get_logger(__name__)


def aggregate_cv_metrics(
    fold_metrics: list[dict[str, float]],
    output_path: str | Path = "outputs/reports/cv_metrics.json",
) -> dict[str, Any]:
    """
    Aggregate per-fold metrics into mean ± std summary.
    DVC reads this file to track experiment performance.
    """
    agg: dict[str, Any] = {}
    keys = fold_metrics[0].keys()
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        agg[f"{k}_mean"] = round(float(np.mean(vals)), 6)
        agg[f"{k}_std"] = round(float(np.std(vals)), 6)
        agg[f"{k}_min"] = round(float(np.min(vals)), 6)
        agg[f"{k}_max"] = round(float(np.max(vals)), 6)

    save_metrics(agg, output_path)
    log.info("CV metrics aggregated", n_folds=len(fold_metrics))
    return agg


def save_oof_predictions(
    oof_df: pd.DataFrame,
    output_path: str | Path = "outputs/models/oof_predictions.parquet",
) -> None:
    """
    Save OOF predictions for post-hoc analysis, error analysis, and stacking.

    Expected columns: id_col, target_col, oof_score, fold
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(output_path, index=False)
    log.info("Saved OOF predictions", path=str(output_path), rows=len(oof_df))


def save_feature_importance(
    importance: pd.Series,
    output_path: str | Path = "outputs/reports/feature_importance.parquet",
    top_n: int = 50,
) -> pd.DataFrame:
    """Save feature importance, sorted descending."""
    df = importance.reset_index()
    df.columns = ["feature", "importance"]
    df = df.sort_values("importance", ascending=False)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info(
        "Saved feature importance",
        path=str(output_path),
        top_feature=df.iloc[0]["feature"],
    )

    # Print top N to console
    log.info("Top features", features=df.head(top_n).to_dict("records"))
    return df


def generate_cv_report(
    fold_metrics: list[dict[str, float]],
    oof_scores: np.ndarray,
    oof_labels: np.ndarray,
    feature_importance: pd.Series | None = None,
    output_dir: str | Path = "outputs/reports",
) -> dict[str, Any]:
    """
    Full CV report: metrics + OOF eval + feature importance.
    Returns the aggregated metrics dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate fold-level metrics
    agg = aggregate_cv_metrics(fold_metrics, output_dir / "cv_metrics.json")

    # Evaluate OOF as a single held-out set
    oof_metrics = evaluate_all(oof_labels, oof_scores, prefix="oof")
    save_metrics(oof_metrics, output_dir / "oof_metrics.json")

    # Feature importance
    if feature_importance is not None:
        save_feature_importance(feature_importance, output_dir / "feature_importance.parquet")

    log.info(
        "CV report generated",
        oof_roc_auc=oof_metrics.get("oof_roc_auc"),
        oof_gini=oof_metrics.get("oof_gini"),
        oof_ks=oof_metrics.get("oof_ks"),
    )
    return {**agg, **oof_metrics}
