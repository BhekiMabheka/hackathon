"""
Evaluation metrics for banking ML tasks.

Beyond raw AUC, banking decisions require:
  - Gini coefficient (2*AUC - 1) — standard in credit risk
  - KS statistic — regulatory metric for scorecards
  - Precision/Recall at various operating points (not just 0.5 threshold)
  - Calibration check — probabilities must be reliable for decision-making
  - Business metrics (e.g., capture rate at top decile)

All functions accept numpy arrays so they work inside CV loops.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.utils.logging import get_logger

log = get_logger(__name__)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under precision-recall curve — better for imbalanced targets."""
    return float(average_precision_score(y_true, y_score))


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini coefficient = 2 * AUC - 1. Standard credit risk metric."""
    return 2.0 * roc_auc(y_true, y_score) - 1.0


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic.
    Max separation between cumulative good and bad distributions.
    Regulatory benchmark in SA credit scoring (NCA context).
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def f1_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> float:
    preds = (y_score >= threshold).astype(int)
    return float(f1_score(y_true, preds, zero_division=0))


def capture_rate_at_decile(
    y_true: np.ndarray,
    y_score: np.ndarray,
    top_pct: float = 0.10,
) -> float:
    """
    What fraction of actual positives fall in the top X% of scored population?
    Core business metric: 'does top decile capture 3x expected positive rate?'
    """
    n = len(y_true)
    n_top = max(1, int(n * top_pct))
    top_idx = np.argsort(y_score)[::-1][:n_top]
    actual_positives = y_true.sum()
    if actual_positives == 0:
        return 0.0
    captured = y_true[top_idx].sum()
    return float(captured / actual_positives)


def optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "f1",
) -> float:
    """Find threshold that maximises F1 (or precision, recall) on OOF."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if metric == "f1":
        f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
        best_idx = np.argmax(f1_scores[:-1])
    elif metric == "precision":
        best_idx = np.argmax(precision[:-1])
    else:
        best_idx = np.argmax(recall[:-1])
    return float(thresholds[best_idx])


def calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).
    Low ECE means probabilities are reliable — important for risk pricing.
    """
    fraction_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins)
    bin_sizes = np.histogram(y_score, bins=n_bins)[0]
    weights = bin_sizes / bin_sizes.sum()
    n_bins_actual = len(fraction_pos)
    ece = float(np.sum(np.abs(fraction_pos - mean_pred[:n_bins_actual]) * weights[:n_bins_actual]))
    return ece


def evaluate_all(
    y_true: np.ndarray,
    y_score: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """Compute the full suite of metrics and return as a dict."""
    sep = "_" if prefix else ""
    metrics = {
        f"{prefix}{sep}roc_auc": roc_auc(y_true, y_score),
        f"{prefix}{sep}pr_auc": pr_auc(y_true, y_score),
        f"{prefix}{sep}gini": gini(y_true, y_score),
        f"{prefix}{sep}ks": ks_statistic(y_true, y_score),
        f"{prefix}{sep}f1_at_05": f1_at_threshold(y_true, y_score, 0.5),
        f"{prefix}{sep}capture_top10pct": capture_rate_at_decile(y_true, y_score, 0.10),
        f"{prefix}{sep}capture_top20pct": capture_rate_at_decile(y_true, y_score, 0.20),
        f"{prefix}{sep}ece": calibration_error(y_true, y_score),
        f"{prefix}{sep}pos_rate": float(y_true.mean()),
    }
    log.info("Evaluation metrics", **{k: round(v, 4) for k, v in metrics.items()})
    return metrics


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()}, f, indent=2)
    log.info("Saved metrics", path=str(path))
