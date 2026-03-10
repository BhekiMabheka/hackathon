"""
Feature selection — remove noise before model training.

Methods:
  1. variance_threshold  — drop near-zero variance features
  2. correlation_filter  — drop one of each highly-correlated pair
  3. importance_filter   — drop features with near-zero LGBM importance
  4. leakage_screen      — flag features suspiciously correlated with target
                           (can indicate data leakage, not just good signal)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from src.utils.logging import get_logger

log = get_logger(__name__)


def variance_filter(df: pd.DataFrame, threshold: float = 0.01) -> list[str]:
    """Return column names to drop due to near-zero variance."""
    num_df = df.select_dtypes(include="number")
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(num_df.fillna(0))
    drop = num_df.columns[~sel.get_support()].tolist()
    if drop:
        log.info("Variance filter", dropping=len(drop), cols=drop[:10])
    return drop


def correlation_filter(df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """
    Return one column from each pair with |corr| > threshold.
    Keeps the column that appears first (arbitrary but stable).
    """
    num_df = df.select_dtypes(include="number").fillna(0)
    corr = num_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if drop:
        log.info("Correlation filter", dropping=len(drop), threshold=threshold)
    return drop


def importance_filter(
    feature_importance: pd.Series,
    threshold_pct: float = 0.001,
) -> list[str]:
    """
    Return feature names whose importance is below threshold_pct of total.

    Parameters
    ----------
    feature_importance: Series indexed by feature name, values = importance.
    threshold_pct:      Fraction of total importance below which to drop.
    """
    total = feature_importance.sum()
    if total == 0:
        return []
    relative = feature_importance / total
    drop = relative[relative < threshold_pct].index.tolist()
    log.info("Importance filter", dropping=len(drop), threshold_pct=threshold_pct)
    return drop


def leakage_screen(
    df: pd.DataFrame,
    target_col: str,
    correlation_alarm: float = 0.90,
) -> list[str]:
    """
    Flag features with unusually high correlation to the target.

    A correlation > 0.90 to the target on training data almost always
    indicates data leakage in banking datasets — not genuine signal.

    Returns list of suspicious column names (does not auto-drop).
    """
    if target_col not in df.columns:
        return []
    num_df = df.select_dtypes(include="number")
    target = df[target_col]
    suspicious = []
    for col in num_df.columns:
        if col == target_col:
            continue
        corr_val = abs(num_df[col].corr(target))
        if corr_val > correlation_alarm:
            suspicious.append(col)
            log.warning(
                "Potential leakage detected",
                feature=col,
                corr_with_target=round(float(corr_val), 4),
            )
    return suspicious


def apply_selection(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    date_col: str | None = None,
    feature_importance: pd.Series | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Run all selection filters and return filtered DataFrame + dropped columns.

    Leakage-flagged columns are logged but NOT auto-dropped — you must
    investigate and decide deliberately.
    """
    protected = {target_col, id_col, date_col} - {None}
    feature_cols = [c for c in df.columns if c not in protected]
    feat_df = df[feature_cols]

    drop = set()
    drop.update(variance_filter(feat_df))
    drop.update(correlation_filter(feat_df))
    if feature_importance is not None:
        drop.update(importance_filter(feature_importance))

    # Leakage screen — informational only
    leakage_screen(df[[*feature_cols, target_col]], target_col=target_col)

    drop_list = list(drop)
    log.info("Total features dropped by selection", count=len(drop_list))
    return df.drop(columns=drop_list, errors="ignore"), drop_list
