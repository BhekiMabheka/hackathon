"""
Feature engineering — banking-domain temporal and aggregate features.

Design principles:
  - All features are computed in a single pass per entity (customer/account).
  - Lag features use strict date ordering to avoid look-ahead.
  - Features are namespaced (e.g., txn_30d_mean) for interpretability.
  - Fit only on training window; test features use the same lookback periods.

Run standalone: python -m src.features.engineering
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Rolling / lag feature builders
# ---------------------------------------------------------------------------


def add_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    value_cols: list[str],
    windows: list[int],
    agg_fns: list[str] = ("mean", "std", "min", "max", "sum"),
) -> pd.DataFrame:
    """
    Add rolling window aggregate features per entity.

    Parameters
    ----------
    df:         DataFrame sorted by (group_col, date_col).
    group_col:  Grouping key (e.g., customer_id).
    date_col:   Date column for ordering.
    value_cols: Numeric columns to aggregate.
    windows:    List of lookback windows in days.
    agg_fns:    Aggregation functions to apply.

    Notes
    -----
    Uses pandas rolling with min_periods=1 so rows with fewer observations
    than the window still get a value (avoid NaN explosion in early history).
    """
    df = df.sort_values([group_col, date_col]).reset_index(drop=True)

    for col in value_cols:
        if col not in df.columns:
            log.warning("Value column not found, skipping", col=col)
            continue
        for window in windows:
            grp = df.groupby(group_col)[col]
            for fn in agg_fns:
                feat_name = f"{col}_{window}d_{fn}"
                if fn == "std":
                    df[feat_name] = grp.transform(
                        lambda s, w=window: s.rolling(w, min_periods=1).std().fillna(0)
                    )
                else:
                    df[feat_name] = grp.transform(
                        lambda s, w=window, f=fn: s.rolling(w, min_periods=1).agg(f)
                    )

    log.info(
        "Added rolling features",
        cols=value_cols,
        windows=windows,
        total_new_features=len(value_cols) * len(windows) * len(agg_fns),
    )
    return df


def add_lag_features(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    value_cols: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """
    Add point-in-time lag features (value N rows back within the group).

    Note: 'rows back' is only a proxy for 'N days back' if data is daily.
    For irregular time series, prefer rolling window features instead.
    """
    df = df.sort_values([group_col, date_col]).reset_index(drop=True)
    for col in value_cols:
        if col not in df.columns:
            continue
        for lag in lags:
            feat_name = f"{col}_lag{lag}"
            df[feat_name] = df.groupby(group_col)[col].shift(lag)

    log.info("Added lag features", cols=value_cols, lags=lags)
    return df


def add_temporal_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Extract calendar features from a datetime column.

    Banking patterns often follow calendar cycles:
    month-end spikes, salary day inflows, year-end bonuses.
    """
    if date_col not in df.columns:
        log.warning("Date column not found", col=date_col)
        return df

    dt = df[date_col]
    df["dow"] = dt.dt.dayofweek                    # 0=Mon..6=Sun
    df["dom"] = dt.dt.day                          # day of month
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)

    # SA salary cycle proxy: last 3 days of month are typically high-activity
    df["is_salary_window"] = (dt.dt.day >= 25).astype(int)

    log.info("Added temporal features", date_col=date_col)
    return df


def add_ratio_features(
    df: pd.DataFrame,
    numerator_cols: list[str],
    denominator_cols: list[str],
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Add ratio features (e.g., credit/debit ratio, balance/limit).

    Division-by-zero is handled with eps.
    """
    for num, denom in zip(numerator_cols, denominator_cols):
        if num in df.columns and denom in df.columns:
            feat_name = f"ratio_{num}_over_{denom}"
            df[feat_name] = df[num] / (df[denom].abs() + eps)

    return df


def add_velocity_features(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    amount_col: str,
    windows: list[int] = (7, 30),
) -> pd.DataFrame:
    """
    Transaction velocity features — count and volume per window.
    Critical for fraud / AML risk scoring.
    """
    df = df.sort_values([group_col, date_col]).reset_index(drop=True)
    grp = df.groupby(group_col)[amount_col]
    for w in windows:
        df[f"txn_count_{w}d"] = grp.transform(
            lambda s, ww=w: s.rolling(ww, min_periods=1).count()
        )
        df[f"txn_volume_{w}d"] = grp.transform(
            lambda s, ww=w: s.rolling(ww, min_periods=1).sum()
        )
        df[f"txn_avg_{w}d"] = df[f"txn_volume_{w}d"] / (df[f"txn_count_{w}d"] + 1e-6)

    log.info("Added velocity features", amount_col=amount_col, windows=list(windows))
    return df


# ---------------------------------------------------------------------------
# Main feature pipeline
# ---------------------------------------------------------------------------


class FeatureEngineer:
    """
    Orchestrates all feature creation steps.

    Usage
    -----
    >>> fe = FeatureEngineer(config)
    >>> train_features = fe.fit_transform(train_df)
    >>> test_features  = fe.transform(test_df)
    >>> fe.save_meta("data/processed/feature_meta.json")
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.feature_cols: list[str] = []
        self._meta: dict = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("Starting feature engineering", rows=len(df))
        df = self._build_features(df)
        self.feature_cols = [
            c for c in df.columns
            if c not in (
                self.config.get("id_col"),
                self.config.get("target_col"),
                self.config.get("date_col"),
            )
        ]
        self._meta = {"feature_cols": self.feature_cols, "config": self.config}
        log.info("Feature engineering complete", n_features=len(self.feature_cols))
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply same transforms to test. No fit state needed for most features."""
        return self._build_features(df)

    def save_meta(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._meta, f, indent=2)
        log.info("Saved feature meta", path=str(path), n_features=len(self.feature_cols))

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        date_col = cfg.get("date_col")
        id_col = cfg.get("id_col")
        amount_col = cfg.get("amount_col")  # may be None until brief

        # Calendar features
        if date_col and date_col in df.columns:
            df = add_temporal_features(df, date_col)

        # Rolling aggregates over numeric cols (excluding id/target/date)
        skip = {id_col, cfg.get("target_col"), date_col}
        numeric_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in skip
        ]

        windows = cfg.get("lag_windows", [7, 14, 30, 90])
        agg_fns = cfg.get("rolling_agg_fns", ["mean", "std", "sum"])

        if id_col and date_col and numeric_cols:
            # Limit to most important numeric cols to avoid feature explosion
            top_cols = numeric_cols[:10]
            df = add_rolling_features(
                df,
                group_col=id_col,
                date_col=date_col,
                value_cols=top_cols,
                windows=windows,
                agg_fns=agg_fns,
            )

        # Velocity (if amount column available)
        if amount_col and amount_col in df.columns and id_col and date_col:
            df = add_velocity_features(df, id_col, date_col, amount_col, windows=[7, 30])

        return df


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    train = pd.read_parquet("data/interim/train.parquet")
    test = pd.read_parquet("data/interim/test.parquet")

    fe_config = {
        **params["data"],
        **params["features"],
    }
    fe = FeatureEngineer(config=fe_config)
    train_out = fe.fit_transform(train)
    test_out = fe.transform(test)
    fe.save_meta("data/processed/feature_meta.json")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    train_out.to_parquet("data/processed/train_features.parquet", index=False)
    test_out.to_parquet("data/processed/test_features.parquet", index=False)
    log.info("Done", train_shape=train_out.shape, test_shape=test_out.shape)
