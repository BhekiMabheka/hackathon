"""
Train/validation splitting strategies for banking time-series data.

Key design decisions:
  - stratified_time:  Stratified K-Fold that respects temporal ordering.
    Each fold's validation set comes from a later period than its training
    set, preventing look-ahead bias while still stratifying on the target.
  - time_series:      Pure walk-forward CV — no shuffling, strict temporal
    ordering. Use when seasonality / trends are strong.
  - stratified_kfold: Standard stratified CV — only use when the data has
    no meaningful temporal structure.

Banking note: ALWAYS prefer temporal splits when a date column exists.
Standard K-Fold on transaction data almost always leaks future information.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from src.utils.logging import get_logger

log = get_logger(__name__)

FoldIndices = tuple[np.ndarray, np.ndarray]


def get_cv_splits(
    df: pd.DataFrame,
    target_col: str,
    strategy: str = "stratified_time",
    n_folds: int = 5,
    date_col: str | None = None,
    random_state: int = 42,
) -> list[FoldIndices]:
    """
    Return a list of (train_idx, val_idx) index pairs.

    Parameters
    ----------
    df:           Full training DataFrame.
    target_col:   Binary target column.
    strategy:     'stratified_time' | 'time_series' | 'stratified_kfold'
    n_folds:      Number of CV folds.
    date_col:     Date column used for temporal splits.
    random_state: Seed for reproducibility.
    """
    if strategy == "stratified_time":
        if date_col is None:
            log.warning("stratified_time requires date_col — falling back to stratified_kfold")
            strategy = "stratified_kfold"
        else:
            return list(_stratified_time_splits(df, target_col, date_col, n_folds, random_state))

    if strategy == "time_series":
        if date_col is None:
            raise ValueError("time_series strategy requires date_col")
        return list(_time_series_splits(df, date_col, n_folds))

    # Default: standard stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return list(skf.split(df, df[target_col]))


def _stratified_time_splits(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    n_folds: int,
    random_state: int,
) -> Iterator[FoldIndices]:
    """
    Stratified splits where each validation period is strictly after training.

    Implementation:
      - Sort by date, create n_folds temporal buckets.
      - For fold k: train = buckets [0..k-1], val = bucket k.
      - Stratification is preserved within each bucket by oversampling
        the minority class in the training window.
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    bucket_labels = pd.qcut(
        df_sorted[date_col].rank(method="first"),
        q=n_folds,
        labels=False,
    )

    for fold in range(1, n_folds + 1):
        train_mask = bucket_labels < fold
        val_mask = bucket_labels == fold

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        pos_rate_train = df_sorted.iloc[train_idx][target_col].mean()
        pos_rate_val = df_sorted.iloc[val_idx][target_col].mean()
        log.info(
            "Fold split",
            fold=fold,
            train_n=len(train_idx),
            val_n=len(val_idx),
            train_pos_rate=round(float(pos_rate_train), 4),
            val_pos_rate=round(float(pos_rate_val), 4),
        )
        yield train_idx, val_idx


def _time_series_splits(
    df: pd.DataFrame,
    date_col: str,
    n_folds: int,
) -> Iterator[FoldIndices]:
    """Pure walk-forward split using sklearn TimeSeriesSplit."""
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    tss = TimeSeriesSplit(n_splits=n_folds)
    for train_idx, val_idx in tss.split(df_sorted):
        yield train_idx, val_idx


def train_val_holdout(
    df: pd.DataFrame,
    target_col: str,
    date_col: str | None = None,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train / val / holdout.

    If date_col is provided, split is temporal (no shuffling).
    Otherwise split is random stratified.
    """
    if date_col:
        df_s = df.sort_values(date_col).reset_index(drop=True)
        n = len(df_s)
        n_holdout = int(n * test_size)
        n_val = int(n * val_size)
        holdout = df_s.iloc[n - n_holdout:]
        val = df_s.iloc[n - n_holdout - n_val: n - n_holdout]
        train = df_s.iloc[: n - n_holdout - n_val]
    else:
        from sklearn.model_selection import train_test_split
        remaining, holdout = train_test_split(
            df, test_size=test_size, stratify=df[target_col], random_state=random_state
        )
        train, val = train_test_split(
            remaining,
            test_size=val_size / (1 - test_size),
            stratify=remaining[target_col],
            random_state=random_state,
        )

    log.info(
        "Train/val/holdout split",
        train_n=len(train),
        val_n=len(val),
        holdout_n=len(holdout),
    )
    return train, val, holdout
