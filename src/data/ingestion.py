"""
Data ingestion — loads raw competition data into a canonical DataFrame.

Responsibilities:
  - Discover and load CSV/Parquet/Excel from raw data directory
  - Apply basic type coercions from data dictionary
  - Emit a structured log entry with row/column counts + null summary
  - Never mutate business logic here (that belongs in preprocessing.py)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logging import get_logger
from src.utils.io import read_parquet_or_csv

log = get_logger(__name__)


def load_raw(
    raw_dir: str | Path,
    filename: str,
    date_col: Optional[str] = None,
    id_col: Optional[str] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a single raw data file.

    Parameters
    ----------
    raw_dir:   Directory containing raw data files.
    filename:  File name (csv, parquet, or xlsx).
    date_col:  If provided, parse this column as datetime.
    id_col:    If provided, ensure this column is string type.
    nrows:     Optional row limit for debug runs.

    Returns
    -------
    pd.DataFrame with at least the id and date columns correctly typed.
    """
    path = Path(raw_dir) / filename
    log.info("Loading raw data", path=str(path), nrows=nrows)

    df = read_parquet_or_csv(path, nrows=nrows)

    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
        log.info("Parsed date column", col=date_col, dtype=str(df[date_col].dtype))

    if id_col and id_col in df.columns:
        df[id_col] = df[id_col].astype(str)

    _log_shape(df, label=filename)
    return df


def load_train_test(
    raw_dir: str | Path,
    train_file: str = "train.csv",
    test_file: str = "test.csv",
    date_col: Optional[str] = None,
    id_col: Optional[str] = None,
    nrows: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper to load both train and test splits."""
    train = load_raw(raw_dir, train_file, date_col=date_col, id_col=id_col, nrows=nrows)
    test = load_raw(raw_dir, test_file, date_col=date_col, id_col=id_col, nrows=nrows)
    return train, test


def audit_load(df: pd.DataFrame, save_path: Optional[str | Path] = None) -> dict:
    """
    Produce a structured audit record of a loaded DataFrame.
    Useful for detecting data drift between competition data drops.
    """
    audit = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "null_ratio": (df.isnull().mean().round(4)).to_dict(),
    }
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(audit, f, indent=2)
        log.info("Saved ingestion audit", path=str(save_path))
    return audit


def _log_shape(df: pd.DataFrame, label: str) -> None:
    null_pct = df.isnull().mean().mul(100).round(1)
    high_null_cols = null_pct[null_pct > 20].to_dict()
    log.info(
        "Loaded dataframe",
        label=label,
        rows=len(df),
        cols=len(df.columns),
        high_null_cols=high_null_cols,
    )
