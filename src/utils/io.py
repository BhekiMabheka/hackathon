"""
IO utilities — unified file reading with format detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

SUPPORTED = {".csv", ".parquet", ".xlsx", ".xls", ".json"}


def read_parquet_or_csv(
    path: str | Path,
    nrows: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load a data file with automatic format detection.

    Supports: .parquet, .csv, .xlsx/.xls, .json
    Raises ValueError for unsupported extensions.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {SUPPORTED}")

    log.info("Reading file", path=str(path), format=suffix, nrows=nrows)

    if suffix == ".parquet":
        df = pd.read_parquet(path, **kwargs)
        if nrows:
            df = df.head(nrows)
    elif suffix == ".csv":
        df = pd.read_csv(path, nrows=nrows, **kwargs)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, nrows=nrows, **kwargs)
    elif suffix == ".json":
        df = pd.read_json(path, **kwargs)
        if nrows:
            df = df.head(nrows)
    else:
        raise ValueError(f"Unhandled extension: {suffix}")

    return df


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
