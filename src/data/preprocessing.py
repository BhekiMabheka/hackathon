"""
Preprocessing — schema-level transformations applied identically to train and test.

Rules:
  - All transforms are fit on train only, applied to test (no leakage).
  - A PreprocessingMeta object is serialised after fitting so inference can
    replay identical transforms without re-fitting.
  - Banking-specific: anonymised data may have numeric codes for categorical
    variables — handle gracefully.

Run standalone: python -m src.data.preprocessing
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PreprocessingMeta:
    """Serialisable record of all fit state needed to replay preprocessing."""
    drop_cols: list[str] = field(default_factory=list)
    fill_values: dict[str, Any] = field(default_factory=dict)
    ordinal_encoder_categories: dict[str, list] = field(default_factory=dict)
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    bool_cols: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "PreprocessingMeta":
        with open(path) as f:
            return cls(**json.load(f))


class Preprocessor:
    """
    Stateful preprocessor that fits on train and transforms both train + test.

    Usage
    -----
    >>> pp = Preprocessor(target_col="target", id_col="customer_id",
    ...                   date_col="transaction_date", max_null_ratio=0.8)
    >>> train_out = pp.fit_transform(train_df)
    >>> test_out  = pp.transform(test_df)
    >>> pp.save_meta("data/interim/preprocessing_meta.json")
    """

    def __init__(
        self,
        target_col: str,
        id_col: str,
        date_col: Optional[str] = None,
        max_null_ratio: float = 0.80,
        max_cardinality: int = 500,
    ) -> None:
        self.target_col = target_col
        self.id_col = id_col
        self.date_col = date_col
        self.max_null_ratio = max_null_ratio
        self.max_cardinality = max_cardinality
        self.meta = PreprocessingMeta()
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("Fitting preprocessor on train data", rows=len(df))
        df = df.copy()

        # 1. Identify columns to drop (high null, constants, identity-like)
        self.meta.drop_cols = self._find_drop_cols(df)
        df = df.drop(columns=self.meta.drop_cols)
        log.info("Dropped columns", dropped=self.meta.drop_cols)

        # 2. Classify columns
        feature_cols = [c for c in df.columns if c not in (self.target_col, self.id_col, self.date_col)]
        self.meta.numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
        self.meta.categorical_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
        self.meta.bool_cols = df[feature_cols].select_dtypes(include="bool").columns.tolist()

        # 3. Imputation — fit on train
        self.meta.fill_values = self._compute_fill_values(df)
        df = self._apply_fill(df)

        # 4. Categorical encoding — fit on train
        df = self._fit_encode_categoricals(df)

        # 5. Boolean to int
        for c in self.meta.bool_cols:
            if c in df.columns:
                df[c] = df[c].astype(int)

        self._fitted = True
        log.info("Fit complete", numeric=len(self.meta.numeric_cols), cat=len(self.meta.categorical_cols))
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform.")
        df = df.copy()
        df = df.drop(columns=[c for c in self.meta.drop_cols if c in df.columns])
        df = self._apply_fill(df)
        df = self._apply_encode_categoricals(df)
        for c in self.meta.bool_cols:
            if c in df.columns:
                df[c] = df[c].astype(int)
        return df

    def save_meta(self, path: str | Path) -> None:
        self.meta.save(path)
        log.info("Saved preprocessing meta", path=str(path))

    @classmethod
    def from_meta(cls, meta_path: str | Path, target_col: str, id_col: str) -> "Preprocessor":
        pp = cls(target_col=target_col, id_col=id_col)
        pp.meta = PreprocessingMeta.load(meta_path)
        pp._fitted = True
        return pp

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_drop_cols(self, df: pd.DataFrame) -> list[str]:
        drop = []
        null_ratio = df.isnull().mean()
        drop += null_ratio[null_ratio > self.max_null_ratio].index.tolist()

        # Constant columns
        for c in df.columns:
            if c in (self.target_col, self.id_col, self.date_col):
                continue
            if df[c].nunique(dropna=False) <= 1:
                drop.append(c)

        # High cardinality free text (likely anonymised IDs)
        for c in self.meta.categorical_cols if self.meta.categorical_cols else []:
            if df[c].nunique() > self.max_cardinality:
                drop.append(c)
                log.warning("Dropping high-cardinality categorical", col=c, unique=df[c].nunique())

        return list(set(drop))

    def _compute_fill_values(self, df: pd.DataFrame) -> dict[str, Any]:
        fills: dict[str, Any] = {}
        for c in self.meta.numeric_cols:
            if c in df.columns:
                fills[c] = float(df[c].median())
        for c in self.meta.categorical_cols:
            if c in df.columns:
                fills[c] = "__MISSING__"
        return fills

    def _apply_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, val in self.meta.fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        return df

    def _fit_encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        self._enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if self.meta.categorical_cols:
            cols_present = [c for c in self.meta.categorical_cols if c in df.columns]
            df[cols_present] = self._enc.fit_transform(df[cols_present].astype(str))
            self.meta.ordinal_encoder_categories = {
                col: list(cats)
                for col, cats in zip(cols_present, self._enc.categories_)
            }
        return df

    def _apply_encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.meta.categorical_cols and hasattr(self, "_enc"):
            cols_present = [c for c in self.meta.categorical_cols if c in df.columns]
            df[cols_present] = self._enc.transform(df[cols_present].astype(str))
        return df


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    from src.data.ingestion import load_train_test

    d = params["data"]
    train, test = load_train_test(
        raw_dir=d["raw_path"],
        date_col=d["date_col"],
        id_col=d["id_col"],
    )

    pp = Preprocessor(
        target_col=d["target_col"],
        id_col=d["id_col"],
        date_col=d["date_col"],
        max_null_ratio=params["features"]["drop_high_null_thresh"],
    )
    train_out = pp.fit_transform(train)
    test_out = pp.transform(test)
    pp.save_meta("data/interim/preprocessing_meta.json")

    Path("data/interim").mkdir(parents=True, exist_ok=True)
    train_out.to_parquet("data/interim/train.parquet", index=False)
    test_out.to_parquet("data/interim/test.parquet", index=False)
    log.info("Preprocessing complete", train_shape=train_out.shape, test_shape=test_out.shape)
