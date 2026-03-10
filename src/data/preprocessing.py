"""
Preprocessing — schema-level transformations applied identically to train and test.

Rules:
  - All transforms are fit on train only, applied to test (no leakage).
  - A PreprocessingMeta object is serialised after fitting so inference can
    replay identical transforms without re-fitting.

Loan dataset specifics:
  - grade_subgrade ("B5") → grade_letter ("B") + grade_num (5)
    Grade letter is ordinal (A best → G worst); subgrade refines within grade.
  - installment is mathematically derived: installment = f(loan_amount, interest_rate, loan_term).
    Keeping it risks leaking rate/term signal in a disguised form. Controlled via
    params.yaml:features.drop_leaky — default False (keep, but flagged).
  - monthly_income = annual_income / 12 — redundant; dropped by default.

Run standalone: python -m src.data.preprocessing
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.utils.logging import get_logger

log = get_logger(__name__)

# Grade ordering for ordinal encoding (A = lowest risk, G = highest)
GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]


@dataclass
class PreprocessingMeta:
    """Serialisable record of all fit state needed to replay preprocessing."""
    drop_cols: list[str] = field(default_factory=list)
    fill_values: dict[str, Any] = field(default_factory=dict)
    ordinal_encoder_categories: dict[str, list] = field(default_factory=dict)
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    bool_cols: list[str] = field(default_factory=list)
    grade_subgrade_parsed: bool = False

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
    >>> pp = Preprocessor(target_col="loan_paid_back", id_col="loan_id")
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
        drop_redundant: bool = True,   # drop monthly_income
        drop_leaky: bool = False,       # drop installment
    ) -> None:
        self.target_col = target_col
        self.id_col = id_col
        self.date_col = date_col
        self.max_null_ratio = max_null_ratio
        self.max_cardinality = max_cardinality
        self.drop_redundant = drop_redundant
        self.drop_leaky = drop_leaky
        self.meta = PreprocessingMeta()
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("Fitting preprocessor on train data", rows=len(df))
        df = df.copy()

        # 1. Loan-specific: parse grade_subgrade before anything else
        df = self._parse_grade_subgrade(df)
        self.meta.grade_subgrade_parsed = True

        # 2. Drop leaky / redundant columns
        explicit_drops = self._explicit_drop_list(df)
        self.meta.drop_cols = list(set(self._find_drop_cols(df) + explicit_drops))
        df = df.drop(columns=[c for c in self.meta.drop_cols if c in df.columns])
        log.info("Dropped columns", dropped=self.meta.drop_cols)

        # 3. Classify columns
        protected = {self.target_col, self.id_col, self.date_col} - {None}
        feature_cols = [c for c in df.columns if c not in protected]
        self.meta.numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
        self.meta.categorical_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
        self.meta.bool_cols = df[feature_cols].select_dtypes(include="bool").columns.tolist()

        # 4. Imputation — fit on train
        self.meta.fill_values = self._compute_fill_values(df)
        df = self._apply_fill(df)

        # 5. Categorical encoding — fit on train
        df = self._fit_encode_categoricals(df)

        # 6. Boolean to int
        for c in self.meta.bool_cols:
            if c in df.columns:
                df[c] = df[c].astype(int)

        self._fitted = True
        log.info(
            "Fit complete",
            numeric=len(self.meta.numeric_cols),
            categorical=len(self.meta.categorical_cols),
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform.")
        df = df.copy()
        df = self._parse_grade_subgrade(df)
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
    # Loan-specific transforms
    # ------------------------------------------------------------------

    def _parse_grade_subgrade(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse 'grade_subgrade' column (e.g. "B5") into two features:
          - grade_letter: ordinal-encoded A=0 … G=6 (lower risk to higher)
          - grade_num:    integer 1-5 (subgrade within the letter band)
        The original grade_subgrade column is dropped after parsing.
        """
        col = "grade_subgrade"
        if col not in df.columns:
            return df

        # Extract letter and number from strings like "B5", "AA", "F1"
        parsed = df[col].astype(str).str.extract(r"^([A-Ga-g]+)(\d)?$")
        df["grade_letter"] = parsed[0].str.upper().map(
            {g: i for i, g in enumerate(GRADE_ORDER)}
        ).fillna(-1).astype(int)
        df["grade_num"] = parsed[1].fillna(0).astype(int)
        df = df.drop(columns=[col])
        return df

    def _explicit_drop_list(self, df: pd.DataFrame) -> list[str]:
        """Drop dataset-specific redundant/leaky columns."""
        drops = []
        if self.drop_redundant and "monthly_income" in df.columns:
            drops.append("monthly_income")
            log.info("Dropping redundant column", col="monthly_income", reason="= annual_income / 12")

        if self.drop_leaky and "installment" in df.columns:
            drops.append("installment")
            log.warning(
                "Dropping leaky column",
                col="installment",
                reason="derived from loan_amount * f(interest_rate, loan_term)",
            )
        elif "installment" in df.columns:
            log.warning(
                "Leakage risk retained",
                col="installment",
                note="installment is derived from loan_amount/interest_rate/loan_term. "
                     "Set features.drop_leaky=true in params.yaml to exclude.",
            )
        return drops

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_drop_cols(self, df: pd.DataFrame) -> list[str]:
        protected = {self.target_col, self.id_col, self.date_col} - {None}
        drop = []
        null_ratio = df.isnull().mean()
        drop += null_ratio[null_ratio > self.max_null_ratio].index.tolist()

        for c in df.columns:
            if c in protected:
                continue
            if df[c].nunique(dropna=False) <= 1:
                drop.append(c)

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for c in cat_cols:
            if c in protected:
                continue
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

    from src.data.ingestion import load_single_file
    from src.data.splits import train_val_holdout

    d = params["data"]
    df = load_single_file(d["raw_path"], d["train_file"], id_col=d["id_col"])

    # For standalone run: split first, then preprocess
    train, val, holdout = train_val_holdout(
        df,
        target_col=d["target_col"],
        date_col=None,
        test_size=d["test_size"],
        val_size=d["val_size"],
        random_state=d["random_state"],
    )

    feat_params = params["features"]
    pp = Preprocessor(
        target_col=d["target_col"],
        id_col=d["id_col"],
        date_col=None,
        max_null_ratio=feat_params["drop_high_null_thresh"],
        drop_redundant=feat_params.get("drop_redundant", True),
        drop_leaky=feat_params.get("drop_leaky", False),
    )
    train_out = pp.fit_transform(train)
    val_out = pp.transform(val)
    holdout_out = pp.transform(holdout)
    pp.save_meta("data/interim/preprocessing_meta.json")

    Path("data/interim").mkdir(parents=True, exist_ok=True)
    train_out.to_parquet("data/interim/train.parquet", index=False)
    val_out.to_parquet("data/interim/val.parquet", index=False)
    holdout_out.to_parquet("data/interim/holdout.parquet", index=False)
    log.info(
        "Preprocessing complete",
        train_shape=train_out.shape,
        val_shape=val_out.shape,
        holdout_shape=holdout_out.shape,
    )
