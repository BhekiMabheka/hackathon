"""
Feature engineering — credit risk domain features for loan default prediction.

This dataset is cross-sectional (no date column), so all features are
derived from the static loan application snapshot.

Feature groups:
  1. Credit utilisation         — balance vs. total credit limit
  2. Affordability ratios       — loan_amount vs. income, installment vs. income
  3. Risk grade features        — grade_letter / grade_num interactions
  4. Delinquency aggregates     — combining delinquency_history + num_of_delinquencies
  5. Loan structure             — interaction of loan_term × interest_rate
  6. Income / leverage          — log transforms, income bands
  7. Compound risk flags        — binary indicators from domain knowledge

Design rules:
  - No fit state needed for ratio/interaction features (safe for train+test).
  - New feature names are prefixed with 'feat_' for easy identification.
  - Called AFTER preprocessing so grade_letter/grade_num already exist.

Run standalone: python -m src.features.engineering
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

EPS = 1e-6  # division guard


# ---------------------------------------------------------------------------
# 1. Credit utilisation
# ---------------------------------------------------------------------------

def add_credit_utilisation(df: pd.DataFrame) -> pd.DataFrame:
    """current_balance / total_credit_limit — core credit bureau metric."""
    if "current_balance" in df.columns and "total_credit_limit" in df.columns:
        df["feat_credit_util"] = df["current_balance"] / (df["total_credit_limit"] + EPS)
        df["feat_credit_util_capped"] = df["feat_credit_util"].clip(0, 1)
        df["feat_high_util_flag"] = (df["feat_credit_util"] > 0.8).astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. Affordability ratios
# ---------------------------------------------------------------------------

def add_affordability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loan amount and repayment burden relative to income.

    installment may be absent (controlled by params.yaml:features.drop_leaky).
    A first-principles approximation is computed regardless.
    """
    if "loan_amount" in df.columns and "annual_income" in df.columns:
        df["feat_loan_to_income"] = df["loan_amount"] / (df["annual_income"] + EPS)

    if "installment" in df.columns and "annual_income" in df.columns:
        monthly_income = df["annual_income"] / 12
        df["feat_installment_to_monthly"] = df["installment"] / (monthly_income + EPS)

    if "loan_amount" in df.columns and "annual_income" in df.columns and "loan_term" in df.columns:
        # Simple monthly repayment approximation (no interest) — leakage-free alternative
        monthly_income_approx = df["annual_income"] / 12
        simple_monthly_payment = df["loan_amount"] / (df["loan_term"] + EPS)
        df["feat_simple_payment_to_income"] = simple_monthly_payment / (monthly_income_approx + EPS)

    if "debt_to_income_ratio" in df.columns:
        df["feat_dti_band"] = pd.cut(
            df["debt_to_income_ratio"],
            bins=[0, 0.1, 0.2, 0.35, 0.5, np.inf],
            labels=[0, 1, 2, 3, 4],
            right=False,
        ).astype(float)
        df["feat_high_dti_flag"] = (df["debt_to_income_ratio"] > 0.35).astype(int)

    return df


# ---------------------------------------------------------------------------
# 3. Risk grade features
# ---------------------------------------------------------------------------

def add_grade_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interactions between grade_letter, grade_num, and interest_rate.
    grade_letter is ordinal 0=A (safest) … 6=G (riskiest) — set in preprocessing.
    """
    if "grade_letter" not in df.columns:
        return df

    if "interest_rate" in df.columns:
        grade_avg_rate = df.groupby("grade_letter")["interest_rate"].transform("mean")
        df["feat_rate_vs_grade_avg"] = df["interest_rate"] - grade_avg_rate

    if "grade_num" in df.columns:
        df["feat_grade_position"] = df["grade_letter"] * 5 + df["grade_num"]

    return df


# ---------------------------------------------------------------------------
# 4. Delinquency aggregates
# ---------------------------------------------------------------------------

def add_delinquency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine and normalise delinquency signals."""
    if "delinquency_history" in df.columns and "num_of_delinquencies" in df.columns:
        df["feat_total_delinquency"] = (
            df["delinquency_history"] + df["num_of_delinquencies"]
        )
        df["feat_any_delinquency"] = (df["feat_total_delinquency"] > 0).astype(int)
        df["feat_severe_delinquency"] = (df["feat_total_delinquency"] >= 3).astype(int)

        if "num_of_open_accounts" in df.columns:
            df["feat_delinq_per_account"] = df["feat_total_delinquency"] / (
                df["num_of_open_accounts"] + 1
            )

    if "public_records" in df.columns:
        df["feat_has_public_record"] = (df["public_records"] > 0).astype(int)

    return df


# ---------------------------------------------------------------------------
# 5. Loan structure features
# ---------------------------------------------------------------------------

def add_loan_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction features between loan amount, term, and interest rate."""
    if "interest_rate" in df.columns and "loan_term" in df.columns:
        df["feat_total_interest_burden"] = (
            df["interest_rate"] / 100 * df["loan_term"] / 12
        )
        df["feat_long_high_rate"] = (
            (df["loan_term"] >= 60) & (df["interest_rate"] > 15)
        ).astype(int)

    if "loan_amount" in df.columns and "loan_term" in df.columns:
        df["feat_loan_per_term_month"] = df["loan_amount"] / (df["loan_term"] + EPS)

    if "credit_score" in df.columns:
        df["feat_credit_band"] = pd.cut(
            df["credit_score"],
            bins=[0, 580, 670, 740, 800, np.inf],
            labels=[0, 1, 2, 3, 4],
            right=False,
        ).astype(float)
        df["feat_subprime"] = (df["credit_score"] < 620).astype(int)
        df["feat_prime"] = (df["credit_score"] >= 740).astype(int)

    return df


# ---------------------------------------------------------------------------
# 6. Income & leverage features
# ---------------------------------------------------------------------------

def add_income_features(df: pd.DataFrame) -> pd.DataFrame:
    if "annual_income" in df.columns:
        df["feat_log_income"] = np.log1p(df["annual_income"])
        df["feat_income_band"] = pd.cut(
            df["annual_income"],
            bins=[0, 20_000, 40_000, 70_000, 120_000, np.inf],
            labels=[0, 1, 2, 3, 4],
            right=False,
        ).astype(float)

    if "current_balance" in df.columns:
        df["feat_log_balance"] = np.log1p(df["current_balance"].clip(0))

    if "loan_amount" in df.columns:
        df["feat_log_loan_amount"] = np.log1p(df["loan_amount"])

    return df


# ---------------------------------------------------------------------------
# 7. Compound risk flags
# ---------------------------------------------------------------------------

def add_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary domain-knowledge flags capturing compound risk scenarios."""
    if "debt_to_income_ratio" in df.columns and "feat_any_delinquency" in df.columns:
        df["feat_dti_x_delinquency"] = (
            df["debt_to_income_ratio"] * df["feat_any_delinquency"]
        )

    if "credit_score" in df.columns and "grade_letter" in df.columns:
        df["feat_score_x_grade"] = df["credit_score"] * df["grade_letter"]
        # Flag mismatches: low score but good grade (or vice versa)
        score_norm = df["credit_score"] / 850
        grade_norm = df["grade_letter"] / 6
        df["feat_score_grade_mismatch"] = (score_norm - (1 - grade_norm)).abs()

    return df


# ---------------------------------------------------------------------------
# Main FeatureEngineer class
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Orchestrates all credit-risk feature creation steps.

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
        df = self._build_features(df.copy())
        protected = {
            self.config.get("id_col"),
            self.config.get("target_col"),
            self.config.get("date_col"),
        } - {None}
        self.feature_cols = [c for c in df.columns if c not in protected]
        self._meta = {"feature_cols": self.feature_cols, "config": self.config}
        log.info("Feature engineering complete", n_features=len(self.feature_cols))
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._build_features(df.copy())

    def save_meta(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._meta, f, indent=2)
        log.info("Saved feature meta", path=str(path), n_features=len(self.feature_cols))

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_credit_utilisation(df)
        df = add_affordability_features(df)
        df = add_grade_features(df)
        df = add_delinquency_features(df)
        df = add_loan_structure_features(df)
        df = add_income_features(df)
        df = add_risk_flags(df)
        return df


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    train = pd.read_parquet("data/interim/train.parquet")
    holdout_path = Path("data/interim/holdout.parquet")
    test = pd.read_parquet(holdout_path) if holdout_path.exists() else pd.read_parquet("data/interim/val.parquet")

    fe_config = {**params["data"], **params["features"]}
    fe = FeatureEngineer(config=fe_config)
    train_out = fe.fit_transform(train)
    test_out = fe.transform(test)
    fe.save_meta("data/processed/feature_meta.json")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    train_out.to_parquet("data/processed/train_features.parquet", index=False)
    test_out.to_parquet("data/processed/test_features.parquet", index=False)
    log.info("Done", train_shape=train_out.shape, test_shape=test_out.shape)
