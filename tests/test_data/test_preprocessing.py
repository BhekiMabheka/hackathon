"""Tests for Preprocessor — including loan-dataset-specific transforms."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocessing import Preprocessor


def test_grade_subgrade_parsing(raw_loan_df):
    """grade_subgrade strings should be parsed into grade_letter + grade_num columns."""
    import numpy as np
    df = raw_loan_df.copy()
    # Use varied values so neither parsed column is constant (constant cols are dropped)
    rng = np.random.default_rng(0)
    grades = ["A1", "B5", "C3", "D2", "E4", "F1", "G3"]
    df["grade_subgrade"] = rng.choice(grades, size=len(df))
    df.insert(0, "loan_id", [f"L{i:06d}" for i in range(len(df))])

    pp = Preprocessor(target_col="loan_paid_back", id_col="loan_id")
    result = pp.fit_transform(df)

    assert "grade_letter" in result.columns, f"grade_letter missing; cols={result.columns.tolist()}"
    assert "grade_num" in result.columns
    assert "grade_subgrade" not in result.columns
    # grade_letter must be 0-6 (A=0 … G=6)
    assert result["grade_letter"].between(0, 6).all()
    # grade_num must be 1-5 (or 0 for unmatched — none expected here)
    assert result["grade_num"].isin(range(6)).all()


def test_monthly_income_dropped_by_default(raw_loan_df):
    """monthly_income is redundant (= annual_income/12) and should be dropped."""
    df = raw_loan_df.copy()
    df.insert(0, "loan_id", [f"L{i:06d}" for i in range(len(df))])
    pp = Preprocessor(target_col="loan_paid_back", id_col="loan_id", drop_redundant=True)
    result = pp.fit_transform(df)
    assert "monthly_income" not in result.columns


def test_monthly_income_kept_when_not_dropping(raw_loan_df):
    df = raw_loan_df.copy()
    df.insert(0, "loan_id", [f"L{i:06d}" for i in range(len(df))])
    pp = Preprocessor(target_col="loan_paid_back", id_col="loan_id", drop_redundant=False)
    result = pp.fit_transform(df)
    assert "monthly_income" in result.columns


def test_installment_retained_by_default(raw_loan_df):
    """installment is leaky but kept by default (drop_leaky=False)."""
    df = raw_loan_df.copy()
    df.insert(0, "loan_id", [f"L{i:06d}" for i in range(len(df))])
    pp = Preprocessor(target_col="loan_paid_back", id_col="loan_id", drop_leaky=False)
    result = pp.fit_transform(df)
    assert "installment" in result.columns


def test_installment_dropped_when_leaky(raw_loan_df):
    df = raw_loan_df.copy()
    df.insert(0, "loan_id", [f"L{i:06d}" for i in range(len(df))])
    pp = Preprocessor(target_col="loan_paid_back", id_col="loan_id", drop_leaky=True)
    result = pp.fit_transform(df)
    assert "installment" not in result.columns


def test_transform_matches_fit_transform(raw_loan_df):
    """Calling transform on the same data must give identical output to fit_transform."""
    df = raw_loan_df.copy()
    df.insert(0, "loan_id", [f"L{i:06d}" for i in range(len(df))])
    pp = Preprocessor(target_col="loan_paid_back", id_col="loan_id")
    fit_result = pp.fit_transform(df.copy())
    transform_result = pp.transform(df.copy())
    pd.testing.assert_frame_equal(
        fit_result.reset_index(drop=True),
        transform_result.reset_index(drop=True),
    )


def test_no_nulls_after_preprocessing(raw_loan_df):
    """After imputation, no numeric or categorical feature should have nulls."""
    df = raw_loan_df.copy()
    df.insert(0, "loan_id", [f"L{i:06d}" for i in range(len(df))])
    pp = Preprocessor(target_col="loan_paid_back", id_col="loan_id")
    result = pp.fit_transform(df)
    feature_cols = [c for c in result.columns if c not in ("loan_id", "loan_paid_back")]
    assert result[feature_cols].isnull().sum().sum() == 0
