"""Tests for credit-risk feature engineering functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    add_credit_utilisation,
    add_affordability_features,
    add_grade_features,
    add_delinquency_features,
    add_loan_structure_features,
    add_income_features,
    add_risk_flags,
    FeatureEngineer,
)


def test_credit_utilisation_added(sample_train_df):
    df = add_credit_utilisation(sample_train_df)
    assert "feat_credit_util" in df.columns
    assert "feat_high_util_flag" in df.columns
    assert df["feat_high_util_flag"].isin([0, 1]).all()


def test_credit_utilisation_no_division_zero(sample_train_df):
    df = sample_train_df.copy()
    df["total_credit_limit"] = 0
    result = add_credit_utilisation(df)
    assert not result["feat_credit_util"].isnull().any()
    assert not np.isinf(result["feat_credit_util"]).any()


def test_affordability_features_added(sample_train_df):
    df = add_affordability_features(sample_train_df)
    assert "feat_loan_to_income" in df.columns
    assert "feat_dti_band" in df.columns
    assert "feat_high_dti_flag" in df.columns
    assert df["feat_high_dti_flag"].isin([0, 1]).all()


def test_grade_features_added(sample_train_df):
    df = add_grade_features(sample_train_df)
    assert "feat_grade_position" in df.columns
    # grade_position = grade_letter * 5 + grade_num; grade_letter 0-6, grade_num 1-5
    assert df["feat_grade_position"].between(0, 35).all()


def test_delinquency_features_added(sample_train_df):
    df = add_delinquency_features(sample_train_df)
    assert "feat_total_delinquency" in df.columns
    assert "feat_any_delinquency" in df.columns
    assert "feat_severe_delinquency" in df.columns
    assert df["feat_any_delinquency"].isin([0, 1]).all()
    # feat_total_delinquency >= 0
    assert (df["feat_total_delinquency"] >= 0).all()


def test_delinquency_per_account_no_zero_division(sample_train_df):
    df = sample_train_df.copy()
    df["num_of_open_accounts"] = 0
    result = add_delinquency_features(df)
    assert not result["feat_delinq_per_account"].isnull().any()


def test_loan_structure_credit_bands(sample_train_df):
    df = add_loan_structure_features(sample_train_df)
    assert "feat_credit_band" in df.columns
    assert "feat_subprime" in df.columns
    assert "feat_prime" in df.columns
    assert df["feat_subprime"].isin([0, 1]).all()
    # subprime and prime should not both be 1 for the same row
    both = (df["feat_subprime"] == 1) & (df["feat_prime"] == 1)
    assert not both.any(), "A loan cannot be both subprime and prime"


def test_income_log_features(sample_train_df):
    df = add_income_features(sample_train_df)
    assert "feat_log_income" in df.columns
    assert "feat_log_loan_amount" in df.columns
    # log1p of positive values should all be positive
    assert (df["feat_log_income"] > 0).all()


def test_risk_flags_added(sample_train_df):
    df = add_delinquency_features(sample_train_df)  # create feat_any_delinquency first
    df = add_risk_flags(df)
    assert "feat_score_grade_mismatch" in df.columns
    assert (df["feat_score_grade_mismatch"] >= 0).all()


def test_feature_engineer_fit_transform(sample_train_df):
    fe = FeatureEngineer(config={
        "id_col": "loan_id",
        "target_col": "loan_paid_back",
        "date_col": None,
    })
    result = fe.fit_transform(sample_train_df)
    # Should have more columns than input
    assert result.shape[1] > sample_train_df.shape[1]
    # Feature cols should not include id or target
    assert "loan_id" not in fe.feature_cols
    assert "loan_paid_back" not in fe.feature_cols


def test_feature_engineer_no_nulls_introduced(sample_train_df):
    """Feature engineering must not introduce NaNs in non-null input columns."""
    fe = FeatureEngineer(config={
        "id_col": "loan_id",
        "target_col": "loan_paid_back",
        "date_col": None,
    })
    result = fe.fit_transform(sample_train_df)
    feat_cols = [c for c in result.columns if c.startswith("feat_")]
    null_counts = result[feat_cols].isnull().sum()
    assert null_counts.sum() == 0, f"Null values introduced: {null_counts[null_counts > 0].to_dict()}"


def test_feature_engineer_transform_matches_fit_transform(sample_train_df):
    """transform() on the same data should produce identical results to fit_transform()."""
    fe = FeatureEngineer(config={
        "id_col": "loan_id",
        "target_col": "loan_paid_back",
        "date_col": None,
    })
    result_fit = fe.fit_transform(sample_train_df.copy())
    result_transform = fe.transform(sample_train_df.copy())
    feat_cols = [c for c in result_fit.columns if c.startswith("feat_")]
    pd.testing.assert_frame_equal(
        result_fit[feat_cols].reset_index(drop=True),
        result_transform[feat_cols].reset_index(drop=True),
    )
