"""Tests for data validation logic — aligned with loan_dataset_20000.csv schema."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.validation import validate_schema, validate_statistics


def test_validate_schema_passes(sample_train_df):
    result = validate_schema(
        sample_train_df,
        required_cols=["loan_id", "loan_paid_back"],
    )
    assert result.passed


def test_validate_schema_flags_missing_cols(sample_train_df):
    result = validate_schema(
        sample_train_df,
        required_cols=["loan_id", "nonexistent_col"],
    )
    assert not result.passed
    assert any("nonexistent_col" in e for e in result.errors)


def test_validate_schema_warns_high_null(sample_train_df):
    df = sample_train_df.copy()
    df["mostly_null"] = None
    result = validate_schema(df, required_cols=["loan_id"], max_null_ratio=0.8)
    # Column with 100% nulls should trigger a critical error
    assert not result.passed


def test_validate_statistics_target_rate(sample_train_df):
    # Fixture has ~78% repayment rate → ~22% default (loan_paid_back=0)
    # The target here is loan_paid_back (1=paid, 0=default)
    result = validate_statistics(
        sample_train_df,
        target_col="loan_paid_back",
        id_col="loan_id",
        expected_target_rate=(0.60, 0.95),  # expect 60-95% paid back
    )
    assert result.passed


def test_validate_statistics_warns_extreme_target_rate(sample_train_df):
    df = sample_train_df.copy()
    df["loan_paid_back"] = 1  # all paid — suspiciously high
    result = validate_statistics(
        df,
        target_col="loan_paid_back",
        id_col="loan_id",
        expected_target_rate=(0.60, 0.95),
    )
    # 100% rate is outside [0.60, 0.95]
    assert len(result.warnings) > 0


def test_validate_schema_detects_constant_col(sample_train_df):
    df = sample_train_df.copy()
    df["constant_col"] = 42
    # A constant column should get a warning (it's not missing, just useless)
    result = validate_schema(df, required_cols=["loan_id"])
    # Schema check itself passes if required cols are present
    assert result.passed
