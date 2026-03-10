"""Tests for data validation logic."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.validation import validate_schema, validate_statistics, validate_no_leakage


def test_validate_schema_passes(sample_train_df):
    result = validate_schema(
        sample_train_df,
        required_cols=["customer_id", "target"],
    )
    assert result.passed


def test_validate_schema_flags_missing_cols(sample_train_df):
    result = validate_schema(
        sample_train_df,
        required_cols=["customer_id", "nonexistent_col"],
    )
    assert not result.passed
    assert any("nonexistent_col" in e for e in result.errors)


def test_validate_schema_warns_high_null(sample_train_df):
    df = sample_train_df.copy()
    df["mostly_null"] = None
    result = validate_schema(df, required_cols=["customer_id"], max_null_ratio=0.8)
    # Column with 100% nulls should trigger an error
    assert not result.passed


def test_validate_statistics_target_rate(sample_train_df):
    result = validate_statistics(
        sample_train_df,
        target_col="target",
        id_col="customer_id",
        expected_target_rate=(0.01, 0.20),
    )
    # ~8% target rate — should pass [1%, 20%] range
    assert result.passed


def test_validate_no_leakage_detects_date_overlap(sample_train_df, sample_test_df):
    # Force overlap: set test dates to same as train dates
    sample_test_df_copy = sample_test_df.copy()
    sample_test_df_copy["transaction_date"] = sample_train_df["transaction_date"]
    result = validate_no_leakage(
        sample_train_df,
        sample_test_df_copy,
        id_col="customer_id",
        date_col="transaction_date",
    )
    # Should detect temporal overlap
    assert not result.passed


def test_validate_no_leakage_clean_temporal_split(sample_train_df, sample_test_df):
    result = validate_no_leakage(
        sample_train_df,
        sample_test_df,
        id_col="customer_id",
        date_col="transaction_date",
    )
    # Test dates are 1 year after train dates — no overlap
    assert result.passed
