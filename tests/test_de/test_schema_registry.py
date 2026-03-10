"""Tests for schema drift detection."""

from __future__ import annotations

import pandas as pd
import pytest

from src.de.schema_registry import SchemaDefinition, SchemaDriftDetector, DriftEvent


@pytest.fixture
def base_df():
    return pd.DataFrame({
        "customer_id": ["C001", "C002"],
        "amount": [100.0, 200.0],
        "transaction_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
    })


@pytest.fixture
def schema(base_df):
    return SchemaDefinition.infer_from_df(base_df, name="transactions", version="1.0.0")


def test_no_drift_detected(schema, base_df):
    detector = SchemaDriftDetector(schema)
    events = detector.detect(base_df)
    errors = [e for e in events if e.severity == "error"]
    assert len(errors) == 0, f"Unexpected errors: {errors}"


def test_missing_column_detected(schema, base_df):
    df_missing = base_df.drop(columns=["amount"])
    detector = SchemaDriftDetector(schema)
    events = detector.detect(df_missing)
    error_types = [e.drift_type for e in events if e.severity == "error"]
    assert "missing_column" in error_types


def test_new_column_is_warning(schema, base_df):
    df_extra = base_df.copy()
    df_extra["new_col"] = 0
    detector = SchemaDriftDetector(schema)
    events = detector.detect(df_extra)
    warning_types = [e.drift_type for e in events if e.severity == "warning"]
    assert "unexpected_column" in warning_types


def test_assert_no_errors_raises(schema, base_df):
    df_missing = base_df.drop(columns=["amount"])
    detector = SchemaDriftDetector(schema)
    events = detector.detect(df_missing)
    with pytest.raises(RuntimeError, match="Schema drift errors"):
        detector.assert_no_errors(events)


def test_schema_roundtrip(schema, tmp_path):
    """Schema can be saved and loaded without data loss."""
    path = tmp_path / "schema.json"
    schema.save(path)
    loaded = SchemaDefinition.load(path)
    assert loaded.name == schema.name
    assert len(loaded.columns) == len(schema.columns)
