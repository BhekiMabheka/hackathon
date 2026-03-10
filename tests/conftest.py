"""Shared pytest fixtures for all test modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    """Minimal training DataFrame with realistic banking columns."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(n)],
        "transaction_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "amount": np.random.lognormal(mean=7, sigma=1.5, size=n),
        "balance": np.random.normal(loc=50_000, scale=30_000, size=n),
        "n_transactions_30d": np.random.poisson(lam=15, size=n),
        "account_age_days": np.random.randint(30, 3650, size=n),
        "product_code": np.random.choice(["CHQ", "SAV", "CRD", "INV"], size=n),
        "risk_band": np.random.choice(["A", "B", "C", "D"], size=n, p=[0.5, 0.3, 0.15, 0.05]),
        "target": np.random.binomial(1, 0.08, size=n),  # ~8% positive rate
    })


@pytest.fixture
def sample_test_df(sample_train_df) -> pd.DataFrame:
    """Test DataFrame with same schema but no target column."""
    df = sample_train_df.copy()
    df["customer_id"] = [f"T{i:06d}" for i in range(len(df))]
    df["transaction_date"] = df["transaction_date"] + pd.Timedelta(days=365)
    return df.drop(columns=["target"])


@pytest.fixture
def binary_preds() -> tuple[np.ndarray, np.ndarray]:
    """(y_true, y_score) pair for metric testing."""
    np.random.seed(0)
    y_true = np.random.binomial(1, 0.1, size=5000)
    y_score = np.clip(y_true * 0.7 + np.random.beta(2, 10, size=5000), 0, 1)
    return y_true, y_score
