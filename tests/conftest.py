"""
Shared pytest fixtures for all test modules.

Fixtures mirror the actual loan_dataset_20000.csv schema so tests
exercise code paths that will run on real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    """
    Synthetic loan dataset fixture matching loan_dataset_20000.csv schema.

    Columns: loan_id (synthetic), all 21 original feature columns, loan_paid_back target.
    grade_subgrade is pre-parsed to grade_letter + grade_num (as preprocessing does).
    """
    np.random.seed(42)
    n = 500

    grades = ["A", "B", "C", "D", "E", "F", "G"]
    grade_subgrades = [f"{g}{s}" for g in grades for s in range(1, 6)]

    df = pd.DataFrame({
        "loan_id": [f"L{i:06d}" for i in range(n)],
        # Demographics
        "age": np.random.randint(22, 75, size=n),
        "gender": np.random.choice(["Male", "Female", "Other"], size=n),
        "marital_status": np.random.choice(["Single", "Married", "Divorced", "Widowed"], size=n),
        "education_level": np.random.choice(
            ["High School", "Bachelor's", "Master's", "PhD", "Other"], size=n
        ),
        # Income
        "annual_income": np.random.lognormal(mean=10.8, sigma=0.6, size=n),
        # employment_status — encoded as int (0-4) simulating ordinal encoding
        "employment_status": np.random.choice([0, 1, 2, 3, 4], size=n),
        # Credit metrics
        "debt_to_income_ratio": np.random.beta(2, 8, size=n),
        "credit_score": np.random.randint(440, 820, size=n),
        # Loan details
        "loan_amount": np.random.lognormal(mean=9.3, sigma=0.9, size=n).clip(500, 40_000),
        "loan_purpose": np.random.choice(
            ["Debt consolidation", "Car", "Home", "Medical", "Business", "Education", "Other"],
            size=n,
        ),
        "interest_rate": np.random.uniform(5, 25, size=n).round(2),
        "loan_term": np.random.choice([36, 60], size=n),
        "installment": np.random.uniform(50, 1000, size=n).round(2),
        # Credit bureau
        "num_of_open_accounts": np.random.randint(1, 15, size=n),
        "total_credit_limit": np.random.lognormal(mean=11, sigma=0.7, size=n),
        "current_balance": np.random.lognormal(mean=10, sigma=1, size=n),
        "delinquency_history": np.random.choice([0, 1, 2, 3, 4, 5, 6], size=n, p=[0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02]),
        "public_records": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        "num_of_delinquencies": np.random.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.5, 0.2, 0.15, 0.08, 0.05, 0.02]),
        # Grade (parsed from grade_subgrade in preprocessing)
        "grade_letter": np.random.choice(range(7), size=n),   # 0=A … 6=G
        "grade_num": np.random.randint(1, 6, size=n),
        # Target
        "loan_paid_back": np.random.binomial(1, 0.78, size=n),  # ~78% repay
    })
    return df


@pytest.fixture
def sample_test_df(sample_train_df) -> pd.DataFrame:
    """Holdout-style DataFrame — same schema minus target column."""
    df = sample_train_df.copy()
    df["loan_id"] = [f"T{i:06d}" for i in range(len(df))]
    return df.drop(columns=["loan_paid_back"])


@pytest.fixture
def raw_loan_df() -> pd.DataFrame:
    """
    Fixture mimicking the raw CSV before preprocessing.
    Includes grade_subgrade as original string and monthly_income.
    """
    np.random.seed(7)
    n = 200
    grades = ["A", "B", "C", "D", "E", "F", "G"]
    grade_subgrades = [f"{g}{s}" for g in grades for s in range(1, 6)]
    annual_incomes = np.random.lognormal(mean=10.8, sigma=0.6, size=n)
    return pd.DataFrame({
        "age": np.random.randint(22, 75, size=n),
        "gender": np.random.choice(["Male", "Female", "Other"], size=n),
        "marital_status": np.random.choice(["Single", "Married", "Divorced", "Widowed"], size=n),
        "education_level": np.random.choice(["High School", "Bachelor's", "Master's", "PhD", "Other"], size=n),
        "annual_income": annual_incomes,
        "monthly_income": annual_incomes / 12,       # redundant — dropped in preprocessing
        "employment_status": np.random.choice(["Employed", "Unemployed", "Self-employed", "Student", "Retired"], size=n),
        "debt_to_income_ratio": np.random.beta(2, 8, size=n),
        "credit_score": np.random.randint(440, 820, size=n),
        "loan_amount": np.random.lognormal(mean=9.3, sigma=0.9, size=n).clip(500, 40_000),
        "loan_purpose": np.random.choice(["Debt consolidation", "Car", "Home", "Medical", "Business"], size=n),
        "interest_rate": np.random.uniform(5, 25, size=n).round(2),
        "loan_term": np.random.choice([36, 60], size=n),
        "installment": np.random.uniform(50, 1000, size=n).round(2),
        "grade_subgrade": np.random.choice(grade_subgrades, size=n),
        "num_of_open_accounts": np.random.randint(1, 15, size=n),
        "total_credit_limit": np.random.lognormal(mean=11, sigma=0.7, size=n),
        "current_balance": np.random.lognormal(mean=10, sigma=1, size=n),
        "delinquency_history": np.random.randint(0, 7, size=n),
        "public_records": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        "num_of_delinquencies": np.random.randint(0, 6, size=n),
        "loan_paid_back": np.random.binomial(1, 0.78, size=n),
    })


@pytest.fixture
def binary_preds() -> tuple[np.ndarray, np.ndarray]:
    """(y_true, y_score) pair for metric testing."""
    np.random.seed(0)
    y_true = np.random.binomial(1, 0.22, size=5000)  # ~22% default rate
    y_score = np.clip(y_true * 0.6 + np.random.beta(2, 8, size=5000), 0, 1)
    return y_true, y_score
