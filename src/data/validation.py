"""
Data validation — Great Expectations-style checks without the overhead.

Two entry points:
  1. validate_schema()   — structural checks (columns, dtypes, nulls)
  2. validate_statistics() — distribution checks (target rate, range, uniqueness)

Raises DataValidationError on critical failures.
Logs warnings for non-critical anomalies (e.g., unexpectedly high null rate).

Run standalone: python -m src.data.validation --stage raw
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


class DataValidationError(RuntimeError):
    """Raised when a critical validation expectation fails."""


@dataclass
class ValidationResult:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


def validate_schema(
    df: pd.DataFrame,
    required_cols: list[str],
    max_null_ratio: float = 0.80,
) -> ValidationResult:
    """
    Check that required columns exist and null rates are within bounds.

    Parameters
    ----------
    df:             DataFrame to validate.
    required_cols:  Columns that must be present.
    max_null_ratio: Features above this threshold raise a warning.
    """
    result = ValidationResult()

    # Required columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        result.add_error(f"Missing required columns: {missing}")

    # High null columns
    null_ratios = df.isnull().mean()
    very_high_null = null_ratios[null_ratios > max_null_ratio].index.tolist()
    critical_null = null_ratios[null_ratios > 0.99].index.tolist()

    if critical_null:
        result.add_error(f"Columns with >99% nulls (likely empty): {critical_null}")
    if very_high_null:
        result.add_warning(f"Columns with >{max_null_ratio*100:.0f}% nulls: {very_high_null}")

    # Duplicate rows
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        result.add_warning(f"{n_dupes} duplicate rows detected")

    result.details["null_ratios"] = null_ratios.round(4).to_dict()
    result.details["shape"] = {"rows": len(df), "cols": len(df.columns)}

    _log_result(result, stage="schema")
    return result


def validate_statistics(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    expected_target_rate: tuple[float, float] = (0.01, 0.50),
) -> ValidationResult:
    """
    Check dataset statistics for anomalies.

    Parameters
    ----------
    df:                    DataFrame to validate.
    target_col:            Binary target column name.
    id_col:                Identifier column (check for uniqueness on test).
    expected_target_rate:  (min, max) acceptable positive class rate.
    """
    result = ValidationResult()

    # Target distribution
    if target_col in df.columns:
        target_rate = df[target_col].mean()
        lo, hi = expected_target_rate
        if not (lo <= target_rate <= hi):
            result.add_warning(
                f"Target rate {target_rate:.4f} outside expected range [{lo}, {hi}]. "
                "Check for class imbalance or data leakage."
            )
        result.details["target_rate"] = round(float(target_rate), 6)
        result.details["class_counts"] = df[target_col].value_counts().to_dict()

    # ID uniqueness
    if id_col in df.columns:
        n_unique_ids = df[id_col].nunique()
        n_rows = len(df)
        dup_ids = n_rows - n_unique_ids
        if dup_ids > 0:
            result.add_warning(
                f"ID column '{id_col}' has {dup_ids} duplicates — "
                "expected for transaction-level data; verify vs. expected grain."
            )
        result.details["unique_ids"] = n_unique_ids

    _log_result(result, stage="statistics")
    return result


def validate_no_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str,
    date_col: str,
) -> ValidationResult:
    """
    Critical leakage check: ensure no test IDs appear in train set
    and that all train dates precede all test dates.

    Banking datasets often have temporal structure — violating this
    inflates CV scores and causes silent production degradation.
    """
    result = ValidationResult()

    # ID overlap
    train_ids = set(train_df[id_col].astype(str))
    test_ids = set(test_df[id_col].astype(str))
    overlap = train_ids & test_ids
    if overlap:
        sample = list(overlap)[:5]
        result.add_warning(
            f"{len(overlap)} IDs appear in both train and test. "
            f"Sample: {sample}. "
            "If ID is customer-level and data is transactional, this may be expected. "
            "Verify grain carefully."
        )

    # Temporal ordering
    if date_col in train_df.columns and date_col in test_df.columns:
        train_max = train_df[date_col].max()
        test_min = test_df[date_col].min()
        if train_max >= test_min:
            result.add_error(
                f"Temporal leakage risk: train max date ({train_max}) >= "
                f"test min date ({test_min}). "
                "Ensure a clean temporal split with no overlap."
            )
        result.details["train_date_range"] = [str(train_df[date_col].min()), str(train_max)]
        result.details["test_date_range"] = [str(test_min), str(test_df[date_col].max())]

    _log_result(result, stage="leakage_check")
    return result


def _log_result(result: ValidationResult, stage: str) -> None:
    if result.passed and not result.warnings:
        log.info("Validation passed", stage=stage)
    elif result.warnings:
        for w in result.warnings:
            log.warning("Validation warning", stage=stage, message=w)
    for e in result.errors:
        log.error("Validation error", stage=stage, message=e)


def save_report(result: ValidationResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    log.info("Saved validation report", path=str(path))


# ---------------------------------------------------------------------------
# Standalone CLI entry point (called by DVC stage)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["raw", "processed"], default="raw")
    args = parser.parse_args()

    import yaml
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    from src.data.ingestion import load_train_test

    train, test = load_train_test(
        raw_dir=params["data"]["raw_path"],
        date_col=params["data"]["date_col"],
        id_col=params["data"]["id_col"],
    )

    required = [params["data"]["id_col"], params["data"]["target_col"]]
    result = validate_schema(train, required_cols=required)
    result2 = validate_statistics(
        train,
        target_col=params["data"]["target_col"],
        id_col=params["data"]["id_col"],
    )
    result3 = validate_no_leakage(
        train, test,
        id_col=params["data"]["id_col"],
        date_col=params["data"]["date_col"],
    )

    combined = {
        "schema": result.to_dict(),
        "statistics": result2.to_dict(),
        "leakage": result3.to_dict(),
    }
    save_report(
        ValidationResult(
            passed=all(r.passed for r in [result, result2, result3]),
            details=combined,
        ),
        path="outputs/reports/raw_validation_report.json",
    )

    critical = [r for r in [result, result2, result3] if not r.passed]
    if critical:
        raise DataValidationError("Critical validation failures — see report.")
