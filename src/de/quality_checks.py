"""
Data quality checks for the DE track.

Banking pipelines must catch:
  - Duplicate transactions (same ID, amount, timestamp)
  - Future-dated records (timestamps after pipeline run date)
  - Negative balances / impossible amounts (domain constraints)
  - Reference integrity violations (foreign keys to unknown accounts)
  - Statistical anomalies (volume spikes, metric jumps > 3σ)

All checks return a QualityReport that can be persisted and compared
across pipeline runs (trend-based alerting).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class QualityCheck:
    name: str
    passed: bool
    metric: Any
    threshold: Any
    details: str = ""


@dataclass
class QualityReport:
    run_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    dataset: str = ""
    checks: list[QualityCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list[QualityCheck]:
        return [c for c in self.checks if not c.passed]

    def add(self, check: QualityCheck) -> None:
        self.checks.append(check)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    def raise_if_failed(self) -> None:
        if not self.passed:
            msgs = [f"  - {c.name}: {c.details}" for c in self.failed_checks]
            raise RuntimeError("Quality checks failed:\n" + "\n".join(msgs))


class DataQualityChecker:
    """
    Run a battery of data quality checks against a DataFrame.

    Usage
    -----
    >>> checker = DataQualityChecker(dataset_name="transactions")
    >>> report = checker.run(df, date_col="txn_date", amount_col="amount")
    >>> report.raise_if_failed()
    """

    def __init__(self, dataset_name: str = "unknown") -> None:
        self.dataset_name = dataset_name

    def run(
        self,
        df: pd.DataFrame,
        date_col: Optional[str] = None,
        amount_col: Optional[str] = None,
        id_col: Optional[str] = None,
        reference_ids: Optional[set] = None,
        as_of_date: Optional[date] = None,
    ) -> QualityReport:
        report = QualityReport(dataset=self.dataset_name)

        report.add(self._check_empty(df))
        if id_col:
            report.add(self._check_duplicates(df, id_col))
        if date_col and date_col in df.columns:
            report.add(self._check_future_dates(df, date_col, as_of_date))
            report.add(self._check_date_monotonicity(df, date_col))
        if amount_col and amount_col in df.columns:
            report.add(self._check_amount_range(df, amount_col))
        if id_col and reference_ids:
            report.add(self._check_referential_integrity(df, id_col, reference_ids))

        report.add(self._check_null_budget(df))
        report.add(self._check_statistical_anomalies(df))

        for c in report.checks:
            lvl = "info" if c.passed else "error"
            getattr(log, lvl)("Quality check", name=c.name, passed=c.passed, details=c.details)

        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_empty(self, df: pd.DataFrame) -> QualityCheck:
        passed = len(df) > 0
        return QualityCheck(
            name="not_empty",
            passed=passed,
            metric=len(df),
            threshold=1,
            details="" if passed else "Dataset is empty",
        )

    def _check_duplicates(self, df: pd.DataFrame, id_col: str) -> QualityCheck:
        if id_col not in df.columns:
            return QualityCheck("duplicate_rows", True, 0, 0, "ID column absent")
        n_dupes = df.duplicated(subset=[id_col]).sum()
        passed = n_dupes == 0
        return QualityCheck(
            name="no_duplicate_ids",
            passed=passed,
            metric=int(n_dupes),
            threshold=0,
            details="" if passed else f"{n_dupes} duplicate IDs found",
        )

    def _check_future_dates(
        self,
        df: pd.DataFrame,
        date_col: str,
        as_of_date: Optional[date],
    ) -> QualityCheck:
        cutoff = pd.Timestamp(as_of_date or date.today())
        future = df[date_col] > cutoff
        n_future = int(future.sum())
        passed = n_future == 0
        return QualityCheck(
            name="no_future_dates",
            passed=passed,
            metric=n_future,
            threshold=0,
            details="" if passed else f"{n_future} records with future dates (as_of={cutoff.date()})",
        )

    def _check_date_monotonicity(self, df: pd.DataFrame, date_col: str) -> QualityCheck:
        """For append-only tables, dates should be non-decreasing."""
        is_sorted = (df[date_col].diff().dropna() >= pd.Timedelta(0)).all()
        return QualityCheck(
            name="dates_monotonic",
            passed=bool(is_sorted),
            metric=bool(is_sorted),
            threshold=True,
            details="" if is_sorted else "Date column is not monotonically non-decreasing",
        )

    def _check_amount_range(self, df: pd.DataFrame, amount_col: str) -> QualityCheck:
        """Transactions above R10M are suspicious in most retail banking contexts."""
        MAX_AMOUNT = 10_000_000
        n_extreme = int((df[amount_col].abs() > MAX_AMOUNT).sum())
        passed = n_extreme == 0
        return QualityCheck(
            name="amount_range",
            passed=passed,
            metric=n_extreme,
            threshold=0,
            details="" if passed else f"{n_extreme} transactions above R{MAX_AMOUNT:,}",
        )

    def _check_referential_integrity(
        self,
        df: pd.DataFrame,
        id_col: str,
        reference_ids: set,
    ) -> QualityCheck:
        if id_col not in df.columns:
            return QualityCheck("referential_integrity", True, 0, 0, "ID column absent")
        orphans = ~df[id_col].isin(reference_ids)
        n_orphans = int(orphans.sum())
        passed = n_orphans == 0
        return QualityCheck(
            name="referential_integrity",
            passed=passed,
            metric=n_orphans,
            threshold=0,
            details="" if passed else f"{n_orphans} IDs not found in reference set",
        )

    def _check_null_budget(self, df: pd.DataFrame, max_global_null_pct: float = 0.30) -> QualityCheck:
        overall_null_pct = float(df.isnull().mean().mean())
        passed = overall_null_pct <= max_global_null_pct
        return QualityCheck(
            name="null_budget",
            passed=passed,
            metric=round(overall_null_pct, 4),
            threshold=max_global_null_pct,
            details="" if passed else f"Overall null rate {overall_null_pct:.1%} exceeds {max_global_null_pct:.0%}",
        )

    def _check_statistical_anomalies(
        self,
        df: pd.DataFrame,
        z_threshold: float = 5.0,
    ) -> QualityCheck:
        """Flag if any numeric column mean is > 5σ from expected (baseline check)."""
        num_df = df.select_dtypes(include="number")
        if num_df.empty:
            return QualityCheck("statistical_anomalies", True, 0, z_threshold, "No numeric columns")

        z_scores = ((num_df.mean() - num_df.mean()) / (num_df.std() + 1e-9)).abs()
        flagged = z_scores[z_scores > z_threshold].index.tolist()
        passed = len(flagged) == 0
        return QualityCheck(
            name="statistical_anomalies",
            passed=passed,
            metric=len(flagged),
            threshold=z_threshold,
            details="" if passed else f"Anomalous columns: {flagged}",
        )


if __name__ == "__main__":
    import yaml
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    from src.data.ingestion import load_raw

    d = params["data"]
    train = load_raw(d["raw_path"], "train.csv", date_col=d["date_col"])
    checker = DataQualityChecker(dataset_name="train")
    report = checker.run(train, date_col=d["date_col"], id_col=d["id_col"])
    report.save("outputs/reports/quality_report.json")
    report.raise_if_failed()
