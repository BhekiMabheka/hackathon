"""
Backfill pipeline — reprocess historical data for a given date range.

Real-world trigger scenarios:
  - A bug in a feature transformation is discovered and fixed.
  - A source system retroactively corrected historical records.
  - A new feature needs to be computed over historical windows.

Design:
  - Backfill is idempotent: running it twice produces the same result.
  - Backfill is partition-aware: each date partition is processed independently,
    written to a staging area, then promoted atomically (rename).
  - The pipeline logs every partition processed with its row count and checksum.

Run: python -m src.de.backfill --start-date 2024-01-01 --end-date 2024-03-31
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def date_range(start: date, end: date) -> list[date]:
    """Generate inclusive list of dates from start to end."""
    days = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(days)]


def partition_path(base_dir: Path, dt: date, fmt: str = "%Y/%m/%d") -> Path:
    """Return Hive-style partition path: base_dir/2024/03/15/data.parquet"""
    return base_dir / dt.strftime(fmt) / "data.parquet"


def checksum(df: pd.DataFrame) -> str:
    """Stable checksum for a DataFrame (column names + shape + sample values)."""
    fingerprint = f"{list(df.columns)}|{df.shape}|{df.head(5).to_csv()}"
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:12]


class BackfillPipeline:
    """
    Partition-by-date backfill with idempotency and audit logging.

    Parameters
    ----------
    source_fn:    Callable(date) -> pd.DataFrame — loads raw data for a date.
    transform_fn: Callable(pd.DataFrame) -> pd.DataFrame — applies transforms.
    output_dir:   Base directory for partitioned output.
    date_col:     Column to filter on per partition.
    overwrite:    If False (default), skip partitions that already exist.
    """

    def __init__(
        self,
        source_fn: Callable[[date], pd.DataFrame],
        transform_fn: Callable[[pd.DataFrame], pd.DataFrame],
        output_dir: str | Path,
        date_col: str,
        overwrite: bool = False,
    ) -> None:
        self.source_fn = source_fn
        self.transform_fn = transform_fn
        self.output_dir = Path(output_dir)
        self.date_col = date_col
        self.overwrite = overwrite

    def run(self, start: date, end: date) -> dict[str, int]:
        """
        Run backfill for [start, end] inclusive.

        Returns
        -------
        Dict mapping date string -> rows processed (0 if skipped).
        """
        log.info("Starting backfill", start=str(start), end=str(end), overwrite=self.overwrite)
        results: dict[str, int] = {}

        for dt in date_range(start, end):
            results[str(dt)] = self._process_partition(dt)

        total_rows = sum(results.values())
        skipped = sum(1 for v in results.values() if v == 0)
        log.info(
            "Backfill complete",
            partitions=len(results),
            total_rows=total_rows,
            skipped=skipped,
        )
        return results

    def _process_partition(self, dt: date) -> int:
        out_path = partition_path(self.output_dir, dt)

        if out_path.exists() and not self.overwrite:
            log.info("Partition exists, skipping", date=str(dt), path=str(out_path))
            return 0

        try:
            raw = self.source_fn(dt)
        except Exception as exc:
            log.error("Failed to load partition", date=str(dt), error=str(exc))
            return 0

        if raw.empty:
            log.info("No data for partition", date=str(dt))
            return 0

        # Filter to date partition
        if self.date_col in raw.columns:
            raw = raw[pd.to_datetime(raw[self.date_col]).dt.date == dt]

        transformed = self.transform_fn(raw)
        cs = checksum(transformed)

        # Write atomically via staging
        staging_path = out_path.with_suffix(".tmp.parquet")
        staging_path.parent.mkdir(parents=True, exist_ok=True)
        transformed.to_parquet(staging_path, index=False)
        staging_path.rename(out_path)

        log.info(
            "Partition processed",
            date=str(dt),
            rows=len(transformed),
            checksum=cs,
            path=str(out_path),
        )
        return len(transformed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backfill pipeline")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    start_dt = date.fromisoformat(args.start_date)
    end_dt = date.fromisoformat(args.end_date)

    # Placeholder implementations — replace with real source/transform logic
    def _dummy_source(dt: date) -> pd.DataFrame:
        """Replace with actual data source query."""
        log.warning("Using dummy source — replace with real implementation", date=str(dt))
        return pd.DataFrame()

    def _dummy_transform(df: pd.DataFrame) -> pd.DataFrame:
        """Replace with actual transformation logic."""
        return df

    pipeline = BackfillPipeline(
        source_fn=_dummy_source,
        transform_fn=_dummy_transform,
        output_dir="data/processed/partitioned",
        date_col="transaction_date",
        overwrite=args.overwrite,
    )
    pipeline.run(start=start_dt, end=end_dt)
