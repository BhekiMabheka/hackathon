"""
Submission validator — run before every Zindi upload.

Checks:
  1. Required columns present and correctly named
  2. ID column uniqueness
  3. Score column is in [0, 1] (probability)
  4. No null scores
  5. Score variance (not a constant predictor)
  6. Row count matches expected test set size (if known)

Usage
-----
    python scripts/validate_submission.py path/to/submission.csv
    make submit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def validate(submission_path: str | Path, expected_rows: int | None = None) -> bool:
    path = Path(submission_path)
    if not path.exists():
        log.error("Submission file not found", path=str(path))
        return False

    df = pd.read_csv(path)
    errors = []

    # Column presence
    if len(df.columns) != 2:
        errors.append(f"Expected 2 columns (ID + score), got {len(df.columns)}: {df.columns.tolist()}")

    score_col = df.columns[-1]
    id_col = df.columns[0]

    # Nulls
    if df[score_col].isnull().any():
        n = df[score_col].isnull().sum()
        errors.append(f"Null scores: {n} rows")

    # Range
    if not ((df[score_col] >= 0) & (df[score_col] <= 1)).all():
        n_out = ((df[score_col] < 0) | (df[score_col] > 1)).sum()
        errors.append(f"{n_out} scores outside [0, 1]")

    # Variance
    if df[score_col].std() < 1e-6:
        errors.append("Score std ≈ 0 — model may have collapsed to a constant")

    # Duplicates
    n_dup = df[id_col].duplicated().sum()
    if n_dup > 0:
        errors.append(f"{n_dup} duplicate IDs")

    # Row count
    if expected_rows and len(df) != expected_rows:
        errors.append(f"Expected {expected_rows} rows, got {len(df)}")

    if errors:
        for e in errors:
            log.error("Validation failed", message=e)
        return False

    log.info(
        "Submission valid",
        path=str(path),
        rows=len(df),
        score_mean=round(float(df[score_col].mean()), 4),
        score_std=round(float(df[score_col].std()), 4),
        score_min=round(float(df[score_col].min()), 4),
        score_max=round(float(df[score_col].max()), 4),
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", nargs="?", help="Path to submission CSV")
    parser.add_argument("--expected-rows", type=int, default=None)
    args = parser.parse_args()

    # Auto-detect latest submission if not specified
    if args.submission is None:
        sub_dir = Path("outputs/submissions")
        csvs = sorted(sub_dir.glob("*.csv"))
        if not csvs:
            log.error("No submission files found in outputs/submissions/")
            sys.exit(1)
        submission_path = csvs[-1]
        log.info("Auto-selected latest submission", path=str(submission_path))
    else:
        submission_path = args.submission

    ok = validate(submission_path, expected_rows=args.expected_rows)
    sys.exit(0 if ok else 1)
