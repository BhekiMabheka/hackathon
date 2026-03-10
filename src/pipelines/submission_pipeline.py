"""
Submission pipeline — builds and validates the Zindi CSV submission.

Zindi format: two columns — ID and score (probability of positive class).
Submission is validated before write to catch format errors early.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def build_submission(
    test_df: pd.DataFrame,
    scores: pd.Series,
    id_col: str,
    score_col: str = "score",
    output_dir: str | Path = "outputs/submissions",
    tag: str | None = None,
) -> Path:
    """
    Construct and save a Zindi-format submission CSV.

    Parameters
    ----------
    test_df:    Test DataFrame (must contain id_col).
    scores:     Model scores, same length as test_df.
    id_col:     ID column name.
    score_col:  Column name for scores in submission file.
    output_dir: Directory to write submission to.
    tag:        Optional tag appended to filename (e.g., model version).

    Returns
    -------
    Path to the written submission file.
    """
    if len(test_df) != len(scores):
        raise ValueError(f"Length mismatch: test_df {len(test_df)} vs scores {len(scores)}")

    submission = pd.DataFrame({
        id_col: test_df[id_col].values,
        score_col: scores.values,
    })

    _validate_submission(submission, id_col, score_col)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_str = f"_{tag}" if tag else ""
    filename = f"submission_{ts}{tag_str}.csv"
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission.to_csv(output_path, index=False)
    log.info(
        "Submission saved",
        path=str(output_path),
        rows=len(submission),
        score_range=[round(float(scores.min()), 4), round(float(scores.max()), 4)],
    )
    return output_path


def _validate_submission(
    submission: pd.DataFrame,
    id_col: str,
    score_col: str,
) -> None:
    """Raise on any submission format violation."""
    errors = []

    if id_col not in submission.columns:
        errors.append(f"Missing ID column: {id_col}")
    if score_col not in submission.columns:
        errors.append(f"Missing score column: {score_col}")

    if score_col in submission.columns:
        scores = submission[score_col]
        if scores.isnull().any():
            errors.append(f"Null values in score column ({scores.isnull().sum()} rows)")
        if not ((scores >= 0) & (scores <= 1)).all():
            errors.append("Scores outside [0, 1] — expected probability scores")
        if scores.std() < 1e-6:
            errors.append("Scores have near-zero variance — model may have collapsed")

    if id_col in submission.columns:
        n_dup = submission[id_col].duplicated().sum()
        if n_dup > 0:
            errors.append(f"{n_dup} duplicate IDs in submission")

    if errors:
        for e in errors:
            log.error("Submission validation error", message=e)
        raise ValueError(f"Submission invalid:\n" + "\n".join(errors))

    log.info("Submission validation passed")
