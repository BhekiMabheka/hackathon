"""
Inference pipeline — load a fitted model and produce test set predictions.

Deliberately simple: the complexity lives in training.
Test-time inference must be fast, stateless, and identical to training transforms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.models.base import BaseModel
from src.utils.logging import get_logger

log = get_logger(__name__)


class InferencePipeline:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._model: BaseModel | None = None

    def load(self) -> "InferencePipeline":
        self._model = BaseModel.load(self.model_path)
        return self

    @property
    def model_feature_cols(self) -> list[str]:
        """Feature list the model was trained on — authoritative source of truth."""
        if self._model is None:
            raise RuntimeError("Call .load() first.")
        return self._model.feature_cols

    def predict(self, test_df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Call .load() first.")
        scores = self._model.predict_proba(test_df[feature_cols])
        log.info(
            "Inference complete",
            rows=len(test_df),
            score_mean=round(float(scores.mean()), 4),
            score_std=round(float(scores.std()), 4),
        )
        return pd.Series(scores, name="score")
