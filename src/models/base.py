"""
Abstract base model — all model implementations extend this.

Enforces a consistent interface so pipelines are model-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


class BaseModel(ABC):
    """Minimal interface every model must implement."""

    def __init__(self, params: dict[str, Any], random_state: int = 42) -> None:
        self.params = params
        self.random_state = random_state
        self.model: Any = None
        self.feature_cols: list[str] = []
        self.feature_importance_: Optional[pd.Series] = None

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "BaseModel":
        """Fit the model. Return self for chaining."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of positive class, shape (n_samples,)."""
        ...

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("Saved model", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        model = joblib.load(path)
        log.info("Loaded model", path=str(path), type=type(model).__name__)
        return model

    def _get_feature_importance(self) -> Optional[pd.Series]:
        """Override in subclass to expose importance scores."""
        return None
