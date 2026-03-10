"""
Ensemble model — weighted average of base model OOF predictions.

Strategy:
  1. Train each base model on the same CV folds (OOF predictions).
  2. Optimise blend weights using OOF scores (scipy minimize or grid search).
  3. Generate final test predictions as weighted average of all base model
     test predictions.

This is typically the submission-winning approach in Zindi competitions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.base import BaseModel
from src.utils.logging import get_logger

log = get_logger(__name__)


class WeightedEnsemble:
    """
    Blends OOF predictions from multiple base models.

    Parameters
    ----------
    models:       List of fitted BaseModel instances.
    metric_fn:    Callable(y_true, y_pred) -> float (higher is better).
    """

    def __init__(
        self,
        models: list[BaseModel],
        metric_fn=None,
    ) -> None:
        self.models = models
        self.metric_fn = metric_fn
        self.weights_: Optional[np.ndarray] = None

    def fit_weights(
        self,
        oof_preds: list[np.ndarray],
        y_true: np.ndarray,
    ) -> "WeightedEnsemble":
        """
        Find optimal blend weights by minimising negative metric on OOF.

        Parameters
        ----------
        oof_preds: List of 1-D arrays, one per model, shape (n_train,).
        y_true:    Ground truth labels.
        """
        n_models = len(oof_preds)
        stack = np.column_stack(oof_preds)

        def neg_metric(weights: np.ndarray) -> float:
            w = np.abs(weights) / (np.abs(weights).sum() + 1e-9)
            blended = stack @ w
            return -self.metric_fn(y_true, blended)

        x0 = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0)] * n_models
        result = minimize(
            neg_metric,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        )
        raw = np.abs(result.x)
        self.weights_ = raw / raw.sum()

        log.info(
            "Ensemble weights optimised",
            weights={m.__class__.__name__: round(float(w), 4) for m, w in zip(self.models, self.weights_)},
            oof_score=round(-result.fun, 6),
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.weights_ is None:
            # Equal weights if not fitted
            self.weights_ = np.ones(len(self.models)) / len(self.models)

        preds = np.column_stack([m.predict_proba(X) for m in self.models])
        return preds @ self.weights_

    def equal_blend(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([m.predict_proba(X) for m in self.models])
        return preds.mean(axis=1)


def build_model(model_type: str, params: dict, random_state: int = 42) -> BaseModel:
    """Factory function — returns a model instance from config string."""
    if model_type == "lgbm":
        from src.models.lgbm_model import LGBMModel
        return LGBMModel(params=dict(params), random_state=random_state)
    elif model_type == "xgb":
        from src.models.xgb_model import XGBModel
        return XGBModel(params=dict(params), random_state=random_state)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'lgbm' or 'xgb'.")
