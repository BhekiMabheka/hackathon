"""
LightGBM model wrapper with early stopping, OOF predictions, and SHAP support.

LightGBM is typically the first-choice model for tabular banking data:
  - Handles high cardinality categoricals natively
  - Efficient on large datasets (histogram-based splits)
  - Built-in handling of missing values
  - Fast iteration speed for competition cycles
"""

from __future__ import annotations

from typing import Any, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.utils.logging import get_logger

log = get_logger(__name__)


class LGBMModel(BaseModel):
    """
    LightGBM binary classifier.

    Parameters
    ----------
    params:       LightGBM hyperparameters (from conf/model/lgbm.yaml).
    random_state: Seed passed to LightGBM.
    """

    def __init__(self, params: dict[str, Any], random_state: int = 42) -> None:
        super().__init__(params, random_state)
        self.params.setdefault("random_state", random_state)
        self.params.setdefault("verbose", -1)
        self._booster: Optional[lgb.Booster] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LGBMModel":
        self.feature_cols = X_train.columns.tolist()

        callbacks = [lgb.log_evaluation(period=100)]
        early_stopping_rounds = self.params.pop("early_stopping_rounds", 100)
        if X_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

        train_set = lgb.Dataset(X_train, label=y_train)
        val_sets = []
        if X_val is not None and y_val is not None:
            val_sets = [lgb.Dataset(X_val, label=y_val, reference=train_set)]

        # Separate params for lgb.train vs constructor
        n_estimators = self.params.pop("n_estimators", 2000)
        eval_metric = self.params.pop("eval_metric", "auc")

        self._booster = lgb.train(
            params={**self.params, "objective": "binary", "metric": eval_metric},
            train_set=train_set,
            num_boost_round=n_estimators,
            valid_sets=val_sets if val_sets else None,
            callbacks=callbacks,
        )

        # Restore params so object is re-usable
        self.params["n_estimators"] = n_estimators
        self.params["eval_metric"] = eval_metric
        self.params["early_stopping_rounds"] = early_stopping_rounds

        self.feature_importance_ = pd.Series(
            self._booster.feature_importance(importance_type="gain"),
            index=self.feature_cols,
        ).sort_values(ascending=False)

        log.info(
            "LightGBM training complete",
            best_iteration=self._booster.best_iteration,
            n_features=len(self.feature_cols),
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Model not fitted.")
        return self._booster.predict(X[self.feature_cols], num_iteration=self._booster.best_iteration)

    def _get_feature_importance(self) -> pd.Series:
        return self.feature_importance_

    def explain_shap(self, X: pd.DataFrame, n_samples: int = 500) -> np.ndarray:
        """
        Compute SHAP values for interpretability.
        Requires shap package. Returns array shape (n_samples, n_features).
        """
        try:
            import shap
        except ImportError:
            raise ImportError("Install shap: pip install shap")

        sample = X[self.feature_cols].sample(min(n_samples, len(X)), random_state=42)
        explainer = shap.TreeExplainer(self._booster)
        return explainer.shap_values(sample)
