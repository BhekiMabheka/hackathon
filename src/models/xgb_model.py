"""
XGBoost model wrapper — useful as a diversity component in ensembles.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.base import BaseModel
from src.utils.logging import get_logger

log = get_logger(__name__)


class XGBModel(BaseModel):
    def __init__(self, params: dict[str, Any], random_state: int = 42) -> None:
        super().__init__(params, random_state)
        self.params.setdefault("seed", random_state)
        self._model: Optional[xgb.XGBClassifier] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBModel":
        self.feature_cols = X_train.columns.tolist()

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val[self.feature_cols], y_val)]
            fit_kwargs["verbose"] = False

        self._model = xgb.XGBClassifier(**self.params)
        self._model.fit(X_train[self.feature_cols], y_train, **fit_kwargs)

        self.feature_importance_ = pd.Series(
            self._model.feature_importances_,
            index=self.feature_cols,
        ).sort_values(ascending=False)

        log.info("XGBoost training complete", n_features=len(self.feature_cols))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.predict_proba(X[self.feature_cols])[:, 1]
