"""
Training pipeline — orchestrates CV, OOF prediction collection, and MLflow logging.

Key design decisions:
  1. Models are trained fold-by-fold; OOF preds accumulate across folds.
  2. Feature importance is averaged across folds (more stable than single-run).
  3. A final model is retrained on all training data after CV (for submission).
  4. Everything is logged to MLflow — params, metrics, model artifact.
  5. The whole pipeline is deterministic given the same seed (verified in tests).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd

from src.data.splits import get_cv_splits
from src.evaluation.metrics import evaluate_all
from src.evaluation.reports import generate_cv_report, save_oof_predictions
from src.models.base import BaseModel
from src.models.ensemble import build_model
from src.utils.logging import get_logger
from src.utils.reproducibility import seed_everything

log = get_logger(__name__)


class TrainingPipeline:
    """
    Cross-validated training pipeline with MLflow tracking.

    Parameters
    ----------
    cfg:          Full config dict (merged from Hydra).
    mlflow_uri:   MLflow tracking URI.
    experiment:   MLflow experiment name.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        mlflow_uri: str = "sqlite:///outputs/mlflow.db",
        experiment: str = "nedbank-masters-2026",
    ) -> None:
        self.cfg = cfg
        self.mlflow_uri = mlflow_uri
        self.experiment = experiment
        self._oof_scores: Optional[np.ndarray] = None
        self._oof_labels: Optional[np.ndarray] = None
        self._final_model: Optional[BaseModel] = None
        self._fold_importances: list[pd.Series] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        id_col: str,
        date_col: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Execute the full training pipeline.

        Returns
        -------
        dict with aggregated CV metrics (also saved to outputs/reports/).
        """
        seed = self.cfg.get("training", {}).get("random_state", 42)
        seed_everything(seed)

        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.experiment)

        with mlflow.start_run(run_name=self.cfg.get("project", {}).get("experiment", "run")):
            mlflow.log_params(self._flatten_cfg())

            cv_metrics, oof_df = self._cv_loop(
                train_df, feature_cols, target_col, id_col, date_col
            )

            # Retrain on full data
            log.info("Retraining on full training data for submission model")
            self._final_model = self._train_single(
                train_df[feature_cols],
                train_df[target_col],
            )

            # Average feature importance across folds
            avg_importance = pd.concat(self._fold_importances, axis=1).mean(axis=1)

            # Generate and save report
            report = generate_cv_report(
                fold_metrics=cv_metrics,
                oof_scores=self._oof_scores,
                oof_labels=self._oof_labels,
                feature_importance=avg_importance,
            )
            save_oof_predictions(oof_df)

            # Log to MLflow
            mlflow.log_metrics({k: v for k, v in report.items() if isinstance(v, float)})
            self._save_model()

        return report

    def get_final_model(self) -> BaseModel:
        if self._final_model is None:
            raise RuntimeError("Pipeline has not been run yet.")
        return self._final_model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cv_loop(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        id_col: str,
        date_col: Optional[str],
    ) -> tuple[list[dict], pd.DataFrame]:
        training_cfg = self.cfg.get("training", {})
        n_folds = training_cfg.get("n_folds", 5)
        cv_strategy = training_cfg.get("cv_strategy", "stratified_time")

        folds = get_cv_splits(
            df=df,
            target_col=target_col,
            strategy=cv_strategy,
            n_folds=n_folds,
            date_col=date_col,
        )

        oof_scores = np.zeros(len(df))
        oof_labels = df[target_col].to_numpy()
        fold_metrics = []
        oof_records = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
            log.info("Training fold", fold=fold_idx, n_folds=n_folds)
            X_tr = df.iloc[train_idx][feature_cols]
            y_tr = df.iloc[train_idx][target_col]
            X_val = df.iloc[val_idx][feature_cols]
            y_val = df.iloc[val_idx][target_col]

            model = self._train_single(X_tr, y_tr, X_val=X_val, y_val=y_val)
            fold_preds = model.predict_proba(X_val)
            oof_scores[val_idx] = fold_preds

            fold_m = evaluate_all(y_val.to_numpy(), fold_preds, prefix=f"fold{fold_idx}")
            fold_metrics.append({k.replace(f"fold{fold_idx}_", ""): v for k, v in fold_m.items()})

            if model.feature_importance_ is not None:
                self._fold_importances.append(model.feature_importance_)

            # Collect OOF records
            fold_df = df.iloc[val_idx][[id_col, target_col]].copy()
            fold_df["oof_score"] = fold_preds
            fold_df["fold"] = fold_idx
            oof_records.append(fold_df)

        self._oof_scores = oof_scores
        self._oof_labels = oof_labels
        return fold_metrics, pd.concat(oof_records, ignore_index=True)

    def _train_single(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> BaseModel:
        model_cfg = self.cfg.get("model", {})
        model_type = model_cfg.get("type", "lgbm")
        model_params = dict(model_cfg.get(model_type, {}))

        model = build_model(model_type, model_params)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        return model

    def _save_model(self) -> None:
        path = Path("outputs/models/model.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._final_model.save(path)
        mlflow.log_artifact(str(path))

    def _flatten_cfg(self) -> dict:
        """Flatten nested config for MLflow param logging."""
        flat: dict = {}
        for section, vals in self.cfg.items():
            if isinstance(vals, dict):
                for k, v in vals.items():
                    flat[f"{section}.{k}"] = str(v)
            else:
                flat[section] = str(vals)
        return flat
