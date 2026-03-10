"""
Inference entry point — generates a submission CSV from a trained model.

Usage
-----
    python scripts/predict.py
    make predict
"""

from __future__ import annotations

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data.preprocessing import Preprocessor
from src.features.engineering import FeatureEngineer
from src.pipelines.inference_pipeline import InferencePipeline
from src.pipelines.submission_pipeline import build_submission
from src.utils.logging import get_logger

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    data_cfg = cfg_dict["data"]

    # ── Load processed test features (built during training pipeline) ─
    try:
        test = pd.read_parquet("data/processed/test_features.parquet")
        log.info("Loaded processed test features", rows=len(test))
    except FileNotFoundError:
        log.warning("Processed test not found — running preprocessing from raw")
        from src.data.ingestion import load_raw
        test = load_raw(
            data_cfg["raw_path"],
            "test.csv",
            date_col=data_cfg["date_col"],
            id_col=data_cfg["id_col"],
        )
        pp = Preprocessor.from_meta(
            "data/interim/preprocessing_meta.json",
            target_col=data_cfg["target_col"],
            id_col=data_cfg["id_col"],
        )
        test = pp.transform(test)

        import json
        with open("data/processed/feature_meta.json") as f:
            meta = json.load(f)
        fe = FeatureEngineer(config=meta["config"])
        test = fe.transform(test)

    # ── Run inference ─────────────────────────────────────────────────
    inference = InferencePipeline(model_path="outputs/models/model.pkl").load()

    # Use the feature list from the model itself — it records exactly what
    # columns it was trained on, after feature selection.
    feature_cols = inference.model_feature_cols

    scores = inference.predict(test, feature_cols=feature_cols)

    # ── Build and validate submission ─────────────────────────────────
    sub_cfg = cfg_dict.get("submission", {})
    submission_path = build_submission(
        test_df=test,
        scores=scores,
        id_col=data_cfg["id_col"],
        output_dir=sub_cfg.get("output_dir", "outputs/submissions"),
    )
    log.info("Submission ready", path=str(submission_path))


if __name__ == "__main__":
    main()
