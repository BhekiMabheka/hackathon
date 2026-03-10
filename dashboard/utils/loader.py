"""Cached data and model loading utilities for the Streamlit dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# hackathon/ root is two levels up from dashboard/utils/
ROOT = Path(__file__).resolve().parent.parent.parent


@st.cache_resource
def load_model():
    """Load trained model. Cached as a resource — survives re-runs."""
    path = ROOT / "outputs" / "models" / "model.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    path = ROOT / "data" / "raw" / "loan_dataset_20000.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_train_features() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "train_features.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_test_features() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "test_features.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_oof_predictions() -> pd.DataFrame | None:
    path = ROOT / "outputs" / "models" / "oof_predictions.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data
def load_oof_metrics() -> dict:
    path = ROOT / "outputs" / "reports" / "oof_metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_cv_metrics() -> dict:
    path = ROOT / "outputs" / "reports" / "cv_metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_feature_importance() -> pd.DataFrame | None:
    path = ROOT / "outputs" / "reports" / "feature_importance.parquet"
    return pd.read_parquet(path) if path.exists() else None
