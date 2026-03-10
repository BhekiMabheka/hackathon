"""📈 Model Performance — ROC, KS, Gini, calibration, feature importance."""
from __future__ import annotations

import sys
from pathlib import Path

DASH_ROOT = Path(__file__).resolve().parent.parent
ROOT = DASH_ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DASH_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from utils.loader import (
    load_cv_metrics,
    load_feature_importance,
    load_model,
    load_oof_metrics,
    load_oof_predictions,
)

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

st.title("📈 Model Performance")
st.caption("LightGBM binary classifier — 5-fold stratified cross-validation on 20,000 loans")

# ── Load ──────────────────────────────────────────────────────────────────────
model = load_model()
oof_df = load_oof_predictions()
oof_metrics = load_oof_metrics()
cv_metrics = load_cv_metrics()
importance_df = load_feature_importance()

if model is None:
    st.warning("No trained model found. Run `make train` first, then refresh.")
    st.stop()

if not oof_metrics:
    st.warning("No OOF metrics found. Run `make train` first, then refresh.")
    st.stop()

# ── Headline metrics ──────────────────────────────────────────────────────────
st.subheader("Out-of-Fold (OOF) Metrics")

def _m(key: str, fmt: str = "{:.4f}") -> str:
    v = oof_metrics.get(key)
    return fmt.format(v) if v is not None else "—"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("ROC-AUC",         _m("oof_roc_auc"))
c2.metric("Gini Coefficient", _m("oof_gini"))
c3.metric("KS Statistic",    _m("oof_ks"))
c4.metric("PR-AUC",          _m("oof_pr_auc"))
c5.metric("Top-10% Capture", _m("oof_capture_top10pct", "{:.1%}"))

st.divider()

# ── Plots ─────────────────────────────────────────────────────────────────────
if (
    oof_df is not None
    and "oof_score" in oof_df.columns
    and "loan_paid_back" in oof_df.columns
):
    y_true  = oof_df["loan_paid_back"].to_numpy()
    y_score = oof_df["oof_score"].to_numpy()

    # ROC + KS
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = oof_metrics.get("oof_roc_auc", 0.0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"LightGBM  (AUC = {auc_val:.4f})",
            fill="tozeroy", line=dict(color="#2196F3", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Random",
            mode="lines", line=dict(dash="dash", color="#aaa", width=1),
        ))
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=380, legend=dict(x=0.55, y=0.05),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("KS Separation Plot")
        thresholds = np.linspace(0, 1, 300)
        n_pos = (y_true == 1).sum()
        n_neg = (y_true == 0).sum()
        good_cdf = [((y_score <= t) & (y_true == 1)).sum() / n_pos for t in thresholds]
        bad_cdf  = [((y_score <= t) & (y_true == 0)).sum() / n_neg for t in thresholds]
        ks_val = oof_metrics.get("oof_ks", 0.0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=good_cdf, name="Good — Repaid",  line=dict(color="#27ae60", width=2)))
        fig.add_trace(go.Scatter(x=thresholds, y=bad_cdf,  name="Bad — Default", line=dict(color="#e74c3c", width=2)))
        fig.update_layout(
            title=f"KS = {ks_val:.4f}  |  max separation between cumulative good & bad",
            xaxis_title="Score Threshold",
            yaxis_title="Cumulative %",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Precision-Recall + Calibration
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc_val = oof_metrics.get("oof_pr_auc", 0.0)
        baseline = float(y_true.mean())
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            name=f"LightGBM  (PR-AUC = {pr_auc_val:.4f})",
            fill="tozeroy", line=dict(color="#9b59b6", width=2),
        ))
        fig.add_shape(
            type="line", x0=0, y0=baseline, x1=1, y1=baseline,
            line=dict(dash="dash", color="#aaa", width=1),
        )
        fig.add_annotation(
            x=0.5, y=baseline + 0.015,
            text=f"Baseline — {baseline:.1%} positive rate",
            showarrow=False, font=dict(color="#888"),
        )
        fig.update_layout(
            xaxis_title="Recall", yaxis_title="Precision",
            height=380, legend=dict(x=0.3, y=0.9),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Calibration Curve")
        frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10)
        ece = oof_metrics.get("oof_ece", None)
        title = f"ECE = {ece:.4f}" if ece else "Reliability diagram"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mean_pred, y=frac_pos, mode="lines+markers",
            name="Model", line=dict(color="#e67e22", width=2), marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Perfect calibration",
            mode="lines", line=dict(dash="dash", color="#aaa", width=1),
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            height=380, legend=dict(x=0.05, y=0.9),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Score distribution
    st.subheader("Score Distribution by Outcome")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=y_score[y_true == 1], name="Repaid (1)",
        opacity=0.7, marker_color="#27ae60", nbinsx=60,
    ))
    fig.add_trace(go.Histogram(
        x=y_score[y_true == 0], name="Default (0)",
        opacity=0.7, marker_color="#e74c3c", nbinsx=60,
    ))
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Predicted Default Probability",
        yaxis_title="Count",
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("OOF predictions parquet not found — run `make train` to generate them.")

st.divider()

# ── Feature importance ────────────────────────────────────────────────────────
st.subheader("Feature Importance (Gain) — Top 25")

imp_df = importance_df
if imp_df is None and model is not None and model.feature_importance_ is not None:
    imp_df = model.feature_importance_.reset_index()
    imp_df.columns = ["feature", "importance"]

if imp_df is not None and not imp_df.empty:
    top = imp_df.head(25).copy()
    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Blues",
        text=top["importance"].map("{:.0f}".format),
        labels={"importance": "Gain Importance", "feature": "Feature"},
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=580, showlegend=False, margin=dict(t=10, b=10, l=200))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Feature importance not available — run `make train` to generate it.")

# ── CV fold breakdown ─────────────────────────────────────────────────────────
if cv_metrics:
    with st.expander("📋 Cross-Validation Fold Summary  (mean ± std across 5 folds)"):
        rows = []
        for key in ["roc_auc", "gini", "ks", "f1_at_05", "capture_top10pct", "pr_auc"]:
            mean_v = cv_metrics.get(f"{key}_mean")
            if mean_v is None:
                continue
            rows.append({
                "Metric": key,
                "Mean":   f"{cv_metrics.get(f'{key}_mean', 0):.4f}",
                "Std":    f"{cv_metrics.get(f'{key}_std',  0):.4f}",
                "Min":    f"{cv_metrics.get(f'{key}_min',  0):.4f}",
                "Max":    f"{cv_metrics.get(f'{key}_max',  0):.4f}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
