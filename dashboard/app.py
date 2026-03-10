"""
Nedbank Credit Risk — MLOps Demo Dashboard
==========================================
Landing page. Navigate via the sidebar to explore the dataset,
review model performance, or score a new loan application.

Run from hackathon/:
    streamlit run dashboard/app.py
    # or
    make dashboard
"""
from __future__ import annotations

import sys
from pathlib import Path

DASH_ROOT = Path(__file__).resolve().parent
ROOT = DASH_ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DASH_ROOT))

import streamlit as st

from utils.loader import load_model, load_oof_metrics

st.set_page_config(
    page_title="Nedbank Credit Risk | MLOps Demo",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Nedbank MLOps")
    st.caption("Data & Analytics Masters Challenge 2026")
    st.divider()
    st.markdown(
        "**Stack**\n"
        "- LightGBM · XGBoost\n"
        "- SHAP · Optuna\n"
        "- DVC · MLflow\n"
        "- Hydra · structlog\n"
        "- Streamlit\n\n"
        "**Data**\n"
        "- 20,000 anonymised Nedbank loans\n"
        "- 22 raw → 50+ engineered features"
    )

# ── Hero ──────────────────────────────────────────────────────────────────────
st.title("🏦 Nedbank Credit Risk — MLOps Demo")
st.markdown(
    "> **Nedbank Data & Analytics Masters Challenge 2026 — ML/DS Track**  \n"
    "> Binary classification: predict `loan_paid_back` (1 = repaid, 0 = default)  \n"
    "> Production-grade MLOps pipeline from raw CSV to scored submission."
)

st.divider()

# ── Live stats (load lazily, no crash if not trained yet) ─────────────────────
model = load_model()
oof_metrics = load_oof_metrics()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Portfolio Size", "20,000 loans")
c2.metric("Default Rate", "~22%")
c3.metric("Raw Features", "22")
c4.metric("Engineered Features", "50+")
if oof_metrics:
    auc = oof_metrics.get("oof_roc_auc")
    gini = oof_metrics.get("oof_gini")
    c5.metric("OOF ROC-AUC", f"{auc:.4f}" if auc else "—", delta=f"Gini {gini:.4f}" if gini else None)
else:
    c5.metric("Model Status", "✅ Ready" if model else "⚠ Run make train")

st.divider()

# ── Navigation cards ──────────────────────────────────────────────────────────
st.subheader("Pages")
a, b, c = st.columns(3)

with a:
    st.info(
        "### 📊 Dataset Explorer\n"
        "Understand the raw loan portfolio:\n"
        "- Repayment vs default distribution\n"
        "- Default rates by grade (A–G) & purpose\n"
        "- Income, loan amount & credit score distributions\n"
        "- Feature correlations with target"
    )

with b:
    st.success(
        "### 📈 Model Performance\n"
        "Full evaluation of the trained LightGBM model:\n"
        "- ROC-AUC · Gini coefficient · KS statistic\n"
        "- ROC, Precision-Recall & calibration curves\n"
        "- Score distribution by outcome\n"
        "- Feature importance (gain, top 25)"
    )

with c:
    st.warning(
        "### 🔮 Predict & Explain\n"
        "Score any loan application:\n"
        "- Pick a held-out test record  \n"
        "  *or* tweak features with sliders\n"
        "- Default probability + risk tier gauge\n"
        "- SHAP waterfall — *why* this score?"
    )

st.divider()

# ── Pipeline overview ─────────────────────────────────────────────────────────
st.subheader("Pipeline")
st.markdown(
    "```\n"
    "loan_dataset_20000.csv\n"
    "  → Ingestion       (synthetic loan_id injection)\n"
    "  → Preprocessing   (grade_subgrade parsing, ordinal encoding, imputation)\n"
    "  → Feature Eng.    (credit utilisation, DTI ratios, delinquency, log transforms …)\n"
    "  → Feature Select. (variance + correlation filters)\n"
    "  → 5-Fold Strat.CV (LightGBM, early stopping, OOF predictions)\n"
    "  → MLflow Tracking (params, metrics, model artifact)\n"
    "  → Submission CSV  (loan_id + default_probability)\n"
    "```"
)
