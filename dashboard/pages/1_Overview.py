"""📊 Dataset Explorer — EDA of the raw loan portfolio."""
from __future__ import annotations

import sys
from pathlib import Path

DASH_ROOT = Path(__file__).resolve().parent.parent
ROOT = DASH_ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DASH_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.loader import load_raw_data

st.set_page_config(page_title="Dataset Explorer", page_icon="📊", layout="wide")

st.title("📊 Dataset Explorer")
st.caption("Raw loan portfolio — `loan_dataset_20000.csv`")

# ── Load ──────────────────────────────────────────────────────────────────────
df = load_raw_data()
if df.empty:
    st.error("Raw data not found at `data/raw/loan_dataset_20000.csv`. Check path.")
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Loans", f"{len(df):,}")
c2.metric("Features", str(df.shape[1]))
c3.metric("Repayment Rate", f"{df['loan_paid_back'].mean():.1%}")
c4.metric("Default Rate", f"{1 - df['loan_paid_back'].mean():.1%}")
c5.metric("Avg Loan Amount", f"R {df['loan_amount'].mean():,.0f}")

st.divider()

# ── Target distribution & grade default rate ──────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Repayment Distribution")
    counts = df["loan_paid_back"].value_counts().rename({1: "Paid Back", 0: "Default"})
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        color=counts.index,
        color_discrete_map={"Paid Back": "#27ae60", "Default": "#e74c3c"},
        hole=0.45,
    )
    fig.update_traces(textinfo="percent+label", pull=[0, 0.06])
    fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Default Rate by Grade  (A = safest → G = riskiest)")
    if "grade_subgrade" in df.columns:
        df["_grade"] = df["grade_subgrade"].str[0]
        grade_df = (
            df.groupby("_grade")["loan_paid_back"]
            .agg(default_rate=lambda x: 1 - x.mean(), count="count")
            .reset_index()
            .sort_values("_grade")
        )
        grade_df.columns = ["Grade", "Default Rate", "Count"]
        fig = px.bar(
            grade_df,
            x="Grade",
            y="Default Rate",
            color="Default Rate",
            text=grade_df["Default Rate"].map("{:.1%}".format),
            color_continuous_scale="RdYlGn_r",
            hover_data={"Count": True},
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=280, showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Loan purpose & loan amount ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Default Rate by Loan Purpose")
    purpose_df = (
        df.groupby("loan_purpose")["loan_paid_back"]
        .apply(lambda x: 1 - x.mean())
        .reset_index()
        .sort_values("loan_paid_back", ascending=True)
    )
    purpose_df.columns = ["Loan Purpose", "Default Rate"]
    fig = px.bar(
        purpose_df,
        x="Default Rate",
        y="Loan Purpose",
        orientation="h",
        text=purpose_df["Default Rate"].map("{:.1%}".format),
        color="Default Rate",
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(height=330, showlegend=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Loan Amount Distribution by Outcome")
    plot_df = df.copy()
    plot_df["Outcome"] = plot_df["loan_paid_back"].map({1: "Paid Back", 0: "Default"})
    fig = px.histogram(
        plot_df,
        x="loan_amount",
        color="Outcome",
        color_discrete_map={"Paid Back": "#27ae60", "Default": "#e74c3c"},
        barmode="overlay",
        opacity=0.7,
        nbins=50,
        labels={"loan_amount": "Loan Amount (R)"},
    )
    fig.update_layout(height=330, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Credit score & income ─────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Credit Score by Outcome")
    plot_df = df.copy()
    plot_df["Outcome"] = plot_df["loan_paid_back"].map({1: "Paid Back", 0: "Default"})
    fig = px.histogram(
        plot_df,
        x="credit_score",
        color="Outcome",
        color_discrete_map={"Paid Back": "#27ae60", "Default": "#e74c3c"},
        barmode="overlay",
        opacity=0.7,
        nbins=40,
        labels={"credit_score": "Credit Score"},
    )
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Annual Income by Outcome  (capped at 99th pct)")
    p99 = df["annual_income"].quantile(0.99)
    plot_df = df[df["annual_income"] <= p99].copy()
    plot_df["Outcome"] = plot_df["loan_paid_back"].map({1: "Paid Back", 0: "Default"})
    fig = px.histogram(
        plot_df,
        x="annual_income",
        color="Outcome",
        color_discrete_map={"Paid Back": "#27ae60", "Default": "#e74c3c"},
        barmode="overlay",
        opacity=0.7,
        nbins=50,
        labels={"annual_income": "Annual Income (R)"},
    )
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Debt-to-income & interest rate ────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Debt-to-Income Ratio by Outcome")
    plot_df = df.copy()
    plot_df["Outcome"] = plot_df["loan_paid_back"].map({1: "Paid Back", 0: "Default"})
    fig = px.box(
        plot_df,
        x="Outcome",
        y="debt_to_income_ratio",
        color="Outcome",
        color_discrete_map={"Paid Back": "#27ae60", "Default": "#e74c3c"},
        labels={"debt_to_income_ratio": "Debt-to-Income Ratio"},
    )
    fig.update_layout(height=300, showlegend=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Interest Rate by Outcome")
    plot_df = df.copy()
    plot_df["Outcome"] = plot_df["loan_paid_back"].map({1: "Paid Back", 0: "Default"})
    fig = px.box(
        plot_df,
        x="Outcome",
        y="interest_rate",
        color="Outcome",
        color_discrete_map={"Paid Back": "#27ae60", "Default": "#e74c3c"},
        labels={"interest_rate": "Interest Rate (%)"},
    )
    fig.update_layout(height=300, showlegend=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Feature correlation with target ──────────────────────────────────────────
st.subheader("Numeric Feature Correlation with Target  (`loan_paid_back`)")
num_df = df.select_dtypes(include="number")
corr = num_df.corr()["loan_paid_back"].drop("loan_paid_back").sort_values()
fig = px.bar(
    x=corr.values,
    y=corr.index,
    orientation="h",
    color=corr.values,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    labels={"x": "Pearson Correlation with loan_paid_back", "y": "Feature"},
)
fig.update_layout(height=420, showlegend=False, margin=dict(t=10, b=10, l=180))
st.plotly_chart(fig, use_container_width=True)

# ── Raw data sample ───────────────────────────────────────────────────────────
with st.expander("📋 Raw data sample (20 rows)"):
    st.dataframe(df.drop(columns=["_grade"], errors="ignore").sample(20, random_state=42).reset_index(drop=True), use_container_width=True)
