"""🔮 Predict & Explain — score a loan application with SHAP explainability."""
from __future__ import annotations

import sys
from pathlib import Path

DASH_ROOT = Path(__file__).resolve().parent.parent
ROOT = DASH_ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DASH_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.loader import load_model, load_test_features, load_train_features

st.set_page_config(page_title="Predict & Explain", page_icon="🔮", layout="wide")

st.title("🔮 Predict & Explain")
st.caption("Score a loan application and get a plain-English explanation of the decision.")

# ── Load ──────────────────────────────────────────────────────────────────────
model = load_model()
if model is None:
    st.error("Model not found at `outputs/models/model.pkl`. Run `make train` first.")
    st.stop()

feature_cols: list[str] = model.feature_cols

test_df  = load_test_features()
train_df = load_train_features()
ref_df   = test_df if not test_df.empty else train_df

if ref_df.empty:
    st.error("No processed features found. Run `make train` first.")
    st.stop()

available = [c for c in feature_cols if c in ref_df.columns]
if not available:
    st.error(
        "Feature mismatch: model features not found in processed data. "
        "Re-run `make train` to regenerate consistent artifacts."
    )
    st.stop()


# ── Plain-English translation layer ───────────────────────────────────────────
_GRADE_LETTERS = ["A", "B", "C", "D", "E", "F", "G"]
_INCOME_BANDS  = ["Very Low", "Low", "Middle", "Upper-Middle", "High"]
_CREDIT_BANDS  = ["Poor (< 580)", "Fair (580–670)", "Good (670–740)",
                   "Very Good (740–800)", "Excellent (800+)"]
_DTI_BANDS     = ["Very Low (< 10%)", "Low (10–20%)", "Moderate (20–35%)",
                   "High (35–50%)", "Very High (> 50%)"]


def _plain_reason(feat: str, value: float, shap_val: float) -> str:
    """Return a plain-English sentence for a feature + value combination."""
    v = value
    risk = shap_val > 0   # True = pushes toward default

    def _grade(code: float) -> str:
        idx = int(round(code))
        return _GRADE_LETTERS[idx] if 0 <= idx <= 6 else "?"

    def _band(val: float, labels: list[str]) -> str:
        idx = min(int(round(val)), len(labels) - 1)
        return labels[max(idx, 0)]

    mapping: dict[str, str] = {
        # ── Raw / preprocessed features ───────────────────────────────────────
        "credit_score": (
            f"Credit score of {v:.0f} — "
            + ("below the sub-prime threshold of 620, which is a significant concern"
               if v < 620 else
               "average range, some caution advised" if v < 720 else
               "good standing" if v < 780 else
               "excellent credit history")
        ),
        "debt_to_income_ratio": (
            f"Debt repayments are {v:.0%} of income — "
            + ("dangerously high, leaving little room for unexpected costs" if v > 0.5 else
               "high, indicating financial strain" if v > 0.35 else
               "moderate, manageable with discipline" if v > 0.2 else
               "low, strong financial buffer")
        ),
        "annual_income": (
            f"Annual income of R {v:,.0f} — "
            + ("below R 20,000 which limits repayment capacity" if v < 20_000 else
               "moderate income" if v < 60_000 else
               "good income level" if v < 120_000 else
               "high income, strong repayment capacity")
        ),
        "loan_amount": (
            f"Loan of R {v:,.0f} — "
            + ("very large loan" if v > 30_000 else
               "large loan" if v > 15_000 else
               "moderate loan" if v > 5_000 else
               "small loan")
        ),
        "interest_rate": (
            f"Interest rate of {v:.1f}% — "
            + ("very high, significantly increases total cost" if v > 20 else
               "high rate" if v > 15 else
               "moderate rate" if v > 10 else
               "low rate, affordable borrowing cost")
        ),
        "loan_term": (
            f"{'60-month (5-year)' if v >= 60 else '36-month (3-year)'} loan term — "
            + ("longer term means more total interest paid" if v >= 60 else
               "shorter term, lower total interest cost")
        ),
        "grade_letter": (
            f"Loan grade {_grade(v)} — "
            + ("highest-risk tier" if v >= 5 else
               "high-risk tier" if v >= 3 else
               "medium-risk tier" if v >= 1 else
               "low-risk tier")
        ),
        "grade_num": f"Sub-grade position {v:.0f} within letter grade",
        "num_of_open_accounts": (
            f"{v:.0f} open credit accounts — "
            + ("very few accounts, thin credit history" if v < 3 else
               "moderate number of accounts" if v < 8 else
               "many accounts, monitor for overextension")
        ),
        "delinquency_history": (
            "No past missed payments on record" if v == 0 else
            f"{v:.0f} past missed payment{'s' if v > 1 else ''} — "
            + ("serious concern" if v >= 3 else "worth investigating")
        ),
        "num_of_delinquencies": (
            "No delinquencies recorded" if v == 0 else
            f"{v:.0f} delinquency record{'s' if v > 1 else ''} — "
            + ("multiple delinquencies are a red flag" if v >= 3 else "concerning")
        ),
        "public_records": (
            "No public records (clean legal/credit history)" if v == 0 else
            f"{v:.0f} public record{'s' if v > 1 else ''} on file (e.g. court judgment, insolvency)"
        ),
        "current_balance": f"Current outstanding balance of R {v:,.0f}",
        "total_credit_limit": f"Total credit limit of R {v:,.0f}",
        "installment": (
            f"Monthly instalment of R {v:,.0f} — "
            + ("very high monthly commitment" if v > 800 else
               "moderate monthly commitment" if v > 400 else
               "low monthly commitment")
        ),
        # ── Engineered features ────────────────────────────────────────────────
        "feat_credit_util": (
            f"Using {v:.0%} of available credit — "
            + ("dangerously overextended" if v > 0.9 else
               "high usage, limited financial flexibility" if v > 0.7 else
               "moderate usage" if v > 0.4 else
               "healthy, well within limits")
        ),
        "feat_credit_util_capped": (
            f"Credit utilisation at {min(v, 1.0):.0%}"
        ),
        "feat_high_util_flag": (
            "Credit utilisation exceeds 80% — a sign of financial stress"
            if v >= 1 else
            "Credit utilisation is within healthy limits"
        ),
        "feat_loan_to_income": (
            f"Loan is {v:.1f}× annual income — "
            + ("very heavy debt load relative to earnings" if v > 0.5 else
               "significant debt relative to income" if v > 0.3 else
               "moderate" if v > 0.15 else
               "modest loan relative to income")
        ),
        "feat_installment_to_monthly": (
            f"Monthly instalment is {v:.0%} of monthly income — "
            + ("more than a third of take-home pay, very tight budget" if v > 0.33 else
               "significant portion of income" if v > 0.2 else
               "affordable" if v > 0.1 else
               "easily manageable")
        ),
        "feat_simple_payment_to_income": (
            f"Estimated repayment takes up {v:.0%} of monthly income"
        ),
        "feat_dti_band": f"Debt-to-income risk band: {_band(v, _DTI_BANDS)}",
        "feat_high_dti_flag": (
            "Debt-to-income ratio is above 35% — exceeds the standard risk threshold"
            if v >= 1 else
            "Debt-to-income ratio is within the acceptable range"
        ),
        "feat_rate_vs_grade_avg": (
            f"Interest rate is {abs(v):.1f}% "
            + ("higher than average for this grade — potential pricing mismatch"
               if v > 0 else
               "lower than average for this grade — relatively favourable rate")
        ),
        "feat_grade_position": f"Overall grade ranking: {v:.0f} out of 35 (lower = safer)",
        "feat_total_delinquency": (
            "No missed payments at all — positive repayment history" if v == 0 else
            f"{v:.0f} total missed payment{'s' if v > 1 else ''} across all accounts"
        ),
        "feat_any_delinquency": (
            "Has a history of missed payments" if v >= 1 else
            "No history of missed payments — reliable payer"
        ),
        "feat_severe_delinquency": (
            "Three or more missed payments — indicates serious repayment difficulties"
            if v >= 1 else
            "No severe delinquency pattern"
        ),
        "feat_delinq_per_account": (
            f"{v:.2f} missed payments per account on average"
        ),
        "feat_has_public_record": (
            "Has a public legal or credit record on file (e.g. judgment or insolvency)"
            if v >= 1 else
            "No public records — clean legal and credit standing"
        ),
        "feat_total_interest_burden": (
            f"Total interest over the loan life is {v:.0%} of principal — "
            + ("very high cost of borrowing" if v > 0.5 else
               "significant cost" if v > 0.3 else
               "moderate cost")
        ),
        "feat_long_high_rate": (
            "Long-term loan combined with a high interest rate — increases total repayment risk"
            if v >= 1 else
            "No long-term + high-rate combination"
        ),
        "feat_loan_per_term_month": (
            f"Loan principal of R {v:,.0f} per month of term"
        ),
        "feat_credit_band": f"Credit quality band: {_band(v, _CREDIT_BANDS)}",
        "feat_subprime": (
            "Sub-prime borrower — credit score below 620, considered high default risk"
            if v >= 1 else
            "Not classified as sub-prime"
        ),
        "feat_prime": (
            "Prime-rated borrower — credit score 740+, low risk profile"
            if v >= 1 else
            "Not prime-rated"
        ),
        "feat_income_band": f"Income band: {_band(v, _INCOME_BANDS)}",
        "feat_log_income":      f"Log income: {v:.2f}",
        "feat_log_balance":     f"Log outstanding balance: {v:.2f}",
        "feat_log_loan_amount": f"Log loan amount: {v:.2f}",
        "feat_dti_x_delinquency": (
            "High debt burden combined with missed payments — compounded risk"
            if v > 0 else
            "No high-DTI + delinquency combination"
        ),
        "feat_score_grade_mismatch": (
            f"Credit score and loan grade {'are inconsistent with each other' if v > 0.3 else 'are aligned'} "
            f"(mismatch index: {v:.2f})"
        ),
        "feat_score_x_grade": "Credit score × grade interaction score",
    }

    result = mapping.get(feat)
    if result:
        return result

    # Generic fallback for any unmapped feat_* column
    readable = feat.replace("feat_", "").replace("_", " ").title()
    return f"{readable}: {v:.3g}"


def _plain_summary(
    prob_default: float,
    tier: str,
    risk_reasons: list[str],
    protect_reasons: list[str],
) -> str:
    """One-paragraph plain-English verdict for a non-technical reader."""
    if prob_default < 0.10:
        opening = (
            "This applicant presents **strong creditworthiness signals**. "
            "The model considers them very unlikely to default."
        )
    elif prob_default < 0.25:
        opening = (
            "This applicant shows **mostly positive indicators** but has a few areas "
            "of concern that should be reviewed before final approval."
        )
    elif prob_default < 0.50:
        opening = (
            "This applicant carries **notable risk factors**. "
            "There is a meaningful chance of non-repayment and the application "
            "warrants careful review or additional conditions."
        )
    else:
        opening = (
            "This applicant has **multiple high-risk characteristics**. "
            "The model predicts a high likelihood of default. "
            "Lending should only proceed with strong mitigants in place."
        )

    risks    = "; ".join(risk_reasons[:2])    if risk_reasons    else "no major risk factors identified"
    protects = "; ".join(protect_reasons[:2]) if protect_reasons else "no strong positive factors identified"

    return (
        f"{opening}  \n\n"
        f"**Main concerns:** {risks}.  \n"
        f"**Working in their favour:** {protects}."
    )


# ── Mode selector ─────────────────────────────────────────────────────────────
mode = st.radio(
    "Input mode:",
    ["📋 Select a test record", "🎚️ What-if: tweak feature values"],
    horizontal=True,
)
st.divider()

# ── Build input row ───────────────────────────────────────────────────────────
if mode == "📋 Select a test record":
    n = len(ref_df)
    col_sl, col_info = st.columns([3, 1])
    with col_sl:
        idx = st.slider(
            "Record index (held-out test set)",
            0, n - 1, 0,
            help="Scroll through held-out test loans",
        )
    with col_info:
        actual = ref_df.iloc[idx].get("loan_paid_back", None)
        if actual is not None:
            label = "✅ Repaid" if actual == 1 else "❌ Default"
            st.metric("Actual Outcome", label)

    X = ref_df.iloc[[idx]][available]

    with st.expander("📋 Feature values for this record"):
        disp = X.T.reset_index()
        disp.columns = ["Feature", "Value"]
        disp["Value"] = disp["Value"].round(4)
        st.dataframe(disp, use_container_width=True, height=300)

else:
    st.subheader("Adjust Feature Values")
    st.caption(
        "Sliders show the top-10 most important model features. "
        "All others are fixed at the training-set median."
    )

    if model.feature_importance_ is not None:
        top_feats = [f for f in model.feature_importance_.head(10).index if f in available]
    else:
        top_feats = available[:10]

    medians = (
        train_df[available].median() if not train_df.empty
        else ref_df[available].median()
    )

    custom_vals: dict[str, float] = {}
    c1, c2 = st.columns(2)
    for i, feat in enumerate(top_feats):
        col = c1 if i % 2 == 0 else c2
        min_v     = float(ref_df[feat].min())
        max_v     = float(ref_df[feat].max())
        default_v = float(medians.get(feat, (min_v + max_v) / 2))
        step      = max((max_v - min_v) / 200, 1e-6)
        with col:
            custom_vals[feat] = st.slider(
                feat.replace("_", " ").title(),
                min_value=min_v, max_value=max_v,
                value=default_v, step=step, format="%.3f",
            )

    for feat in available:
        if feat not in custom_vals:
            custom_vals[feat] = float(medians.get(feat, 0.0))

    X = pd.DataFrame([custom_vals])[available]

# ── Score button ──────────────────────────────────────────────────────────────
st.divider()
if not st.button("🚀 Score this application", type="primary", use_container_width=True):
    st.info("👆 Click **Score this application** to get the prediction and explanation.")
    st.stop()

# ── Predict ───────────────────────────────────────────────────────────────────
prob_default = float(model.predict_proba(X)[0])
prob_repay   = 1.0 - prob_default

if prob_default < 0.10:
    tier, colour, tier_emoji = "LOW RISK",       "#27ae60", "🟢"
elif prob_default < 0.25:
    tier, colour, tier_emoji = "MEDIUM RISK",    "#f39c12", "🟡"
elif prob_default < 0.50:
    tier, colour, tier_emoji = "HIGH RISK",      "#e67e22", "🟠"
else:
    tier, colour, tier_emoji = "VERY HIGH RISK", "#e74c3c", "🔴"

# Metric cards
st.subheader("Prediction Result")
rc1, rc2, rc3 = st.columns(3)
rc1.metric("Default Probability",   f"{prob_default:.2%}")
rc2.metric("Repayment Probability", f"{prob_repay:.2%}")
rc3.metric("Risk Tier", f"{tier_emoji} {tier}")

# Gauge
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prob_default * 100,
    number={"suffix": "%", "font": {"size": 44}},
    title={"text": "Default Risk Score"},
    gauge={
        "axis": {"range": [0, 100], "tickformat": ".0f"},
        "bar":  {"color": colour, "thickness": 0.3},
        "steps": [
            {"range": [0,  10], "color": "#d5f5e3"},
            {"range": [10, 25], "color": "#fef9e7"},
            {"range": [25, 50], "color": "#fdebd0"},
            {"range": [50,100], "color": "#fadbd8"},
        ],
        "threshold": {
            "line": {"color": "black", "width": 4},
            "thickness": 0.75,
            "value": prob_default * 100,
        },
    },
))
fig.update_layout(height=300, margin=dict(t=30, b=10, l=40, r=40))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# PLAIN-LANGUAGE DECISION SUMMARY  (non-technical audience)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📋 Decision Summary")
st.caption("Plain-English explanation — no technical knowledge required.")

try:
    import shap

    booster   = model._booster
    explainer = shap.TreeExplainer(booster)
    shap_exp  = explainer(X)
    sv        = shap_exp.values[0]       # (n_features,)
    base_val  = float(shap_exp.base_values[0])

    sv_series = pd.Series(sv, index=available)

    # Top 5 risk-increasing factors (positive SHAP = more default risk)
    top_risk_feats = sv_series.nlargest(5)
    # Top 5 risk-reducing factors (negative SHAP = less default risk)
    top_protect_feats = sv_series.nsmallest(5)

    risk_reasons    = [_plain_reason(f, float(X[f].iloc[0]), v) for f, v in top_risk_feats.items()]
    protect_reasons = [_plain_reason(f, float(X[f].iloc[0]), v) for f, v in top_protect_feats.items()]

    # ── Verdict banner ────────────────────────────────────────────────────────
    verdict_md = _plain_summary(prob_default, tier, risk_reasons, protect_reasons)
    if prob_default < 0.10:
        st.success(verdict_md)
    elif prob_default < 0.25:
        st.info(verdict_md)
    elif prob_default < 0.50:
        st.warning(verdict_md)
    else:
        st.error(verdict_md)

    # ── Risk factors vs positive factors ──────────────────────────────────────
    col_risk, col_prot = st.columns(2)

    with col_risk:
        st.markdown("#### 🔴 Factors Raising Risk")
        st.caption("Each item below increased this applicant's default probability.")
        for reason in risk_reasons[:5]:
            st.markdown(f"- {reason}")

    with col_prot:
        st.markdown("#### 🟢 Factors Reducing Risk")
        st.caption("Each item below reduced this applicant's default probability.")
        for reason in protect_reasons[:5]:
            st.markdown(f"- {reason}")

    # ── Recommendation ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 💡 Suggested Action")
    if prob_default < 0.10:
        st.success(
            "**Approve** — Strong credit profile. Standard terms and conditions apply. "
            "Routine monitoring recommended."
        )
    elif prob_default < 0.25:
        st.info(
            "**Review** — Generally acceptable risk. Consider verifying income documents "
            "and confirming employment status before approval."
        )
    elif prob_default < 0.50:
        st.warning(
            "**Caution** — Elevated risk. Consider requesting additional collateral, "
            "reducing the loan amount, or applying a higher interest rate to compensate "
            "for the increased risk."
        )
    else:
        st.error(
            "**Decline or Escalate** — High probability of default. If the application "
            "must proceed, require strong collateral, a co-signatory, and senior credit "
            "officer approval."
        )

except ImportError:
    st.info("Install SHAP for plain-language explanations: `pip install shap`")
    sv, sv_series = None, None

except Exception as exc:
    st.error(f"Could not generate explanation: {exc}")
    sv, sv_series = None, None

# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL SHAP SECTION  (analysts & data scientists)
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander("📊 Technical Analysis — SHAP Details (for analysts)", expanded=False):
    st.caption(
        "SHAP (SHapley Additive exPlanations) is a method from game theory that assigns "
        "each feature a contribution score showing how much it shifted the predicted "
        "probability away from the model's baseline."
    )

    try:
        import shap  # already imported above; re-import is a no-op

        if sv is None:
            booster   = model._booster
            explainer = shap.TreeExplainer(booster)
            shap_exp  = explainer(X)
            sv        = shap_exp.values[0]
            base_val  = float(shap_exp.base_values[0])

        # Waterfall plot
        try:
            shap.plots.waterfall(shap_exp[0], max_display=15, show=False)
            wf_fig = plt.gcf()
            wf_fig.set_size_inches(10, 6)
            st.pyplot(wf_fig, clear_figure=True)
        except Exception:
            # Fallback: plotly bar
            shap_df = pd.DataFrame({"feature": available, "shap": sv})
            shap_df["abs"] = shap_df["shap"].abs()
            shap_df = shap_df.sort_values("abs", ascending=False).head(15).drop(columns="abs")
            shap_df["Effect"] = shap_df["shap"].apply(
                lambda v: "Increases Default Risk" if v > 0 else "Reduces Default Risk"
            )
            fig = px.bar(
                shap_df, x="shap", y="feature", orientation="h", color="Effect",
                color_discrete_map={
                    "Increases Default Risk": "#e74c3c",
                    "Reduces Default Risk":   "#27ae60",
                },
                title=f"SHAP Contributions  |  Baseline: {base_val:.3f}  →  Prediction: {prob_default:.3f}",
                labels={"shap": "SHAP Value", "feature": "Feature"},
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=480)
            st.plotly_chart(fig, use_container_width=True)

        # Global beeswarm
        st.markdown("---")
        st.markdown("**Global Feature Impact** — how each feature affects predictions across 200 training samples")
        if not train_df.empty:
            train_avail = [c for c in available if c in train_df.columns]
            sample     = train_df[train_avail].sample(200, random_state=42)
            global_exp = explainer(sample)
            try:
                shap.plots.beeswarm(global_exp, max_display=15, show=False)
                bee_fig = plt.gcf()
                bee_fig.set_size_inches(10, 7)
                st.pyplot(bee_fig, clear_figure=True)
            except Exception as e:
                st.info(f"Beeswarm unavailable: {e}")
        else:
            st.info("Training features not found.")

    except ImportError:
        st.info("Install SHAP: `pip install shap`")
    except Exception as exc:
        st.error(f"SHAP computation failed: {exc}")
        st.exception(exc)

# ── Full feature table ────────────────────────────────────────────────────────
with st.expander("📋 All feature values used for scoring"):
    disp = X.T.reset_index()
    disp.columns = ["Feature", "Value"]
    disp["Value"] = disp["Value"].round(4)
    st.dataframe(disp, use_container_width=True)
