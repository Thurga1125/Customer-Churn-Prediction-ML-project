import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: #f0f4f8; }
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Tabs ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #fff;
    border-bottom: 1px solid #e8ecf0;
    gap: 0;
    padding: 0 60px;
    position: sticky;
    top: 0;
    z-index: 200;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.stTabs [data-baseweb="tab"] {
    font-size: 13.5px;
    font-weight: 500;
    color: #64748b;
    padding: 16px 24px;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    letter-spacing: .01em;
}
.stTabs [aria-selected="true"] {
    color: #2563eb !important;
    border-bottom-color: #2563eb !important;
    font-weight: 600;
}
.stTabs [data-baseweb="tab-panel"] { padding: 0; }

/* ── Hero ──────────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(160deg, #f8fafc 0%, #eef2ff 100%);
    text-align: center;
    padding: 80px 60px 64px;
    border-bottom: 1px solid #e8ecf0;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #fff;
    border: 1px solid #dde3ee;
    border-radius: 999px;
    padding: 7px 18px;
    font-size: 12.5px;
    font-weight: 600;
    color: #3b82f6;
    letter-spacing: .03em;
    margin-bottom: 28px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.hero-title {
    font-size: 58px;
    font-weight: 900;
    line-height: 1.08;
    color: #0f172a;
    margin-bottom: 4px;
    letter-spacing: -.02em;
}
.hero-accent { color: #2563eb; }
.hero-sub {
    font-size: 17px;
    color: #64748b;
    max-width: 560px;
    margin: 16px auto 40px;
    line-height: 1.65;
    font-weight: 400;
}
.hero-stats {
    display: inline-flex;
    align-items: center;
    background: #fff;
    border: 1px solid #dde3ee;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.hero-stat-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 16px 28px;
    border-right: 1px solid #e8ecf0;
}
.hero-stat-item:last-child { border-right: none; }
.hero-stat-icon {
    width: 34px; height: 34px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.icon-blue   { background: #eff6ff; }
.icon-green  { background: #f0fdf4; }
.icon-purple { background: #faf5ff; }
.icon-amber  { background: #fffbeb; }
.hero-stat-val { font-size: 16px; font-weight: 700; color: #0f172a; line-height: 1; }
.hero-stat-lbl { font-size: 11.5px; color: #94a3b8; font-weight: 500; margin-top: 2px; }

/* ── Sections ──────────────────────────────────────────────────────────── */
.section     { padding: 60px; }
.section-alt { background: #fff; }
.sec-title   { font-size: 30px; font-weight: 800; color: #0f172a; text-align: center; letter-spacing: -.02em; margin-bottom: 6px; }
.sec-sub     { font-size: 14.5px; color: #64748b; text-align: center; margin-bottom: 40px; }
.inner       { padding: 0 60px; }

/* ── Metric cards ──────────────────────────────────────────────────────── */
.metric-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; }
.metric-card {
    background: #fff; border-radius: 16px; padding: 28px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06); text-align: center; border: 1px solid #f1f5f9;
}
.metric-label { font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: .07em; margin-bottom: 10px; }
.metric-value { font-size: 38px; font-weight: 800; color: #0f172a; line-height: 1; letter-spacing: -.02em; }
.metric-note  { font-size: 12px; color: #94a3b8; margin-top: 7px; }

/* ── Tags ──────────────────────────────────────────────────────────────── */
.tag-cloud { display: flex; flex-wrap: wrap; justify-content: center; gap: 9px; margin-top: 28px; }
.tag {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 999px;
    padding: 5px 15px; font-size: 12.5px; font-weight: 500; color: #475569;
}

/* ── Model cards ───────────────────────────────────────────────────────── */
.model-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.model-card {
    background: #fff; border-radius: 18px; padding: 36px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06); border: 1.5px solid #f1f5f9; position: relative;
}
.model-card.winner {
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37,99,235,.08), 0 1px 4px rgba(0,0,0,.06);
}
.winner-badge {
    position: absolute; top: -13px; left: 28px;
    background: #2563eb; color: #fff;
    font-size: 11px; font-weight: 700; padding: 4px 14px;
    border-radius: 999px; letter-spacing: .05em; text-transform: uppercase;
}
.model-name   { font-size: 20px; font-weight: 700; color: #0f172a; margin-bottom: 24px; }
.model-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
.mm-label { font-size: 10.5px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: .07em; margin-bottom: 4px; }
.mm-value { font-size: 30px; font-weight: 800; color: #0f172a; letter-spacing: -.01em; }
.mm-value.blue { color: #2563eb; }
.model-divider { height: 1px; background: #f1f5f9; margin-bottom: 18px; }
.model-points  { list-style: none; }
.model-points li { display: flex; align-items: center; gap: 9px; font-size: 13.5px; color: #64748b; margin-bottom: 9px; }
.dot-blue { width: 6px; height: 6px; border-radius: 50%; background: #2563eb; flex-shrink: 0; }
.dot-gray { width: 6px; height: 6px; border-radius: 50%; background: #cbd5e1; flex-shrink: 0; }

/* ── Driver cards ──────────────────────────────────────────────────────── */
.driver-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 18px; }
.driver-card {
    background: #fff; border-radius: 16px; padding: 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06); border: 1px solid #f1f5f9;
}
.driver-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px; }
.driver-icon-wrap { width: 42px; height: 42px; border-radius: 12px; background: #eff6ff; display: flex; align-items: center; justify-content: center; }
.driver-pct   { font-size: 26px; font-weight: 800; color: #2563eb; letter-spacing: -.01em; }
.driver-title { font-size: 15px; font-weight: 700; color: #0f172a; margin-bottom: 8px; }
.driver-desc  { font-size: 13.5px; color: #64748b; line-height: 1.6; }

/* ── Insights ──────────────────────────────────────────────────────────── */
.insight-card { border-radius: 14px; padding: 22px 24px; }
.insight-blue  { background: #eff6ff; border: 1px solid #bfdbfe; }
.insight-amber { background: #fffbeb; border: 1px solid #fde68a; }
.insight-red   { background: #fef2f2; border: 1px solid #fecaca; }
.insight-head  { font-size: 13.5px; font-weight: 700; margin-bottom: 7px; }
.insight-blue  .insight-head { color: #1d4ed8; }
.insight-amber .insight-head { color: #92400e; }
.insight-red   .insight-head { color: #991b1b; }
.insight-body  { font-size: 13px; line-height: 1.6; }
.insight-blue  .insight-body { color: #3b82f6; }
.insight-amber .insight-body { color: #b45309; }
.insight-red   .insight-body { color: #b91c1c; }

/* ── Metrics table ─────────────────────────────────────────────────────── */
.mtable {
    width: 100%; border-collapse: collapse;
    background: #fff; border-radius: 16px; overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,.06); border: 1px solid #f1f5f9;
}
.mtable th {
    background: #f8fafc; font-size: 11px; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: .07em; padding: 14px 22px;
    text-align: left; border-bottom: 1px solid #e8ecf0;
}
.mtable td { padding: 15px 22px; font-size: 14.5px; color: #334155; border-bottom: 1px solid #f8fafc; font-weight: 500; }
.mtable tr:last-child td { border-bottom: none; }
.mtable .best { color: #2563eb; font-weight: 700; }
.mtable .dim  { color: #94a3b8; }

/* ── Image cards ───────────────────────────────────────────────────────── */
.img-card { background: #fff; border-radius: 16px; padding: 22px; box-shadow: 0 1px 3px rgba(0,0,0,.06); border: 1px solid #f1f5f9; }
.img-title { font-size: 13px; font-weight: 700; color: #0f172a; margin-bottom: 14px; letter-spacing: .01em; }

/* ── Why cards ─────────────────────────────────────────────────────────── */
.why-card {
    background: #fff; border-radius: 16px; padding: 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06); border: 1px solid #f1f5f9; margin-bottom: 16px;
}
.why-title { font-size: 14.5px; font-weight: 700; color: #0f172a; margin-bottom: 7px; }
.why-body  { font-size: 13.5px; color: #64748b; line-height: 1.6; }

/* ── Prediction results ────────────────────────────────────────────────── */
.result-wrap  { border-radius: 16px; padding: 28px 30px; }
.result-churn { background: #fef2f2; border: 1px solid #fca5a5; }
.result-stay  { background: #f0fdf4; border: 1px solid #86efac; }
.result-label { font-size: 20px; font-weight: 800; margin-bottom: 6px; }
.result-label.churn { color: #dc2626; }
.result-label.stay  { color: #16a34a; }
.result-prob  { font-size: 13px; color: #64748b; }
.result-prob b { color: #0f172a; }
.model-tag {
    font-size: 11px; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: .07em; margin-bottom: 10px;
}

/* ── Risk bar ──────────────────────────────────────────────────────────── */
.risk-card {
    background: #fff; border-radius: 16px; padding: 28px 32px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06); border: 1px solid #f1f5f9; margin-top: 20px;
}
.risk-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 16px; }
.risk-head-title { font-size: 14px; font-weight: 700; color: #0f172a; }
.risk-score  { font-size: 28px; font-weight: 800; letter-spacing: -.01em; }
.risk-track  { height: 10px; background: #f1f5f9; border-radius: 999px; overflow: hidden; margin-bottom: 8px; }
.risk-fill   { height: 100%; border-radius: 999px; }
.risk-labels { display: flex; justify-content: space-between; font-size: 11.5px; color: #94a3b8; font-weight: 500; }
.risk-verdict { margin-top: 14px; font-size: 13.5px; font-weight: 600; }

/* ── Form labels ───────────────────────────────────────────────────────── */
div[data-testid="stSelectbox"] > label,
div[data-testid="stSlider"] > label,
div[data-testid="stNumberInput"] > label {
    font-size: 12px !important;
    font-weight: 700 !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: .05em !important;
}
div[data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
    font-size: 14px !important;
    background: #fff !important;
}
div[data-testid="stNumberInput"] > div {
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
    background: #fff !important;
}
.form-sep {
    font-size: 11px; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: .1em;
    padding: 24px 0 10px; border-bottom: 1px solid #f1f5f9; margin-bottom: 8px;
}

/* ── Submit button ─────────────────────────────────────────────────────── */
div[data-testid="stFormSubmitButton"] button {
    background: #2563eb !important; color: #fff !important;
    border-radius: 12px !important; font-size: 14.5px !important;
    font-weight: 700 !important; padding: 14px !important; border: none !important;
    width: 100% !important; letter-spacing: .02em !important;
    box-shadow: 0 4px 14px rgba(37,99,235,.3) !important;
}
div[data-testid="stFormSubmitButton"] button:hover { background: #1d4ed8 !important; }

.gap16 { height: 16px; }
.gap24 { height: 24px; }
.gap48 { height: 48px; }
</style>
""", unsafe_allow_html=True)

# ── Models ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    lr     = joblib.load("models/logistic_regression_churn_model.pkl")
    rf     = joblib.load("models/random_forest_churn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return lr, rf, scaler

lr_model, rf_model, scaler = load_models()

FEATURE_COLS = [
    'SeniorCitizen','tenure','MonthlyCharges','TotalCharges',
    'gender_Male','Partner_Yes','Dependents_Yes','PhoneService_Yes',
    'MultipleLines_No phone service','MultipleLines_Yes',
    'InternetService_Fiber optic','InternetService_No',
    'OnlineSecurity_No internet service','OnlineSecurity_Yes',
    'OnlineBackup_No internet service','OnlineBackup_Yes',
    'DeviceProtection_No internet service','DeviceProtection_Yes',
    'TechSupport_No internet service','TechSupport_Yes',
    'StreamingTV_No internet service','StreamingTV_Yes',
    'StreamingMovies_No internet service','StreamingMovies_Yes',
    'Contract_One year','Contract_Two year','PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check','PaymentMethod_Mailed check',
]
RESULTS = Path("results")

# ── SVG icons ─────────────────────────────────────────────────────────────────
def svg(path_d, color="#2563eb", w=18, h=18, extra=""):
    return f'<svg width="{w}" height="{h}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" {extra}>{path_d}</svg>'

ICN_USERS  = svg('<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>', "#2563eb")
ICN_DB     = svg('<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>', "#16a34a")
ICN_ACT    = svg('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>', "#7c3aed")
ICN_TGT    = svg('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>', "#d97706")
ICN_BADGE  = svg('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>', "#3b82f6", 13, 13)
ICN_DOC    = svg('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>', "#2563eb", 20, 20)
ICN_CLK    = svg('<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>', "#2563eb", 20, 20)
ICN_WIFI   = svg('<path d="M1.42 9a16 16 0 0 1 21.16 0"/><path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/>', "#2563eb", 20, 20)


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div class="hero-badge">{ICN_BADGE}&nbsp;Supervised ML · Classification</div>
  <div class="hero-title">Customer Churn<br><span class="hero-accent">Prediction</span></div>
  <div class="hero-sub">
    Predicting telecom customer attrition using Logistic Regression and Random Forest
    on 7,043 customer records with 19 features.
  </div>
  <div class="hero-stats">
    <div class="hero-stat-item">
      <div class="hero-stat-icon icon-blue">{ICN_USERS}</div>
      <div><div class="hero-stat-val">7,043</div><div class="hero-stat-lbl">Customers</div></div>
    </div>
    <div class="hero-stat-item">
      <div class="hero-stat-icon icon-green">{ICN_DB}</div>
      <div><div class="hero-stat-val">19</div><div class="hero-stat-lbl">Features</div></div>
    </div>
    <div class="hero-stat-item">
      <div class="hero-stat-icon icon-purple">{ICN_ACT}</div>
      <div><div class="hero-stat-val">2</div><div class="hero-stat-lbl">Algorithms</div></div>
    </div>
    <div class="hero-stat-item">
      <div class="hero-stat-icon icon-amber">{ICN_TGT}</div>
      <div><div class="hero-stat-val">80.55%</div><div class="hero-stat-lbl">Best Accuracy</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Predict Churn", "Model Analysis", "Visualizations"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown("""
    <div class="section section-alt">
      <div class="sec-title">Dataset at a Glance</div>
      <div class="sec-sub">Telco Customer Churn dataset from Kaggle · mosapabdelghany</div>
      <div class="metric-grid">
        <div class="metric-card">
          <div class="metric-label">Customers</div>
          <div class="metric-value">7,043</div>
          <div class="metric-note">total records</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Features</div>
          <div class="metric-value">19</div>
          <div class="metric-note">excl. customerID &amp; target</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Churn Rate</div>
          <div class="metric-value">26.5%</div>
          <div class="metric-note">1,869 churned customers</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Train / Test Split</div>
          <div class="metric-value" style="font-size:28px">80 / 20</div>
          <div class="metric-note">stratified · random_state = 42</div>
        </div>
      </div>
      <div style="text-align:center;margin-top:36px;font-size:11px;font-weight:700;
                  color:#94a3b8;text-transform:uppercase;letter-spacing:.08em">Key Features Used</div>
      <div class="tag-cloud">
        <span class="tag">Gender</span><span class="tag">Senior Citizen</span>
        <span class="tag">Partner</span><span class="tag">Dependents</span>
        <span class="tag">Tenure</span><span class="tag">Phone Service</span>
        <span class="tag">Multiple Lines</span><span class="tag">Internet Service</span>
        <span class="tag">Online Security</span><span class="tag">Online Backup</span>
        <span class="tag">Device Protection</span><span class="tag">Tech Support</span>
        <span class="tag">Streaming TV</span><span class="tag">Streaming Movies</span>
        <span class="tag">Contract</span><span class="tag">Paperless Billing</span>
        <span class="tag">Payment Method</span><span class="tag">Monthly Charges</span>
        <span class="tag">Total Charges</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section">
      <div class="sec-title">Model Performance</div>
      <div class="sec-sub">Head-to-head comparison on the test set · 1,409 samples</div>
      <div class="model-grid">

        <div class="model-card winner">
          <div class="winner-badge">Winner</div>
          <div class="model-name">Logistic Regression</div>
          <div class="model-metrics">
            <div><div class="mm-label">Accuracy</div><div class="mm-value blue">80.55%</div></div>
            <div><div class="mm-label">ROC-AUC</div><div class="mm-value blue">0.842</div></div>
            <div><div class="mm-label">F1 Score</div><div class="mm-value">60.40%</div></div>
            <div><div class="mm-label">Recall</div><div class="mm-value">55.88%</div></div>
          </div>
          <div class="model-divider"></div>
          <ul class="model-points">
            <li><span class="dot-blue"></span>Higher accuracy and ROC-AUC on this dataset</li>
            <li><span class="dot-blue"></span>Interpretable coefficients for business use</li>
            <li><span class="dot-blue"></span>Faster training — approx. 0.3 s vs 12 s</li>
            <li><span class="dot-blue"></span>Better generalisation with moderate class imbalance</li>
          </ul>
        </div>

        <div class="model-card">
          <div class="model-name">Random Forest</div>
          <div class="model-metrics">
            <div><div class="mm-label">Accuracy</div><div class="mm-value">78.28%</div></div>
            <div><div class="mm-label">ROC-AUC</div><div class="mm-value">0.826</div></div>
            <div><div class="mm-label">F1 Score</div><div class="mm-value">54.60%</div></div>
            <div><div class="mm-label">Recall</div><div class="mm-value">49.20%</div></div>
          </div>
          <div class="model-divider"></div>
          <ul class="model-points">
            <li><span class="dot-gray"></span>Handles non-linear feature interactions</li>
            <li><span class="dot-gray"></span>Built-in feature importance rankings</li>
            <li><span class="dot-gray"></span>Robust to outliers — no scaling required</li>
            <li><span class="dot-gray"></span>100 decision trees · n_estimators = 100</li>
          </ul>
        </div>

      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="section section-alt">
      <div class="sec-title">Key Churn Drivers</div>
      <div class="sec-sub">The most influential features identified by both models</div>
      <div class="driver-grid">
        <div class="driver-card">
          <div class="driver-top">
            <div class="driver-icon-wrap">{ICN_DOC}</div>
            <div class="driver-pct">42%</div>
          </div>
          <div class="driver-title">Month-to-Month Contracts</div>
          <div class="driver-desc">Customers without long-term commitments show significantly higher churn rates compared to one- or two-year contracts. Incentivising upgrades is the single biggest retention lever.</div>
        </div>
        <div class="driver-card">
          <div class="driver-top">
            <div class="driver-icon-wrap">{ICN_CLK}</div>
            <div class="driver-pct">34%</div>
          </div>
          <div class="driver-title">Low Customer Tenure</div>
          <div class="driver-desc">Newer customers are far more likely to leave, with churn dropping sharply after the first 12 months. Early onboarding and proactive engagement are critical in this window.</div>
        </div>
        <div class="driver-card">
          <div class="driver-top">
            <div class="driver-icon-wrap">{ICN_WIFI}</div>
            <div class="driver-pct">24%</div>
          </div>
          <div class="driver-title">Fiber Optic Service</div>
          <div class="driver-desc">Fiber optic subscribers churn at higher rates, potentially driven by pricing expectations or perceived service quality gaps relative to the premium cost.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section">
      <div class="sec-title">Business Recommendations</div>
      <div class="sec-sub">Actionable strategies derived from the predictive models</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='inner' style='padding-bottom:60px'>", unsafe_allow_html=True)
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown("""<div class="insight-card insight-blue">
          <div class="insight-head">Convert Month-to-Month Customers</div>
          <div class="insight-body">Offer discounts or loyalty rewards to monthly-plan customers to switch to 1- or 2-year contracts, dramatically reducing churn risk across the base.</div>
        </div>""", unsafe_allow_html=True)
    with rc2:
        st.markdown("""<div class="insight-card insight-amber">
          <div class="insight-head">Strengthen Early Onboarding</div>
          <div class="insight-body">Implement structured 30/60/90-day journeys. The first year is the highest-risk window — proactive engagement reduces attrition sharply in this period.</div>
        </div>""", unsafe_allow_html=True)
    with rc3:
        st.markdown("""<div class="insight-card insight-red">
          <div class="insight-head">Audit Fiber Optic Pricing</div>
          <div class="insight-body">Investigate quality-to-price perception among fiber customers. Address service gaps and introduce loyalty tiers to retain high-value subscribers.</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT CHURN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section section-alt" style="padding-bottom:32px">
      <div class="sec-title">Predict Customer Churn</div>
      <div class="sec-sub">Enter customer details — both models return a churn probability in real time</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='inner' style='padding-top:8px;padding-bottom:60px'>", unsafe_allow_html=True)
    with st.form("churn_form"):
        st.markdown('<div class="form-sep">Demographics</div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        gender     = d1.selectbox("Gender", ["Male", "Female"])
        senior     = d2.selectbox("Senior Citizen", ["No", "Yes"])
        partner    = d3.selectbox("Partner", ["No", "Yes"])
        dependents = d4.selectbox("Dependents", ["No", "Yes"])

        st.markdown('<div class="form-sep">Phone Services</div>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        phone_service  = p1.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = p2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

        st.markdown('<div class="form-sep">Internet Services</div>', unsafe_allow_html=True)
        i1, i2, i3, i4 = st.columns(4)
        internet        = i1.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = i2.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup   = i3.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protect  = i4.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        i5, i6, i7 = st.columns(3)
        tech_support     = i5.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv     = i6.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = i7.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.markdown('<div class="form-sep">Account & Billing</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        contract       = a1.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless      = a2.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = a3.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        b1, b2, b3 = st.columns(3)
        tenure          = b1.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = b2.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
        total_charges   = b3.number_input("Total Charges ($)", 0.0, 10000.0,
                                          float(tenure * monthly_charges), step=1.0)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        row = {col: 0 for col in FEATURE_COLS}
        row['SeniorCitizen']    = 1 if senior == "Yes" else 0
        row['tenure']           = tenure
        row['MonthlyCharges']   = monthly_charges
        row['TotalCharges']     = total_charges
        row['gender_Male']      = 1 if gender == "Male" else 0
        row['Partner_Yes']      = 1 if partner == "Yes" else 0
        row['Dependents_Yes']   = 1 if dependents == "Yes" else 0
        row['PhoneService_Yes'] = 1 if phone_service == "Yes" else 0
        if multiple_lines == "No phone service":
            row['MultipleLines_No phone service'] = 1
        elif multiple_lines == "Yes":
            row['MultipleLines_Yes'] = 1
        if internet == "Fiber optic":  row['InternetService_Fiber optic'] = 1
        elif internet == "No":         row['InternetService_No'] = 1
        for feat, val in [
            ('OnlineSecurity', online_security), ('OnlineBackup', online_backup),
            ('DeviceProtection', device_protect), ('TechSupport', tech_support),
            ('StreamingTV', streaming_tv), ('StreamingMovies', streaming_movies),
        ]:
            if val == "No internet service": row[f'{feat}_No internet service'] = 1
            elif val == "Yes":               row[f'{feat}_Yes'] = 1
        if contract == "One year":   row['Contract_One year'] = 1
        elif contract == "Two year": row['Contract_Two year'] = 1
        row['PaperlessBilling_Yes'] = 1 if paperless == "Yes" else 0
        if payment_method == "Credit card (automatic)":
            row['PaymentMethod_Credit card (automatic)'] = 1
        elif payment_method == "Electronic check":
            row['PaymentMethod_Electronic check'] = 1
        elif payment_method == "Mailed check":
            row['PaymentMethod_Mailed check'] = 1

        X = pd.DataFrame([row])[FEATURE_COLS]
        X[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(
            X[['tenure','MonthlyCharges','TotalCharges']])

        lr_pred = lr_model.predict(X)[0]
        lr_prob = lr_model.predict_proba(X)[0][1]
        rf_pred = rf_model.predict(X)[0]
        rf_prob = rf_model.predict_proba(X)[0][1]
        avg     = (lr_prob + rf_prob) / 2

        st.markdown("""
        <div class="section section-alt" style="padding-top:40px">
          <div class="sec-title">Prediction Results</div>
          <div class="sec-sub">Both models evaluated on the supplied customer profile</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='inner' style='padding-bottom:0'>", unsafe_allow_html=True)

        def render_result(col, name, pred, prob):
            cls   = "result-churn" if pred == 1 else "result-stay"
            lcls  = "churn"        if pred == 1 else "stay"
            label = "Will Churn"   if pred == 1 else "Likely to Stay"
            col.markdown(f"""
            <div class="model-tag">{name}</div>
            <div class="result-wrap {cls}">
              <div class="result-label {lcls}">{label}</div>
              <div class="result-prob">Churn probability: <b>{prob*100:.1f}%</b></div>
            </div>""", unsafe_allow_html=True)

        rc1, rc2 = st.columns(2)
        render_result(rc1, "Logistic Regression", lr_pred, lr_prob)
        render_result(rc2, "Random Forest", rf_pred, rf_prob)

        risk_col  = "#22c55e" if avg < 0.3 else ("#f59e0b" if avg < 0.6 else "#ef4444")
        risk_lbl  = ("Low risk — customer is unlikely to churn." if avg < 0.3 else
                     "Medium risk — consider proactive retention action." if avg < 0.6 else
                     "High risk — immediate retention intervention recommended.")
        st.markdown(f"""
        <div class="risk-card">
          <div class="risk-head">
            <span class="risk-head-title">Combined Risk Score</span>
            <span class="risk-score" style="color:{risk_col}">{avg*100:.1f}%</span>
          </div>
          <div class="risk-track">
            <div class="risk-fill" style="width:{avg*100:.1f}%;background:{risk_col}"></div>
          </div>
          <div class="risk-labels"><span>Low</span><span>Medium</span><span>High</span></div>
          <div class="risk-verdict" style="color:{risk_col}">{risk_lbl}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='gap48'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="section section-alt">
      <div class="sec-title">Full Performance Metrics</div>
      <div class="sec-sub">All evaluation metrics on the held-out test set · 1,409 samples</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='inner'>", unsafe_allow_html=True)
    st.markdown("""
    <table class="mtable">
      <thead>
        <tr><th>Metric</th><th>Logistic Regression</th><th>Random Forest</th><th>Winner</th></tr>
      </thead>
      <tbody>
        <tr><td>Accuracy</td>  <td class="best">80.55%</td><td>78.28%</td><td class="dim">LR</td></tr>
        <tr><td>Precision</td> <td class="best">65.72%</td><td>61.33%</td><td class="dim">LR</td></tr>
        <tr><td>Recall</td>    <td class="best">55.88%</td><td>49.20%</td><td class="dim">LR</td></tr>
        <tr><td>F1 Score</td>  <td class="best">60.40%</td><td>54.60%</td><td class="dim">LR</td></tr>
        <tr><td>ROC-AUC</td>   <td class="best">0.842</td> <td>0.826</td> <td class="dim">LR</td></tr>
      </tbody>
    </table>
    <div class="gap24"></div>
    """, unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="img-card"><div class="img-title">Confusion Matrices</div>', unsafe_allow_html=True)
        if (RESULTS / "confusion_matrices.png").exists():
            st.image(str(RESULTS / "confusion_matrices.png"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with cb:
        st.markdown('<div class="img-card"><div class="img-title">ROC Curves</div>', unsafe_allow_html=True)
        if (RESULTS / "roc_curves.png").exists():
            st.image(str(RESULTS / "roc_curves.png"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div class='gap16'></div>", unsafe_allow_html=True)

    cc, cd = st.columns(2)
    with cc:
        st.markdown('<div class="img-card"><div class="img-title">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
        if (RESULTS / "feature_importance.png").exists():
            st.image(str(RESULTS / "feature_importance.png"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with cd:
        st.markdown('<div class="img-card"><div class="img-title">Cross-Validation Scores (k = 5)</div>', unsafe_allow_html=True)
        if (RESULTS / "cross_validation.png").exists():
            st.image(str(RESULTS / "cross_validation.png"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div class='gap16'></div>", unsafe_allow_html=True)

    ce, cf = st.columns(2)
    with ce:
        st.markdown('<div class="img-card"><div class="img-title">Overall Model Comparison</div>', unsafe_allow_html=True)
        if (RESULTS / "model_comparison.png").exists():
            st.image(str(RESULTS / "model_comparison.png"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with cf:
        st.markdown('<div class="img-card"><div class="img-title">Error Analysis — Misclassified Samples</div>', unsafe_allow_html=True)
        if (RESULTS / "error_analysis.png").exists():
            st.image(str(RESULTS / "error_analysis.png"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section">
      <div class="sec-title">Why Logistic Regression Won</div>
      <div class="sec-sub">Despite being the simpler model, LR outperformed the ensemble on every metric</div>
    </div>
    """, unsafe_allow_html=True)

    WHY = [
        ("Approximately Linear Decision Boundary",
         "The churn/stay boundary in this feature space is close to linear. LR captures this directly, while Random Forest wastes capacity modelling non-linearities that do not exist in the data."),
        ("Dataset Size vs. Model Complexity",
         "Random Forest's 100 trees introduce high variance on ~5,600 training samples. LR's lower variance leads to better generalisation on the held-out test set."),
        ("Class Imbalance Handling",
         "With only 26.5% positive class, LR's probabilistic framework generalises more robustly, while RF tends to bias toward the majority class under moderate imbalance."),
        ("Feature Scaling Benefit",
         "StandardScaler normalises tenure, MonthlyCharges, and TotalCharges — directly benefiting LR's gradient-based optimisation. RF is scale-invariant so gains nothing from this step."),
    ]

    st.markdown("<div class='inner' style='padding-bottom:60px'>", unsafe_allow_html=True)
    wa, wb = st.columns(2)
    for i, (title, body) in enumerate(WHY):
        col = wa if i % 2 == 0 else wb
        col.markdown(f"""<div class="why-card">
          <div class="why-title">{title}</div>
          <div class="why-body">{body}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="section section-alt">
      <div class="sec-title">Exploratory Data Analysis</div>
      <div class="sec-sub">Visual insights from the Telco Customer Churn dataset before modelling</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='inner'>", unsafe_allow_html=True)

    st.markdown('<div class="img-card" style="margin-bottom:16px"><div class="img-title">Churn Distribution — Overall Dataset</div>', unsafe_allow_html=True)
    if (RESULTS / "churn_distribution.png").exists():
        st.image(str(RESULTS / "churn_distribution.png"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    v1, v2 = st.columns(2)
    with v1:
        st.markdown('<div class="img-card"><div class="img-title">Churn Rate by Contract Type</div>', unsafe_allow_html=True)
        if (RESULTS / "churn_by_contract.png").exists():
            st.image(str(RESULTS / "churn_by_contract.png"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with v2:
        st.markdown('<div class="img-card"><div class="img-title">Churn Rate by Internet Service</div>', unsafe_allow_html=True)
        if (RESULTS / "churn_by_internet.png").exists():
            st.image(str(RESULTS / "churn_by_internet.png"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div class='gap16'></div>", unsafe_allow_html=True)

    v3, v4 = st.columns(2)
    with v3:
        st.markdown('<div class="img-card"><div class="img-title">Monthly Charges Distribution by Churn</div>', unsafe_allow_html=True)
        if (RESULTS / "monthly_charges_by_churn.png").exists():
            st.image(str(RESULTS / "monthly_charges_by_churn.png"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with v4:
        st.markdown('<div class="img-card"><div class="img-title">Tenure Distribution by Churn</div>', unsafe_allow_html=True)
        if (RESULTS / "tenure_by_churn.png").exists():
            st.image(str(RESULTS / "tenure_by_churn.png"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section">
      <div class="sec-title">Key EDA Findings</div>
      <div class="sec-sub">Statistical takeaways from the exploratory analysis</div>
      <div class="metric-grid" style="margin-top:32px">
        <div class="metric-card">
          <div class="metric-label">Churn Rate</div>
          <div class="metric-value">26.54%</div>
          <div class="metric-note">moderately imbalanced</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Avg Monthly Charge</div>
          <div class="metric-value">$64.76</div>
          <div class="metric-note">across all customers</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Avg Tenure — Stay</div>
          <div class="metric-value">37.6 mo</div>
          <div class="metric-note">loyal customers</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Avg Tenure — Churn</div>
          <div class="metric-value">18.0 mo</div>
          <div class="metric-note">2x lower than stayers</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='gap48'></div>", unsafe_allow_html=True)
