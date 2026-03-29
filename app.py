import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="",
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
.stApp {
    background:
        radial-gradient(circle at top, rgba(96,165,250,.16) 0%, rgba(96,165,250,0) 30%),
        linear-gradient(180deg, #ffffff 0%, #f5f9ff 100%);
}
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

:root {
    --bg-main: #ffffff;
    --bg-panel: #f4f8ff;
    --bg-card: #f8fbff;
    --bg-soft: #eef5ff;
    --line-soft: rgba(96, 165, 250, 0.16);
    --text-main: #0f172a;
    --text-soft: #64748b;
    --text-dim: #7f8ca8;
    --accent: #4f8ff7;
    --accent-soft: #7cb7ff;
    --depth-shadow-soft: 0 12px 28px rgba(96, 165, 250, 0.10), 0 4px 12px rgba(15, 23, 42, 0.04);
    --depth-shadow-mid: 0 18px 40px rgba(96, 165, 250, 0.12), 0 6px 16px rgba(15, 23, 42, 0.05);
    --depth-highlight: inset 0 1px 0 rgba(255,255,255,0.92), inset 0 -1px 0 rgba(148,163,184,0.06);
}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    width: fit-content;
    margin: 26px auto 22px;
    background: linear-gradient(180deg, #fbfdff 0%, #f1f7ff 100%);
    border: 1px solid rgba(96, 165, 250, 0.14);
    border-radius: 28px;
    gap: 12px;
    padding: 12px;
    justify-content: center;
    position: sticky;
    top: 14px;
    z-index: 200;
    box-shadow: var(--depth-shadow-mid), var(--depth-highlight);
}
.stTabs [data-baseweb="tab"] {
    font-size: 18px;
    font-weight: 600;
    color: var(--accent) !important;
    padding: 14px 28px;
    border: 1px solid transparent;
    border-radius: 18px;
    margin-bottom: 0;
    letter-spacing: .01em;
    background: rgba(96,165,250,.06);
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom-color: transparent !important;
    background: linear-gradient(180deg, #7cb7ff 0%, #4f8ff7 100%) !important;
    box-shadow: 0 10px 24px rgba(96,165,250,.24), inset 0 1px 0 rgba(255,255,255,.22);
}
.stTabs [data-baseweb="tab-panel"] { padding: 0; }

/* ── Hero ──────────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(160deg, #fbfdff 0%, #f1f7ff 100%);
    text-align: center;
    padding: 80px 60px 64px;
    border: 1px solid var(--line-soft);
    border-radius: 26px;
    margin: 8px 60px 28px;
    box-shadow: var(--depth-shadow-mid), var(--depth-highlight);
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(96,165,250,.10);
    border: 1px solid rgba(96,165,250,.14);
    border-radius: 999px;
    padding: 7px 18px;
    font-size: 12.5px;
    font-weight: 600;
    color: var(--accent-soft);
    letter-spacing: .03em;
    margin-bottom: 28px;
    box-shadow: var(--depth-shadow-soft), var(--depth-highlight);
}
.hero-title {
    font-size: 58px;
    font-weight: 900;
    line-height: 1.08;
    color: var(--text-main);
    margin-bottom: 4px;
    letter-spacing: -.02em;
}
.hero-accent { color: #2563eb; }
.hero-sub {
    font-size: 17px;
    color: var(--text-soft);
    max-width: 560px;
    margin: 16px auto 40px;
    line-height: 1.65;
    font-weight: 400;
}
.hero-stats {
    display: inline-flex;
    align-items: center;
    background: linear-gradient(180deg, rgba(255,255,255,.95) 0%, rgba(245,249,255,.98) 100%);
    border: 1px solid var(--line-soft);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: var(--depth-shadow-mid), var(--depth-highlight);
    transform: translateY(-1px);
}
.hero-stat-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 16px 28px;
    border-right: 1px solid var(--line-soft);
    background: transparent;
}
.hero-stat-item:last-child { border-right: none; }
.hero-stat-icon {
    width: 34px; height: 34px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.icon-blue   { background: rgba(59,130,246,.14); }
.icon-green  { background: rgba(34,197,94,.14); }
.icon-purple { background: rgba(168,85,247,.14); }
.icon-amber  { background: rgba(245,158,11,.14); }
.hero-stat-val { font-size: 16px; font-weight: 700; color: var(--text-main); line-height: 1; }
.hero-stat-lbl { font-size: 11.5px; color: var(--text-soft); font-weight: 500; margin-top: 2px; }

/* ── Sections ──────────────────────────────────────────────────────────── */
.section     { padding: 60px; }
.section-alt { background: transparent; }
.sec-title   { font-size: 30px; font-weight: 800; color: var(--accent); text-align: left; letter-spacing: -.02em; margin-bottom: 6px; }
.sec-sub     { font-size: 14.5px; color: var(--text-soft); text-align: left; margin-bottom: 40px; }
.inner       { padding: 0 60px; }

/* ── Metric cards ──────────────────────────────────────────────────────── */
.metric-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; }
.metric-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%); border-radius: 20px; padding: 28px 24px;
    box-shadow: var(--depth-shadow-soft), var(--depth-highlight); text-align: center; border: 1px solid var(--line-soft);
    transform: translateY(-1px);
}
.metric-label { font-size: 11px; font-weight: 700; color: var(--text-soft); text-transform: uppercase; letter-spacing: .07em; margin-bottom: 10px; }
.metric-value { font-size: 38px; font-weight: 800; color: var(--accent); line-height: 1; letter-spacing: -.02em; }
.metric-note  { font-size: 12px; color: var(--text-soft); margin-top: 7px; }

/* ── Tags ──────────────────────────────────────────────────────────────── */
.tag-cloud { display: flex; flex-wrap: wrap; justify-content: center; gap: 9px; margin-top: 28px; }
.tag {
    background: rgba(96,165,250,.08); border: 1px solid rgba(96,165,250,.16); border-radius: 999px;
    padding: 7px 18px; font-size: 12.5px; font-weight: 500; color: var(--accent);
}

/* ── Model cards ───────────────────────────────────────────────────────── */
.model-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.model-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%); border-radius: 18px; padding: 36px;
    box-shadow: var(--depth-shadow-soft), var(--depth-highlight); border: 1.5px solid var(--line-soft); position: relative;
}
.model-card.winner {
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37,99,235,.08), var(--depth-shadow-soft), var(--depth-highlight);
}
.winner-badge {
    position: absolute; top: -13px; left: 28px;
    background: #2563eb; color: #fff;
    font-size: 11px; font-weight: 700; padding: 4px 14px;
    border-radius: 999px; letter-spacing: .05em; text-transform: uppercase;
}
.model-name   { font-size: 20px; font-weight: 700; color: var(--text-main); margin-bottom: 24px; }
.model-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
.mm-label { font-size: 10.5px; font-weight: 700; color: var(--text-soft); text-transform: uppercase; letter-spacing: .07em; margin-bottom: 4px; }
.mm-value { font-size: 30px; font-weight: 800; color: var(--text-main); letter-spacing: -.01em; }
.mm-value.blue { color: #2563eb; }
.model-divider { height: 1px; background: rgba(255,255,255,.06); margin-bottom: 18px; }
.model-points  { list-style: none; }
.model-points li { display: flex; align-items: center; gap: 9px; font-size: 13.5px; color: var(--text-soft); margin-bottom: 9px; }
.dot-blue { width: 6px; height: 6px; border-radius: 50%; background: #2563eb; flex-shrink: 0; }
.dot-gray { width: 6px; height: 6px; border-radius: 50%; background: #cbd5e1; flex-shrink: 0; }

/* ── Driver cards ──────────────────────────────────────────────────────── */
.driver-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 18px; }
.driver-card {
    background: linear-gradient(180deg, #ffffff 0%, #f4f8ff 100%); border-radius: 20px; padding: 28px;
    box-shadow: var(--depth-shadow-soft), var(--depth-highlight); border: 1px solid var(--line-soft);
    transform: translateY(-1px);
}
.driver-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px; }
.driver-icon-wrap { width: 54px; height: 54px; border-radius: 999px; background: linear-gradient(180deg, rgba(59,130,246,.26) 0%, rgba(59,130,246,.16) 100%); display: flex; align-items: center; justify-content: center; }
.driver-pct   { font-size: 26px; font-weight: 800; color: #2563eb; letter-spacing: -.01em; }
.driver-title { font-size: 15px; font-weight: 700; color: var(--text-main); margin-bottom: 8px; }
.driver-desc  { font-size: 13.5px; color: var(--text-soft); line-height: 1.6; }

/* ── Insights ──────────────────────────────────────────────────────────── */
.insight-card { border-radius: 14px; padding: 22px 24px; }
.insight-blue  { background: linear-gradient(180deg, #ffffff 0%, #f4f8ff 100%); border: 1px solid var(--line-soft); }
.insight-amber { background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%); border: 1px solid var(--line-soft); }
.insight-red   { background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%); border: 1px solid var(--line-soft); }
.insight-head  { font-size: 13.5px; font-weight: 700; margin-bottom: 7px; }
.insight-blue  .insight-head { color: #1d4ed8; }
.insight-amber .insight-head { color: #92400e; }
.insight-red   .insight-head { color: #991b1b; }
.insight-body  { font-size: 13px; line-height: 1.6; }
.insight-blue  .insight-body { color: var(--text-soft); }
.insight-amber .insight-body { color: var(--text-soft); }
.insight-red   .insight-body { color: var(--text-soft); }

/* ── Metrics table ─────────────────────────────────────────────────────── */
.mtable {
    width: 100%; border-collapse: collapse;
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%); border-radius: 16px; overflow: hidden;
    box-shadow: var(--depth-shadow-soft), var(--depth-highlight); border: 1px solid var(--line-soft);
}
.mtable th {
    background: rgba(96,165,250,.05); font-size: 11px; font-weight: 700; color: var(--text-soft);
    text-transform: uppercase; letter-spacing: .07em; padding: 11px 16px;
    text-align: left; border-bottom: 1px solid rgba(255,255,255,.06);
}
.mtable td { padding: 11px 16px; font-size: 13px; color: var(--text-main); border-bottom: 1px solid rgba(59,130,246,.05); font-weight: 500; }
.mtable tr:last-child td { border-bottom: none; }
.mtable .best { color: #2563eb; font-weight: 700; }
.mtable .dim  { color: var(--text-soft); }

/* ── Image cards ───────────────────────────────────────────────────────── */
.img-card { background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%); border-radius: 16px; padding: 16px; box-shadow: var(--depth-shadow-soft), var(--depth-highlight); border: 1px solid var(--line-soft); transform: translateY(-1px); }
.img-title { font-size: 12px; font-weight: 700; color: var(--text-main); margin-bottom: 10px; letter-spacing: .01em; }
.analysis-narrow { max-width: 980px; margin: 0 auto; }

/* ── Why cards ─────────────────────────────────────────────────────────── */
.why-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%); border-radius: 16px; padding: 28px;
    box-shadow: var(--depth-shadow-soft), var(--depth-highlight); border: 1px solid var(--line-soft); margin-bottom: 16px;
}
.why-title { font-size: 14.5px; font-weight: 700; color: var(--text-main); margin-bottom: 7px; }
.why-body  { font-size: 13.5px; color: var(--text-soft); line-height: 1.6; }

/* ── Prediction results ────────────────────────────────────────────────── */
.result-wrap  { border-radius: 16px; padding: 28px 30px; box-shadow: var(--depth-shadow-soft), var(--depth-highlight); transform: translateY(-1px); }
.result-churn { background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%); border: 1px solid rgba(248,113,113,.18); }
.result-stay  { background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%); border: 1px solid rgba(74,222,128,.16); }
.result-label { font-size: 20px; font-weight: 800; margin-bottom: 6px; }
.result-label.churn { color: #dc2626; }
.result-label.stay  { color: #4ade80; }
.result-prob  { font-size: 13px; color: var(--text-soft); }
.result-prob b { color: var(--text-main); }
.model-tag {
    font-size: 11px; font-weight: 700; color: var(--text-soft);
    text-transform: uppercase; letter-spacing: .07em; margin-bottom: 10px;
}

/* ── Risk bar ──────────────────────────────────────────────────────────── */
.risk-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%); border-radius: 16px; padding: 28px 32px;
    box-shadow: var(--depth-shadow-soft), var(--depth-highlight); border: 1px solid var(--line-soft); margin-top: 20px;
    transform: translateY(-1px);
}
.risk-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 16px; }
.risk-head-title { font-size: 14px; font-weight: 700; color: var(--text-main); }
.risk-score  { font-size: 28px; font-weight: 800; letter-spacing: -.01em; }
.risk-track  { height: 10px; background: rgba(96,165,250,.10); border-radius: 999px; overflow: hidden; margin-bottom: 8px; }
.risk-fill   { height: 100%; border-radius: 999px; }
.risk-labels { display: flex; justify-content: space-between; font-size: 11.5px; color: var(--text-soft); font-weight: 500; }
.risk-verdict { margin-top: 14px; font-size: 13.5px; font-weight: 600; }

/* ── Form labels ───────────────────────────────────────────────────────── */
div[data-testid="stSelectbox"] > label,
div[data-testid="stSlider"] > label,
div[data-testid="stNumberInput"] > label {
    font-size: 12px !important;
    font-weight: 700 !important;
    color: var(--text-soft) !important;
    text-transform: uppercase !important;
    letter-spacing: .05em !important;
}
div[data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important;
    border: 1px solid rgba(96,132,189,.24) !important;
    font-size: 13px !important;
    background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%) !important;
    min-height: 40px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.92), 0 6px 16px rgba(96,165,250,.08) !important;
}
div[data-testid="stNumberInput"] > div {
    border-radius: 10px !important;
    border: 1px solid rgba(96,132,189,.24) !important;
    background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.92), 0 6px 16px rgba(96,165,250,.08) !important;
}
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] input {
    font-size: 13px !important;
    color: var(--text-main) !important;
}
div[data-testid="stSelectbox"] svg,
div[data-testid="stNumberInput"] svg {
    color: #64748b !important;
}
div[data-testid="stSlider"] {
    padding-top: 2px !important;
}
.form-sep {
    font-size: 13px; font-weight: 800; color: var(--accent);
    text-transform: uppercase; letter-spacing: .12em;
    padding: 20px 0 8px; border-bottom: 1px solid rgba(96,165,250,.10); margin-bottom: 8px;
}

/* ── Submit button ─────────────────────────────────────────────────────── */
div[data-testid="stFormSubmitButton"] {
    margin-top: 10px !important;
}
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(180deg, #3b82f6 0%, #2563eb 58%, #1d4ed8 100%) !important;
    color: #fff !important;
    border-radius: 14px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 0 20px !important;
    min-height: 50px !important;
    border: 1px solid rgba(255,255,255,.16) !important;
    width: 100% !important; letter-spacing: .02em !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 18px 32px rgba(37,99,235,.28), inset 0 1px 0 rgba(255,255,255,.30), inset 0 -2px 0 rgba(29,78,216,.35) !important;
    transition: transform .18s ease, box-shadow .18s ease, background-color .18s ease !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(180deg, #4f8ff7 0%, #2563eb 60%, #1d4ed8 100%) !important;
    box-shadow: 0 22px 36px rgba(37,99,235,.32), inset 0 1px 0 rgba(255,255,255,.34), inset 0 -2px 0 rgba(29,78,216,.4) !important;
    transform: translateY(-2px) !important;
}

.gap16 { height: 16px; }
.gap24 { height: 24px; }
.gap48 { height: 48px; }

/* ── Predict layout ────────────────────────────────────────────────────── */
.predict-shell { padding: 36px 60px 60px; }
.predict-panel {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid var(--line-soft);
    border-radius: 22px;
    box-shadow: var(--depth-shadow-mid), var(--depth-highlight);
    padding: 28px;
    transform: translateY(-1px);
}
.predict-panel-head {
    margin-bottom: 6px;
}
.predict-panel-title {
    font-size: 34px;
    font-weight: 800;
    color: var(--text-main);
    letter-spacing: -.02em;
    margin-bottom: 6px;
}
.predict-panel-sub {
    font-size: 14px;
    color: var(--text-soft);
    line-height: 1.6;
}
.result-placeholder {
    border: 1px dashed rgba(96,132,189,.24);
    border-radius: 16px;
    padding: 26px;
    background: rgba(96,165,250,.05);
    color: var(--text-soft);
    font-size: 14px;
    line-height: 1.7;
    margin-top: 8px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.92);
}

@media (max-width: 1100px) {
    .metric-grid { grid-template-columns: repeat(2,1fr); }
    .driver-grid { grid-template-columns: 1fr; }
}

@media (max-width: 768px) {
    .hero { padding: 56px 24px 48px; margin: 8px 24px 28px; }
    .hero-title { font-size: 42px; }
    .section { padding: 40px 24px; }
    .inner { padding: 0 24px; }
    .predict-shell { padding: 24px 24px 48px; }
    .hero-stats {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        width: 100%;
    }
    .hero-stat-item {
        border-right: none;
        border-bottom: 1px solid var(--line-soft);
    }
    .hero-stat-item:nth-last-child(-n+2) { border-bottom: none; }
    .model-grid,
    .metric-grid { grid-template-columns: 1fr; }
}
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


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Overview", "Predict Churn"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT CHURN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='predict-shell'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="predict-panel">
      <div class="predict-panel-head">
        <div class="predict-panel-title">Predict Customer Churn</div>
        <div class="predict-panel-sub">Adjust the 19 customer features below and run the prediction to see results.</div>
      </div>
    """, unsafe_allow_html=True)

    with st.form("churn_form"):
        st.markdown('<div class="form-sep">Demographics</div>', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        gender     = d1.selectbox("Gender", ["Male", "Female"])
        senior     = d2.selectbox("Senior Citizen", ["No", "Yes"])
        d3, d4 = st.columns(2)
        partner    = d3.selectbox("Partner", ["No", "Yes"])
        dependents = d4.selectbox("Dependents", ["No", "Yes"])

        st.markdown('<div class="form-sep">Phone Services</div>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        phone_service  = p1.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = p2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

        st.markdown('<div class="form-sep">Internet Services</div>', unsafe_allow_html=True)
        i1, i2 = st.columns(2)
        internet        = i1.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = i2.selectbox("Online Security", ["No", "Yes", "No internet service"])
        i3, i4 = st.columns(2)
        online_backup   = i3.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protect  = i4.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        i5, i6 = st.columns(2)
        tech_support     = i5.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv     = i6.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        i7, i8 = st.columns(2)
        streaming_movies = i7.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        i8.empty()

        st.markdown('<div class="form-sep">Account & Billing</div>', unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        contract       = a1.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless      = a2.selectbox("Paperless Billing", ["Yes", "No"])
        a3 = st.columns(1)[0]
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
        <div class="predict-panel">
          <div class="predict-panel-head">
            <div class="predict-panel-title">Prediction Results</div>
            <div class="predict-panel-sub">Both models were evaluated using the customer profile you entered.</div>
          </div>
        """, unsafe_allow_html=True)

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
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='gap48'></div>", unsafe_allow_html=True)
