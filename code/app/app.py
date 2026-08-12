# ============================================================
# Telecom Churn XAI — HOME  (multi-page app)
# MSc Dissertation — Rajarajachozhan V K (03189049)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(page_title="Churn XAI — Home", page_icon="📡", layout="wide")

# ---------- palette ----------
NAVY, OCEAN, SKY, AMBER, TIGER = "#023047", "#219ebc", "#8ecae6", "#ffb703", "#fb8500"

# ---------- shared CSS ----------
st.markdown(f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1300px; }}
h1, h2, h3 {{ color: {NAVY}; }}
.hero {{
  background: linear-gradient(120deg, {NAVY} 0%, {OCEAN} 100%);
  padding: 2.2rem 2.4rem; border-radius: 16px; color: white; margin-bottom: 1.5rem;
}}
.hero h1 {{ color: white; margin: 0 0 .3rem 0; font-size: 2.1rem; }}
.hero p  {{ color: #d8eef5; margin: 0; font-size: 1.05rem; }}
.card {{
  background: white; border: 1px solid #e4e9ee; border-left: 6px solid {OCEAN};
  border-radius: 12px; padding: 1.1rem 1.3rem; height: 100%;
}}
.card h4 {{ color: {NAVY}; margin: 0 0 .4rem 0; }}
.card p  {{ color: #46606e; margin: 0; font-size: .92rem; }}
.kpi {{ background: {SKY}22; border-radius: 12px; padding: 1rem 1.2rem; text-align: center; }}
.kpi .v {{ font-size: 1.9rem; font-weight: 700; color: {NAVY}; }}
.kpi .l {{ font-size: .8rem; color: {OCEAN}; text-transform: uppercase; letter-spacing: .5px; }}
</style>
""", unsafe_allow_html=True)

# ---------- load minimal artefacts for KPIs ----------
BASE    = Path(__file__).resolve().parent.parent
MODELS  = BASE / "models"
RESULTS = BASE / "outputs" / "results"

@st.cache_resource
def load_home():
    with open(MODELS / "test_data.pkl", "rb") as f:  td  = pickle.load(f)
    with open(RESULTS / "segmentation.pkl", "rb") as f: seg = pickle.load(f)
    return td, seg

td, seg = load_home()
n_cust = len(td["X_test_scaled"]); n_feat = len(td["feature_names"])
churn_rate = td["y_test"].mean() * 100

# ---------- hero ----------
st.markdown(f"""
<div class="hero">
  <h1>📡 Telecom Customer Churn — Explainable AI Platform</h1>
  <p>Predict churn · explain every decision with SHAP &amp; LIME · segment customers into
     actionable retention profiles. Built on the Cell2Cell dataset (Duke University).</p>
</div>
""", unsafe_allow_html=True)

# ---------- KPIs ----------
k1, k2, k3, k4 = st.columns(4)
for col, v, l in [(k1, f"{n_cust:,}", "Customers analysed"),
                  (k2, n_feat, "Behavioural features"),
                  (k3, f"{churn_rate:.0f}%", "Churn rate"),
                  (k4, seg["best_k"], "Retention segments")]:
    col.markdown(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>',
                 unsafe_allow_html=True)

st.write("")

# ---------- audience cards ----------
st.subheader("Choose your view")
c1, c2 = st.columns(2)
c1.markdown(f"""
<div class="card" style="border-left-color:{TIGER};">
  <h4>📈 Marketing View</h4>
  <p>Explore all customers in a sortable, filterable table by churn risk and segment.
     See each customer's recommended retention strategy. Upload your own customer file
     for batch scoring, and export results to Excel / CSV for Power BI.</p>
</div>""", unsafe_allow_html=True)
c2.markdown(f"""
<div class="card" style="border-left-color:{OCEAN};">
  <h4>🧠 AI / Engineer View</h4>
  <p>Inspect the SHAP vs LIME explanation methods head-to-head, the four-metric
     evaluation (fidelity, stability, sparsity, agreement), and the disagreement
     analysis that underpins the model's trustworthiness.</p>
</div>""", unsafe_allow_html=True)

# ---------- navigation buttons ----------
st.write("")
nav1, nav2 = st.columns(2)
if nav1.button("📈  Open Marketing View", use_container_width=True, type="primary"):
    st.switch_page("pages/marketing.py")
if nav2.button("🧠  Open AI / Engineer View", use_container_width=True):
    st.switch_page("pages/ai_engineer.py")

st.info("👈 You can also use the sidebar to switch between views.")

st.divider()
st.caption("MSc Dissertation — Comparative Evaluation of Explainable AI Methods for "
           "Customer Churn Prediction · Rajarajachozhan V K (03189049)")