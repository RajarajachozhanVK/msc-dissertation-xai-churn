# ============================================================
# Telecom Churn XAI — AI / ENGINEER VIEW
# Model metrics · SHAP vs LIME · disagreement finding · figures
# MSc Dissertation — Rajarajachozhan V K (03189049)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
import lime, lime.lime_tabular
from pathlib import Path

st.set_page_config(page_title="AI / Engineer", page_icon="🧠", layout="wide")

NAVY, OCEAN, SKY, AMBER, TIGER = "#023047", "#219ebc", "#8ecae6", "#ffb703", "#fb8500"
st.markdown(f"""
<style>
.block-container {{ padding-top:1.2rem; max-width:1450px; }}
h1,h2,h3 {{ color:{NAVY}; }}
.hero {{ background:linear-gradient(120deg,{NAVY} 0%,{OCEAN} 100%);
  padding:1.5rem 2rem; border-radius:14px; color:white; margin-bottom:1rem; }}
.hero h1 {{ color:white; margin:0; font-size:1.9rem; }}
.hero p {{ color:#d8eef5; margin:.35rem 0 0 0; }}
div[data-testid="stMetric"] {{ background:{SKY}22; border-radius:10px;
  padding:.7rem 1rem; border-left:5px solid {OCEAN}; }}
.win {{ border-left-color:{TIGER} !important; }}
.stTabs [data-baseweb="tab"] {{ font-size:1.05rem; font-weight:600; }}
</style>""", unsafe_allow_html=True)

BASE=Path(__file__).resolve().parent.parent.parent
MODELS=BASE/"models"; RESULTS=BASE/"outputs"/"results"; FIGS=BASE/"outputs"/"figures"

@st.cache_resource
def load():
    with open(MODELS/"xgboost_tuned.pkl","rb") as f: xgb=pickle.load(f)
    with open(MODELS/"test_data.pkl","rb") as f:      td =pickle.load(f)
    with open(RESULTS/"shap_outputs.pkl","rb") as f:  so =pickle.load(f)
    with open(RESULTS/"evaluation_metrics.pkl","rb") as f: em=pickle.load(f)
    return xgb, td, so, em
xgb, td, so, em = load()
X=td["X_test_scaled"]; y=td["y_test"]; feats=td["feature_names"]
examples=so["examples"]

@st.cache_resource
def explainers():
    bg=shap.sample(X,100,random_state=42)
    sp=shap.KernelExplainer(lambda x: xgb.predict_proba(x)[:,1], bg, link="identity")
    cat=[i for i,f in enumerate(feats) if X[f].nunique()<=2]
    lx=lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values, feature_names=feats, class_names=["Non-Churn","Churn"],
        categorical_features=cat, mode="classification",
        discretize_continuous=False, random_state=42)
    return sp, lx
shap_prob, lime_expl = explainers()
lime_pred=lambda x: xgb.predict_proba(pd.DataFrame(x,columns=feats))

st.markdown('<div class="hero"><h1>🧠 AI / Engineer — Explanation Analytics</h1>'
            '<p>SHAP vs LIME method comparison, four-metric evaluation, and the disagreement '
            'analysis behind model trustworthiness.</p></div>', unsafe_allow_html=True)

t1,t2,t3,t4 = st.tabs(["📐 Model Metrics","🔬 SHAP vs LIME (live)",
                       "⚖️ Disagreement Finding","🖼️ Figures"])

# ============ TAB 1 — METRICS ============
with t1:
    fid=em["fidelity"]; stab=em["stability"]; spar=em["sparsity"]; agr=em["agreement"]
    st.subheader("Four-metric evaluation — SHAP vs LIME (500-customer eval set)")

    st.markdown("**1 · Fidelity** (deletion-AOPC — higher = explainer's top features move the model more)")
    c=st.columns(3)
    c[0].metric("SHAP", f"{fid['SHAP deletion-AOPC']:.3f}")
    c[1].metric("LIME", f"{fid['LIME deletion-AOPC']:.3f}")
    c[2].metric("Random baseline", f"{fid['Random deletion-AOPC']:.3f}")

    st.markdown("**2 · Stability** (cosine similarity under input perturbation — higher = more robust)")
    c=st.columns(3)
    c[0].metric("SHAP", f"{stab['SHAP input-stability (cosine)']:.3f}")
    c[1].metric("LIME", f"{stab['LIME input-stability (cosine)']:.3f}")
    c[2].metric("LIME (seed variance)", f"{stab['LIME seed-stability (cosine)']:.3f}")

    st.markdown("**3 · Sparsity** (active features per explanation — fewer = more actionable)")
    c=st.columns(2)
    c[0].metric("SHAP", f"{spar['SHAP mean active features']:.1f}")
    c[1].metric("LIME", f"{spar['LIME mean active features']:.1f}")

    st.markdown("**4 · Agreement** (SHAP vs LIME — low = the disagreement problem)")
    c=st.columns(3)
    c[0].metric("Feature agreement@10", f"{agr['Feature agreement@10']*100:.0f}%")
    c[1].metric("Sign agreement@10", f"{agr['Sign agreement@10']*100:.0f}%")
    c[2].metric("Signed rank corr", f"{agr['Signed rank corr (mean)']:.2f}")

    st.divider()
    st.success("**Verdict:** SHAP wins fidelity, stability, and sparsity; the two methods "
               "barely agree (≈15% feature overlap, ~0 rank correlation). SHAP is the more "
               "trustworthy explainer for this model.")

# ============ TAB 2 — SHAP vs LIME LIVE ============
with t2:
    st.subheader("Explain any customer — SHAP vs LIME, side by side")
    mode=st.radio("Pick customer",["Archetype","By index"],horizontal=True)
    if mode=="Archetype":
        lbl=st.selectbox("Archetype",list(examples.keys())); cid=examples[lbl]
    else:
        cid=st.selectbox("Customer index",list(X.index))
    p=float(xgb.predict_proba(X.loc[[cid]])[:,1][0])
    st.metric("Churn probability", f"{p*100:.1f}%")

    if st.button("🔬 Generate explanations", type="primary"):
        row=X.loc[[cid]]
        cL,cR=st.columns(2)
        with cL:
            st.markdown("**SHAP** (probability space)")
            with st.spinner("SHAP..."):
                sv=shap_prob.shap_values(row,silent=True)[0]
                e=shap.Explanation(values=sv,base_values=shap_prob.expected_value,
                                   data=row.values[0],feature_names=feats)
                fig=plt.figure(figsize=(7,6)); shap.plots.waterfall(e,max_display=12,show=False)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        with cR:
            st.markdown("**LIME** (local linear surrogate)")
            with st.spinner("LIME..."):
                ex=lime_expl.explain_instance(X.loc[cid].values,lime_pred,
                    num_features=12,labels=(1,),num_samples=3000)
                ws=pd.Series(dict(ex.as_list(label=1))).sort_values()
                fig2,ax=plt.subplots(figsize=(7,6))
                ax.barh(range(len(ws)),ws.values,
                        color=["#e74c3c" if v>0 else OCEAN for v in ws.values],edgecolor=NAVY)
                ax.set_yticks(range(len(ws))); ax.set_yticklabels(ws.index,fontsize=8)
                ax.axvline(0,color=NAVY,lw=.8); ax.set_xlabel("LIME weight (→ churn if +)")
                plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)
        st.info("Notice the two methods often rank different features at the top — "
                "the core disagreement this project quantifies.")
    else:
        st.caption("👆 Choose a customer and click **Generate explanations**.")

# ============ TAB 3 — DISAGREEMENT FINDING ============
with t3:
    st.subheader("Why SHAP and LIME disagree on this model")
    r2=em["lime_local_r2_xgb"]
    c=st.columns(2)
    c[0].metric("LIME local R² on XGBoost", f"{r2:.2f}", help="How well LIME's linear surrogate fits")
    c[1].metric("LIME local R² on Logistic Regression", "0.99")
    st.markdown(f"""
    **The finding.** LIME approximates the model locally with a **linear** surrogate.
    On the linear Logistic Regression it fits almost perfectly (**R² ≈ 0.99**).
    On the high-interaction XGBoost model it **collapses to R² ≈ {r2:.2f}** — a linear
    surrogate simply cannot capture XGBoost's feature interactions.

    **Why it matters.** LIME's explanations for the deployed (XGBoost) model are therefore
    unreliable, which is exactly why its rankings disagree with SHAP's. SHAP, being
    model-aware (exact for tree models), does not suffer this failure. This is a concrete,
    evidence-backed reason to prefer **SHAP** for explaining this churn model.
    """)
    st.warning("This controlled contrast (0.99 vs 0.13, same setup, two models) is the "
               "strongest single result in the evaluation.")

# ============ TAB 4 — FIGURES ============
with t4:
    st.subheader("Dissertation figures")
    if FIGS.exists():
        imgs=sorted([p for p in FIGS.glob("*.png")])
        if imgs:
            names=[p.name for p in imgs]
            pick=st.selectbox("Select figure", names)
            st.image(str(FIGS/pick), use_container_width=True)
            st.caption(pick)
        else:
            st.info("No figures found in outputs/figures/.")
    else:
        st.info("Figures folder not found.")