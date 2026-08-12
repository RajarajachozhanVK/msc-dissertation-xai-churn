# ============================================================
# Churn XAI Dashboard — Phase 6: polished final
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

st.set_page_config(page_title="Telecom Churn XAI", page_icon="📊", layout="wide")

BASE    = Path(__file__).resolve().parent.parent
MODELS  = BASE / "models"
RESULTS = BASE / "outputs" / "results"

@st.cache_resource
def load_artifacts():
    with open(MODELS / "xgboost_tuned.pkl", "rb") as f:  xgb_model = pickle.load(f)
    with open(MODELS / "test_data.pkl", "rb") as f:      test_data = pickle.load(f)
    with open(RESULTS / "shap_outputs.pkl", "rb") as f:  shap_out  = pickle.load(f)
    with open(RESULTS / "segmentation.pkl", "rb") as f:  seg       = pickle.load(f)
    return xgb_model, test_data, shap_out, seg

xgb_model, test_data, shap_out, seg = load_artifacts()
X_test        = test_data["X_test_scaled"]
y_test        = test_data["y_test"]
feature_names = test_data["feature_names"]
segment_df    = seg["segment_df"]
examples      = shap_out["examples"]
seg_scaler, seg_pca, seg_kmeans = seg["scaler"], seg["pca"], seg["kmeans"]

@st.cache_data
def all_probs():
    return pd.Series(xgb_model.predict_proba(X_test)[:, 1], index=X_test.index)
probs = all_probs()

@st.cache_resource
def build_explainers():
    background = shap.sample(X_test, 100, random_state=42)
    shap_prob = shap.KernelExplainer(lambda x: xgb_model.predict_proba(x)[:, 1],
                                     background, link="identity")
    shap_tree = shap.TreeExplainer(xgb_model)
    cat_idx = [i for i, f in enumerate(feature_names) if X_test[f].nunique() <= 2]
    lime_expl = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_test.values, feature_names=feature_names,
        class_names=["Non-Churn", "Churn"], categorical_features=cat_idx,
        mode="classification", discretize_continuous=False, random_state=42)
    return shap_prob, shap_tree, lime_expl

shap_prob, shap_tree, lime_expl = build_explainers()
lime_pred = lambda x: xgb_model.predict_proba(pd.DataFrame(x, columns=feature_names))

def treeshap_logodds(rows_df):
    raw = np.asarray(shap_tree.shap_values(rows_df))
    if raw.ndim == 3: raw = raw[:, :, 1]
    return raw

def assign_segments(rows_df):
    sv = treeshap_logodds(rows_df)
    return seg_kmeans.predict(seg_pca.transform(seg_scaler.transform(sv)))

def band_plain(p):   # for CSV export (no emoji)
    return "HIGH" if p >= 0.66 else ("MEDIUM" if p >= 0.40 else "LOW")
def band_emoji(p):   # for on-screen display
    return "🔴 HIGH" if p >= 0.66 else ("🟠 MEDIUM" if p >= 0.40 else "🟢 LOW")

# ---------------- Header ----------------
st.title("📊 Telecom Churn — Explainable AI Dashboard")
st.caption("XGBoost churn prediction · SHAP & LIME explanations · marketer-oriented segmentation  |  "
           "MSc Dissertation — Rajarajachozhan V K")

tab_single, tab_batch = st.tabs(["🔍 Single customer", "📁 Batch scoring (file upload)"])

# =====================================================================
# TAB 1 — SINGLE CUSTOMER
# =====================================================================
with tab_single:
    st.sidebar.header("Select a customer")
    mode = st.sidebar.radio("Selection method", ["Archetype (quick-pick)", "By index number"])
    if mode == "Archetype (quick-pick)":
        label = st.sidebar.selectbox("Archetype", list(examples.keys()))
        cust_idx = examples[label]
    else:
        label = "Custom selection"
        cust_idx = st.sidebar.selectbox("Customer index", list(X_test.index), index=0)
    st.sidebar.markdown(f"**Selected ID:** `{cust_idx}`")
    st.sidebar.divider()
    st.sidebar.caption("Tip: try 'High-Risk Churner' then 'Missed Churner' to see the "
                       "model's blind spot to quiet churners.")

    p = float(probs.loc[cust_idx]); actual = int(y_test.loc[cust_idx])
    actual_txt = "Churned" if actual == 1 else "Did not churn"
    if   p >= 0.66: band, colour = "HIGH risk", "🔴"
    elif p >= 0.40: band, colour = "MEDIUM risk", "🟠"
    else:           band, colour = "LOW risk", "🟢"

    st.subheader(f"Customer `{cust_idx}` — {label}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{p*100:.1f}%")
    c2.metric("Risk band", f"{colour} {band}")
    c3.metric("Actual outcome", actual_txt)
    st.progress(min(max(p, 0.0), 1.0))
    pred = int(p >= 0.5)
    if pred == actual:
        st.success(f"✅ Model prediction ({'Churn' if pred else 'No churn'}) matches the actual outcome.")
    else:
        st.warning(f"⚠️ Model predicted **{'Churn' if pred else 'No churn'}** but the customer actually "
                   f"**{actual_txt.lower()}** — a {'false positive' if pred==1 else 'false negative'}.")
    st.divider()

    st.header("🎯 Segment & recommended action")
    seg_id = int(assign_segments(X_test.loc[[cust_idx]])[0])
    seg_row = segment_df.loc[seg_id]
    s1, s2 = st.columns([1, 2])
    with s1:
        st.metric("Assigned segment", seg_row["segment"])
        st.caption(f"Cluster {seg_id} · avg P(churn) {seg_row['mean_P_churn']:.2f} · "
                   f"{int(seg_row['n'])} churners")
    with s2:
        st.markdown("**Recommended retention strategy**")
        st.success(seg_row["strategy"])
    st.divider()

    st.header("Why did the model predict this?")
    st.caption("SHAP (probability space, left) vs LIME (right). The two methods often disagree — "
               "LIME's linear surrogate fits XGBoost poorly (R²≈0.12), a core finding of this project.")
    row = X_test.loc[[cust_idx]]
    colL, colR = st.columns(2)
    with colL:
        st.subheader("SHAP explanation")
        with st.spinner("Computing SHAP..."):
            sv = shap_prob.shap_values(row, silent=True)[0]
            expl = shap.Explanation(values=sv, base_values=shap_prob.expected_value,
                                    data=row.values[0], feature_names=feature_names)
            fig = plt.figure(figsize=(8, 6))
            shap.plots.waterfall(expl, max_display=12, show=False)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("Exact contribution of each feature to churn probability. Red → churn, blue → retain.")
    with colR:
        st.subheader("LIME explanation")
        with st.spinner("Computing LIME..."):
            e = lime_expl.explain_instance(X_test.loc[cust_idx].values, lime_pred,
                                           num_features=12, labels=(1,), num_samples=3000)
            wser = pd.Series(dict(e.as_list(label=1))).sort_values()
            fig2, ax = plt.subplots(figsize=(8, 6))
            cols = ["#e74c3c" if v > 0 else "#3498db" for v in wser.values]
            ax.barh(range(len(wser)), wser.values, color=cols, edgecolor="black")
            ax.set_yticks(range(len(wser))); ax.set_yticklabels(wser.index, fontsize=8)
            ax.axvline(0, color="black", lw=0.8); ax.set_xlabel("LIME weight (→ churn if positive)")
            plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)
        st.caption("Local linear approximation. Its ranking often differs from SHAP's on this model.")

# =====================================================================
# TAB 2 — BATCH SCORING
# =====================================================================
with tab_batch:
    st.header("📁 Batch scoring")
    st.write("Upload a CSV or Excel file of customers to score them all at once — "
             "each row gets a churn probability, risk band, segment, and retention strategy.")

    d1, d2 = st.columns(2)
    template = pd.DataFrame(columns=feature_names)
    d1.download_button("⬇️ Download blank template (CSV)",
                       template.to_csv(index=False).encode("utf-8-sig"),
                       "churn_template.csv", "text/csv",
                       help="Empty file with the exact 66 columns the model expects")
    sample = X_test.head(20).copy()
    d2.download_button("⬇️ Download sample (20 customers, CSV)",
                       sample.to_csv(index=False).encode("utf-8-sig"),
                       "churn_sample.csv", "text/csv",
                       help="A ready-to-score example you can re-upload immediately")

    st.divider()
    up = st.file_uploader("Upload customer file", type=["csv", "xlsx"])

    if up is not None:
        try:
            df_in = pd.read_excel(up) if up.name.endswith("xlsx") else pd.read_csv(up)
            st.write(f"Uploaded **{len(df_in)}** rows, **{df_in.shape[1]}** columns.")

            missing = [c for c in feature_names if c not in df_in.columns]
            if missing:
                st.error(f"❌ File is missing {len(missing)} required columns, e.g.: "
                         f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")
                st.info("Use the blank template above to see the exact required columns.")
            else:
                Xin = df_in[feature_names].copy()
                with st.spinner(f"Scoring {len(Xin)} customers..."):
                    p_all   = xgb_model.predict_proba(Xin)[:, 1]
                    seg_all = assign_segments(Xin)
                    # display version (emoji) and export version (plain) kept separate
                    disp = pd.DataFrame({
                        "churn_probability": np.round(p_all, 4),
                        "risk_band":  [band_emoji(v) for v in p_all],
                        "prediction": np.where(p_all >= 0.5, "Churn", "No churn"),
                        "segment":    [segment_df.loc[int(s), "segment"] for s in seg_all],
                        "strategy":   [segment_df.loc[int(s), "strategy"] for s in seg_all],
                    }, index=df_in.index)
                    export = disp.copy()
                    export["risk_band"] = [band_plain(v) for v in p_all]   # clean text for CSV

                st.success(f"✅ Scored {len(disp)} customers.")
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted churners", int((p_all >= 0.5).sum()))
                m2.metric("High risk", int((p_all >= 0.66).sum()))
                m3.metric("Avg churn probability", f"{p_all.mean()*100:.1f}%")

                st.dataframe(disp, use_container_width=True)
                st.download_button("⬇️ Download results (CSV)",
                                   export.to_csv().encode("utf-8-sig"),
                                   "churn_results.csv", "text/csv")

                st.subheader("Segment breakdown")
                counts = disp["segment"].value_counts()
                fig3, ax3 = plt.subplots(figsize=(8, 3.5))
                ax3.bar(range(len(counts)), counts.values,
                        color=["#e74c3c", "#3498db", "#2ecc71"][:len(counts)], edgecolor="black")
                ax3.set_xticks(range(len(counts)))
                ax3.set_xticklabels(counts.index, rotation=15, ha="right", fontsize=9)
                ax3.set_ylabel("Customers")
                plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)
        except Exception as e:
            st.error(f"❌ Could not process the file:\n\n{e}")

# ---------------- Footer ----------------
st.divider()
with st.expander("Dataset & model overview"):
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Customers", f"{len(X_test):,}"); o2.metric("Features", len(feature_names))
    o3.metric("Churn rate", f"{y_test.mean()*100:.1f}%"); o4.metric("Segments", seg["best_k"])
    st.write(f"Best model: **XGBoost** (test AUC ≈ 0.68) · Segmentation: **{seg['method']}** "
             f"(silhouette {seg['silhouette']:.3f})")
    st.caption("Note: probability-space SHAP (KernelExplainer) is used for on-screen readability; "
               "all quantitative evaluation in the dissertation uses exact log-odds TreeSHAP.")
with st.expander("All churn segments & strategies"):
    st.dataframe(segment_df[["n", "mean_P_churn", "segment", "strategy"]],
                 use_container_width=True)