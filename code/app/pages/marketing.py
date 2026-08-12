# ============================================================
# Telecom Churn XAI — MARKETING VIEW (4 tabs)
# Overview · Data Computation · Churn Groups · What-If + Leaderboard
# MSc Dissertation — Rajarajachozhan V K (03189049)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle, io, random
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from pathlib import Path

st.set_page_config(page_title="Marketing", page_icon="📈", layout="wide")

NAVY, OCEAN, SKY, AMBER, TIGER = "#023047", "#219ebc", "#8ecae6", "#ffb703", "#fb8500"
SEG_COLORS = [TIGER, OCEAN, SKY, AMBER, "#8d99ae"]

st.markdown(f"""
<style>
.block-container {{ padding-top: 1.2rem; max-width: 1450px; }}
h1,h2,h3 {{ color:{NAVY}; }}
.hero {{ background:linear-gradient(120deg,{NAVY} 0%,{OCEAN} 100%);
  padding:1.5rem 2rem; border-radius:14px; color:white; margin-bottom:1rem; }}
.hero h1 {{ color:white; margin:0; font-size:1.9rem; }}
.hero p {{ color:#d8eef5; margin:.35rem 0 0 0; font-size:1.05rem; }}
div[data-testid="stMetric"] {{ background:{SKY}22; border-radius:10px;
  padding:.7rem 1rem; border-left:5px solid {OCEAN}; }}
div[data-testid="stMetricValue"] {{ font-size:2rem; color:{NAVY}; }}
.stTabs [data-baseweb="tab"] {{ font-size:1.05rem; font-weight:600; }}
</style>""", unsafe_allow_html=True)

BASE    = Path(__file__).resolve().parent.parent.parent
MODELS  = BASE / "models"; RESULTS = BASE / "outputs" / "results"

@st.cache_resource
def load():
    with open(MODELS/"xgboost_tuned.pkl","rb") as f: xgb=pickle.load(f)
    with open(MODELS/"test_data.pkl","rb") as f:      td =pickle.load(f)
    with open(RESULTS/"segmentation.pkl","rb") as f:  seg=pickle.load(f)
    return xgb, td, seg
xgb, td, seg = load()
X=td["X_test_scaled"]; y=td["y_test"]; feats=td["feature_names"]
segdf=seg["segment_df"]; sc,pca,km = seg["scaler"],seg["pca"],seg["kmeans"]
med = X.median()

@st.cache_resource
def get_tree_explainer():
    import shap as _shap
    return _shap.TreeExplainer(xgb)
tree = get_tree_explainer()

def band(p): return "HIGH" if p>=0.66 else ("MEDIUM" if p>=0.40 else "LOW")

def safe_sheet(name):
    for ch in r'[]:*?/\\':
        name = name.replace(ch, "-")
    return name[:31]

def seg_of(rows_df):
    raw=np.asarray(tree.shap_values(rows_df))
    S=raw[:,:,1] if raw.ndim==3 else raw
    return km.predict(pca.transform(sc.transform(S)))

@st.cache_data(show_spinner="Scoring all customers (first load only)...")
def score_all():
    proba=xgb.predict_proba(X)[:,1]
    csv=RESULTS/"xgb_shap_values.csv"; S=None
    if csv.exists():
        t=pd.read_csv(csv,index_col=0)
        if t.shape[0]==X.shape[0] and all(c in t.columns for c in feats):
            S=t.reindex(columns=feats).values
    if S is None:
        raw=np.asarray(tree.shap_values(X))
        S=raw[:,:,1] if raw.ndim==3 else raw
    ids=km.predict(pca.transform(sc.transform(S)))
    return pd.DataFrame({
        "customer_id":X.index,"churn_probability":np.round(proba,4),
        "risk_band":[band(v) for v in proba],
        "prediction":np.where(proba>=0.5,"Churn","No churn"),
        "actual":np.where(y.values==1,"Churned","Retained"),
        "segment":[segdf.loc[int(s),"segment"] for s in ids],
        "strategy":[segdf.loc[int(s),"strategy"] for s in ids]})
scored=score_all()

# ---------- chart helpers ----------
def donut(counts, title):
    fig,ax=plt.subplots(figsize=(4.2,4.2))
    ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%", pctdistance=0.8,
           colors=SEG_COLORS[:len(counts)], wedgeprops=dict(width=0.42,edgecolor="white"),
           textprops=dict(color=NAVY,fontsize=9))
    ax.set_title(title,color=NAVY,fontsize=12,fontweight="bold")
    ctr=plt.Circle((0,0),0.58,color="white"); ax.add_artist(ctr)
    ax.text(0,0,f"{int(counts.sum()):,}\ncustomers",ha="center",va="center",
            color=NAVY,fontsize=11,fontweight="bold")
    return fig

def gauge(p):
    fig,ax=plt.subplots(figsize=(4.2,2.6),subplot_kw={"aspect":"equal"})
    col=TIGER if p>=0.66 else (AMBER if p>=0.40 else OCEAN)
    ax.add_patch(mp.Wedge((0,0),1,0,180,width=0.32,facecolor="#e6ecf0"))
    ax.add_patch(mp.Wedge((0,0),1,180-180*p,180,width=0.32,facecolor=col))
    ax.text(0,-0.05,f"{p*100:.0f}%",ha="center",va="center",fontsize=26,
            color=NAVY,fontweight="bold")
    ax.text(0,-0.4,band(p)+" RISK",ha="center",va="center",fontsize=11,color=col,fontweight="bold")
    ax.set_xlim(-1.1,1.1); ax.set_ylim(-0.5,1.1); ax.axis("off")
    return fig

# ---------- hero ----------
st.markdown('<div class="hero"><h1>📈 Marketing — Customer Churn Intelligence</h1>'
            '<p>Explore, score, and action the customer base by churn risk and retention segment.</p></div>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🧮 Data Computation", "👥 Churn Groups", "🎛️ What-If & Leaderboard"])

# ================= TAB 1 — OVERVIEW =================
with tab1:
    k=st.columns(4)
    k[0].metric("Customers", f"{len(scored):,}")
    k[1].metric("Predicted churners", f"{(scored.prediction=='Churn').sum():,}")
    k[2].metric("High risk", f"{(scored.risk_band=='HIGH').sum():,}")
    k[3].metric("Avg churn prob", f"{scored.churn_probability.mean()*100:.1f}%")
    st.write("")
    d1,d2=st.columns(2)
    d1.pyplot(donut(scored["segment"].value_counts(), "Segment share"))
    d2.pyplot(donut(scored["risk_band"].value_counts().reindex(["HIGH","MEDIUM","LOW"]).fillna(0),
                    "Risk share"))
    st.divider()
    st.subheader("🔎 Filter & sort customers")
    f=st.columns(3)
    rs=f[0].multiselect("Risk band",["HIGH","MEDIUM","LOW"],["HIGH","MEDIUM","LOW"])
    ss=f[1].multiselect("Segment",list(segdf["segment"]),list(segdf["segment"]))
    lo,hi=f[2].slider("Churn probability",0.0,1.0,(0.0,1.0),0.01)
    v=scored[scored.risk_band.isin(rs)&scored.segment.isin(ss)&
             scored.churn_probability.between(lo,hi)]
    st.caption(f"Showing **{len(v):,}** of {len(scored):,} customers.")
    st.dataframe(v,use_container_width=True,height=400,hide_index=True,
        column_config={"churn_probability":st.column_config.ProgressColumn(
            "Churn prob",min_value=0.0,max_value=1.0,format="%.2f"),"customer_id":"ID"})
    x1,x2,_=st.columns([1,1,3])
    x1.download_button("⬇️ CSV",v.to_csv(index=False).encode("utf-8-sig"),
                       "customers.csv","text/csv",use_container_width=True)
    def xls(df):
        b=io.BytesIO()
        with pd.ExcelWriter(b,engine="openpyxl") as w:
            df.to_excel(w,index=False,sheet_name="Customers")
            segdf.to_excel(w,index=False,sheet_name="Segments")
        return b.getvalue()
    x2.download_button("⬇️ Excel",xls(v),"customers.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)

# ================= TAB 2 — DATA COMPUTATION =================
with tab2:
    st.subheader("Score a customer three ways")

    @st.cache_data
    def demo_identity(cid):
        r = random.Random(int(cid))
        first = ["Alex","Sam","Jordan","Priya","Wei","Maria","John","Aisha","Tom","Nina",
                 "Omar","Leah","Raj","Sofia","Ken","Ivy","Ben","Zara","Luca","Mei"]
        last  = ["Smith","Patel","Chen","Garcia","Khan","Jones","Kumar","Silva","Lee","Ali",
                 "Brown","Diaz","Nguyen","Wang","Roy","Costa","Park","Shah","Ivanov","Cruz"]
        return f"{r.choice(first)} {r.choice(last)}", "+1 "+"".join(str(r.randint(0,9)) for _ in range(10))

    method = st.radio("Input method",
        ["Pick existing customer","Manual entry (key drivers)","Upload file"], horizontal=True)

    def show_result(xrow_df, cid=None):
        p = float(xgb.predict_proba(xrow_df)[:,1][0])
        sid = int(seg_of(xrow_df)[0]); srow = segdf.loc[sid]
        if cid is not None:
            nm, mob = demo_identity(cid)
            st.caption(f"🧪 Demo identity (synthetic): **{nm}** · {mob} · real ID `{cid}`")
        g, info = st.columns([1, 1.4])
        g.pyplot(gauge(p))
        info.metric("Segment", srow["segment"])
        info.metric("Prediction", "Churn" if p>=0.5 else "No churn")
        info.success(f"**Action:** {srow['strategy']}")

    if method == "Pick existing customer":
        s1, s2 = st.columns([1,2])
        by = s1.selectbox("Search by", ["Customer ID","Demo name","Demo mobile"])
        q  = s2.text_input(f"Type {by.lower()} (partial ok)", "")
        ids = list(X.index)
        if q.strip():
            ql = q.strip().lower()
            if by == "Customer ID":
                matches = [i for i in ids if ql in str(i).lower()][:50]
            elif by == "Demo name":
                matches = [i for i in ids if ql in demo_identity(i)[0].lower()][:50]
            else:
                matches = [i for i in ids if ql in demo_identity(i)[1].lower()][:50]
        else:
            matches = ids[:50]
        if not matches:
            st.warning("No match. Try a different search.")
        else:
            labels = {i: f"{i} — {demo_identity(i)[0]} ({demo_identity(i)[1]})" for i in matches}
            cid = st.selectbox("Select customer", matches, format_func=lambda i: labels[i])
            show_result(X.loc[[cid]], cid)

    elif method == "Manual entry (key drivers)":
        st.info("ℹ️ The model uses **all 66 features**. Position this customer on the **10 strongest "
                "churn drivers** using percentiles (0% = lowest, 100% = highest vs all customers). "
                "The other 56 features stay at typical (median) values. "
                "Nothing computes until you click **Calculate churn**.")
        LEGEND = {"retcalls":"Retention calls made","mou":"Minutes of use (monthly)",
            "eqpdays":"Days on current handset (equipment age)","months":"Tenure",
            "changem":"Change in minutes vs prior period","overage":"Overage minutes billed",
            "recchrge":"Recurring monthly charge","uniqsubs":"Unique subscriptions",
            "dropvce":"Dropped calls","custcare":"Customer-care calls"}
        drivers = [d for d in LEGEND if d in feats][:10]
        def pctl_to_scaled(f, pct): return float(np.percentile(X[f].values, pct))
        with st.form("manual_entry"):
            st.caption("Each slider = where this customer sits vs the whole base (50% = median customer).")
            entered = {}; cols = st.columns(2)
            for i, f in enumerate(drivers):
                entered[f] = cols[i%2].slider(f"{LEGEND[f]}",0,100,50,5,
                    help="0% = lowest · 100% = highest", key=f"pct_{f}")
            submitted = st.form_submit_button("🔮 Calculate churn", type="primary",
                                              use_container_width=True)
        if submitted:
            base = med.copy()
            for f in drivers: base[f] = pctl_to_scaled(f, entered[f])
            show_result(pd.DataFrame([base], columns=feats))
        else:
            st.caption("👆 Set the percentiles and click **Calculate churn**.")

    else:
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Template",
            pd.DataFrame(columns=feats).to_csv(index=False).encode("utf-8-sig"),
            "template.csv","text/csv")
        c2.download_button("⬇️ Sample",
            X.head(20).to_csv(index=False).encode("utf-8-sig"),"sample.csv","text/csv")
        up = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx"])
        if up is not None:
            di = pd.read_excel(up) if up.name.endswith("xlsx") else pd.read_csv(up)
            miss = [c for c in feats if c not in di.columns]
            if miss:
                st.error(f"❌ Missing {len(miss)} columns, e.g. {miss[:6]}")
            else:
                Xi = di[feats].copy()
                with st.spinner(f"Scoring {len(Xi)}..."):
                    pa = xgb.predict_proba(Xi)[:,1]; sids = seg_of(Xi)
                    out = pd.DataFrame({"churn_probability":np.round(pa,4),
                        "risk_band":[band(v) for v in pa],
                        "prediction":np.where(pa>=0.5,"Churn","No churn"),
                        "segment":[segdf.loc[int(s),"segment"] for s in sids],
                        "strategy":[segdf.loc[int(s),"strategy"] for s in sids]},index=di.index)
                st.success(f"✅ Scored {len(out)}.")
                st.dataframe(out, use_container_width=True, height=300)
                st.download_button("⬇️ Results CSV",
                    out.to_csv().encode("utf-8-sig"),"results.csv","text/csv")

# ================= TAB 3 — CHURN GROUPS =================
with tab3:
    st.subheader("Customers grouped by churn type (segment)")
    st.caption("Each marketing team can download only their segment's customers.")
    for i,segname in enumerate(segdf["segment"]):
        grp=scored[scored.segment==segname]
        with st.container(border=True):
            a,b=st.columns([3,1])
            a.markdown(f"### {segname}")
            a.caption(f"{len(grp):,} customers · avg churn "
                      f"{grp.churn_probability.mean()*100:.1f}% · "
                      f"strategy: {segdf.iloc[i]['strategy']}")
            b.download_button("⬇️ CSV",grp.to_csv(index=False).encode("utf-8-sig"),
                              f"{segname.replace(' ','_').replace('/','-')}.csv","text/csv",
                              key=f"csv{i}",use_container_width=True)
            bb=io.BytesIO()
            with pd.ExcelWriter(bb,engine="openpyxl") as w:
                grp.to_excel(w,index=False,sheet_name=safe_sheet(segname))
            b.download_button("⬇️ Excel",bb.getvalue(),
                f"{segname.replace(' ','_').replace('/','-')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"xls{i}",use_container_width=True)
    st.divider()
    allb=io.BytesIO()
    with pd.ExcelWriter(allb,engine="openpyxl") as w:
        for segname in segdf["segment"]:
            scored[scored.segment==segname].to_excel(
                w,index=False,sheet_name=safe_sheet(segname))
    st.download_button("⬇️ Download ALL segments (multi-sheet Excel)",allb.getvalue(),
        "all_segments.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ================= TAB 4 — WHAT-IF & LEADERBOARD =================
with tab4:
    # ---- Top-N at-risk leaderboard ----
    st.subheader("🏆 Top at-risk customers")
    n = st.slider("How many to show", 10, 200, 50, 10)
    top = scored.sort_values("churn_probability", ascending=False).head(n).reset_index(drop=True)
    top.index = top.index + 1
    st.dataframe(top[["customer_id","churn_probability","risk_band","segment","strategy"]],
        use_container_width=True, height=380,
        column_config={"churn_probability":st.column_config.ProgressColumn(
            "Churn prob",min_value=0.0,max_value=1.0,format="%.2f"),"customer_id":"ID"})
    st.download_button(f"⬇️ Download top {n} (CSV)",
        top.to_csv(index=False).encode("utf-8-sig"),
        f"top_{n}_at_risk.csv","text/csv")

    st.divider()

    # ---- What-If simulator ----
    st.subheader("🎛️ What-if simulator")
    st.caption("Pick a customer, then adjust their driver percentiles to see how churn risk changes.")
    wc = st.selectbox("Customer", list(X.index), key="whatif_cust")
    base_row = X.loc[[wc]].copy()
    base_p = float(xgb.predict_proba(base_row)[:,1][0])

    LEG = {"retcalls":"Retention calls","mou":"Minutes of use","eqpdays":"Handset age (days)",
           "months":"Tenure","changem":"Change in minutes","overage":"Overage minutes",
           "recchrge":"Monthly charge","custcare":"Care calls"}
    wdrivers = [d for d in LEG if d in feats][:8]

    # current percentile of this customer per driver (for slider defaults)
    def current_pct(f, val):
        return int(round((X[f].values < val).mean()*100))

    with st.form("whatif"):
        cols = st.columns(2); newvals = {}
        for i,f in enumerate(wdrivers):
            cur = current_pct(f, float(X.loc[wc, f]))
            newvals[f] = cols[i%2].slider(f"{LEG[f]} (now ~{cur}%)",0,100,cur,5, key=f"wi_{f}")
        go = st.form_submit_button("▶️ Simulate", type="primary", use_container_width=True)

    if go:
        sim = base_row.copy()
        for f in wdrivers:
            sim.iloc[0, feats.index(f)] = float(np.percentile(X[f].values, newvals[f]))
        new_p = float(xgb.predict_proba(sim)[:,1][0])
        delta = new_p - base_p
        m = st.columns(3)
        m[0].metric("Original churn", f"{base_p*100:.1f}%")
        m[1].metric("Simulated churn", f"{new_p*100:.1f}%", f"{delta*100:+.1f} pts")
        arrow = "🔻 reduced" if delta<0 else ("🔺 increased" if delta>0 else "unchanged")
        m[2].metric("Effect", arrow)
        if delta < -0.05:
            st.success(f"This intervention **lowers** churn risk by {abs(delta)*100:.1f} points — worth pursuing.")
        elif delta > 0.05:
            st.warning(f"These changes **raise** churn risk by {delta*100:.1f} points.")
        else:
            st.info("Minimal effect on churn risk from these changes.")
    else:
        st.caption("👆 Adjust sliders and click **Simulate** to see the churn-risk change.")