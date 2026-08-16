# Comparative Evaluation of Explainable AI Methods for Customer Churn Prediction in Telecommunications

**A SHAP- and LIME-Based Framework with Marketer-Oriented Segmentation**

MSc Artificial Intelligence and Computer Science — Northeastern University London
Author: Rajarajachozhan V K (03189049) · Supervisor: Ibukun Afolabi · Course: LDSCI7237

---

## Overview

This project compares SHAP and LIME as post-hoc explanation methods for telecom
customer churn prediction, evaluates them across four quantitative metrics
(fidelity, stability, sparsity, agreement), extends SHAP outputs into a
marketer-oriented segmentation of churners, and delivers the whole pipeline
through an interactive multi-page Streamlit dashboard.

The dataset is the Cell2Cell telecom churn dataset (Duke University), 71,047
customer records across ~66 usable features after cleanup, with ~29% churn.

## Key findings

- **Model performance:** XGBoost was the best model (test AUC ≈ 0.68), consistent
  with published Cell2Cell benchmarks. SMOTE was applied inside the cross-validation
  folds (imbalanced-learn pipeline) to avoid oversampling leakage.
- **SHAP vs LIME:** SHAP outperformed LIME on fidelity (deletion-AOPC 0.58 vs 0.54,
  both above random 0.38), stability, and sparsity. The two methods showed near-zero
  agreement (~15% top-10 overlap, signed rank correlation ≈ 0) — the disagreement problem.
- **Why they disagree:** LIME's local linear surrogate fits XGBoost poorly (R² ≈ 0.12)
  but fits the linear Logistic Regression almost perfectly (R² ≈ 0.99), showing LIME's
  fidelity collapses on high-interaction models.
- **Segmentation:** PCA(10) + k-means on churner SHAP vectors yielded three interpretable
  segments (silhouette 0.18) — At-Risk/Retention-Contact, Low-Engagement/Passive, and
  High-Usage/Handset-Gap — each mapped to a retention strategy. The model under-flags the
  quiet (low-engagement) churners.

## Repository structure

```text
code/
├── notebooks/          Day1-5 pipeline + figure export
│   ├── Day1_Hyperparameter_Tuning.ipynb    Model training, SMOTE-in-CV, tuning
│   ├── Day2_SHAP_Explanations.ipynb         TreeSHAP / LinearExplainer
│   ├── Day3_LIME_Explanations.ipynb         LIME + SHAP-vs-LIME comparison
│   ├── Day4_Evaluation_Metrics.ipynb        Four-metric evaluation
│   ├── Day5_SHAP_Segmentation.ipynb         PCA + k-means churn segments
│   └── Export_Figures.ipynb                 Regenerates all 23 figures from artefacts
├── app/                Streamlit dashboard (multi-page)
│   ├── app.py                               Home page
│   ├── pages/marketing.py                   Marketing view (4 tabs)
│   ├── pages/ai_engineer.py                 AI / Engineer view (4 tabs)
│   └── .streamlit/config.toml               Theme
├── data/               Cell2Cell CSVs (see Data note below)
├── models/             Trained models (.pkl)
├── outputs/
│   ├── figures/        23 dissertation figures (300 DPI PNG)
│   └── results/        SHAP/LIME matrices, metrics, segmentation (.csv/.pkl)
└── requirements.txt
```

## Running the analysis (notebooks)

1. Install dependencies: `pip install -r requirements.txt`
2. Run the notebooks in order (Day1 → Day5). Each saves its outputs to the
   models/results folders that the next notebook reads.
3. `Export_Figures.ipynb` regenerates all figures from saved artefacts (no re-training).

Notebooks were developed in Google Colab; random seeds are fixed at 42 throughout.

## Running the dashboard (Streamlit)

The dashboard runs locally and loads the saved models/results (it does not retrain).

```bash
conda create -n churn-app python=3.11 -y
conda activate churn-app
pip install -r requirements.txt
cd code/app
streamlit run app.py
```
- **`app/app_v1_backup.py`** is an earlier single-page prototype, retained to
  document the dashboard's evolution to the current multi-page design.

The app opens at `http://localhost:8501` with three views:
- **Home** — project overview, KPIs, navigation.
- **Marketing** — Overview (donut charts, sortable/filterable customer table, Excel/CSV
  export), Data Computation (score a customer via search / percentile inputs / file upload),
  Churn Groups (per-segment downloads), What-If & Leaderboard (top-N at-risk + what-if simulator).
- **AI / Engineer** — Model Metrics (four-metric evaluation), SHAP vs LIME (live per-customer
  comparison), Disagreement Finding (R² 0.99 vs 0.12), Figures (browse the 23 figures).

## Notes

- **`random_forest_tuned.pkl` (133 MB) is excluded from version control** as it exceeds
  GitHub's 100 MB limit. Regenerate it by running `Day1_Hyperparameter_Tuning.ipynb`.
- **Data:** the Cell2Cell dataset originates from the Duke University Teradata Center,
  distributed via Kaggle. Refer to the Kaggle licence for redistribution terms.
- **Demo identities:** names/mobile numbers shown in the dashboard's customer search are
  synthetic and clearly labelled — the real dataset is anonymised.
- Library versions are pinned in `requirements.txt` for reproducibility.

## Deliverables

- Trained models: Logistic Regression, Random Forest, XGBoost (tuned)
- SHAP and LIME explanations on a canonical 500-customer evaluation set
- Four-metric SHAP-vs-LIME comparison
- SHAP-based churn segmentation with retention-strategy mapping
- Interactive multi-page Streamlit dashboard (marketing + technical views)
