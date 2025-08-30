---
title: "Telco Churn — Zero-Shot (Pro: SHAP + Gallery + Scoring)"
emoji: "📈"
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.31.0
app_file: app.py
pinned: false
---

# Telco Customer Churn — Zero‑Shot (AI‑assisted)

**What this Space does**
- Downloads the Kaggle Telco churn dataset via Space secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`)
- Trains **CatBoost** with a compact **Optuna** sweep (trials/time sliders)
- Tunes thresholds (F2, Youden) and reports **ROC‑AUC / PR‑AUC / Accuracy / Brier**
- Exports a detailed **.ipynb** with **context, provenance, master prompt, code, and plots**
- Shows **ROC, PR, calibration, confusion matrices**, **SHAP summary**, and **SHAP dependence** plots
- Provides a **scoring form** to compute a churn probability for a single customer

## Local dev (optional)
```bash
pip install -r requirements.txt
export KAGGLE_USERNAME=...; export KAGGLE_KEY=...
python app.py  # launches Gradio locally
```
