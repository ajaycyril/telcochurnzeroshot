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

This Space is a compact, production-minded demo showing the end-to-end ML lifecycle for a structured classification problem (Telco customer churn). It's intentionally educational and reproducible — the front-end explains the pipeline steps and provides safe defaults for hosting on Hugging Face Spaces.

## What the Space demonstrates
- Data ingestion: CSV upload or automatic Kaggle download (requires Space secrets `KAGGLE_USERNAME` and `KAGGLE_KEY`).
- Data preparation: type coercion, missing-value handling, domain-derived features (tenure buckets, charge ratios, service counts).
- Modeling: logistic regression, random forest, gradient boosting, optional XGBoost/CatBoost when available. Fast/Full modes control search breadth.
- Evaluation: Stratified CV, ROC/AUC, confusion matrices, calibrated probabilities, and a small ensemble option.
- Explainability: SHAP summary plots and per-sample explanations to help stakeholders interpret predictions.

## Pipeline contract (inputs / outputs / error modes)
- Inputs: CSV with standard Telco fields or upload any CSV with similar schema.
- Outputs: scorecard (table) of model metrics, best model summary, ROC and confusion plots, and SHAP visualizations.
- Errors: clear error messages if dataset cannot be found or if optional heavy libraries are missing. The UI uses `Fast` mode by default to avoid heavy installs at startup.

## Running locally
```powershell
# Create venv and install core deps
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt

# Optionally set Kaggle creds to allow automatic download
setx KAGGLE_USERNAME "<your-username>"
setx KAGGLE_KEY "<your-key>"

# Start the app
.venv\Scripts\python app.py
```

## Deploying to Hugging Face Spaces
1. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` under Space Settings -> Secrets.
2. Rebuild the Space. The app will attempt to write `~/.kaggle/kaggle.json` and use the Kaggle API to download the dataset.
3. If downloading during build fails, the app will attempt to pip-install `kaggle` at runtime; adding `kaggle` to `requirements.txt` makes this deterministic.

## Notes for reviewers
- The UI provides a narrative and streaming status updates for each pipeline phase so non-technical stakeholders can follow along.
- Fast mode is the safe default for hosting: it avoids heavy booster installs and uses compact hyperparameter grids.
- For production/SLA deployments, replace the local training step with a scheduled training job and export a serialized model artifact (job + model registry).
