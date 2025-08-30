import os
import pytest
from datetime import datetime

# Use the app module functions directly
import app


def test_smoke_fast_train_runs(tmp_path):
    """Smoke test: load local CSV and run a quick Fast-mode training for Logistic Regression only.
    This test is resource-light and suitable for CI: it uses small grids and skips heavy boosters.
    """
    # Ensure the dataset exists in repo (the canonical Telco file should be present)
    csv_path = os.path.join(os.path.dirname(app.__file__), app.TELCO_CSV)
    assert os.path.exists(csv_path), f"Telco CSV not found at {csv_path}"

    # Call the loader explicitly with the local CSV
    df = app.load_telco_data(uploaded_file=csv_path)
    assert df is not None and not df.empty

    # Run fast training for Logistic Regression only
    selected_models = ["Logistic Regression"]
    result = app.train_selected_models_only(selected_models, allow_install=False, fast=True, uploaded_file=csv_path)

    assert isinstance(result, dict)
    assert result.get("success") is True, f"Training failed or not successful: {result.get('error')}"
    assert "scorecard" in result and not result["scorecard"].empty
    assert "model_paths" in result
    # One model should have been persisted
    assert any(result["model_paths"].values())

    # Ensure saved model files exist
    for p in result["model_paths"].values():
        assert os.path.exists(p)

    # Cleanup persisted models that the test created
    for p in list(result["model_paths"].values()):
        try:
            os.remove(p)
        except Exception:
            pass
