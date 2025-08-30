import os
import joblib
from datetime import datetime
from .data import engineer_features, get_preprocessor, split_xy
from .modeling import get_tuned_models_selected if False else None

def train_selected_models_only(selected_models, allow_install=False, fast=False, uploaded_file=None):
    # This is a lightweight wrapper kept as a placeholder; the full logic lives in app.py until migrated fully.
    raise NotImplementedError("train_selected_models_only is moved in the refactor; use the original app.py implementation or re-run migration step.")
