import importlib


def train_selected_models_only(selected_models, allow_install=False, fast=False, uploaded_file=None):
    """Delegate to the legacy app implementation while refactoring progresses.

    This avoids duplicating large orchestration logic and keeps the public API
    stable for the UI and tests.
    """
    try:
        app = importlib.import_module("app")
        return app.train_selected_models_only(selected_models, allow_install=allow_install, fast=fast, uploaded_file=uploaded_file)
    except Exception:
        # If import fails for some reason, raise a clear error
        raise
