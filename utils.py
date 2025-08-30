import os
import sys
import json
import subprocess
import zipfile
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

APP_BUILD_ID = datetime.now().strftime('%Y%m%d_%H%M%S')

# Constants
TELCO_KAGGLE_REF = "blastchar/telco-customer-churn"
TELCO_ZIP = "telco-customer-churn.zip"
import os
import sys
import subprocess
import hashlib
from datetime import datetime

# CSV location relative to this module
TELCO_CSV = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def generate_hash(input_string):
    """Generate a hash from an input string for use as a unique ID"""
    return hashlib.md5(str(input_string).encode()).hexdigest()[:8]

def ensure_optional_libs(allow_install):
    available = {"xgboost": False, "lightgbm": False, "catboost": False}
    try:
        import xgboost  # noqa
        available["xgboost"] = True
    except Exception:
        if allow_install:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost==1.7.6"]) 
                import xgboost  # noqa
                available["xgboost"] = True
            except Exception:
                pass
    try:
        import lightgbm  # noqa
        available["lightgbm"] = True
    except Exception:
        if allow_install:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm>=3.3.0"], timeout=60)
                import lightgbm  # noqa
                available["lightgbm"] = True
            except Exception:
                available["lightgbm"] = False
    try:
        import catboost  # noqa
        available["catboost"] = True
    except Exception:
        if allow_install:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost==1.2.5"])
                import catboost  # noqa
                available["catboost"] = True
            except Exception:
                pass
    return available


def kaggle_download_telco():
    """Attempt to download the Telco dataset via Kaggle API or CLI."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Debug: Checking Kaggle credentials...")
    print(f"KAGGLE_USERNAME present: {'KAGGLE_USERNAME' in os.environ}")
    print(f"KAGGLE_KEY present: {'KAGGLE_KEY' in os.environ}")

    if os.path.exists(TELCO_CSV):
        return True, f"Found existing CSV: {TELCO_CSV}"

    # Try Python API
    try:
        kaggle_user = os.environ.get("KAGGLE_USERNAME")
        kaggle_key = os.environ.get("KAGGLE_KEY")
        if kaggle_user and kaggle_key:
            kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
            os.makedirs(kaggle_dir, exist_ok=True)
            kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
            if not os.path.exists(kaggle_json):
                with open(kaggle_json, "w") as fh:
                    json.dump({"username": kaggle_user, "key": kaggle_key}, fh)
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(TELCO_KAGGLE_REF, path=".", unzip=True)
        if os.path.exists(TELCO_CSV):
            return True, f"Downloaded via Python API: {TELCO_CSV}"
    except Exception as e:
        print(f"Python API failed: {e}")

    # Fallback to CLI
    try:
        subprocess.check_call(["kaggle", "datasets", "download", "-d", TELCO_KAGGLE_REF, "-p", "."]) 
        # unzip
        if os.path.exists(TELCO_ZIP):
            with zipfile.ZipFile(TELCO_ZIP, "r") as zip_ref:
                zip_ref.extractall(".")
            os.remove(TELCO_ZIP)
        if os.path.exists(TELCO_CSV):
            return True, f"Downloaded via CLI: {TELCO_CSV}"
    except Exception as e:
        print(f"Kaggle CLI failed: {e}")

    return False, "Both Kaggle methods failed. Provide KAGGLE_USERNAME and KAGGLE_KEY as env vars."
