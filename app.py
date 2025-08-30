# app.py
import os
import sys
import io
import json
import subprocess
import zipfile
from datetime import datetime
import warnings
# app.py
import os
import sys
import io
import json
import subprocess
import zipfile
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Build marker to help verify the deployed Space has reloaded this file
APP_BUILD_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
print(f"[APP_BUILD] Build ID: {APP_BUILD_ID}")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight

import shap

import gradio as gr
import joblib
import threading
import time
import builtins

# Runtime compatibility: print Gradio version and suggest upgrade when old
try:
    _gr_version = getattr(gr, "__version__", "0")
    print(f"[DEBUG] Gradio version detected: {_gr_version}")
    # Graceful suggestion for Spaces logs (non-fatal)
    try:
        from packaging import version as _pv
        if _pv.parse(_gr_version) < _pv.parse("4.44.1"):
            print("[WARN] Gradio older than 4.44.1 detected. Consider upgrading in the environment to match tested behavior.")
    except Exception:
        pass
except Exception:
    pass


# -------------------------------
# Constants / Defaultss
# -------------------------------
TELCO_KAGGLE_REF = "blastchar/telco-customer-churn"
TELCO_ZIP = "telco-customer-churn.zip"
TELCO_CSV = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

RANDOM_STATE = 42
N_FOLDS = 5

# -------------------------------
# Utility: Optional installs
# -------------------------------
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
                print("Attempting to install LightGBM...")
                # Try to install a pre-built wheel first
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm>=3.3.0"], 
                                        timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    import lightgbm  # noqa
                    available["lightgbm"] = True
                    print("LightGBM installed successfully!")
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    print("LightGBM installation failed or timed out, continuing without it...")
                    available["lightgbm"] = False
            except Exception:
                print("LightGBM installation failed, continuing without it...")
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

# Global availability cache used at runtime to avoid NameError
AVAILABLE = None

# -------------------------------
# Kaggle download
# -------------------------------
def kaggle_download_telco():
    """
    Try to download the Telco dataset via Kaggle Python API or CLI.
    Works with both Hugging Face secrets and local Kaggle CLI.
    """
    # Debug: Show environment variables for Hugging Face debugging
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Debug: Checking Kaggle credentials...")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] KAGGLE_USERNAME present: {'KAGGLE_USERNAME' in os.environ}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] KAGGLE_KEY present: {'KAGGLE_KEY' in os.environ}")
    
    # already present?
    if os.path.exists(TELCO_CSV):
        return True, f"[{datetime.now().strftime('%H:%M:%S')}] Found existing CSV: {TELCO_CSV}"

    # Try Python API first (works with Hugging Face secrets)
    try:
        # If secrets are provided as environment variables, ensure kaggle.json exists
        kaggle_user = os.environ.get("KAGGLE_USERNAME")
        kaggle_key = os.environ.get("KAGGLE_KEY")
        if kaggle_user and kaggle_key:
            kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
            os.makedirs(kaggle_dir, exist_ok=True)
            kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
            if not os.path.exists(kaggle_json):
                try:
                    with open(kaggle_json, "w") as fh:
                        json.dump({"username": kaggle_user, "key": kaggle_key}, fh)
                    # Set restrictive permissions where supported
                    try:
                        os.chmod(kaggle_json, 0o600)
                    except Exception:
                        pass
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Wrote kaggle.json to {kaggle_json}")
                except Exception as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to write kaggle.json: {e}")

        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Using Kaggle Python API...")
        api.dataset_download_files(TELCO_KAGGLE_REF, path=".", unzip=True)
        
        if os.path.exists(TELCO_CSV):
            return True, f"[{datetime.now().strftime('%H:%M:%S')}] Downloaded via Python API: {TELCO_CSV}"
        else:
            return False, f"[{datetime.now().strftime('%H:%M:%S')}] Python API download failed - CSV not found"
            
    except ImportError:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Kaggle Python API not available, attempting to install kaggle package...")
        # Try to install kaggle package and re-import
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402
            api = KaggleApi()
            api.authenticate()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Kaggle package installed and authenticated")
            api.dataset_download_files(TELCO_KAGGLE_REF, path=".", unzip=True)
            if os.path.exists(TELCO_CSV):
                return True, f"[{datetime.now().strftime('%H:%M:%S')}] Downloaded via Python API after installing kaggle: {TELCO_CSV}"
        except Exception:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to install or use kaggle Python package; falling back to CLI approach")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Python API failed: {str(e)}")
        # Check if it's a credentials issue
        if "No API found" in str(e) or "authentication" in str(e).lower():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Authentication failed - check KAGGLE_USERNAME and KAGGLE_KEY environment variables in Hugging Face Space secrets")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Current environment: KAGGLE_USERNAME={'KAGGLE_USERNAME' in os.environ}, KAGGLE_KEY={'KAGGLE_KEY' in os.environ}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Trying CLI as fallback...")

    # Fallback to CLI
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Trying Kaggle CLI...")
        cmd = ["kaggle", "datasets", "download", "-d", TELCO_KAGGLE_REF, "-p", "."]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        # unzip
        if os.path.exists(TELCO_ZIP):
            with zipfile.ZipFile(TELCO_ZIP, "r") as zip_ref:
                zip_ref.extractall(".")
            os.remove(TELCO_ZIP)
            
        if os.path.exists(TELCO_CSV):
            return True, f"[{datetime.now().strftime('%H:%M:%S')}] Downloaded via CLI: {TELCO_CSV}"
        else:
            return False, f"[{datetime.now().strftime('%H:%M:%S')}] CLI download failed - CSV not found after extraction"
            
    except subprocess.CalledProcessError:
        return False, f"[{datetime.now().strftime('%H:%M:%S')}] Kaggle CLI failed. Please ensure kaggle is installed and configured."
    except Exception as e:
        return False, f"[{datetime.now().strftime('%H:%M:%S')}] Download error: {str(e)}"
    
    # If both methods failed, provide clear error message
    return False, f"[{datetime.now().strftime('%H:%M:%S')}] Both Kaggle methods failed. Please check your Kaggle credentials in Hugging Face Space secrets (KAGGLE_USERNAME and KAGGLE_KEY)."

# -------------------------------
# Data loading & preprocessing
# -------------------------------
def load_telco_data(uploaded_file=None):
    """Load dataset. If uploaded_file (path-like) is provided, use it; otherwise fall back to the local Telco CSV or try Kaggle."""
    # If user uploaded a CSV via UI, prefer it
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded uploaded dataset: {df.shape}")
            df = engineer_features(df)
            return df
        except Exception as e:
            raise FileNotFoundError(f"Failed to read uploaded file: {e}")

    # Otherwise, prefer Kaggle as the default source on Hugging Face Spaces
    # This will attempt to use Kaggle API (with secrets KAGGLE_USERNAME/KAGGLE_KEY) or CLI.
    print(f"[{datetime.now().strftime('%H:%M:%S')}] No upload provided — attempting Kaggle download as default source")
    success, msg = kaggle_download_telco()
    if success:
        # Ensure we load the CSV from the most likely locations
        possible_paths = [
            os.path.abspath(TELCO_CSV),
            os.path.join(os.path.dirname(__file__), TELCO_CSV),
        ]
        found_path = None
        for p in possible_paths:
            if os.path.exists(p):
                found_path = p
                break
        if found_path:
            df = pd.read_csv(found_path)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded dataset from Kaggle download at: {found_path} shape={df.shape}")
            df = engineer_features(df)
            return df
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Kaggle reported success but CSV not found in expected locations: {possible_paths}. Message: {msg}")

    # Fallback: try to load CSV bundled with the repo (useful if network/download fails)
    possible_paths = [
        os.path.abspath(TELCO_CSV),
        os.path.join(os.path.dirname(__file__), TELCO_CSV),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded dataset from local repo path: {p} shape={df.shape}")
            df = engineer_features(df)
            return df

    # If we reach here, nothing worked. Don't raise here during UI build —
    # return an empty DataFrame and log a warning so the Space can still
    # start and surface a clear message to the user.
    try:
        warn_msg = msg
    except Exception:
        warn_msg = "(no kaggle message available)"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Dataset not found: {TELCO_CSV}. Kaggle attempt message: {warn_msg}")
    return pd.DataFrame()
    return pd.DataFrame()

def engineer_features(df):
    """
    Advanced feature engineering based on top Kaggle solutions.
    """
    df = df.copy()
    
    # Handle TotalCharges - convert to numeric and fill missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with MonthlyCharges (new customers)
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
    
    # Create ratio features
    df['MonthlyToTotalRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1e-8)
    df['ContractValue'] = df['MonthlyCharges'] * df['Contract'].map({'Month-to-month': 1, 'One year': 12, 'Two year': 24})
    
    # Create interaction features
    df['InternetService_Contract'] = df['InternetService'] + '_' + df['Contract']
    df['PaymentMethod_Contract'] = df['PaymentMethod'] + '_' + df['Contract']
    
    # Create tenure-based features
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72, 100], labels=['0-1y', '1-2y', '2-4y', '4-6y', '6y+'])
    df['IsNewCustomer'] = (df['tenure'] <= 1).astype(int)
    df['IsLongTermCustomer'] = (df['tenure'] >= 24).astype(int)
    
    # Create charge-based features
    df['HighSpender'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
    df['LowSpender'] = (df['MonthlyCharges'] < df['MonthlyCharges'].quantile(0.25)).astype(int)
    
    # Create service count features
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['TotalServices'] = df[service_cols].apply(lambda x: (x != 'No').sum(), axis=1)
    df['InternetServices'] = df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']].apply(
        lambda x: (x != 'No').sum(), axis=1)
    
    # Create customer value features
    df['CustomerValue'] = df['MonthlyCharges'] * df['tenure']
    df['ValuePerTenure'] = df['CustomerValue'] / (df['tenure'] + 1e-8)
    
    # Handle categorical variables with more sophisticated encoding
    df['Contract_Monthly'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['Contract_Yearly'] = (df['Contract'] == 'One year').astype(int)
    df['Contract_TwoYear'] = (df['Contract'] == 'Two year').astype(int)
    
    df['PaymentMethod_Electronic'] = df['PaymentMethod'].isin(['Electronic check', 'Mailed check']).astype(int)
    df['PaymentMethod_Auto'] = df['PaymentMethod'].isin(['Bank transfer (automatic)', 'Credit card (automatic)']).astype(int)
    
    # Create binary features for key services
    df['HasInternet'] = (df['InternetService'] != 'No').astype(int)
    df['HasPhone'] = (df['PhoneService'] == 'Yes').astype(int)
    df['HasMultipleLines'] = (df['MultipleLines'] == 'Yes').astype(int)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Feature engineering complete. New shape: {df.shape}")
    return df

def get_preprocessor(df):
    """
    Create advanced preprocessor with feature selection.
    """
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target and ID columns
    numeric_cols = [col for col in numeric_cols if col not in ['Churn', 'customerID']]
    categorical_cols = [col for col in categorical_cols if col not in ['Churn', 'customerID']]
    
    # Numeric pipeline with robust scaling
    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler())
    ])
    
    # Categorical pipeline with OHE
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Combine transformers - use column names as they will be preserved in the pipeline
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols)
        ],
        remainder="drop"
    )
    
    return pre, numeric_cols, categorical_cols

def split_xy(df):
    """
    Split data into features and target with stratified sampling.
    """
    # Map target
    if "Churn" not in df.columns:
        raise ValueError("Column 'Churn' not found in dataset.")
    y = df["Churn"].map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("Target 'Churn' contains unexpected values. Expected 'Yes'/'No'.")

    # Drop ID & target to form X
    drop_cols = [c for c in ["Churn", "customerID"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    
    # Use stratified split for better representation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Ensure X_train and X_test remain as DataFrames with column names
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    
    return X_train, X_test, y_train, y_test

# -------------------------------
# Advanced Modeling with Hyperparameter Tuning
# -------------------------------
# The full, modern `get_tuned_models_selected` implementation (with `fast` support)
# lives later in the file. The older duplicate was removed to avoid confusion.

def get_tuned_models(preprocessor, X_train, y_train):
    """
    Returns list of (name, pipeline) tuples with hyperparameter tuning.
    """
    models = []
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting hyperparameter tuning for all models...")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Training data shape: {X_train.shape}, Target distribution: {np.bincount(y_train)}")
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚖️ Class weights calculated: {class_weight_dict}")
    
    # Logistic Regression with tuning
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Tuning Logistic Regression...")
    lr_param_grid = {
        'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'clf__penalty': ['l1', 'l2'],
        'clf__solver': ['liblinear', 'saga'],
        'clf__class_weight': ['balanced', None]
    }
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 LR Grid: {len(lr_param_grid['clf__C'])} × {len(lr_param_grid['clf__penalty'])} × {len(lr_param_grid['clf__solver'])} × {len(lr_param_grid['clf__class_weight'])} = {len(lr_param_grid['clf__C']) * len(lr_param_grid['clf__penalty']) * len(lr_param_grid['clf__solver']) * len(lr_param_grid['clf__class_weight'])} combinations")

    lr_base = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    
    lr_tuned = GridSearchCV(
        lr_base, lr_param_grid, cv=3, scoring='roc_auc', 
        n_jobs=-1
    )
    lr_tuned.fit(X_train, y_train)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Logistic Regression tuned! Best CV AUC: {lr_tuned.best_score_:.4f}")
    models.append(("LogisticRegression", lr_tuned.best_estimator_))

    # Random Forest with tuning
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🌲 Tuning Random Forest...")
    rf_param_grid = {
        'clf__n_estimators': [300, 500, 700],
        'clf__max_depth': [8, 12, 16, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__class_weight': ['balanced', 'balanced_subsample']
    }
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 RF Grid: {len(rf_param_grid['clf__n_estimators'])} × {len(rf_param_grid['clf__max_depth'])} × {len(rf_param_grid['clf__min_samples_split'])} × {len(rf_param_grid['clf__min_samples_leaf'])} × {len(rf_param_grid['clf__class_weight'])} = {len(rf_param_grid['clf__n_estimators']) * len(rf_param_grid['clf__max_depth']) * len(rf_param_grid['clf__min_samples_split']) * len(rf_param_grid['clf__min_samples_leaf']) * len(rf_param_grid['clf__class_weight'])} combinations")

    rf_base = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_jobs=-1))
    ])

    rf_tuned = GridSearchCV(
        rf_base, rf_param_grid, cv=3, scoring='roc_auc', 
        n_jobs=-1
    )
    rf_tuned.fit(X_train, y_train)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Random Forest tuned! Best CV AUC: {rf_tuned.best_score_:.4f}")
    models.append(("RandomForest", rf_tuned.best_estimator_))

    # Gradient Boosting
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📈 Tuning Gradient Boosting...")
    gb_param_grid = {
        'clf__n_estimators': [200, 400, 600],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__max_depth': [3, 5, 7],
        'clf__subsample': [0.8, 0.9, 1.0]
    }
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 GB Grid: {len(gb_param_grid['clf__n_estimators'])} × {len(gb_param_grid['clf__learning_rate'])} × {len(gb_param_grid['clf__max_depth'])} × {len(gb_param_grid['clf__subsample'])} = {len(gb_param_grid['clf__n_estimators']) * len(gb_param_grid['clf__learning_rate']) * len(gb_param_grid['clf__max_depth']) * len(gb_param_grid['clf__subsample'])} combinations")

    gb_base = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", GradientBoostingClassifier())
    ])

    gb_tuned = GridSearchCV(
        gb_base, gb_param_grid, cv=3, scoring='roc_auc', 
        n_jobs=-1
    )
    gb_tuned.fit(X_train, y_train)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Gradient Boosting tuned! Best CV AUC: {gb_tuned.best_score_:.4f}")
    models.append(("GradientBoosting", gb_tuned.best_estimator_))

    # CatBoost with tuning
    try:
        from catboost import CatBoostClassifier
        cb_param_grid = {
            'clf__iterations': [400, 600, 800],
            'clf__learning_rate': [0.03, 0.05, 0.1],
            'clf__depth': [4, 6, 8],
            'clf__l2_leaf_reg': [1, 3, 5, 7]
        }
        
        cb_base = Pipeline(steps=[
            ("pre", preprocessor),
            ("clf", CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                verbose=False
            ))
        ])
        
        cb_tuned = GridSearchCV(
            cb_base, cb_param_grid, cv=3, scoring='roc_auc', 
            n_jobs=-1
        )
        cb_tuned.fit(X_train, y_train)
        models.append(("CatBoost", cb_tuned.best_estimator_))
    except Exception:
        pass

    # XGBoost with tuning
    try:
        from xgboost import XGBClassifier
        xgb_param_grid = {
            'clf__n_estimators': [400, 600, 800],
            'clf__max_depth': [3, 5, 7],
            'clf__learning_rate': [0.01, 0.05, 0.1],
            'clf__subsample': [0.8, 0.9, 1.0],
            'clf__colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        xgb_base = Pipeline(steps=[
            ("pre", preprocessor),
            ("clf", XGBClassifier(
                scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                n_jobs=-1,
                tree_method="hist",
                eval_metric="logloss"
            ))
        ])
        
        xgb_tuned = GridSearchCV(
            xgb_base, xgb_param_grid, cv=3, scoring='roc_auc', 
            n_jobs=-1
        )
        xgb_tuned.fit(X_train, y_train)
        models.append(("XGBoost", xgb_tuned.best_estimator_))
    except Exception:
        pass

    # LightGBM with tuning
    try:
        from lightgbm import LGBMClassifier
        lgbm_param_grid = {
            'clf__n_estimators': [400, 600, 800],
            'clf__max_depth': [3, 5, 7, -1],
            'clf__learning_rate': [0.01, 0.05, 0.1],
            'clf__subsample': [0.8, 0.9, 1.0],
            'clf__colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        lgbm_base = Pipeline(steps=[
            ("pre", preprocessor),
            ("clf", LGBMClassifier(
                class_weight='balanced',
                n_jobs=-1,
                verbose=-1
            ))
        ])
        
        lgbm_tuned = GridSearchCV(
            lgbm_base, lgbm_param_grid, cv=3, scoring='roc_auc', 
            n_jobs=-1
        )
        lgbm_tuned.fit(X_train, y_train)
        models.append(("LightGBM", lgbm_tuned.best_estimator_))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] LightGBM model added successfully")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] LightGBM not available: {e}")
        pass

    return models


def _get_final_estimator(obj):
    """Return the final estimator for common wrappers (Pipeline) or the object itself.

    This avoids indexing like obj[-1] which fails for non-subscriptable estimators
    such as VotingClassifier.
    """
    try:
        # sklearn Pipeline exposes `steps`
        if hasattr(obj, 'steps') and isinstance(obj.steps, (list, tuple)) and len(obj.steps) > 0:
            return obj.steps[-1][1]
    except Exception:
        pass
    # Fallback: return the object itself
    return obj

def create_ensemble_model(models, preprocessor, X_train, y_train):
    """
    Create a voting ensemble from the best models.
    """
    # Get the top 3 models by CV performance
    cv_scores = []
    for name, model in models:
        cv_auc = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=3, n_jobs=-1)
        cv_scores.append((name, model, cv_auc.mean()))
    
    cv_scores.sort(key=lambda x: x[2], reverse=True)
    top_models = cv_scores[:3]
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model, _ in top_models],
        voting='soft'
    )
    
    # Fit ensemble directly (don't wrap in pipeline to avoid ColumnTransformer issues)
    ensemble.fit(X_train, y_train)
    
    models.append(("Ensemble", ensemble))
    return models

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """Evaluate a list of (name, estimator) pairs and return scorecard, plots and trained objects.

    This implementation avoids indexing into pipeline-like objects (e.g. pipe[-1]) and
    uses _get_final_estimator to extract the underlying estimator where needed.
    """
    results = []
    trained = {}
    fprs, tprs, aucs = [], [], []
    names_for_roc = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # CV first, pick best by mean CV AUC
    cv_auc_by_model = []
    for name, pipe in models:
        try:
            cv_auc = cross_val_score(pipe, X_train, y_train, scoring="roc_auc", cv=skf, n_jobs=-1)
            cv_auc_by_model.append((name, cv_auc.mean(), cv_auc.std()))
        except Exception:
            cv_auc_by_model.append((name, np.nan, np.nan))

    # Sort by CV AUC
    cv_auc_by_model.sort(key=lambda t: (t[1] if not np.isnan(t[1]) else -1.0), reverse=True)

    # Fit and evaluate all models
    for name, pipe in models:
        try:
            pipe.fit(X_train, y_train)
        except Exception:
            # If a model fails to fit, skip it but keep moving
            continue
        trained[name] = pipe

        # Resolve final estimator and try to get probabilities
        final_est = _get_final_estimator(pipe)

        y_proba = None
        # Prefer top-level predict_proba
        if hasattr(pipe, "predict_proba"):
            try:
                y_proba = pipe.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None

        # Fallbacks
        if y_proba is None and hasattr(final_est, "predict_proba"):
            try:
                y_proba = final_est.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None

        if y_proba is None:
            # Try decision_function then scale to [0,1]
            if hasattr(pipe, "decision_function"):
                raw = pipe.decision_function(X_test)
                y_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
            elif hasattr(final_est, "decision_function"):
                raw = final_est.decision_function(X_test)
                y_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
            else:
                y_proba = np.zeros_like(y_test, dtype=float)

        y_pred = (y_proba >= 0.5).astype(int)

        # Calculate comprehensive metrics
        try:
            test_auc = roc_auc_score(y_test, y_proba)
        except Exception:
            test_auc = float('nan')

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        cv_mean = next((m for n, m, s in cv_auc_by_model if n == name), np.nan)
        cv_std = next((s for n, m, s in cv_auc_by_model if n == name), np.nan)

        results.append({
            "Model": name,
            "CV AUC Mean": round(cv_mean, 4) if not np.isnan(cv_mean) else np.nan,
            "CV AUC Std": round(cv_std, 4) if not np.isnan(cv_std) else np.nan,
            "Test AUC": round(test_auc, 4) if not np.isnan(test_auc) else np.nan,
            "Test Accuracy": round(acc, 4),
            "Test F1": round(f1, 4),
            "Test Precision": round(precision, 4),
            "Test Recall": round(recall, 4),
        })

        # ROC components
        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fprs.append(fpr); tprs.append(tpr)
            aucs.append(auc(fpr, tpr))
            names_for_roc.append(name)
        except Exception:
            pass

    # Sort by test accuracy for final ranking
    scorecard_df = pd.DataFrame(results)
    if not scorecard_df.empty:
        scorecard_df = scorecard_df.sort_values("Test Accuracy", ascending=False).reset_index(drop=True)

    # ROC plot (all models)
    roc_path = f"roc_all_{datetime.now().strftime('%H%M%S')}.png"
    try:
        plt.figure(figsize=(10, 8))
        for fpr, tpr, name in zip(fprs, tprs, names_for_roc):
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (Test)")
        plt.legend(loc="lower right", frameon=True)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(roc_path, dpi=140, bbox_inches='tight')
        plt.close()
    except Exception:
        roc_path = None

    # Confusion matrix (best by test accuracy)
    best_name = None
    conf_path = None
    if not scorecard_df.empty:
        best_row = scorecard_df.iloc[0]
        best_name = best_row["Model"]
        best_pipe = trained.get(best_name)
        if best_pipe is not None:
            best_final = _get_final_estimator(best_pipe)
            best_proba = None
            if hasattr(best_pipe, "predict_proba"):
                try:
                    best_proba = best_pipe.predict_proba(X_test)[:, 1]
                except Exception:
                    best_proba = None
            if best_proba is None and hasattr(best_final, "predict_proba"):
                try:
                    best_proba = best_final.predict_proba(X_test)[:, 1]
                except Exception:
                    best_proba = None
            if best_proba is None:
                if hasattr(best_pipe, "decision_function"):
                    raw = best_pipe.decision_function(X_test)
                    best_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
                elif hasattr(best_final, "decision_function"):
                    raw = best_final.decision_function(X_test)
                    best_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
                else:
                    best_proba = np.zeros_like(y_test, dtype=float)

            best_pred = (best_proba >= 0.5).astype(int)
            conf = confusion_matrix(y_test, best_pred)
            conf_path = f"confusion_best_{datetime.now().strftime('%H%M%S')}.png"
            try:
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title(f"Confusion Matrix (Best: {best_name})")
                plt.tight_layout()
                plt.savefig(conf_path, dpi=140, bbox_inches='tight')
                plt.close()
            except Exception:
                conf_path = None

    return scorecard_df, roc_path, conf_path, best_name, trained

# -------------------------------
# SHAP Explainability
# -------------------------------
def compute_shap_images(trained_pipe, preprocessor, X_sample, feature_names_out, shap_sample=100):
    """Compute SHAP summary and bar plots for a trained pipeline/estimator.

    Returns list of file paths written. On any failure returns an empty list.
    """
    paths = []
    try:
        import shap
    except Exception:
        print("shap library not available; skipping SHAP plots")
        return []

    try:
        # Transform X through preprocessor so SHAP sees the exact model input
        X_trans = preprocessor.transform(X_sample)
        model = _get_final_estimator(trained_pipe)

        n_sample = min(shap_sample, X_trans.shape[0])

        # Choose explainer based on estimator characteristics
        explainer = None
        try:
            model_name = type(model).__name__.lower()
            if hasattr(model, "get_booster") or hasattr(model, "feature_importances_") or 'xgb' in model_name or 'catboost' in model_name or 'lgbm' in model_name:
                explainer = shap.TreeExplainer(model)
            else:
                # KernelExplainer is slow; use a small background sample
                background = X_trans[: max(20, n_sample)]
                explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x), background)
        except Exception:
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                print("Failed to construct SHAP explainer for model")
                return []

        shap_values = explainer.shap_values(X_trans[:n_sample])

        # Handle multi-output case (classification)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]

        # Summary plot
        summary_path = f"shap_summary_{datetime.now().strftime('%H%M%S')}.png"
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_trans[:n_sample], feature_names=feature_names_out, show=False)
        plt.tight_layout()
        plt.savefig(summary_path, dpi=130, bbox_inches='tight')
        plt.close()
        paths.append(summary_path)

        # Bar plot
        bar_path = f"shap_bar_{datetime.now().strftime('%H%M%S')}.png"
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_trans[:n_sample], feature_names=feature_names_out, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(bar_path, dpi=130, bbox_inches='tight')
        plt.close()
        paths.append(bar_path)

    except Exception as exc:
        print(f"SHAP computation failed: {exc}")
        return []

    return paths

    if "Churn" not in df.columns:
        raise ValueError("Column 'Churn' not found in dataset.")
    y = df["Churn"].map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("Target 'Churn' contains unexpected values. Expected 'Yes'/'No'.")

    # Drop ID & target to form X
    drop_cols = [c for c in ["Churn", "customerID"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    
    # Use stratified split for better representation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Ensure X_train and X_test remain as DataFrames with column names
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    
    return X_train, X_test, y_train, y_test

# -------------------------------
# Advanced Modeling with Hyperparameter Tuning
# -------------------------------
def get_tuned_models_selected(preprocessor, X_train, y_train, selected_models, fast=False):
    """
    Returns list of (name, pipeline) tuples with hyperparameter tuning for SELECTED models only.
    """
    models = []
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting hyperparameter tuning for selected models: {selected_models}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Training data shape: {X_train.shape}, Target distribution: {np.bincount(y_train)}")
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚖️ Class weights calculated: {class_weight_dict}")
    
    # Only train the selected models
    if "Logistic Regression" in selected_models:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Tuning Logistic Regression...")
        if fast:
            lr_param_grid = {'clf__C': [0.1, 1.0], 'clf__penalty': ['l2'], 'clf__solver': ['liblinear'], 'clf__class_weight': ['balanced', None]}
        else:
            lr_param_grid = {
                'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'clf__penalty': ['l1', 'l2'],
                'clf__solver': ['liblinear', 'saga'],
                'clf__class_weight': ['balanced', None]
            }
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 LR Grid: {len(lr_param_grid['clf__C'])} × {len(lr_param_grid['clf__penalty'])} × {len(lr_param_grid['clf__solver'])} × {len(lr_param_grid['clf__class_weight'])} = {len(lr_param_grid['clf__C']) * len(lr_param_grid['clf__penalty']) * len(lr_param_grid['clf__solver']) * len(lr_param_grid['clf__class_weight'])} combinations")
        
        lr_base = Pipeline(steps=[
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=2000))
        ])
        
        lr_tuned = GridSearchCV(
            lr_base, lr_param_grid, cv=3, scoring='roc_auc', 
            n_jobs=-1
        )
        lr_tuned.fit(X_train, y_train)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Logistic Regression tuned! Best CV AUC: {lr_tuned.best_score_:.4f}")
        models.append(("LogisticRegression", lr_tuned.best_estimator_))

    if "Random Forest" in selected_models:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🌲 Tuning Random Forest...")
        if fast:
            rf_param_grid = {'clf__n_estimators': [100, 200], 'clf__max_depth': [8, None], 'clf__min_samples_split': [2], 'clf__min_samples_leaf': [1], 'clf__class_weight': ['balanced']}
        else:
            rf_param_grid = {
                'clf__n_estimators': [300, 500, 700],
                'clf__max_depth': [8, 12, 16, None],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4],
                'clf__class_weight': ['balanced', 'balanced_subsample']
            }
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 RF Grid: {len(rf_param_grid['clf__n_estimators'])} × {len(rf_param_grid['clf__max_depth'])} × {len(rf_param_grid['clf__min_samples_split'])} × {len(rf_param_grid['clf__min_samples_leaf'])} × {len(rf_param_grid['clf__class_weight'])} = {len(rf_param_grid['clf__n_estimators']) * len(rf_param_grid['clf__max_depth']) * len(rf_param_grid['clf__min_samples_split']) * len(rf_param_grid['clf__min_samples_leaf']) * len(rf_param_grid['clf__class_weight'])} combinations")
        
        rf_base = Pipeline(steps=[
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(n_jobs=-1))
        ])
        
        rf_tuned = GridSearchCV(
            rf_base, rf_param_grid, cv=3, scoring='roc_auc', 
            n_jobs=-1
        )
        rf_tuned.fit(X_train, y_train)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Random Forest tuned! Best CV AUC: {rf_tuned.best_score_:.4f}")
        models.append(("RandomForest", rf_tuned.best_estimator_))

    if "Gradient Boosting" in selected_models:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📈 Tuning Gradient Boosting...")
        if fast:
            gb_param_grid = {'clf__n_estimators': [100, 200], 'clf__learning_rate': [0.05, 0.1], 'clf__max_depth': [3], 'clf__subsample': [0.9]}
        else:
            gb_param_grid = {
                'clf__n_estimators': [200, 400, 600],
                'clf__learning_rate': [0.01, 0.05, 0.1],
                'clf__max_depth': [3, 5, 7],
                'clf__subsample': [0.8, 0.9, 1.0]
            }
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 GB Grid: {len(gb_param_grid['clf__n_estimators'])} × {len(gb_param_grid['clf__learning_rate'])} × {len(gb_param_grid['clf__max_depth'])} × {len(gb_param_grid['clf__subsample'])} = {len(gb_param_grid['clf__n_estimators']) * len(gb_param_grid['clf__learning_rate']) * len(gb_param_grid['clf__max_depth']) * len(gb_param_grid['clf__subsample'])} combinations")
        
        gb_base = Pipeline(steps=[
            ("pre", preprocessor),
            ("clf", GradientBoostingClassifier())
        ])
        
        gb_tuned = GridSearchCV(
            gb_base, gb_param_grid, cv=3, scoring='roc_auc', 
            n_jobs=-1
        )
        gb_tuned.fit(X_train, y_train)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Gradient Boosting tuned! Best CV AUC: {gb_tuned.best_score_:.4f}")
        models.append(("GradientBoosting", gb_tuned.best_estimator_))

    if "XGBoost" in selected_models:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Tuning XGBoost...")
        if fast:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Skipping XGBoost in Fast mode to avoid heavy installs/training.")
        else:
            xgb_param_grid = {
                'clf__n_estimators': [400, 600, 800],
                'clf__max_depth': [3, 5, 7],
                'clf__learning_rate': [0.01, 0.05, 0.1],
                'clf__subsample': [0.8, 0.9, 1.0],
                'clf__colsample_bytree': [0.7, 0.8, 0.9]
            }
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 XGB Grid: {len(xgb_param_grid['clf__n_estimators'])} × {len(xgb_param_grid['clf__max_depth'])} × {len(xgb_param_grid['clf__learning_rate'])} × {len(xgb_param_grid['clf__subsample'])} × {len(xgb_param_grid['clf__colsample_bytree'])} = {len(xgb_param_grid['clf__n_estimators']) * len(xgb_param_grid['clf__max_depth']) * len(xgb_param_grid['clf__learning_rate']) * len(xgb_param_grid['clf__subsample']) * len(xgb_param_grid['clf__colsample_bytree'])} combinations")
            try:
                from xgboost import XGBClassifier
                xgb_base = Pipeline(steps=[
                    ("pre", preprocessor),
                    ("clf", XGBClassifier(
                        scale_pos_weight=len(y_train[y_train==0]) / max(1, len(y_train[y_train==1])),
                        n_jobs=-1,
                        tree_method="hist",
                        eval_metric="logloss"
                    ))
                ])
                xgb_tuned = GridSearchCV(
                    xgb_base, xgb_param_grid, cv=3, scoring='roc_auc', 
                    n_jobs=-1
                )
                xgb_tuned.fit(X_train, y_train)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ XGBoost tuned! Best CV AUC: {xgb_tuned.best_score_:.4f}")
                models.append(("XGBoost", xgb_tuned.best_estimator_))
            except Exception as ex:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] XGBoost not available or failed to train: {ex}")

    if "CatBoost" in selected_models:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🐱 Tuning CatBoost...")
        if fast:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Skipping CatBoost in Fast mode to avoid heavy installs/training.")
        else:
            cb_param_grid = {
                'clf__iterations': [300, 500, 700],
                'clf__learning_rate': [0.01, 0.05, 0.1],
                'clf__depth': [4, 6, 8],
                'clf__l2_leaf_reg': [1, 3, 5, 7]
            }
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 CB Grid: {len(cb_param_grid['clf__iterations'])} × {len(cb_param_grid['clf__learning_rate'])} × {len(cb_param_grid['clf__depth'])} × {len(cb_param_grid['clf__l2_leaf_reg'])} = {len(cb_param_grid['clf__iterations']) * len(cb_param_grid['clf__learning_rate']) * len(cb_param_grid['clf__depth']) * len(cb_param_grid['clf__l2_leaf_reg'])} combinations")
        
        if not fast:
            try:
                from catboost import CatBoostClassifier
                cb_base = Pipeline(steps=[
                    ("pre", preprocessor),
                    ("clf", CatBoostClassifier(
                        loss_function="Logloss",
                        eval_metric="AUC",
                        verbose=False
                    ))
                ])
                cb_tuned = GridSearchCV(
                    cb_base, cb_param_grid, cv=3, scoring='roc_auc', 
                    n_jobs=-1
                )
                cb_tuned.fit(X_train, y_train)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ CatBoost tuned! Best CV AUC: {cb_tuned.best_score_:.4f}")
                models.append(("CatBoost", cb_tuned.best_estimator_))
            except Exception as ex:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] CatBoost not available or failed to train: {ex}")

    return models

def get_tuned_models(preprocessor, X_train, y_train):
    """
    Returns list of (name, pipeline) tuples with hyperparameter tuning.
    """
    models = []
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting hyperparameter tuning for all models...")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Training data shape: {X_train.shape}, Target distribution: {np.bincount(y_train)}")
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚖️ Class weights calculated: {class_weight_dict}")
    
    # Logistic Regression with tuning
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Tuning Logistic Regression...")
    lr_param_grid = {
        'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'clf__penalty': ['l1', 'l2'],
        'clf__solver': ['liblinear', 'saga'],
        'clf__class_weight': ['balanced', None]
    }
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 LR Grid: {len(lr_param_grid['clf__C'])} × {len(lr_param_grid['clf__penalty'])} × {len(lr_param_grid['clf__solver'])} × {len(lr_param_grid['clf__class_weight'])} = {len(lr_param_grid['clf__C']) * len(lr_param_grid['clf__penalty']) * len(lr_param_grid['clf__solver']) * len(lr_param_grid['clf__class_weight'])} combinations")
    
    lr_base = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    
    lr_tuned = GridSearchCV(
        lr_base, lr_param_grid, cv=3, scoring='roc_auc', 
        n_jobs=-1
    )
    lr_tuned.fit(X_train, y_train)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Logistic Regression tuned! Best CV AUC: {lr_tuned.best_score_:.4f}")
    models.append(("LogisticRegression", lr_tuned.best_estimator_))

    # Random Forest with tuning
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🌲 Tuning Random Forest...")
    rf_param_grid = {
        'clf__n_estimators': [300, 500, 700],
        'clf__max_depth': [8, 12, 16, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__class_weight': ['balanced', 'balanced_subsample']
    }
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 RF Grid: {len(rf_param_grid['clf__n_estimators'])} × {len(rf_param_grid['clf__max_depth'])} × {len(rf_param_grid['clf__min_samples_split'])} × {len(rf_param_grid['clf__min_samples_leaf'])} × {len(rf_param_grid['clf__class_weight'])} = {len(rf_param_grid['clf__n_estimators']) * len(rf_param_grid['clf__max_depth']) * len(rf_param_grid['clf__min_samples_split']) * len(rf_param_grid['clf__min_samples_leaf']) * len(rf_param_grid['clf__class_weight'])} combinations")
    
    rf_base = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_jobs=-1))
    ])
    
    rf_tuned = GridSearchCV(
        rf_base, rf_param_grid, cv=3, scoring='roc_auc', 
        n_jobs=-1
    )
    rf_tuned.fit(X_train, y_train)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Random Forest tuned! Best CV AUC: {rf_tuned.best_score_:.4f}")
    models.append(("RandomForest", rf_tuned.best_estimator_))

    # Gradient Boosting
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📈 Tuning Gradient Boosting...")
    gb_param_grid = {
        'clf__n_estimators': [200, 400, 600],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__max_depth': [3, 5, 7],
        'clf__subsample': [0.8, 0.9, 1.0]
    }
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 GB Grid: {len(gb_param_grid['clf__n_estimators'])} × {len(gb_param_grid['clf__learning_rate'])} × {len(gb_param_grid['clf__max_depth'])} × {len(gb_param_grid['clf__subsample'])} = {len(gb_param_grid['clf__n_estimators']) * len(gb_param_grid['clf__learning_rate']) * len(gb_param_grid['clf__max_depth']) * len(gb_param_grid['clf__subsample'])} combinations")
    
    gb_base = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", GradientBoostingClassifier())
    ])
    
    gb_tuned = GridSearchCV(
        gb_base, gb_param_grid, cv=3, scoring='roc_auc', 
        n_jobs=-1
    )
    gb_tuned.fit(X_train, y_train)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Gradient Boosting tuned! Best CV AUC: {gb_tuned.best_score_:.4f}")
    models.append(("GradientBoosting", gb_tuned.best_estimator_))

    # CatBoost with tuning
    try:
        from catboost import CatBoostClassifier
        cb_param_grid = {
            'clf__iterations': [400, 600, 800],
            'clf__learning_rate': [0.03, 0.05, 0.1],
            'clf__depth': [4, 6, 8],
            'clf__l2_leaf_reg': [1, 3, 5, 7]
        }
        
        cb_base = Pipeline(steps=[
            ("pre", preprocessor),
            ("clf", CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                verbose=False
            ))
        ])
        
        cb_tuned = GridSearchCV(
            cb_base, cb_param_grid, cv=3, scoring='roc_auc', 
            n_jobs=-1
        )
        cb_tuned.fit(X_train, y_train)
        models.append(("CatBoost", cb_tuned.best_estimator_))
    except Exception:
        pass

    # XGBoost with tuning
    try:
        from xgboost import XGBClassifier
        xgb_param_grid = {
            'clf__n_estimators': [400, 600, 800],
            'clf__max_depth': [3, 5, 7],
            'clf__learning_rate': [0.01, 0.05, 0.1],
            'clf__subsample': [0.8, 0.9, 1.0],
            'clf__colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        xgb_base = Pipeline(steps=[
            ("pre", preprocessor),
            ("clf", XGBClassifier(
                scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                n_jobs=-1,
                tree_method="hist",
                eval_metric="logloss"
            ))
        ])
        
        xgb_tuned = GridSearchCV(
            xgb_base, xgb_param_grid, cv=3, scoring='roc_auc', 
            n_jobs=-1
        )
        xgb_tuned.fit(X_train, y_train)
        models.append(("XGBoost", xgb_tuned.best_estimator_))
    except Exception:
        pass

    # LightGBM with tuning
    try:
        from lightgbm import LGBMClassifier
        lgbm_param_grid = {
            'clf__n_estimators': [400, 600, 800],
            'clf__max_depth': [3, 5, 7, -1],
            'clf__learning_rate': [0.01, 0.05, 0.1],
            'clf__subsample': [0.8, 0.9, 1.0],
            'clf__colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        lgbm_base = Pipeline(steps=[
            ("pre", preprocessor),
            ("clf", LGBMClassifier(
                class_weight='balanced',
                n_jobs=-1,
                verbose=-1
            ))
        ])
        
        lgbm_tuned = GridSearchCV(
            lgbm_base, lgbm_param_grid, cv=3, scoring='roc_auc', 
            n_jobs=-1
        )
        lgbm_tuned.fit(X_train, y_train)
        models.append(("LightGBM", lgbm_tuned.best_estimator_))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] LightGBM model added successfully")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] LightGBM not available: {e}")
        pass

    return models

def create_ensemble_model(models, preprocessor, X_train, y_train):
    """
    Create a voting ensemble from the best models.
    """
    # Get the top 3 models by CV performance
    cv_scores = []
    for name, model in models:
        cv_auc = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=3, n_jobs=-1)
        cv_scores.append((name, model, cv_auc.mean()))
    
    cv_scores.sort(key=lambda x: x[2], reverse=True)
    top_models = cv_scores[:3]
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model, _ in top_models],
        voting='soft'
    )
    
    # Fit ensemble directly (don't wrap in pipeline to avoid ColumnTransformer issues)
    ensemble.fit(X_train, y_train)
    
    models.append(("Ensemble", ensemble))
    return models

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Advanced evaluation with multiple metrics and ensemble creation.
    """
    results = []
    trained = {}
    fprs, tprs, aucs = [], [], []
    names_for_roc = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # CV first, pick best by mean CV AUC
    cv_auc_by_model = []
    for name, pipe in models:
        cv_auc = cross_val_score(pipe, X_train, y_train, scoring="roc_auc", cv=skf, n_jobs=-1)
        cv_auc_by_model.append((name, cv_auc.mean(), cv_auc.std()))

    # Sort by CV AUC
    cv_auc_by_model.sort(key=lambda t: t[1], reverse=True)
    
    # Fit all models
    for name, pipe in models:
        pipe.fit(X_train, y_train)
        trained[name] = pipe
        # Get final estimator safely (handles Pipeline, VotingClassifier, etc.)
        final_est = _get_final_estimator(pipe)
        y_proba = None
        # Prefer pipeline-level predict_proba if available
        if hasattr(pipe, "predict_proba"):
            try:
                y_proba = pipe.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None
        # Fall back to final estimator's predict_proba
        if y_proba is None and hasattr(final_est, "predict_proba"):
            try:
                y_proba = final_est.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None
        # Last resort: use decision_function or zeros
        if y_proba is None:
            if hasattr(pipe, "decision_function"):
                raw = pipe.decision_function(X_test)
                y_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
            elif hasattr(final_est, "decision_function"):
                raw = final_est.decision_function(X_test)
                y_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
            else:
                y_proba = np.zeros_like(y_test, dtype=float)

        y_pred = (y_proba >= 0.5).astype(int)

        # Calculate comprehensive metrics
        test_auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        cv_mean = next((m for n, m, s in cv_auc_by_model if n == name), np.nan)
        cv_std = next((s for n, m, s in cv_auc_by_model if n == name), np.nan)

        results.append({
            "Model": name,
            "CV AUC Mean": round(cv_mean, 4),
            "CV AUC Std": round(cv_std, 4),
            "Test AUC": round(test_auc, 4),
            "Test Accuracy": round(acc, 4),
            "Test F1": round(f1, 4),
            "Test Precision": round(precision, 4),
            "Test Recall": round(recall, 4),
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fprs.append(fpr); tprs.append(tpr)
        aucs.append(auc(fpr, tpr))
        names_for_roc.append(name)

    # Sort by test accuracy for final ranking
    scorecard_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)

    # ROC plot (all models) - use seaborn styling and clearer legend
    roc_path = f"roc_all_{datetime.now().strftime('%H%M%S')}.png"
    plt.figure(figsize=(10, 8))
    for fpr, tpr, name in zip(fprs, tprs, names_for_roc):
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Test)")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=140, bbox_inches='tight')
    plt.close()

    # Confusion matrix (best by test accuracy)
    best_row = scorecard_df.iloc[0]
    best_name = best_row["Model"]
    best_pipe = trained[best_name]
    
    best_final = _get_final_estimator(best_pipe)
    if hasattr(best_pipe, "predict_proba"):
        try:
            best_proba = best_pipe.predict_proba(X_test)[:, 1]
        except Exception:
            if hasattr(best_final, "predict_proba"):
                best_proba = best_final.predict_proba(X_test)[:, 1]
            else:
                best_proba = np.zeros_like(y_test, dtype=float)
    elif hasattr(best_final, "predict_proba"):
        best_proba = best_final.predict_proba(X_test)[:, 1]
    else:
        if hasattr(best_pipe, "decision_function"):
            raw = best_pipe.decision_function(X_test)
            best_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        elif hasattr(best_final, "decision_function"):
            raw = best_final.decision_function(X_test)
            best_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        else:
            best_proba = np.zeros_like(y_test, dtype=float)
    
    best_pred = (best_proba >= 0.5).astype(int)

    conf = confusion_matrix(y_test, best_pred)
    conf_path = f"confusion_best_{datetime.now().strftime('%H%M%S')}.png"
    # Use seaborn heatmap for a nicer confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f"Confusion Matrix (Best: {best_name})")
    plt.tight_layout()
    plt.savefig(conf_path, dpi=140, bbox_inches='tight')
    plt.close()

    return scorecard_df, roc_path, conf_path, best_name, trained

# -------------------------------
# Main processing function
# -------------------------------
def train_selected_models_only(selected_models, allow_install=False, fast=False, uploaded_file=None):
    """Train only the selected models based on UI selections. Returns a result dict.

    - fast: when True use compact grids and smaller SHAP samples to run on resource-limited hosts.
    - uploaded_file: optional path-like or file object for a user-supplied CSV; forwarded to data loader.
    """
    try:
        # Ensure optional libraries (may attempt install if allow_install)
        available = ensure_optional_libs(allow_install)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Available libraries: {available}")

        # Load and prepare data
        df = load_telco_data(uploaded_file=uploaded_file)
        preprocessor, numeric_cols, categorical_cols = get_preprocessor(df)

        X_train, X_test, y_train, y_test = split_xy(df)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Data split. Train: {X_train.shape}, Test: {X_test.shape}")

        # Train only the selected models (fast flag forwarded)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting training for selected models: {selected_models} (fast={fast})")
        models = get_tuned_models_selected(preprocessor, X_train, y_train, selected_models, fast=fast)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Created {len(models)} tuned models")

        # Optionally create voting ensemble if several models present
        if len(models) > 1:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏆 Creating ensemble from selected models...")
            models = create_ensemble_model(models, preprocessor, X_train, y_train)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Ensemble model added! Total models: {len(models)}")

        # Preprocess once for evaluation
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Evaluate models
        scorecard_df, roc_path, conf_path, best_name, trained = evaluate_models(
            X_train, y_train, X_test, y_test, models
        )

        feature_names_out = preprocessor.get_feature_names_out()

        # Persist trained models for inspection / reuse
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_paths = {}
        try:
            for name, pipe in trained.items():
                safe_name = name.replace(" ", "_")
                path = os.path.join(model_dir, f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
                joblib.dump(pipe, path)
                model_paths[name] = path
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Persisted models to {model_dir}")
        except Exception as e:
            print(f"Failed to persist models: {e}")

        return {
            "success": True,
            "scorecard": scorecard_df,
            "roc_path": roc_path,
            "conf_path": conf_path,
            "best_name": best_name,
            "trained": trained,
            "model_paths": model_paths,
            "feature_names": feature_names_out,
            "X_sample": X_test[:100],
            "preprocessor": preprocessor
        }

    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error during selected-model training: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# Compatibility wrapper: some deployed environments may have an older
# version of this function without the `uploaded_file` parameter.
def _call_train(selected_models, allow_install=True, fast=False, uploaded_file=None):
    try:
        return train_selected_models_only(selected_models, allow_install=allow_install, fast=fast, uploaded_file=uploaded_file)
    except TypeError:
        # Fall back to older signature
        return train_selected_models_only(selected_models, allow_install=allow_install, fast=fast)


# Resilient alias: rebind the public name to a wrapper that tolerates both
# the old and new signatures. This helps when a deployed runtime still
# references a different implementation variant.
_train_impl = train_selected_models_only

def train_selected_models_only(selected_models, allow_install=False, fast=False, uploaded_file=None):
    try:
        return _train_impl(selected_models, allow_install=allow_install, fast=fast, uploaded_file=uploaded_file)
    except TypeError:
        # Older impl didn't accept uploaded_file kw; call without it
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: train_selected_models_only fallback used (no uploaded_file kw accepted)")
        except Exception:
            pass
        return _train_impl(selected_models, allow_install=allow_install, fast=fast)

def process_telco_data(allow_install=False):
    """
    Main function to process Telco data with all improvements.
    """
    try:
        # Ensure libraries
        available = ensure_optional_libs(allow_install)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Available libraries: {available}")

        # Load and engineer data
        df = load_telco_data()
        
        # Get preprocessor
        preprocessor, numeric_cols, categorical_cols = get_preprocessor(df)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Preprocessor created. Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")

        # Split data
        X_train, X_test, y_train, y_test = split_xy(df)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Data split. Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

        # Get tuned models
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting model training phase...")
        models = get_tuned_models(preprocessor, X_train, y_train)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Created {len(models)} tuned models")

        # Create ensemble
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏆 Creating ensemble model...")
        models = create_ensemble_model(models, preprocessor, X_train, y_train)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Ensemble model added! Total models: {len(models)}")
        
        # Preprocess data once for evaluation
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Preprocessing data for evaluation...")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Data preprocessing complete!")

        # Evaluate all models
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Starting model evaluation phase...")
        scorecard_df, roc_path, conf_path, best_name, trained = evaluate_models(
            X_train, y_train, X_test, y_test, models
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Model evaluation complete!")

        # Get feature names for SHAP
        feature_names_out = preprocessor.get_feature_names_out()

        return {
            "success": True,
            "scorecard": scorecard_df,
            "roc_path": roc_path,
            "conf_path": conf_path,
            "best_name": best_name,
            "trained": trained,
            "feature_names": feature_names_out,
            "X_sample": X_test[:100],  # sample for SHAP
            "preprocessor": preprocessor
        }

    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# -------------------------------
# Configuration & Time Estimation
# -------------------------------
def estimate_training_time(model_config):
    """Estimate training time based on model configuration."""
    base_time = 30  # Base time in seconds
    
    # Calculate total combinations
    total_combinations = 1
    for param, values in model_config.items():
        if isinstance(values, list):
            total_combinations *= len(values)
    
    # Estimate time per combination (including CV)
    time_per_combination = 2  # seconds per combination × 3-fold CV
    estimated_time = (total_combinations * time_per_combination) + base_time
    
    return estimated_time, total_combinations

def get_model_configurations():
    """Get all available model configurations with time estimates."""
    configs = {
        "Logistic Regression": {
            "params": {
                'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'clf__penalty': ['l1', 'l2'],
                'clf__solver': ['liblinear', 'saga'],
                'clf__class_weight': ['balanced', None]
            },
            "description": "Linear model with L1/L2 regularization",
            "complexity": "Low",
            "best_for": "Interpretability, baseline performance"
        },
        "Random Forest": {
            "params": {
                'clf__n_estimators': [300, 500, 700],
                'clf__max_depth': [8, 12, 16, None],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4],
                'clf__class_weight': ['balanced', 'balanced_subsample']
            },
            "description": "Ensemble of decision trees with bagging",
            "complexity": "Medium",
            "best_for": "Robust performance, feature importance"
        },
        "Gradient Boosting": {
            "params": {
                'clf__n_estimators': [200, 400, 600],
                'clf__learning_rate': [0.01, 0.05, 0.1],
                'clf__max_depth': [3, 5, 7],
                'clf__subsample': [0.8, 0.9, 1.0]
            },
            "description": "Sequential boosting with gradient descent",
            "complexity": "Medium-High",
            "best_for": "High accuracy, complex patterns"
        },
        "XGBoost": {
            "params": {
                'clf__n_estimators': [400, 600, 800],
                'clf__max_depth': [3, 5, 7],
                'clf__learning_rate': [0.01, 0.05, 0.1],
                'clf__subsample': [0.8, 0.9, 1.0],
                'clf__colsample_bytree': [0.7, 0.8, 0.9]
            },
            "description": "Optimized gradient boosting with regularization",
            "complexity": "High",
            "best_for": "Competition-level performance, scalability"
        },
        "CatBoost": {
            "params": {
                'clf__iterations': [300, 500, 700],
                'clf__learning_rate': [0.01, 0.05, 0.1],
                'clf__depth': [4, 6, 8],
                'clf__l2_leaf_reg': [1, 3, 5, 7]
            },
            "description": "Categorical boosting with advanced preprocessing",
            "complexity": "Medium-High",
            "best_for": "Categorical features, robust training"
        }
    }
    
    # Calculate time estimates
    for model_name, config in configs.items():
        time_est, combinations = estimate_training_time(config["params"])
        config["estimated_time"] = time_est
        config["total_combinations"] = combinations
    
    return configs

# -------------------------------
# Professional Gradio Interface
# -------------------------------
# Module-level storage for latest training result (used by SHAP compute)
TRAINED_RESULT = None

# -------------------------------
# Frontend handler functions (top-level)
# -------------------------------
def train_models_handler(cv_folds, ensemble_size, feature_selection, lr_selected, rf_selected, gb_selected, xgb_selected, cat_selected, mode, uploaded_file=None):
    """Wrapper handler for Gradio. Uses train_selected_models_only and stores result."""
    global TRAINED_RESULT
    # Mirror previous nested train_models behavior
    # Set defaults
    if cv_folds is None:
        cv_folds = 3
    if ensemble_size is None:
        ensemble_size = 3
    if lr_selected is None:
        lr_selected = False
    if rf_selected is None:
        rf_selected = False
    if gb_selected is None:
        gb_selected = False
    if xgb_selected is None:
        xgb_selected = False
    if cat_selected is None:
        cat_selected = False

    # Build selected models
    selected_models = []
    if lr_selected:
        selected_models.append("Logistic Regression")
    if rf_selected:
        selected_models.append("Random Forest")
    if gb_selected:
        selected_models.append("Gradient Boosting")
    if xgb_selected:
        selected_models.append("XGBoost")
    if cat_selected:
        selected_models.append("CatBoost")

    if not selected_models:
        return (pd.DataFrame(), "Error: No models selected", None, None, "Status: No models selected", "Estimated Time: 0s")

    fast_mode = (mode == "Fast")

    # start training
    result = _call_train(selected_models, allow_install=True, fast=fast_mode, uploaded_file=uploaded_file)
    if result.get("success"):
        TRAINED_RESULT = result
        best_row = result["scorecard"].iloc[0]
        best_summary = f"""
### Best Model: {best_row['Model']}

Test Performance:
- Accuracy: {best_row['Test Accuracy']:.3f}
- F1 Score: {best_row['Test F1']:.3f}
- AUC: {best_row['Test AUC']:.3f}
- Precision: {best_row['Test Precision']:.3f}
- Recall: {best_row['Test Recall']:.3f}

Cross-Validation:
- CV AUC: {best_row['CV AUC Mean']:.3f} ± {best_row['CV AUC Std']:.3f}
"""
        return (result["scorecard"], best_summary, result.get("roc_path"), result.get("conf_path"), "Status: Training completed successfully! ✅", "")
    else:
        err = result.get('error', 'Unknown error')
        return (pd.DataFrame(), f"Error: {err}", None, None, f"Status: Training failed ❌ - {err}", "Estimated Time: Error occurred")


def test_ui_handler():
    return "UI is working! Status updated successfully.", "Test completed"


def compute_shap_handler():
    global TRAINED_RESULT
    if TRAINED_RESULT:
        result = TRAINED_RESULT
        shap_paths = compute_shap_images(
            result["trained"][result["best_name"]],
            result["preprocessor"],
            result["X_sample"],
            result["feature_names"]
        )
        if len(shap_paths) >= 2:
            return shap_paths[0], shap_paths[1]
        return None, None
    return None, None


def preview_dataset_handler(uploaded_file=None):
    try:
        if uploaded_file:
            path = uploaded_file.name if hasattr(uploaded_file, 'name') else uploaded_file
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(TELCO_CSV)
        return df.head(50)
    except Exception:
        return pd.DataFrame()


def download_dataset_handler():
    """Module-level download handler used by the UI buttons (calls kaggle downloader)."""
    try:
        success, msg = kaggle_download_telco()
        if success:
            try:
                df = pd.read_csv(TELCO_CSV)
                preview = df.head(20)
            except Exception:
                preview = pd.DataFrame()
            return msg, preview
        else:
            return msg, pd.DataFrame()
    except Exception as e:
        return f"Download error: {e}", pd.DataFrame()


def train_streaming_handler(cv_folds, ensemble_size, feature_selection, lr_selected, rf_selected, gb_selected, xgb_selected, cat_selected, mode, uploaded_file=None):
    """Module-level streaming training generator compatible with Gradio Blocks.

    Mirrors previous nested `train_models` generator but stores results in the
    module-level TRAINED_RESULT so other handlers can consume it.
    """
    global TRAINED_RESULT
    # Set defaults
    cv_folds = 3 if cv_folds is None else cv_folds
    ensemble_size = 3 if ensemble_size is None else ensemble_size
    lr_selected = bool(lr_selected)
    rf_selected = bool(rf_selected)
    gb_selected = bool(gb_selected)
    xgb_selected = bool(xgb_selected)
    cat_selected = bool(cat_selected)

    selected_models = []
    if lr_selected: selected_models.append("Logistic Regression")
    if rf_selected: selected_models.append("Random Forest")
    if gb_selected: selected_models.append("Gradient Boosting")
    if xgb_selected: selected_models.append("XGBoost")
    if cat_selected: selected_models.append("CatBoost")

    if not selected_models:
        yield (pd.DataFrame(), "Error: No models selected", None, None, "Status: No models selected", "Estimated Time: 0s", "", "", "<div></div>")
        return

    # Rough estimate
    est = 0
    est += 0 if not lr_selected else 40
    est += 0 if not rf_selected else 72
    est += 0 if not gb_selected else 81
    est += 0 if not xgb_selected else 243
    est += 0 if not cat_selected else 108
    time_update = f"Estimated Time: {est//60}m {est%60}s"

    fast_mode = (mode == "Fast")

    # Initial UI pulse
    yield (pd.DataFrame(), "Preparing data and environment...", None, None, f"Status: Preparing - fast={fast_mode}", time_update, "", "", "<div style='width:100%;background:#eef2ff;border-radius:8px;padding:6px'><div style='width:5%;background:#4f46e5;color:white;padding:6px;border-radius:6px;text-align:center;'>5%</div></div>")

    # Log capture
    log_lines = []
    lock = threading.Lock()
    orig_print = builtins.print

    def proxy_print(*args, **kwargs):
        try:
            text = " ".join(str(a) for a in args)
        except Exception:
            text = str(args)
        try:
            orig_print(*args, **kwargs)
        except Exception:
            pass
        with lock:
            log_lines.append(text)

    result_container = {}

    def worker():
        try:
            builtins.print = proxy_print
            res = _call_train(selected_models, allow_install=True, fast=fast_mode, uploaded_file=uploaded_file)
            result_container['result'] = res
        except Exception as e:
            result_container['result'] = {"success": False, "error": str(e)}
        finally:
            builtins.print = orig_print

    th = threading.Thread(target=worker)
    th.start()

    # Stream logs while worker runs
    while th.is_alive():
        time.sleep(0.7)
        with lock:
            log_text = "\n".join(log_lines[-200:])
        prog = min(95, max(5, int(min(100, len(log_lines) / 5))))
        prog_html = f"<div style='width:100%;background:#eef2ff;border-radius:8px;padding:6px'><div style='width:{prog}%;background:#4f46e5;color:white;padding:6px;border-radius:6px;text-align:center;'>{prog}%</div></div>"
        yield (pd.DataFrame(), "Training in progress...", None, None, "Status: Training...", time_update, log_text, "", prog_html)

    # Finished
    result = result_container.get('result', {"success": False, "error": "No result"})
    final_log = "\n".join(log_lines)
    if result.get('success'):
        TRAINED_RESULT = result
        try:
            best_row = result["scorecard"].iloc[0]
            best_summary = f"""
### Best Model: {best_row['Model']}

Test Performance:
- Accuracy: {best_row['Test Accuracy']:.3f}
- F1 Score: {best_row['Test F1']:.3f}
- AUC: {best_row['Test AUC']:.3f}
- Precision: {best_row['Test Precision']:.3f}
- Recall: {best_row['Test Recall']:.3f}

Cross-Validation:
- CV AUC: {best_row['CV AUC Mean']:.3f} ± {best_row['CV AUC Std']:.3f}
"""
        except Exception:
            best_summary = "Training completed"

        key_md = "### Key takeaways\n"
        try:
            best_pipe = result["trained"].get(result.get("best_name")) if result.get("best_name") else None
            final_est = _get_final_estimator(best_pipe) if best_pipe is not None else None
            if final_est is not None and hasattr(final_est, 'feature_importances_'):
                importances = getattr(final_est, 'feature_importances_')
                fnames = result.get('feature_names', [])
                if len(importances) == len(fnames):
                    idx = list(np.argsort(importances)[::-1][:6])
                    top = [f"{fnames[i]} ({importances[i]:.3f})" for i in idx]
                    key_md += "**Top features:**\n\n"
                    for t in top:
                        key_md += f"- {t}\n"
        except Exception:
            pass

        prog_html = "<div style='width:100%;background:#eef2ff;border-radius:8px;padding:6px'><div style='width:100%;background:#4f46e5;color:white;padding:6px;border-radius:6px;text-align:center;'>100%</div></div>"
        yield (result["scorecard"], best_summary, result.get("roc_path"), result.get("conf_path"), "Status: Training completed successfully! ✅", time_update, final_log, key_md, prog_html)
        return
    else:
        err = result.get('error', 'Unknown error')
        final_log = final_log + "\nERROR: " + str(err)
        prog_html = "<div style='width:100%;background:#fee2e2;border-radius:8px;padding:6px'><div style='width:100%;background:#ef4444;color:white;padding:6px;border-radius:6px;text-align:center;'>ERROR</div></div>"
        yield (pd.DataFrame(), f"Error: {err}", None, None, f"Status: Training failed ❌ - {err}", "Estimated Time: Error occurred", final_log, "", prog_html)
        return

def create_gradio_interface():
    """
    Create an absolutely minimal interface that works with Gradio 4.31.0.
    All components are created at top level inside a single Blocks() context,
    and all event handlers are registered at the end before exiting context.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: Creating bare minimal interface for Gradio 4.31.0")
    
    # Define demo outside any with blocks to ensure clean scope
    demo = gr.Blocks()
    
    # ALL components are created inside this single with block
    with demo:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: Creating base components")
        
        # Basic styling
        gr.HTML("""<style>body { font-family: sans-serif; } .gr-button { margin: 5px; }</style>""")
        
        # UI Header
        gr.Markdown("# Telco Churn Prediction")
        
        # Create ALL components first
        # Input components
        file_input = gr.File(file_count="single", file_types=['.csv'], label="Upload CSV (optional)")
        preview_btn = gr.Button("Preview Dataset")
        download_btn = gr.Button("Download dataset (Kaggle)")
        dataset_preview = gr.DataFrame(headers=None, interactive=False)
        download_log = gr.Textbox(label="Download log", lines=6)
        
        # Status and outputs
        run_log = gr.Textbox(label="Run log", lines=10, interactive=False)
        progress_bar = gr.HTML("<div style='width:100%;background:#eee;border-radius:8px;padding:6px'><div style='width:0%;background:blue;color:white;text-align:center;'>0%</div></div>")
        status_text = gr.Markdown("Status: Ready") 
        time_estimate = gr.Markdown("Time: Calculating...")
        
        # Output components
        out_scorecard = gr.DataFrame()
        out_best_model = gr.Markdown()
        out_key_takeaways = gr.Markdown()
        out_roc = gr.Image(type="filepath")
        out_conf = gr.Image(type="filepath")
        
        # SHAP components
        shap_summary = gr.HTML("<div id='shap-area'></div>")
        shap_bar = gr.HTML("<div id='shap-area'></div>")
        
        # Model selection
        gr.Markdown("## Model Selection")
        lr_checkbox = gr.Checkbox(label="Logistic Regression", value=True)
        rf_checkbox = gr.Checkbox(label="Random Forest", value=True) 
        gb_checkbox = gr.Checkbox(label="Gradient Boosting", value=True)
        xgb_checkbox = gr.Checkbox(label="XGBoost", value=False)
        cat_checkbox = gr.Checkbox(label="CatBoost", value=False)
        
        # Options
        gr.Markdown("## Options")
        cv_folds = gr.Number(value=3, label="CV Folds (3-10)")
        ensemble_size = gr.Number(value=3, label="Ensemble Size (2-5)") 
        feature_selection = gr.Checkbox(label="Feature Selection", value=True)
        mode_toggle = gr.Radio(choices=["Fast", "Full"], value="Fast", label="Mode")
        
        # Buttons
        gr.Markdown("## Actions")
        btn_train = gr.Button("Start Training")
        btn_test = gr.Button("Test UI")
        btn_shap = gr.Button("Compute SHAP")
        
        # ======================================================
        # REGISTER ALL EVENT HANDLERS AT THE END OF BLOCKS CONTEXT
        # This ensures all handlers are registered while the context
        # is still active, which is critical for Gradio 4.31.0
        # ======================================================
        print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: Registering ALL button handlers in flat structure")
        
        # Register click handlers INSIDE the Blocks context but AFTER creating ALL components
        preview_btn.click(
            fn=preview_dataset_handler, 
            inputs=[file_input], 
            outputs=[dataset_preview]
        )
        
        download_btn.click(
            fn=download_dataset_handler, 
            inputs=None, 
            outputs=[download_log, dataset_preview]
        )
        
        btn_train.click(
            fn=train_streaming_handler,
            inputs=[cv_folds, ensemble_size, feature_selection, lr_checkbox, rf_checkbox, gb_checkbox, xgb_checkbox, cat_checkbox, mode_toggle, file_input],
            outputs=[out_scorecard, out_best_model, out_roc, out_conf, status_text, time_estimate, run_log, out_key_takeaways, progress_bar]
        )
        
        btn_test.click(
            fn=test_ui_handler,
            outputs=[status_text, time_estimate]
        )
        
        btn_shap.click(
            fn=compute_shap_handler, 
            outputs=[shap_summary, shap_bar]
        )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: ALL click handlers registered successfully")
        
        # Try to load data on startup (directly instead of using demo.load)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: Attempting to load initial data")
        try:
            df = load_telco_data()
            dataset_preview.update(value=df.head(20))
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Initial data load failed: {e}")
    
    # Return the demo after all components and event handlers are registered
    print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: Successfully created interface with all handlers")
    return demo

# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
                "- Data ingestion: accept CSV uploads or automatically fetch the canonical Telco dataset from Kaggle (Space secrets required).\n"
                "- Feature engineering: domain-driven derived features (tenure buckets, charge ratios, service counts) so simple linear models and trees can learn robust signals.\n"
                "- Model training: configurable selection of models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, CatBoost).\n"
                "- Safe hosting defaults: `Fast` mode uses small hyperparameter grids and skips heavy libraries when not available; `Full` mode expands search when resources allow.\n"
                "- Explainability: SHAP summary and per-feature contributions so predictions are actionable and auditable.\n\n"
                "Design goals: reproducibility, clear decision records, and an educational narrative that explains tradeoffs (speed vs. accuracy, interpretability vs. capacity)."
            )

            gr.Markdown("## ML Features")
            gr.Markdown("- Advanced Feature Engineering (tenure groups, charge ratios, service interactions)")
            gr.Markdown("- Stratified cross-validation & calibrated scoring")
            gr.Markdown("- Hyperparameter search (safe grids in Fast mode, wider search in Full mode)")
            gr.Markdown("- Optional tree boosters (XGBoost / CatBoost) when available")
            gr.Markdown("- Ensemble via voting and simple stacking")
            gr.Markdown("- Explainability via SHAP: global summary and per-sample contributions")

            gr.Markdown("---")
            gr.Markdown("## How to interpret the UI and outputs")
            gr.Markdown(
                "- 'Status' updates stream key pipeline steps: data load, preprocessing, model search, evaluation.\n"
                "- 'Estimated Time' shows a coarse time heuristic based on selected models for transparency.\n"
                "- Use `Preview Dataset` to inspect uploaded or downloaded CSV before training.\n"
                "- After training, explore ROC and confusion plots to understand class tradeoffs; use SHAP to justify individual predictions for stakeholders."
            )

            gr.Markdown("---")
            gr.Markdown("## Getting started (step-by-step)")
            gr.Markdown(
                "Follow these steps to run the full ML flow and reproduce strong results on the Telco dataset:\n\n"
                "1. Data source: By default, the app attempts to download the canonical Telco dataset from Kaggle. If you prefer, upload your CSV using 'Upload CSV'.\n"
                "2. Preview / EDA: Click 'Preview Dataset' to inspect the top rows and ensure data types are correct.\n"
                "3. Feature engineering: The pipeline constructs domain features (tenure groups, charge ratios, service counts). Inspect features locally if you wish to modify.\n"
                "4. Mode selection: Use 'Fast' for quick iterations (safe grids and skipped heavy boosters). Switch to 'Full' for thorough tuning and including XGBoost/CatBoost.\n"
                "5. Model selection: Select one or more models; combining models yields a more robust ensemble.\n"
                "6. Train: Click 'Start Training' and follow the Status messages to watch progress.\n"
                "7. Evaluate & explain: Inspect scorecard, ROC, confusion matrix; click 'Compute SHAP' for explainability.\n"
                "8. Persist & export: Models are saved under the `/models` folder; download artifacts for production deployment."
            )

            gr.Markdown("## How to reach production-quality accuracy")
            gr.Markdown(
                "Recommended path to maximize performance while remaining auditable:\n\n"
                "- Start in `Fast` mode to validate the pipeline.\n"
                "- Switch to `Full` mode and enable XGBoost/CatBoost in an environment where those packages are available.\n"
                "- Increase CV folds (5-10), expand search spaces, and consider RandomizedSearch/Optuna for faster coverage.\n"
                "- Use targeted feature selection (SelectKBest or model importances) to remove noisy columns.\n"
                "- Calibrate predicted probabilities (Platt or isotonic) for better thresholding.\n"
                "- Ensemble top performers and validate on a holdout set for robustness.\n"
            )

            with gr.Column():
                gr.Markdown("## Configuration")

                # Model Selection (labeled)
                gr.Markdown("### Model Selection")
                lr_checkbox = gr.Checkbox(label="Logistic Regression")
                rf_checkbox = gr.Checkbox(label="Random Forest")
                gb_checkbox = gr.Checkbox(label="Gradient Boosting")
                xgb_checkbox = gr.Checkbox(label="XGBoost")
                cat_checkbox = gr.Checkbox(label="CatBoost")

                # Options
                gr.Markdown("### Options")
                cv_folds = gr.Number(value=3, label="CV Folds (3-10)")
                ensemble_size = gr.Number(value=3, label="Ensemble Size (2-5)")
                feature_selection = gr.Checkbox(label="Feature Selection")
                mode_toggle = gr.Radio(choices=["Fast", "Full"], value="Fast", label="Mode")
                # file_input, preview_btn and download_btn are created earlier
                # (top of Blocks) and their handlers are already registered.
        
                # Controls
                gr.Markdown("## Training")
                # Create button without binding click handler
                btn_train = gr.Button("Start Training")
                
                # We'll register all event handlers at the end of the Blocks context

                # Create button without binding click handler
                btn_test = gr.Button("Test UI")


    # Run log (scrollable) to surface terminal output from training
    # (placeholders created earlier)

                # Results
                gr.Markdown("## Results")
                gr.HTML("</div>")
                gr.HTML("<div class='results-card'>")
        # Plain-English guidance for non-technical viewers
        gr.Markdown(
            "### Quick read: What you just saw (plain English)\n"
            "- Scorecard: a ranked table of models and their core metrics (accuracy, F1, AUC).\n"
            "- Best model: the single model with the best held-out test AUC.\n"
            "- ROC / Confusion: visual tools to understand tradeoffs between false positives and false negatives.\n"
            "- Key takeaways: concise metrics and the top features that most influenced model predictions."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Performance")
                # ...existing code uses the earlier placeholders

            with gr.Column():
                gr.Markdown("### Visualizations")
                # ...existing code uses the earlier placeholders

                # SHAP
                gr.Markdown("## SHAP Analysis")
                with gr.Row():
                    with gr.Column():
                        # Constrain SHAP images to a wrapper so CSS can control sizing
                        shap_summary = gr.HTML("<div id='shap-area'></div>")
                    with gr.Column():
                        shap_bar = gr.HTML("<div id='shap-area'></div>")
                        # Create button without binding click handler
                        btn_shap = gr.Button("Compute SHAP")                # Preview and download handlers are registered earlier next to their components.

                # Quick demo flow – concise and professional guidance for live demos
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("---")
                        gr.Markdown("## Quick demo flow")
                        gr.Markdown(
                            "1. Load or upload a CSV and click 'Preview Dataset' to verify the table and datatypes.\n"
                            "2. Select models and choose 'Fast' for a fast end-to-end run or 'Full' for thorough tuning.\n"
                            "3. Click 'Start Training' and watch the Status messages for data, feature engineering, tuning, and evaluation steps.\n"
                            "4. Inspect the scorecard and ROC/Confusion visualizations to evaluate performance tradeoffs.\n"
                            "5. Use 'Compute SHAP' to produce global and per-feature explainability artifacts.\n"
                            "6. Download or persist trained models from the /models folder for production use."
                )
                gr.Markdown("---")

        # On load (Spaces start), attempt to ensure dataset is present and show a preview
                def _on_startup():
                    # Try to load data via the same loader; returns a small preview or error
                    try:
                        df = load_telco_data()
                        return df.head(20)
                    except Exception as e:
                        print(f"Startup dataset load failed: {e}")
                        return pd.DataFrame()

                # Hook into Gradio load so Spaces will execute the loader at startup
                # Just run the startup function directly instead of using demo.load
                # which is also causing context issues in Gradio 4.31.0
                print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: Running startup function directly")
                try:
                    preview_df = _on_startup()
                    if not preview_df.empty:
                        dataset_preview.update(value=preview_df)
                except Exception as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR in startup: {e}")
                
                # ======================================================
                # REGISTER ALL EVENT HANDLERS AT THE END OF BLOCKS CONTEXT
                # This ensures all handlers are registered while the context
                # is still active, which is critical for Gradio 4.31.0
                # ======================================================
                print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: Registering ALL button click handlers at end of Blocks context")
                
                # Training button
                btn_train.click(
                    fn=train_streaming_handler,
                    inputs=[cv_folds, ensemble_size, feature_selection, lr_checkbox, rf_checkbox, gb_checkbox, xgb_checkbox, cat_checkbox, mode_toggle, file_input],
                    outputs=[out_scorecard, out_best_model, out_roc, out_conf, status_text, time_estimate, run_log, out_key_takeaways, progress_bar]
                )
                
                # Test button
                btn_test.click(
                    fn=test_ui_handler,
                    outputs=[status_text, time_estimate]
                )
                
                # SHAP button
                btn_shap.click(
                    fn=compute_shap_handler, 
                    outputs=[shap_summary, shap_bar]
                )
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: ALL click handlers registered successfully")

    # Critical: Make sure we're exiting the Blocks context correctly
    print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: Exiting create_gradio_interface() and returning demo")
    return demo

# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)