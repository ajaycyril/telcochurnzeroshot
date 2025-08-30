import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer

from utils import TELCO_CSV, kaggle_download_telco

def load_telco_data():
    """Load the Telco customer churn dataset from the local file or download it if not available."""
    import os
    import pandas as pd
    
    if os.path.exists(TELCO_CSV):
        print(f"Loading Telco data from {TELCO_CSV}")
        df = pd.read_csv(TELCO_CSV)
    else:
        print(f"Downloading Telco dataset...")
        kaggle_download_telco()
        df = pd.read_csv(TELCO_CSV)
        
    # Apply feature engineering
    return engineer_features(df)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-informed features for Telco dataset."""
    df = df.copy()
    # Coerce TotalCharges to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')

    # tenure buckets
    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 36, 48, 60, 72], labels=False)

    # Service counts
    svc_cols = [c for c in df.columns if 'Service' in c or 'Services' in c or c in ['PhoneService','MultipleLines','InternetService']]
    df['num_services'] = df[[c for c in ['PhoneService','MultipleLines','InternetService'] if c in df.columns]].apply(lambda r: r.eq('Yes').sum(), axis=1)

    # Basic binary mappings
    df['HasInternet'] = (df.get('InternetService') != 'No').astype(int) if 'InternetService' in df.columns else 0
    df['HasPhone'] = (df.get('PhoneService') == 'Yes').astype(int) if 'PhoneService' in df.columns else 0

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Feature engineering complete. New shape: {df.shape}")
    return df


def get_preprocessor(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['Churn', 'customerID']]
    categorical_cols = [col for col in categorical_cols if col not in ['Churn', 'customerID']]

    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(transformers=[
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ], remainder="drop")

    return pre, numeric_cols, categorical_cols


def split_xy(df):
    if "Churn" not in df.columns:
        raise ValueError("Column 'Churn' not found in dataset.")
    y = df["Churn"].map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("Target 'Churn' contains unexpected values. Expected 'Yes'/'No'.")
    drop_cols = [c for c in ["Churn", "customerID"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    return X_train, X_test, y_train, y_test
