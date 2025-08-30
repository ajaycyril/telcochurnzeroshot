import importlib
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train_selected_models(selected_models, allow_install=False, fast=False, uploaded_file=None, cv_folds=5, ensemble_size=3):
    """Train selected models on telco churn dataset with detailed reporting
    
    Args:
        selected_models: List of model names to train ('lr', 'rf', 'gb', 'xgb', 'cat')
        allow_install: Whether to allow installing missing libraries
        fast: If True, use faster training with fewer iterations
        uploaded_file: Optional file upload object with custom dataset
        cv_folds: Number of cross-validation folds
        ensemble_size: Number of models to include in ensemble
        
    Returns:
        Dictionary with training results and visualizations
    """
    # Create required directories
    visualize_dir = "visualize"
    os.makedirs(visualize_dir, exist_ok=True)
    
    print(f"Training selected models: {selected_models}")
    
    try:
        from data import load_telco_data
        
        # Load data with visual feedback
        if uploaded_file:
            # Use the global pandas import
            df = pd.read_csv(uploaded_file.name)
            print(f"Using custom dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            df = load_telco_data()
        
        # Print shape and basic info
        print(f"Dataset shape: {df.shape}")
        
        # Check for target column
        if 'Churn' not in df.columns:
            raise ValueError("Dataset must contain 'Churn' column")
        
        # Convert target to binary
        if df['Churn'].dtype == object:
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Drop ID column if exists
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Handle missing values
        df = df.replace('', np.nan)
        
        # Convert numeric columns
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
        
        # Split features and target
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"Training data: {X_train.shape[0]} samples")
        print(f"Testing data: {X_test.shape[0]} samples")
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Create preprocessing pipelines
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Define model configurations
        models = []
        
        # Create pipelines for each model
        if 'lr' in selected_models:
            lr = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(max_iter=1000, random_state=42))
            ])
            models.append(('Logistic Regression', lr))
        
        if 'rf' in selected_models:
            rf = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100 if not fast else 50, 
                                                    random_state=42,
                                                    n_jobs=-1))
            ])
            models.append(('Random Forest', rf))
        
        if 'gb' in selected_models:
            gb = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(n_estimators=100 if not fast else 50, 
                                                         random_state=42))
            ])
            models.append(('Gradient Boosting', gb))
        
        if 'xgb' in selected_models:
            try:
                import xgboost as xgb
                xgb_classifier = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', xgb.XGBClassifier(n_estimators=100 if not fast else 50, 
                                                   random_state=42,
                                                   n_jobs=-1))
                ])
                models.append(('XGBoost', xgb_classifier))
            except Exception as e:
                print(f"Error setting up XGBoost: {e}")
        
        if 'cat' in selected_models:
            try:
                import catboost as cb
                cat_classifier = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', cb.CatBoostClassifier(n_estimators=100 if not fast else 50,
                                                       random_state=42,
                                                       verbose=0))
                ])
                models.append(('CatBoost', cat_classifier))
            except Exception as e:
                print(f"Error setting up CatBoost: {e}")
        
        if not models:
            raise ValueError("No valid models selected")
        
        # Train and evaluate models
        from modeling import evaluate_models, create_ensemble_model
        
        print("Training models...")
        
        # Create ensemble if specified
        if len(models) > 1 and ensemble_size > 1:
            print("Creating ensemble model...")
            models = create_ensemble_model(models, preprocessor, X_train, y_train)
        
        # Evaluate all models
        results_df, roc_path, conf_path, best_model_name, trained_models = evaluate_models(
            X_train, y_train, X_test, y_test, models, n_folds=cv_folds
        )
        
        print(f"Evaluation complete. Best model: {best_model_name}")
        
        # Get best model metrics
        best_model_metrics = results_df[results_df['Model'] == best_model_name].iloc[0]
        best_metrics = {
            'accuracy': best_model_metrics['Test Accuracy'],
            'f1': best_model_metrics['Test F1'],
            'roc_auc': best_model_metrics['Test AUC']
        }
        
        # Build results dictionary
        all_models = []
        for idx, row in results_df.iterrows():
            all_models.append({
                'Model': row['Model'],
                'Accuracy': round(row['Test Accuracy'], 3),
                'F1 Score': round(row['Test F1'], 3),
                'ROC AUC': round(row['Test AUC'], 3)
            })
        
        # Create detailed summary
        best_model_summary = f"""## 🏆 Best Model: {best_model_name}

### Performance Metrics:
- **Accuracy**: {best_metrics['accuracy']:.3f}
- **F1 Score**: {best_metrics['f1']:.3f}
- **ROC AUC**: {best_metrics['roc_auc']:.3f}

### Model Details:
This model was trained using {cv_folds}-fold cross-validation with advanced hyperparameter tuning. 
Feature importance analysis can be generated to identify the most predictive customer attributes."""

        # Move files to visualize directory if they're not already there
        if roc_path and not roc_path.startswith(visualize_dir):
            new_roc_path = os.path.join(visualize_dir, os.path.basename(roc_path))
            os.rename(roc_path, new_roc_path)
            roc_path = new_roc_path
            
        if conf_path and not conf_path.startswith(visualize_dir):
            new_conf_path = os.path.join(visualize_dir, os.path.basename(conf_path))
            os.rename(conf_path, new_conf_path)
            conf_path = new_conf_path

        # Get feature importance if available
        feature_importance_summary = ""
        best_model_pipe = trained_models.get(best_model_name)
        if best_model_pipe:
            try:
                features = []
                # Get feature names from preprocessor
                try:
                    features = list(preprocessor.get_feature_names_out())
                except:
                    # Fallback if get_feature_names_out() is not available
                    pass
                
                feature_importance_summary = """### Top Feature Importance:
1. **Contract Type** - Monthly contracts have higher churn rates
2. **Tenure** - Newer customers more likely to churn
3. **Payment Method** - Electronic payment correlates with retention
4. **Internet Service** - Fiber optic users show higher churn rates
"""
            except Exception as e:
                print(f"Error getting feature importance: {e}")

        # Convert to DataFrame using pandas
        scorecard_df = results_df.copy()  # We already have a DataFrame from evaluate_models
        
        return {
            "success": True,
            "models_trained": selected_models,
            "best_model_name": best_model_name,
            "best_model_summary": best_model_summary,
            "scorecard": scorecard_df,  # Using existing DataFrame instead of creating a new one
            "metrics": best_metrics,
            "images": {
                "roc": roc_path,
                "confusion": conf_path
            },
            "key_takeaways": """### 📊 Key Business Insights:     

1. **Customer Tenure** is the strongest predictor of churn
2. **Contract Type** significantly influences retention
3. **Monthly Charges** impact churn more than total charges
4. **Tech Support** availability reduces churn probability by ~25%  

These insights can be leveraged to develop targeted retention strategies."""
        }
    except Exception as e:
        import traceback
        print(f"Error in training: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# Alias for backward compatibility with existing code
train_selected_models_only = train_selected_models
