import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


def _get_final_estimator(obj):
    try:
        if hasattr(obj, 'steps') and isinstance(obj.steps, (list, tuple)) and len(obj.steps) > 0:
            return obj.steps[-1][1]
    except Exception:
        pass
    return obj


def create_ensemble_model(models, preprocessor, X_train, y_train):
    cv_scores = []
    for name, model in models:
        try:
            cv_auc = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=3, n_jobs=-1)
            cv_scores.append((name, model, cv_auc.mean()))
        except Exception:
            cv_scores.append((name, model, np.nan))
    cv_scores.sort(key=lambda x: (x[2] if not np.isnan(x[2]) else -1.0), reverse=True)
    top_models = cv_scores[:3]
    ensemble = VotingClassifier(estimators=[(name, model) for name, model, _ in top_models], voting='soft')
    ensemble.fit(X_train, y_train)
    models.append(("Ensemble", ensemble))
    return models


def evaluate_models(X_train, y_train, X_test, y_test, models, n_folds=5):
    results = []
    trained = {}
    fprs, tprs, names_for_roc = [], [], []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_auc_by_model = []
    for name, pipe in models:
        try:
            cv_auc = cross_val_score(pipe, X_train, y_train, scoring="roc_auc", cv=skf, n_jobs=-1)
            cv_auc_by_model.append((name, cv_auc.mean(), cv_auc.std()))
        except Exception:
            cv_auc_by_model.append((name, np.nan, np.nan))

    for name, pipe in models:
        try:
            pipe.fit(X_train, y_train)
        except Exception:
            continue
        trained[name] = pipe
        final_est = _get_final_estimator(pipe)
        y_proba = None
        if hasattr(pipe, "predict_proba"):
            try:
                y_proba = pipe.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None
        if y_proba is None and hasattr(final_est, "predict_proba"):
            try:
                y_proba = final_est.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None
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

        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fprs.append(fpr); tprs.append(tpr); names_for_roc.append(name)
        except Exception:
            pass

    scorecard_df = pd.DataFrame(results)
    if not scorecard_df.empty:
        scorecard_df = scorecard_df.sort_values("Test Accuracy", ascending=False).reset_index(drop=True)

    roc_path = f"roc_all_{datetime.now().strftime('%H%M%S')}.png"
    try:
        plt.figure(figsize=(10, 8))
        for fpr, tpr, name in zip(fprs, tprs, names_for_roc):
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curves (Test)")
        plt.legend(loc="lower right", frameon=True); plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(roc_path, dpi=140, bbox_inches='tight'); plt.close()
    except Exception:
        roc_path = None

    best_name = None; conf_path = None
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
                fig, ax = plt.subplots(figsize=(6, 6)); sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(f"Confusion Matrix (Best: {best_name})")
                plt.tight_layout(); plt.savefig(conf_path, dpi=140, bbox_inches='tight'); plt.close()
            except Exception:
                conf_path = None

    return scorecard_df, roc_path, conf_path, best_name, trained
