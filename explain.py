from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def compute_shap_images(trained_pipe, preprocessor, X_sample, feature_names_out, shap_sample=100):
    paths = []
    try:
        import shap
    except Exception:
        print("shap not available")
        return []

    try:
        X_trans = preprocessor.transform(X_sample)
        model = trained_pipe
        n_sample = min(shap_sample, X_trans.shape[0])
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:,1] if hasattr(model, 'predict_proba') else model.predict(x), X_trans[:max(20,n_sample)])
        shap_values = explainer.shap_values(X_trans[:n_sample])
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        summary_path = f"shap_summary_{datetime.now().strftime('%H%M%S')}.png"
        plt.figure(figsize=(10,8)); shap.summary_plot(shap_values, X_trans[:n_sample], feature_names=feature_names_out, show=False); plt.tight_layout(); plt.savefig(summary_path, dpi=130, bbox_inches='tight'); plt.close(); paths.append(summary_path)
        bar_path = f"shap_bar_{datetime.now().strftime('%H%M%S')}.png"
        plt.figure(figsize=(10,8)); shap.summary_plot(shap_values, X_trans[:n_sample], feature_names=feature_names_out, plot_type='bar', show=False); plt.tight_layout(); plt.savefig(bar_path, dpi=130, bbox_inches='tight'); plt.close(); paths.append(bar_path)
    except Exception as e:
        print(f"SHAP failed: {e}")
        return []

    return paths
