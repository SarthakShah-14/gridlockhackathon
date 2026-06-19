import pandas as pd
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use('Agg') # Safe headless execution
import matplotlib.pyplot as plt
import shap

def generate_shap_interpretability(model, X: pd.DataFrame, 
                                   cat_features: list, target_name: str, 
                                   output_dir: str = "reports") -> dict:
    """
    Computes SHAP values, generates summary and dependence plots, 
    and saves global_importance.csv and local_explanations.pkl for the target.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prep inputs for CatBoost Pool
    X_cb = X.copy()
    for col in cat_features:
        if col in X_cb.columns:
            X_cb[col] = X_cb[col].astype(str)
            
    # Fit SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_cb)
    
    # Handle shape of SHAP values
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            shap_vals_target = shap_values[1]
        else:
            shap_vals_target = shap_values[0]
    elif len(shap_values.shape) == 3:
        # multiclass / prob outputs
        shap_vals_target = shap_values[:, :, 1]
    else:
        shap_vals_target = shap_values
        
    # 1. Save local explanations pickle
    local_pkl_path = os.path.join(output_dir, f"local_explanations_{target_name}.pkl")
    try:
        with open(local_pkl_path, "wb") as f:
            pickle.dump(shap_vals_target, f)
    except Exception:
        pass
        
    # 2. Generate SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals_target, X_cb, show=False)
    plt.tight_layout()
    summary_path = os.path.join(output_dir, f"shap_summary_{target_name}.png")
    plt.savefig(summary_path, dpi=150)
    plt.close()
    
    # 3. Calculate and save SHAP global importance CSV
    mean_shap = np.abs(shap_vals_target).mean(axis=0)
    shap_importance = pd.DataFrame({
        'feature': X_cb.columns,
        'mean_shap': mean_shap
    }).sort_values(by='mean_shap', ascending=False)
    
    csv_path = os.path.join(output_dir, f"global_importance_shap_{target_name}.csv")
    shap_importance.to_csv(csv_path, index=False)
    
    # Save SHAP Dependence Plot for top feature
    if len(shap_importance) > 0:
        top_feature = shap_importance.iloc[0]['feature']
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(top_feature, shap_vals_target, X_cb, show=False)
        plt.tight_layout()
        dep_path = os.path.join(output_dir, f"shap_dependence_{target_name}_{top_feature}.png")
        plt.savefig(dep_path, dpi=150)
        plt.close()
        
    return {
        'shap_importance': shap_importance.head(30).to_dict(orient='records'),
        'summary_plot_path': summary_path,
        'local_pkl_path': local_pkl_path,
        'global_csv_path': csv_path
    }
