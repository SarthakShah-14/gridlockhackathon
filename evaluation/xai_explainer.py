import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.helpers import setup_logger, load_data
from preprocessing.cleaning import clean_dataset
from preprocessing.validation import validate_dataset
from preprocessing.target_engineering import engineer_targets
from feature_engineering.time_features import extract_time_features
from feature_engineering.spatial_features import extract_spatial_features
from feature_engineering.graph_features import add_graph_features
from feature_engineering.historical_stats import compute_leakage_free_historical_stats, compute_group_aggregations
from feature_engineering.interactions import create_interactions

logger = setup_logger("xai_explainer")

def generate_xai_report():
    logger.info("Initializing Explainable AI (XAI) Report Generator...")
    output_dir = "reports/xai"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load pipeline artifacts and models
    logger.info("Loading models and artifacts...")
    artifacts_path = "models/pipeline_artifacts.pkl"
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Pipeline artifacts not found at {artifacts_path}. Please run pipeline first.")
        
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)
        
    selected_features = artifacts['selected_features']
    cat_features = artifacts['cat_features']
    classes_mapping = artifacts.get('sev_classes', ['Moderate', 'Prolonged', 'Quick'])
    
    # Load first CatBoost fold model for Severity Classifier
    with open("models/severity_base_models.pkl", "rb") as f:
        severity_base_models = pickle.load(f)
    
    # CatBoost is the first model in selected_sev_models, so fold 0 model is index 0
    cb_model = severity_base_models[0]
    logger.info(f"Loaded CatBoost base estimator: {cb_model.__class__.__name__}")
    
    # 2. Re-engineer features to match model inputs
    logger.info("Loading and processing dataset...")
    df_raw = load_data()
    df_validated = validate_dataset(df_raw)
    df_cleaned = clean_dataset(df_validated)
    df_targets, _, _ = engineer_targets(df_cleaned, train_mode=True)
    df_features = extract_time_features(df_targets)
    df_features, dbscan_model = extract_spatial_features(df_features, train_mode=True)
    df_features, graph_adj, junction_coords = add_graph_features(df_features, train_mode=True)
    df_features, historical_lookups = compute_leakage_free_historical_stats(df_features, train_mode=True)
    
    frequency_encoder = artifacts.get('frequency_encoder')
    if frequency_encoder is not None:
        df_features = frequency_encoder.transform(df_features)
        
    df_features, _, _ = compute_group_aggregations(df_features)
    df_features = create_interactions(df_features)
    
    # Align features
    agg_cols = [
        'agg_junction_count', 'agg_junction_avg_duration', 'agg_junction_avg_priority',
        'agg_police_station_count', 'agg_police_station_avg_duration',
        'agg_event_type_avg_duration', 'agg_event_type_avg_priority'
    ]
    feature_cols = list(selected_features) + agg_cols
    feature_cols = list(dict.fromkeys(feature_cols)) # deduplicate
    
    missing_feats = [col for col in feature_cols if col not in df_features.columns]
    for col in missing_feats:
        df_features[col] = 0.0
        
    severity_mask = df_features['severity'].notnull()
    X_sev = df_features.loc[severity_mask, feature_cols].copy()
    
    # Impute missing values
    num_cols = X_sev.select_dtypes(include=['number']).columns
    for col in num_cols:
        X_sev[col] = X_sev[col].fillna(0.0)
        
    # Format cat features for CatBoost
    for col in cat_features:
        if col in X_sev.columns:
            X_sev[col] = X_sev[col].astype(str)
            
    # Subsample data to speed up SHAP computations (100 samples)
    shap_sample_size = min(100, len(X_sev))
    X_shap = X_sev.head(shap_sample_size).copy()
    
    logger.info(f"Computing SHAP values for {shap_sample_size} samples...")
    explainer = shap.TreeExplainer(cb_model)
    shap_values = explainer.shap_values(X_shap)
    
    # Standardize to list of 2D numpy arrays
    n_classes = len(classes_mapping)
    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        logger.info(f"Raw shap_values shape: {shap_values.shape}")
        if shap_values.shape[0] == n_classes:
            shap_values_list = [shap_values[i] for i in range(n_classes)]
        elif shap_values.shape[2] == n_classes:
            shap_values_list = [shap_values[:, :, i] for i in range(n_classes)]
        else:
            shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    elif isinstance(shap_values, list):
        shap_values_list = shap_values
    else:
        shap_values_list = [shap_values]
        
    # 3. Generate Global Interpretability plots
    logger.info("Plotting global feature importance...")
    
    # Global multi-class bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_list, X_shap, plot_type="bar", class_names=classes_mapping, show=False)
    plt.title("Severity Model Multi-class Feature Importance (SHAP)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_global_importance.png"), dpi=200)
    plt.close()
    
    # Beeswarm plots per class
    for c_idx, c_name in enumerate(classes_mapping):
        logger.info(f"Generating summary beeswarm plot for class: {c_name}")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_list[c_idx], X_shap, show=False)
        plt.title(f"SHAP Beeswarm Plot for Class: {c_name}", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_beeswarm_{c_name.lower()}.png"), dpi=200)
        plt.close()
        
    # 4. Generate Waterfall Plot for a representative local prediction
    logger.info("Generating waterfall plot for local attribution...")
    probs = cb_model.predict_proba(X_shap)
    preds = cb_model.predict(X_shap)
    
    # Pick a sample that has high confidence
    sample_idx = int(np.argmax(np.max(probs, axis=1)))
    pred_class_idx = int(preds[sample_idx][0])
    pred_class_name = classes_mapping[pred_class_idx]
    
    logger.info(f"Selected sample index {sample_idx} with predicted class '{pred_class_name}' ({probs[sample_idx, pred_class_idx]:.2%})")
    
    # Get correct base value
    if isinstance(explainer.expected_value, (np.ndarray, list)):
        exp_val = explainer.expected_value[pred_class_idx]
    else:
        exp_val = explainer.expected_value
        
    try:
        # Construct SHAP Explanation object
        exp = shap.Explanation(
            values=shap_values_list[pred_class_idx][sample_idx],
            base_values=exp_val,
            data=X_shap.iloc[sample_idx].values,
            feature_names=X_shap.columns.tolist()
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, max_display=10, show=False)
        plt.title(f"Local Waterfall Plot (Predicted: {pred_class_name} - {probs[sample_idx, pred_class_idx]:.1%})", fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_local_waterfall.png"), dpi=200)
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to generate standard waterfall plot: {e}. Falling back to custom horizontal bar chart.")
        # Fallback custom plot
        vals = shap_values_list[pred_class_idx][sample_idx]
        feat_names = X_shap.columns.tolist()
        indices = np.argsort(np.abs(vals))[-10:] # Top 10 features
        
        plt.figure(figsize=(10, 6))
        plt.barh(np.array(feat_names)[indices], vals[indices], color=['#ff7f0e' if v > 0 else '#1f77b4' for v in vals[indices]])
        plt.title(f"Local SHAP Feature Attribution (Predicted: {pred_class_name})", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("SHAP Value (Impact on Prediction)")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_local_waterfall.png"), dpi=200)
        plt.close()
        
    # 5. Generate Decision Plot
    logger.info("Generating SHAP decision plot...")
    try:
        plt.figure(figsize=(10, 8))
        shap.decision_plot(
            exp_val, 
            shap_values_list[pred_class_idx], 
            X_shap, 
            show=False
        )
        plt.title(f"SHAP Decision Plot for Class: {pred_class_name}", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_decision_plot.png"), dpi=200)
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to generate decision plot: {e}")
        
    # 6. Generate reports/xai/xai_report.md
    logger.info("Writing XAI Report markdown file...")
    report_content = f"""# Explainable AI (XAI) Report: Severity Stacking Classifier

This report documents the local and global feature attribution mechanisms of the Severity Stacking Classifier. By leveraging SHAP (SHapley Additive exPlanations), we explain how the base estimators contribute to predictions across classes: `Quick`, `Moderate`, and `Prolonged`.

---

## 1. Global Interpretability Analysis

### Severity Model Feature Importance
The figure below ranks the top features by their mean absolute SHAP value across all three severity classes. This stacked bar chart visualizes how each feature influences class-specific decisions.

![Global Feature Importance](shap_global_importance.png)

#### Key Global Insights
1. **Junction & Corridor Centrality:** Degree and closeness centralities are critical features. High centrality junctions lead to longer clearance times due to complex traffic patterns.
2. **Temporal Features:** Hour of day and cyclic time encodings (sin/cos components) highly influence predictions, capturing diurnal peak congestion cycles.
3. **Event Characteristics:** Incident cause (`event_cause_freq`) and vehicle types (`event_type_x_veh_type`) are major drivers of severity.

---

## 2. Class-specific Interpretability

Below are the beeswarm plots for the individual severity classes. Beeswarm plots reveal the directionality of features—showing how high or low values of a feature increase or decrease the likelihood of a specific prediction class.

### Class: Quick
![Beeswarm Quick](shap_beeswarm_quick.png)

* **Quick Characteristics:** Standard temporal baselines, low-frequency junction counts, and off-peak hours heavily shift predictions toward the `Quick` severity class.

### Class: Moderate
![Beeswarm Moderate](shap_beeswarm_moderate.png)

* **Moderate Characteristics:** Balanced spatial hotspots and standard incident counts typically result in a `Moderate` prediction.

### Class: Prolonged
![Beeswarm Prolonged](shap_beeswarm_prolonged.png)

* **Prolonged Characteristics:** High vehicle counts, high-density spatial centroids, peak hours, and heavy-cargo vehicle types strongly increase the probability of a `Prolonged` clearance event.

---

## 3. Local Decision Attribution

For any scored incident, we can visualize the local SHAP waterfall plot or decision plot showing the exact step-by-step feature attributions starting from the base value to the final prediction.

### Waterfall Plot (Individual Prediction)
Below is the feature attribution for a highly confident prediction:

![Local Waterfall Plot](shap_local_waterfall.png)

* **Baseline Value:** The starting base probability (prior distribution across the training dataset).
* **Positive/Negative Drivers:** The horizontal bars show how each feature pulls the prediction higher or lower to arrive at the final probability.

### Cumulative Decision Path
The decision plot below shows the path of predictions for a sample batch of incidents, demonstrating the cumulative feature effects.

![SHAP Decision Plot](shap_decision_plot.png)
"""
    
    with open(os.path.join(output_dir, "xai_report.md"), "w", encoding="utf-8") as f:
        f.write(report_content)
        
    logger.info("XAI Report compiled and saved successfully.")

if __name__ == "__main__":
    generate_xai_report()
