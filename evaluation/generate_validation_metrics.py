import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, classification_report, brier_score_loss

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.helpers import setup_logger, load_data
from preprocessing.cleaning import clean_dataset
from preprocessing.validation import validate_dataset
from preprocessing.target_engineering import engineer_targets, calculate_congestion_target
from feature_engineering.time_features import extract_time_features
from feature_engineering.spatial_features import extract_spatial_features
from feature_engineering.graph_features import add_graph_features
from feature_engineering.historical_stats import compute_leakage_free_historical_stats, compute_group_aggregations
from feature_engineering.interactions import create_interactions
from utils.calibration import CalibratorWrapper, MulticlassCalibratorWrapper

logger = setup_logger("generate_validation_metrics")

def compute_multiclass_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) for multi-class classification.
    We bin samples by their confidence score (maximum predicted probability).
    """
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (y_pred == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Samples in the current bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece

def compute_multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes the multi-class Brier score.
    """
    n_classes = y_prob.shape[1]
    # One-hot encode true targets
    y_one_hot = np.zeros_like(y_prob)
    y_one_hot[np.arange(len(y_true)), y_true] = 1.0
    # Mean squared error
    return float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))

def run_validation_evaluation():
    logger.info("Initializing Validation Evaluation Pipeline...")
    os.makedirs("reports", exist_ok=True)
    
    # 1. Load pipeline artifacts and models
    logger.info("Loading models and artifacts...")
    artifacts_path = "models/pipeline_artifacts.pkl"
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Pipeline artifacts not found at {artifacts_path}. Please run pipeline first.")
        
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)
        
    selected_features = artifacts['selected_features']
    cat_features = artifacts['cat_features']
    selected_sev_models = artifacts.get('selected_sev_models', ['catboost', 'lightgbm', 'xgboost', 'random_forest', 'extra_trees'])
    classes_mapping = artifacts.get('sev_classes', ['Moderate', 'Prolonged', 'Quick'])
    n_classes = len(classes_mapping)
    
    logger.info(f"Loaded classes: {classes_mapping}, selected models for severity: {selected_sev_models}")
    
    # Load severity models
    with open("models/severity_model.pkl", "rb") as f:
        severity_meta_model = pickle.load(f)
    with open("models/severity_base_models.pkl", "rb") as f:
        severity_base_models = pickle.load(f)
        
    # 2. Re-engineer features to match the exact training set
    logger.info("Processing dataset features...")
    df_raw = load_data()
    df_validated = validate_dataset(df_raw)
    df_cleaned = clean_dataset(df_validated)
    df_targets, q33, q66 = engineer_targets(df_cleaned, train_mode=True)
    df_features = extract_time_features(df_targets)
    df_features, dbscan_model = extract_spatial_features(df_features, train_mode=True)
    df_features, graph_adj, junction_coords = add_graph_features(df_features, train_mode=True)
    df_features, historical_lookups = compute_leakage_free_historical_stats(df_features, train_mode=True)
    
    freq_cols = ['junction', 'police_station', 'corridor', 'event_type', 'event_cause', 'veh_type', 'cargo_material', 'zone']
    frequency_encoder = artifacts.get('frequency_encoder')
    if frequency_encoder is not None:
        df_features = frequency_encoder.transform(df_features)
    else:
        frequency_encoder = FrequencyEncoder(freq_cols)
        df_features = frequency_encoder.fit_transform(df_features)
        
    df_features, _, _ = compute_group_aggregations(df_features)
    df_features = create_interactions(df_features)
    df_features['congestion_score_target'] = calculate_congestion_target(df_features)
    
    # Align features to model expectations
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
        
    # Get severity records
    severity_mask = df_features['severity'].notnull()
    X_sev = df_features.loc[severity_mask, feature_cols].copy()
    y_sev = df_features.loc[severity_mask, 'severity']
    groups = df_features.loc[severity_mask, 'junction']
    
    # Impute missing values
    num_cols = X_sev.select_dtypes(include=['number']).columns
    for col in num_cols:
        X_sev[col] = X_sev[col].fillna(0.0)
        
    # Target encoding
    y_encoded = y_sev.map({c: i for i, c in enumerate(classes_mapping)}).astype(int)
    
    # 3. Reconstruct Out-Of-Fold predictions
    logger.info("Reconstructing Out-of-Fold severity predictions from fold models...")
    cv = GroupKFold(n_splits=5)
    splits = list(cv.split(X_sev, y_encoded, groups))
    
    oof_probs_matrix = np.zeros((len(y_encoded), len(selected_sev_models) * n_classes))
    
    for m_idx, m_name in enumerate(selected_sev_models):
        logger.info(f"Reconstructing predictions for base estimator: {m_name}")
        oof_preds = np.zeros((len(y_encoded), n_classes))
        
        # Load the 5 fold estimators for this model
        start_idx = m_idx * 5
        end_idx = start_idx + 5
        fold_models = severity_base_models[start_idx:end_idx]
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_val = X_sev.iloc[val_idx].copy()
            
            # Format inputs
            if m_name == 'catboost':
                for col in cat_features:
                    X_val[col] = X_val[col].astype(str)
                X_val_proc = X_val
            else:
                X_val_proc = X_val.copy()
                if cat_features:
                    global_ordinal_encoder = artifacts.get('global_ordinal_encoder')
                    if global_ordinal_encoder is not None:
                        X_val_proc[cat_features] = global_ordinal_encoder.transform(X_val_proc[cat_features].astype(str))
                    else:
                        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        X_val_proc[cat_features] = oe.fit_transform(X_val_proc[cat_features].astype(str))
                    X_val_proc[cat_features] = X_val_proc[cat_features].fillna(-1)
            
            oof_preds[val_idx] = fold_models[fold].predict_proba(X_val_proc)
            
        start_col = m_idx * n_classes
        end_col = start_col + n_classes
        oof_probs_matrix[:, start_col:end_col] = oof_preds
        
    # Apply Meta-Model to generate stacked probabilities
    stacked_probs = severity_meta_model.predict_proba(oof_probs_matrix)
    stacked_preds = np.argmax(stacked_probs, axis=1)
    
    # 4. Generate Reports and Visualizations
    logger.info("Computing metrics and plotting results...")
    
    # Classification report
    report_dict = classification_report(y_encoded, stacked_preds, target_names=classes_mapping, output_dict=True)
    report_str = classification_report(y_encoded, stacked_preds, target_names=classes_mapping)
    print("\n--- Severity Classification Report ---")
    print(report_str)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_encoded, stacked_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot Side-by-Side Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Raw CM
    im = axes[0].imshow(cm, cmap='Blues', interpolation='nearest')
    axes[0].set_title('Raw Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xticks(np.arange(n_classes))
    axes[0].set_yticks(np.arange(n_classes))
    axes[0].set_xticklabels(classes_mapping, fontsize=11)
    axes[0].set_yticklabels(classes_mapping, fontsize=11)
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='semibold', labelpad=10)
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='semibold', labelpad=10)
    
    # Text annotations for Raw
    for i in range(n_classes):
        for j in range(n_classes):
            axes[0].text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > cm.max()/2.0 else "black",
                         fontsize=12, fontweight='bold')
            
    # Normalized CM
    im2 = axes[1].imshow(cm_norm, cmap='Oranges', interpolation='nearest')
    axes[1].set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xticks(np.arange(n_classes))
    axes[1].set_yticks(np.arange(n_classes))
    axes[1].set_xticklabels(classes_mapping, fontsize=11)
    axes[1].set_yticklabels(classes_mapping, fontsize=11)
    axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='semibold', labelpad=10)
    axes[1].set_ylabel('True Label', fontsize=12, fontweight='semibold', labelpad=10)
    
    # Text annotations for Normalized
    for i in range(n_classes):
        for j in range(n_classes):
            axes[1].text(j, i, format(cm_norm[i, j], '.2%'),
                         ha="center", va="center",
                         color="white" if cm_norm[i, j] > cm_norm.max()/2.0 else "black",
                         fontsize=12, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved reports/confusion_matrix.png")
    
    # 5. Model Calibration Curves (Reliability Diagram)
    ece = compute_multiclass_ece(y_encoded.values, stacked_probs, n_bins=10)
    brier = compute_multiclass_brier_score(y_encoded.values, stacked_probs)
    
    logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")
    logger.info(f"Multi-class Brier Score: {brier:.4f}")
    
    # Plot Reliability Diagrams
    # We will plot calibration curves for each class (Quick, Moderate, Prolonged)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot 1: Class-specific Calibration Curves
    for c_idx, c_name in enumerate(classes_mapping):
        y_true_binary = (y_encoded == c_idx).astype(int)
        y_prob_binary = stacked_probs[:, c_idx]
        
        # Calculate reliability diagram bins
        bin_boundaries = np.linspace(0, 1, 11)
        bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
        
        emp_acc = []
        pred_conf = []
        
        for i in range(10):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i+1]
            mask = (y_prob_binary > lower) & (y_prob_binary <= upper)
            if np.sum(mask) > 0:
                emp_acc.append(np.mean(y_true_binary[mask]))
                pred_conf.append(np.mean(y_prob_binary[mask]))
            else:
                emp_acc.append(np.nan)
                pred_conf.append(bin_centers[i])
                
        axes[0].plot(pred_conf, emp_acc, 'o-', label=f'{c_name} (Brier: {brier_score_loss(y_true_binary, y_prob_binary):.4f})', color=colors[c_idx], linewidth=2)
        
    axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[0].set_xlabel('Predicted Probability', fontsize=12, labelpad=10)
    axes[0].set_ylabel('Empirical Accuracy', fontsize=12, labelpad=10)
    axes[0].set_title('One-vs-Rest Reliability Curves', fontsize=14, fontweight='bold', pad=15)
    axes[0].legend(fontsize=10, loc='upper left')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Plot 2: Probability Distribution Histogram
    for c_idx, c_name in enumerate(classes_mapping):
        axes[1].hist(stacked_probs[:, c_idx], bins=15, alpha=0.5, label=c_name, color=colors[c_idx], density=True)
        
    axes[1].set_xlabel('Predicted Probability', fontsize=12, labelpad=10)
    axes[1].set_ylabel('Density', fontsize=12, labelpad=10)
    axes[1].set_title('Severity Class Probability Distributions', fontsize=14, fontweight='bold', pad=15)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle(f"Severity Model Stacking Calibration (ECE = {ece:.4f}, Multi-class Brier = {brier:.4f})", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("reports/calibration_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved reports/calibration_curve.png")
    
    # Save a validation metrics summary JSON for automated reporting
    validation_stats = {
        'ece': float(ece),
        'brier': float(brier),
        'accuracy': float(report_dict['accuracy']),
        'macro_f1': float(report_dict['macro avg']['f1-score']),
        'weighted_f1': float(report_dict['weighted avg']['f1-score']),
        'classes_mapping': classes_mapping,
        'class_wise': {}
    }
    
    for c_name in classes_mapping:
        validation_stats['class_wise'][c_name] = {
            'precision': float(report_dict[c_name]['precision']),
            'recall': float(report_dict[c_name]['recall']),
            'f1-score': float(report_dict[c_name]['f1-score']),
            'support': int(report_dict[c_name]['support'])
        }
        
    stats_path = "models/walkthrough_stats.pkl"
    stats_dict = {}
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "rb") as f:
                stats_dict = pickle.load(f)
        except Exception:
            pass
            
    stats_dict.update(validation_stats)
    
    with open(stats_path, "wb") as f:
        pickle.dump(stats_dict, f)
        
    logger.info("Saved and merged validation_stats to models/walkthrough_stats.pkl")

if __name__ == "__main__":
    run_validation_evaluation()
