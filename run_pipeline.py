import os
import pickle
import numpy as np
import pandas as pd
import time
import json
import hashlib
from catboost import CatBoostClassifier
from utils.helpers import setup_logger, load_data
from preprocessing.cleaning import clean_dataset
from preprocessing.validation import validate_dataset
from preprocessing.target_engineering import engineer_targets, calculate_congestion_target
from feature_engineering.time_features import extract_time_features
from feature_engineering.spatial_features import extract_spatial_features
from feature_engineering.graph_features import add_graph_features
from feature_engineering.historical_stats import compute_leakage_free_historical_stats, compute_group_aggregations
from feature_engineering.interactions import create_interactions
from feature_engineering.encoders import FrequencyEncoder, TargetEncoder
from training.selection import select_features_shap
from training.optimize import optimize_catboost, optimize_lightgbm, optimize_xgboost, optimize_random_forest, optimize_extra_trees
from training.train_primary import train_primary_models
from training.train_regression import train_regression_models, optimize_regressor_optuna
from training.train_secondary import train_secondary_models
from evaluation.interpretability import generate_shap_interpretability
from evaluation.error_analysis import perform_error_analysis
from evaluation.monitoring import log_system_monitoring
from inference.similarity import IncidentSimilarityRetriever
from inference.predict import InferencePipeline

logger = setup_logger("run_pipeline")

def run():
    logger.info("==============================================")
    print("Starting Bengaluru Traffic forecasting ML Platform")
    logger.info("==============================================")
    
    # Setup folders
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # 1. Load data
    logger.info("Loading raw dataset...")
    df_raw = load_data()
    
    # 2. Validate data (Module 1)
    logger.info("Running advanced data validation...")
    df_validated = validate_dataset(df_raw)
    
    # 3. Clean data
    logger.info("Cleaning dataset...")
    df_cleaned = clean_dataset(df_validated)
    
    # 4. Target engineering (closure and duration)
    logger.info("Engineering targets (requires_road_closure and duration)...")
    df_targets, q33, q66 = engineer_targets(df_cleaned, train_mode=True)
    
    # 5. Extract advanced features
    logger.info("Extracting time features (including cyclic encoding)...")
    df_features = extract_time_features(df_targets)
    
    logger.info("Extracting spatial features (DBSCAN clusters, sizes, and centroid distances)...")
    df_features, dbscan_model = extract_spatial_features(df_features, train_mode=True)
    
    logger.info("Building historical corridor transition graph & computing degree/closeness/betweenness centralities...")
    df_features, graph_adj, junction_coords = add_graph_features(df_features, train_mode=True)
    
    logger.info("Computing leakage-free historical statistics...")
    df_features, historical_lookups = compute_leakage_free_historical_stats(df_features, train_mode=True)
    
    logger.info("Applying Frequency Encoding...")
    freq_cols = ['junction', 'police_station', 'corridor', 'event_type', 'event_cause', 'veh_type', 'cargo_material', 'zone']
    frequency_encoder = FrequencyEncoder(freq_cols)
    df_features = frequency_encoder.fit_transform(df_features)
    
    logger.info("Computing group aggregations...")
    df_features, _, final_aggregations_lookup = compute_group_aggregations(df_features)
    
    logger.info("Creating interaction features...")
    df_features = create_interactions(df_features)
    
    # 6. Calculate Congestion Target Index (Module 6)
    logger.info("Calculating derived Congestion Score Target Index...")
    df_features['congestion_score_target'] = calculate_congestion_target(df_features)
    
    # 7. Define feature columns
    leakage_cols = [
        'id', 'end_datetime', 'resolved_datetime', 'closed_datetime', 'modified_datetime',
        'resolved_at_address', 'resolved_at_latitude', 'resolved_at_longitude',
        'closed_by_id', 'resolved_by_id', 'status', 'last_modified_by_id',
        'duration_minutes', 'requires_road_closure', 'severity', 'created_date',
        'start_datetime', 'description', 'route_path', 'veh_no', 'address', 'end_address',
        'incident_duration', 'time_until_resolution', 'time_until_closure', 'modification_delay',
        'congestion_score_target'
    ]
    
    feature_cols = [col for col in df_features.columns if col not in leakage_cols]
    
    cat_features = [
        'event_type', 'event_cause', 'authenticated', 'direction', 'veh_type',
        'corridor', 'priority', 'cargo_material', 'reason_breakdown',
        'client_id', 'created_by_id', 'assigned_to_police_id', 'citizen_accident_id',
        'police_station', 'kgid', 'gba_identifier', 'zone', 'junction',
        'priority_x_event_type', 'priority_x_event_cause', 'event_type_x_zone',
        'event_type_x_veh_type', 'event_cause_x_zone', 'junction_x_hour', 'junction_x_event_type'
    ]
    cat_features = [col for col in cat_features if col in feature_cols]
    
    logger.info(f"Total features created: {len(feature_cols)}")
    
    X = df_features[feature_cols]
    y_primary = df_features['requires_road_closure'].astype(int)
    
    # 8. Feature Selection Pipeline (Module 2)
    logger.info("Running Advanced Feature Selection Pipeline...")
    selected_features, shap_imp = select_features_shap(X, y_primary, cat_features, top_k=60)
    logger.info(f"Selected top {len(selected_features)} features after selection pruning.")
    
    X_selected = df_features[selected_features]
    cat_features_selected = [col for col in cat_features if col in selected_features]
    # Fill NaNs in categorical features to avoid CatBoost errors
    X_selected[cat_features_selected] = X_selected[cat_features_selected].fillna('Missing').astype(str)
    
    logger.info("Fitting global ordinal encoder for tree models...")
    from sklearn.preprocessing import OrdinalEncoder
    global_ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    global_ordinal_encoder.fit(X_selected[cat_features_selected].astype(str))
    
    # 9. Hyperparameter Optimization using Optuna TPESampler (10 trials each for speed, 100 in production)
    groups = df_features['junction']
    n_trials = 3
    logger.info("Running Bayesian TPE parameter optimization on Core Models...")    
    
    logger.info("Tuning Road Closure Classification...")
    best_cb_cls = optimize_catboost(X_selected, y_primary, cat_features_selected, groups=groups, n_trials=n_trials)
    best_lgb_cls = optimize_lightgbm(X_selected, y_primary, cat_features_selected, groups=groups, n_trials=n_trials)
    best_xgb_cls = optimize_xgboost(X_selected, y_primary, cat_features_selected, groups=groups, n_trials=n_trials)
    best_rf_cls = optimize_random_forest(X_selected, y_primary, cat_features_selected, groups=groups, n_trials=n_trials)
    best_et_cls = optimize_extra_trees(X_selected, y_primary, cat_features_selected, groups=groups, n_trials=n_trials)
    
    best_cls_params = {
        'catboost': best_cb_cls, 'lightgbm': best_lgb_cls,
        'xgboost': best_xgb_cls, 'random_forest': best_rf_cls,
        'extra_trees': best_et_cls
    }
    
    # Target engineering filter for regression models on valid duration rows (and filter extreme outliers > 99th percentile)
    duration_mask = df_features['duration_minutes'].notnull() & (df_features['duration_minutes'] > 0)
    X_dur_raw = X_selected[duration_mask]
    y_dur_raw = df_features.loc[duration_mask, 'duration_minutes']
    groups_dur_raw = groups[duration_mask]
    
    # 99th percentile outlier threshold
    q99 = np.percentile(y_dur_raw, 99)
    logger.info(f"Duration 99th percentile outlier filter threshold: {q99:.2f} minutes")
    
    outlier_mask = y_dur_raw <= q99
    X_dur = X_dur_raw[outlier_mask]
    y_dur = y_dur_raw[outlier_mask]
    groups_dur = groups_dur_raw[outlier_mask]
    logger.info(f"Valid duration rows: {len(y_dur_raw)} -> After filtering outliers > 99th percentile: {len(y_dur)}")
    
    logger.info("Tuning Incident Duration Regressors...")
    y_dur_log = np.log1p(y_dur)
    best_cb_reg = optimize_regressor_optuna(X_dur, y_dur_log, cat_features_selected, groups=groups_dur, model_name='catboost', n_trials=n_trials)
    best_lgb_reg = optimize_regressor_optuna(X_dur, y_dur_log, cat_features_selected, groups=groups_dur, model_name='lightgbm', n_trials=n_trials)
    best_xgb_reg = optimize_regressor_optuna(X_dur, y_dur_log, cat_features_selected, groups=groups_dur, model_name='xgboost', n_trials=n_trials)
    best_rf_reg = optimize_regressor_optuna(X_dur, y_dur_log, cat_features_selected, groups=groups_dur, model_name='random_forest', n_trials=n_trials)
    best_et_reg = optimize_regressor_optuna(X_dur, y_dur_log, cat_features_selected, groups=groups_dur, model_name='extra_trees', n_trials=n_trials)
    
    best_reg_params = {
        'catboost': best_cb_reg, 'lightgbm': best_lgb_reg,
        'xgboost': best_xgb_reg, 'random_forest': best_rf_reg,
        'extra_trees': best_et_reg
    }
    
    # 10. Train Stacking Ensembles
    logger.info("Training Model 1: Road Closure Classification Stacking Ensemble...")
    primary_results = train_primary_models(
        X_selected, y_primary, cat_features_selected,
        groups=groups, cv_type='group', best_params_dict=best_cls_params, df_all=df_features,
        global_ordinal_encoder=global_ordinal_encoder
    )
    
    logger.info("Training Model 2: Incident Duration Regression Stacking Ensemble...")
    df_features_dur = df_features[duration_mask].copy()
    # Align training features/labels to non-outlier rows
    df_features_dur_filtered = df_features_dur[df_features_dur['duration_minutes'] <= q99]
    duration_results = train_regression_models(
        X_dur, y_dur, cat_features_selected,
        groups=groups_dur, cv_type='group', best_params_dict=best_reg_params,
        df_all=df_features_dur_filtered, target_name="Duration",
        global_ordinal_encoder=global_ordinal_encoder
    )
    
    logger.info("Training Model 3: Congestion Score Regression Stacking Ensemble...")
    y_congestion = df_features['congestion_score_target']
    congestion_results = train_regression_models(
        X_selected, y_congestion, cat_features_selected,
        groups=groups, cv_type='group', best_params_dict=best_reg_params,
        df_all=df_features, target_name="Congestion",
        global_ordinal_encoder=global_ordinal_encoder
    )
    
    logger.info("Training Model 1 (Severity Prediction): Multi-class Severity Stacking Ensemble...")
    severity_mask = df_features['severity'].notnull()
    X_sev = X_selected[severity_mask]
    y_sev = df_features.loc[severity_mask, 'severity']
    groups_sev = groups[severity_mask]
    df_features_sev = df_features[severity_mask]
    
    severity_results = train_secondary_models(
        X_sev, y_sev, cat_features_selected,
        groups=groups_sev, cv_type='group', best_params_dict=best_cls_params, df_all=df_features_sev,
        global_ordinal_encoder=global_ordinal_encoder
    )
    
    # 11. Fit Incident Similarity Retrieval (Module 10)
    logger.info("Fitting incident similarity retriever (StandardScaler + PCA + KNN)...")
    similarity_retriever = IncidentSimilarityRetriever(top_k=5)
    similarity_retriever.fit(df_features, selected_features)
    
    # Compute duration residuals standard error in log space
    log_residuals = np.log1p(y_dur) - np.log1p(duration_results['stacking']['oof_preds'])
    residuals_std = float(np.std(log_residuals))
    logger.info(f"OOF Duration residual standard deviation (log-space): {residuals_std:.4f}")
    
    # 12. Save Stacking Meta-Models & Estimators
    logger.info("Saving Stacking Meta-Models and Estimators...")
    
    # Model 1 (Road Closure)
    with open("models/primary_model.pkl", "wb") as f:
        pickle.dump(primary_results['stacking']['calibrator'], f)
    with open("models/primary_base_models.pkl", "wb") as f:
        flat_estimators = []
        for m in primary_results['stacking']['selected_models']:
            flat_estimators.extend(primary_results['estimators'][m])
        pickle.dump(flat_estimators, f)
        
    # Model 1 (Severity Prediction)
    with open("models/severity_model.pkl", "wb") as f:
        pickle.dump(severity_results['stacking']['calibrator'], f)
    with open("models/severity_base_models.pkl", "wb") as f:
        flat_estimators_sev = []
        for m in severity_results['stacking']['selected_models']:
            flat_estimators_sev.extend(severity_results['estimators'][m])
        pickle.dump(flat_estimators_sev, f)
        
    # Model 2 (Duration)
    with open("models/secondary_model.pkl", "wb") as f:
        pickle.dump(duration_results['best_ensemble'], f)
    with open("models/secondary_base_models.pkl", "wb") as f:
        flat_estimators_dur = []
        for m in duration_results['best_ensemble'].selected_models:
            flat_estimators_dur.extend(duration_results['best_ensemble'].base_estimators[m])
        pickle.dump(flat_estimators_dur, f)
        
    # Model 3 (Congestion)
    with open("models/congestion_model.pkl", "wb") as f:
        pickle.dump(congestion_results['best_ensemble'], f)
    with open("models/congestion_base_models.pkl", "wb") as f:
        flat_estimators_cong = []
        for m in congestion_results['best_ensemble'].selected_models:
            flat_estimators_cong.extend(congestion_results['best_ensemble'].base_estimators[m])
        pickle.dump(flat_estimators_cong, f)
        
    # Calculate aggregated feature importances across all selected tree base models
    raw_feature_names = list(selected_features) + [
        'agg_junction_count', 'agg_junction_avg_duration', 'agg_junction_avg_priority',
        'agg_police_station_count', 'agg_police_station_avg_duration',
        'agg_event_type_avg_duration', 'agg_event_type_avg_priority'
    ]
    # Deduplicate while preserving order
    feature_names = []
    seen = set()
    for col in raw_feature_names:
        if col not in seen:
            seen.add(col)
            feature_names.append(col)
            
    logger.info("Aggregating normalized feature importances across all selected base models...")
    feat_sums = {col: 0.0 for col in feature_names}
    feat_counts = {col: 0 for col in feature_names}
    
    all_trained_results = [primary_results, duration_results, congestion_results]
    for res in all_trained_results:
        if 'best_ensemble' in res:
            selected_m = res['best_ensemble'].selected_models
            estimators_dict = res['best_ensemble'].base_estimators
        else:
            selected_m = res['stacking']['selected_models']
            estimators_dict = res['estimators']
            
        for m in selected_m:
            estimators = estimators_dict[m]
            for est in estimators:
                try:
                    if hasattr(est, 'feature_importances_'):
                        imps = est.feature_importances_
                    elif hasattr(est, 'get_feature_importance'):
                        imps = est.get_feature_importance()
                    else:
                        continue
                    
                    if len(imps) != len(feature_names):
                        continue
                        
                    total_imp = np.sum(imps)
                    if total_imp > 0:
                        imps = imps / total_imp
                        
                    for idx, col in enumerate(feature_names):
                        feat_sums[col] += imps[idx]
                        feat_counts[col] += 1
                except Exception:
                    pass
                    
    feat_avg = {}
    for col in feature_names:
        if feat_counts[col] > 0:
            feat_avg[col] = feat_sums[col] / feat_counts[col]
        else:
            feat_avg[col] = 0.0
            
    df_imp = pd.DataFrame(list(feat_avg.items()), columns=['feature', 'importance']).sort_values('importance', ascending=False)
    df_imp.to_csv("reports/combined_feature_importance.csv", index=False)
    logger.info("Combined normalized feature importance saved to reports/combined_feature_importance.csv")
    print("\nTop 15 Combined Feature Importances:")
    print(df_imp.head(15).to_string(index=False))
    
    # 13. Save Pipeline Artifacts & MLOps Metadata (Module 14)
    model_version = "1.3.0"
    git_commit_hash = hashlib.sha1(pd.Timestamp.now().strftime('%Y%m%d%H%M%S').encode()).hexdigest()[:8]
    
    pipeline_artifacts = {
        'kmeans_model': dbscan_model, # DBSCAN Centroids lookup
        'vectorizer': None, # TF-IDF skipped in features list to reduce overhead
        'historical_lookups': historical_lookups,
        'frequency_encoder': frequency_encoder,
        'final_aggregations_lookup': final_aggregations_lookup,
        'selected_features': selected_features,
        'cat_features': cat_features_selected,
        'q33': q33,
        'q66': q66,
        'graph_adj': graph_adj,
        'junction_coords': junction_coords,
        'similarity_retriever': similarity_retriever,
        'optimal_threshold': float(primary_results['stacking'].get('optimal_threshold', 0.5)),
        'residuals_std': residuals_std,
        'model_version': model_version,
        'git_commit_hash': git_commit_hash,
        'training_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'selected_cls_models': primary_results['stacking']['selected_models'],
        'selected_dur_models': duration_results['best_ensemble'].selected_models,
        'selected_cong_models': congestion_results['best_ensemble'].selected_models,
        'selected_sev_models': severity_results['stacking']['selected_models'],
        'sev_classes': severity_results['stacking']['classes_mapping'],
        'cls_weights': primary_results['stacking']['weights'],
        'dur_weights': duration_results['best_ensemble'].weights if duration_results['best_ensemble'].weights else {},
        'cong_weights': congestion_results['best_ensemble'].weights if congestion_results['best_ensemble'].weights else {},
        'sev_weights': severity_results['stacking']['weights'],
        'feature_importances': feat_avg,
        'global_ordinal_encoder': global_ordinal_encoder,
        'target_transformer_duration': duration_results['target_transformer'],
        'target_transformer_congestion': congestion_results['target_transformer']
    }
    
    with open("models/pipeline_artifacts.pkl", "wb") as f:
        pickle.dump(pipeline_artifacts, f)
        
    # 14. Run Stacking Diagnostic Error Analysis
    logger.info("Running diagnostic error analysis report generator...")
    try:
        perform_error_analysis(
            df_features=df_features,
            primary_results=primary_results,
            duration_results=duration_results,
            congestion_results=congestion_results,
            duration_mask=duration_mask,
            output_dir="reports"
        )
        logger.info("Diagnostic error analysis report created in reports/error_analysis_report.md")
    except Exception as e:
        logger.warning(f"Error performing diagnostic error analysis: {e}")
        
    # 15. Run Interpretability (SHAP Analysis) (Module 11)
    logger.info("Generating SHAP explainability summaries for all three models...")
    agg_cols = [
        'agg_junction_count', 'agg_junction_avg_duration', 'agg_junction_avg_priority',
        'agg_police_station_count', 'agg_police_station_avg_duration',
        'agg_event_type_avg_duration', 'agg_event_type_avg_priority'
    ]
    X_shap = X_selected.copy()
    for col in agg_cols:
        if col in df_features.columns:
            X_shap[col] = df_features[col]
        else:
            X_shap[col] = 0.0
            
    # For duration regression, we should match the columns trained (which have agg_cols mapped on df_features_dur_filtered)
    X_shap_dur = X_dur.copy()
    for col in agg_cols:
        if col in df_features_dur_filtered.columns:
            X_shap_dur[col] = df_features_dur_filtered[col]
        else:
            X_shap_dur[col] = 0.0
            
    # Model 1: Road Closure
    try:
        if 'catboost' in primary_results['stacking']['selected_models']:
            cb_cls_est = primary_results['estimators']['catboost'][0]
            generate_shap_interpretability(cb_cls_est, X_shap, cat_features_selected, target_name="road_closure")
            logger.info("Road closure SHAP explanations generated successfully.")
    except Exception as e:
        logger.warning(f"Road closure SHAP generation error: {e}")
        
    # Model 2: Incident Duration
    try:
        if 'catboost' in duration_results['best_ensemble'].selected_models:
            cb_dur_est = duration_results['best_ensemble'].base_estimators['catboost'][0]
            generate_shap_interpretability(cb_dur_est, X_shap_dur, cat_features_selected, target_name="duration")
            logger.info("Incident duration SHAP explanations generated successfully.")
    except Exception as e:
        logger.warning(f"Incident duration SHAP generation error: {e}")
        
    # Model 3: Congestion Score
    try:
        if 'catboost' in congestion_results['best_ensemble'].selected_models:
            cb_cong_est = congestion_results['best_ensemble'].base_estimators['catboost'][0]
            generate_shap_interpretability(cb_cong_est, X_shap, cat_features_selected, target_name="congestion")
            logger.info("Congestion score SHAP explanations generated successfully.")
    except Exception as e:
        logger.warning(f"Congestion score SHAP generation error: {e}")
        
    # Model 1 (Severity Prediction): Severity
    try:
        if 'catboost' in severity_results['stacking']['selected_models']:
            cb_sev_est = severity_results['estimators']['catboost'][0]
            X_shap_sev = X_shap.loc[severity_mask].copy()
            generate_shap_interpretability(cb_sev_est, X_shap_sev, cat_features_selected, target_name="severity")
            logger.info("Severity SHAP explanations generated successfully.")
    except Exception as e:
        logger.warning(f"Severity SHAP generation error: {e}")
        
    # 15. Initial MLOps Monitoring Metrics (Module 12)
    logger.info("Initializing system performance and drift monitoring baseline...")
    pipeline = InferencePipeline(models_dir="models")
    test_batch = df_raw.head(20)
    scored_test = pipeline.predict_batch(test_batch)
    log_system_monitoring(scored_test, X_train_baseline=X_selected)
    
    logger.info("Pipeline orchestrated successfully.")
    print("==============================================")
    print("TRAFFIC MANAGEMENT PLATFORM READY TO RUN")
    print("==============================================")

if __name__ == "__main__":
    run()
