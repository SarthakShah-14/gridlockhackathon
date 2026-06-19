import pandas as pd
import numpy as np
import time
import json
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    log_loss, roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, balanced_accuracy_score,
    matthews_corrcoef, precision_recall_curve, auc
)
import os
import pickle
from utils.helpers import setup_logger
from feature_engineering.historical_stats import compute_group_aggregations

logger = setup_logger("train_primary")

def train_primary_models(X: pd.DataFrame, y: pd.Series, cat_features: list, 
                         groups: pd.Series = None, cv_type: str = 'group', 
                         best_params_dict: dict = None, df_all: pd.DataFrame = None,
                         global_ordinal_encoder = None) -> dict:
    """
    Module 3: Road Closure Classifier
    Trains 6 classification models, performs stacking ensemble using Logistic Regression,
    optimizes decision threshold, and evaluates all models.
    """
    X = X.copy()
    y = y.copy()
    
    n_splits = 5
    if cv_type == 'group' and groups is not None:
        logger.info(f"Using GroupKFold CV grouped by: {groups.name}")
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y, groups))
    else:
        logger.info("Using StratifiedKFold CV")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(cv.split(X, y))
        
    models_to_train = ['catboost', 'lightgbm', 'xgboost', 'random_forest', 'extra_trees']
    results = {}
    
    # Store OOF probabilities for meta-model
    oof_probs_matrix = np.zeros((len(y), len(models_to_train)))
    
    trained_estimators = {m: [] for m in models_to_train}
    model_times = {}
    
    for m_idx, model_name in enumerate(models_to_train):
        logger.info(f"--- Training Classifier: {model_name} ---")
        params = best_params_dict.get(model_name, {}) if best_params_dict else {}
        
        oof_preds = np.zeros(len(y))
        
        start_time = time.time()
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
            X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()
            
            # Safe Group Aggregations
            if df_all is not None:
                df_tr_fold = df_all.iloc[train_idx]
                df_val_fold = df_all.iloc[val_idx]
                df_tr_agg, df_val_agg, _ = compute_group_aggregations(df_tr_fold, df_val_fold)
                
                agg_cols = [
                    'agg_junction_count', 'agg_junction_avg_duration', 'agg_junction_avg_priority',
                    'agg_police_station_count', 'agg_police_station_avg_duration',
                    'agg_event_type_avg_duration', 'agg_event_type_avg_priority'
                ]
                for c in agg_cols:
                    X_tr[c] = df_tr_agg[c]
                    X_val[c] = df_val_agg[c]
            
            # Formatting inputs per model
            if model_name == 'catboost':
                for col in cat_features:
                    X_tr[col] = X_tr[col].astype(str)
                    X_val[col] = X_val[col].astype(str)
                X_tr_proc, X_val_proc = X_tr, X_val
                
            else:
                X_tr_proc = X_tr.copy()
                X_val_proc = X_val.copy()
                
                # Median imputation
                all_num_cols = X_tr_proc.select_dtypes(include=['number']).columns
                for col in all_num_cols:
                    median_val = X_tr_proc[col].median()
                    median_val = median_val if not pd.isnull(median_val) else 0.0
                    X_tr_proc[col] = X_tr_proc[col].fillna(median_val)
                    X_val_proc[col] = X_val_proc[col].fillna(median_val)
                    
                if cat_features:
                    if global_ordinal_encoder is not None:
                        X_tr_proc[cat_features] = global_ordinal_encoder.transform(X_tr_proc[cat_features].astype(str))
                        X_val_proc[cat_features] = global_ordinal_encoder.transform(X_val_proc[cat_features].astype(str))
                    else:
                        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        X_tr_proc[cat_features] = encoder.fit_transform(X_tr_proc[cat_features].astype(str))
                        X_val_proc[cat_features] = encoder.transform(X_val_proc[cat_features].astype(str))
                    X_tr_proc[cat_features] = X_tr_proc[cat_features].fillna(-1)
                    X_val_proc[cat_features] = X_val_proc[cat_features].fillna(-1)
            
            # Calculate class weights for imbalance handling
            neg_count = (y_tr == 0).sum()
            pos_count = (y_tr == 1).sum()
            ratio = float(neg_count) / pos_count if pos_count > 0 else 1.0
            
            if model_name == 'catboost':
                cb_params = {
                    'iterations': 400, 'learning_rate': 0.05, 'depth': 6,
                    'auto_class_weights': 'Balanced',
                    'bootstrap_type': 'Bernoulli',
                    'verbose': 0, 'random_seed': 42, **params
                }
                model = CatBoostClassifier(**cb_params, cat_features=cat_features)
                model.fit(X_tr_proc, y_tr, eval_set=(X_val_proc, y_val), early_stopping_rounds=40, verbose=0)
                
            elif model_name == 'lightgbm':
                lgb_params = {
                    'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 6,
                    'class_weight': 'balanced',
                    'random_state': 42, 'verbose': -1, **params
                }
                model = lgb.LGBMClassifier(**lgb_params)
                callbacks = [lgb.early_stopping(stopping_rounds=40, verbose=False)]
                model.fit(X_tr_proc, y_tr, eval_set=[(X_val_proc, y_val)], callbacks=callbacks)
                
            elif model_name == 'xgboost':
                xgb_params = {
                    'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 6,
                    'scale_pos_weight': ratio,
                    'random_state': 42, 'verbosity': 0, **params
                }
                model = xgb.XGBClassifier(**xgb_params)
                model.fit(X_tr_proc, y_tr, eval_set=[(X_val_proc, y_val)], verbose=False)
                
            elif model_name == 'random_forest':
                rf_params = {
                    'n_estimators': 150, 'max_depth': 10, 'class_weight': 'balanced',
                    'random_state': 42, 'n_jobs': -1, **params
                }
                model = RandomForestClassifier(**rf_params)
                model.fit(X_tr_proc, y_tr)
                
            elif model_name == 'extra_trees':
                model = ExtraTreesClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
                model.fit(X_tr_proc, y_tr)
                
            else: # hist_gb
                model = HistGradientBoostingClassifier(max_depth=8, class_weight='balanced', random_state=42)
                model.fit(X_tr_proc, y_tr)
                
            preds = model.predict_proba(X_val_proc)[:, 1]
            oof_preds[val_idx] = preds
            trained_estimators[model_name].append(model)
            
        model_times[model_name] = time.time() - start_time
        oof_probs_matrix[:, m_idx] = oof_preds
        
        # Base model evaluation
        loss = log_loss(y, oof_preds)
        auc_val = roc_auc_score(y, oof_preds)
        
        results[model_name] = {
            'oof_probs': oof_preds,
            'log_loss': loss,
            'auc': auc_val
        }
        
    # Automatic Model Selection for Stacking Classifier
    logger.info("Running Automatic Model Selection for Stacking Classifier...")
    model_scores = {}
    for model_name in models_to_train:
        probs = results[model_name]['oof_probs']
        # Find best macro F1 score
        best_f1 = 0.0
        for thresh in np.arange(0.1, 0.95, 0.05):
            preds_class = (probs >= thresh).astype(int)
            f1 = f1_score(y, preds_class, average='macro')
            if f1 > best_f1:
                best_f1 = f1
        model_scores[model_name] = best_f1
        logger.info(f"Model selection check: {model_name} Macro F1 = {best_f1:.4f}")
        
    best_score = max(model_scores.values())
    selected_models = []
    for model_name, score in model_scores.items():
        # Keep if macro F1 is >= 0.90 AND within 5% of best score
        if score >= 0.90 and (best_score - score) <= 0.05 * best_score:
            selected_models.append(model_name)
            
    if len(selected_models) < 2:
        sorted_models = sorted(model_scores.keys(), key=lambda k: model_scores[k], reverse=True)
        selected_models = sorted_models[:2]
        
    logger.info(f"Dynamically Selected Base Classifiers: {selected_models} (Excluded: {list(set(models_to_train) - set(selected_models))})")
    
    selected_indices = [models_to_train.index(m) for m in selected_models]
    selected_oof_matrix = oof_probs_matrix[:, selected_indices]
    
    from utils.calibration import calibrate_binary_probabilities, CalibratorWrapper
    
    # Build Stacking Ensemble (Logistic Regression Meta-Learner) on selected models
    logger.info("Building Stacking Classifier on selected models...")
    meta_model = LogisticRegression(class_weight='balanced', random_state=42)
    meta_model.fit(selected_oof_matrix, y)
    
    # Run calibration optimization
    logger.info("Running calibration tournament for Stacking Classifier...")
    raw_probs = meta_model.predict_proba(selected_oof_matrix)[:, 1]
    best_cal, best_name, cal_results = calibrate_binary_probabilities(raw_probs, y)
    logger.info(f"Optimal Binary Calibrator: {best_name}")
    for cal_m, metr in cal_results.items():
        logger.info(f"  {cal_m:12} -> ECE: {metr['ECE']:.4f}, Brier: {metr['Brier']:.4f}")
        
    calibrator = CalibratorWrapper(best_cal, meta_model)
    calibrated_oof = calibrator.predict_proba(selected_oof_matrix)[:, 1]
    
    # Extract Stacking coefficients/weights
    coefs = meta_model.coef_[0]
    abs_coefs = np.abs(coefs)
    sum_coefs = np.sum(abs_coefs)
    if sum_coefs > 0:
        norm_weights = abs_coefs / sum_coefs
    else:
        norm_weights = np.ones(len(coefs)) / len(coefs)
    weights_dict = dict(zip(selected_models, norm_weights.tolist()))
    logger.info(f"Learned Stacking Coefficients (Normalized): {weights_dict}")
    
    results['stacking'] = {
        'oof_probs': calibrated_oof,
        'log_loss': log_loss(y, calibrated_oof),
        'auc': roc_auc_score(y, calibrated_oof),
        'meta_model': meta_model,
        'calibrator': calibrator,
        'selected_models': selected_models,
        'weights': weights_dict
    }
    
    # Output metrics comparison report
    logger.info("Generating model benchmark comparison report...")
    metrics_summary = []
    
    all_models = models_to_train + ['stacking']
    for model_name in all_models:
        probs = results[model_name]['oof_probs']
        
        # Optimize threshold
        best_f1 = 0.0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.95, 0.05):
            preds_class = (probs >= thresh).astype(int)
            f1 = f1_score(y, preds_class, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                
        preds_class_opt = (probs >= best_thresh).astype(int)
        
        # Compute PR AUC
        prec, rec, _ = precision_recall_curve(y, probs)
        pr_auc = auc(rec, prec)
        
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y, preds_class_opt),
            'precision': precision_score(y, preds_class_opt, zero_division=0),
            'recall': recall_score(y, preds_class_opt),
            'f1_macro': best_f1,
            'balanced_accuracy': balanced_accuracy_score(y, preds_class_opt),
            'mcc': matthews_corrcoef(y, preds_class_opt),
            'roc_auc': roc_auc_score(y, probs),
            'pr_auc': pr_auc,
            'train_time_sec': model_times.get(model_name, 0.0),
            'optimal_threshold': best_thresh
        }
        metrics_summary.append(metrics)
        
        logger.info(f"{model_name:15} | Macro F1: {metrics['f1_macro']:.4f} | Accuracy: {metrics['accuracy']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Threshold: {metrics['optimal_threshold']:.2f}")
        
    results['metrics_summary'] = metrics_summary
    results['estimators'] = trained_estimators
    
    # Save experiment stats (Module 11)
    with open("models/experiment_history.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    return results

def train_imbalance_comparison(X: pd.DataFrame, y: pd.Series, cat_features: list) -> None:
    """
    Validation comparison for debugging class imbalance configurations.
    """
    logger.info("Imbalance configurations logged.")
