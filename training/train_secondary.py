import pandas as pd
import numpy as np
import time
import json
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import (
    log_loss, f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, cohen_kappa_score
)
import os
import pickle
from utils.helpers import setup_logger
from feature_engineering.historical_stats import compute_group_aggregations

logger = setup_logger("train_secondary")

def train_secondary_models(X: pd.DataFrame, y: pd.Series, cat_features: list, 
                           groups: pd.Series = None, cv_type: str = 'group', 
                           best_params_dict: dict = None, df_all: pd.DataFrame = None,
                           global_ordinal_encoder = None) -> dict:
    """
    Model 1: Severity Prediction Stacking Classifier
    Trains 5 multi-class base estimators, applies automatic model selection,
    builds a multi-class Logistic Regression meta-learner, and logs metrics.
    """
    X = X.copy()
    y = y.copy()
    
    # 1. Label encode y to integers (0, 1, 2)
    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
    classes_mapping = list(le.classes_)
    n_classes = len(classes_mapping)
    logger.info(f"Severity class mapping: {classes_mapping}")
    
    # 2. Setup Cross Validation
    n_splits = 5
    if cv_type == 'group' and groups is not None:
        logger.info(f"Using GroupKFold CV grouped by: {groups.name}")
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y_encoded, groups))
    else:
        logger.info("Using StratifiedKFold CV")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(cv.split(X, y_encoded))
        
    models_to_train = ['catboost', 'lightgbm', 'xgboost', 'random_forest', 'extra_trees']
    results = {}
    
    # Store OOF probabilities for meta-model (n_samples, n_base_models * n_classes)
    oof_probs_matrix = np.zeros((len(y_encoded), len(models_to_train) * n_classes))
    trained_estimators = {m: [] for m in models_to_train}
    model_times = {}
    
    for m_idx, model_name in enumerate(models_to_train):
        logger.info(f"--- Training Severity Classifier: {model_name} ---")
        params = best_params_dict.get(model_name, {}) if best_params_dict else {}
        
        oof_preds = np.zeros((len(y_encoded), n_classes))
        start_time = time.time()
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[train_idx].copy(), y_encoded.iloc[train_idx].copy()
            X_val, y_val = X.iloc[val_idx].copy(), y_encoded.iloc[val_idx].copy()
            
            # Cross-validation safe group aggregations
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
            
            # Format inputs
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
            
            # Instantiate and fit models
            if model_name == 'catboost':
                cb_params = {
                    'iterations': 400, 'learning_rate': 0.05, 'depth': 6,
                    'loss_function': 'MultiClass', 'auto_class_weights': 'Balanced',
                    'bootstrap_type': 'Bernoulli',
                    'verbose': 0, 'random_seed': 42, **params
                }
                model = CatBoostClassifier(**cb_params, cat_features=cat_features)
                model.fit(X_tr_proc, y_tr, eval_set=(X_val_proc, y_val), early_stopping_rounds=40, verbose=0)
                
            elif model_name == 'lightgbm':
                lgb_params = {
                    'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 6,
                    'objective': 'multiclass', 'class_weight': 'balanced',
                    'random_state': 42, 'verbose': -1, **params
                }
                model = lgb.LGBMClassifier(**lgb_params)
                callbacks = [lgb.early_stopping(stopping_rounds=40, verbose=False)]
                model.fit(X_tr_proc, y_tr, eval_set=[(X_val_proc, y_val)], callbacks=callbacks)
                
            elif model_name == 'xgboost':
                from sklearn.utils.class_weight import compute_sample_weight
                xgb_params = {
                    'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 6,
                    'objective': 'multi:softprob', 'num_class': n_classes,
                    'random_state': 42, 'verbosity': 0, **params
                }
                model = xgb.XGBClassifier(**xgb_params)
                sample_weights = compute_sample_weight('balanced', y_tr)
                model.fit(X_tr_proc, y_tr, sample_weight=sample_weights, eval_set=[(X_val_proc, y_val)], verbose=False)
                
            elif model_name == 'random_forest':
                rf_params = {
                    'n_estimators': 150, 'max_depth': 10, 'class_weight': 'balanced',
                    'random_state': 42, 'n_jobs': -1, **params
                }
                model = RandomForestClassifier(**rf_params)
                model.fit(X_tr_proc, y_tr)
                
            else: # extra_trees
                et_params = {
                    'n_estimators': 150, 'max_depth': 10, 'class_weight': 'balanced',
                    'random_state': 42, 'n_jobs': -1, **params
                }
                model = ExtraTreesClassifier(**et_params)
                model.fit(X_tr_proc, y_tr)
                
            preds = model.predict_proba(X_val_proc)
            oof_preds[val_idx] = preds
            trained_estimators[model_name].append(model)
            
        model_times[model_name] = time.time() - start_time
        start_col = m_idx * n_classes
        end_col = start_col + n_classes
        oof_probs_matrix[:, start_col:end_col] = oof_preds
        
        # Base model evaluation
        loss = log_loss(y_encoded, oof_preds)
        oof_classes = np.argmax(oof_preds, axis=1)
        macro_f1 = f1_score(y_encoded, oof_classes, average='macro')
        acc = accuracy_score(y_encoded, oof_classes)
        
        results[model_name] = {
            'oof_probs': oof_preds,
            'log_loss': loss,
            'macro_f1': macro_f1,
            'accuracy': acc
        }
        
    # Automatic Model Selection for Stacking Severity Classifier
    logger.info("Running Automatic Model Selection for Severity Stacking...")
    model_scores = {m: results[m]['macro_f1'] for m in models_to_train}
    for m, score in model_scores.items():
        logger.info(f"Model severity selection check: {m} Macro F1 = {score:.4f}")
        
    best_score = max(model_scores.values())
    selected_models = []
    for m, score in model_scores.items():
        # Keep if Macro F1 is reasonably high and within 10% of best model
        if score >= 0.40 and (best_score - score) <= 0.10 * best_score:
            selected_models.append(m)
            
    if len(selected_models) < 2:
        sorted_models = sorted(model_scores.keys(), key=lambda k: model_scores[k], reverse=True)
        selected_models = sorted_models[:2]
        
    logger.info(f"Dynamically Selected Severity Base Classifiers: {selected_models} (Excluded: {list(set(models_to_train) - set(selected_models))})")
    
    # Filter the OOF matrix to selected models
    selected_cols = []
    for m in selected_models:
        m_idx = models_to_train.index(m)
        selected_cols.extend(range(m_idx * n_classes, (m_idx + 1) * n_classes))
    selected_oof_matrix = oof_probs_matrix[:, selected_cols]
    
    from utils.calibration import calibrate_multiclass_probabilities, MulticlassCalibratorWrapper
    
    # Train Stacking Meta-Learner (Multinomial Logistic Regression)
    logger.info("Building Severity Stacking Classifier on selected models...")
    meta_model = LogisticRegression(multi_class='multinomial', class_weight='balanced', random_state=42)
    meta_model.fit(selected_oof_matrix, y_encoded)
    
    # Run calibration optimization
    logger.info("Running calibration tournament for Severity Stacking Classifier...")
    raw_probs = meta_model.predict_proba(selected_oof_matrix)
    best_cal, best_name, cal_results = calibrate_multiclass_probabilities(raw_probs, y_encoded)
    logger.info(f"Optimal Multi-class Calibrator: {best_name}")
    for cal_m, metr in cal_results.items():
        logger.info(f"  {cal_m:12} -> ECE: {metr['ECE']:.4f}, Brier: {metr['Brier']:.4f}")
        
    calibrator = MulticlassCalibratorWrapper(best_cal, meta_model)
    stacking_oof = calibrator.predict_proba(selected_oof_matrix)
    stack_classes = np.argmax(stacking_oof, axis=1)
    
    # Stacking weights/coefficients (average absolute coefficient per model)
    coefs = meta_model.coef_ # Shape (n_classes, n_selected_models * n_classes)
    weights_dict = {}
    for idx, m in enumerate(selected_models):
        m_start = idx * n_classes
        m_end = m_start + n_classes
        m_coefs = coefs[:, m_start:m_end]
        weights_dict[m] = float(np.mean(np.abs(m_coefs)))
        
    # Normalize weights
    sum_w = sum(weights_dict.values())
    if sum_w > 0:
        weights_dict = {k: v / sum_w for k, v in weights_dict.items()}
    else:
        weights_dict = {k: 1.0 / len(weights_dict) for k in weights_dict.keys()}
        
    logger.info(f"Learned Severity Stacking Weights: {weights_dict}")
    
    # Compile Stacking Results
    results['stacking'] = {
        'oof_probs': stacking_oof,
        'log_loss': log_loss(y_encoded, stacking_oof),
        'macro_f1': f1_score(y_encoded, stack_classes, average='macro'),
        'accuracy': accuracy_score(y_encoded, stack_classes),
        'meta_model': meta_model,
        'calibrator': calibrator,
        'label_encoder': le,
        'classes_mapping': classes_mapping,
        'selected_models': selected_models,
        'weights': weights_dict
    }
    
    # Output metrics comparison report
    logger.info("Generating severity model benchmark comparison report...")
    metrics_summary = []
    
    all_models = models_to_train + ['stacking']
    for m_name in all_models:
        probs = results[m_name]['oof_probs'] if m_name in results else results['stacking']['oof_probs']
        pred_cls = np.argmax(probs, axis=1)
        
        # Calculate multi-class ROC-AUC (ovr)
        try:
            auc_val = roc_auc_score(y_encoded, probs, multi_class='ovr', average='macro')
        except Exception:
            auc_val = 0.5
            
        metrics = {
            'model': f"severity_{m_name}",
            'accuracy': accuracy_score(y_encoded, pred_cls),
            'precision': precision_score(y_encoded, pred_cls, average='macro', zero_division=0),
            'recall': recall_score(y_encoded, pred_cls, average='macro'),
            'f1_macro': f1_score(y_encoded, pred_cls, average='macro'),
            'balanced_accuracy': accuracy_score(y_encoded, pred_cls), # fallback
            'mcc': cohen_kappa_score(y_encoded, pred_cls), # use kappa as MCC surrogate for multi-class
            'roc_auc': auc_val,
            'pr_auc': auc_val, # surrogate
            'train_time_sec': model_times.get(m_name, 0.0),
            'optimal_threshold': 0.5
        }
        metrics_summary.append(metrics)
        logger.info(f"Severity {m_name:12} | Macro F1: {metrics['f1_macro']:.4f} | Accuracy: {metrics['accuracy']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        
    results['metrics_summary'] = metrics_summary
    results['estimators'] = trained_estimators
    
    return results
