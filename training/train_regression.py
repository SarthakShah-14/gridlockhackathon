import pandas as pd
import numpy as np
import optuna
import os
import pickle
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error, explained_variance_score
)
from utils.helpers import setup_logger
from feature_engineering.historical_stats import compute_group_aggregations
from feature_engineering.encoders import TargetEncoder
from utils.transformer import TargetTransformer

logger = setup_logger("train_regression")

class BestRegressionEnsemble:
    """
    Unified predictor wrapper that handles prediction using the selected
    best regressor configuration (either single, stacked, or blended).
    """
    def __init__(self, method, selected_models, base_estimators, meta_model=None, weights=None, target_encoder=None):
        self.method = method # 'single', 'stacking', 'weighted_blending', 'soft_voting'
        self.selected_models = selected_models
        self.base_estimators = base_estimators # dict: model_name -> list of fold estimators
        self.meta_model = meta_model # Ridge meta-learner (for stacking)
        self.weights = weights # dict: model_name -> weight (for blending)
        self.target_encoder = target_encoder # TargetEncoder fitted on full dataset

    def predict(self, X_dict):
        """
        Expects a dictionary mapping model_name -> preprocessed X dataframe.
        """
        X_dict_proc = {}
        for m, X_in in X_dict.items():
            X_in_proc = X_in.copy()
            if self.target_encoder is not None:
                X_in_proc = self.target_encoder.transform(X_in_proc)
            X_dict_proc[m] = X_in_proc

        # Get base model predictions
        preds_list = []
        for m in self.selected_models:
            X_in = X_dict_proc[m]
            # Average predictions across the 5 fold models
            fold_preds = np.mean([est.predict(X_in) for est in self.base_estimators[m]], axis=0)
            preds_list.append(fold_preds)

        if self.method == 'single':
            return preds_list[0]
        elif self.method == 'soft_voting':
            return np.mean(preds_list, axis=0)
        elif self.method == 'weighted_blending':
            blend = np.zeros_like(preds_list[0])
            for idx, m in enumerate(self.selected_models):
                blend += self.weights[m] * preds_list[idx]
            return blend
        elif self.method == 'stacking':
            base_matrix = np.column_stack(preds_list)
            return self.meta_model.predict(base_matrix)
        else:
            raise ValueError(f"Unknown prediction method: {self.method}")

def compute_smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    return 100.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask])

def optimize_regressor_optuna(X: pd.DataFrame, y: pd.Series, cat_features: list, 
                             groups: pd.Series, model_name: str, n_trials: int = 100) -> dict:
    """
    Optuna TPE parameter tuning for regressors with MedianPruner.
    """
    X = X.copy()
    
    # Preprocess categoricals
    if cat_features:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[cat_features] = oe.fit_transform(X[cat_features].astype(str))
        X[cat_features] = X[cat_features].fillna(-1)
        
    def objective(trial):
        if model_name == 'catboost':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'depth': trial.suggest_int('depth', 4, 8),
                'verbose': 0,
                'random_seed': 42
            }
            model = CatBoostRegressor(**params)
            
        elif model_name == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'random_state': 42,
                'verbose': -1
            }
            model = lgb.LGBMRegressor(**params)
            
        elif model_name == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'random_state': 42,
                'verbosity': 0
            }
            model = xgb.XGBRegressor(**params)
            
        elif model_name == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'random_state': 42,
                'n_jobs': -1
            }
            model = RandomForestRegressor(**params)
            
        elif model_name == 'extra_trees':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'random_state': 42,
                'n_jobs': -1
            }
            model = ExtraTreesRegressor(**params)
            
        else: # hist_gb
            params = {
                'max_iter': trial.suggest_int('max_iter', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'random_state': 42
            }
            model = HistGradientBoostingRegressor(**params)
            
        cv = GroupKFold(n_splits=3) if groups is not None else KFold(n_splits=3, shuffle=True, random_state=42)
        splits = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
        
        losses = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            if model_name == 'lightgbm':
                callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
            elif model_name == 'catboost':
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=0)
            else:
                model.fit(X_tr, y_tr)
                
            preds = model.predict(X_val)
            loss = mean_absolute_error(y_val, preds)
            losses.append(loss)
            
            trial.report(loss, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruning()
                
        return np.mean(losses)
        
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def train_regression_models(X: pd.DataFrame, y: pd.Series, cat_features: list, 
                            groups: pd.Series = None, cv_type: str = 'group', 
                            best_params_dict: dict = None, df_all: pd.DataFrame = None,
                            target_name: str = "duration", global_ordinal_encoder = None) -> dict:
    """
    Module 4 & 6: Train regression models, compare, and build a Ridge stacking ensemble.
    """
    X = X.copy()
    y = y.copy()
    
    is_duration = (target_name.lower() == "duration")
    
    # 1. Target Engineering (Study and select best transformation method)
    if is_duration:
        logger.info("Incident Duration Target: Running Target Engineering study...")
        from scipy.stats import skew, kurtosis
        
        methods = ['raw', 'log1p', 'sqrt', 'yeo-johnson', 'box-cox']
        best_method = 'log1p'
        best_skew = 999.0
        transformer_study = {}
        
        for m in methods:
            temp_transformer = TargetTransformer(method=m)
            try:
                temp_transformer.fit(y)
                y_trans = temp_transformer.transform(y)
                s = skew(y_trans)
                k = kurtosis(y_trans)
                transformer_study[m] = {'skewness': s, 'kurtosis': k}
                logger.info(f"Target Transform Study: {m:12} -> Skewness: {s:.4f}, Kurtosis: {k:.4f}")
                
                if abs(s) < abs(best_skew):
                    best_skew = s
                    best_method = m
            except Exception as e:
                logger.warning(f"Transformation method '{m}' failed: {e}")
                
        logger.info(f"Selected Optimal Target Transformation method: '{best_method}' (skewness = {best_skew:.4f})")
        target_transformer = TargetTransformer(method=best_method).fit(y)
        y_train = pd.Series(target_transformer.transform(y), index=y.index)
    else:
        target_transformer = TargetTransformer(method='raw').fit(y)
        y_train = y
        
    n_splits = 5
    if cv_type == 'group' and groups is not None:
        logger.info(f"Using GroupKFold CV grouped by: {groups.name}")
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y_train, groups))
    else:
        logger.info("Using KFold CV")
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(cv.split(X, y_train))
        
    models_to_train = [
        'catboost', 'lightgbm', 'xgboost', 'random_forest', 
        'extra_trees', 'hist_gb', 'elastic_net', 'ridge', 'lasso'
    ]
    results = {}
    
    # Matrix of OOF predictions (log/transformed space)
    oof_preds_matrix = np.zeros((len(y), len(models_to_train)))
    
    trained_estimators = {m: [] for m in models_to_train}
    
    # Setup target encoding target columns
    te_cols = ['junction', 'corridor', 'event_type', 'event_cause']
    te_cols = [c for c in te_cols if c in X.columns]
    
    # Store target encoders per fold
    fold_target_encoders = []
    
    for m_idx, model_name in enumerate(models_to_train):
        logger.info(f"--- Training Regressor: {model_name} ---")
        params = best_params_dict.get(model_name, {}) if best_params_dict else {}
        
        oof_preds = np.zeros(len(y))
        
        # Track metrics fold-by-fold (Step 7)
        fold_maes = []
        fold_rmses = []
        fold_r2s = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[train_idx].copy(), y_train.iloc[train_idx].copy()
            X_val, y_val = X.iloc[val_idx].copy(), y_train.iloc[val_idx].copy()
            
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
            
            # Target Encoding (OOF only) - calculated fold-by-fold
            if te_cols:
                te = TargetEncoder(cols=te_cols)
                # target encoder fits on raw minutes target to represent physical durations
                X_tr = te.fit_transform(X_tr, y.iloc[train_idx])
                X_val = te.transform(X_val)
                if m_idx == 0:
                    fold_target_encoders.append(te)
            
            # Format inputs
            if model_name == 'catboost':
                for col in cat_features:
                    X_tr[col] = X_tr[col].astype(str)
                    X_val[col] = X_val[col].astype(str)
                X_tr_proc, X_val_proc = X_tr, X_val
                
            else:
                X_tr_proc = X_tr.copy()
                X_val_proc = X_val.copy()
                
                # Median impute
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
                    
            # Instantiate Model
            if model_name == 'catboost':
                cb_params = {
                    'iterations': 400, 'learning_rate': 0.05, 'depth': 6,
                    'bootstrap_type': 'Bernoulli', 'verbose': 0, 'random_seed': 42, **params
                }
                if is_duration:
                    cb_params['loss_function'] = 'MAE'
                model = CatBoostRegressor(**cb_params, cat_features=cat_features)
                model.fit(X_tr_proc, y_tr, eval_set=(X_val_proc, y_val), early_stopping_rounds=40, verbose=0)
                
            elif model_name == 'lightgbm':
                lgb_params = {
                    'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 6,
                    'random_state': 42, 'verbose': -1, **params
                }
                if is_duration:
                    lgb_params['objective'] = 'mae'
                model = lgb.LGBMRegressor(**lgb_params)
                callbacks = [lgb.early_stopping(stopping_rounds=40, verbose=False)]
                model.fit(X_tr_proc, y_tr, eval_set=[(X_val_proc, y_val)], callbacks=callbacks)
                
            elif model_name == 'xgboost':
                xgb_params = {
                    'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 6,
                    'random_state': 42, 'verbosity': 0, **params
                }
                if is_duration:
                    xgb_params['objective'] = 'reg:absoluteerror'
                model = xgb.XGBRegressor(**xgb_params)
                model.fit(X_tr_proc, y_tr, eval_set=[(X_val_proc, y_val)], verbose=False)
                
            elif model_name == 'random_forest':
                rf_params = {
                    'n_estimators': 150, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1, **params
                }
                model = RandomForestRegressor(**rf_params)
                model.fit(X_tr_proc, y_tr)
                
            elif model_name == 'extra_trees':
                et_params = {
                    'n_estimators': 150, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1, **params
                }
                model = ExtraTreesRegressor(**et_params)
                model.fit(X_tr_proc, y_tr)
                
            elif model_name == 'hist_gb':
                hgb_params = {
                    'max_iter': 200, 'learning_rate': 0.05, 'max_depth': 6, 'random_state': 42, **params
                }
                if is_duration:
                    hgb_params['loss'] = 'absolute_error'
                model = HistGradientBoostingRegressor(**hgb_params)
                model.fit(X_tr_proc, y_tr)
                
            elif model_name == 'elastic_net':
                model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
                model.fit(X_tr_proc, y_tr)
                
            elif model_name == 'ridge':
                model = Ridge(alpha=1.0, random_state=42)
                model.fit(X_tr_proc, y_tr)
                
            else: # lasso
                model = Lasso(alpha=0.1, random_state=42)
                model.fit(X_tr_proc, y_tr)
                
            preds = model.predict(X_val_proc)
            oof_preds[val_idx] = preds
            trained_estimators[model_name].append(model)
            
            # Step 7: Inverse transform fold predictions and score
            preds_fold_raw = target_transformer.inverse_transform(preds)
            y_val_raw = y.iloc[val_idx]
            fold_maes.append(mean_absolute_error(y_val_raw, preds_fold_raw))
            fold_rmses.append(root_mean_squared_error(y_val_raw, preds_fold_raw))
            fold_r2s.append(r2_score(y_val_raw, preds_fold_raw))
            
        # Store predictions in matrix
        oof_preds_matrix[:, m_idx] = oof_preds
        
        # Invert predictions to evaluate performance on raw target scale
        preds_orig = target_transformer.inverse_transform(oof_preds)
            
        mae = mean_absolute_error(y, preds_orig)
        rmse = root_mean_squared_error(y, preds_orig)
        r2 = r2_score(y, preds_orig)
        mape = mean_absolute_percentage_error(y, preds_orig)
        medae = median_absolute_error(y, preds_orig)
        smape = compute_smape(y.values, preds_orig)
        
        # Verify Fold Averaging Correctness (Step 7)
        logger.info(f"Step 7 Fold Verification for {model_name}:")
        logger.info(f"  Fold MAEs: {[round(x, 2) for x in fold_maes]} -> Mean: {np.mean(fold_maes):.2f} +/- {np.std(fold_maes):.2f}")
        logger.info(f"  Fold RMSEs: {[round(x, 2) for x in fold_rmses]} -> Mean: {np.mean(fold_rmses):.2f}")
        
        results[model_name] = {
            'oof_preds': preds_orig,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'medae': medae,
            'smape': smape,
            'fold_maes': fold_maes,
            'fold_rmses': fold_rmses,
            'fold_r2s': fold_r2s
        }
        
    # Fit full dataset Target Encoder for inference
    full_target_encoder = None
    if te_cols:
        full_target_encoder = TargetEncoder(cols=te_cols)
        full_target_encoder.fit(X, y)
        
    # Tournament: Build and compare meta ensembles on the base models' OOF predictions
    logger.info("Running Automatic Model Selection / Tournament for Regression Configurations...")
    
    # 1. Soft Voting (Simple Average)
    voting_oof_trans = np.mean(oof_preds_matrix, axis=1)
    voting_oof = target_transformer.inverse_transform(voting_oof_trans)
    results['soft_voting'] = {
        'oof_preds': voting_oof,
        'mae': mean_absolute_error(y, voting_oof),
        'rmse': root_mean_squared_error(y, voting_oof),
        'r2': r2_score(y, voting_oof),
        'mape': mean_absolute_percentage_error(y, voting_oof),
        'medae': median_absolute_error(y, voting_oof),
        'smape': compute_smape(y.values, voting_oof)
    }
    
    # 2. Stacking (Ridge Meta-Learner)
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(oof_preds_matrix, y_train)
    stacking_oof_trans = meta_model.predict(oof_preds_matrix)
    stacking_oof = target_transformer.inverse_transform(stacking_oof_trans)
    
    # Extract stacking coefficients
    coefs = meta_model.coef_
    abs_coefs = np.abs(coefs)
    sum_coefs = np.sum(abs_coefs)
    weights = abs_coefs / sum_coefs if sum_coefs > 0 else np.ones(len(coefs)) / len(coefs)
    weights_dict = dict(zip(models_to_train, weights.tolist()))
    
    results['stacking'] = {
        'oof_preds': stacking_oof,
        'mae': mean_absolute_error(y, stacking_oof),
        'rmse': root_mean_squared_error(y, stacking_oof),
        'r2': r2_score(y, stacking_oof),
        'mape': mean_absolute_percentage_error(y, stacking_oof),
        'medae': median_absolute_error(y, stacking_oof),
        'smape': compute_smape(y.values, stacking_oof),
        'meta_model': meta_model,
        'weights': weights_dict,
        'selected_models': models_to_train
    }
    
    # 3. Weighted Blending (weighted average of predictions based on Ridge coefficients)
    blending_oof_trans = np.zeros_like(voting_oof_trans)
    for idx, model_name in enumerate(models_to_train):
        blending_oof_trans += weights[idx] * oof_preds_matrix[:, idx]
    blending_oof = target_transformer.inverse_transform(blending_oof_trans)
    
    results['weighted_blending'] = {
        'oof_preds': blending_oof,
        'mae': mean_absolute_error(y, blending_oof),
        'rmse': root_mean_squared_error(y, blending_oof),
        'r2': r2_score(y, blending_oof),
        'mape': mean_absolute_percentage_error(y, blending_oof),
        'medae': median_absolute_error(y, blending_oof),
        'smape': compute_smape(y.values, blending_oof)
    }
    
    # Print Tournament Leaderboard
    all_configs = models_to_train + ['stacking', 'weighted_blending', 'soft_voting']
    print(f"\n================= {target_name} TOURNAMENT LEADERBOARD ================= ")
    leaderboard = []
    for cfg in all_configs:
        res = results[cfg]
        leaderboard.append({
            'Configuration': cfg,
            'MAE': res['mae'],
            'MedAE': res['medae'],
            'R²': res['r2'],
            'MAPE': res['mape'],
            'SMAPE': res['smape'],
            'RMSE': res['rmse']
        })
    df_lead = pd.DataFrame(leaderboard).sort_values(by='MAE')
    print(df_lead.to_string(index=False))
    print("========================================================================\n")
    
    # Select the overall best model configuration
    best_config = df_lead.iloc[0]['Configuration']
    logger.info(f"Tournament Winner: '{best_config}' with MAE: {df_lead.iloc[0]['MAE']:.2f}")
    
    # Wrap in BestRegressionEnsemble
    if best_config in ['stacking', 'weighted_blending', 'soft_voting']:
        ensemble = BestRegressionEnsemble(
            method=best_config,
            selected_models=models_to_train,
            base_estimators=trained_estimators,
            meta_model=meta_model if best_config == 'stacking' else None,
            weights=weights_dict if best_config == 'weighted_blending' else None,
            target_encoder=full_target_encoder
        )
    else:
        # Single best base model configuration
        ensemble = BestRegressionEnsemble(
            method='single',
            selected_models=[best_config],
            base_estimators={best_config: trained_estimators[best_config]},
            target_encoder=full_target_encoder
        )
        
    results['best_oof_preds'] = results[best_config]['oof_preds']
    results['best_config'] = best_config
    results['best_ensemble'] = ensemble
    results['target_transformer'] = target_transformer
    results['estimators'] = trained_estimators
    
    return results
