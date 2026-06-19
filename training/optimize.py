import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import log_loss
import warnings

warnings.filterwarnings('ignore')

def optimize_catboost(X: pd.DataFrame, y: pd.Series, cat_features: list, groups: pd.Series = None, n_trials: int = 100) -> dict:
    X = X.copy()
    for col in cat_features:
        X[col] = X[col].astype(str)
        
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
            'bootstrap_type': 'Bernoulli',
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'verbose': 0,
            'random_seed': 42
        }
        
        cv = GroupKFold(n_splits=3) if groups is not None else StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        splits = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
        
        losses = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            model = CatBoostClassifier(**params, cat_features=cat_features)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=0)
            
            preds = model.predict_proba(X_val)
            loss = log_loss(y_val, preds)
            losses.append(loss)
            
            # Prune trial after each fold
            trial.report(loss, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruning()
                
        return np.mean(losses)

    # Use MedianPruner to prune bad trials early
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def optimize_lightgbm(X: pd.DataFrame, y: pd.Series, cat_features: list, groups: pd.Series = None, n_trials: int = 100) -> dict:
    X = X.copy()
    if cat_features:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[cat_features] = encoder.fit_transform(X[cat_features].astype(str))
        X[cat_features] = X[cat_features].fillna(-1)
        
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100), # is min_data_in_leaf
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        
        cv = GroupKFold(n_splits=3) if groups is not None else StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        splits = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
        
        losses = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
            
            preds = model.predict_proba(X_val)
            loss = log_loss(y_val, preds)
            losses.append(loss)
            
            trial.report(loss, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruning()
                
        return np.mean(losses)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def optimize_xgboost(X: pd.DataFrame, y: pd.Series, cat_features: list, groups: pd.Series = None, n_trials: int = 100) -> dict:
    X = X.copy()
    if cat_features:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[cat_features] = encoder.fit_transform(X[cat_features].astype(str))
        X[cat_features] = X[cat_features].fillna(-1)
        
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-3, 5.0, log=True),
            'random_state': 42,
            'verbosity': 0
        }
        
        cv = GroupKFold(n_splits=3) if groups is not None else StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        splits = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
        
        losses = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            preds = model.predict_proba(X_val)
            loss = log_loss(y_val, preds)
            losses.append(loss)
            
            trial.report(loss, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruning()
                
        return np.mean(losses)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def optimize_random_forest(X: pd.DataFrame, y: pd.Series, cat_features: list, groups: pd.Series = None, n_trials: int = 100) -> dict:
    X = X.copy()
    
    num_cols = X.select_dtypes(include=['number']).columns
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median() if X[col].median() is not np.nan else 0)
        
    if cat_features:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[cat_features] = encoder.fit_transform(X[cat_features].astype(str))
        X[cat_features] = X[cat_features].fillna(-1)
        
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'random_state': 42,
            'n_jobs': -1
        }
        
        cv = GroupKFold(n_splits=3) if groups is not None else StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        splits = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
        
        losses = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            model = RandomForestClassifier(**params)
            model.fit(X_tr, y_tr)
            
            preds = model.predict_proba(X_val)
            loss = log_loss(y_val, preds)
            losses.append(loss)
            
            trial.report(loss, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruning()
                
        return np.mean(losses)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def optimize_extra_trees(X: pd.DataFrame, y: pd.Series, cat_features: list, groups: pd.Series = None, n_trials: int = 100) -> dict:
    X = X.copy()
    
    num_cols = X.select_dtypes(include=['number']).columns
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median() if X[col].median() is not np.nan else 0)
        
    if cat_features:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[cat_features] = encoder.fit_transform(X[cat_features].astype(str))
        X[cat_features] = X[cat_features].fillna(-1)
        
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'random_state': 42,
            'n_jobs': -1
        }
        
        cv = GroupKFold(n_splits=3) if groups is not None else StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        splits = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
        
        losses = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            model = ExtraTreesClassifier(**params)
            model.fit(X_tr, y_tr)
            
            preds = model.predict_proba(X_val)
            loss = log_loss(y_val, preds)
            losses.append(loss)
            
            trial.report(loss, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruning()
                
        return np.mean(losses)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

