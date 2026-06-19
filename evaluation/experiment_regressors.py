import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, median_absolute_error
from sklearn.linear_model import HuberRegressor, TweedieRegressor, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

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

logger = setup_logger("experiment_regressors")

def run_experiments():
    logger.info("Loading and preprocessing dataset...")
    df_raw = load_data()
    df_validated = validate_dataset(df_raw)
    df_cleaned = clean_dataset(df_validated)
    df_targets, _, _ = engineer_targets(df_cleaned, train_mode=True)
    df_features = extract_time_features(df_targets)
    df_features, dbscan_model = extract_spatial_features(df_features, train_mode=True)
    df_features, graph_adj, junction_coords = add_graph_features(df_features, train_mode=True)
    df_features, historical_lookups = compute_leakage_free_historical_stats(df_features, train_mode=True)
    
    # Load pipeline artifacts to match features
    with open("models/pipeline_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    selected_features = artifacts['selected_features']
    cat_features = artifacts['cat_features']
    
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
    feature_cols = list(dict.fromkeys(feature_cols))
    
    missing_feats = [col for col in feature_cols if col not in df_features.columns]
    for col in missing_feats:
        df_features[col] = 0.0
        
    # Get valid duration rows
    duration_mask = df_features['duration_minutes'].notnull() & (df_features['duration_minutes'] > 0)
    X_dur_all = df_features.loc[duration_mask, feature_cols].copy()
    y_dur_all = df_features.loc[duration_mask, 'duration_minutes']
    groups_all = df_features.loc[duration_mask, 'junction']
    
    # Filter 99th percentile outliers
    q99 = np.percentile(y_dur_all, 99)
    outlier_mask = y_dur_all <= q99
    X_dur = X_dur_all[outlier_mask].copy()
    y_dur = y_dur_all[outlier_mask].copy()
    groups = groups_all[outlier_mask].copy()
    
    # Impute missing values
    num_cols = X_dur.select_dtypes(include=['number']).columns
    for col in num_cols:
        X_dur[col] = X_dur[col].fillna(0.0)
        
    # Setup GroupKFold
    cv = GroupKFold(n_splits=5)
    splits = list(cv.split(X_dur, y_dur, groups))
    
    # Define models to test
    models = {
        'HistGradientBoosting (L1/MAE Loss)': HistGradientBoostingRegressor(loss='absolute_error', random_state=42),
        'Huber Regressor (on raw target)': HuberRegressor(max_iter=1000),
        'Tweedie Regressor (power=1.5)': TweedieRegressor(power=1.5, link='log', max_iter=1000),
        'Ridge Regression (on log target)': Ridge()
    }
    
    # We will also test CatBoost and LightGBM with MAE loss
    
    # Preprocess ordinal features for scikit-learn models
    X_dur_enc = X_dur.copy()
    if cat_features:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_dur_enc[cat_features] = oe.fit_transform(X_dur_enc[cat_features].astype(str))
        X_dur_enc[cat_features] = X_dur_enc[cat_features].fillna(-1)
        
    # Standard scale for linear/Huber models
    scaler = StandardScaler()
    X_dur_scaled = pd.DataFrame(scaler.fit_transform(X_dur_enc), columns=X_dur_enc.columns, index=X_dur_enc.index)
    
    results = []
    
    for name, model in models.items():
        logger.info(f"Evaluating model: {name}")
        oof_preds = np.zeros(len(y_dur))
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X_dur_enc.iloc[train_idx].copy(), y_dur.iloc[train_idx].copy()
            X_val, y_val = X_dur_enc.iloc[val_idx].copy(), y_dur.iloc[val_idx].copy()
            
            if "Huber" in name or "Tweedie" in name:
                X_tr = X_dur_scaled.iloc[train_idx].copy()
                X_val = X_dur_scaled.iloc[val_idx].copy()
                
            if "log target" in name:
                y_tr = np.log1p(y_tr)
                model.fit(X_tr, y_tr)
                preds = np.clip(np.expm1(model.predict(X_val)), 0.0, None)
            else:
                model.fit(X_tr, y_tr)
                preds = np.clip(model.predict(X_val), 0.0, None)
                
            oof_preds[val_idx] = preds
            
        mae = mean_absolute_error(y_dur, oof_preds)
        medae = median_absolute_error(y_dur, oof_preds)
        r2 = r2_score(y_dur, oof_preds)
        rmse = root_mean_squared_error(y_dur, oof_preds)
        
        results.append({
            'Model Configuration': name,
            'MAE (mins)': mae,
            'Median AE (mins)': medae,
            'R² Score': r2,
            'RMSE (mins)': rmse
        })
        logger.info(f"  MAE: {mae:.2f} | MedAE: {medae:.2f} | R²: {r2:.4f}")
        
    # Test LightGBM with MAE (L1) loss in log-space vs raw space
    logger.info("Evaluating LightGBM with L1 loss (log target)...")
    oof_preds_lgb = np.zeros(len(y_dur))
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X_dur_enc.iloc[train_idx].copy(), np.log1p(y_dur.iloc[train_idx]).copy()
        X_val, y_val = X_dur_enc.iloc[val_idx].copy(), np.log1p(y_dur.iloc[val_idx]).copy()
        
        # Fit LightGBM with MAE loss (L1)
        model = lgb.LGBMRegressor(objective='mae', random_state=42, verbose=-1)
        model.fit(X_tr, y_tr)
        preds = np.clip(np.expm1(model.predict(X_val)), 0.0, None)
        oof_preds_lgb[val_idx] = preds
        
    mae_lgb = mean_absolute_error(y_dur, oof_preds_lgb)
    medae_lgb = median_absolute_error(y_dur, oof_preds_lgb)
    r2_lgb = r2_score(y_dur, oof_preds_lgb)
    rmse_lgb = root_mean_squared_error(y_dur, oof_preds_lgb)
    results.append({
        'Model Configuration': 'LightGBM (MAE/L1 loss on log target)',
        'MAE (mins)': mae_lgb,
        'Median AE (mins)': medae_lgb,
        'R² Score': r2_lgb,
        'RMSE (mins)': rmse_lgb
    })
    logger.info(f"  MAE: {mae_lgb:.2f} | MedAE: {medae_lgb:.2f} | R²: {r2_lgb:.4f}")
    
    df_results = pd.DataFrame(results)
    print("\n--- Model Experiments Comparison Table ---")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    run_experiments()
