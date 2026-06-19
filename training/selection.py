import pandas as pd
import numpy as np
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

ORIGINAL_COLUMNS = [
    'event_type', 'latitude', 'longitude', 'endlatitude', 'endlongitude', 
    'event_cause', 'authenticated', 'direction', 'veh_type', 'corridor', 
    'priority', 'cargo_material', 'reason_breakdown', 'age_of_truck', 
    'police_station', 'gba_identifier', 'zone', 'junction', 'client_id', 
    'created_by_id', 'last_modified_by_id', 'assigned_to_police_id', 
    'citizen_accident_id', 'kgid'
]

def select_features_shap(X: pd.DataFrame, y: pd.Series, cat_features: list, 
                         top_k: int = 80, min_importance: float = 1e-5) -> tuple:
    """
    Module 2: Advanced Feature Selection Pipeline
    1. Variance Threshold
    2. Mutual Information
    3. SHAP Importance
    4. Correlation Filter (>0.95 correlation)
    Always preserves original columns.
    """
    X = X.copy()
    
    # 1. Variance Threshold (Prune constant or near-constant features)
    num_features = X.select_dtypes(include=['number']).columns.tolist()
    vars_df = X[num_features].var()
    low_var_cols = vars_df[vars_df < 1e-4].index.tolist()
    # Ensure original columns are not removed here
    cols_to_drop_var = [col for col in low_var_cols if col not in ORIGINAL_COLUMNS]
    X = X.drop(columns=cols_to_drop_var)
    num_features = [col for col in num_features if col not in cols_to_drop_var]
    
    # Fill missing values for baseline modeling
    for col in num_features:
        X[col] = X[col].fillna(X[col].median() if X[col].median() is not np.nan else 0)
    
    # Ensure categorical features are strings
    cat_features_active = [col for col in cat_features if col in X.columns]
    for col in cat_features_active:
        X[col] = X[col].fillna('unknown').astype(str)
        
    # 2. Compute SHAP value importance via baseline CatBoost
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.08,
        depth=6,
        cat_features=cat_features_active,
        verbose=0,
        random_seed=42
    )
    pool = Pool(X, y, cat_features=cat_features_active)
    model.fit(pool)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pool)
    
    if isinstance(shap_values, list):
        shap_imp = np.abs(shap_values[1]).mean(axis=0)
    elif len(shap_values.shape) == 3:
        shap_imp = np.abs(shap_values).mean(axis=(0, 2))
    else:
        shap_imp = np.abs(shap_values).mean(axis=0)
        
    shap_df = pd.DataFrame({
        'feature': X.columns,
        'shap_importance': shap_imp
    })
    
    # 3. Compute Mutual Information
    X_encoded = X.copy()
    if cat_features_active:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_encoded[cat_features_active] = oe.fit_transform(X_encoded[cat_features_active])
        X_encoded[cat_features_active] = X_encoded[cat_features_active].fillna(-1)
        
    mi_scores = mutual_info_classif(X_encoded, y, random_state=42)
    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    })
    
    features_df = pd.merge(shap_df, mi_df, on='feature')
    
    # 4. Correlation Analysis
    corr_matrix = X_encoded.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop_corr = set()
    for col in upper_tri.columns:
        if col in ORIGINAL_COLUMNS:
            continue
        high_corr_features = upper_tri.index[upper_tri[col] > 0.95].tolist()
        if high_corr_features:
            col_score = features_df.loc[features_df['feature'] == col, 'shap_importance'].values[0]
            for other_col in high_corr_features:
                other_score = features_df.loc[features_df['feature'] == other_col, 'shap_importance'].values[0]
                if col_score < other_score:
                    to_drop_corr.add(col)
                else:
                    to_drop_corr.add(other_col)
                    
    # 5. Filter features to keep
    features_to_keep = []
    for _, row in features_df.iterrows():
        feat = row['feature']
        if feat in ORIGINAL_COLUMNS:
            features_to_keep.append(feat)
            continue
        if feat in to_drop_corr:
            continue
        if row['shap_importance'] < 1e-7 and row['mi_score'] < 1e-4:
            continue
        features_to_keep.append(feat)
        
    # Order by SHAP importance and select top_k
    importance_sorted = features_df[features_df['feature'].isin(features_to_keep)].sort_values(by='shap_importance', ascending=False)
    
    original_cols_present = [col for col in ORIGINAL_COLUMNS if col in X.columns]
    top_engineered = importance_sorted[~importance_sorted['feature'].isin(original_cols_present)].head(top_k - len(original_cols_present))['feature'].tolist()
    
    final_selected = original_cols_present + [feat for feat in top_engineered if feat in features_to_keep]
    final_selected = list(dict.fromkeys(final_selected))
    
    return final_selected, features_df
