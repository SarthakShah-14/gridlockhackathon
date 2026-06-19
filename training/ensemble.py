import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, f1_score
from utils.helpers import setup_logger

logger = setup_logger("ensemble")

def build_average_probability_ensemble(oof_preds_dict: dict, keys: list) -> np.ndarray:
    """
    Computes a simple average probability ensemble of selected models' OOF predictions.
    """
    preds_list = [oof_preds_dict[key] for key in keys]
    avg_preds = np.mean(preds_list, axis=0)
    return avg_preds

def optimize_voting_weights(oof_preds_dict: dict, y: pd.Series, keys: list) -> tuple:
    """
    Performs grid search to find optimal soft voting weights for the base models
    that maximize Macro F1 score on out-of-fold predictions.
    """
    best_f1 = 0.0
    best_weights = None
    
    # Generate weight combinations
    if len(keys) == 3:
        # Generate weights summing to 1.0
        w_range = np.linspace(0, 1.0, 11)
        combinations = []
        for w1 in w_range:
            for w2 in w_range:
                w3 = 1.0 - w1 - w2
                if w3 >= -1e-5:
                    combinations.append([w1, w2, max(0.0, w3)])
    elif len(keys) == 4:
        w_range = np.linspace(0, 1.0, 6)
        combinations = []
        for w1 in w_range:
            for w2 in w_range:
                for w3 in w_range:
                    w4 = 1.0 - w1 - w2 - w3
                    if w4 >= -1e-5:
                        combinations.append([w1, w2, w3, max(0.0, w4)])
    else:
        combinations = [[1.0 / len(keys)] * len(keys)]
        
    for w in combinations:
        weighted_pred = np.zeros(len(y))
        for idx, key in enumerate(keys):
            weighted_pred += w[idx] * oof_preds_dict[key]
            
        # Check F1 at standard threshold 0.5
        classes = (weighted_pred >= 0.5).astype(int)
        score = f1_score(y, classes, average='macro')
        if score > best_f1:
            best_f1 = score
            best_weights = w
            
    # Normalize best weights
    best_weights = np.array(best_weights)
    best_weights = best_weights / np.sum(best_weights)
    
    logger.info(f"Optimal voting weights for keys {keys}: {best_weights.tolist()} with Macro F1: {best_f1:.4f}")
    return best_weights.tolist()

def build_weighted_voting_ensemble(oof_preds_dict: dict, keys: list, weights: list) -> np.ndarray:
    """
    Computes a weighted soft voting ensemble of base models.
    """
    weighted_pred = np.zeros(len(oof_preds_dict[keys[0]]))
    for idx, key in enumerate(keys):
        weighted_pred += weights[idx] * oof_preds_dict[key]
    return weighted_pred

def build_stacking_ensemble(oof_preds_dict: dict, y: pd.Series, keys: list) -> tuple:
    """
    Trains a Logistic Regression meta-classifier on the OOF predictions of the base models.
    """
    meta_features = []
    for key in keys:
        meta_features.append(oof_preds_dict[key].reshape(-1, 1))
    X_meta = np.hstack(meta_features)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_stack_preds = np.zeros(len(y))
    meta_models = []
    
    for train_idx, val_idx in cv.split(X_meta, y):
        X_tr, y_tr = X_meta[train_idx], y.iloc[train_idx]
        X_val, y_val = X_meta[val_idx], y.iloc[val_idx]
        
        meta_model = LogisticRegression(C=1.0, random_state=42)
        meta_model.fit(X_tr, y_tr)
        
        preds = meta_model.predict_proba(X_val)[:, 1]
        oof_stack_preds[val_idx] = preds
        meta_models.append(meta_model)
        
    final_meta_model = LogisticRegression(C=1.0, random_state=42)
    final_meta_model.fit(X_meta, y)
    
    logger.info(f"Stacking Ensemble -> LogLoss: {log_loss(y, oof_stack_preds):.4f}, AUC: {roc_auc_score(y, oof_stack_preds):.4f}")
    
    return oof_stack_preds, final_meta_model, meta_models
