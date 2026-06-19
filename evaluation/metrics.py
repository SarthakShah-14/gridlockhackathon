import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve, auc, balanced_accuracy_score,
    confusion_matrix, classification_report
)

def compute_binary_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    
    # Precision recall curve
    precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_pts, precision_pts)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': pr_auc,
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }

def optimize_decision_threshold(y_true: pd.Series, y_prob: np.ndarray) -> dict:
    """
    Grid searches over thresholds to find optimal values for different metrics.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    
    best_acc = 0.0
    best_acc_thresh = 0.5
    
    best_f1 = 0.0
    best_f1_thresh = 0.5
    
    best_balanced = 0.0
    best_balanced_thresh = 0.5
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_acc_thresh = t
            
        f1 = f1_score(y_true, y_pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_f1_thresh = t
            
        bal = balanced_accuracy_score(y_true, y_pred)
        if bal > best_balanced:
            best_balanced = bal
            best_balanced_thresh = t
            
    return {
        'optimal_accuracy': {'threshold': best_acc_thresh, 'value': best_acc},
        'optimal_macro_f1': {'threshold': best_f1_thresh, 'value': best_f1},
        'optimal_balanced_accuracy': {'threshold': best_balanced_thresh, 'value': best_balanced}
    }

def compute_multiclass_metrics(y_true: pd.Series, y_prob: np.ndarray, classes_mapping: list) -> dict:
    """
    Computes classification metrics for multi-class targets.
    """
    y_pred = np.argmax(y_prob, axis=1)
    
    report = classification_report(y_true, y_pred, target_names=classes_mapping, output_dict=True)
    conf = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'classification_report': report,
        'confusion_matrix': conf.tolist()
    }
