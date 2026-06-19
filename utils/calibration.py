import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression as SklearnIsotonicRegression
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss

def compute_ece_mce(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    if len(y_prob.shape) == 2:
        y_pred = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1)
        accuracies = (y_pred == y_true)
    else:
        confidences = y_prob
        y_pred = (y_prob >= 0.5).astype(int)
        accuracies = (y_pred == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            difference = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * difference
            mce = max(mce, difference)
            
    return float(ece), float(mce)

def compute_multiclass_brier(y_true, y_prob):
    n_samples = len(y_true)
    if len(y_prob.shape) == 1:
        return float(np.mean((y_prob - y_true) ** 2))
    else:
        n_classes = y_prob.shape[1]
        y_one_hot = np.zeros_like(y_prob)
        y_one_hot[np.arange(n_samples), y_true] = 1.0
        return float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))

# Binary calibrators

class BetaCalibrator:
    def __init__(self):
        self.lr = None
        
    def fit(self, y_prob, y_true):
        eps = 1e-15
        p = np.clip(y_prob, eps, 1 - eps)
        x1 = np.log(p)
        x2 = -np.log(1 - p)
        X_cal = np.column_stack([x1, x2])
        self.lr = LogisticRegression(C=1e5, max_iter=1000)
        self.lr.fit(X_cal, y_true)
        return self
        
    def predict(self, y_prob):
        eps = 1e-15
        p = np.clip(y_prob, eps, 1 - eps)
        x1 = np.log(p)
        x2 = -np.log(1 - p)
        X_cal = np.column_stack([x1, x2])
        return self.lr.predict_proba(X_cal)[:, 1]

class BinaryTemperatureCalibrator:
    def __init__(self):
        self.t = 1.0
        
    def fit(self, y_prob, y_true):
        eps = 1e-15
        p = np.clip(y_prob, eps, 1 - eps)
        logits = np.log(p / (1 - p))
        
        def objective(t_val):
            scaled = logits / t_val[0]
            cal_p = 1.0 / (1.0 + np.exp(-scaled))
            cal_p = np.clip(cal_p, eps, 1 - eps)
            return -np.mean(y_true * np.log(cal_p) + (1 - y_true) * np.log(1 - cal_p))
            
        res = minimize(objective, x0=[1.0], bounds=[(0.01, 10.0)])
        self.t = res.x[0]
        return self
        
    def predict(self, y_prob):
        eps = 1e-15
        p = np.clip(y_prob, eps, 1 - eps)
        logits = np.log(p / (1 - p))
        scaled = logits / self.t
        return 1.0 / (1.0 + np.exp(-scaled))

class BinaryPlattCalibrator:
    def __init__(self):
        self.lr = None
        
    def fit(self, y_prob, y_true):
        X_cal = y_prob.reshape(-1, 1)
        self.lr = LogisticRegression(C=1e5, max_iter=1000)
        self.lr.fit(X_cal, y_true)
        return self
        
    def predict(self, y_prob):
        X_cal = y_prob.reshape(-1, 1)
        return self.lr.predict_proba(X_cal)[:, 1]

class BinaryIsotonicCalibrator:
    def __init__(self):
        self.ir = None
        
    def fit(self, y_prob, y_true):
        self.ir = SklearnIsotonicRegression(out_of_bounds='clip')
        self.ir.fit(y_prob, y_true)
        return self
        
    def predict(self, y_prob):
        return self.ir.predict(y_prob)

# Multi-class calibrators

class MulticlassTemperatureCalibrator:
    def __init__(self):
        self.t = 1.0
        
    def fit(self, y_prob, y_true):
        eps = 1e-15
        logits = np.log(np.clip(y_prob, eps, 1 - eps))
        
        def objective(t_val):
            scaled_logits = logits / t_val[0]
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            return -np.mean(np.log(softmax_probs[np.arange(len(y_true)), y_true] + eps))
            
        res = minimize(objective, x0=[1.0], bounds=[(0.01, 10.0)])
        self.t = res.x[0]
        return self
        
    def predict(self, y_prob):
        eps = 1e-15
        logits = np.log(np.clip(y_prob, eps, 1 - eps))
        scaled_logits = logits / self.t
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

class MulticlassPlattCalibrator:
    def __init__(self):
        self.lr = None
        
    def fit(self, y_prob, y_true):
        self.lr = LogisticRegression(multi_class='multinomial', C=1e5, max_iter=1000)
        self.lr.fit(y_prob, y_true)
        return self
        
    def predict(self, y_prob):
        return self.lr.predict_proba(y_prob)

class MulticlassIsotonicCalibrator:
    def __init__(self):
        self.calibrators = []
        
    def fit(self, y_prob, y_true):
        n_classes = y_prob.shape[1]
        self.calibrators = []
        for c in range(n_classes):
            ir = SklearnIsotonicRegression(out_of_bounds='clip')
            y_true_binary = (y_true == c).astype(int)
            ir.fit(y_prob[:, c], y_true_binary)
            self.calibrators.append(ir)
        return self
        
    def predict(self, y_prob):
        n_classes = y_prob.shape[1]
        preds = np.zeros_like(y_prob)
        for c in range(n_classes):
            preds[:, c] = self.calibrators[c].predict(y_prob[:, c])
        row_sums = np.sum(preds, axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return preds / row_sums

# SKLearn compatible wrappers

class CalibratorWrapper:
    def __init__(self, calibrator, base_meta_model):
        self.calibrator = calibrator
        self.base_meta_model = base_meta_model
        
    def predict_proba(self, X):
        raw_probs = self.base_meta_model.predict_proba(X)[:, 1]
        if self.calibrator is None:
            cal_probs = raw_probs
        else:
            cal_probs = self.calibrator.predict(raw_probs)
        return np.column_stack([1.0 - cal_probs, cal_probs])

class MulticlassCalibratorWrapper:
    def __init__(self, calibrator, base_meta_model):
        self.calibrator = calibrator
        self.base_meta_model = base_meta_model
        
    def predict_proba(self, X):
        raw_probs = self.base_meta_model.predict_proba(X)
        if self.calibrator is None:
            return raw_probs
        else:
            return self.calibrator.predict(raw_probs)

# Unified Calibrator Selection APIs

def calibrate_binary_probabilities(y_prob, y_true):
    calibrators = {
        'Uncalibrated': None,
        'Platt': BinaryPlattCalibrator().fit(y_prob, y_true),
        'Isotonic': BinaryIsotonicCalibrator().fit(y_prob, y_true),
        'Beta': BetaCalibrator().fit(y_prob, y_true),
        'Temperature': BinaryTemperatureCalibrator().fit(y_prob, y_true)
    }
    
    best_name = 'Uncalibrated'
    best_ece = 1.0
    best_cal = None
    results = {}
    
    for name, cal in calibrators.items():
        if name == 'Uncalibrated':
            p_cal = y_prob
        else:
            p_cal = cal.predict(y_prob)
            
        ece, mce = compute_ece_mce(y_true, p_cal)
        brier = compute_multiclass_brier(y_true, p_cal)
        results[name] = {'ECE': ece, 'MCE': mce, 'Brier': brier}
        
        if ece < best_ece:
            best_ece = ece
            best_name = name
            best_cal = cal
            
    return best_cal, best_name, results

def calibrate_multiclass_probabilities(y_prob, y_true):
    calibrators = {
        'Uncalibrated': None,
        'Platt': MulticlassPlattCalibrator().fit(y_prob, y_true),
        'Isotonic': MulticlassIsotonicCalibrator().fit(y_prob, y_true),
        'Temperature': MulticlassTemperatureCalibrator().fit(y_prob, y_true)
    }
    
    best_name = 'Uncalibrated'
    best_ece = 1.0
    best_cal = None
    results = {}
    
    for name, cal in calibrators.items():
        if name == 'Uncalibrated':
            p_cal = y_prob
        else:
            p_cal = cal.predict(y_prob)
            
        ece, mce = compute_ece_mce(y_true, p_cal)
        brier = compute_multiclass_brier(y_true, p_cal)
        results[name] = {'ECE': ece, 'MCE': mce, 'Brier': brier}
        
        if ece < best_ece:
            best_ece = ece
            best_name = name
            best_cal = cal
            
    return best_cal, best_name, results
