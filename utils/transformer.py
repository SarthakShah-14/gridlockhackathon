import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

class TargetTransformer:
    """
    Handles training target scaling/transformations (raw, log1p, sqrt, box-cox, yeo-johnson)
    and their mathematical inverse transformations.
    """
    def __init__(self, method='log1p'):
        self.method = method
        self.pt = None
        self.offset = 0.0

    def fit(self, y):
        y_clean = np.nan_to_num(y, nan=0.0)
        # Avoid non-positive targets for Box-Cox
        if self.method == 'box-cox':
            min_y = np.min(y_clean)
            if min_y <= 0:
                self.offset = abs(min_y) + 1.0
            else:
                self.offset = 0.0
            y_fit = y_clean + self.offset
        else:
            y_fit = y_clean
            
        y_arr = np.array(y_fit).reshape(-1, 1)
        
        if self.method in ['box-cox', 'yeo-johnson']:
            self.pt = PowerTransformer(method=self.method)
            self.pt.fit(y_arr)
        return self

    def transform(self, y):
        y_clean = np.nan_to_num(y, nan=0.0)
        y_fit = y_clean + self.offset
        y_arr = np.array(y_fit).reshape(-1, 1)
        
        if self.method == 'log1p':
            return np.log1p(y_fit)
        elif self.method == 'sqrt':
            return np.sqrt(y_fit)
        elif self.method == 'raw':
            return y_fit
        elif self.method in ['box-cox', 'yeo-johnson']:
            return self.pt.transform(y_arr).flatten()
        else:
            raise ValueError(f"Unknown transform method: {self.method}")

    def inverse_transform(self, y_trans):
        y_arr = np.array(y_trans).reshape(-1, 1)
        
        if self.method == 'log1p':
            inv = np.expm1(y_trans)
        elif self.method == 'sqrt':
            inv = np.square(y_trans)
        elif self.method == 'raw':
            inv = y_trans
        elif self.method in ['box-cox', 'yeo-johnson']:
            inv = self.pt.inverse_transform(y_arr).flatten()
        else:
            raise ValueError(f"Unknown transform method: {self.method}")
            
        # Subtract offset if applied
        inv = inv - self.offset
        return np.clip(inv, 0.0, None)
