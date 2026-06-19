import pandas as pd
import numpy as np

class FrequencyEncoder:
    def __init__(self, cols):
        self.cols = cols
        self.freq_maps = {}
        
    def fit(self, df: pd.DataFrame):
        for col in self.cols:
            if col in df.columns:
                # Value counts ratio
                freq = df[col].value_counts(normalize=True).to_dict()
                self.freq_maps[col] = freq
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.cols:
            if col in df.columns:
                freq_map = self.freq_maps.get(col, {})
                # Map frequency and fill unmapped categories with 0.0
                df[f"{col}_freq"] = df[col].map(freq_map).fillna(0.0)
        return df
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

class TargetEncoder:
    def __init__(self, cols, smoothing=10.0):
        self.cols = cols
        self.smoothing = smoothing
        self.target_maps = {}
        self.global_mean = 0.0
        
    def fit(self, df: pd.DataFrame, target: pd.Series):
        self.global_mean = target.mean() if target.mean() is not np.nan else 0.0
        df = df.copy()
        df['target'] = target
        
        for col in self.cols:
            if col in df.columns:
                stats = df.groupby(col)['target'].agg(['count', 'mean'])
                # Smoothed target encoding
                smooth = (stats['count'] * stats['mean'] + self.smoothing * self.global_mean) / (stats['count'] + self.smoothing)
                self.target_maps[col] = smooth.to_dict()
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.cols:
            if col in df.columns:
                mapping = self.target_maps.get(col, {})
                df[f"{col}_target_enc"] = df[col].map(mapping).fillna(self.global_mean)
        return df
        
    def fit_transform(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        return self.fit(df, target).transform(df)
