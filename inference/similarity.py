import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

class IncidentSimilarityRetriever:
    """
    Module 10: Incident Similarity Retrieval
    Standardizes features, runs PCA (retaining 95% variance), and fits Nearest Neighbors
    to retrieve similar historical traffic incidents.
    """
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95, random_state=42)
        self.knn = NearestNeighbors(n_neighbors=top_k, metric='minkowski')
        self.features_list = []
        self.train_data_df = None
        self.train_features_scaled_pca = None
        self.cat_cols = []
        self.num_cols = []
        self.encoder = None
        
    def fit(self, df_train: pd.DataFrame, features: list):
        self.features_list = [f for f in features if f in df_train.columns]
        self.train_data_df = df_train.copy().reset_index(drop=True)
        
        # Extract features
        X_num = df_train[self.features_list].copy()
        
        # Identify categorical vs numerical columns
        self.cat_cols = X_num.select_dtypes(exclude=[np.number]).columns.tolist()
        self.num_cols = X_num.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.cat_cols:
            self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_num[self.cat_cols] = self.encoder.fit_transform(X_num[self.cat_cols].astype(str))
            X_num[self.cat_cols] = X_num[self.cat_cols].fillna(-1)
            
        # Fill missing values for numerical columns
        for col in self.num_cols:
            median_val = X_num[col].median()
            median_val = median_val if not pd.isnull(median_val) else 0.0
            X_num[col] = X_num[col].fillna(median_val)
            
        # 1. Standardize
        X_scaled = self.scaler.fit_transform(X_num)
        
        # 2. PCA (retaining 95% variance to reduce noise)
        X_pca = self.pca.fit_transform(X_scaled)
        self.train_features_scaled_pca = X_pca
        
        # 3. Fit KNN
        self.knn.fit(X_pca)
        return self
        
    def retrieve_similar(self, new_event_features: pd.DataFrame) -> list:
        """
        Retrieves the top_k most similar historical records for the input event.
        Returns a list of dicts with records and their similarity confidence.
        """
        # Align features
        X_new = new_event_features[self.features_list].copy()
        
        if self.encoder is not None and self.cat_cols:
            X_new[self.cat_cols] = self.encoder.transform(X_new[self.cat_cols].astype(str))
            X_new[self.cat_cols] = X_new[self.cat_cols].fillna(-1)
            
        # Fill missing values for numerical columns
        for col in self.num_cols:
            median_val = self.train_data_df[col].median()
            median_val = median_val if not pd.isnull(median_val) else 0.0
            X_new[col] = X_new[col].fillna(median_val)
            
        # Scale and PCA
        X_new_scaled = self.scaler.transform(X_new)
        X_new_pca = self.pca.transform(X_new_scaled)
        
        distances, indices = self.knn.kneighbors(X_new_pca)
        
        results = []
        for q_idx in range(len(new_event_features)):
            q_recs = []
            for d, idx in zip(distances[q_idx], indices[q_idx]):
                hist_row = self.train_data_df.iloc[idx].to_dict()
                # Calculate simple similarity confidence percentage based on distance
                sim_score = float(np.clip(100.0 / (1.0 + d), 50.0, 100.0))
                hist_row['similarity_score'] = sim_score
                q_recs.append(hist_row)
            results.append(q_recs)
            
        return results
