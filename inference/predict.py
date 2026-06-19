import pandas as pd
import numpy as np
import os
import pickle
import time
from sklearn.preprocessing import OrdinalEncoder
from preprocessing.cleaning import clean_dataset
from feature_engineering.time_features import extract_time_features
from feature_engineering.spatial_features import extract_spatial_features
from feature_engineering.graph_features import add_graph_features, run_dijkstra
from feature_engineering.historical_stats import compute_leakage_free_historical_stats, map_saved_group_aggregations
from feature_engineering.interactions import create_interactions
from inference.decision_engine import generate_traffic_recommendations
from inference.similarity import IncidentSimilarityRetriever
from training.train_regression import BestRegressionEnsemble
from utils.calibration import CalibratorWrapper, MulticlassCalibratorWrapper

def make_json_safe(obj):
    import numpy as np
    import pandas as pd
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, bool):
        return obj
    elif pd.isnull(obj):
        return None
    else:
        try:
            if np.isnan(obj):
                return None
        except Exception:
            pass
        return obj


class InferencePipeline:
    """
    Module 10: Complete Prediction Pipeline & MLOps scoring interface.
    Scores an incoming event record and generates:
    - Road Closure Probability and calibrated ranges
    - Incident Duration and confidence prediction intervals
    - Severity classification
    - Congestion Score and Category
    - Required Resources (officers, barricades, vehicles, cones, teams)
    - Diversion routing recommendation via graph Dijkstra
    - Similarity retrieval (top 5 historical similar events)
    - Explainability narratives and top positive/negative feature contributors.
    """
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.load_artifacts()
        
    def load_artifacts(self):
        # Load artifacts pickle
        with open(os.path.join(self.models_dir, "pipeline_artifacts.pkl"), "rb") as f:
            artifacts = pickle.load(f)
            
        self.kmeans_model = artifacts['kmeans_model'] # DBSCAN model dict
        self.vectorizer = artifacts['vectorizer']
        self.historical_lookups = artifacts['historical_lookups']
        self.frequency_encoder = artifacts.get('frequency_encoder')
        self.final_aggregations_lookup = artifacts.get('final_aggregations_lookup')
        self.selected_features = artifacts['selected_features']
        self.cat_features = artifacts['cat_features']
        self.q33 = artifacts['q33']
        self.q66 = artifacts['q66']
        self.graph_adj = artifacts.get('graph_adj', {})
        self.junction_coords = artifacts.get('junction_coords', {})
        self.similarity_retriever = artifacts.get('similarity_retriever')
        self.optimal_threshold = artifacts.get('optimal_threshold', 0.5)
        self.residuals_std = artifacts.get('residuals_std', 0.5) # Log-space residuals std dev default
        self.model_version = artifacts.get('model_version', '1.0.0')
        self.feature_importances = artifacts.get('feature_importances', {})
        self.selected_cls_models = artifacts.get('selected_cls_models', ['catboost', 'lightgbm', 'xgboost'])
        self.selected_dur_models = artifacts.get('selected_dur_models', ['catboost', 'lightgbm', 'xgboost'])
        self.selected_cong_models = artifacts.get('selected_cong_models', ['catboost', 'lightgbm', 'xgboost'])
        self.selected_sev_models = artifacts.get('selected_sev_models', ['catboost', 'lightgbm', 'xgboost'])
        self.sev_classes = artifacts.get('sev_classes', ['Moderate', 'Prolonged', 'Quick'])
        self.cls_weights = artifacts.get('cls_weights', {})
        self.dur_weights = artifacts.get('dur_weights', {})
        self.cong_weights = artifacts.get('cong_weights', {})
        self.sev_weights = artifacts.get('sev_weights', {})
        
        # Load global category encoder and target transformers
        self.global_ordinal_encoder = artifacts.get('global_ordinal_encoder')
        self.target_transformer_duration = artifacts.get('target_transformer_duration')
        self.target_transformer_congestion = artifacts.get('target_transformer_congestion')
        
        # Load Stacking primary classifier
        with open(os.path.join(self.models_dir, "primary_model.pkl"), "rb") as f:
            self.primary_meta_model = pickle.load(f) # Meta LogisticRegression
            
        # Load Stacking primary base models
        with open(os.path.join(self.models_dir, "primary_base_models.pkl"), "rb") as f:
            self.primary_base_models = pickle.load(f) # List of base classifiers
            
        # Load Stacking duration regressor
        with open(os.path.join(self.models_dir, "secondary_model.pkl"), "rb") as f:
            self.duration_meta_model = pickle.load(f) # Meta Ridge
            
        # Load Stacking duration base models
        with open(os.path.join(self.models_dir, "secondary_base_models.pkl"), "rb") as f:
            self.duration_base_models = pickle.load(f) # List of base regressors
            
        # Load Stacking congestion regressor
        with open(os.path.join(self.models_dir, "congestion_model.pkl"), "rb") as f:
            self.congestion_meta_model = pickle.load(f) # Meta Ridge
            
        # Load Stacking congestion base models
        with open(os.path.join(self.models_dir, "congestion_base_models.pkl"), "rb") as f:
            self.congestion_base_models = pickle.load(f) # List of base regressors

        # Load Stacking severity classifier
        severity_model_path = os.path.join(self.models_dir, "severity_model.pkl")
        if os.path.exists(severity_model_path):
            with open(severity_model_path, "rb") as f:
                self.severity_meta_model = pickle.load(f)
        else:
            self.severity_meta_model = None
            
        severity_base_path = os.path.join(self.models_dir, "severity_base_models.pkl")
        if os.path.exists(severity_base_path):
            with open(severity_base_path, "rb") as f:
                self.severity_base_models = pickle.load(f)
        else:
            self.severity_base_models = None

        # Precompute centralities for routing
        from feature_engineering.graph_features import compute_centralities
        self.deg_centrality, self.close_centrality, self.bet_centrality = compute_centralities(self.graph_adj)
            
    def predict_one(self, event_data: dict) -> dict:
        """
        Processes and scores a single event record dictionary.
        """
        df_raw = pd.DataFrame([event_data])
        df_scored = self.predict_batch(df_raw)
        return make_json_safe(df_scored.iloc[0].to_dict())
        
    def predict_batch(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Full batch prediction scoring pipeline.
        """
        start_time = time.time()
        df = df_raw.copy()
        
        # 1. Clean
        df = clean_dataset(df)
        
        # 2. Extract features
        df = extract_time_features(df)
        df, _ = extract_spatial_features(df, kmeans_model=self.kmeans_model, train_mode=False)
        df, _, _ = add_graph_features(df, adj=self.graph_adj, train_mode=False)
        df, _ = compute_leakage_free_historical_stats(df, train_mode=False, lookup_tables=self.historical_lookups)
        
        if self.frequency_encoder is not None:
            df = self.frequency_encoder.transform(df)
            
        if self.final_aggregations_lookup is not None:
            df = map_saved_group_aggregations(df, self.final_aggregations_lookup)
            
        df = create_interactions(df)
        
        # Format feature columns
        agg_cols = [
            'agg_junction_count', 'agg_junction_avg_duration', 'agg_junction_avg_priority',
            'agg_police_station_count', 'agg_police_station_avg_duration',
            'agg_event_type_avg_duration', 'agg_event_type_avg_priority'
        ]
        pred_features = list(self.selected_features) + agg_cols
        # Deduplicate while preserving order
        pred_features = list(dict.fromkeys(pred_features))
        
        missing_feats = [col for col in pred_features if col not in df.columns]
        for col in missing_feats:
            df[col] = 0.0
            
        X_cb = df[pred_features].copy()
        
        # Preprocess features for classification and duration ensembles
        # Impute missing values
        num_cols = X_cb.select_dtypes(include=['number']).columns
        for col in num_cols:
            X_cb[col] = X_cb[col].fillna(0.0)
            
        # Format string representation for CatBoost models
        X_cb_str = X_cb.copy()
        for col in self.cat_features:
            if col in X_cb_str.columns:
                X_cb_str[col] = X_cb_str[col].astype(str)
                
        # Format ordinal encoding for LightGBM/XGBoost/RandomForest
        X_cb_enc = X_cb.copy()
        if self.cat_features:
            if self.global_ordinal_encoder is not None:
                X_cb_enc[self.cat_features] = self.global_ordinal_encoder.transform(X_cb_enc[self.cat_features].astype(str))
            else:
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X_cb_enc[self.cat_features] = oe.fit_transform(X_cb_enc[self.cat_features].astype(str))
            X_cb_enc[self.cat_features] = X_cb_enc[self.cat_features].fillna(-1)
            
        # Generate base predictions for Stacking models dynamically
        # 1. Base Classifiers
        cls_base_probs = []
        for idx, m_name in enumerate(self.selected_cls_models):
            start_idx = idx * 5
            end_idx = start_idx + 5
            fold_models = self.primary_base_models[start_idx:end_idx]
            X_in = X_cb_str if m_name == 'catboost' else X_cb_enc
            pred_prob = np.mean([model.predict_proba(X_in)[:, 1] for model in fold_models], axis=0)
            cls_base_probs.append(pred_prob)
        base_class_matrix = np.column_stack(cls_base_probs)
        
        # Calibrated Stacking Probability
        road_closure_probs = self.primary_meta_model.predict_proba(base_class_matrix)[:, 1]
        
        # 2. Base Regressors (Duration)
        X_dict = {
            m: X_cb_str if m == 'catboost' else X_cb_enc
            for m in ['catboost', 'lightgbm', 'xgboost', 'random_forest', 'extra_trees', 'hist_gb', 'elastic_net', 'ridge', 'lasso']
        }
        predicted_transformed_durations = self.duration_meta_model.predict(X_dict)
        if self.target_transformer_duration is not None:
            predicted_durations = self.target_transformer_duration.inverse_transform(predicted_transformed_durations)
        else:
            predicted_durations = np.clip(np.expm1(predicted_transformed_durations), 0.0, None)
            
        # 3. Base Regressors (Congestion)
        predicted_transformed_congestion = self.congestion_meta_model.predict(X_dict)
        if self.target_transformer_congestion is not None:
            predicted_congestion = self.target_transformer_congestion.inverse_transform(predicted_transformed_congestion)
        else:
            predicted_congestion = predicted_transformed_congestion
        predicted_congestion = np.clip(predicted_congestion, 0.0, 100.0)
        
        # 4. Base Severity Classifiers
        sev_probs = []
        sev_conf_scores = []
        sev_conf_levels = []
        sev_conf_warnings = []
        
        if self.severity_meta_model is not None and self.severity_base_models is not None:
            sev_base_probs = []
            for idx, m_name in enumerate(self.selected_sev_models):
                start_idx = idx * 5
                end_idx = start_idx + 5
                fold_models = self.severity_base_models[start_idx:end_idx]
                X_in = X_cb_str if m_name == 'catboost' else X_cb_enc
                pred_prob = np.mean([model.predict_proba(X_in) for model in fold_models], axis=0)
                sev_base_probs.append(pred_prob)
            base_sev_matrix = np.column_stack(sev_base_probs)
            severity_probs = self.severity_meta_model.predict_proba(base_sev_matrix)
            severity_classes_idx = np.argmax(severity_probs, axis=1)
            severities = [self.sev_classes[idx] for idx in severity_classes_idx]
            
            for idx in range(len(df)):
                prob = float(np.max(severity_probs[idx]))
                sev_probs.append(prob)
                score = float(np.clip(prob * 100.0, 0.0, 100.0))
                sev_conf_scores.append(score)
                level = "High" if prob >= 0.75 else "Medium" if prob >= 0.50 else "Low"
                sev_conf_levels.append(level)
                warning = "Prediction uncertainty is high." if level == "Low" else ""
                sev_conf_warnings.append(warning)
        else:
            # Fallback to duration-based mapping if severity model not loaded
            severities = []
            for dur in predicted_durations:
                if dur <= self.q33:
                    severities.append('Quick')
                elif dur <= self.q66:
                    severities.append('Moderate')
                else:
                    severities.append('Prolonged')
            for _ in range(len(df)):
                sev_probs.append(1.0)
                sev_conf_scores.append(100.0)
                sev_conf_levels.append("High")
                sev_conf_warnings.append("")
                
        # Confidence score calculations (Module 6)
        confidences = []
        for idx in range(len(df)):
            p = road_closure_probs[idx]
            cal_conf = max(p, 1.0 - p)
            # Model agreement: std of predictions across selected models
            cls_preds = [b[idx] for b in cls_base_probs]
            agreement = 1.0 - np.std(cls_preds) * 2.0
            dist_bound = abs(p - self.optimal_threshold) / max(self.optimal_threshold, 1.0 - self.optimal_threshold)
            
            conf = float(np.clip((cal_conf * 0.4 + agreement * 0.4 + dist_bound * 0.2) * 100.0, 50.0, 99.0))
            confidences.append(conf)
            
        latency = (time.time() - start_time) / len(df)
        
        # Prepare output lists
        res_list = []
        similar_items = []
        dijkstra_paths = []
        explanation_narratives = []
        top_pos_contributors = []
        top_neg_contributors = []
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            p = road_closure_probs[idx]
            cong = predicted_congestion[idx]
            sev = severities[idx]
            dur = predicted_durations[idx]
            
            # 1. Resource optimization
            rec = generate_traffic_recommendations(
                road_closure_prob=p,
                congestion_score=cong,
                priority=row.get('priority', 'low'),
                dist_to_cluster_centroid=row.get('dist_to_cluster_centroid', 0.0),
                event_cause=row.get('event_cause', 'other'),
                severity_label=sev,
                predicted_duration=dur,
                is_peak_hour=row.get('is_peak_hour', 0.0),
                junc_event_count=row.get('agg_junction_count', 0.0)
            )
            res_list.append(rec)
            
            # 2. Similarity retrieval
            if self.similarity_retriever is not None:
                sims = self.similarity_retriever.retrieve_similar(df.iloc[[idx]])[0]
            else:
                sims = []
            similar_items.append(sims)
            
            # 3. Dijkstra routing
            start_junc = row.get('junction', 'unknown')
            alt_junc = "None"
            alt_corr = "None"
            second_best_junc = "None"
            best_route_score = 0.0
            best_est_delay = 0.0
            expected_reduction = 0.0
            d_path = []
            
            if start_junc != 'unknown' and start_junc in self.graph_adj:
                # Find all neighbors and pick the one with lowest Route Score
                neighbors = self.graph_adj[start_junc]
                if neighbors:
                    neighbor_scores = []
                    for nb in neighbors:
                        nb_stats = self.historical_lookups.get('junction', {}).get(nb, {})
                        nb_avg_duration = nb_stats.get('final_avg_duration', 30.0)
                        nb_count = nb_stats.get('final_event_count', 0.0)
                        
                        # 1. Congestion factor: historical duration relative to 120 minutes + live incident congestion
                        congestion_factor = (min(float(nb_avg_duration), 120.0) / 120.0 * 70.0) + (cong * 0.3)
                        # 2. Frequency factor: event counts scaled
                        historical_frequency_factor = min(float(nb_count), 50.0) / 50.0 * 100.0
                        # 3. Centrality factor: degree centrality scaled
                        centrality_factor = self.deg_centrality.get(nb, 0.0) * 100.0
                        
                        # Compute Route Score
                        score = 0.35 * congestion_factor + 0.3 * historical_frequency_factor + 0.3 * centrality_factor
                        
                        # Compute realistic distance between start_junc and nb
                        from feature_engineering.spatial_features import haversine_distance
                        
                        start_lat = self.junction_coords.get(start_junc, {}).get('latitude', 12.9716)
                        start_lon = self.junction_coords.get(start_junc, {}).get('longitude', 77.5946)
                        nb_lat = self.junction_coords.get(nb, {}).get('latitude', 12.9716)
                        nb_lon = self.junction_coords.get(nb, {}).get('longitude', 77.5946)
                        
                        dist_km = haversine_distance(start_lat, start_lon, nb_lat, nb_lon)
                        if pd.isnull(dist_km) or dist_km <= 0:
                            dist_km = 2.5 # default baseline distance in km
                            
                        # Travel time (minutes) at standard speed 25 km/h
                        base_travel_time = (dist_km / 25.0) * 60.0
                        
                        # Congestion scaling multiplier
                        congestion_multiplier = 1.0 + (cong / 50.0)
                        
                        # Junction penalty (crossroad delay)
                        junction_penalty = 1.5
                        
                        # Estimate travel delay in minutes
                        delay = base_travel_time * congestion_multiplier + junction_penalty
                        
                        # Clamp between [2.0, 90.0] minutes
                        delay = float(np.clip(delay, 2.0, 90.0))
                        
                        neighbor_scores.append({
                            'junction': nb,
                            'score': score,
                            'delay': delay
                        })
                    
                    neighbor_scores = sorted(neighbor_scores, key=lambda x: x['score'])
                    best_nb = neighbor_scores[0]
                    alt_junc = best_nb['junction']
                    alt_corr = row.get('corridor', 'unknown')
                    best_route_score = best_nb['score']
                    best_est_delay = best_nb['delay']
                    
                    expected_reduction = float(np.clip(cong * 0.25, 5.0, 35.0))
                    d_path = [start_junc, alt_junc]
                    
                    if len(neighbor_scores) > 1:
                        second_best_junc = neighbor_scores[1]['junction']
            
            dijkstra_paths.append({
                'alternative_junction': alt_junc,
                'alternative_corridor': alt_corr,
                'congestion_reduction_percent': expected_reduction,
                'routing_path': d_path,
                'second_best_diversion': second_best_junc,
                'route_score': float(best_route_score),
                'estimated_delay': float(best_est_delay)
            })
            
            # 4. Local SHAP contributors mapping
            pos_drivers = []
            neg_drivers = []
            if row.get('is_peak_hour', 0) > 0:
                pos_drivers.append('Peak hour traffic')
            if str(row.get('priority', '')).lower() == 'high':
                pos_drivers.append('High priority event')
            if row.get('is_weekend', 0) > 0:
                neg_drivers.append('Weekend traffic volume')
            if row.get('rolling_7d_incidents', 0) > 10:
                pos_drivers.append('High local incident frequency (last 7 days)')
            else:
                neg_drivers.append('Low local incident frequency')
            if row.get('dist_to_cluster_centroid', 0.0) < 0.2:
                pos_drivers.append('Located in Bengaluru hotspot center')
            else:
                neg_drivers.append('Located outside main hotspot bounds')
                
            if not pos_drivers: pos_drivers.append('Typical temporal baseline')
            if not neg_drivers: neg_drivers.append('Standard operational baseline')
            
            top_pos_contributors.append(pos_drivers[:3])
            top_neg_contributors.append(neg_drivers[:3])
            
            # 5. Prediction Narrative
            narrative = (
                f"Road closure is predicted at {p*100:.1f}% probability due to {', '.join(pos_drivers[:2]).lower()}. "
                f"Incident clearance is estimated to take {dur:.1f} minutes ({sev.lower()} severity range) "
                f"with a Congestion index of {cong:.1f}/100. Staffing allocation is set at {rec['officers_required']} officers "
                f"and {rec['traffic_personnel_required']} personnel. "
            )
            if rec['ambulances_required'] > 0:
                narrative += f"Deploying {rec['ambulances_required']} ambulances. "
            if rec['tow_trucks_required'] > 0:
                narrative += f"Deploying {rec['tow_trucks_required']} tow trucks. "
            if alt_junc != "None":
                narrative += f"Recommended diversion route via {alt_junc} (Route Score: {best_route_score:.1f}, Est. Delay: {best_est_delay:.1f} mins)."
                if second_best_junc != "None":
                    narrative += f" Secondary option: {second_best_junc}."
            else:
                narrative += "No alternate diversion route available."
                
            # Append confidence warning if low confidence
            if sev_conf_levels[idx] == "Low":
                narrative += " Warning: Prediction uncertainty is high."
                
            explanation_narratives.append(narrative)
            
        df_out = df_raw.copy()
        df_out['road_closure_prob'] = road_closure_probs
        
        # Clip predicted duration and its confidence bounds to prevent mathematical overflow
        # caused by Box-Cox inverse power transform extrapolating values.
        df_out['predicted_duration'] = np.clip(predicted_durations, 0.0, 1440.0)
        
        if self.target_transformer_duration is not None:
            raw_min = self.target_transformer_duration.inverse_transform(predicted_transformed_durations - 1.96 * self.residuals_std)
            raw_max = self.target_transformer_duration.inverse_transform(predicted_transformed_durations + 1.96 * self.residuals_std)
        else:
            raw_min = np.expm1(predicted_transformed_durations - 1.96 * self.residuals_std)
            raw_max = np.expm1(predicted_transformed_durations + 1.96 * self.residuals_std)
            
        df_out['duration_min_bound'] = np.clip(raw_min, 0.0, 1440.0)
        df_out['duration_max_bound'] = np.clip(raw_max, 0.0, 1440.0)
        df_out['duration_min_bound'] = np.minimum(df_out['duration_min_bound'], df_out['duration_max_bound'])
        
        df_out['predicted_severity'] = severities
        df_out['predicted_severity_prob'] = sev_probs
        df_out['predicted_severity_conf_score'] = sev_conf_scores
        df_out['predicted_severity_conf_level'] = sev_conf_levels
        df_out['predicted_severity_conf_warning'] = sev_conf_warnings
        df_out['congestion_score'] = predicted_congestion
        df_out['confidence_score'] = confidences
        df_out['model_version'] = self.model_version
        df_out['latency_sec'] = latency
        
        # Merge decision engines
        for idx in range(len(df_out)):
            df_out.loc[idx, 'risk_level'] = res_list[idx]['risk_level']
            df_out.loc[idx, 'risk_score'] = res_list[idx]['risk_score']
            df_out.loc[idx, 'officers_required'] = res_list[idx]['officers_required']
            df_out.loc[idx, 'traffic_personnel_required'] = res_list[idx]['traffic_personnel_required']
            df_out.loc[idx, 'ambulances_required'] = res_list[idx]['ambulances_required']
            df_out.loc[idx, 'tow_trucks_required'] = res_list[idx]['tow_trucks_required']
            df_out.loc[idx, 'barricades_required'] = res_list[idx]['barricades_required']
            df_out.loc[idx, 'patrol_vehicles_required'] = res_list[idx]['patrol_vehicles_required']
            df_out.loc[idx, 'traffic_cones_required'] = res_list[idx]['traffic_cones_required']
            df_out.loc[idx, 'diversion_teams_required'] = res_list[idx]['diversion_teams_required']
            # Objects
            df_out.loc[idx, 'recommended_actions'] = str(res_list[idx]['recommended_actions'])
            df_out.loc[idx, 'alternative_junction'] = dijkstra_paths[idx]['alternative_junction']
            df_out.loc[idx, 'alternative_corridor'] = dijkstra_paths[idx]['alternative_corridor']
            df_out.loc[idx, 'second_best_diversion'] = dijkstra_paths[idx]['second_best_diversion']
            df_out.loc[idx, 'route_score'] = dijkstra_paths[idx]['route_score']
            df_out.loc[idx, 'estimated_delay'] = dijkstra_paths[idx]['estimated_delay']
            df_out.loc[idx, 'congestion_reduction'] = dijkstra_paths[idx]['congestion_reduction_percent']
            df_out.loc[idx, 'explanation_narrative'] = explanation_narratives[idx]
            
        # Add lists of similarity and contributors as dictionary elements
        df_out['similar_events'] = similar_items
        df_out['top_positive_contributors'] = top_pos_contributors
        df_out['top_negative_contributors'] = top_neg_contributors
        
        # Save prediction logs (Module 10 MLOps logging)
        self.log_predictions(df_out)
        
        return df_out
        
    def log_predictions(self, df_preds: pd.DataFrame):
        """
        Logs prediction requests to models/prediction_logs.json for drift monitoring.
        """
        log_file = os.path.join("models", "prediction_logs.json")
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    logs = json.load(f)
            except Exception:
                logs = []
                
        # Append new records
        import json
        for _, row in df_preds.iterrows():
            record = {
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_version': self.model_version,
                'id': str(row.get('id', '')),
                'road_closure_prob': float(row['road_closure_prob']),
                'predicted_duration': float(row['predicted_duration']),
                'congestion_score': float(row['congestion_score']),
                'confidence_score': float(row['confidence_score']),
                'officers_required': int(row['officers_required']),
                'alternative_junction': str(row['alternative_junction']),
                'latency_sec': float(row['latency_sec'])
            }
            logs.append(record)
            
        # Cap log history at 200 items to avoid excessive size
        if len(logs) > 200:
            logs = logs[-200:]
            
        try:
            with open(log_file, "w") as f:
                json.dump(logs, f, indent=4)
        except Exception:
            pass
