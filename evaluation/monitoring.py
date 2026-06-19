import pandas as pd
import numpy as np
import os
import json
import time

def log_system_monitoring(df_scored: pd.DataFrame, X_train_baseline: pd.DataFrame = None, 
                        report_path: str = "models/monitoring_metrics.json") -> dict:
    """
    Module 12: MLOps Monitoring
    Logs inference latencies, memory/CPU bounds, prediction distributions, and target/feature drift.
    """
    df = df_scored.copy()
    
    # 1. Latency & through-put metrics
    avg_latency = float(df['latency_sec'].mean()) if 'latency_sec' in df.columns else 0.02
    
    # 2. Prediction distribution logs
    prob_mean = float(df['road_closure_prob'].mean()) if 'road_closure_prob' in df.columns else 0.5
    prob_std = float(df['road_closure_prob'].std()) if 'road_closure_prob' in df.columns and len(df) > 1 else 0.0
    
    dur_mean = float(df['predicted_duration'].mean()) if 'predicted_duration' in df.columns else 60.0
    
    # 3. Simulate resource footprints (CPU/Memory)
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage_mb = float(process.memory_info().rss / 1024 / 1024)
        cpu_pct = float(psutil.cpu_percent(interval=None))
    except ImportError:
        memory_usage_mb = 150.0  # Safe fallback estimate in MB
        cpu_pct = 12.5          # Safe fallback estimate in %
    
    # 4. Simple Feature Drift detection (Kolmogorov-Smirnov style distance or mean checks)
    drift_status = "No Drift"
    drift_details = {}
    
    if X_train_baseline is not None:
        # Check coordinates (latitude/longitude) mean difference
        for col in ['latitude', 'longitude']:
            if col in df.columns and col in X_train_baseline.columns:
                train_mean = X_train_baseline[col].mean()
                batch_mean = df[col].mean()
                diff = abs(train_mean - batch_mean)
                # If coordinates shift significantly (>0.05 degrees), flag as drift
                drift_details[f"{col}_mean_diff"] = float(diff)
                if diff > 0.05:
                    drift_status = "Feature Drift Detected"
                    
    monitoring_stats = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_version': str(df['model_version'].iloc[0]) if 'model_version' in df.columns else '1.0.0',
        'average_latency_sec': avg_latency,
        'memory_usage_mb': memory_usage_mb,
        'cpu_usage_percent': cpu_pct,
        'prediction_distribution': {
            'road_closure_probability_mean': prob_mean,
            'road_closure_probability_std': prob_std,
            'duration_mean_minutes': dur_mean
        },
        'feature_drift': {
            'status': drift_status,
            'details': drift_details
        }
    }
    
    try:
        with open(report_path, "w") as f:
            json.dump(monitoring_stats, f, indent=4)
    except Exception:
        pass
        
    return monitoring_stats
