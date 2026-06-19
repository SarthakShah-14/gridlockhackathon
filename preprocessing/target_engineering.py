import pandas as pd
import numpy as np

def calculate_duration_minutes(df: pd.DataFrame) -> pd.Series:
    """
    Calculate duration ONLY using explicit datetime columns.
    No fallback to modified_datetime.
    """
    s_dt = pd.to_datetime(df['start_datetime'], errors='coerce')
    c_dt = pd.to_datetime(df['closed_datetime'], errors='coerce')
    r_dt = pd.to_datetime(df['resolved_datetime'], errors='coerce')
    e_dt = pd.to_datetime(df['end_datetime'], errors='coerce')
    
    # Combined end datetime using only explicit fields
    end_dt = c_dt.fillna(r_dt).fillna(e_dt)
    
    # Calculate duration in minutes
    duration = (end_dt - s_dt).dt.total_seconds() / 60.0
    return duration

def compute_all_ground_truth_durations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    s_dt = pd.to_datetime(df['start_datetime'], errors='coerce')
    e_dt = pd.to_datetime(df['end_datetime'], errors='coerce')
    r_dt = pd.to_datetime(df['resolved_datetime'], errors='coerce')
    c_dt = pd.to_datetime(df['closed_datetime'], errors='coerce')
    m_dt = pd.to_datetime(df['modified_datetime'], errors='coerce')
    
    df['incident_duration'] = (e_dt - s_dt).dt.total_seconds() / 60.0
    df['time_until_resolution'] = (r_dt - s_dt).dt.total_seconds() / 60.0
    df['time_until_closure'] = (c_dt - s_dt).dt.total_seconds() / 60.0
    df['modification_delay'] = (m_dt - s_dt).dt.total_seconds() / 60.0
    return df

def engineer_targets(df: pd.DataFrame, train_mode: bool = True, q33: float = None, q66: float = None) -> tuple:
    """
    Calculate targets for the dataset.
    Returns:
        df: modified dataframe
        q33: 33rd percentile threshold
        q66: 66th percentile threshold
    """
    df = df.copy()
    df = compute_all_ground_truth_durations(df)
    
    # 1. Primary Target
    # Ensure requires_road_closure is boolean
    if 'requires_road_closure' in df.columns:
        df['requires_road_closure'] = df['requires_road_closure'].astype(bool)
    else:
        df['requires_road_closure'] = False
        
    # 2. Secondary Target (Severity Classification)
    duration = calculate_duration_minutes(df)
    df['duration_minutes'] = duration
    
    # Filter for valid positive durations to define severity classes
    valid_mask = duration.notnull() & (duration > 0)
    
    if train_mode:
        valid_durations = duration[valid_mask]
        if len(valid_durations) > 0:
            q33 = valid_durations.quantile(0.3333)
            q66 = valid_durations.quantile(0.6667)
        else:
            q33, q66 = 30.0, 120.0  # Safe defaults
            
    # Assign severity classes
    def get_severity_class(minutes):
        if pd.isnull(minutes) or minutes <= 0:
            return np.nan
        if minutes <= q33:
            return 'Quick'
        elif minutes <= q66:
            return 'Moderate'
        else:
            return 'Prolonged'
            
    df['severity'] = df['duration_minutes'].apply(get_severity_class)
    
    return df, q33, q66

def calculate_congestion_target(df: pd.DataFrame) -> pd.Series:
    """
    Module 6: Congestion Score Target Index Computation
    Calculates a rich operational congestion target index (0-100) from historical fields.
    """
    df = df.copy()
    
    # 1. Duration Percentile
    dur = df['duration_minutes'].fillna(df['duration_minutes'].median() if df['duration_minutes'].median() is not np.nan else 30.0)
    dur_pct = dur.rank(pct=True) * 100
    
    # 2. Closure Indicator (historical closure target)
    closure = df['requires_road_closure'].astype(float) * 100
    
    # 3. Location Density (DBSCAN cluster size if available, else fallback)
    if 'cluster_size' in df.columns:
        density = df['cluster_size'].fillna(1.0)
        density_pct = density.rank(pct=True) * 100
    else:
        density_pct = pd.Series(50.0, index=df.index)
        
    # 4. Priority Weight
    priority_map = {'high': 1.0, 'medium': 0.5, 'low': 0.1, 'standard': 0.3}
    prio_w = df['priority'].astype(str).str.lower().map(priority_map).fillna(0.3) * 100
    
    # 5. Graph Centrality (Degree Centrality if available, else fallback)
    if 'degree_centrality' in df.columns:
        centrality = df['degree_centrality'].fillna(0.0)
        centrality_pct = centrality.rank(pct=True) * 100
    else:
        centrality_pct = pd.Series(50.0, index=df.index)
        
    # 6. Peak Hour
    peak = df.get('is_peak_hour', pd.Series(0, index=df.index)).fillna(0) * 100
    
    # Weighted congestion target index
    congestion = (
        0.35 * dur_pct +
        0.25 * closure +
        0.15 * density_pct +
        0.10 * prio_w +
        0.05 * centrality_pct +
        0.10 * peak
    )
    
    return congestion.clip(0, 100)
