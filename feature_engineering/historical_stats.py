import pandas as pd
import numpy as np

def compute_leakage_free_historical_stats(df: pd.DataFrame, train_mode: bool = True, lookup_tables: dict = None) -> tuple:
    """
    Computes leakage-free historical statistics for columns: junction, corridor, zone, event_type.
    For each group, calculates:
    - average closure probability (closure rate)
    - average duration
    - event count
    - breakdown frequency
    - high priority frequency
    
    If train_mode=True, computes cumulative rolling statistics to avoid target leakage.
    If train_mode=False, maps the lookup_tables onto the dataframe.
    """
    df = df.copy()
    
    # Save original index to restore order at the end
    df['original_index'] = df.index
    
    # Ensure start_datetime is datetime
    df['start_datetime_dt'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    # Convert to timezone naive if it is tz-aware
    if df['start_datetime_dt'].dt.tz is not None:
        df['start_datetime_dt'] = df['start_datetime_dt'].dt.tz_localize(None)
    # If missing datetime, fill with a dummy to prevent sort issues
    df['start_datetime_dt'] = df['start_datetime_dt'].fillna(pd.Timestamp('2024-01-01'))
    
    # Retrieve or compute global priors safely
    if not train_mode and lookup_tables and 'global_priors' in lookup_tables:
        priors = lookup_tables['global_priors']
        global_closure_rate = priors.get('global_closure_rate', 0.1)
        global_avg_duration = priors.get('global_avg_duration', 60.0)
        global_breakdown_freq = priors.get('global_breakdown_freq', 0.2)
        global_priority_freq = priors.get('global_priority_freq', 0.3)
    else:
        if 'requires_road_closure' in df.columns:
            global_closure_rate = df['requires_road_closure'].astype(float).mean()
        else:
            global_closure_rate = 0.1
        if 'duration_minutes' in df.columns:
            global_avg_duration = df['duration_minutes'].mean()
        else:
            global_avg_duration = 60.0
        global_breakdown_freq = (df['event_cause'] == 'vehicle_breakdown').astype(float).mean() if 'event_cause' in df.columns else 0.2
        global_priority_freq = (df['priority'] == 'High').astype(float).mean() if 'priority' in df.columns else 0.3

    # Ensure targets are correct type (if present)
    if 'requires_road_closure' in df.columns:
        df['is_closed_target'] = df['requires_road_closure'].astype(float)
    else:
        df['is_closed_target'] = 0.0
    df['has_breakdown'] = (df['event_cause'] == 'vehicle_breakdown').astype(float) if 'event_cause' in df.columns else 0.0
    df['is_high_priority'] = (df['priority'] == 'High').astype(float) if 'priority' in df.columns else 0.0

    
    # Sort chronologically to compute cumulative stats
    df = df.sort_values(by='start_datetime_dt')
    
    group_cols = ['junction', 'corridor', 'zone', 'event_type']
    
    if train_mode:
        # Compute rolling time-aware metrics using closed='left' (strictly leakage-free)
        df_temp = df.set_index('start_datetime_dt')
        df['rolling_7d_incidents'] = df_temp.rolling('7D', closed='left')['id'].count().values
        df['rolling_30d_incidents'] = df_temp.rolling('30D', closed='left')['id'].count().values
        df['rolling_7d_closure_rate'] = df_temp.rolling('7D', closed='left')['is_closed_target'].mean().fillna(global_closure_rate).values
        if 'duration_minutes' in df.columns:
            df['rolling_30d_avg_duration'] = df_temp.rolling('30D', closed='left')['duration_minutes'].mean().fillna(global_avg_duration).values
        else:
            df['rolling_30d_avg_duration'] = global_avg_duration

        lookup_tables = {}
        
        for col in group_cols:
            # Shift by 1 within each group to ensure no leakage of the current record's target
            grouped = df.groupby(col)
            
            # Cumulative count of previous events
            prev_event_count = grouped.cumcount()
            
            # Cumulative sum of closures
            prev_closures = grouped['is_closed_target'].cumsum() - df['is_closed_target']
            
            # Cumulative breakdown counts
            prev_breakdowns = grouped['has_breakdown'].cumsum() - df['has_breakdown']
            
            # Cumulative high priority counts
            prev_priority = grouped['is_high_priority'].cumsum() - df['is_high_priority']
            
            # Cumulative duration
            # Note: duration_minutes is a future column! We must NOT leak it.
            # But the historical average duration of PREVIOUS events is allowed.
            if 'duration_minutes' in df.columns:
                dur_series = df['duration_minutes'].fillna(0)
                # Count of previous events with valid duration
                valid_dur_mask = df['duration_minutes'].notnull().astype(float)
                prev_valid_dur_count = grouped[col].transform(lambda x: valid_dur_mask.loc[x.index].cumsum() - valid_dur_mask.loc[x.index])
                prev_dur_sum = grouped[col].transform(lambda x: dur_series.loc[x.index].cumsum() - dur_series.loc[x.index])
            else:
                prev_dur_sum = pd.Series(0, index=df.index)
                prev_valid_dur_count = pd.Series(0, index=df.index)
            
            # Compute rolling features
            df[f'hist_{col}_event_count'] = prev_event_count
            
            # Closure probability
            df[f'hist_{col}_closure_prob'] = (prev_closures / prev_event_count).fillna(global_closure_rate)
            
            # Breakdown freq
            df[f'hist_{col}_breakdown_freq'] = (prev_breakdowns / prev_event_count).fillna(global_breakdown_freq)
            
            # High priority freq
            df[f'hist_{col}_priority_freq'] = (prev_priority / prev_event_count).fillna(global_priority_freq)
            
            # Avg duration
            df[f'hist_{col}_avg_duration'] = (prev_dur_sum / prev_valid_dur_count).fillna(global_avg_duration)
            
            # Build lookup tables for inference (using final values of the training set)
            final_stats = df.groupby(col).agg(
                final_event_count=('is_closed_target', 'count'),
                final_closures=('is_closed_target', 'sum'),
                final_breakdowns=('has_breakdown', 'sum'),
                final_priority=('is_high_priority', 'sum'),
                final_dur_sum=('duration_minutes', 'sum') if 'duration_minutes' in df.columns else ('is_closed_target', lambda x: 0),
                final_valid_dur_count=('duration_minutes', lambda x: x.notnull().sum()) if 'duration_minutes' in df.columns else ('is_closed_target', lambda x: 0)
            )
            
            final_stats['final_closure_prob'] = (final_stats['final_closures'] / final_stats['final_event_count']).fillna(global_closure_rate)
            final_stats['final_breakdown_freq'] = (final_stats['final_breakdowns'] / final_stats['final_event_count']).fillna(global_breakdown_freq)
            final_stats['final_priority_freq'] = (final_stats['final_priority'] / final_stats['final_event_count']).fillna(global_priority_freq)
            final_stats['final_avg_duration'] = (final_stats['final_dur_sum'] / final_stats['final_valid_dur_count']).fillna(global_avg_duration)
            
            lookup_tables[col] = final_stats.to_dict(orient='index')
            
        # Save historical events database for rolling inference (Module 10 prediction history)
        lookup_tables['historical_events'] = df[['start_datetime', 'is_closed_target', 'duration_minutes', 'id']].rename(
            columns={'is_closed_target': 'is_closed'}
        ).to_dict(orient='records')
        
        # Save global priors
        lookup_tables['global_priors'] = {
            'global_closure_rate': float(global_closure_rate),
            'global_avg_duration': float(global_avg_duration),
            'global_breakdown_freq': float(global_breakdown_freq),
            'global_priority_freq': float(global_priority_freq)
        }

            
    else:
        # Train mode is False, map values from lookup_tables
        for col in group_cols:
            table = lookup_tables.get(col, {})
            
            # Default values if key not found in table
            df[f'hist_{col}_event_count'] = df[col].apply(lambda x: table.get(x, {}).get('final_event_count', 0))
            df[f'hist_{col}_closure_prob'] = df[col].apply(lambda x: table.get(x, {}).get('final_closure_prob', global_closure_rate))
            df[f'hist_{col}_breakdown_freq'] = df[col].apply(lambda x: table.get(x, {}).get('final_breakdown_freq', global_breakdown_freq))
            df[f'hist_{col}_priority_freq'] = df[col].apply(lambda x: table.get(x, {}).get('final_priority_freq', global_priority_freq))
            df[f'hist_{col}_avg_duration'] = df[col].apply(lambda x: table.get(x, {}).get('final_avg_duration', global_avg_duration))
            
        # Dynamically compute rolling features using training history
        hist_events = lookup_tables.get('historical_events', [])
        if hist_events:
            hist_df = pd.DataFrame(hist_events)
            hist_df['start_datetime_dt'] = pd.to_datetime(hist_df['start_datetime'], errors='coerce')
            if hist_df['start_datetime_dt'].dt.tz is not None:
                hist_df['start_datetime_dt'] = hist_df['start_datetime_dt'].dt.tz_localize(None)
            
            r_7d = []
            r_30d = []
            r_7d_closed = []
            r_30d_dur = []
            
            for idx, row in df.iterrows():
                dt = pd.to_datetime(row['start_datetime'], errors='coerce')
                if dt is not None and getattr(dt, 'tz', None) is not None:
                    dt = dt.tz_localize(None)
                if pd.isnull(dt):
                    dt = pd.Timestamp('2024-01-01')
                # Find historical events in window [dt - 7 days, dt)
                mask_7d = (hist_df['start_datetime_dt'] >= dt - pd.Timedelta(days=7)) & (hist_df['start_datetime_dt'] < dt)
                mask_30d = (hist_df['start_datetime_dt'] >= dt - pd.Timedelta(days=30)) & (hist_df['start_datetime_dt'] < dt)
                
                r_7d.append(int(mask_7d.sum()))
                r_30d.append(int(mask_30d.sum()))
                r_7d_closed.append(float(hist_df.loc[mask_7d, 'is_closed'].mean() if mask_7d.any() else global_closure_rate))
                r_30d_dur.append(float(hist_df.loc[mask_30d, 'duration_minutes'].mean() if mask_30d.any() else global_avg_duration))
                
            df['rolling_7d_incidents'] = r_7d
            df['rolling_30d_incidents'] = r_30d
            df['rolling_7d_closure_rate'] = r_7d_closed
            df['rolling_30d_avg_duration'] = r_30d_dur
        else:
            df['rolling_7d_incidents'] = 0
            df['rolling_30d_incidents'] = 0
            df['rolling_7d_closure_rate'] = global_closure_rate
            df['rolling_30d_avg_duration'] = global_avg_duration
            
    # Clean up temporary columns
    df = df.drop(columns=['start_datetime_dt', 'is_closed_target', 'has_breakdown', 'is_high_priority'])
    
    # Sort back to original index
    df = df.sort_values(by='original_index').drop(columns=['original_index'])
    
    return df, lookup_tables

def compute_group_aggregations(train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> tuple:
    """
    Computes cross-validation safe group aggregations on train_df and maps them back.
    Avoids data leakage by using only train_df for computing statistics.
    """
    train_df = train_df.copy()
    
    # Map priority to numeric
    train_df['priority_num'] = train_df['priority'].map({'High': 1.0, 'Low': 0.0}).fillna(0.0)
    
    # Duration column (ensure it exists, otherwise fill with standard value)
    dur_col = 'duration_minutes' if 'duration_minutes' in train_df.columns else 'incident_duration'
    if dur_col not in train_df.columns:
        train_df[dur_col] = 60.0 # dummy
        
    # Global priors from train_df
    global_count = len(train_df)
    global_duration = train_df[dur_col].mean()
    if pd.isnull(global_duration) or global_duration <= 0:
        global_duration = 60.0
    global_priority = train_df['priority_num'].mean()
    
    # 1. Junction stats
    junc_stats = train_df.groupby('junction').agg(
        junc_count=(dur_col, 'count'),
        junc_avg_duration=(dur_col, 'mean'),
        junc_avg_priority=('priority_num', 'mean')
    ).to_dict(orient='index')
    
    # 2. Police station stats
    ps_stats = train_df.groupby('police_station').agg(
        ps_count=(dur_col, 'count'),
        ps_avg_duration=(dur_col, 'mean')
    ).to_dict(orient='index')
    
    # 3. Event type stats
    et_stats = train_df.groupby('event_type').agg(
        et_avg_duration=(dur_col, 'mean'),
        et_avg_priority=('priority_num', 'mean')
    ).to_dict(orient='index')
    
    def map_features(target_df):
        target_df = target_df.copy()
        
        # Helper to safely map values from dictionary
        target_df['agg_junction_count'] = target_df['junction'].apply(lambda x: junc_stats.get(x, {}).get('junc_count', 0.0))
        target_df['agg_junction_avg_duration'] = target_df['junction'].apply(lambda x: junc_stats.get(x, {}).get('junc_avg_duration', global_duration))
        target_df['agg_junction_avg_priority'] = target_df['junction'].apply(lambda x: junc_stats.get(x, {}).get('junc_avg_priority', global_priority))
        
        target_df['agg_police_station_count'] = target_df['police_station'].apply(lambda x: ps_stats.get(x, {}).get('ps_count', 0.0))
        target_df['agg_police_station_avg_duration'] = target_df['police_station'].apply(lambda x: ps_stats.get(x, {}).get('ps_avg_duration', global_duration))
        
        target_df['agg_event_type_avg_duration'] = target_df['event_type'].apply(lambda x: et_stats.get(x, {}).get('et_avg_duration', global_duration))
        target_df['agg_event_type_avg_priority'] = target_df['event_type'].apply(lambda x: et_stats.get(x, {}).get('et_avg_priority', global_priority))
        
        # Fill any remaining NaNs
        fill_vals = {
            'agg_junction_count': 0.0,
            'agg_junction_avg_duration': global_duration,
            'agg_junction_avg_priority': global_priority,
            'agg_police_station_count': 0.0,
            'agg_police_station_avg_duration': global_duration,
            'agg_event_type_avg_duration': global_duration,
            'agg_event_type_avg_priority': global_priority
        }
        for c, val in fill_vals.items():
            target_df[c] = target_df[c].fillna(val)
            
        return target_df
        
    train_mapped = map_features(train_df)
    # Drop temp priority_num
    if 'priority_num' in train_mapped.columns:
        train_mapped = train_mapped.drop(columns=['priority_num'])
        
    val_mapped = None
    if val_df is not None:
        val_mapped = map_features(val_df)
        
    # Build a final lookup dictionary to save for inference
    lookup_dict = {
        'junc_stats': junc_stats,
        'ps_stats': ps_stats,
        'et_stats': et_stats,
        'global_priors': {
            'global_duration': global_duration,
            'global_priority': global_priority
        }
    }
    
    return train_mapped, val_mapped, lookup_dict

def map_saved_group_aggregations(df: pd.DataFrame, lookup_dict: dict) -> pd.DataFrame:
    """
    Maps saved group statistics onto a dataframe for inference.
    """
    df = df.copy()
    junc_stats = lookup_dict.get('junc_stats', {})
    ps_stats = lookup_dict.get('ps_stats', {})
    et_stats = lookup_dict.get('et_stats', {})
    priors = lookup_dict.get('global_priors', {})
    global_duration = priors.get('global_duration', 60.0)
    global_priority = priors.get('global_priority', 0.5)
    
    df['agg_junction_count'] = df['junction'].apply(lambda x: junc_stats.get(x, {}).get('junc_count', 0.0))
    df['agg_junction_avg_duration'] = df['junction'].apply(lambda x: junc_stats.get(x, {}).get('junc_avg_duration', global_duration))
    df['agg_junction_avg_priority'] = df['junction'].apply(lambda x: junc_stats.get(x, {}).get('junc_avg_priority', global_priority))
    
    df['agg_police_station_count'] = df['police_station'].apply(lambda x: ps_stats.get(x, {}).get('ps_count', 0.0))
    df['agg_police_station_avg_duration'] = df['police_station'].apply(lambda x: ps_stats.get(x, {}).get('ps_avg_duration', global_duration))
    
    df['agg_event_type_avg_duration'] = df['event_type'].apply(lambda x: et_stats.get(x, {}).get('et_avg_duration', global_duration))
    df['agg_event_type_avg_priority'] = df['event_type'].apply(lambda x: et_stats.get(x, {}).get('et_avg_priority', global_priority))
    
    return df
