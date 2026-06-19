import pandas as pd
import numpy as np

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Drop constant/empty columns
    cols_to_drop = ['map_file', 'comment', 'meta_data']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # Ensure start_datetime exists
    if 'start_datetime' not in df.columns:
        df['start_datetime'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        df['start_datetime'] = df['start_datetime'].fillna(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Ensure latitude and longitude exist
    if 'latitude' not in df.columns:
        df['latitude'] = 12.9716
    else:
        df['latitude'] = df['latitude'].fillna(12.9716)
        
    if 'longitude' not in df.columns:
        df['longitude'] = 77.5946
    else:
        df['longitude'] = df['longitude'].fillna(77.5946)

    # Ensure endlatitude and endlongitude columns exist (can be missing in single API requests)
    if 'endlatitude' not in df.columns:
        df['endlatitude'] = np.nan
    if 'endlongitude' not in df.columns:
        df['endlongitude'] = np.nan
    
    # 2. Treat coordinates at 0.0 as NaN
    if 'endlatitude' in df.columns:
        df['endlatitude'] = df['endlatitude'].replace(0.0, np.nan)
    if 'endlongitude' in df.columns:
        df['endlongitude'] = df['endlongitude'].replace(0.0, np.nan)
        
    # 3. Standardize age_of_truck relative to event start year
    if 'age_of_truck' in df.columns and 'start_datetime' in df.columns:
        start_year = pd.to_datetime(df['start_datetime'], errors='coerce').dt.year
        # Replace NaN start_year with current year or creation year
        if 'created_date' in df.columns:
            creation_year = pd.to_datetime(df['created_date'], errors='coerce').dt.year
            start_year = start_year.fillna(creation_year)
        start_year = start_year.fillna(2024) # Fallback to 2024
        
        # Calculate clean age
        is_year = df['age_of_truck'] > 1900
        clean_age = df['age_of_truck'].copy()
        clean_age[is_year] = start_year[is_year] - df['age_of_truck'][is_year]
        
        # Clip invalid ages (e.g. negative or excessively large) to reasonable limits [0, 40]
        clean_age = clean_age.apply(lambda x: np.nan if (pd.isnull(x) or x < 0 or x > 60) else x)
        df['age_of_truck_clean'] = clean_age
    else:
        df['age_of_truck_clean'] = np.nan
        
    # 4. Standardize cargo_material and reason_breakdown strings
    for col in ['cargo_material', 'reason_breakdown']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            df[col] = df[col].replace({'nan': 'unknown', '': 'unknown'})
            
    # 5. Clean priority and fill missing values
    if 'priority' in df.columns:
        df['priority'] = df['priority'].fillna('Low').str.strip().str.capitalize()
    else:
        df['priority'] = 'Low'
        
    # 6. Fill nulls for standard categoricals with 'unknown' and ensure they exist
    cat_cols = ['corridor', 'zone', 'junction', 'police_station', 'veh_type', 'direction', 'gba_identifier', 'event_type', 'event_cause']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str).str.strip()
        else:
            df[col] = 'unknown'
            
    # 7. Convert client_id to categorical/string
    if 'client_id' in df.columns:
        df['client_id'] = df['client_id'].fillna(1).astype(int).astype(str)
        
    return df

