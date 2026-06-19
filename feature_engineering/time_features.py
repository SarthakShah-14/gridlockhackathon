import pandas as pd
import numpy as np

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Parse start_datetime
    start_dt = pd.to_datetime(df['start_datetime'], errors='coerce')
    
    # 1. Standard Date/Time components
    df['hour'] = start_dt.dt.hour.fillna(-1).astype(int)
    df['day'] = start_dt.dt.day.fillna(-1).astype(int)
    df['weekday'] = start_dt.dt.weekday.fillna(-1).astype(int)
    df['month'] = start_dt.dt.month.fillna(-1).astype(int)
    df['quarter'] = start_dt.dt.quarter.fillna(-1).astype(int)
    df['week_of_year'] = start_dt.dt.isocalendar().week.astype(float).fillna(-1).astype(int)
    
    # 2. Cyclic Time Encodings (Module 2)
    # Map hour (0-23) and weekday (0-6) to sin and cos
    df['sin_hour'] = np.where(df['hour'] >= 0, np.sin(2 * np.pi * df['hour'] / 24.0), 0.0)
    df['cos_hour'] = np.where(df['hour'] >= 0, np.cos(2 * np.pi * df['hour'] / 24.0), 0.0)
    df['sin_dayofweek'] = np.where(df['weekday'] >= 0, np.sin(2 * np.pi * df['weekday'] / 7.0), 0.0)
    df['cos_dayofweek'] = np.where(df['weekday'] >= 0, np.cos(2 * np.pi * df['weekday'] / 7.0), 0.0)
    
    # 3. Weekend flag
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # 4. Peak hour (8-10 AM, 5-7 PM)
    df['is_peak_hour'] = (((df['hour'] >= 8) & (df['hour'] <= 10)) | 
                          ((df['hour'] >= 17) & (df['hour'] <= 19))).astype(int)
    
    # 5. Rush hour (7-10 AM, 4-7 PM)
    df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 10)) | 
                          ((df['hour'] >= 16) & (df['hour'] <= 19))).astype(int)
    
    # 6. Day parts
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 16)).astype(int)
    df['is_evening'] = ((df['hour'] >= 16) & (df['hour'] < 21)).astype(int)
    df['is_night'] = ((df['hour'] >= 21) | (df['hour'] < 6)).astype(int)
    
    # 7. Additional calendar features
    df['quarter'] = start_dt.dt.quarter.fillna(-1).astype(int)
    
    return df
