import pandas as pd

def create_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Intersecting fields
    interactions = [
        ('priority', 'event_type'),
        ('priority', 'event_cause'),
        ('event_type', 'zone'),
        ('event_type', 'veh_type'),
        ('event_cause', 'zone')
    ]
    
    for col1, col2 in interactions:
        if col1 in df.columns and col2 in df.columns:
            new_col = f"{col1}_x_{col2}"
            df[new_col] = df[col1].astype(str) + "_" + df[col2].astype(str)
            
    if 'junction' in df.columns and 'hour' in df.columns:
        df['junction_x_hour'] = df['junction'].astype(str) + "_" + df['hour'].astype(str)
        
    if 'junction' in df.columns and 'event_type' in df.columns:
        df['junction_x_event_type'] = df['junction'].astype(str) + "_" + df['event_type'].astype(str)
            
    return df
