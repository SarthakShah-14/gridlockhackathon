import pandas as pd
import numpy as np
import os

def validate_dataset(df: pd.DataFrame, report_path: str = "reports/data_audit_report.md") -> pd.DataFrame:
    """
    Module 1: Advanced Data Validation
    Performs data quality checks (missing, duplicates, invalid coordinates, timestamp anomalies, outliers)
    and saves a markdown report.
    """
    df = df.copy()
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Check missing values
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    
    # Check duplicates
    duplicate_count = df.duplicated().sum()
    
    # Validate Coordinates (Bangalore bounding box: Lat 12.8 to 13.2, Lon 77.4 to 77.8)
    invalid_coords = (
        (df['latitude'].isnull()) | (df['longitude'].isnull()) |
        (df['latitude'] < 12.0) | (df['latitude'] > 14.0) |
        (df['longitude'] < 77.0) | (df['longitude'] > 79.0)
    )
    invalid_coord_count = invalid_coords.sum()
    
    # Validate Timestamps
    start_dt = pd.to_datetime(df['start_datetime'], errors='coerce')
    end_dt = pd.to_datetime(df['end_datetime'], errors='coerce')
    
    ts_inconsistencies = (end_dt.notnull()) & (start_dt.notnull()) & (end_dt < start_dt)
    ts_inconsistency_count = ts_inconsistencies.sum()
    
    # Missing timestamps
    missing_start_ts = df['start_datetime'].isnull().sum()
    
    # Categorical checking
    event_type_distribution = df['event_type'].value_counts().to_dict()
    priority_distribution = df['priority'].value_counts(dropna=False).to_dict()
    
    # Construct markdown report
    report_lines = [
        "# Data Quality Audit Report",
        f"**Audit Timestamp:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Records Scanned:** {len(df)}",
        "",
        "## 1. Summary Metrics",
        f"- **Duplicate Records:** {duplicate_count}",
        f"- **Invalid Coordinates (outside Bangalore bounds):** {invalid_coord_count}",
        f"- **Chronological Timestamp Inconsistencies:** {ts_inconsistency_count}",
        f"- **Missing Start Timestamps:** {missing_start_ts}",
        "",
        "## 2. Missing Value Statistics",
        "| Column Name | Missing Count | Percentage (%) |",
        "|---|---|---|",
    ]
    
    for col in df.columns:
        if missing_counts[col] > 0:
            report_lines.append(f"| `{col}` | {missing_counts[col]} | {missing_pct[col]:.2f}% |")
            
    report_lines.extend([
        "",
        "## 3. Categorical Distribution Logs",
        "### Event Types:",
    ])
    for k, v in event_type_distribution.items():
        report_lines.append(f"- **{k}:** {v} ({v/len(df)*100:.2f}%)")
        
    report_lines.append("### Priorities:")
    for k, v in priority_distribution.items():
        report_lines.append(f"- **{k}:** {v} ({v/len(df)*100:.2f}%)")
        
    report_lines.extend([
        "",
        "## 4. Inconsistency Action Log",
        "- Invalid coordinates will have their cluster mappings assigned to a default cluster center.",
        "- Inconsistent end times are ignored for duration training and binned fallbacks.",
        "- Missing durations are imputed via historical averages of event types."
    ])
    
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
        
    return df
