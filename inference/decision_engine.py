import pandas as pd
import numpy as np

def generate_traffic_recommendations(road_closure_prob: float, congestion_score: float, 
                                     priority: str, dist_to_cluster_centroid: float, 
                                     event_cause: str, severity_label: str,
                                     predicted_duration: float = 30.0,
                                     is_peak_hour: float = 0.0,
                                     junc_event_count: float = 0.0) -> dict:
    """
    Model 3: Resource Optimization Engine
    Computes a scoring-based Resource Priority Score and allocates optimal available resources:
    - Police (Staffing bounded [2, 10])
    - Traffic Personnel (Staffing bounded [1, 5])
    - Ambulances (Staffing bounded [0, 3])
    - Tow Trucks (Staffing bounded [0, 2])
    - Barricades (Bounded [0, 15])
    - Traffic Cones (Bounded [5, 30])
    - Diversion Teams (Bounded [0, 3])
    """
    priority = str(priority).lower()
    
    # 1. Severity weight
    sev_map = {'quick': 0.2, 'moderate': 0.6, 'prolonged': 1.0}
    severity_weight = sev_map.get(str(severity_label).lower(), 0.5)
    
    # 2. Duration weight (capped at 240 mins)
    duration_weight = min(float(predicted_duration), 240.0) / 240.0
    
    # 3. Congestion weight
    congestion_weight = float(congestion_score) / 100.0
    
    # 4. Priority level weight
    prio_map = {'high': 1.0, 'medium': 0.5, 'low': 0.1, 'standard': 0.3}
    priority_weight = prio_map.get(priority, 0.3)
    
    # 5. Incident frequency weight (junction logs count, scaled relative to 50 counts)
    frequency_weight = min(float(junc_event_count), 50.0) / 50.0
    
    # 6. Junction Importance (DBSCAN cluster centroid distance inverse)
    dist_to_cluster_centroid = float(dist_to_cluster_centroid) if not pd.isnull(dist_to_cluster_centroid) else 1.0
    junction_importance = 1.0 / (1.0 + dist_to_cluster_centroid)
    
    # 7. Peak Hour weight
    peak_hour_weight = float(is_peak_hour)
    
    # Resource Priority Score (0-100)
    risk_score = (
        0.25 * severity_weight +
        0.20 * duration_weight +
        0.15 * congestion_weight +
        0.15 * priority_weight +
        0.10 * frequency_weight +
        0.10 * junction_importance +
        0.05 * peak_hour_weight
    ) * 100.0
    
    # Staffing allocations based on the score
    officers = int(np.clip(round(2.0 + 8.0 * (risk_score / 100.0)), 2, 10))
    personnel = int(np.clip(round(1.0 + 4.0 * (risk_score / 100.0)), 1, 5))
    
    # Ambulances: allocated if severe or high priority
    if severity_label == 'Prolonged' or priority == 'high':
        ambulances = int(np.clip(round(1.0 + 2.0 * (risk_score / 100.0)), 1, 3))
    else:
        ambulances = 0
        
    # Tow Trucks: allocated if breakdown/accident blocking
    if event_cause in ['vehicle_breakdown', 'accident']:
        tow_trucks = int(np.clip(round(1.0 + 1.0 * duration_weight), 1, 2))
    else:
        tow_trucks = 0
        
    # Barricades and Cones
    barricades = int(np.clip(round(15.0 * road_closure_prob), 0, 15))
    if road_closure_prob > 0.4 and barricades < 2:
        barricades = 2
    cones = int(np.clip(round(5.0 + 25.0 * road_closure_prob), 5, 30))
    
    # Diversion Teams
    if road_closure_prob > 0.5 or congestion_score >= 75.0:
        teams = int(np.clip(round(1.0 + 2.0 * road_closure_prob), 1, 3))
    else:
        teams = 0
        
    # Patrol Vehicles
    patrol_vehicles = int(max(1, officers // 2))
        
    # Categorize Risk Level
    if risk_score >= 75.0:
        risk_level = "Critical"
    elif risk_score >= 50.0:
        risk_level = "High"
    elif risk_score >= 25.0:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
        
    # Resource recommendations reasoning
    reasons = []
    if officers > 0: reasons.append(f"{officers} Police Officers")
    if personnel > 0: reasons.append(f"{personnel} Traffic Personnel")
    if patrol_vehicles > 0: reasons.append(f"{patrol_vehicles} Patrol Vehicles")
    if ambulances > 0: reasons.append(f"{ambulances} Ambulances")
    if tow_trucks > 0: reasons.append(f"{tow_trucks} Tow Trucks")
    
    explanation = f"Severity predicted as {severity_label} (Clearance: {predicted_duration:.1f} mins). Priority Score of {risk_score:.1f}/100. Allocating: {', '.join(reasons)}."
    
    actions = [
        f"Deploy {officers} Police Officers & {personnel} Traffic Personnel",
        f"Setup {barricades} Barricades and {cones} Cones"
    ]
    if patrol_vehicles > 0:
        actions.append(f"Deploy {patrol_vehicles} Patrol Vehicles")
    if ambulances > 0:
        actions.append(f"Deploy {ambulances} Ambulances")
    if tow_trucks > 0:
        actions.append(f"Deploy {tow_trucks} Tow Trucks")
    if teams > 0:
        actions.append(f"Deploy {teams} Active Diversion Teams")
        
    return {
        'risk_level': risk_level,
        'risk_score': risk_score, # Resource Priority Score
        'officers_required': officers,
        'traffic_personnel_required': personnel,
        'patrol_vehicles_required': patrol_vehicles,
        'ambulances_required': ambulances,
        'tow_trucks_required': tow_trucks,
        'barricades_required': barricades,
        'traffic_cones_required': cones,
        'diversion_teams_required': teams,
        'recommended_actions': actions,
        'resource_explanation': explanation
    }

def batch_recommendations(df_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the resource optimization rules to a batch of predictions.
    """
    recs = []
    for _, row in df_preds.iterrows():
        p = row['road_closure_prob']
        cong = row['congestion_score']
        prio = row.get('priority', 'low')
        dist = row.get('dist_to_cluster_centroid', 0.0)
        cause = row.get('event_cause', 'other')
        sev = row.get('predicted_severity', 'Quick')
        dur = row.get('predicted_duration', 30.0)
        is_peak = row.get('is_peak_hour', 0.0)
        hist_count = row.get('hist_junction_event_count', 0.0)
        
        rec = generate_traffic_recommendations(
            p, cong, prio, dist, cause, sev,
            predicted_duration=dur, is_peak_hour=is_peak, junc_event_count=hist_count
        )
        recs.append(rec)
        
    df_recs = pd.DataFrame(recs)
    return pd.concat([df_preds.reset_index(drop=True), df_recs], axis=1)
