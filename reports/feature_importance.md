# Feature Importance & Engineering Insights Report

This report documents the top 20 features ranked by their normalized combined importance across all stacking models (Road Closure, Clearance Duration, Congestion Score, and Incident Severity).

---

## 1. Feature Importance Ranking (Top 20)

| Rank | Feature Name | Combined Importance | Category | Description & Engineering Rationale |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `mid_latitude` | **7.56%** | Spatial | The middle latitude coordinate of the corridor segment. Helps models capture localized geographical cluster baselines. |
| **2** | `endlatitude` | **7.14%** | Spatial | Ending latitude of the transit corridor. High importance indicates strong directional bottleneck correlation. |
| **3** | `event_cause_freq` | **7.08%** | Historical | Frequency frequency rate of incident causes. Captures baseline incident rate probabilities. |
| **4** | `bearing` | **6.89%** | Spatial | Segment compass direction bearing in degrees. Helps identify one-way bottlenecks and inbound/outbound lanes. |
| **5** | `endlongitude` | **6.24%** | Spatial | Ending longitude of the transit corridor segment. Works together with latitude to map 2D coordinates. |
| **6** | `mid_longitude` | **6.07%** | Spatial | Segment middle longitude coordinate. Helps isolate grid squares and local density hotspots. |
| **7** | `authenticated` | **4.51%** | Quality | Indicator of whether an incident was validated by police dispatch. Heavily correlates with true severe accidents. |
| **8** | `event_type_x_veh_type` | **3.95%** | Interaction | Multiplicative cross-feature representing type of incident and vehicle (e.g. broken down heavy container trucks). |
| **9** | `event_cause` | **3.81%** | Categorical | Primary cause of traffic disturbance (breakdowns, crashes, waterlogging, or VIP movement). |
| **10** | `cluster_size` | **3.69%** | Spatial | Size of the DBSCAN density coordinate cluster. Represents persistent hotspot concentration. |
| **11** | `priority_x_event_cause` | **3.07%** | Interaction | Interaction crossing of report priority and cause. Dispatches resources based on combined criticality. |
| **12** | `agg_junction_avg_duration` | **2.85%** | Historical | Average historical clearance duration at that specific junction. Encodes congestion profiles. |
| **13** | `sin_hour` | **2.00%** | Temporal | Cyclic sin-transform of hour-of-day. Distinguishes AM rush-hours from off-peak periods. |
| **14** | `cos_hour` | **1.56%** | Temporal | Cyclic cos-transform of hour-of-day. Captures mid-day and PM peak-hour shifts. |
| **15** | `veh_type` | **1.54%** | Categorical | Class of vehicle involved (e.g. heavy cargo trucks, light commercial, or passenger). |
| **16** | `priority_x_event_type` | **1.52%** | Interaction | Crossed feature representing dispatch severity and the incident nature. |
| **17** | `hist_event_type_breakdown_freq` | **1.46%** | Historical | Historical frequency breakdown rate of incident types. |
| **18** | `kgid` | **1.40%** | Categorical | Unique administrative jurisdiction ID. Correlates with local police division efficiency. |
| **19** | `agg_police_station_avg_duration` | **1.31%** | Historical | Average historical clearance duration for the dispatching police station. |
| **20** | `created_by_id` | **1.28%** | Quality | Operator identifier reporting the incident, representing logging accuracy and speed. |

---

## 2. Key Insights for Hackathon Judges

1. **Spatial Features Dominate:** Over **33%** of overall model decision weights are driven by geographic features (`mid_latitude`, `endlatitude`, `bearing`, `endlongitude`, `mid_longitude`, `cluster_size`). This indicates that Bengaluru's traffic bottlenecks are highly geolocalized, meaning models gain massive predictability by knowing the *exact lane and segment location*.
2. **Interaction Cross-Features Add Signal:** Combined features like `event_type_x_veh_type` and `priority_x_event_cause` successfully capture non-linear relationship patterns (e.g. a vehicle breakdown of a cargo truck vs a private scooter) that single features alone would miss.
3. **Calibrated Timings:** Cyclic time encodings (`sin_hour`, `cos_hour`) outperform raw continuous hours by representing midnight-to-morning continuity correctly, giving the model precise awareness of Bengaluru's morning and evening rush hour profiles.
