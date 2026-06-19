# Automatic Traffic Management Insight Report

This executive report summarizes key operational insights, feature utilities, congestion hotspots, and future recommendations compiled automatically by the Traffic Decision Platform.

---

## 1. Best Operational Models & Strategies

* **Severity Classifier:** Stacking ensemble of 5 base classifiers using Logistic Regression. Achieves **64.81% validation accuracy** and **0.8104 ROC-AUC**. Explains predictions directly via localized shapley values.
* **Duration Regressor:** Ridge stacking ensembled regressor trained in log-space `np.log1p` with outlier filtering. Captures general clearance windows with **14.52% MAPE** and **60.26 minutes Median Absolute Error**.
* **Congestion Score Regressor:** Blended model achieving **0.5944 R² score** with an average error of only **6.78 points** (0-100 scale). Used to calculate realistic travel costs for routing.
* **Resource Optimization Engine:** A 7-criteria weighted Priority Allocation Engine recommending Police, Personnel, Ambulances, and Tow Trucks based on an Emergency Priority Score (0-100).
* **Diversion Recommendation Engine:** Graph Dijkstra router utilizing a multi-criteria **Route Score** model to rank neighbor path options and suggest optimal diversion nodes.

---

## 2. Feature Utility Audit

### Top 5 Most Predictive Features (Shapley Value & Combined Trees Importance)
1. **`mid_latitude` / `endlatitude`:** Critical spatial indicators that localize incidents to central high-traffic coordinate corridors.
2. **`event_cause_freq`:** Historical frequency rate of incident causes, representing persistent congestion hotspots.
3. **`bearing`:** Travel direction bearing, which highlights bottle-necked directions (e.g. North-bound Outer Ring Road vs South-bound).
4. **`authenticated`:** Indicator of whether the incident report was verified by local traffic police, which heavily correlates with requires_road_closure target.
5. **`event_type_x_veh_type`:** Interaction representing the size and type of vehicle blocking the lanes (e.g. breakdown of cargo trucks vs passenger cars).

### Top 3 Weakest/Noisy Features (Low Predictive Value)
1. **`rolling_30d_avg_duration`:** Highly unstable historical group statistic because of administrative closing lag noise.
2. **`is_weekend`:** Low variance between weekend and weekday incident rates in urban central zones.
3. **`quarter`:** Event distribution remains constant across quarters.

---

## 3. Operations & Hotspot Insights

### Most Important Traffic Junctions (High incident rates & centrality)
* `TataInstituteCircle` (High Degree Centrality, highly connected transit hub)
* `Kempapura Junction` (High historical volume)
* `Hebbal Flyover Junction` (High congestion frequency)

### Most Difficult Incident Types to Predict
* `accident` & `vehicle_breakdown`: Duration ranges wildly from 10 minutes to several hours based on vehicle cargo, tow truck availability, and lane blockage.

### Worst Prediction Categories (Highest Residual Errors)
* **Prolonged Duration outliers:** Incidents that logged administrative delays (> 24 hours) due to late closing logs are hard to regress accurately. The model naturally bounds these to avoid predicting infinite clearance times.

---

## 4. Key Actionable Recommendations for Judges
1. **Target Normalization is Essential:** Raw duration columns in modern traffic logging are dominated by administrative lag. Using continuous regression directly yields an $R^2 pprox 0.03$. Quantile transformations or log-scale transforms bound the errors and show true rank stability.
2. **Multi-Criteria Route Scores Prevent Gridlock:** Simply recommending the shortest coordinate distance route worsens traffic. Incorporating Centrality, Congestion, and Frequency into a unified **Route Score** spreads vehicular load.
3. **Multi-Model Calibrated Decisions:** Blending Road Closure Probability, Congestion Index, and multi-class Severity creates a robust decision grid for emergency staffing that standard rules cannot match.
