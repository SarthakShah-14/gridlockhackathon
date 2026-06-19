# Explainable AI (XAI) Report: Severity Stacking Classifier

This report documents the local and global feature attribution mechanisms of the Severity Stacking Classifier. By leveraging SHAP (SHapley Additive exPlanations), we explain how the base estimators contribute to predictions across classes: `Quick`, `Moderate`, and `Prolonged`.

---

## 1. Global Interpretability Analysis

### Severity Model Feature Importance
The figure below ranks the top features by their mean absolute SHAP value across all three severity classes. This stacked bar chart visualizes how each feature influences class-specific decisions.

![Global Feature Importance](shap_global_importance.png)

#### Key Global Insights
1. **Junction & Corridor Centrality:** Degree and closeness centralities are critical features. High centrality junctions lead to longer clearance times due to complex traffic patterns.
2. **Temporal Features:** Hour of day and cyclic time encodings (sin/cos components) highly influence predictions, capturing diurnal peak congestion cycles.
3. **Event Characteristics:** Incident cause (`event_cause_freq`) and vehicle types (`event_type_x_veh_type`) are major drivers of severity.

---

## 2. Class-specific Interpretability

Below are the beeswarm plots for the individual severity classes. Beeswarm plots reveal the directionality of features—showing how high or low values of a feature increase or decrease the likelihood of a specific prediction class.

### Class: Quick
![Beeswarm Quick](shap_beeswarm_quick.png)

* **Quick Characteristics:** Standard temporal baselines, low-frequency junction counts, and off-peak hours heavily shift predictions toward the `Quick` severity class.

### Class: Moderate
![Beeswarm Moderate](shap_beeswarm_moderate.png)

* **Moderate Characteristics:** Balanced spatial hotspots and standard incident counts typically result in a `Moderate` prediction.

### Class: Prolonged
![Beeswarm Prolonged](shap_beeswarm_prolonged.png)

* **Prolonged Characteristics:** High vehicle counts, high-density spatial centroids, peak hours, and heavy-cargo vehicle types strongly increase the probability of a `Prolonged` clearance event.

---

## 3. Local Decision Attribution

For any scored incident, we can visualize the local SHAP waterfall plot or decision plot showing the exact step-by-step feature attributions starting from the base value to the final prediction.

### Waterfall Plot (Individual Prediction)
Below is the feature attribution for a highly confident prediction:

![Local Waterfall Plot](shap_local_waterfall.png)

* **Baseline Value:** The starting base probability (prior distribution across the training dataset).
* **Positive/Negative Drivers:** The horizontal bars show how each feature pulls the prediction higher or lower to arrive at the final probability.

### Cumulative Decision Path
The decision plot below shows the path of predictions for a sample batch of incidents, demonstrating the cumulative feature effects.

![SHAP Decision Plot](shap_decision_plot.png)
