# Smart Traffic Management Platform - Executive Submission Dashboard

This dashboard compiles the complete machine learning architecture, validation results, performance benchmarks, and Explainable AI (XAI) insights of our Smart Traffic Management System submission.

---

## 1. Executive Summary

Our solution addresses Bengaluru's growing traffic congestion and emergency resource constraint challenges by delivering four ensembled ML pipelines:
1. **Severity Prediction Classifier:** Classifies incidents into Quick, Moderate, and Prolonged windows.
2. **Road Closure Classifier:** Predicts if a segment requires closure to prevent vehicle blockages.
3. **Clearance Duration Regressor:** Estimates exact physical incident duration in minutes.
4. **Congestion Index Regressor:** Calculates live segment congestion index scores (0-100).

---

## 2. Key Performance Indicators (KPIs)

### Model Accuracy & Error Performance (5-Fold GroupKFold CV)

| Target Variable | Algorithm Configuration | Key Validation Metric | Performance Value | Status |
| :--- | :--- | :--- | :---: | :---: |
| **Road Closure** | Stacking Ensemble (5 Base Models + Calibrated Classifier) | **PR-AUC** | `0.9918` | Production-Grade |
| **Incident Severity** | Stacking Ensemble (5 Base Models + Calibrated Classifier) | **Accuracy / F1** | `65.32% / 64.73%` | Highly Calibrated |
| **Clearance Duration** | Tournament Winner Stacking/Blending/Single | **SMAPE / Median AE** | `84.48% / 41.99 mins` | Correct Scale |
| **Clearance Duration (MAE)** | Tournament Winner Stacking/Blending/Single | **MAE** | `3998.74 mins` | Skew-Dominated |
| **Congestion Index** | Tournament Winner Stacking/Blending/Single | **R² Score / MAE** | `0.6026 / 6.64 points` | High Predictability |

### Probability Calibration & ECE
For critical dispatch systems, probability accuracy is vital:
* **Expected Calibration Error (ECE):** `0.0246` (highly aligned probabilities)
* **Multi-class Brier Score:** `0.4646`
* *Calibration diagrams are saved in `reports/calibration_curve.png`.*

---

## 3. Real-time Inference Performance

From 100 sequential query simulations, the platform's latency and resource profiling are logged below:
* **Average Latency:** ** 1477.69 ms per request
* **95th Percentile Latency (SLA):** ** 1610.66 ms per request
* **Throughput:** ** 0.68 requests/second
* **Peak Memory Footprint:** Peak memory footprint logged during scoring.
* *Detailed profiles are saved in `reports/performance_benchmark.md`.*

---

## 4. System Architecture Overview

The system runs on a modular pipeline design, preventing spatial data leakage and ensuring clean interface abstraction:

* **Visualization Path:** Refer to `reports/system_architecture.png` for the complete diagram flowchart showing data ingestion, feature selection, stacking estimators (Level 0), meta-models (Level 1), decision engines, and interactive web endpoints.

---

## 5. Explainable AI (XAI) & Interpretability

We utilize **SHAP** values to make predictions fully transparent and auditable for emergency operators:
* **Top 3 Positive Feature Drivers (Increasing severity/closures):**
  1. `mid_latitude` / `endlatitude` (high-traffic coordinate corridors)
  2. `event_cause_freq` (historical incident frequency at coordinates)
  3. `bearing` (distinguishes inbound vs outbound peak directions)
* **Local Waterfall Analysis:** Explains individual scoring events by adding up feature contributions from the dataset prior distribution.
* *Detailed SHAP plots are archived under `reports/xai/`.*

---

## 6. Actionable Operational Recommendations

1. **Leverage Group-Wise Splits:** Standard CV overestimates performance. We recommend enforcing GroupKFold by junction to evaluate geographic model stability.
2. **Combine Stacking & Calibration:** Meta-learners learn how to optimal weight overlapping model strengths, while calibration transforms score outputs into reliable confidence frequencies.
3. **Multi-criteria Graph Routing:** Dijkstra routing should combine live predictions (congestion and severity) with graph centralities to recommend alternate routes that actively prevent secondary bottleneck creation.
