---
title: Bengaluru Traffic Decision TMC
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Bengaluru Event-Driven Traffic Decision Support Platform (TMC) 🚦


A production-grade, end-to-end Machine Learning and decision intelligence platform built to predict, analyze, and mitigate traffic incidents across Bengaluru's corridor networks. 

This platform leverages advanced feature engineering, multi-model stacking ensembles, Explainable AI (SHAP), graph transition centralities, and real-time inference latency monitoring, served through a custom glassmorphism web dashboard.

---

## 🚀 Key Platform Features

- **Multi-Target Decision Intel**:
  1. **Road Closure Classifier**: Stacking ensemble of tree-based algorithms calibrated using probability scaling (ECE/Brier score optimized).
  2. **Incident Duration Regressor**: Forecasts incident clearance times (using log-transformed target space and 99th percentile outlier pruning).
  3. **Congestion Index Regressor**: Formulates dynamic congestion score impact metrics.
  4. **Multi-class Incident Severity Classifier**: Estimates emergency dispatch levels.
- **Advanced Feature Engineering**:
  - **Graph Topology**: Extracts corridor degree, closeness, and betweenness centralities from historical transit network graphs.
  - **Spatial Hotspots**: Dynamic clustering of coordinates via DBSCAN.
  - **Cyclic Time Encodings**: Transforms hour & weekday to cyclic sin/cos spaces.
  - **Leakage-Free Statistics**: Historical corridor aggregates computed without lookahead bias.
- **Explainable AI (XAI)**:
  - Global and local SHAP explanations integrated directly into prediction pathways.
- **Interactive Decision Dashboard**:
  - Custom UI to simulate incident triggers, view automated resource allocation metrics (officers, barricades, cones required), request dynamic routing alternatives, and export operational reports (PDF & CSV).

---

## 🛠️ Project Structure

```text
├── dashboard/
│   ├── app.py                     # API backend & SimpleHTTP Server
│   └── index.html                 # Glassmorphic responsive frontend
├── preprocessing/                 # Data validation & cleaning pipelines
├── feature_engineering/           # Graph, Spatial, Temporal, & Target encodings
├── training/                      # Optuna hyperparameter tuning & stacking meta-models
├── evaluation/                    # SHAP XAI & diagnostic error analysis reports
├── inference/                     # Live prediction pipeline & similarity retrievers
├── models/                        # Serialized pipeline parameters (gitignored binary files)
├── reports/                       # Auto-generated diagnostics (error analysis, plots)
├── run_pipeline.py                # Main orchestrator (End-to-End training)
└── requirements.txt               # Project dependency versions
```

---

## ⚡ Quick Start

### 1. Prerequisites & Setup
Ensure you have a Python virtual environment activated:
```bash
# Activate your environment
.\gridhackenvanti\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the ML Pipeline
This will clean the data, engineer graph/temporal features, run Optuna Bayesian hyperparameter search, train base models, stack them with a calibrated meta-learner, and export prediction parameters:
```bash
python run_pipeline.py
```
*Note: This generates trained model artifacts under the `models/` directory and creates an `error_analysis_report.md` in the `reports/` folder.*

### 3. Launch the Dashboard
Run the server and open the operational interface in your browser:
```bash
python dashboard/app.py
```
Open **[http://localhost:8085](http://localhost:8085)** in your web browser.

---

## 📊 Stacking Model Performance & Tuning

- Models optimized using **Optuna (TPE Sampler)** across 5 algorithms: **CatBoost, LightGBM, XGBoost, Random Forest, and Extra Trees**.
- Dynamic base-model selection drops weak estimators before meta-learning to minimize online latency.
- Out-of-Fold (OOF) cross-validation grouped by `junction` to prevent spatial data leakage.
