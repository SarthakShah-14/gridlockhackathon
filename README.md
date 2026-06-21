# Bengaluru Event-Driven Traffic Decision Support Platform (TMC) 🚦

This project was built to help traffic management teams make faster and better decisions during road incidents in Bengaluru. It uses machine learning models to predict road closures, incident duration, congestion impact, and incident severity, then presents the results through an interactive dashboard.

## 🌐 Live Demo

https://sarthak1410-gridlock-hackathon.hf.space/

---

## What This Project Does

The platform takes incident-related information and generates predictions that can help traffic authorities respond more effectively.

It includes:

- **Road Closure Prediction** using a stacking ensemble of tree-based models.
- **Incident Duration Prediction** to estimate how long an incident may take to clear.
- **Congestion Index Prediction** to measure expected traffic impact.
- **Incident Severity Classification** to estimate response priority levels.

The system also provides SHAP-based explanations so users can understand which features influenced the predictions.

---

## Key Features

### Feature Engineering

The project uses several engineered features, including:

- Graph-based metrics such as corridor degree, closeness and betweenness centrality.
- Spatial hotspot detection using DBSCAN clustering.
- Cyclical time features using sine and cosine transformations.
- Historical corridor statistics generated without look-ahead bias.

### Explainable AI

- Global and local SHAP explanations for model predictions.

### Dashboard Features

The dashboard allows users to:

- Simulate traffic incidents.
- View prediction results in real time.
- Estimate resource requirements such as officers, barricades and cones.
- Explore alternate routing suggestions.
- Export reports in PDF and CSV formats.

---

## Project Structure

```text
├── dashboard/
│   ├── app.py                     # API backend & SimpleHTTP Server
│   └── index.html                 # Glassmorphic responsive frontend
├── preprocessing/                 # Data validation & cleaning pipelines
├── feature_engineering/           # Graph, Spatial, Temporal, & Target encodings
├── training/                      # Optuna hyperparameter tuning & stacking meta-models
├── evaluation/                    # SHAP XAI & diagnostic error analysis reports
├── inference/                     # Live prediction pipeline & similarity retrievers
├── models/                        # Serialized pipeline parameters
├── reports/                       # Auto-generated diagnostics
├── run_pipeline.py                # Main training pipeline
└── requirements.txt               # Project dependencies
```

## Technologies Used

| Layer | Technologies / Tools |
|---------|----------------------|
| Frontend | HTML5, CSS3, JavaScript (ES6+), Leaflet.js, English & Kannada Support |
| Backend | Python 3.10, http.server, socketserver |
| Database | MongoDB Atlas, PyMongo, dnspython |
| Machine Learning | CatBoost, XGBoost, LightGBM, Optuna, SHAP |
| Graph Analytics | NetworkX, Dijkstra's Algorithm |
| Visualization & Reporting | Matplotlib, Custom PDF Report Generator |
| Deployment | Docker, Hugging Face Spaces |

---

## Quick Start

### 1. Install Dependencies

Make sure your virtual environment is activated.

```bash
.\gridhackenvanti\Scripts\activate

pip install -r requirements.txt
```

### 2. Run the Training Pipeline

```bash
python run_pipeline.py
```

This will:

- Clean and preprocess the data.
- Generate graph and temporal features.
- Run Optuna hyperparameter tuning.
- Train and stack multiple models.
- Generate model artifacts and evaluation reports.

### 3. Launch the Dashboard

```bash
python dashboard/app.py
```

Then open:

```text
http://localhost:8085
```

---

## Model Training Details

- Optuna (TPE Sampler) is used for hyperparameter optimization.
- Models include CatBoost, LightGBM, XGBoost, Random Forest and Extra Trees.
- Weak base models are removed before stacking to reduce inference latency.
- Out-of-Fold cross-validation is grouped by `junction` to avoid spatial data leakage.

---

## Notes

- `reports/` contains evaluation reports, SHAP summaries and diagnostics.
- Run `run_pipeline.py` before launching the dashboard.
- Model files are automatically generated inside `/models`.
- The project follows a modular structure from training to inference and deployment.
- Designed as a practical traffic management solution for Bengaluru road networks.

---

## Summary

A machine learning-based traffic management platform that converts incident data into actionable predictions and operational insights for Bengaluru's traffic network.
