# DriftForge AI - Python FastAPI Machine Learning Core

This backend provides real Scikit-learn model training, Optuna hyperparameter optimization studies, distribution shift simulation, statistical drift analysis (Kolmogorov-Smirnov 2-sample tests, Wasserstein distance, Population Stability Index), and Explainable AI (SHAP / Feature Attribution).

## Quickstart

### 1. Setup Virtual Environment
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI Service
```bash
python run.py
```
Or directly with Uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive OpenAPI docs will be available at:
`http://localhost:8000/docs`

## Endpoints

- `GET /health`: Engine status, available libraries, and uptime.
- `POST /train`: Trains Logistic Regression, Random Forest, or Gradient Boosting with full cross-validation and evaluation metrics.
- `POST /optimize`: Executes Optuna multi-trial Bayesian/TPE optimization studies.
- `POST /drift`: Simulates feature mean/variance/noise shifts, computes Kolmogorov-Smirnov test and PSI, and evaluates model degradation.
- `POST /explain`: Computes SHAP attributions and permutation importances.
