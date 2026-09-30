# DriftForge AI - Python FastAPI Machine Learning Core

This backend provides scikit-learn model training, Optuna hyperparameter optimization, distribution shift simulation, statistical drift analysis (Kolmogorov-Smirnov 2-sample test, Wasserstein distance, Population Stability Index), and explainability (SHAP, with permutation importance as fallback).

When you run `npm run dev` from the project root, this backend is started automatically on port 8001 using `backend/.venv`. The steps below are only needed to run it on its own.

## Quickstart

### 1. Set up the virtual environment
Run from this `backend` folder.

Windows (Command Prompt):
```cmd
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

macOS / Linux:
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### 2. Start the FastAPI service
```cmd
.venv\Scripts\python run.py
```
(macOS/Linux: `.venv/bin/python run.py`)

Interactive OpenAPI docs: `http://localhost:8000/docs`

## Endpoints

Each endpoint is also available under an `/api` prefix (e.g. `/api/train`).

- `GET /health`: Engine status and available frameworks.
- `POST /api/datasets/parse`: Parses CSV text and returns a column summary.
- `POST /train`: Trains Logistic Regression, Random Forest, or Gradient Boosting and returns test-split metrics, confusion matrix, ROC curve, and feature importances.
- `POST /optimize`: Runs an Optuna TPE study (5–30 trials) maximizing F1, accuracy, or ROC-AUC, and returns the trial log and the tuned model.
- `POST /drift`: Applies a seeded mean/variance/noise shift to numeric features, computes KS, Wasserstein, and PSI per feature, and evaluates the given model configuration's degradation and prediction flip rate.
- `POST /explain`: Computes SHAP global importances and a local attribution breakdown for one test sample.
- `POST /predict`: Trains on the provided rows and scores unseen rows (the target column is excluded from inputs).

`/train`, `/drift`, `/explain`, and `/predict` accept an optional `hyperparameters` object in either UI (camelCase, e.g. `nEstimators`) or Optuna (snake_case, e.g. `n_estimators`) form.
