# DriftForge AI – Adaptive Machine Learning Optimization & Data Drift Detection Platform

[![Platform](https://img.shields.io/badge/Platform-MLOps%20%26%20Drift%20Detection-blue)](https://github.com)
[![Tech Stack](https://img.shields.io/badge/Tech-React%20%7C%20FastAPI%20%7C%20Scikit--Learn%20%7C%20Optuna-purple)](https://github.com)

**DriftForge AI** is a machine learning optimization, distribution shift simulation, data drift detection, and model explainability platform. It tunes models with Optuna, stress-tests them under simulated distribution shifts, detects drift with statistical tests (Kolmogorov-Smirnov, Wasserstein distance, Population Stability Index), explains predictions with SHAP, and compares accuracy/latency tradeoffs.

---

## 🌟 Key Features

1. **Dataset Management & Validation**
   - CSV upload (drag & drop, up to 15MB) or built-in presets: Customer Churn, Credit Risk, Housing Valuation, Fraud Transactions.
   - Data preview with column type detection, null value auditing, and target column selection.
   - Binary classification. Numeric targets with more than two values are binarized at the median.

2. **Model Training**
   - **Logistic Regression**, **Random Forest**, and **Gradient Boosting** (scikit-learn).
   - Configurable train/test split and random seed.
   - Metrics on the held-out test split: **Accuracy, Precision, Recall, F1-Score, ROC-AUC**.
   - **Confusion Matrix** and **ROC Curve**, plus inference latency (ms) and serialized model size.

3. **Hyperparameter Optimization with Optuna**
   - Tree-structured Parzen Estimator (TPE) search, 5–30 trials, seeded for reproducibility.
   - Single objective per study: maximize F1, Accuracy, or ROC-AUC.
   - **Optimization history** chart, full trial log, and baseline vs tuned comparison.

4. **Data Drift & Robustness Evaluation**
   - Shift simulator applied to numeric features: mean shift (in standard deviations), variance scaling, and noise. Seeded, so identical settings give identical results.
   - Evaluates the leading model (highest F1, shown as "Leader" in the header) with its exact hyperparameters, not a generic default.
   - Per-feature drift statistics:
     - **Kolmogorov-Smirnov 2-sample test** (statistic and p-value)
     - **Wasserstein distance**
     - **Population Stability Index** (<0.1 stable, 0.1–0.25 moderate, >0.25 high)
   - Baseline vs shifted distribution histograms.
   - Robustness verdict from accuracy drop, F1 drop, and prediction flip rate: `ROBUST`, `MODERATE_DEGRADATION`, or `CRITICAL_FAILURE`.

5. **Explainable AI (XAI)**
   - Global feature importance from mean absolute SHAP values (TreeSHAP for tree models, LinearSHAP for logistic regression, permutation importance as fallback).
   - Local waterfall plot showing how each feature pushed a test sample's probability away from the base value $E[f(x)]$.

6. **Model Comparison**
   - Side-by-side accuracy, F1, precision, recall, ROC-AUC, latency, and memory.
   - Latency vs F1 tradeoff scatter plot and JSON export.

7. **Test Prediction**
   - Score a single manually entered record or a batch CSV of unseen rows with the leading model.

---

## 🏗️ Architecture

```
Browser (React + Vite)
   │  /api/*
   ▼
Express server (server.ts, port 3000)
   │  forwards requests
   ▼
Python FastAPI core (backend/main.py, port 8001, started automatically)
   scikit-learn · Optuna · SHAP · SciPy · pandas
```

- `npm run dev` starts the Express server, which launches the Python backend automatically using `backend/.venv` (or the interpreter in the `PYTHON` environment variable).
- **Fallback:** if the Python backend can't start, the app keeps working on a built-in TypeScript engine. That engine is a simplified approximation (its "tuning" is a fixed search, not Optuna), so always confirm Python is connected before presenting results (see [Check it's working](#-check-its-working)).
- No API keys are required. `.env.example` is left over from the project template and is not used.

---

## 🚀 Running Locally

**Prerequisites:** Node.js 18+ and Python 3.10+.

### 1. Install dependencies (once)

**Windows (Command Prompt)**
```cmd
npm install
python -m venv backend\.venv
backend\.venv\Scripts\python -m pip install -r backend\requirements.txt
```

**macOS / Linux**
```bash
npm install
python3 -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt
```

> **Windows tip:** if you have several Pythons installed (e.g. MSYS2 and Miniconda), `python` may not be the one you expect. Use the full path of the one you want, e.g. `C:\Users\<you>\miniconda3\python.exe -m venv backend\.venv`. Don't re-run the `venv` command with a different Python on an existing `backend\.venv`; delete the folder first.

Verify the Python install:
```cmd
backend\.venv\Scripts\python -c "import uvicorn, sklearn, optuna, shap; print('OK')"
```
(macOS/Linux: `backend/.venv/bin/python -c ...`)

### 2. Start the app

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Keep this terminal open. After changing backend code, stop it with **Ctrl+C** and start it again.

---

## ✅ Check it's working

1. Open [http://localhost:3000/api/health](http://localhost:3000/api/health) (or run `curl.exe http://localhost:3000/api/health` in PowerShell). It should show:
   ```json
   "fastApiBridgeReady": true
   ```
   It can take a few seconds after startup.
2. In the app, trained models should be named **"… (Scikit-Learn)"** and tuned models **"… (Optuna Tuned)"**. If you see **"(Optuna Optimized)"**, the app is on the TypeScript fallback.
3. Sanity checks:
   - Drift with 0% mean shift, 1.0x variance, and 0% noise → about 0% accuracy drop, status `ROBUST`.
   - Increasing the mean shift → the drop and flip rate grow.
   - The best Optuna trial is never below trial #1.

---

## 📊 End-to-End Workflow

1. **Dataset**: click a preset (e.g. "Customer Churn (500)") or upload a CSV, then pick the target column.
2. **Model Training**: choose a model, adjust hyperparameters, click "Execute Model Training".
3. **Optuna Tuning**: set the number of trials and launch the study. The drift, explain, and prediction tabs use whichever model has the highest F1 (the "Leader" in the header).
4. **Drift & Robustness**: set mean shift, variance, and noise, then inject the shift to see per-feature KS/Wasserstein/PSI and the model's degradation.
5. **Explainable AI**: view global SHAP importance and the waterfall for any test sample.
6. **Model Comparison**: compare all trained models and export the results as JSON.
7. **Test Prediction**: score new records with the leading model.

---

## 🐍 Running the Python backend on its own

```cmd
cd backend
.venv\Scripts\python run.py
```
(macOS/Linux: `.venv/bin/python run.py`)

The API runs at `http://localhost:8000` with interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs). To have the browser call it directly, open **Settings** in the app and select **External Python FastAPI Core**. See [backend/README.md](backend/README.md) for the endpoint list.

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| `EADDRINUSE: address already in use :3000` | Another dev server is running. Stop it (Ctrl+C in its window), or on Windows: `for /f "tokens=5" %a in ('netstat -ano ^| findstr :3000 ^| findstr LISTENING') do taskkill /PID %a /F` |
| `[FastAPI Backend] exited with code 1` / `fastApiBridgeReady: false` | `backend/.venv` is missing or incomplete. Re-run step 1 and the verify command, or set `PYTHON` to an interpreter that has the requirements installed. |
| `No module named uvicorn` | The venv was created with a different Python than the packages were installed for. Delete `backend/.venv` and repeat step 1. |
| PowerShell `curl` shows a "Script Execution Risk" prompt | PowerShell's `curl` is an alias for `Invoke-WebRequest`. Use `curl.exe` instead. |

---

## ⚠️ Current Limitations

- Hyperparameter search optimizes a single metric; there is no multi-objective (Pareto) search yet.
- Optuna trials are scored on the test split; there is no separate validation split or k-fold cross-validation yet.
- Binary classification only.
- No fairness or calibration metrics yet.
- Drift is simulated and detected; the model is not automatically re-optimized after drift.
