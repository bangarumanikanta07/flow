# DriftForge AI – Adaptive Machine Learning Optimization & Data Drift Detection Platform

[![Platform](https://img.shields.io/badge/Platform-MLOps%20%26%20Drift%20Detection-blue)](https://github.com)
[![Tech Stack](https://img.shields.io/badge/Tech-React%20%7C%20FastAPI%20%7C%20Scikit--Learn%20%7C%20Optuna-purple)](https://github.com)

**DriftForge AI** is a machine learning optimization, distribution shift simulation, data drift detection, and model explainability platform. It evaluates model robustness under distribution shifts, detects data drift via rigorous statistical tests (Kolmogorov-Smirnov, Wasserstein distance, Population Stability Index), explains predictions via TreeSHAP and marginal contributions, and compares efficiency tradeoffs.

---

## 🌟 Key Features

1. **Dataset Management & Validation**:
   - CSV upload (drag & drop, file size validation <= 15MB) or instant enterprise presets (Telco Churn, Credit Risk, Housing Valuation).
   - Data preview with column data type detection, null value auditing, and target variable selection.
   - Classification or regression formulation support.

2. **Model Training**:
   - Supports **Logistic Regression**, **Random Forest**, and **Gradient Boosting**.
   - Configurable train/test partition ratio and reproducible random seeds.
   - Computes empirical evaluation metrics: **Accuracy, Precision, Recall, F1-Score, ROC-AUC**.
   - Interactive **Confusion Matrix** (TP, FP, TN, FN) and **ROC Curve** plotting.
   - Exact inference latency (ms) and model memory complexity calculations.

3. **Hyperparameter Optimization with Optuna**:
   - Multi-trial Bayesian / Tree-structured Parzen Estimator (TPE) parameter search.
   - Optimizes against objective metrics (Maximize F1, Accuracy, or ROC-AUC).
   - Visualizes **Optimization History trajectory** and detailed trial logs.
   - Directly compares baseline vs tuned architectures with delta improvements.

4. **Data Drift & Robustness Evaluation**:
   - Synthetic distribution shift simulator: mean shifts ($\Delta \mu$), variance scaling ($s$), and noise contamination.
   - Statistical drift calculations:
     - **Kolmogorov-Smirnov (KS) 2-Sample Test**: empirical CDF maximum vertical deviation and asymptotic $p$-values.
     - **Wasserstein Distance**: Earth Mover's Distance across continuous distributions.
     - **Population Stability Index (PSI)**: quantile divergence test ($<0.1$ stable, $0.1-0.25$ moderate, $>0.25$ high drift).
   - Overlay density histograms comparing baseline vs drifted feature distributions.
   - Model Robustness scoring: measures accuracy/F1 degradation and prediction flip rates under distribution shifts.

5. **Explainable AI (XAI)**:
   - Global Feature Importance bar charts.
   - Local sample prediction decomposition (Waterfall attribution plot) showing each feature's contribution ($\phi_i$) pushing probability away from expected base value $E[f(x)]$.

6. **Model Comparison & Tradeoff Matrix**:
   - Compares multiple models across accuracy, F1, precision, recall, latency, parameters, and drift retention.
   - Latency vs F1 Pareto efficiency scatter plot.
   - JSON export of experimental runs.

---

## 🏗️ Architecture

DriftForge AI features a **Dual-Engine Architecture**:
- **Integrated Full-Stack Core**: Runs on Express/Node.js serving Vite, implementing real mathematical machine learning algorithms, exact KS-tests, Wasserstein distance, PSI, and Shapley attribution. Zero external dependencies required.
- **Python FastAPI Core (`/backend`)**: Full Python backend powered by `fastapi`, `uvicorn`, `scikit-learn`, `optuna`, `pandas`, `numpy`, `scipy`, and `shap`.

---

## 🚀 Running Locally

### Option A: Complete Web Application (Frontend + Integrated Backend)
```bash
# Install dependencies
npm install

# Start development server on port 3000
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### Option B: Running the Python FastAPI Backend
```bash
# Navigate to backend folder
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python requirements
pip install -r requirements.txt

# Start FastAPI server
python run.py
# Or with uvicorn:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
The FastAPI backend will run on `http://localhost:8000` with Swagger docs at `http://localhost:8000/docs`. In the DriftForge UI under **Settings**, select **External Python FastAPI Core** to connect.

---

## 📊 End-to-End Workflow

1. **Dataset**: In the *Dataset* tab, click "Customer Churn (500)" or upload your CSV.
2. **Train**: Navigate to *Model Training*, choose "Random Forest" or "Gradient Boosting", adjust estimators/depth, and click "Execute Model Training".
3. **Optimize**: In *Optuna Tuning*, set 15 trials and click "Launch Optuna Optimization Study" to observe Bayesian convergence.
4. **Drift**: Open *Drift & Robustness*, adjust Mean Shift to +35%, and click "Inject Distribution Shift & Compute Drift Metrics" to inspect KS-statistics, PSI, and accuracy drop.
5. **Explain**: Visit *Explainable AI* to view global importance and local Waterfall Shapley attributions for any test instance.
6. **Compare**: In *Model Comparison*, view the Pareto efficiency tradeoff between latency and accuracy.
