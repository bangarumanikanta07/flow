"""
DriftForge AI - Machine Learning Optimization & Drift Detection API
Powered by FastAPI, Scikit-learn, Optuna, Pandas, NumPy, SciPy, and SHAP.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
import io
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from sklearn.inspection import permutation_importance
from scipy.stats import ks_2samp, wasserstein_distance
import optuna
import shap

app = FastAPI(
    title="DriftForge AI ML & Drift Detection Engine",
    description="Production-grade adaptive ML optimization, hyperparameter tuning with Optuna, data drift detection (KS-test, Wasserstein, PSI), and explainability with SHAP.",
    version="1.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Request Models
class TrainRequest(BaseModel):
    rows: List[Dict[str, Any]]
    targetCol: Optional[str] = None
    targetColumn: Optional[str] = None
    modelType: str = "random_forest"
    hyperparameters: Optional[Dict[str, Any]] = None
    testRatio: float = 0.25
    testSize: Optional[float] = None
    seed: int = 42
    randomState: Optional[int] = None

    def get_target(self) -> str:
        t = self.targetCol or self.targetColumn
        if not t:
            raise ValueError("targetCol or targetColumn is required")
        return t

    def get_test_ratio(self) -> float:
        return self.testSize if self.testSize is not None else self.testRatio

    def get_seed(self) -> int:
        return self.randomState if self.randomState is not None else self.seed

class OptimizeRequest(BaseModel):
    rows: List[Dict[str, Any]]
    targetCol: Optional[str] = None
    targetColumn: Optional[str] = None
    modelType: str = "random_forest"
    nTrials: int = 15
    objectiveMetric: str = "f1Score"

    def get_target(self) -> str:
        t = self.targetCol or self.targetColumn
        if not t:
            raise ValueError("targetCol or targetColumn is required")
        return t

class ShiftConfig(BaseModel):
    meanShiftPct: float = 25.0
    varianceScale: float = 1.3
    noiseLevel: float = 0.15
    affectedFeatures: List[str] = Field(default_factory=list)

class DriftRequest(BaseModel):
    rows: List[Dict[str, Any]]
    targetCol: Optional[str] = None
    targetColumn: Optional[str] = None
    modelType: str = "random_forest"
    shiftConfig: ShiftConfig

    def get_target(self) -> str:
        t = self.targetCol or self.targetColumn
        if not t:
            raise ValueError("targetCol or targetColumn is required")
        return t

class ExplainRequest(BaseModel):
    rows: List[Dict[str, Any]]
    targetCol: Optional[str] = None
    targetColumn: Optional[str] = None
    modelType: str = "random_forest"
    sampleIndex: int = 0

    def get_target(self) -> str:
        t = self.targetCol or self.targetColumn
        if not t:
            raise ValueError("targetCol or targetColumn is required")
        return t

class PredictRequest(BaseModel):
    trainingRows: List[Dict[str, Any]]
    testRows: List[Dict[str, Any]]
    targetCol: Optional[str] = None
    targetColumn: Optional[str] = None
    modelType: str = "random_forest"
    hyperparameters: Optional[Dict[str, Any]] = None

    def get_target(self) -> str:
        t = self.targetCol or self.targetColumn
        if not t:
            raise ValueError("targetCol or targetColumn is required")
        return t

class ParseCsvRequest(BaseModel):
    csvText: str
    targetCol: Optional[str] = None


def calculate_psi_and_bins(baseline: np.ndarray, drifted: np.ndarray, num_bins: int = 10):
    """Calculates PSI and histogram bins for distribution visualization."""
    if len(baseline) == 0 or len(drifted) == 0:
        return 0.0, []

    min_val = float(np.min(baseline))
    max_val = float(np.max(baseline))
    if min_val == max_val:
        max_val += 1.0

    bins = np.linspace(min_val, max_val, num_bins + 1)
    base_counts, _ = np.histogram(baseline, bins=bins)
    drift_counts, _ = np.histogram(drifted, bins=bins)

    eps = 1e-4
    base_pct = (base_counts + eps) / (len(baseline) + eps * num_bins)
    drift_pct = (drift_counts + eps) / (len(drifted) + eps * num_bins)

    psi_val = float(np.sum((drift_pct - base_pct) * np.log(drift_pct / base_pct)))

    bin_data = []
    for i in range(num_bins):
        bin_data.append({
            "binLabel": f"{round(bins[i], 1)} - {round(bins[i+1], 1)}",
            "binStart": round(float(bins[i]), 2),
            "binEnd": round(float(bins[i+1]), 2),
            "baselineFrequency": round(float(base_pct[i]), 3),
            "driftedFrequency": round(float(drift_pct[i]), 3)
        })

    return round(psi_val, 4), bin_data


def prepare_data(rows: List[Dict[str, Any]], target_col: str, test_ratio: float = 0.25, seed: int = 42):
    df = pd.DataFrame(rows)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    # Drop common identifier columns
    drop_cols = [c for c in ['id', 'customer_id', 'Unnamed: 0'] if c in df.columns]
    df = df.drop(columns=drop_cols)

    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])

    # Convert target to binary classification
    if y_raw.dtype == 'object' or not np.issubdtype(y_raw.dtype, np.number):
        unique_targets = list(y_raw.dropna().unique())
        if len(unique_targets) == 2:
            target_map = {unique_targets[0]: 0, unique_targets[1]: 1}
            y = y_raw.map(target_map).fillna(0).astype(int)
        else:
            y = (y_raw.astype(str).str.lower().isin(['1', 'true', 'yes', 'y', 'positive'])).astype(int)
    else:
        # If numeric continuous or binary
        unique_vals = np.unique(y_raw.dropna())
        if len(unique_vals) <= 2:
            y = (y_raw >= np.max(unique_vals) if len(unique_vals) > 1 else y_raw >= 1).astype(int)
        else:
            y = (y_raw >= y_raw.median()).astype(int)

    # Encode categorical features via pandas get_dummies
    X = pd.get_dummies(X_raw, drop_first=True)
    # Ensure all numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    feature_names = list(X.columns)

    strat = y.values if len(np.unique(y.values)) > 1 and len(y) >= 10 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_ratio, random_state=seed, stratify=strat
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, X_test, y.values


@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "engine": "FastAPI Python ML Core (Scikit-Learn, Optuna & SHAP)",
        "version": "1.4.0",
        "frameworks": ["fastapi", "scikit-learn", "optuna", "pandas", "numpy", "scipy", "shap"]
    }


@app.post("/api/datasets/parse")
def parse_csv_endpoint(payload: ParseCsvRequest):
    try:
        csv_io = io.StringIO(payload.csvText)
        df = pd.read_csv(csv_io)
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="Empty CSV provided")

        chosen_target = payload.targetCol if payload.targetCol and payload.targetCol in df.columns else df.columns[-1]

        columns_summary = []
        for col in df.columns:
            non_null_count = int(df[col].notnull().sum())
            null_count = int(df[col].isnull().sum())
            unique_count = int(df[col].nunique())
            is_num = pd.api.types.is_numeric_dtype(df[col])

            col_info = {
                "name": col,
                "type": "numerical" if is_num else "categorical",
                "nonNullCount": non_null_count,
                "nullCount": null_count,
                "uniqueCount": unique_count,
                "sampleValues": [str(v) for v in df[col].dropna().unique()[:5]]
            }
            if is_num and non_null_count > 0:
                col_info["mean"] = round(float(df[col].mean()), 3)
                col_info["std"] = round(float(df[col].std(ddof=0)), 3)
                col_info["min"] = round(float(df[col].min()), 3)
                col_info["max"] = round(float(df[col].max()), 3)

            columns_summary.append(col_info)

        head_rows = df.head(15).fillna("").to_dict(orient="records")

        return {
            "summary": {
                "id": f"uploaded_{int(time.time())}",
                "name": "Uploaded Dataset (Python Evaluated)",
                "rowCount": len(df),
                "columnCount": len(df.columns),
                "columns": columns_summary,
                "targetColumn": chosen_target,
                "taskType": "classification",
                "features": [c for c in df.columns if c != chosen_target and c != "id"],
                "headRows": head_rows,
                "missingValuesTotal": int(df.isnull().sum().sum())
            },
            "rawCsv": payload.csvText
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train")
@app.post("/api/train")
def train_model(payload: TrainRequest):
    try:
        target_name = payload.get_target()
        test_ratio = payload.get_test_ratio()
        seed = payload.get_seed()
        X_train, X_test, y_train, y_test, feature_names, _, _ = prepare_data(
            payload.rows, target_name, test_ratio, seed
        )

        params = payload.hyperparameters or {}
        start_time = time.perf_counter()

        if payload.modelType == "logistic_regression":
            model = LogisticRegression(
                C=float(params.get("l2Reg", params.get("C", 1.0))),
                max_iter=int(params.get("maxIter", params.get("max_iter", 200))),
                random_state=payload.seed
            )
        elif payload.modelType == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=int(params.get("nEstimators", params.get("n_estimators", 40))),
                learning_rate=float(params.get("learningRate", params.get("learning_rate", 0.1))),
                max_depth=int(params.get("maxDepth", params.get("max_depth", 4))),
                random_state=payload.seed
            )
        else:
            model = RandomForestClassifier(
                n_estimators=int(params.get("nEstimators", params.get("n_estimators", 40))),
                max_depth=int(params.get("maxDepth", params.get("max_depth", 6))),
                min_samples_split=int(params.get("minSamplesSplit", params.get("min_samples_split", 4))),
                random_state=payload.seed
            )

        model.fit(X_train, y_train)
        train_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Measure inference latency with high-precision timer
        lat_start = time.perf_counter()
        y_prob = model.predict_proba(X_test)[:, 1]
        latency_ms = round((time.perf_counter() - lat_start) * 1000, 2)
        y_pred = (y_prob >= 0.5).astype(int)

        acc = float(accuracy_score(y_test, y_pred)) if not np.isnan(accuracy_score(y_test, y_pred)) else 0.0
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.5
        except Exception:
            auc = 0.5
        if np.isnan(auc):
            auc = 0.5

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        # ROC Curve (Sanitize NaN and Inf for JSON compliance)
        roc_pts = []
        if len(np.unique(y_test)) > 1:
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            step = max(1, len(fpr) // 20)
            for i in range(0, len(fpr), step):
                f_val = 0.0 if np.isnan(fpr[i]) else round(float(fpr[i]), 3)
                t_val = 0.0 if np.isnan(tpr[i]) else round(float(tpr[i]), 3)
                thresh_val = 1.0 if (np.isinf(thresholds[i]) or np.isnan(thresholds[i])) else round(float(thresholds[i]), 3)
                roc_pts.append({
                    "fpr": f_val,
                    "tpr": t_val,
                    "threshold": thresh_val
                })
            if not roc_pts or roc_pts[-1]["fpr"] != 1.0:
                roc_pts.append({"fpr": 1.0, "tpr": 1.0, "threshold": 0.0})
        else:
            roc_pts = [
                {"fpr": 0.0, "tpr": 0.0, "threshold": 1.0},
                {"fpr": 1.0, "tpr": 1.0, "threshold": 0.0}
            ]

        # Feature importances
        if hasattr(model, "feature_importances_"):
            raw_imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            raw_imp = np.abs(model.coef_[0])
        else:
            raw_imp = np.ones(len(feature_names)) / len(feature_names)

        total_imp = float(np.sum(raw_imp)) or 1.0
        importances = [
            {"feature": feature_names[i], "importance": round(float(raw_imp[i] / total_imp), 3)}
            for i in range(len(feature_names))
        ]
        importances.sort(key=lambda x: x["importance"], reverse=True)

        # Actual Model Complexity & Memory
        model_bytes = len(pickle.dumps(model))
        model_kb = round(model_bytes / 1024, 1)

        names = {
            "logistic_regression": "Logistic Regression (Scikit-Learn)",
            "random_forest": "Random Forest Classifier (Scikit-Learn)",
            "gradient_boosting": "Gradient Boosting Classifier (Scikit-Learn)"
        }

        return {
            "model": {
                "id": f"{payload.modelType}_{int(time.time() * 1000)}",
                "modelType": payload.modelType,
                "modelName": names.get(payload.modelType, "Classifier (Scikit-Learn)"),
                "taskType": "classification",
                "trainedAt": pd.Timestamp.now().isoformat(),
                "hyperparameters": params,
                "metrics": {
                    "accuracy": round(acc, 3),
                    "precision": round(prec, 3),
                    "recall": round(rec, 3),
                    "f1Score": round(f1, 3),
                    "rocAuc": round(auc, 3),
                    "latencyMs": latency_ms,
                    "trainTimeMs": train_time_ms,
                    "modelComplexity": {
                        "parameterCount": int(len(feature_names) * (getattr(model, 'n_estimators', 10))),
                        "memoryKb": model_kb
                    },
                    "confusionMatrix": {
                        "truePositive": int(tp),
                        "falsePositive": int(fp),
                        "trueNegative": int(tn),
                        "falseNegative": int(fn),
                        "classes": ["Negative (0)", "Positive (1)"]
                    },
                    "rocCurve": roc_pts
                },
                "featureImportances": importances,
                "isOptimized": False
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
@app.post("/api/optimize")
def optimize_model(payload: OptimizeRequest):
    try:
        target_name = payload.get_target()
        X_train, X_test, y_train, y_test, feature_names, _, _ = prepare_data(
            payload.rows, target_name
        )

        trials_log = []
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            t_start = time.perf_counter()
            if payload.modelType == "logistic_regression":
                c_val = trial.suggest_float("C", 1e-3, 10.0, log=True)
                clf = LogisticRegression(C=c_val, max_iter=200, random_state=42)
            elif payload.modelType == "gradient_boosting":
                n_est = trial.suggest_int("n_estimators", 15, 60, step=5)
                lr = trial.suggest_float("learning_rate", 0.02, 0.25)
                depth = trial.suggest_int("max_depth", 2, 5)
                clf = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, max_depth=depth, random_state=42)
            else:
                n_est = trial.suggest_int("n_estimators", 20, 80, step=10)
                depth = trial.suggest_int("max_depth", 3, 10)
                min_s = trial.suggest_int("min_samples_split", 2, 8)
                clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, min_samples_split=min_s, random_state=42)

            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)

            if payload.objectiveMetric == "accuracy":
                score = accuracy_score(y_test, pred)
            elif payload.objectiveMetric == "rocAuc" and len(np.unique(y_test)) > 1:
                prob = clf.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, prob)
            else:
                score = f1_score(y_test, pred, zero_division=0)

            elapsed = round((time.perf_counter() - t_start) * 1000, 1)

            trials_log.append({
                "trialNumber": len(trials_log) + 1,
                "params": trial.params,
                "score": round(float(score), 3),
                "durationMs": elapsed,
                "status": "COMPLETED"
            })
            return score

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=min(30, max(5, payload.nTrials)))

        best_score = float(study.best_value)
        baseline_score = trials_log[0]["score"] if trials_log else best_score * 0.9

        # Fit final optimized model with the best parameters
        best_params = study.best_params
        if payload.modelType == "logistic_regression":
            best_clf = LogisticRegression(C=best_params["C"], max_iter=200, random_state=42)
        elif payload.modelType == "gradient_boosting":
            best_clf = GradientBoostingClassifier(
                n_estimators=best_params["n_estimators"],
                learning_rate=best_params["learning_rate"],
                max_depth=best_params["max_depth"],
                random_state=42
            )
        else:
            best_clf = RandomForestClassifier(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                min_samples_split=best_params["min_samples_split"],
                random_state=42
            )

        best_clf.fit(X_train, y_train)
        lat_start = time.perf_counter()
        opt_probs = best_clf.predict_proba(X_test)[:, 1]
        lat_ms = round((time.perf_counter() - lat_start) * 1000, 2)
        opt_preds = (opt_probs >= 0.5).astype(int)

        opt_acc = float(accuracy_score(y_test, opt_preds))
        opt_f1 = float(f1_score(y_test, opt_preds, zero_division=0))
        opt_prec = float(precision_score(y_test, opt_preds, zero_division=0))
        opt_rec = float(recall_score(y_test, opt_preds, zero_division=0))
        opt_auc = float(roc_auc_score(y_test, opt_probs)) if len(np.unique(y_test)) > 1 else 0.5

        # Feature importances
        if hasattr(best_clf, "feature_importances_"):
            raw_imp = best_clf.feature_importances_
        elif hasattr(best_clf, "coef_"):
            raw_imp = np.abs(best_clf.coef_[0])
        else:
            raw_imp = np.ones(len(feature_names)) / len(feature_names)

        total_imp = float(np.sum(raw_imp)) or 1.0
        opt_importances = [
            {"feature": feature_names[i], "importance": round(float(raw_imp[i] / total_imp), 3)}
            for i in range(len(feature_names))
        ]
        opt_importances.sort(key=lambda x: x["importance"], reverse=True)

        cm = confusion_matrix(y_test, opt_preds, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        names = {
            "logistic_regression": "Logistic Regression (Optuna Tuned)",
            "random_forest": "Random Forest (Optuna Tuned)",
            "gradient_boosting": "Gradient Boosting (Optuna Tuned)"
        }

        optimized_model = {
            "id": f"opt_{payload.modelType}_{int(time.time() * 1000)}",
            "modelType": payload.modelType,
            "modelName": names.get(payload.modelType, "Optimized Model"),
            "taskType": "classification",
            "trainedAt": pd.Timestamp.now().isoformat(),
            "hyperparameters": best_params,
            "metrics": {
                "accuracy": round(opt_acc, 3),
                "precision": round(opt_prec, 3),
                "recall": round(opt_rec, 3),
                "f1Score": round(opt_f1, 3),
                "rocAuc": round(opt_auc, 3),
                "latencyMs": lat_ms,
                "trainTimeMs": int(sum(t["durationMs"] for t in trials_log)),
                "modelComplexity": {
                    "parameterCount": int(len(feature_names) * (getattr(best_clf, 'n_estimators', 10))),
                    "memoryKb": round(len(pickle.dumps(best_clf)) / 1024, 1)
                },
                "confusionMatrix": {
                    "truePositive": int(tp),
                    "falsePositive": int(fp),
                    "trueNegative": int(tn),
                    "falseNegative": int(fn),
                    "classes": ["Negative (0)", "Positive (1)"]
                },
                "rocCurve": []
            },
            "featureImportances": opt_importances,
            "isOptimized": True
        }

        return {
            "modelType": payload.modelType,
            "objectiveMetric": payload.objectiveMetric,
            "nTrials": len(trials_log),
            "bestTrialNumber": study.best_trial.number + 1,
            "bestParams": best_params,
            "bestScore": round(best_score, 3),
            "baselineScore": round(baseline_score, 3),
            "improvementDelta": round(best_score - baseline_score, 3),
            "trials": trials_log,
            "optimizedModel": optimized_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/drift")
@app.post("/api/drift")
def detect_drift(payload: DriftRequest):
    try:
        target_name = payload.get_target()
        X_train, X_test, y_train, y_test, feature_names, _, _ = prepare_data(
            payload.rows, target_name
        )

        if payload.modelType == "logistic_regression":
            clf = LogisticRegression(random_state=42)
        elif payload.modelType == "gradient_boosting":
            clf = GradientBoostingClassifier(n_estimators=30, max_depth=4, random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=35, max_depth=5, random_state=42)

        clf.fit(X_train, y_train)

        # Baseline performance
        base_preds = clf.predict(X_test)
        base_acc = float(accuracy_score(y_test, base_preds))
        base_f1 = float(f1_score(y_test, base_preds, zero_division=0))

        # Apply shifts
        cfg = payload.shiftConfig
        shifted_X_test = X_test.copy()
        drift_metrics = []
        distribution_comparison = {}
        drift_count = 0

        for col_idx, feat in enumerate(feature_names):
            if not cfg.affectedFeatures or feat in cfg.affectedFeatures:
                shifted_col = (shifted_X_test[:, col_idx] + (cfg.meanShiftPct / 100.0)) * cfg.varianceScale
                if cfg.noiseLevel > 0:
                    noise = np.random.normal(0, cfg.noiseLevel, len(shifted_col))
                    shifted_col += noise
                shifted_X_test[:, col_idx] = shifted_col

            base_col = X_test[:, col_idx]
            drift_col = shifted_X_test[:, col_idx]

            ks_res = ks_2samp(base_col, drift_col)
            w_dist = wasserstein_distance(base_col, drift_col)
            psi, bins = calculate_psi_and_bins(base_col, drift_col)

            is_drift = bool(psi > 0.1 or (ks_res.pvalue < 0.05 and ks_res.statistic > 0.15))
            if is_drift:
                drift_count += 1

            drift_metrics.append({
                "feature": feat,
                "baselineMean": round(float(np.mean(base_col)), 2),
                "driftedMean": round(float(np.mean(drift_col)), 2),
                "meanShiftPct": round(float(cfg.meanShiftPct), 1),
                "baselineStd": round(float(np.std(base_col)), 2),
                "driftedStd": round(float(np.std(drift_col)), 2),
                "wassersteinDistance": round(float(w_dist), 3),
                "ksStatistic": round(float(ks_res.statistic), 3),
                "ksPValue": round(float(ks_res.pvalue), 4),
                "psi": psi,
                "driftDetected": is_drift,
                "severity": "high" if psi > 0.25 else "moderate" if psi > 0.1 else "low"
            })
            distribution_comparison[feat] = bins

        # Evaluate performance on shifted data
        shifted_preds = clf.predict(shifted_X_test)
        shifted_acc = float(accuracy_score(y_test, shifted_preds))
        shifted_f1 = float(f1_score(y_test, shifted_preds, zero_division=0))

        flips = np.sum(base_preds != shifted_preds)
        flip_pct = round(float((flips / len(y_test)) * 100), 1)

        acc_drop = round(float(((base_acc - shifted_acc) / (base_acc or 1)) * 100), 1)
        f1_drop = round(float(((base_f1 - shifted_f1) / (base_f1 or 1)) * 100), 1)

        status = "ROBUST"
        if acc_drop > 20 or f1_drop > 25:
            status = "CRITICAL_FAILURE"
        elif acc_drop > 8 or f1_drop > 10:
            status = "MODERATE_DEGRADATION"

        names = {
            "logistic_regression": "Logistic Regression",
            "random_forest": "Random Forest Classifier",
            "gradient_boosting": "Gradient Boosting Classifier"
        }

        return {
            "isSimulated": True,
            "sampleCountBaseline": len(X_test),
            "sampleCountDrifted": len(shifted_X_test),
            "featureDriftMetrics": drift_metrics,
            "overallDriftIndex": round(float(np.mean([m["psi"] for m in drift_metrics])), 3),
            "driftedFeaturesCount": drift_count,
            "distributionComparison": distribution_comparison,
            "modelRobustness": {
                "modelName": names.get(payload.modelType, "Random Forest Classifier"),
                "baselineAccuracy": round(base_acc, 3),
                "driftedAccuracy": round(shifted_acc, 3),
                "accuracyDropPct": acc_drop,
                "baselineF1": round(base_f1, 3),
                "driftedF1": round(shifted_f1, 3),
                "f1DropPct": f1_drop,
                "predictionFlipPct": flip_pct,
                "status": status
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
@app.post("/api/explain")
def explain_predictions(payload: ExplainRequest):
    try:
        target_name = payload.get_target()
        X_train, X_test, y_train, y_test, feature_names, raw_X_test, _ = prepare_data(
            payload.rows, target_name
        )

        if payload.modelType == "logistic_regression":
            model = LogisticRegression(random_state=42)
        elif payload.modelType == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=30, max_depth=4, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=35, max_depth=5, random_state=42)

        model.fit(X_train, y_train)

        # Baseline expected value E[f(x)]
        all_probs = model.predict_proba(X_test)[:, 1]
        base_value = float(np.mean(all_probs))

        # Real SHAP Explainer
        try:
            if payload.modelType in ["random_forest", "gradient_boosting"]:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                # Handle binary classification 2D or 3D array
                if isinstance(shap_values, list):
                    shap_matrix = shap_values[1]
                elif shap_values.ndim == 3:
                    shap_matrix = shap_values[:, :, 1]
                else:
                    shap_matrix = shap_values
                method_name = "TreeSHAP"
            else:
                # Linear / Logit
                explainer = shap.LinearExplainer(model, X_train)
                shap_matrix = explainer.shap_values(X_test)
                method_name = "LinearSHAP"
        except Exception:
            # Fallback to Permutation Importance if SHAP C-tree fails on specific numpy build
            perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
            shap_matrix = np.tile(perm.importances_mean, (len(X_test), 1))
            method_name = "PermutationImportance"

        # Global feature importance
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        total_shap = float(np.sum(mean_abs_shap)) or 1.0

        global_importances = []
        for i, feat in enumerate(feature_names):
            imp_val = float(mean_abs_shap[i])
            global_importances.append({
                "feature": feat,
                "importance": round(imp_val, 3),
                "relativePct": round((imp_val / total_shap) * 100, 1)
            })
        global_importances.sort(key=lambda x: x["importance"], reverse=True)

        # Local sample explanation
        sample_idx = max(0, min(len(X_test) - 1, payload.sampleIndex))
        sample_shap = shap_matrix[sample_idx]
        sample_prob = float(all_probs[sample_idx])
        pred_label = int(sample_prob >= 0.5)
        actual_label = int(y_test[sample_idx])

        # Raw sample feature values
        raw_row = payload.rows[sample_idx] if sample_idx < len(payload.rows) else {}

        local_contributions = []
        running_prob = base_value

        sorted_indices = np.argsort(-np.abs(sample_shap))
        for idx in sorted_indices:
            feat_name = feature_names[idx]
            attr_val = round(float(sample_shap[idx]), 3)
            running_prob = max(0.0, min(1.0, running_prob + attr_val))
            local_contributions.append({
                "feature": feat_name,
                "featureValue": raw_row.get(feat_name, round(float(X_test[sample_idx, idx]), 2)),
                "attribution": attr_val,
                "runningProbability": round(running_prob, 3)
            })

        return {
            "method": method_name,
            "baseValue": round(base_value, 3),
            "globalImportances": global_importances,
            "samplePrediction": {
                "sampleIndex": sample_idx,
                "actualLabel": f"Positive ({actual_label})" if actual_label == 1 else f"Negative ({actual_label})",
                "predictedProbability": round(sample_prob, 3),
                "predictedLabel": "Positive (1)" if pred_label == 1 else "Negative (0)",
                "localContributions": local_contributions
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
@app.post("/api/predict")
def predict_transactions(payload: PredictRequest):
    """
    Trains the requested model (Random Forest by default) on the verified trainingRows
    and executes real model inference on unseen testRows without using targetCol as an input feature.
    """
    try:
        target_name = payload.get_target()
        if not payload.trainingRows or len(payload.trainingRows) < 5:
            raise HTTPException(status_code=400, detail="Sufficient training rows required to train predictor model")
        if not payload.testRows or len(payload.testRows) == 0:
            raise HTTPException(status_code=400, detail="At least one test row is required for prediction")

        train_df = pd.DataFrame(payload.trainingRows)
        if target_name not in train_df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_name}' not found in training dataset")

        # Exclude IDs and target column from input features
        drop_cols = [c for c in ['id', 'customer_id', 'transaction_id', 'Unnamed: 0'] if c in train_df.columns]
        train_df = train_df.drop(columns=drop_cols)

        y_train_raw = train_df[target_name]
        X_train_raw = train_df.drop(columns=[target_name])

        # Convert target to binary classification
        if y_train_raw.dtype == 'object' or not np.issubdtype(y_train_raw.dtype, np.number):
            unique_targets = list(y_train_raw.dropna().unique())
            if len(unique_targets) == 2:
                target_map = {unique_targets[0]: 0, unique_targets[1]: 1}
                y_train = y_train_raw.map(target_map).fillna(0).astype(int)
            else:
                y_train = (y_train_raw.astype(str).str.lower().isin(['1', 'true', 'yes', 'y', 'positive', 'fraud'])).astype(int)
        else:
            unique_vals = np.unique(y_train_raw.dropna())
            if len(unique_vals) <= 2:
                y_train = (y_train_raw >= np.max(unique_vals) if len(unique_vals) > 1 else y_train_raw >= 1).astype(int)
            else:
                y_train = (y_train_raw >= y_train_raw.median()).astype(int)

        # Build feature matrix for training
        X_train_encoded = pd.get_dummies(X_train_raw, drop_first=True)
        X_train_encoded = X_train_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)
        feature_columns = list(X_train_encoded.columns)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_encoded.values)

        # Fit model
        params = payload.hyperparameters or {}
        if payload.modelType == "logistic_regression":
            model = LogisticRegression(
                C=float(params.get("l2Reg", params.get("C", 1.0))),
                max_iter=int(params.get("maxIter", params.get("max_iter", 200))),
                random_state=42
            )
        elif payload.modelType == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=int(params.get("nEstimators", params.get("n_estimators", 40))),
                learning_rate=float(params.get("learningRate", params.get("learning_rate", 0.1))),
                max_depth=int(params.get("maxDepth", params.get("max_depth", 4))),
                random_state=42
            )
        else:
            model = RandomForestClassifier(
                n_estimators=int(params.get("nEstimators", params.get("n_estimators", 40))),
                max_depth=int(params.get("maxDepth", params.get("max_depth", 6))),
                min_samples_split=int(params.get("minSamplesSplit", params.get("min_samples_split", 4))),
                random_state=42
            )

        model.fit(X_train_scaled, y_train.values)

        # Process unseen test rows - CRITICAL: Strip target column if present in unseen data!
        test_df = pd.DataFrame(payload.testRows)
        original_test_cols = list(test_df.columns)
        has_ground_truth = target_name in test_df.columns

        actual_ground_truth = []
        if has_ground_truth:
            actual_ground_truth = test_df[target_name].tolist()
            test_df_features = test_df.drop(columns=[target_name])
        else:
            test_df_features = test_df.copy()

        # Remove identifiers
        test_df_features = test_df_features.drop(columns=[c for c in drop_cols if c in test_df_features.columns])

        # Strict feature validation: check that required training feature columns exist in test data
        expected_raw_features = [c for c in X_train_raw.columns]
        missing_features = [c for c in expected_raw_features if c not in test_df_features.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded prediction CSV is missing required feature column(s): {', '.join(missing_features)}. Expected columns: {', '.join(expected_raw_features)}"
            )

        # Check types for numeric columns
        for c in X_train_raw.columns:
            if pd.api.types.is_numeric_dtype(X_train_raw[c]):
                # verify test column contains numeric values
                coerced = pd.to_numeric(test_df_features[c], errors='coerce')
                if coerced.isnull().all() and len(test_df_features) > 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Column '{c}' expects numeric values, but non-numeric data was provided in the prediction CSV."
                    )

        # One-hot encode test rows matching train columns exactly
        test_encoded = pd.get_dummies(test_df_features, drop_first=True)
        test_encoded = test_encoded.reindex(columns=feature_columns, fill_value=0)
        test_encoded = test_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)

        test_scaled = scaler.transform(test_encoded.values)

        # Model inference
        t_start = time.perf_counter()
        probs = model.predict_proba(test_scaled)[:, 1]
        inference_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        preds = (probs >= 0.5).astype(int)

        results = []
        fraud_count = 0
        legit_count = 0

        for idx in range(len(payload.testRows)):
            is_fraud = int(preds[idx]) == 1
            fraud_prob = round(float(probs[idx]), 4)
            if is_fraud:
                fraud_count += 1
            else:
                legit_count += 1

            item: Dict[str, Any] = {
                "rowIndex": idx + 1,
                "inputFeatures": {k: payload.testRows[idx].get(k) for k in payload.testRows[idx] if k != target_name},
                "predictedClass": "Fraud" if is_fraud else "Legitimate",
                "predictedLabel": 1 if is_fraud else 0,
                "fraudProbability": fraud_prob,
                "confidenceScore": round(float(fraud_prob if is_fraud else (1.0 - fraud_prob)) * 100, 1),
                "isVerifiedOutcome": False
            }

            if has_ground_truth and idx < len(actual_ground_truth):
                verified_val = actual_ground_truth[idx]
                verified_num = 1 if str(verified_val).lower() in ['1', 'true', 'yes', 'fraud'] else 0
                item["verifiedOutcome"] = "Fraud" if verified_num == 1 else "Legitimate"
                item["isCorrect"] = (item["predictedClass"] == item["verifiedOutcome"])
            else:
                item["verifiedOutcome"] = "Unverified (Unseen Transaction)"

            results.append(item)

        return {
            "modelUsed": "Random Forest Classifier (Scikit-Learn)",
            "totalSamples": len(results),
            "fraudCount": fraud_count,
            "legitimateCount": legit_count,
            "fraudRatePct": round((fraud_count / (len(results) or 1)) * 100, 1),
            "inferenceLatencyMs": inference_latency_ms,
            "predictions": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
