import React, { useState, useEffect } from 'react';
import {
  Settings,
  Server,
  Terminal,
  Copy,
  Check,
  Activity,
  Code2,
  Sliders,
  ShieldCheck,
  Cpu
} from 'lucide-react';
import {
  getEngineSettings,
  saveEngineSettings,
  checkFastApiHealth,
  EngineSettings
} from '../lib/api-client';

export const SettingsView: React.FC = () => {
  const [settings, setSettings] = useState<EngineSettings>(getEngineSettings());
  const [fastApiUrlInput, setFastApiUrlInput] = useState(settings.fastApiUrl);
  const [pingStatus, setPingStatus] = useState<{ testing: boolean; result?: string; error?: string; latency?: number }>({
    testing: false
  });
  const [copiedCode, setCopiedCode] = useState(false);
  const [copiedBash, setCopiedBash] = useState(false);

  // Global thresholds
  const [testSplit, setTestSplit] = useState(0.25);
  const [randomSeed, setRandomSeed] = useState(42);
  const [psiThreshold, setPsiThreshold] = useState(0.25);

  const handleSaveMode = (mode: 'integrated' | 'fastapi') => {
    const updated: EngineSettings = { ...settings, mode, fastApiUrl: fastApiUrlInput };
    setSettings(updated);
    saveEngineSettings(updated);
  };

  const handleTestConnection = async () => {
    setPingStatus({ testing: true });
    const start = performance.now();
    const res = await checkFastApiHealth(fastApiUrlInput);
    const elapsed = Math.round(performance.now() - start);

    if (res.healthy) {
      setPingStatus({
        testing: false,
        result: `Connected to ${res.details?.engine || 'FastAPI Server'} (v${res.details?.version || '1.0'})`,
        latency: elapsed
      });
      // automatically save if working
      saveEngineSettings({ ...settings, fastApiUrl: fastApiUrlInput });
    } else {
      setPingStatus({
        testing: false,
        error: res.error || 'Connection failed',
        latency: elapsed
      });
    }
  };

  const pythonScript = `"""
DriftForge AI - Standalone Reproduction Pipeline
Trained with Scikit-learn, Optuna, and Scipy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from scipy.stats import ks_2samp
import optuna

# 1. Load Data
df = pd.read_csv("dataset.csv")
target_col = "churn"  # or target variable
X = pd.get_dummies(df.drop(columns=[target_col, "id"], errors="ignore"), drop_first=True)
y = (df[target_col] >= df[target_col].median()).astype(int)

# 2. Train / Test Split (${(testSplit * 100).toFixed(0)}% split, seed=${randomSeed})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=${testSplit}, random_state=${randomSeed}, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training
clf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=${randomSeed})
clf.fit(X_train_scaled, y_train)

# 4. Evaluation
preds = clf.predict(X_test_scaled)
print("Classification Report:\\n", classification_report(y_test, preds))

# 5. Drift Detection Test (KS-Test)
shifted_feature = X_test_scaled[:, 0] + 0.35  # simulate +35% mean shift
ks_res = ks_2samp(X_test_scaled[:, 0], shifted_feature)
print(f"KS Statistic: {ks_res.statistic:.3f}, p-value: {ks_res.pvalue:.4f}")
`;

  const copyToClipboard = (text: string, type: 'code' | 'bash') => {
    navigator.clipboard.writeText(text);
    if (type === 'code') {
      setCopiedCode(true);
      setTimeout(() => setCopiedCode(false), 2000);
    } else {
      setCopiedBash(true);
      setTimeout(() => setCopiedBash(false), 2000);
    }
  };

  return (
    <div className="space-y-6">
      {/* Backend Engine Orchestration Card */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <Server className="h-4 w-4 text-cyan-400" />
              Machine Learning Backend Engine Configuration
            </h3>
            <p className="text-xs text-slate-400">
              Choose between the self-contained integrated MLOps engine or an external Python FastAPI microservice
            </p>
          </div>

          <span className="text-xs font-mono text-cyan-400">Dual-Engine Architecture</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
          {/* Integrated Engine Option */}
          <div
            onClick={() => handleSaveMode('integrated')}
            className={`p-4 rounded-xl border cursor-pointer transition ${
              settings.mode === 'integrated'
                ? 'bg-cyan-950/30 border-cyan-500/80 shadow-sm'
                : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-white flex items-center gap-2">
                <Cpu className="h-4 w-4 text-cyan-400" />
                Integrated Engine (Zero Setup)
              </span>
              {settings.mode === 'integrated' && (
                <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-900/80 text-cyan-300 border border-cyan-700">
                  ACTIVE
                </span>
              )}
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Executes high-precision mathematical models (Gradient Descent Logistic Regression, Random Forest Ensembles, Kolmogorov-Smirnov test, Wasserstein Distance, Population Stability Index, and Shapley attribution) with instant zero-latency responses.
            </p>
          </div>

          {/* External FastAPI Option */}
          <div
            onClick={() => handleSaveMode('fastapi')}
            className={`p-4 rounded-xl border cursor-pointer transition ${
              settings.mode === 'fastapi'
                ? 'bg-purple-950/30 border-purple-500/80 shadow-sm'
                : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-white flex items-center gap-2">
                <Terminal className="h-4 w-4 text-purple-400" />
                External Python FastAPI Core
              </span>
              {settings.mode === 'fastapi' && (
                <span className="text-[10px] px-2 py-0.5 rounded bg-purple-900/80 text-purple-300 border border-purple-700">
                  ACTIVE
                </span>
              )}
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Direct connection to a local or remote Python FastAPI backend running Scikit-learn, Optuna Bayesian tuning, Pandas, and SHAP. Fallbacks to Integrated Core if temporarily unreachable.
            </p>
          </div>
        </div>

        {/* FastAPI Server URL & Health Check */}
        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
          <label className="text-xs font-medium text-slate-300">FastAPI Host URL Endpoint</label>
          <div className="flex flex-col sm:flex-row gap-2">
            <input
              type="text"
              value={fastApiUrlInput}
              onChange={e => setFastApiUrlInput(e.target.value)}
              placeholder="http://localhost:8000"
              className="flex-1 px-3.5 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs text-slate-200 font-mono focus:outline-none focus:border-cyan-500"
            />
            <button
              onClick={handleTestConnection}
              disabled={pingStatus.testing}
              className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-medium text-slate-200 flex items-center justify-center gap-2 transition cursor-pointer disabled:opacity-50"
            >
              {pingStatus.testing ? (
                <>
                  <Activity className="h-3.5 w-3.5 animate-spin text-cyan-400" />
                  <span>Pinging...</span>
                </>
              ) : (
                <>
                  <Activity className="h-3.5 w-3.5 text-cyan-400" />
                  <span>Test Connection</span>
                </>
              )}
            </button>
          </div>

          {pingStatus.result && (
            <div className="p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-800 text-xs text-emerald-300 flex items-center justify-between">
              <span>✓ {pingStatus.result}</span>
              <span className="font-mono text-[11px]">{pingStatus.latency}ms ping</span>
            </div>
          )}

          {pingStatus.error && (
            <div className="p-2.5 rounded-lg bg-rose-950/40 border border-rose-800 text-xs text-rose-300 flex items-center justify-between">
              <span>✕ {pingStatus.error}</span>
              <span className="text-[11px] text-slate-400">Integrated Core active as fallback</span>
            </div>
          )}
        </div>
      </div>

      {/* Global Configuration Parameters */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
        <h4 className="text-sm font-semibold text-white flex items-center gap-2">
          <Sliders className="h-4 w-4 text-purple-400" />
          Global Experiment & Drift Detection Defaults
        </h4>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-1">
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Default Test Partition Ratio</span>
              <span className="font-mono text-cyan-400">{(testSplit * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.4"
              step="0.05"
              value={testSplit}
              onChange={e => setTestSplit(Number(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer"
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Deterministic Seed</span>
              <span className="font-mono text-cyan-400">{randomSeed}</span>
            </div>
            <input
              type="number"
              value={randomSeed}
              onChange={e => setRandomSeed(Number(e.target.value))}
              className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-xs text-slate-200 font-mono"
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">PSI Alert Threshold</span>
              <span className="font-mono text-rose-400">{psiThreshold}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.5"
              step="0.05"
              value={psiThreshold}
              onChange={e => setPsiThreshold(Number(e.target.value))}
              className="w-full accent-rose-400 cursor-pointer"
            />
          </div>
        </div>
      </div>

      {/* Local Python FastAPI Startup Instructions */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-semibold text-white flex items-center gap-2">
            <Terminal className="h-4 w-4 text-emerald-400" />
            Local Python FastAPI Server Launcher
          </h4>
          <button
            onClick={() => copyToClipboard('cd backend && pip install -r requirements.txt && python run.py', 'bash')}
            className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1 cursor-pointer font-medium"
          >
            {copiedBash ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
            <span>{copiedBash ? 'Copied Shell Command!' : 'Copy Commands'}</span>
          </button>
        </div>

        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-slate-300 space-y-2 overflow-x-auto">
          <div className="text-slate-500"># 1. Navigate to backend directory and setup environment</div>
          <div>cd backend</div>
          <div>python3 -m venv venv &amp;&amp; source venv/bin/activate</div>
          <div className="text-slate-500 pt-1"># 2. Install requirements (FastAPI, Scikit-learn, Optuna, SHAP)</div>
          <div>pip install -r requirements.txt</div>
          <div className="text-slate-500 pt-1"># 3. Start server on port 8000</div>
          <div className="text-emerald-400">uvicorn main:app --host 0.0.0.0 --port 8000 --reload</div>
        </div>
      </div>

      {/* Standalone Scikit-Learn Python Reproduction Script */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-semibold text-white flex items-center gap-2">
            <Code2 className="h-4 w-4 text-indigo-400" />
            Standalone Scikit-Learn Reproduction Pipeline
          </h4>
          <button
            onClick={() => copyToClipboard(pythonScript, 'code')}
            className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1 cursor-pointer font-medium"
          >
            {copiedCode ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
            <span>{copiedCode ? 'Copied Script!' : 'Copy Python Code'}</span>
          </button>
        </div>

        <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-slate-300 overflow-x-auto max-h-72">
          {pythonScript}
        </pre>
      </div>
    </div>
  );
};
