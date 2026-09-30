import Papa from 'papaparse';
import {
  DatasetSummary,
  ModelType,
  TrainedModelResult,
  OptimizationResult,
  DriftAnalysisResult,
  ExplainabilityResult,
  BatchPredictionResponse
} from './ml-engine/types';
import { generateCustomerChurnDataset, generateCreditRiskDataset, generateHousingPricingDataset, generateCreditCardFraudDataset, buildColumnSummaries } from './ml-engine/sample-datasets';
import { preprocessTabularData, calculateClassificationMetrics } from './ml-engine/stats';
import { LogisticRegressionModel, RandomForestClassifierModel, GradientBoostingClassifierModel, ModelEstimator } from './ml-engine/algorithms';
import { runHyperparameterOptimization } from './ml-engine/optimizer';
import { simulateAndAnalyzeDrift, ShiftConfiguration } from './ml-engine/drift';
import { computeExplainability } from './ml-engine/explain';
import { predictTransactionsEngine } from './ml-engine/predict';

export interface EngineSettings {
  mode: 'integrated' | 'fastapi';
  fastApiUrl: string;
}

let currentSettings: EngineSettings = {
  mode: 'integrated',
  fastApiUrl: 'http://localhost:8000'
};

export function getEngineSettings(): EngineSettings {
  try {
    const saved = localStorage.getItem('driftforge_settings');
    if (saved) {
      currentSettings = JSON.parse(saved);
    }
  } catch (e) {
    // fallback
  }
  return currentSettings;
}

export function saveEngineSettings(settings: EngineSettings): void {
  currentSettings = settings;
  try {
    localStorage.setItem('driftforge_settings', JSON.stringify(settings));
  } catch (e) {
    // ignore
  }
}

export async function checkFastApiHealth(url: string = currentSettings.fastApiUrl): Promise<{ healthy: boolean; details?: any; error?: string }> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);
    const resp = await fetch(`${url.replace(/\/$/, '')}/health`, { signal: controller.signal });
    clearTimeout(timeoutId);
    if (resp.ok) {
      const data = await resp.json();
      return { healthy: true, details: data };
    }
    return { healthy: false, error: `HTTP ${resp.status}` };
  } catch (err: any) {
    return { healthy: false, error: err.message || 'Connection refused' };
  }
}

export async function fetchSampleDataset(type: 'churn' | 'credit' | 'housing' | 'fraud'): Promise<{ rawCsv: string; summary: DatasetSummary }> {
  try {
    const resp = await fetch(`/api/datasets/sample/${type}`);
    if (resp.ok) {
      return await resp.json();
    }
  } catch (e) {
    // fallback
  }

  // Client-side fallback
  if (type === 'churn') return generateCustomerChurnDataset();
  if (type === 'credit') return generateCreditRiskDataset();
  if (type === 'fraud') return generateCreditCardFraudDataset();
  return generateHousingPricingDataset();
}

export async function parseCsvDataset(csvText: string, targetCol?: string): Promise<{ summary: DatasetSummary; rawCsv: string }> {
  try {
    const resp = await fetch('/api/datasets/parse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ csvText, targetCol })
    });
    if (resp.ok) {
      return await resp.json();
    }
  } catch (e) {
    // fallback
  }

  // Client-side fallback parsing with PapaParse RFC 4180
  const parsed = Papa.parse<Record<string, any>>(csvText, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: 'greedy'
  });

  if (parsed.errors && parsed.errors.length > 0 && parsed.data.length === 0) {
    throw new Error(parsed.errors[0].message || 'Failed to parse CSV');
  }

  const headers = (parsed.meta.fields || []).map(h => h.trim());
  const rows: Record<string, string | number>[] = [];

  for (const rawRow of parsed.data) {
    const row: Record<string, string | number> = {};
    for (const h of headers) {
      const val = rawRow[h];
      if (val === null || val === undefined || val === '') {
        row[h] = '';
      } else if (typeof val === 'number') {
        row[h] = val;
      } else {
        row[h] = String(val).trim();
      }
    }
    rows.push(row);
  }

  if (rows.length < 2) {
    throw new Error('CSV must contain at least a header and two data rows');
  }

  const chosenTarget = targetCol && headers.includes(targetCol) ? targetCol : headers[headers.length - 1];
  const columns = buildColumnSummaries(rows, chosenTarget);

  return {
    summary: {
      id: `custom_${Date.now()}`,
      name: 'Uploaded Custom Dataset',
      rowCount: rows.length,
      columnCount: headers.length,
      columns,
      targetColumn: chosenTarget,
      taskType: 'classification',
      features: headers.filter(h => h !== chosenTarget && h !== 'id'),
      headRows: rows.slice(0, 15),
      missingValuesTotal: columns.reduce((acc, col) => acc + col.nullCount, 0)
    },
    rawCsv: csvText
  };
}

export async function trainModelApi(params: {
  rows: Record<string, string | number>[];
  targetCol: string;
  modelType: ModelType;
  hyperparameters?: Record<string, any>;
  testRatio?: number;
  seed?: number;
}): Promise<TrainedModelResult> {
  const settings = getEngineSettings();

  // Try FastAPI if configured
  if (settings.mode === 'fastapi') {
    try {
      const resp = await fetch(`${settings.fastApiUrl.replace(/\/$/, '')}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      if (resp.ok) {
        const data = await resp.json();
        return data.model;
      }
    } catch (e) {
      console.warn('FastAPI unavailable, switching to integrated engine', e);
    }
  }

  // Integrated / Express Server
  try {
    const resp = await fetch('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    if (resp.ok) {
      const data = await resp.json();
      return data.model;
    }
  } catch (e) {
    // fallback
  }

  // Client-side execution fallback
  const prepData = preprocessTabularData(params.rows, params.targetCol, params.testRatio || 0.25, params.seed || 42);
  const startTime = performance.now();

  let model: ModelEstimator;
  const hp = params.hyperparameters || {};

  if (params.modelType === 'logistic_regression') {
    model = new LogisticRegressionModel(hp.learningRate ?? 0.05, hp.maxIter ?? 120, hp.l2Reg ?? 0.01);
  } else if (params.modelType === 'gradient_boosting') {
    model = new GradientBoostingClassifierModel(hp.nEstimators ?? 30, hp.learningRate ?? 0.1, hp.maxDepth ?? 4);
  } else {
    model = new RandomForestClassifierModel(hp.nEstimators ?? 35, hp.maxDepth ?? 6, hp.minSamplesSplit ?? 4, hp.maxFeaturesRatio ?? 0.75);
  }

  model.fit(prepData.X_train, prepData.y_train);
  const trainTime = Math.round(performance.now() - startTime);

  const latStart = performance.now();
  const probs = model.predictProba(prepData.X_test);
  const latencyMs = Math.round((performance.now() - latStart) * 10) / 10;

  const metrics = calculateClassificationMetrics(prepData.y_test, probs);
  const featImportances = model.getFeatureImportances(prepData.featureNames);

  const names: Record<ModelType, string> = {
    logistic_regression: 'Logistic Regression',
    random_forest: 'Random Forest Classifier',
    gradient_boosting: 'Gradient Boosting Classifier'
  };

  return {
    id: `${params.modelType}_${Date.now()}`,
    modelType: params.modelType,
    modelName: names[params.modelType],
    taskType: 'classification',
    trainedAt: new Date().toISOString(),
    hyperparameters: hp,
    metrics: {
      ...metrics,
      latencyMs,
      trainTimeMs: trainTime,
      modelComplexity: model.getComplexity()
    },
    featureImportances: featImportances,
    isOptimized: false
  };
}

export async function optimizeModelApi(params: {
  rows: Record<string, string | number>[];
  targetCol: string;
  modelType: ModelType;
  nTrials?: number;
  objectiveMetric?: 'f1Score' | 'accuracy' | 'rocAuc';
}): Promise<OptimizationResult> {
  const settings = getEngineSettings();

  if (settings.mode === 'fastapi') {
    try {
      const resp = await fetch(`${settings.fastApiUrl.replace(/\/$/, '')}/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      if (resp.ok) {
        return await resp.json();
      }
    } catch (e) {
      console.warn('FastAPI unavailable, switching to integrated engine', e);
    }
  }

  try {
    const resp = await fetch('/api/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    if (resp.ok) {
      return await resp.json();
    }
  } catch (e) {
    // fallback
  }

  const prepData = preprocessTabularData(params.rows, params.targetCol);
  return runHyperparameterOptimization(params.modelType, prepData, params.nTrials || 12, params.objectiveMetric || 'f1Score');
}

export async function analyzeDriftApi(params: {
  rows: Record<string, string | number>[];
  targetCol: string;
  modelType: ModelType;
  hyperparameters?: Record<string, any>;
  shiftConfig: ShiftConfiguration;
}): Promise<DriftAnalysisResult> {
  const settings = getEngineSettings();

  if (settings.mode === 'fastapi') {
    try {
      const resp = await fetch(`${settings.fastApiUrl.replace(/\/$/, '')}/drift`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      if (resp.ok) {
        return await resp.json();
      }
    } catch (e) {
      console.warn('FastAPI unavailable, switching to integrated engine', e);
    }
  }

  try {
    const resp = await fetch('/api/drift', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    if (resp.ok) {
      return await resp.json();
    }
  } catch (e) {
    // fallback
  }

  const prepData = preprocessTabularData(params.rows, params.targetCol);
  let model: ModelEstimator;
  if (params.modelType === 'logistic_regression') {
    model = new LogisticRegressionModel();
  } else if (params.modelType === 'gradient_boosting') {
    model = new GradientBoostingClassifierModel();
  } else {
    model = new RandomForestClassifierModel();
  }
  model.fit(prepData.X_train, prepData.y_train);

  return simulateAndAnalyzeDrift(
    prepData,
    model,
    params.modelType === 'random_forest' ? 'Random Forest' : params.modelType === 'gradient_boosting' ? 'Gradient Boosting' : 'Logistic Regression',
    params.shiftConfig
  );
}

export async function explainPredictionsApi(params: {
  rows: Record<string, string | number>[];
  targetCol: string;
  modelType: ModelType;
  hyperparameters?: Record<string, any>;
  sampleIndex?: number;
}): Promise<ExplainabilityResult> {
  try {
    const resp = await fetch('/api/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    if (resp.ok) {
      return await resp.json();
    }
  } catch (e) {
    // fallback
  }

  const prepData = preprocessTabularData(params.rows, params.targetCol);
  let model: ModelEstimator;
  if (params.modelType === 'logistic_regression') {
    model = new LogisticRegressionModel();
  } else if (params.modelType === 'gradient_boosting') {
    model = new GradientBoostingClassifierModel();
  } else {
    model = new RandomForestClassifierModel();
  }
  model.fit(prepData.X_train, prepData.y_train);

  return computeExplainability(model, params.modelType, prepData, params.sampleIndex || 0);
}

export async function predictTransactionsApi(params: {
  trainingRows: Record<string, string | number>[];
  testRows: Record<string, string | number>[];
  targetCol: string;
  modelType?: ModelType;
  hyperparameters?: Record<string, any>;
}): Promise<BatchPredictionResponse> {
  const settings = getEngineSettings();

  // Try external FastAPI if active
  if (settings.mode === 'fastapi') {
    try {
      const resp = await fetch(`${settings.fastApiUrl.replace(/\/$/, '')}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      if (resp.ok) {
        return await resp.json();
      } else {
        const errJson = await resp.json().catch(() => ({}));
        throw new Error(errJson.detail || errJson.error || `FastAPI error (HTTP ${resp.status})`);
      }
    } catch (e: any) {
      if (e.message && !e.message.includes('fetch')) {
        throw e;
      }
      console.warn('FastAPI predict unreachable, falling back to local engine', e);
    }
  }

  // Try Express server proxy
  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    if (resp.ok) {
      return await resp.json();
    } else {
      const errJson = await resp.json().catch(() => ({}));
      throw new Error(errJson.error || errJson.detail || `Server error (HTTP ${resp.status})`);
    }
  } catch (e: any) {
    if (e.message && !e.message.includes('fetch') && !e.message.includes('Failed to fetch')) {
      throw e;
    }
    // Network failure: fallback to in-browser execution
  }

  // In-browser client engine fallback
  return predictTransactionsEngine(
    params.trainingRows,
    params.testRows,
    params.targetCol,
    params.modelType || 'random_forest',
    params.hyperparameters
  );
}
