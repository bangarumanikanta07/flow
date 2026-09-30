import express, { Request, Response } from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { existsSync } from 'fs';
import { fileURLToPath } from 'url';
import Papa from 'papaparse';
import {
  generateCustomerChurnDataset,
  generateCreditRiskDataset,
  generateHousingPricingDataset,
  generateCreditCardFraudDataset,
  buildColumnSummaries
} from './src/lib/ml-engine/sample-datasets';
import { preprocessTabularData, calculateClassificationMetrics } from './src/lib/ml-engine/stats';
import {
  LogisticRegressionModel,
  RandomForestClassifierModel,
  GradientBoostingClassifierModel,
  ModelEstimator
} from './src/lib/ml-engine/algorithms';
import { runHyperparameterOptimization } from './src/lib/ml-engine/optimizer';
import { simulateAndAnalyzeDrift } from './src/lib/ml-engine/drift';
import { computeExplainability } from './src/lib/ml-engine/explain';
import { predictTransactionsEngine } from './src/lib/ml-engine/predict';
import { ModelType } from './src/lib/ml-engine/types';
import { spawn, ChildProcess } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const FASTAPI_INTERNAL_URL = 'http://127.0.0.1:8001';
let pythonProcess: ChildProcess | null = null;

function ensureFastApiProcess() {
  if (pythonProcess && !pythonProcess.killed) {
    return;
  }

  try {
    const backendDir = path.resolve(__dirname, 'backend');
    // Prefer the backend virtualenv so the interpreter actually has fastapi/sklearn/optuna installed
    const venvPython = process.platform === 'win32'
      ? path.join(backendDir, '.venv', 'Scripts', 'python.exe')
      : path.join(backendDir, '.venv', 'bin', 'python');
    const pythonCmd = process.env.PYTHON
      || (existsSync(venvPython) ? venvPython : process.platform === 'win32' ? 'python' : 'python3');
    pythonProcess = spawn(pythonCmd,['-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8001'], {
      cwd: backendDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: false
    });

    pythonProcess.stdout?.on('data', (data) => {
      const msg = data.toString();
      if (msg.includes('Uvicorn running') || msg.includes('Application startup complete')) {
        console.log(`[FastAPI Backend]: ${msg.trim()}`);
      }
    });

    pythonProcess.stderr?.on('data', (data) => {
      const err = data.toString();
      if (err.includes('ERROR') || err.includes('Traceback')) {
        console.warn(`[FastAPI Backend Error]: ${err.trim()}`);
      }
    });

    pythonProcess.on('exit', (code, signal) => {
      console.log(`[FastAPI Backend] exited with code ${code}, signal ${signal}. Restarting if needed...`);
      pythonProcess = null;
    });

    process.on('exit', () => {
      if (pythonProcess) {
        pythonProcess.kill();
      }
    });
  } catch (err) {
    console.warn('Could not spawn background Python FastAPI process:', err);
  }
}

async function waitForFastApiReady(maxRetries = 15, delayMs = 400): Promise<boolean> {
  ensureFastApiProcess();
  for (let i = 0; i < maxRetries; i++) {
    try {
      const resp = await fetch(`${FASTAPI_INTERNAL_URL}/api/health`, { signal: AbortSignal.timeout(1000) });
      if (resp.ok) {
        return true;
      }
    } catch {
      // wait and retry
    }
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }
  return false;
}

async function startServer() {
  ensureFastApiProcess();
  const app = express();
  const PORT = process.env.PORT || 3000;
  const isProd = process.env.NODE_ENV === 'production';

  app.use(express.json({ limit: '25mb' }));
  app.use(express.urlencoded({ extended: true, limit: '25mb' }));

  // CORS
  app.use((_req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
    res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    next();
  });

  // Health check endpoint
  app.get('/api/health', async (_req: Request, res: Response) => {
    let fastApiStatus = false;
    let fastApiEngine = '';
    try {
      const resp = await fetch(`${FASTAPI_INTERNAL_URL}/api/health`, { signal: AbortSignal.timeout(1500) });
      if (resp.ok) {
        const data = (await resp.json()) as any;
        fastApiStatus = true;
        fastApiEngine = data.engine || 'Python FastAPI Core';
      }
    } catch {
      // offline
    }

    res.json({
      status: 'healthy',
      engine: fastApiStatus ? `Integrated + ${fastApiEngine}` : 'DriftForge Dual ML Engine',
      version: '1.4.0',
      fastApiBridgeReady: fastApiStatus,
      timestamp: new Date().toISOString()
    });
  });

  // Sample datasets provider
  app.get('/api/datasets/sample/:type', (req: Request, res: Response) => {
    const { type } = req.params;
    let result;
    if (type === 'churn') {
      result = generateCustomerChurnDataset();
    } else if (type === 'credit') {
      result = generateCreditRiskDataset();
    } else if (type === 'fraud') {
      result = generateCreditCardFraudDataset();
    } else {
      result = generateHousingPricingDataset();
    }
    res.json(result);
  });

  // CSV Parse and Dataset Summary with PapaParse
  app.post('/api/datasets/parse', (req: Request, res: Response) => {
    try {
      const { csvText, targetCol } = req.body;
      if (!csvText || typeof csvText !== 'string') {
        return res.status(400).json({ error: 'Missing csvText in request body' });
      }

      const parsed = Papa.parse<Record<string, any>>(csvText, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: 'greedy'
      });

      if (parsed.errors && parsed.errors.length > 0 && parsed.data.length === 0) {
        return res.status(400).json({ error: parsed.errors[0].message });
      }

      const rows: Record<string, string | number>[] = [];
      const headers = (parsed.meta.fields || []).map(h => h.trim());

      for (const rawRow of parsed.data) {
        const cleanRow: Record<string, string | number> = {};
        for (const h of headers) {
          const val = rawRow[h];
          if (val === null || val === undefined || val === '') {
            cleanRow[h] = '';
          } else if (typeof val === 'number') {
            cleanRow[h] = val;
          } else {
            cleanRow[h] = String(val).trim();
          }
        }
        rows.push(cleanRow);
      }

      if (rows.length < 2) {
        return res.status(400).json({ error: 'CSV must contain at least a header and two data rows' });
      }

      const chosenTarget = targetCol && headers.includes(targetCol) ? targetCol : headers[headers.length - 1];
      const columns = buildColumnSummaries(rows, chosenTarget);

      return res.json({
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
      });
    } catch (err: any) {
      return res.status(500).json({ error: err.message || 'Failed to process CSV' });
    }
  });

  // Train endpoint - bridges directly to Python FastAPI with fallback
  app.post('/api/train', async (req: Request, res: Response) => {
    try {
      const { rows, targetCol, targetColumn, modelType, hyperparameters, testRatio = 0.25, testSize, seed = 42, randomState } = req.body;
      const target = targetCol || targetColumn;
      const finalTestRatio = testRatio || testSize || 0.25;
      const finalSeed = seed || randomState || 42;

      if (!rows || !Array.isArray(rows) || rows.length < 5) {
        return res.status(400).json({ error: 'Valid dataset rows required (minimum 5 samples)' });
      }

      // Forward to Python FastAPI with Scikit-learn
      try {
        ensureFastApiProcess();
        const pyResp = await fetch(`${FASTAPI_INTERNAL_URL}/train`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            rows,
            targetCol: target,
            modelType,
            hyperparameters,
            testRatio: finalTestRatio,
            seed: finalSeed
          }),
          signal: AbortSignal.timeout(15000)
        });

        if (pyResp.ok) {
          const pyData = await pyResp.json();
          return res.json(pyData);
        }
      } catch (err: any) {
        // Silently use the local engine without printing scary error traces if connection refused
        if (err.cause?.code !== 'ECONNREFUSED' && !err.message?.includes('ECONNREFUSED')) {
          console.warn('FastAPI backend forwarding error, using local engine:', err.message || err);
        }
      }

      // High-precision local fallback
      const prepData = preprocessTabularData(rows, target, finalTestRatio, finalSeed);
      const startTime = performance.now();

      let model: ModelEstimator;
      const params = hyperparameters || {};

      if (modelType === 'logistic_regression') {
        model = new LogisticRegressionModel(
          params.learningRate ?? 0.05,
          params.maxIter ?? 120,
          params.l2Reg ?? 0.01
        );
      } else if (modelType === 'gradient_boosting') {
        model = new GradientBoostingClassifierModel(
          params.nEstimators ?? 30,
          params.learningRate ?? 0.1,
          params.maxDepth ?? 4
        );
      } else {
        model = new RandomForestClassifierModel(
          params.nEstimators ?? 35,
          params.maxDepth ?? 6,
          params.minSamplesSplit ?? 4,
          params.maxFeaturesRatio ?? 0.75
        );
      }

      model.fit(prepData.X_train, prepData.y_train);
      const trainTime = Math.round(performance.now() - startTime);

      const latencyStart = performance.now();
      const probs = model.predictProba(prepData.X_test);
      const latencyMs = Math.round((performance.now() - latencyStart) * 10) / 10;

      const metrics = calculateClassificationMetrics(prepData.y_test, probs);
      const featImportances = model.getFeatureImportances(prepData.featureNames);

      const modelNames: Record<ModelType, string> = {
        logistic_regression: 'Logistic Regression',
        random_forest: 'Random Forest Classifier',
        gradient_boosting: 'Gradient Boosting Classifier'
      };

      res.json({
        model: {
          id: `${modelType}_${Date.now()}`,
          modelType,
          modelName: modelNames[modelType as ModelType] || 'Classifier',
          taskType: 'classification',
          trainedAt: new Date().toISOString(),
          hyperparameters: params,
          metrics: {
            ...metrics,
            latencyMs,
            trainTimeMs: trainTime,
            modelComplexity: model.getComplexity()
          },
          featureImportances: featImportances,
          isOptimized: false
        }
      });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Training failed' });
    }
  });

  // Hyperparameter Optimization endpoint - bridges to Optuna in Python
  app.post('/api/optimize', async (req: Request, res: Response) => {
    try {
      const { rows, targetCol, targetColumn, modelType, nTrials = 12, objectiveMetric = 'f1Score' } = req.body;
      const target = targetCol || targetColumn;

      // Forward to Optuna in FastAPI
      try {
        ensureFastApiProcess();
        const pyResp = await fetch(`${FASTAPI_INTERNAL_URL}/optimize`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            rows,
            targetCol: target,
            modelType,
            nTrials,
            objectiveMetric
          }),
          signal: AbortSignal.timeout(30000)
        });

        if (pyResp.ok) {
          const pyData = await pyResp.json();
          return res.json(pyData);
        }
      } catch (err: any) {
        if (err.cause?.code !== 'ECONNREFUSED' && !err.message?.includes('ECONNREFUSED')) {
          console.warn('Optuna FastAPI forwarding error, using local engine:', err.message || err);
        }
      }

      const prepData = preprocessTabularData(rows, target);
      const result = runHyperparameterOptimization(modelType as ModelType, prepData, nTrials, objectiveMetric);
      res.json(result);
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Optimization failed' });
    }
  });

  // Drift Analysis & Robustness endpoint - bridges to SciPy (KS-test, Wasserstein) in Python
  app.post('/api/drift', async (req: Request, res: Response) => {
    try {
      const { rows, targetCol, targetColumn, modelType, hyperparameters, shiftConfig } = req.body;
      const target = targetCol || targetColumn;

      try {
        ensureFastApiProcess();
        const pyResp = await fetch(`${FASTAPI_INTERNAL_URL}/drift`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            rows,
            targetCol: target,
            modelType,
            shiftConfig: shiftConfig || { meanShiftPct: 25, varianceScale: 1.3, noiseLevel: 0.15, affectedFeatures: [] }
          }),
          signal: AbortSignal.timeout(20000)
        });

        if (pyResp.ok) {
          const pyData = await pyResp.json();
          return res.json(pyData);
        }
      } catch (err: any) {
        if (err.cause?.code !== 'ECONNREFUSED' && !err.message?.includes('ECONNREFUSED')) {
          console.warn('FastAPI drift forwarding error, using local engine:', err.message || err);
        }
      }

      const prepData = preprocessTabularData(rows, target);

      let model: ModelEstimator;
      if (modelType === 'logistic_regression') {
        model = new LogisticRegressionModel();
      } else if (modelType === 'gradient_boosting') {
        model = new GradientBoostingClassifierModel();
      } else {
        model = new RandomForestClassifierModel();
      }
      model.fit(prepData.X_train, prepData.y_train);

      const result = simulateAndAnalyzeDrift(
        prepData,
        model,
        modelType === 'random_forest' ? 'Random Forest' : modelType === 'gradient_boosting' ? 'Gradient Boosting' : 'Logistic Regression',
        shiftConfig || { meanShiftPct: 25, varianceScale: 1.3, noiseLevel: 0.15, affectedFeatures: [] }
      );

      res.json(result);
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Drift analysis failed' });
    }
  });

  // Explainability endpoint - bridges to SHAP in Python
  app.post('/api/explain', async (req: Request, res: Response) => {
    try {
      const { rows, targetCol, targetColumn, modelType, sampleIndex = 0 } = req.body;
      const target = targetCol || targetColumn;

      try {
        ensureFastApiProcess();
        const pyResp = await fetch(`${FASTAPI_INTERNAL_URL}/explain`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            rows,
            targetCol: target,
            modelType,
            sampleIndex
          }),
          signal: AbortSignal.timeout(25000)
        });

        if (pyResp.ok) {
          const pyData = await pyResp.json();
          return res.json(pyData);
        }
      } catch (err: any) {
        if (err.cause?.code !== 'ECONNREFUSED' && !err.message?.includes('ECONNREFUSED')) {
          console.warn('FastAPI explainability forwarding error, using local engine:', err.message || err);
        }
      }

      const prepData = preprocessTabularData(rows, target);

      let model: ModelEstimator;
      if (modelType === 'logistic_regression') {
        model = new LogisticRegressionModel();
      } else if (modelType === 'gradient_boosting') {
        model = new GradientBoostingClassifierModel();
      } else {
        model = new RandomForestClassifierModel();
      }
      model.fit(prepData.X_train, prepData.y_train);

      const result = computeExplainability(model, modelType as ModelType, prepData, sampleIndex);
      res.json(result);
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Explainability computation failed' });
    }
  });

  // Test Prediction endpoint - executes real model inference on unseen transaction data
  app.post('/api/predict', async (req: Request, res: Response) => {
    try {
      const { trainingRows, testRows, targetCol, targetColumn, modelType = 'random_forest', hyperparameters } = req.body;
      const target = targetCol || targetColumn || 'is_fraud';

      if (!trainingRows || !Array.isArray(trainingRows) || trainingRows.length < 5) {
        return res.status(400).json({ error: 'Valid training dataset rows required (minimum 5 samples)' });
      }
      if (!testRows || !Array.isArray(testRows) || testRows.length === 0) {
        return res.status(400).json({ error: 'At least one transaction record is required for prediction' });
      }

      // Try Python FastAPI endpoint first
      try {
        ensureFastApiProcess();
        const pyResp = await fetch(`${FASTAPI_INTERNAL_URL}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            trainingRows,
            testRows,
            targetCol: target,
            modelType,
            hyperparameters
          }),
          signal: AbortSignal.timeout(20000)
        });

        if (pyResp.ok) {
          const pyData = await pyResp.json();
          return res.json(pyData);
        } else if (pyResp.status === 400 || pyResp.status === 422) {
          const errData = (await pyResp.json()) as any;
          return res.status(400).json({ error: errData.detail || errData.error || 'Invalid prediction request data' });
        }
      } catch (err: any) {
        if (err.cause?.code !== 'ECONNREFUSED' && !err.message?.includes('ECONNREFUSED')) {
          console.warn('FastAPI predict forwarding error, using local engine:', err.message || err);
        }
      }

      // Execute high-precision native engine prediction
      const result = predictTransactionsEngine(
        trainingRows,
        testRows,
        target,
        modelType as ModelType,
        hyperparameters
      );

      res.json(result);
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Prediction failed' });
    }
  });

  // Mount Vite middlewares in development or serve static files in production
  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa'
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(Number(PORT), '0.0.0.0', () => {
    console.log(`DriftForge AI platform running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch(err => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
