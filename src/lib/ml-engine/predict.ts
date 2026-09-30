import { ModelType, BatchPredictionResponse, PredictionResultItem } from './types';
import { RandomForestClassifierModel, LogisticRegressionModel, GradientBoostingClassifierModel, ModelEstimator } from './algorithms';

/**
 * Predicts whether unseen transactions are Fraud or Legitimate using
 * the trained model and features, strictly excluding the target column (is_fraud)
 * from input features.
 */
export function predictTransactionsEngine(
  trainingRows: Record<string, string | number>[],
  testRows: Record<string, string | number>[],
  targetColumn: string = 'is_fraud',
  modelType: ModelType = 'random_forest',
  hyperparameters?: Record<string, any>
): BatchPredictionResponse {
  if (trainingRows.length < 5) {
    throw new Error('At least 5 training rows are required to fit the predictive model');
  }
  if (testRows.length === 0) {
    throw new Error('At least one transaction record is required for prediction');
  }

  // 1. Identify feature columns excluding ID columns and the target column
  const allHeaders = Object.keys(trainingRows[0]);
  const featureCols = allHeaders.filter(
    h => h !== targetColumn && h !== 'id' && h !== 'customer_id' && h !== 'transaction_id' && h !== 'Unnamed: 0'
  );

  // 2. Discover categorical values and numerical columns
  const categoricalMaps: Record<string, Map<string, number>> = {};
  const numericalCols: string[] = [];

  for (const col of featureCols) {
    const isNum = trainingRows.every(r => {
      const val = r[col];
      return val === null || val === undefined || val === '' || !isNaN(Number(val));
    });

    if (isNum) {
      numericalCols.push(col);
    } else {
      const uniqueVals = Array.from(new Set(trainingRows.map(r => String(r[col]))));
      const valMap = new Map<string, number>();
      uniqueVals.forEach((val, idx) => valMap.set(val, idx));
      categoricalMaps[col] = valMap;
    }
  }

  // Feature vector labels
  const featureNames: string[] = [];
  numericalCols.forEach(col => featureNames.push(col));
  Object.keys(categoricalMaps).forEach(col => {
    categoricalMaps[col].forEach((_, catVal) => {
      featureNames.push(`${col}_${catVal}`);
    });
  });

  const nFeatures = featureNames.length;

  // 3. Vectorize training rows
  const X_train_raw: number[][] = [];
  const y_train_raw: number[] = [];

  for (const row of trainingRows) {
    const vec: number[] = [];
    for (const col of numericalCols) {
      const v = Number(row[col]);
      vec.push(isNaN(v) ? 0 : v);
    }
    for (const col of Object.keys(categoricalMaps)) {
      const catMap = categoricalMaps[col];
      const actualVal = String(row[col]);
      catMap.forEach((_, catVal) => {
        vec.push(catVal === actualVal ? 1 : 0);
      });
    }
    X_train_raw.push(vec);

    const rawT = row[targetColumn];
    let tNum = Number(rawT);
    if (isNaN(tNum)) {
      const lower = String(rawT).toLowerCase();
      tNum = lower === 'yes' || lower === 'true' || lower === 'fraud' || lower === '1' ? 1 : 0;
    }
    y_train_raw.push(tNum >= 0.5 ? 1 : 0);
  }

  // 4. Compute scaling parameters (mean, std) from training data only
  const means: number[] = new Array(nFeatures).fill(0);
  const stds: number[] = new Array(nFeatures).fill(1);

  for (let i = 0; i < trainingRows.length; i++) {
    for (let f = 0; f < nFeatures; f++) {
      means[f] += X_train_raw[i][f];
    }
  }
  for (let f = 0; f < nFeatures; f++) {
    means[f] /= trainingRows.length || 1;
  }

  for (let i = 0; i < trainingRows.length; i++) {
    for (let f = 0; f < nFeatures; f++) {
      stds[f] += Math.pow(X_train_raw[i][f] - means[f], 2);
    }
  }
  for (let f = 0; f < nFeatures; f++) {
    stds[f] = Math.sqrt(stds[f] / (trainingRows.length || 1));
    if (stds[f] < 1e-6) stds[f] = 1;
  }

  const scaleVector = (vec: number[]) => {
    return vec.map((val, f) => (val - means[f]) / stds[f]);
  };

  const X_train = X_train_raw.map(v => scaleVector(v));

  // 5. Instantiate and fit model
  let model: ModelEstimator;
  const hp = hyperparameters || {};
  if (modelType === 'logistic_regression') {
    model = new LogisticRegressionModel(hp.learningRate ?? 0.05, hp.maxIter ?? 150, hp.l2Reg ?? 0.01);
  } else if (modelType === 'gradient_boosting') {
    model = new GradientBoostingClassifierModel(hp.nEstimators ?? 35, hp.learningRate ?? 0.1, hp.maxDepth ?? 4);
  } else {
    model = new RandomForestClassifierModel(
      hp.nEstimators ?? 40,
      hp.maxDepth ?? 6,
      hp.minSamplesSplit ?? 4,
      hp.maxFeaturesRatio ?? 0.75
    );
  }

  model.fit(X_train, y_train_raw);

  // 6. Strict validation of test rows schema
  if (testRows.length > 0) {
    const testCols = Object.keys(testRows[0]);
    const missing = featureCols.filter(col => !testCols.includes(col));
    if (missing.length > 0) {
      throw new Error(
        `Uploaded prediction CSV is missing required feature column(s): ${missing.join(', ')}. Expected columns: ${featureCols.join(', ')}`
      );
    }

    // Check numerical type compatibility
    for (const numCol of numericalCols) {
      const allInvalid = testRows.every(r => {
        const val = r[numCol];
        return val === null || val === undefined || val === '' || isNaN(Number(val));
      });
      if (allInvalid && testRows.length > 0) {
        throw new Error(`Column '${numCol}' expects numeric values, but non-numeric data was provided in the prediction CSV.`);
      }
    }
  }

  // Vectorize unseen test rows (STRICTLY ignoring targetColumn even if present)
  const X_test_raw: number[][] = [];
  const hasGroundTruth = testRows.length > 0 && targetColumn in testRows[0];

  for (const row of testRows) {
    const vec: number[] = [];
    for (const col of numericalCols) {
      const v = Number(row[col]);
      vec.push(isNaN(v) ? 0 : v);
    }
    for (const col of Object.keys(categoricalMaps)) {
      const catMap = categoricalMaps[col];
      const actualVal = String(row[col]);
      catMap.forEach((_, catVal) => {
        vec.push(catVal === actualVal ? 1 : 0);
      });
    }
    X_test_raw.push(vec);
  }

  const X_test = X_test_raw.map(v => scaleVector(v));

  // 7. Execute real inference with high-resolution timer
  const startTime = performance.now();
  const probs = model.predictProba(X_test);
  const latencyMs = Math.round((performance.now() - startTime) * 10) / 10;

  const results: PredictionResultItem[] = [];
  let fraudCount = 0;
  let legitCount = 0;

  for (let idx = 0; idx < testRows.length; idx++) {
    const prob = Math.round(probs[idx] * 10000) / 10000;
    const isFraud = prob >= 0.5;

    if (isFraud) fraudCount++;
    else legitCount++;

    const inputFeats: Record<string, string | number> = {};
    for (const k of Object.keys(testRows[idx])) {
      if (k !== targetColumn) {
        inputFeats[k] = testRows[idx][k];
      }
    }

    const item: PredictionResultItem = {
      rowIndex: idx + 1,
      inputFeatures: inputFeats,
      predictedClass: isFraud ? 'Fraud' : 'Legitimate',
      predictedLabel: isFraud ? 1 : 0,
      fraudProbability: prob,
      confidenceScore: Math.round((isFraud ? prob : 1.0 - prob) * 1000) / 10,
      isVerifiedOutcome: false,
      verifiedOutcome: 'Unverified (Unseen Transaction)'
    };

    if (hasGroundTruth) {
      const verifiedRaw = testRows[idx][targetColumn];
      const verifiedNum =
        String(verifiedRaw).toLowerCase() === '1' ||
        String(verifiedRaw).toLowerCase() === 'true' ||
        String(verifiedRaw).toLowerCase() === 'fraud' ||
        Number(verifiedRaw) >= 0.5
          ? 1
          : 0;
      item.verifiedOutcome = verifiedNum === 1 ? 'Fraud' : 'Legitimate';
      item.isCorrect = item.predictedClass === item.verifiedOutcome;
    }

    results.push(item);
  }

  const names = {
    random_forest: 'Random Forest Classifier',
    logistic_regression: 'Logistic Regression',
    gradient_boosting: 'Gradient Boosting Classifier'
  };

  return {
    modelUsed: names[modelType] || 'Random Forest Classifier',
    totalSamples: results.length,
    fraudCount,
    legitimateCount: legitCount,
    fraudRatePct: Math.round((fraudCount / (results.length || 1)) * 1000) / 10,
    inferenceLatencyMs: latencyMs,
    predictions: results
  };
}
