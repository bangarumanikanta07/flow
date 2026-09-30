import { ConfusionMatrix, RocCurvePoint, DriftMetric, DistributionBin } from './types';

export interface PreprocessedData {
  X_train: number[][];
  y_train: number[];
  X_test: number[][];
  y_test: number[];
  featureNames: string[];
  means: number[];
  stds: number[];
  rawTestRows: Record<string, string | number>[];
}

/**
 * Preprocesses tabular data: encodes categoricals, standardizes numerical features,
 * and splits into train and test partitions.
 */
export function preprocessTabularData(
  rows: Record<string, string | number>[],
  targetCol: string,
  testRatio = 0.25,
  seed = 42
): PreprocessedData {
  if (rows.length === 0) {
    throw new Error('Empty dataset provided for preprocessing');
  }

  // 1. Identify feature columns (exclude target)
  const allHeaders = Object.keys(rows[0]);
  const featureCols = allHeaders.filter(h => h !== targetCol && h !== 'id' && h !== 'customer_id');

  // 2. Discover categories for categorical columns
  const categoricalMaps: Record<string, Map<string, number>> = {};
  const numericalCols: string[] = [];

  for (const col of featureCols) {
    const isNum = rows.every(r => {
      const val = r[col];
      return val === null || val === undefined || val === '' || !isNaN(Number(val));
    });

    if (isNum) {
      numericalCols.push(col);
    } else {
      const uniqueVals = Array.from(new Set(rows.map(r => String(r[col]))));
      const valMap = new Map<string, number>();
      uniqueVals.forEach((val, idx) => valMap.set(val, idx));
      categoricalMaps[col] = valMap;
    }
  }

  // Feature vector names
  const featureNames: string[] = [];
  numericalCols.forEach(col => featureNames.push(col));
  Object.keys(categoricalMaps).forEach(col => {
    const catMap = categoricalMaps[col];
    catMap.forEach((_, catVal) => {
      featureNames.push(`${col}_${catVal}`);
    });
  });

  // 3. Convert rows to vectors
  const X_raw: number[][] = [];
  const y_raw: number[] = [];

  for (const row of rows) {
    const vec: number[] = [];
    // Numerical
    for (const col of numericalCols) {
      const v = Number(row[col]);
      vec.push(isNaN(v) ? 0 : v);
    }
    // Categorical one-hot
    for (const col of Object.keys(categoricalMaps)) {
      const catMap = categoricalMaps[col];
      const actualVal = String(row[col]);
      catMap.forEach((_, catVal) => {
        vec.push(catVal === actualVal ? 1 : 0);
      });
    }

    X_raw.push(vec);

    // Target parsing
    const rawTarget = row[targetCol];
    let targetNum = Number(rawTarget);
    if (isNaN(targetNum)) {
      targetNum = (String(rawTarget).toLowerCase() === 'yes' || String(rawTarget).toLowerCase() === 'true') ? 1 : 0;
    }
    y_raw.push(targetNum >= 0.5 ? 1 : 0);
  }

  // 4. Stratified / Seeded Shuffle & Train/Test Split
  const indices = rows.map((_, i) => i);
  // deterministic Fisher-Yates
  let currentSeed = seed;
  for (let i = indices.length - 1; i > 0; i--) {
    currentSeed = (currentSeed * 9301 + 49297) % 233280;
    const j = Math.floor((currentSeed / 233280) * (i + 1));
    const temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }

  const testCount = Math.max(1, Math.round(rows.length * testRatio));
  const testIndices = indices.slice(0, testCount);
  const trainIndices = indices.slice(testCount);

  // Compute means and standard deviations from training set only
  const nFeatures = featureNames.length;
  const means: number[] = new Array(nFeatures).fill(0);
  const stds: number[] = new Array(nFeatures).fill(1);

  for (const idx of trainIndices) {
    for (let f = 0; f < nFeatures; f++) {
      means[f] += X_raw[idx][f];
    }
  }
  for (let f = 0; f < nFeatures; f++) {
    means[f] /= trainIndices.length || 1;
  }

  for (const idx of trainIndices) {
    for (let f = 0; f < nFeatures; f++) {
      stds[f] += Math.pow(X_raw[idx][f] - means[f], 2);
    }
  }
  for (let f = 0; f < nFeatures; f++) {
    stds[f] = Math.sqrt(stds[f] / (trainIndices.length || 1));
    if (stds[f] < 1e-6) stds[f] = 1; // avoid div by zero
  }

  // Standardize X
  const scaleVector = (vec: number[]) => {
    return vec.map((val, f) => (val - means[f]) / stds[f]);
  };

  const X_train = trainIndices.map(idx => scaleVector(X_raw[idx]));
  const y_train = trainIndices.map(idx => y_raw[idx]);
  const X_test = testIndices.map(idx => scaleVector(X_raw[idx]));
  const y_test = testIndices.map(idx => y_raw[idx]);
  const rawTestRows = testIndices.map(idx => rows[idx]);

  return {
    X_train,
    y_train,
    X_test,
    y_test,
    featureNames,
    means,
    stds,
    rawTestRows
  };
}

/**
 * Calculates classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
 */
export function calculateClassificationMetrics(
  y_true: number[],
  y_prob: number[],
  threshold = 0.5
): {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  rocAuc: number;
  confusionMatrix: ConfusionMatrix;
  rocCurve: RocCurvePoint[];
} {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;

  for (let i = 0; i < y_true.length; i++) {
    const trueVal = y_true[i] >= 0.5 ? 1 : 0;
    const predVal = y_prob[i] >= threshold ? 1 : 0;

    if (trueVal === 1 && predVal === 1) tp++;
    else if (trueVal === 0 && predVal === 1) fp++;
    else if (trueVal === 0 && predVal === 0) tn++;
    else if (trueVal === 1 && predVal === 0) fn++;
  }

  const total = y_true.length || 1;
  const accuracy = (tp + tn) / total;
  const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const f1Score = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  // ROC Curve and AUC calculation
  const rocPoints: { prob: number; actual: number }[] = y_prob.map((prob, i) => ({
    prob,
    actual: y_true[i] >= 0.5 ? 1 : 0
  }));
  rocPoints.sort((a, b) => b.prob - a.prob);

  const totalPositives = y_true.filter(y => y >= 0.5).length;
  const totalNegatives = y_true.length - totalPositives;

  const rocCurve: RocCurvePoint[] = [{ fpr: 0, tpr: 0, threshold: 1 }];
  let cumTp = 0;
  let cumFp = 0;
  let rocAuc = 0.5;

  if (totalPositives > 0 && totalNegatives > 0) {
    for (let i = 0; i < rocPoints.length; i++) {
      if (rocPoints[i].actual === 1) {
        cumTp++;
      } else {
        cumFp++;
      }

      const fpr = cumFp / totalNegatives;
      const tpr = cumTp / totalPositives;
      rocCurve.push({
        fpr: Math.round(fpr * 1000) / 1000,
        tpr: Math.round(tpr * 1000) / 1000,
        threshold: Math.round(rocPoints[i].prob * 1000) / 1000
      });
    }

    // Trapezoidal integration for AUC
    rocAuc = 0;
    for (let i = 1; i < rocCurve.length; i++) {
      const dFpr = rocCurve[i].fpr - rocCurve[i - 1].fpr;
      const avgTpr = (rocCurve[i].tpr + rocCurve[i - 1].tpr) / 2;
      rocAuc += dFpr * avgTpr;
    }
    rocAuc = Math.max(0.5, Math.min(1.0, rocAuc));
  } else {
    rocCurve.push({ fpr: 1, tpr: 1, threshold: 0 });
    rocAuc = 0.5;
  }

  // Downsample ROC curve to ~20 points for smooth charting
  const downsampledRoc: RocCurvePoint[] = [];
  const step = Math.max(1, Math.floor(rocCurve.length / 20));
  for (let i = 0; i < rocCurve.length; i += step) {
    downsampledRoc.push(rocCurve[i]);
  }
  if (downsampledRoc[downsampledRoc.length - 1].fpr !== 1) {
    downsampledRoc.push({ fpr: 1, tpr: 1, threshold: 0 });
  }

  return {
    accuracy: Math.round(accuracy * 1000) / 1000,
    precision: Math.round(precision * 1000) / 1000,
    recall: Math.round(recall * 1000) / 1000,
    f1Score: Math.round(f1Score * 1000) / 1000,
    rocAuc: Math.round(rocAuc * 1000) / 1000,
    confusionMatrix: {
      truePositive: tp,
      falsePositive: fp,
      trueNegative: tn,
      falseNegative: fn,
      classes: ['Class 0 (Negative)', 'Class 1 (Positive)']
    },
    rocCurve: downsampledRoc
  };
}

/**
 * Calculates Kolmogorov-Smirnov 2-sample statistic and asymptotic p-value.
 */
export function calculateKs2Sample(sample1: number[], sample2: number[]): { ksStat: number; pValue: number } {
  if (sample1.length === 0 || sample2.length === 0) {
    return { ksStat: 0, pValue: 1 };
  }

  const s1 = [...sample1].sort((a, b) => a - b);
  const s2 = [...sample2].sort((a, b) => a - b);

  const n1 = s1.length;
  const n2 = s2.length;

  let i = 0;
  let j = 0;
  let maxD = 0;

  while (i < n1 && j < n2) {
    const v1 = s1[i];
    const v2 = s2[j];

    if (v1 <= v2) {
      i++;
    }
    if (v2 <= v1) {
      j++;
    }

    const cdf1 = i / n1;
    const cdf2 = j / n2;
    const diff = Math.abs(cdf1 - cdf2);
    if (diff > maxD) {
      maxD = diff;
    }
  }

  // Asymptotic p-value approximation via Kolmogorov distribution
  const en = Math.sqrt((n1 * n2) / (n1 + n2));
  const lambda = (en + 0.12 + 0.11 / en) * maxD;

  let pVal = 0;
  if (lambda > 0) {
    for (let k = 1; k <= 20; k++) {
      const term = 2 * Math.pow(-1, k - 1) * Math.exp(-2 * k * k * lambda * lambda);
      pVal += term;
    }
    pVal = Math.max(0, Math.min(1, pVal));
  } else {
    pVal = 1;
  }

  return {
    ksStat: Math.round(maxD * 1000) / 1000,
    pValue: Math.round(pVal * 1000) / 1000
  };
}

/**
 * Computes 1D Wasserstein distance (Earth Mover's Distance)
 */
export function calculateWassersteinDistance(s1: number[], s2: number[]): number {
  if (s1.length === 0 || s2.length === 0) return 0;
  const sorted1 = [...s1].sort((a, b) => a - b);
  const sorted2 = [...s2].sort((a, b) => a - b);

  // Evaluate quantile differences across 100 evenly spaced points
  const points = 100;
  let distanceSum = 0;

  for (let q = 1; q <= points; q++) {
    const fraction = q / points;
    const idx1 = Math.min(sorted1.length - 1, Math.floor(fraction * sorted1.length));
    const idx2 = Math.min(sorted2.length - 1, Math.floor(fraction * sorted2.length));
    distanceSum += Math.abs(sorted1[idx1] - sorted2[idx2]);
  }

  return Math.round((distanceSum / points) * 1000) / 1000;
}

/**
 * Computes Population Stability Index (PSI) between baseline and drifted distributions.
 * Partitions baseline into 10 quantiles.
 */
export function calculatePsi(
  baseline: number[],
  drifted: number[]
): { psi: number; bins: DistributionBin[] } {
  if (baseline.length === 0 || drifted.length === 0) {
    return { psi: 0, bins: [] };
  }

  const sortedBaseline = [...baseline].sort((a, b) => a - b);
  const numBins = 10;
  const minVal = sortedBaseline[0];
  const maxVal = sortedBaseline[sortedBaseline.length - 1];
  const range = maxVal - minVal || 1;
  const binWidth = range / numBins;

  const bins: DistributionBin[] = [];
  const baselineCounts = new Array(numBins).fill(0);
  const driftedCounts = new Array(numBins).fill(0);

  // Fill counts
  for (const v of baseline) {
    const binIdx = Math.min(numBins - 1, Math.max(0, Math.floor((v - minVal) / binWidth)));
    baselineCounts[binIdx]++;
  }
  for (const v of drifted) {
    const binIdx = Math.min(numBins - 1, Math.max(0, Math.floor((v - minVal) / binWidth)));
    driftedCounts[binIdx]++;
  }

  const eps = 1e-4; // Laplace smoothing
  let totalPsi = 0;

  for (let i = 0; i < numBins; i++) {
    const binStart = minVal + i * binWidth;
    const binEnd = minVal + (i + 1) * binWidth;
    const baseFreq = (baselineCounts[i] + eps) / (baseline.length + eps * numBins);
    const driftFreq = (driftedCounts[i] + eps) / (drifted.length + eps * numBins);

    const psiComponent = (driftFreq - baseFreq) * Math.log(driftFreq / baseFreq);
    totalPsi += psiComponent;

    bins.push({
      binLabel: `${Math.round(binStart * 10) / 10} - ${Math.round(binEnd * 10) / 10}`,
      binStart: Math.round(binStart * 100) / 100,
      binEnd: Math.round(binEnd * 100) / 100,
      baselineFrequency: Math.round(baseFreq * 1000) / 1000,
      driftedFrequency: Math.round(driftFreq * 1000) / 1000
    });
  }

  return {
    psi: Math.round(totalPsi * 1000) / 1000,
    bins
  };
}
