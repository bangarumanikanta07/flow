import { DriftAnalysisResult, DriftMetric, DistributionBin } from './types';
import { PreprocessedData, calculateKs2Sample, calculateWassersteinDistance, calculatePsi, calculateClassificationMetrics } from './stats';
import { ModelEstimator } from './algorithms';

export interface ShiftConfiguration {
  meanShiftPct: number; // e.g. +30% means +0.3 * std
  varianceScale: number; // e.g. 1.5 means 50% wider dispersion
  noiseLevel: number; // e.g. 0.2 means add uniform noise
  affectedFeatures: string[]; // which features to apply shift to (or empty for all numerical)
}

export function simulateAndAnalyzeDrift(
  data: PreprocessedData,
  model: ModelEstimator,
  modelName: string,
  config: ShiftConfiguration
): DriftAnalysisResult {
  const nTest = data.X_test.length;
  const nFeatures = data.featureNames.length;

  // 1. Generate shifted test dataset
  const shiftedX_test: number[][] = [];
  const shiftedValuesByFeature: number[][] = Array.from({ length: nFeatures }, () => []);
  const baselineValuesByFeature: number[][] = Array.from({ length: nFeatures }, () => []);

  for (let i = 0; i < nTest; i++) {
    const originalRow = data.X_test[i];
    const shiftedRow: number[] = [];

    for (let f = 0; f < nFeatures; f++) {
      const featName = data.featureNames[f];
      const isAffected = config.affectedFeatures.length === 0 || config.affectedFeatures.includes(featName);
      let val = originalRow[f];

      if (isAffected) {
        // Apply mean shift (in normalized scale)
        val += (config.meanShiftPct / 100);
        // Apply variance scaling
        val = val * config.varianceScale;
        // Apply noise
        if (config.noiseLevel > 0) {
          const pseudoNoise = ((Math.sin(i * 13 + f * 29) + Math.cos(i * 31)) / 2) * config.noiseLevel;
          val += pseudoNoise;
        }
      }

      shiftedRow.push(val);
      shiftedValuesByFeature[f].push(val);
      baselineValuesByFeature[f].push(originalRow[f]);
    }
    shiftedX_test.push(shiftedRow);
  }

  // 2. Compute Drift Metrics for each feature
  const featureDriftMetrics: DriftMetric[] = [];
  const distributionComparison: Record<string, DistributionBin[]> = {};
  let driftedCount = 0;
  let totalPsiSum = 0;

  for (let f = 0; f < nFeatures; f++) {
    const featName = data.featureNames[f];
    const baseVals = baselineValuesByFeature[f];
    const driftVals = shiftedValuesByFeature[f];

    const baseMean = baseVals.reduce((a, b) => a + b, 0) / (baseVals.length || 1);
    const driftMean = driftVals.reduce((a, b) => a + b, 0) / (driftVals.length || 1);
    const meanShiftPct = baseMean !== 0 ? ((driftMean - baseMean) / Math.abs(baseMean)) * 100 : driftMean * 100;

    const baseStd = Math.sqrt(baseVals.reduce((acc, v) => acc + Math.pow(v - baseMean, 2), 0) / (baseVals.length || 1));
    const driftStd = Math.sqrt(driftVals.reduce((acc, v) => acc + Math.pow(v - driftMean, 2), 0) / (driftVals.length || 1));

    const { ksStat, pValue } = calculateKs2Sample(baseVals, driftVals);
    const wasserstein = calculateWassersteinDistance(baseVals, driftVals);
    const { psi, bins } = calculatePsi(baseVals, driftVals);

    totalPsiSum += psi;

    const isDrifted = psi > 0.1 || (pValue < 0.05 && ksStat > 0.15);
    if (isDrifted) driftedCount++;

    let severity: 'low' | 'moderate' | 'high' = 'low';
    if (psi > 0.25 || ksStat > 0.35) {
      severity = 'high';
    } else if (psi > 0.1 || ksStat > 0.2) {
      severity = 'moderate';
    }

    featureDriftMetrics.push({
      feature: featName,
      baselineMean: Math.round(baseMean * 100) / 100,
      driftedMean: Math.round(driftMean * 100) / 100,
      meanShiftPct: Math.round(meanShiftPct * 10) / 10,
      baselineStd: Math.round(baseStd * 100) / 100,
      driftedStd: Math.round(driftStd * 100) / 100,
      wassersteinDistance: wasserstein,
      ksStatistic: ksStat,
      ksPValue: pValue,
      psi,
      driftDetected: isDrifted,
      severity
    });

    distributionComparison[featName] = bins;
  }

  // 3. Evaluate Model Robustness Degradation
  const baselineProbs = model.predictProba(data.X_test);
  const baselinePredictions = baselineProbs.map(p => (p >= 0.5 ? 1 : 0));
  const baselineMetrics = calculateClassificationMetrics(data.y_test, baselineProbs);

  const shiftedProbs = model.predictProba(shiftedX_test);
  const shiftedPredictions = shiftedProbs.map(p => (p >= 0.5 ? 1 : 0));
  const shiftedMetrics = calculateClassificationMetrics(data.y_test, shiftedProbs);

  let flippedPredictions = 0;
  for (let i = 0; i < nTest; i++) {
    if (baselinePredictions[i] !== shiftedPredictions[i]) {
      flippedPredictions++;
    }
  }

  const accuracyDropPct = baselineMetrics.accuracy > 0
    ? ((baselineMetrics.accuracy - shiftedMetrics.accuracy) / baselineMetrics.accuracy) * 100
    : 0;

  const f1DropPct = baselineMetrics.f1Score > 0
    ? ((baselineMetrics.f1Score - shiftedMetrics.f1Score) / baselineMetrics.f1Score) * 100
    : 0;

  const predictionFlipPct = (flippedPredictions / (nTest || 1)) * 100;

  let robustnessStatus: 'ROBUST' | 'MODERATE_DEGRADATION' | 'CRITICAL_FAILURE' = 'ROBUST';
  if (accuracyDropPct > 20 || f1DropPct > 25 || predictionFlipPct > 30) {
    robustnessStatus = 'CRITICAL_FAILURE';
  } else if (accuracyDropPct > 8 || f1DropPct > 10 || predictionFlipPct > 12) {
    robustnessStatus = 'MODERATE_DEGRADATION';
  }

  return {
    isSimulated: true,
    sampleCountBaseline: nTest,
    sampleCountDrifted: nTest,
    featureDriftMetrics,
    overallDriftIndex: Math.round((totalPsiSum / (nFeatures || 1)) * 1000) / 1000,
    driftedFeaturesCount: driftedCount,
    distributionComparison,
    modelRobustness: {
      modelName,
      baselineAccuracy: baselineMetrics.accuracy,
      driftedAccuracy: shiftedMetrics.accuracy,
      accuracyDropPct: Math.round(accuracyDropPct * 10) / 10,
      baselineF1: baselineMetrics.f1Score,
      driftedF1: shiftedMetrics.f1Score,
      f1DropPct: Math.round(f1DropPct * 10) / 10,
      predictionFlipPct: Math.round(predictionFlipPct * 10) / 10,
      status: robustnessStatus
    }
  };
}
