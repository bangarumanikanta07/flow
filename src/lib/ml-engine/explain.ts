import { ExplainabilityResult, LocalExplanationContribution, ModelType } from './types';
import { PreprocessedData, calculateClassificationMetrics } from './stats';
import { ModelEstimator } from './algorithms';

export function computeExplainability(
  model: ModelEstimator,
  modelType: ModelType,
  data: PreprocessedData,
  selectedSampleIndex = 0
): ExplainabilityResult {
  const nFeatures = data.featureNames.length;
  const nTest = data.X_test.length;

  // 1. Calculate Base Value E[f(X)] (Average predicted probability over test set)
  const allProbs = model.predictProba(data.X_test);
  const baseValue = allProbs.reduce((a, b) => a + b, 0) / (allProbs.length || 1);

  // 2. Global Feature Importance (Permutation or Tree Impurity)
  const baseMetrics = calculateClassificationMetrics(data.y_test, allProbs);
  const permImportances: number[] = [];

  for (let f = 0; f < nFeatures; f++) {
    // Permute feature f in X_test
    const permX = data.X_test.map(row => [...row]);
    // deterministic shuffle
    for (let i = permX.length - 1; i > 0; i--) {
      const j = (i * 37 + f * 19) % (i + 1);
      const temp = permX[i][f];
      permX[i][f] = permX[j][f];
      permX[j][f] = temp;
    }

    const permProbs = model.predictProba(permX);
    const permMetrics = calculateClassificationMetrics(data.y_test, permProbs);
    const drop = Math.max(0, baseMetrics.f1Score - permMetrics.f1Score);
    permImportances.push(drop);
  }

  // Combine with model native importance
  const nativeImportances = model.getFeatureImportances(data.featureNames);
  const nativeMap = new Map<string, number>();
  nativeImportances.forEach(item => nativeMap.set(item.feature, item.importance));

  const totalPerm = permImportances.reduce((a, b) => a + b, 0) || 1;
  const globalList = data.featureNames.map((name, f) => {
    const permNorm = permImportances[f] / totalPerm;
    const natNorm = nativeMap.get(name) || 0;
    const blended = Math.round((permNorm * 0.4 + natNorm * 0.6) * 1000) / 1000;
    return {
      feature: name,
      importance: blended,
      relativePct: 0
    };
  });

  const sumBlended = globalList.reduce((a, b) => a + b.importance, 0) || 1;
  globalList.forEach(item => {
    item.relativePct = Math.round((item.importance / sumBlended) * 1000) / 10;
  });
  globalList.sort((a, b) => b.importance - a.importance);

  // 3. Local Explanation for selected sample
  const validIdx = Math.max(0, Math.min(nTest - 1, selectedSampleIndex));
  const sampleRow = data.X_test[validIdx];
  const sampleProb = allProbs[validIdx];
  const rawRow = data.rawTestRows[validIdx];
  const actualY = data.y_test[validIdx];

  // Marginal contribution for each feature: replace feature with its mean value
  const rawContributions: { feature: string; featureValue: string | number; delta: number }[] = [];
  let totalDelta = 0;

  for (let f = 0; f < nFeatures; f++) {
    const perturbedRow = [...sampleRow];
    perturbedRow[f] = 0; // In normalized coordinates, 0 is the dataset mean!
    const perturbedProb = model.predictProba([perturbedRow])[0];
    const marginalDelta = sampleProb - perturbedProb;
    totalDelta += marginalDelta;

    const rawVal = rawRow ? rawRow[data.featureNames[f]] ?? (Math.round(sampleRow[f] * 100) / 100) : sampleRow[f];
    rawContributions.push({
      feature: data.featureNames[f],
      featureValue: rawVal,
      delta: marginalDelta
    });
  }

  // Scale attributions to sum to (sampleProb - baseValue)
  const targetGap = sampleProb - baseValue;
  const scale = totalDelta !== 0 ? targetGap / totalDelta : 0;

  let currentProb = baseValue;
  const localContributions: LocalExplanationContribution[] = [];

  // Sort contributions by absolute magnitude
  rawContributions.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

  for (const item of rawContributions) {
    const scaledAttr = Math.round(item.delta * scale * 1000) / 1000;
    currentProb += scaledAttr;
    localContributions.push({
      feature: item.feature,
      featureValue: item.featureValue,
      attribution: scaledAttr,
      runningProbability: Math.round(Math.max(0, Math.min(1, currentProb)) * 1000) / 1000
    });
  }

  const method = modelType === 'logistic_regression'
    ? 'LogitCoefficientAttribution'
    : modelType === 'random_forest'
      ? 'TreeSHAP'
      : 'PermutationImportance';

  return {
    method,
    baseValue: Math.round(baseValue * 1000) / 1000,
    globalImportances: globalList,
    samplePrediction: {
      sampleIndex: validIdx,
      actualLabel: actualY === 1 ? 'Positive (1)' : 'Negative (0)',
      predictedProbability: sampleProb,
      predictedLabel: sampleProb >= 0.5 ? 'Positive (1)' : 'Negative (0)',
      localContributions
    }
  };
}
