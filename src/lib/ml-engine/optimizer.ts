import { ModelType, OptimizationResult, OptunaTrial, TrainedModelResult } from './types';
import { PreprocessedData, calculateClassificationMetrics } from './stats';
import { LogisticRegressionModel, RandomForestClassifierModel, GradientBoostingClassifierModel, ModelEstimator } from './algorithms';

export function runHyperparameterOptimization(
  modelType: ModelType,
  data: PreprocessedData,
  nTrials = 12,
  objectiveMetric: 'f1Score' | 'accuracy' | 'rocAuc' = 'f1Score'
): OptimizationResult {
  const trials: OptunaTrial[] = [];
  let bestScore = -1;
  let bestTrialNum = 1;
  let bestParams: Record<string, number | string | boolean> = {};
  let bestModelInstance: ModelEstimator | null = null;
  let baselineScore = 0;

  // Evaluate baseline first
  const baselineModel = createModel(modelType, getDefaultParams(modelType));
  baselineModel.fit(data.X_train, data.y_train);
  const baselineProbs = baselineModel.predictProba(data.X_test);
  const baseMetrics = calculateClassificationMetrics(data.y_test, baselineProbs);
  baselineScore = baseMetrics[objectiveMetric];

  for (let trial = 1; trial <= nTrials; trial++) {
    const startTime = performance.now();
    const params = sampleHyperparameters(modelType, trial);

    const model = createModel(modelType, params);
    model.fit(data.X_train, data.y_train);

    const probs = model.predictProba(data.X_test);
    const metrics = calculateClassificationMetrics(data.y_test, probs);
    const duration = Math.round((performance.now() - startTime) * 10) / 10;
    const score = metrics[objectiveMetric];

    trials.push({
      trialNumber: trial,
      params,
      score: Math.round(score * 1000) / 1000,
      durationMs: duration,
      status: 'COMPLETED'
    });

    if (score > bestScore) {
      bestScore = score;
      bestTrialNum = trial;
      bestParams = params;
      bestModelInstance = model;
    }
  }

  // Construct optimized model result
  const latStart = performance.now();
  const finalProbs = bestModelInstance!.predictProba(data.X_test);
  const latencyMs = Math.round((performance.now() - latStart) * 10) / 10;
  const finalMetrics = calculateClassificationMetrics(data.y_test, finalProbs);
  const featImportances = bestModelInstance!.getFeatureImportances(data.featureNames);

  const optimizedModel: TrainedModelResult = {
    id: `opt_${modelType}_${Date.now()}`,
    modelType,
    modelName: `${formatModelName(modelType)} (Optuna Optimized)`,
    taskType: 'classification',
    trainedAt: new Date().toISOString(),
    hyperparameters: bestParams,
    metrics: {
      ...finalMetrics,
      latencyMs,
      trainTimeMs: Math.round(trials.reduce((a, b) => a + b.durationMs, 0)),
      modelComplexity: bestModelInstance!.getComplexity()
    },
    featureImportances: featImportances,
    isOptimized: true
  };

  return {
    modelType,
    objectiveMetric,
    nTrials,
    bestTrialNumber: bestTrialNum,
    bestParams,
    bestScore: Math.round(bestScore * 1000) / 1000,
    baselineScore: Math.round(baselineScore * 1000) / 1000,
    improvementDelta: Math.round((bestScore - baselineScore) * 1000) / 1000,
    trials,
    optimizedModel
  };
}

function getDefaultParams(type: ModelType): Record<string, number | string | boolean> {
  switch (type) {
    case 'logistic_regression':
      return { learningRate: 0.05, maxIter: 100, l2Reg: 0.01 };
    case 'random_forest':
      return { nEstimators: 20, maxDepth: 5, minSamplesSplit: 4, maxFeaturesRatio: 0.7 };
    case 'gradient_boosting':
      return { nEstimators: 20, learningRate: 0.1, maxDepth: 3 };
  }
}

function sampleHyperparameters(type: ModelType, trialIdx: number): Record<string, number | string | boolean> {
  const pseudoSeed = (trialIdx * 9301 + 49297) % 233280;
  const rand1 = (pseudoSeed % 1000) / 1000;
  const rand2 = ((pseudoSeed * 37) % 1000) / 1000;
  const rand3 = ((pseudoSeed * 139) % 1000) / 1000;

  switch (type) {
    case 'logistic_regression': {
      const lrValues = [0.01, 0.03, 0.05, 0.08, 0.12];
      const maxIterValues = [80, 120, 160, 200, 250];
      const l2Values = [0.0001, 0.001, 0.01, 0.05, 0.1];
      return {
        learningRate: lrValues[Math.floor(rand1 * lrValues.length)],
        maxIter: maxIterValues[Math.floor(rand2 * maxIterValues.length)],
        l2Reg: l2Values[Math.floor(rand3 * l2Values.length)]
      };
    }
    case 'random_forest': {
      const nEst = [15, 25, 40, 60, 80];
      const depths = [3, 5, 7, 9, 11];
      const minSplits = [2, 4, 6, 8];
      const featRatios = [0.5, 0.7, 0.85, 1.0];
      return {
        nEstimators: nEst[Math.floor(rand1 * nEst.length)],
        maxDepth: depths[Math.floor(rand2 * depths.length)],
        minSamplesSplit: minSplits[Math.floor(rand3 * minSplits.length)],
        maxFeaturesRatio: featRatios[Math.floor(((rand1 + rand2) % 1) * featRatios.length)]
      };
    }
    case 'gradient_boosting': {
      const nEst = [15, 25, 35, 50, 70];
      const lrs = [0.03, 0.06, 0.1, 0.15, 0.2];
      const depths = [2, 3, 4, 5];
      return {
        nEstimators: nEst[Math.floor(rand1 * nEst.length)],
        learningRate: lrs[Math.floor(rand2 * lrs.length)],
        maxDepth: depths[Math.floor(rand3 * depths.length)]
      };
    }
  }
}

function createModel(type: ModelType, params: Record<string, any>): ModelEstimator {
  switch (type) {
    case 'logistic_regression':
      return new LogisticRegressionModel(params.learningRate, params.maxIter, params.l2Reg);
    case 'random_forest':
      return new RandomForestClassifierModel(
        params.nEstimators,
        params.maxDepth,
        params.minSamplesSplit,
        params.maxFeaturesRatio
      );
    case 'gradient_boosting':
      return new GradientBoostingClassifierModel(
        params.nEstimators,
        params.learningRate,
        params.maxDepth
      );
  }
}

function formatModelName(type: ModelType): string {
  switch (type) {
    case 'logistic_regression':
      return 'Logistic Regression';
    case 'random_forest':
      return 'Random Forest';
    case 'gradient_boosting':
      return 'Gradient Boosting';
  }
}
