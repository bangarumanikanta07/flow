export type TaskType = 'classification' | 'regression';

export type ModelType = 'logistic_regression' | 'random_forest' | 'gradient_boosting';

export interface DatasetColumnInfo {
  name: string;
  type: 'numerical' | 'categorical';
  nonNullCount: number;
  nullCount: number;
  uniqueCount: number;
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  sampleValues: (string | number)[];
}

export interface DatasetSummary {
  id: string;
  name: string;
  rowCount: number;
  columnCount: number;
  columns: DatasetColumnInfo[];
  targetColumn: string;
  taskType: TaskType;
  features: string[];
  headRows: Record<string, string | number>[];
  missingValuesTotal: number;
}

export interface ConfusionMatrix {
  truePositive: number;
  falsePositive: number;
  trueNegative: number;
  falseNegative: number;
  classes: [string, string];
}

export interface RocCurvePoint {
  fpr: number;
  tpr: number;
  threshold: number;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  rocAuc: number;
  latencyMs: number;
  trainTimeMs: number;
  modelComplexity: {
    parameterCount: number;
    depth?: number;
    treeCount?: number;
    memoryKb: number;
  };
  confusionMatrix: ConfusionMatrix;
  rocCurve: RocCurvePoint[];
}

export interface TrainedModelResult {
  id: string;
  modelType: ModelType;
  modelName: string;
  taskType: TaskType;
  trainedAt: string;
  hyperparameters: Record<string, number | string | boolean>;
  metrics: ModelMetrics;
  featureImportances: { feature: string; importance: number }[];
  isOptimized?: boolean;
}

export interface OptunaTrial {
  trialNumber: number;
  params: Record<string, number | string | boolean>;
  score: number;
  durationMs: number;
  status: 'COMPLETED' | 'PRUNED';
}

export interface OptimizationResult {
  modelType: ModelType;
  objectiveMetric: 'f1Score' | 'accuracy' | 'rocAuc';
  nTrials: number;
  bestTrialNumber: number;
  bestParams: Record<string, number | string | boolean>;
  bestScore: number;
  baselineScore: number;
  improvementDelta: number;
  trials: OptunaTrial[];
  optimizedModel: TrainedModelResult;
}

export interface DriftMetric {
  feature: string;
  baselineMean: number;
  driftedMean: number;
  meanShiftPct: number;
  baselineStd: number;
  driftedStd: number;
  wassersteinDistance: number;
  ksStatistic: number;
  ksPValue: number;
  psi: number; // Population Stability Index
  driftDetected: boolean;
  severity: 'low' | 'moderate' | 'high';
}

export interface DistributionBin {
  binLabel: string;
  binStart: number;
  binEnd: number;
  baselineFrequency: number;
  driftedFrequency: number;
}

export interface DriftAnalysisResult {
  isSimulated: boolean;
  sampleCountBaseline: number;
  sampleCountDrifted: number;
  featureDriftMetrics: DriftMetric[];
  overallDriftIndex: number;
  driftedFeaturesCount: number;
  distributionComparison: Record<string, DistributionBin[]>;
  modelRobustness: {
    modelName: string;
    baselineAccuracy: number;
    driftedAccuracy: number;
    accuracyDropPct: number;
    baselineF1: number;
    driftedF1: number;
    f1DropPct: number;
    predictionFlipPct: number;
    status: 'ROBUST' | 'MODERATE_DEGRADATION' | 'CRITICAL_FAILURE';
  };
}

export interface LocalExplanationContribution {
  feature: string;
  featureValue: string | number;
  attribution: number; // positive increases probability of class 1, negative decreases
  runningProbability: number;
}

export interface ExplainabilityResult {
  method: 'TreeSHAP' | 'PermutationImportance' | 'LogitCoefficientAttribution';
  baseValue: number; // expected probability E[f(x)]
  globalImportances: { feature: string; importance: number; relativePct: number }[];
  samplePrediction?: {
    sampleIndex: number;
    actualLabel: string | number;
    predictedProbability: number;
    predictedLabel: string | number;
    localContributions: LocalExplanationContribution[];
  };
}

export interface ExperimentRecord {
  id: string;
  timestamp: string;
  name: string;
  datasetName: string;
  modelType: ModelType;
  isOptimized: boolean;
  accuracy: number;
  f1Score: number;
  latencyMs: number;
  driftRobustnessScore?: number;
}

export interface PredictionResultItem {
  rowIndex: number;
  inputFeatures: Record<string, string | number>;
  predictedClass: 'Fraud' | 'Legitimate';
  predictedLabel: number;
  fraudProbability: number;
  confidenceScore: number;
  isVerifiedOutcome: boolean;
  verifiedOutcome: string;
  isCorrect?: boolean;
}

export interface BatchPredictionResponse {
  modelUsed: string;
  totalSamples: number;
  fraudCount: number;
  legitimateCount: number;
  fraudRatePct: number;
  inferenceLatencyMs: number;
  predictions: PredictionResultItem[];
}
