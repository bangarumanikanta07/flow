export interface ModelEstimator {
  fit(X: number[][], y: number[]): void;
  predict(X: number[][]): number[];
  predictProba(X: number[][]): number[];
  getFeatureImportances(featureNames: string[]): { feature: string; importance: number }[];
  getParameterCount(): number;
  getComplexity(): { parameterCount: number; depth?: number; treeCount?: number; memoryKb: number };
}

/**
 * Logistic Regression with L2 Regularization & Sigmoid activation
 */
export class LogisticRegressionModel implements ModelEstimator {
  private weights: number[] = [];
  private bias = 0;
  private learningRate: number;
  private maxIter: number;
  private l2Reg: number;

  constructor(learningRate = 0.05, maxIter = 150, l2Reg = 0.01) {
    this.learningRate = learningRate;
    this.maxIter = maxIter;
    this.l2Reg = l2Reg;
  }

  fit(X: number[][], y: number[]): void {
    const nSamples = X.length;
    if (nSamples === 0) return;
    const nFeatures = X[0].length;

    this.weights = new Array(nFeatures).fill(0);
    this.bias = 0;

    for (let iter = 0; iter < this.maxIter; iter++) {
      const gradW = new Array(nFeatures).fill(0);
      let gradB = 0;

      for (let i = 0; i < nSamples; i++) {
        let z = this.bias;
        for (let j = 0; j < nFeatures; j++) {
          z += this.weights[j] * X[i][j];
        }
        const p = 1 / (1 + Math.exp(-Math.max(-25, Math.min(25, z))));
        const err = p - y[i];

        for (let j = 0; j < nFeatures; j++) {
          gradW[j] += err * X[i][j];
        }
        gradB += err;
      }

      const lr = this.learningRate / (1 + 0.005 * iter);
      for (let j = 0; j < nFeatures; j++) {
        const regTerm = this.l2Reg * this.weights[j];
        this.weights[j] -= lr * (gradW[j] / nSamples + regTerm);
      }
      this.bias -= lr * (gradB / nSamples);
    }
  }

  predictProba(X: number[][]): number[] {
    const probs: number[] = [];
    const nFeatures = this.weights.length;

    for (let i = 0; i < X.length; i++) {
      let z = this.bias;
      for (let j = 0; j < nFeatures; j++) {
        z += (this.weights[j] || 0) * (X[i][j] || 0);
      }
      const p = 1 / (1 + Math.exp(-Math.max(-25, Math.min(25, z))));
      probs.push(Math.round(p * 1000) / 1000);
    }
    return probs;
  }

  predict(X: number[][]): number[] {
    return this.predictProba(X).map(p => (p >= 0.5 ? 1 : 0));
  }

  getFeatureImportances(featureNames: string[]): { feature: string; importance: number }[] {
    const absWeights = this.weights.map(w => Math.abs(w));
    const total = absWeights.reduce((a, b) => a + b, 0) || 1;

    const list = featureNames.map((name, i) => ({
      feature: name,
      importance: Math.round(((absWeights[i] || 0) / total) * 1000) / 1000
    }));

    return list.sort((a, b) => b.importance - a.importance);
  }

  getParameterCount(): number {
    return this.weights.length + 1;
  }

  getComplexity(): { parameterCount: number; depth?: number; treeCount?: number; memoryKb: number } {
    return {
      parameterCount: this.weights.length + 1,
      memoryKb: Math.round(((this.weights.length * 8 + 32) / 1024) * 10) / 10
    };
  }
}

// Tree Node for Decision Tree & Ensembles
class TreeNode {
  isLeaf: boolean;
  value: number; // probability or prediction
  featureIndex: number;
  threshold: number;
  left: TreeNode | null;
  right: TreeNode | null;
  impurity: number;
  sampleCount: number;

  constructor(isLeaf = false, value = 0) {
    this.isLeaf = isLeaf;
    this.value = value;
    this.featureIndex = -1;
    this.threshold = 0;
    this.left = null;
    this.right = null;
    this.impurity = 0;
    this.sampleCount = 0;
  }
}

class DecisionTree {
  root: TreeNode | null = null;
  maxDepth: number;
  minSamplesSplit: number;
  maxFeaturesRatio: number;
  featureImportancesAccum: number[] = [];

  constructor(maxDepth = 6, minSamplesSplit = 4, maxFeaturesRatio = 1.0) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.maxFeaturesRatio = maxFeaturesRatio;
  }

  fit(X: number[][], y: number[], seed = 42): void {
    if (X.length === 0) return;
    const nFeatures = X[0].length;
    this.featureImportancesAccum = new Array(nFeatures).fill(0);
    this.root = this.buildTree(X, y, 0, nFeatures, seed);
  }

  private buildTree(X: number[][], y: number[], depth: number, nFeatures: number, seed: number): TreeNode {
    const nSamples = X.length;
    const posCount = y.filter(v => v >= 0.5).length;
    const probPos = nSamples > 0 ? posCount / nSamples : 0;

    // Gini impurity: 1 - (p^2 + (1-p)^2) = 2 * p * (1-p)
    const currentImpurity = 2 * probPos * (1 - probPos);

    const node = new TreeNode();
    node.sampleCount = nSamples;
    node.impurity = currentImpurity;
    node.value = probPos;

    // Base conditions for leaf
    if (depth >= this.maxDepth || nSamples < this.minSamplesSplit || currentImpurity === 0) {
      node.isLeaf = true;
      return node;
    }

    // Feature subspace sampling
    const featureIndices = Array.from({ length: nFeatures }, (_, i) => i);
    const numSubFeatures = Math.max(1, Math.round(nFeatures * this.maxFeaturesRatio));
    // deterministic shuffle
    for (let i = featureIndices.length - 1; i > 0; i--) {
      const j = Math.floor(((seed * 31 + depth * 17 + i) % 1000) / 1000 * (i + 1));
      const t = featureIndices[i];
      featureIndices[i] = featureIndices[j];
      featureIndices[j] = t;
    }
    const chosenFeatures = featureIndices.slice(0, numSubFeatures);

    let bestGain = -1;
    let bestFeature = -1;
    let bestThreshold = 0;

    for (const f of chosenFeatures) {
      const vals = X.map(row => row[f]);
      const uniqueVals = Array.from(new Set(vals)).sort((a, b) => a - b);
      if (uniqueVals.length <= 1) continue;

      // Evaluate candidate thresholds (up to 10 split candidates)
      const step = Math.max(1, Math.floor(uniqueVals.length / 10));
      for (let k = 0; k < uniqueVals.length - 1; k += step) {
        const threshold = (uniqueVals[k] + uniqueVals[k + 1]) / 2;

        let leftPos = 0;
        let leftCount = 0;
        let rightPos = 0;
        let rightCount = 0;

        for (let i = 0; i < nSamples; i++) {
          if (X[i][f] <= threshold) {
            leftCount++;
            if (y[i] >= 0.5) leftPos++;
          } else {
            rightCount++;
            if (y[i] >= 0.5) rightPos++;
          }
        }

        if (leftCount === 0 || rightCount === 0) continue;

        const pL = leftPos / leftCount;
        const pR = rightPos / rightCount;
        const impL = 2 * pL * (1 - pL);
        const impR = 2 * pR * (1 - pR);

        const weightedImp = (leftCount / nSamples) * impL + (rightCount / nSamples) * impR;
        const gain = currentImpurity - weightedImp;

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = f;
          bestThreshold = threshold;
        }
      }
    }

    if (bestGain <= 1e-5 || bestFeature === -1) {
      node.isLeaf = true;
      return node;
    }

    // Split node
    node.featureIndex = bestFeature;
    node.threshold = bestThreshold;
    this.featureImportancesAccum[bestFeature] += bestGain * nSamples;

    const leftX: number[][] = [];
    const leftY: number[] = [];
    const rightX: number[][] = [];
    const rightY: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      if (X[i][bestFeature] <= bestThreshold) {
        leftX.push(X[i]);
        leftY.push(y[i]);
      } else {
        rightX.push(X[i]);
        rightY.push(y[i]);
      }
    }

    node.left = this.buildTree(leftX, leftY, depth + 1, nFeatures, seed + 1);
    node.right = this.buildTree(rightX, rightY, depth + 1, nFeatures, seed + 2);

    return node;
  }

  predictRowProba(row: number[], node: TreeNode | null = this.root): number {
    if (!node) return 0.5;
    if (node.isLeaf) return node.value;
    if (row[node.featureIndex] <= node.threshold) {
      return this.predictRowProba(row, node.left);
    } else {
      return this.predictRowProba(row, node.right);
    }
  }
}

/**
 * Random Forest Classifier
 */
export class RandomForestClassifierModel implements ModelEstimator {
  private trees: DecisionTree[] = [];
  private nEstimators: number;
  private maxDepth: number;
  private minSamplesSplit: number;
  private maxFeaturesRatio: number;
  private featureImportances: number[] = [];

  constructor(nEstimators = 35, maxDepth = 6, minSamplesSplit = 4, maxFeaturesRatio = 0.75) {
    this.nEstimators = nEstimators;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.maxFeaturesRatio = maxFeaturesRatio;
  }

  fit(X: number[][], y: number[]): void {
    const nSamples = X.length;
    if (nSamples === 0) return;
    const nFeatures = X[0].length;
    this.trees = [];
    this.featureImportances = new Array(nFeatures).fill(0);

    for (let t = 0; t < this.nEstimators; t++) {
      // Bootstrap sampling (sampling with replacement)
      const bootX: number[][] = [];
      const bootY: number[] = [];
      let pseudoRand = (t + 1) * 317 + 89;

      for (let i = 0; i < nSamples; i++) {
        pseudoRand = (pseudoRand * 16807) % 2147483647;
        const randIdx = Math.floor((pseudoRand / 2147483647) * nSamples);
        bootX.push(X[randIdx]);
        bootY.push(y[randIdx]);
      }

      const tree = new DecisionTree(this.maxDepth, this.minSamplesSplit, this.maxFeaturesRatio);
      tree.fit(bootX, bootY, t * 100);
      this.trees.push(tree);

      for (let f = 0; f < nFeatures; f++) {
        this.featureImportances[f] += tree.featureImportancesAccum[f] || 0;
      }
    }

    const totalImp = this.featureImportances.reduce((a, b) => a + b, 0) || 1;
    this.featureImportances = this.featureImportances.map(v => v / totalImp);
  }

  predictProba(X: number[][]): number[] {
    const probs: number[] = [];
    for (let i = 0; i < X.length; i++) {
      let sumProb = 0;
      for (const tree of this.trees) {
        sumProb += tree.predictRowProba(X[i]);
      }
      const avg = sumProb / (this.trees.length || 1);
      probs.push(Math.round(avg * 1000) / 1000);
    }
    return probs;
  }

  predict(X: number[][]): number[] {
    return this.predictProba(X).map(p => (p >= 0.5 ? 1 : 0));
  }

  getFeatureImportances(featureNames: string[]): { feature: string; importance: number }[] {
    const list = featureNames.map((name, i) => ({
      feature: name,
      importance: Math.round((this.featureImportances[i] || 0) * 1000) / 1000
    }));
    return list.sort((a, b) => b.importance - a.importance);
  }

  getParameterCount(): number {
    return this.trees.length * Math.pow(2, this.maxDepth);
  }

  getComplexity(): { parameterCount: number; depth?: number; treeCount?: number; memoryKb: number } {
    const count = this.getParameterCount();
    return {
      parameterCount: count,
      depth: this.maxDepth,
      treeCount: this.trees.length,
      memoryKb: Math.round(((count * 24) / 1024) * 10) / 10
    };
  }
}

/**
 * Gradient Boosting Classifier (Residual Tree Boosting)
 */
export class GradientBoostingClassifierModel implements ModelEstimator {
  private trees: DecisionTree[] = [];
  private baseOdds = 0;
  private nEstimators: number;
  private learningRate: number;
  private maxDepth: number;
  private featureImportances: number[] = [];

  constructor(nEstimators = 30, learningRate = 0.1, maxDepth = 4) {
    this.nEstimators = nEstimators;
    this.learningRate = learningRate;
    this.maxDepth = maxDepth;
  }

  fit(X: number[][], y: number[]): void {
    const nSamples = X.length;
    if (nSamples === 0) return;
    const nFeatures = X[0].length;

    // Prior log odds
    const p1 = Math.max(0.01, Math.min(0.99, y.filter(v => v >= 0.5).length / nSamples));
    this.baseOdds = Math.log(p1 / (1 - p1));

    const fCurrent = new Array(nSamples).fill(this.baseOdds);
    this.trees = [];
    this.featureImportances = new Array(nFeatures).fill(0);

    for (let t = 0; t < this.nEstimators; t++) {
      // Compute negative gradient residuals: r_i = y_i - p_i
      const residuals: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        const p = 1 / (1 + Math.exp(-Math.max(-25, Math.min(25, fCurrent[i]))));
        residuals.push(y[i] - p);
      }

      // Convert residuals into binary pseudo-target for tree split finding
      const pseudoBinaryY = residuals.map(r => (r >= 0 ? 1 : 0));

      const tree = new DecisionTree(this.maxDepth, 4, 0.85);
      tree.fit(X, pseudoBinaryY, t * 73);
      this.trees.push(tree);

      for (let f = 0; f < nFeatures; f++) {
        this.featureImportances[f] += tree.featureImportancesAccum[f] || 0;
      }

      // Update predictions
      for (let i = 0; i < nSamples; i++) {
        const treeProb = tree.predictRowProba(X[i]);
        const step = (treeProb - 0.5) * 2; // scale between -1 and 1
        fCurrent[i] += this.learningRate * step;
      }
    }

    const totalImp = this.featureImportances.reduce((a, b) => a + b, 0) || 1;
    this.featureImportances = this.featureImportances.map(v => v / totalImp);
  }

  predictProba(X: number[][]): number[] {
    const probs: number[] = [];

    for (let i = 0; i < X.length; i++) {
      let f = this.baseOdds;
      for (const tree of this.trees) {
        const treeProb = tree.predictRowProba(X[i]);
        f += this.learningRate * (treeProb - 0.5) * 2;
      }
      const p = 1 / (1 + Math.exp(-Math.max(-25, Math.min(25, f))));
      probs.push(Math.round(p * 1000) / 1000);
    }

    return probs;
  }

  predict(X: number[][]): number[] {
    return this.predictProba(X).map(p => (p >= 0.5 ? 1 : 0));
  }

  getFeatureImportances(featureNames: string[]): { feature: string; importance: number }[] {
    const list = featureNames.map((name, i) => ({
      feature: name,
      importance: Math.round((this.featureImportances[i] || 0) * 1000) / 1000
    }));
    return list.sort((a, b) => b.importance - a.importance);
  }

  getParameterCount(): number {
    return this.trees.length * Math.pow(2, this.maxDepth);
  }

  getComplexity(): { parameterCount: number; depth?: number; treeCount?: number; memoryKb: number } {
    const count = this.getParameterCount();
    return {
      parameterCount: count,
      depth: this.maxDepth,
      treeCount: this.trees.length,
      memoryKb: Math.round(((count * 32) / 1024) * 10) / 10
    };
  }
}
