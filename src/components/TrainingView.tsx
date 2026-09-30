import React, { useState } from 'react';
import {
  DatasetSummary,
  ModelType,
  TrainedModelResult
} from '../lib/ml-engine/types';
import {
  Cpu,
  Play,
  RotateCcw,
  CheckCircle2,
  Sliders,
  Layers,
  Clock,
  Zap,
  Activity,
  BarChart2,
  TrendingUp,
  FileCheck
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid
} from 'recharts';
import { trainModelApi } from '../lib/api-client';

interface TrainingViewProps {
  dataset: DatasetSummary | null;
  rawRows: Record<string, string | number>[];
  trainedModels: TrainedModelResult[];
  onModelTrained: (model: TrainedModelResult) => void;
}

export const TrainingView: React.FC<TrainingViewProps> = ({
  dataset,
  rawRows,
  trainedModels,
  onModelTrained
}) => {
  const [selectedModelType, setSelectedModelType] = useState<ModelType>('random_forest');
  const [testSplitRatio, setTestSplitRatio] = useState(0.25);
  const [randomSeed, setRandomSeed] = useState(42);

  // Hyperparameters
  const [nEstimators, setNEstimators] = useState(35);
  const [maxDepth, setMaxDepth] = useState(6);
  const [learningRate, setLearningRate] = useState(0.1);
  const [l2Reg, setL2Reg] = useState(0.01);
  const [maxIter, setMaxIter] = useState(150);

  const [isTraining, setIsTraining] = useState(false);
  const [currentModel, setCurrentModel] = useState<TrainedModelResult | null>(
    trainedModels.length > 0 ? trainedModels[trainedModels.length - 1] : null
  );

  const handleTrain = async () => {
    if (!dataset || rawRows.length === 0) return;
    setIsTraining(true);

    try {
      const hyperparameters: Record<string, any> = {};
      if (selectedModelType === 'logistic_regression') {
        hyperparameters.learningRate = learningRate;
        hyperparameters.maxIter = maxIter;
        hyperparameters.l2Reg = l2Reg;
      } else if (selectedModelType === 'gradient_boosting') {
        hyperparameters.nEstimators = nEstimators;
        hyperparameters.learningRate = learningRate;
        hyperparameters.maxDepth = maxDepth;
      } else {
        hyperparameters.nEstimators = nEstimators;
        hyperparameters.maxDepth = maxDepth;
        hyperparameters.minSamplesSplit = 4;
        hyperparameters.maxFeaturesRatio = 0.75;
      }

      const trainedResult = await trainModelApi({
        rows: rawRows,
        targetCol: dataset.targetColumn,
        modelType: selectedModelType,
        hyperparameters,
        testRatio: testSplitRatio,
        seed: randomSeed
      });

      setCurrentModel(trainedResult);
      onModelTrained(trainedResult);
    } catch (err) {
      console.error('Training failed', err);
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Configuration & Training Control Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model Architecture Selector & Parameters */}
        <div className="lg:col-span-1 p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-5">
          <div className="space-y-1">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Cpu className="h-4 w-4 text-cyan-400" />
              Model Configuration
            </h3>
            <p className="text-xs text-slate-400">Select model algorithm and tune initial parameters</p>
          </div>

          {/* Model Type Selector */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-300">Model Architecture</label>
            <div className="grid grid-cols-1 gap-2">
              {[
                { id: 'random_forest' as ModelType, name: 'Random Forest', desc: 'Ensemble bagging of decision trees' },
                { id: 'gradient_boosting' as ModelType, name: 'Gradient Boosting', desc: 'Sequential residual gradient boosting' },
                { id: 'logistic_regression' as ModelType, name: 'Logistic Regression', desc: 'Convex linear model with L2 regularization' }
              ].map(item => (
                <button
                  key={item.id}
                  onClick={() => setSelectedModelType(item.id)}
                  className={`p-3 rounded-xl text-left border transition cursor-pointer ${
                    selectedModelType === item.id
                      ? 'bg-cyan-950/40 border-cyan-500/80 shadow-sm'
                      : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="text-xs font-semibold text-white">{item.name}</div>
                  <div className="text-[11px] text-slate-400 mt-0.5">{item.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Dynamic Hyperparameter Sliders */}
          <div className="space-y-3.5 pt-2 border-t border-slate-800/80">
            <span className="text-xs font-medium text-slate-300 flex items-center gap-1.5">
              <Sliders className="h-3.5 w-3.5 text-purple-400" />
              Hyperparameters
            </span>

            {selectedModelType !== 'logistic_regression' && (
              <>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-400">Number of Estimators (Trees)</span>
                    <span className="font-mono text-cyan-400">{nEstimators}</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    step="5"
                    value={nEstimators}
                    onChange={e => setNEstimators(Number(e.target.value))}
                    className="w-full accent-cyan-400 cursor-pointer"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-400">Max Depth</span>
                    <span className="font-mono text-cyan-400">{maxDepth}</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="12"
                    step="1"
                    value={maxDepth}
                    onChange={e => setMaxDepth(Number(e.target.value))}
                    className="w-full accent-cyan-400 cursor-pointer"
                  />
                </div>
              </>
            )}

            {selectedModelType === 'gradient_boosting' && (
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Learning Rate (Shrinkage)</span>
                  <span className="font-mono text-cyan-400">{learningRate}</span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="0.3"
                  step="0.01"
                  value={learningRate}
                  onChange={e => setLearningRate(Number(e.target.value))}
                  className="w-full accent-cyan-400 cursor-pointer"
                />
              </div>
            )}

            {selectedModelType === 'logistic_regression' && (
              <>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-400">Learning Rate</span>
                    <span className="font-mono text-cyan-400">{learningRate}</span>
                  </div>
                  <input
                    type="range"
                    min="0.01"
                    max="0.2"
                    step="0.01"
                    value={learningRate}
                    onChange={e => setLearningRate(Number(e.target.value))}
                    className="w-full accent-cyan-400 cursor-pointer"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-400">L2 Regularization (Weight Decay)</span>
                    <span className="font-mono text-cyan-400">{l2Reg}</span>
                  </div>
                  <input
                    type="range"
                    min="0.0001"
                    max="0.1"
                    step="0.005"
                    value={l2Reg}
                    onChange={e => setL2Reg(Number(e.target.value))}
                    className="w-full accent-cyan-400 cursor-pointer"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-400">Max Iterations (Epochs)</span>
                    <span className="font-mono text-cyan-400">{maxIter}</span>
                  </div>
                  <input
                    type="range"
                    min="50"
                    max="300"
                    step="10"
                    value={maxIter}
                    onChange={e => setMaxIter(Number(e.target.value))}
                    className="w-full accent-cyan-400 cursor-pointer"
                  />
                </div>
              </>
            )}

            {/* Split & Seed */}
            <div className="pt-2 border-t border-slate-800/80 space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Test Partition Ratio</span>
                <span className="font-mono text-slate-300">{(testSplitRatio * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.4"
                step="0.05"
                value={testSplitRatio}
                onChange={e => setTestSplitRatio(Number(e.target.value))}
                className="w-full accent-purple-400 cursor-pointer"
              />
            </div>
          </div>

          {/* Train Button */}
          <button
            onClick={handleTrain}
            disabled={isTraining || !dataset}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-cyan-500 via-indigo-600 to-purple-600 hover:from-cyan-400 hover:to-purple-500 text-white text-xs font-semibold shadow-lg shadow-cyan-500/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition cursor-pointer"
          >
            {isTraining ? (
              <>
                <Activity className="h-4 w-4 animate-spin text-white" />
                <span>Fitting Model on Partitions...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4 fill-white" />
                <span>Execute Model Training</span>
              </>
            )}
          </button>
        </div>

        {/* Results, Metrics & Curves */}
        <div className="lg:col-span-2 space-y-6">
          {currentModel ? (
            <>
              {/* Metric Scorecards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm">
                  <span className="text-xs text-slate-400">Test Accuracy</span>
                  <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">
                    {(currentModel.metrics.accuracy * 100).toFixed(1)}%
                  </div>
                  <span className="text-[11px] text-slate-500">Correct classifications</span>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm">
                  <span className="text-xs text-slate-400">F1-Score</span>
                  <div className="text-2xl font-bold font-mono text-cyan-400 mt-1">
                    {(currentModel.metrics.f1Score * 100).toFixed(1)}%
                  </div>
                  <span className="text-[11px] text-slate-500">Harmonic mean P & R</span>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm">
                  <span className="text-xs text-slate-400">ROC-AUC</span>
                  <div className="text-2xl font-bold font-mono text-purple-400 mt-1">
                    {currentModel.metrics.rocAuc.toFixed(3)}
                  </div>
                  <span className="text-[11px] text-slate-500">Area under curve</span>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm">
                  <span className="text-xs text-slate-400">Inference Latency</span>
                  <div className="text-2xl font-bold font-mono text-indigo-300 mt-1">
                    {currentModel.metrics.latencyMs} <span className="text-xs text-slate-400">ms</span>
                  </div>
                  <span className="text-[11px] text-slate-500">Train: {currentModel.metrics.trainTimeMs}ms</span>
                </div>
              </div>

              {/* Confusion Matrix and ROC Curve */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Confusion Matrix Card */}
                <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
                  <div className="flex items-center justify-between">
                    <h4 className="text-xs font-semibold text-white flex items-center gap-1.5">
                      <Layers className="h-4 w-4 text-cyan-400" />
                      Confusion Matrix
                    </h4>
                    <span className="text-[11px] text-slate-400 font-mono">
                      Precision: {(currentModel.metrics.precision * 100).toFixed(1)}% · Recall: {(currentModel.metrics.recall * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-center">
                    {/* True Positive */}
                    <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-800/50">
                      <span className="text-[10px] font-mono uppercase text-emerald-400 block mb-1">True Positive (TP)</span>
                      <span className="text-3xl font-bold font-mono text-emerald-300">
                        {currentModel.metrics.confusionMatrix.truePositive}
                      </span>
                    </div>

                    {/* False Positive */}
                    <div className="p-4 rounded-xl bg-rose-950/30 border border-rose-800/40">
                      <span className="text-[10px] font-mono uppercase text-rose-400 block mb-1">False Positive (FP)</span>
                      <span className="text-3xl font-bold font-mono text-rose-300">
                        {currentModel.metrics.confusionMatrix.falsePositive}
                      </span>
                    </div>

                    {/* False Negative */}
                    <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-800/40">
                      <span className="text-[10px] font-mono uppercase text-amber-400 block mb-1">False Negative (FN)</span>
                      <span className="text-3xl font-bold font-mono text-amber-300">
                        {currentModel.metrics.confusionMatrix.falseNegative}
                      </span>
                    </div>

                    {/* True Negative */}
                    <div className="p-4 rounded-xl bg-indigo-950/40 border border-indigo-800/50">
                      <span className="text-[10px] font-mono uppercase text-indigo-400 block mb-1">True Negative (TN)</span>
                      <span className="text-3xl font-bold font-mono text-indigo-300">
                        {currentModel.metrics.confusionMatrix.trueNegative}
                      </span>
                    </div>
                  </div>
                </div>

                {/* ROC Curve Chart */}
                <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
                  <div className="flex items-center justify-between">
                    <h4 className="text-xs font-semibold text-white flex items-center gap-1.5">
                      <TrendingUp className="h-4 w-4 text-purple-400" />
                      ROC Curve (Receiver Operating Characteristic)
                    </h4>
                    <span className="text-xs font-mono font-semibold text-purple-400">
                      AUC: {currentModel.metrics.rocAuc.toFixed(3)}
                    </span>
                  </div>

                  <div className="h-48 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={currentModel.metrics.rocCurve} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="fpr" type="number" domain={[0, 1]} stroke="#64748b" fontSize={10} tickLine={false} label={{ value: 'FPR', position: 'insideBottomRight', offset: -5, fontSize: 10, fill: '#64748b' }} />
                        <YAxis dataKey="tpr" type="number" domain={[0, 1]} stroke="#64748b" fontSize={10} tickLine={false} label={{ value: 'TPR', angle: -90, position: 'insideLeft', fontSize: 10, fill: '#64748b' }} />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                        <Line type="monotone" dataKey="tpr" stroke="#a855f7" strokeWidth={2.5} dot={false} name="TPR" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="p-12 text-center border border-dashed border-slate-800 rounded-2xl space-y-3">
              <Cpu className="h-10 w-10 text-slate-600 mx-auto" />
              <div className="text-sm font-medium text-slate-300">Ready to Train</div>
              <p className="text-xs text-slate-500 max-w-md mx-auto">
                Configure hyperparameters on the left and click "Execute Model Training" to fit models and generate empirical performance metrics.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Model Comparison Table */}
      {trainedModels.length > 0 && (
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-white">Trained Models Benchmark Matrix</h4>
            <span className="text-xs text-slate-400 font-mono">{trainedModels.length} models trained</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950 text-slate-400 font-mono border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-4 font-normal">Model Name</th>
                  <th className="py-2.5 px-4 font-normal">Accuracy</th>
                  <th className="py-2.5 px-4 font-normal">Precision</th>
                  <th className="py-2.5 px-4 font-normal">Recall</th>
                  <th className="py-2.5 px-4 font-normal">F1-Score</th>
                  <th className="py-2.5 px-4 font-normal">ROC-AUC</th>
                  <th className="py-2.5 px-4 font-normal">Latency</th>
                  <th className="py-2.5 px-4 font-normal">Parameters</th>
                  <th className="py-2.5 px-4 font-normal">Select</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {trainedModels.map(m => (
                  <tr
                    key={m.id}
                    className={`hover:bg-slate-800/40 transition cursor-pointer ${
                      currentModel?.id === m.id ? 'bg-slate-800/30' : ''
                    }`}
                    onClick={() => setCurrentModel(m)}
                  >
                    <td className="py-2.5 px-4 font-medium text-white flex items-center gap-1.5">
                      <span>{m.modelName}</span>
                      {m.isOptimized && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
                          Optuna
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-4 font-mono text-emerald-400">
                      {(m.metrics.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-4 font-mono">
                      {(m.metrics.precision * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-4 font-mono">
                      {(m.metrics.recall * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-4 font-mono font-semibold text-cyan-400">
                      {(m.metrics.f1Score * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-4 font-mono text-purple-400">
                      {m.metrics.rocAuc.toFixed(3)}
                    </td>
                    <td className="py-2.5 px-4 font-mono text-slate-400">
                      {m.metrics.latencyMs} ms
                    </td>
                    <td className="py-2.5 px-4 font-mono text-slate-400">
                      {m.metrics.modelComplexity.parameterCount}
                    </td>
                    <td className="py-2.5 px-4">
                      <button
                        onClick={e => {
                          e.stopPropagation();
                          setCurrentModel(m);
                        }}
                        className="text-xs text-cyan-400 hover:text-cyan-300 transition"
                      >
                        {currentModel?.id === m.id ? 'Active' : 'Inspect'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
