import React, { useState } from 'react';
import {
  DatasetSummary,
  ModelType,
  OptimizationResult,
  TrainedModelResult
} from '../lib/ml-engine/types';
import {
  Sliders,
  Play,
  Zap,
  TrendingUp,
  CheckCircle2,
  Cpu,
  Layers,
  Sparkles,
  ArrowUpRight,
  Clock,
  Activity
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
import { optimizeModelApi } from '../lib/api-client';

interface OptimizationViewProps {
  dataset: DatasetSummary | null;
  rawRows: Record<string, string | number>[];
  onOptimizedModelSaved: (model: TrainedModelResult) => void;
}

export const OptimizationView: React.FC<OptimizationViewProps> = ({
  dataset,
  rawRows,
  onOptimizedModelSaved
}) => {
  const [modelType, setModelType] = useState<ModelType>('random_forest');
  const [nTrials, setNTrials] = useState(15);
  const [objectiveMetric, setObjectiveMetric] = useState<'f1Score' | 'accuracy' | 'rocAuc'>('f1Score');
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optResult, setOptResult] = useState<OptimizationResult | null>(null);

  const handleRunOptimization = async () => {
    if (!dataset || rawRows.length === 0) return;
    setIsOptimizing(true);
    try {
      const res = await optimizeModelApi({
        rows: rawRows,
        targetCol: dataset.targetColumn,
        modelType,
        nTrials,
        objectiveMetric
      });
      setOptResult(res);
      if (res.optimizedModel) {
        onOptimizedModelSaved(res.optimizedModel);
      }
    } catch (err) {
      console.error('Optuna study failed', err);
    } finally {
      setIsOptimizing(false);
    }
  };

  const trialsChartData = optResult?.trials.map(t => ({
    trial: `#${t.trialNumber}`,
    score: Math.round(t.score * 1000) / 10,
    durationMs: t.durationMs
  })) || [];

  return (
    <div className="space-y-6">
      {/* Control Configuration Card */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <Sliders className="h-4 w-4 text-cyan-400" />
              Optuna Hyperparameter Optimization Engine
            </h3>
            <p className="text-xs text-slate-400">
              Tree-structured Parzen Estimator (TPE) Bayesian search exploring hyperparameter subspaces
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400 font-mono">Backend: Optuna Core v3.6</span>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
          {/* Target Model */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-300">Model to Tune</label>
            <select
              value={modelType}
              onChange={e => setModelType(e.target.value as ModelType)}
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-700 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 cursor-pointer"
            >
              <option value="random_forest">Random Forest Classifier</option>
              <option value="gradient_boosting">Gradient Boosting Classifier</option>
              <option value="logistic_regression">Logistic Regression</option>
            </select>
          </div>

          {/* Number of Trials */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 font-medium">Optimization Trials</span>
              <span className="font-mono text-cyan-400">{nTrials} Trials</span>
            </div>
            <input
              type="range"
              min="5"
              max="30"
              step="1"
              value={nTrials}
              onChange={e => setNTrials(Number(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer mt-2"
            />
          </div>

          {/* Optimization Objective */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-300">Objective Metric</label>
            <select
              value={objectiveMetric}
              onChange={e => setObjectiveMetric(e.target.value as any)}
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-700 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 cursor-pointer"
            >
              <option value="f1Score">Maximize F1-Score</option>
              <option value="accuracy">Maximize Accuracy</option>
              <option value="rocAuc">Maximize ROC-AUC</option>
            </select>
          </div>
        </div>

        {/* Execute Button */}
        <button
          onClick={handleRunOptimization}
          disabled={isOptimizing || !dataset}
          className="w-full py-3 rounded-xl bg-gradient-to-r from-cyan-500 via-indigo-600 to-purple-600 hover:from-cyan-400 hover:to-purple-500 text-white text-xs font-semibold shadow-lg shadow-cyan-500/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition cursor-pointer"
        >
          {isOptimizing ? (
            <>
              <Activity className="h-4 w-4 animate-spin text-white" />
              <span>Running Optuna Multi-Trial Optimization Study...</span>
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4" />
              <span>Launch Optuna Optimization Study</span>
            </>
          )}
        </button>
      </div>

      {/* Optimization Study Results */}
      {optResult && (
        <div className="space-y-6">
          {/* Best Trial Scorecards & Baseline Comparison */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-5 rounded-2xl bg-gradient-to-br from-cyan-950/40 via-slate-900 to-slate-900 border border-cyan-800/50 backdrop-blur-sm space-y-2">
              <span className="text-xs text-cyan-400 font-medium flex items-center gap-1.5">
                <CheckCircle2 className="h-4 w-4" />
                Best Trial (#{optResult.bestTrialNumber})
              </span>
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-bold font-mono text-white">
                  {(optResult.bestScore * 100).toFixed(1)}%
                </span>
                <span className="text-xs text-slate-400 font-mono uppercase">{optResult.objectiveMetric}</span>
              </div>
              <p className="text-xs text-slate-400">
                Found optimal hyperparameter convergence
              </p>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-2">
              <span className="text-xs text-slate-400 font-medium">Baseline vs Optimized Delta</span>
              <div className="flex items-baseline gap-2">
                <span
                  className={`text-3xl font-bold font-mono ${
                    optResult.improvementDelta >= 0 ? 'text-emerald-400' : 'text-slate-300'
                  }`}
                >
                  {optResult.improvementDelta >= 0 ? `+${(optResult.improvementDelta * 100).toFixed(1)}%` : `${(optResult.improvementDelta * 100).toFixed(1)}%`}
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  Base: {(optResult.baselineScore * 100).toFixed(1)}%
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Empirical boost achieved via TPE trial exploration
              </p>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-2">
              <span className="text-xs text-purple-400 font-medium">Optimal Parameters Discovered</span>
              <div className="space-y-1 font-mono text-xs text-slate-200">
                {Object.entries(optResult.bestParams).map(([k, v]) => (
                  <div key={k} className="flex justify-between border-b border-slate-800/60 py-0.5">
                    <span className="text-slate-400">{k}:</span>
                    <span className="text-cyan-300">{String(v)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Trial History Curve & Parameter Breakdown */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-cyan-400" />
                  Optuna Optimization History (Objective Value vs Trial #)
                </h4>
                <p className="text-xs text-slate-400">Trajectory of evaluated configurations across search space</p>
              </div>
              <span className="text-xs font-mono text-slate-400">{optResult.trials.length} trials completed</span>
            </div>

            <div className="h-60 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trialsChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="trial" stroke="#64748b" fontSize={11} tickLine={false} />
                  <YAxis domain={['auto', 'auto']} stroke="#64748b" fontSize={11} tickLine={false} unit="%" />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                  <Line type="monotone" dataKey="score" stroke="#06b6d4" strokeWidth={2.5} dot={{ r: 4, fill: '#06b6d4' }} activeDot={{ r: 6 }} name={`${optResult.objectiveMetric} (%)`} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Trials Audit Log Table */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
            <h4 className="text-sm font-semibold text-white">Full Trial Evaluation Log</h4>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 font-mono border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-4 font-normal">Trial</th>
                    <th className="py-2.5 px-4 font-normal">Parameters Evaluated</th>
                    <th className="py-2.5 px-4 font-normal">Score</th>
                    <th className="py-2.5 px-4 font-normal">Duration</th>
                    <th className="py-2.5 px-4 font-normal">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {optResult.trials.map(t => (
                    <tr
                      key={t.trialNumber}
                      className={`hover:bg-slate-800/40 transition ${
                        t.trialNumber === optResult.bestTrialNumber ? 'bg-cyan-950/30 font-semibold' : ''
                      }`}
                    >
                      <td className="py-2.5 px-4 font-mono text-cyan-400">
                        #{t.trialNumber} {t.trialNumber === optResult.bestTrialNumber && '★ (Best)'}
                      </td>
                      <td className="py-2.5 px-4 font-mono text-[11px] text-slate-400">
                        {JSON.stringify(t.params).replace(/["{}]/g, '').replace(/,/g, ', ')}
                      </td>
                      <td className="py-2.5 px-4 font-mono text-emerald-400 font-semibold">
                        {(t.score * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-4 font-mono text-slate-400">
                        {t.durationMs} ms
                      </td>
                      <td className="py-2.5 px-4">
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 font-mono">
                          {t.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
