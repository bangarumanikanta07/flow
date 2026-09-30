import React from 'react';
import {
  DatasetSummary,
  TrainedModelResult,
  DriftAnalysisResult,
  ExperimentRecord
} from '../lib/ml-engine/types';
import {
  Activity,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Layers,
  Zap,
  ShieldAlert,
  ArrowRight,
  Database,
  BarChart3,
  TrendingUp,
  Sliders
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  LineChart,
  Line
} from 'recharts';

interface DashboardViewProps {
  dataset: DatasetSummary | null;
  models: TrainedModelResult[];
  activeDrift: DriftAnalysisResult | null;
  history: ExperimentRecord[];
  onNavigate: (tab: any) => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  dataset,
  models,
  activeDrift,
  history,
  onNavigate
}) => {
  const bestModel = models.length > 0
    ? [...models].sort((a, b) => b.metrics.f1Score - a.metrics.f1Score)[0]
    : null;

  const comparisonData = models.map(m => ({
    name: m.modelName.replace(' Classifier', '').replace(' (Optuna Optimized)', '*'),
    accuracy: Math.round(m.metrics.accuracy * 1000) / 10,
    f1Score: Math.round(m.metrics.f1Score * 1000) / 10,
    latency: m.metrics.latencyMs
  }));

  return (
    <div className="space-y-6">
      {/* Top Banner if no dataset or no models */}
      {!dataset && (
        <div className="p-6 rounded-2xl bg-gradient-to-r from-purple-950/40 via-slate-900 to-cyan-950/30 border border-purple-800/40 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="space-y-1">
            <h3 className="text-base font-semibold text-white">No Dataset Loaded</h3>
            <p className="text-xs text-slate-400">
              Load a pre-configured enterprise sample dataset or upload your custom CSV to start training and drift detection.
            </p>
          </div>
          <button
            onClick={() => onNavigate('dataset')}
            className="px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white text-xs font-medium shadow-md shadow-cyan-500/20 flex items-center gap-2 transition cursor-pointer"
          >
            <span>Open Dataset Management</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Dataset Status */}
        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Dataset Status</span>
            <Database className="h-4 w-4 text-purple-400" />
          </div>
          {dataset ? (
            <div>
              <div className="text-xl font-bold text-slate-100 truncate">{dataset.name}</div>
              <div className="text-xs text-slate-400 mt-1 flex items-center gap-2">
                <span>{dataset.rowCount} rows</span>
                <span>·</span>
                <span>{dataset.columnCount} columns</span>
                <span>·</span>
                <span className="text-emerald-400">{dataset.missingValuesTotal === 0 ? '0 missing' : `${dataset.missingValuesTotal} nulls`}</span>
              </div>
            </div>
          ) : (
            <div className="text-slate-500 text-sm italic">Dataset awaiting upload</div>
          )}
        </div>

        {/* Best Model F1 Score */}
        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Best Model Performance</span>
            <Zap className="h-4 w-4 text-cyan-400" />
          </div>
          {bestModel ? (
            <div>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold font-mono text-cyan-400">
                  {(bestModel.metrics.f1Score * 100).toFixed(1)}%
                </span>
                <span className="text-xs text-slate-400">F1-Score</span>
              </div>
              <div className="text-xs text-slate-400 mt-1 flex items-center gap-2">
                <span className="truncate max-w-[140px] text-slate-300">{bestModel.modelName}</span>
                <span>·</span>
                <span className="font-mono text-emerald-400">Acc: {(bestModel.metrics.accuracy * 100).toFixed(1)}%</span>
              </div>
            </div>
          ) : (
            <div className="text-slate-500 text-sm italic">No models trained yet</div>
          )}
        </div>

        {/* Inference Latency & Efficiency */}
        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Inference & Complexity</span>
            <Clock className="h-4 w-4 text-indigo-400" />
          </div>
          {bestModel ? (
            <div>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold font-mono text-indigo-300">
                  {bestModel.metrics.latencyMs} <span className="text-sm font-normal text-slate-400">ms</span>
                </span>
              </div>
              <div className="text-xs text-slate-400 mt-1 flex items-center gap-2">
                <span>Memory: {bestModel.metrics.modelComplexity.memoryKb} KB</span>
                <span>·</span>
                <span>{bestModel.metrics.modelComplexity.parameterCount} params</span>
              </div>
            </div>
          ) : (
            <div className="text-slate-500 text-sm italic">Inference benchmark idle</div>
          )}
        </div>

        {/* Data Drift Status */}
        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Data Drift Health</span>
            <ShieldAlert className="h-4 w-4 text-rose-400" />
          </div>
          {activeDrift ? (
            <div>
              <div className="flex items-baseline gap-2">
                <span
                  className={`text-2xl font-bold font-mono ${
                    activeDrift.driftedFeaturesCount > 0 ? 'text-amber-400' : 'text-emerald-400'
                  }`}
                >
                  {activeDrift.driftedFeaturesCount > 0 ? 'Shift Detected' : 'Stable'}
                </span>
              </div>
              <div className="text-xs text-slate-400 mt-1 flex items-center gap-2">
                <span>PSI: {activeDrift.overallDriftIndex}</span>
                <span>·</span>
                <span>{activeDrift.driftedFeaturesCount} of {activeDrift.featureDriftMetrics.length} features drifted</span>
              </div>
            </div>
          ) : (
            <div>
              <div className="text-base font-semibold text-slate-300">Drift Monitor Ready</div>
              <div className="text-xs text-slate-500 mt-1">Run distribution shift simulation to benchmark</div>
            </div>
          )}
        </div>
      </div>

      {/* Main Charts & Analytics Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model Performance Comparison Chart */}
        <div className="lg:col-span-2 p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-cyan-400" />
                Trained Models Benchmark (Accuracy vs F1-Score)
              </h4>
              <p className="text-xs text-slate-400">Actual evaluation metrics computed on held-out test split</p>
            </div>
            {models.length > 0 && (
              <button
                onClick={() => onNavigate('training')}
                className="text-xs text-cyan-400 hover:text-cyan-300 font-medium flex items-center gap-1 cursor-pointer"
              >
                <span>Train New</span>
                <ArrowRight className="h-3 w-3" />
              </button>
            )}
          </div>

          {comparisonData.length > 0 ? (
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="name" stroke="#64748b" fontSize={11} tickLine={false} />
                  <YAxis domain={[40, 100]} stroke="#64748b" fontSize={11} tickLine={false} unit="%" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                  />
                  <Bar dataKey="accuracy" name="Accuracy (%)" fill="#818cf8" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="f1Score" name="F1-Score (%)" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex flex-col items-center justify-center text-center p-6 border border-dashed border-slate-800 rounded-xl space-y-3">
              <Activity className="h-8 w-8 text-slate-600 animate-pulse" />
              <div className="space-y-1">
                <div className="text-sm font-medium text-slate-300">No Model Trained Yet</div>
                <p className="text-xs text-slate-500 max-w-sm">
                  Upload a dataset and execute training on Logistic Regression, Random Forest, or Gradient Boosting to generate metrics.
                </p>
              </div>
              <button
                onClick={() => onNavigate(dataset ? 'training' : 'dataset')}
                className="px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 transition cursor-pointer"
              >
                {dataset ? 'Start Training' : 'Select Dataset'}
              </button>
            </div>
          )}
        </div>

        {/* Quick Action Workflow & Drift Status */}
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4 flex flex-col justify-between">
          <div>
            <h4 className="text-sm font-semibold text-white flex items-center gap-2 mb-1">
              <TrendingUp className="h-4 w-4 text-purple-400" />
              ML Lifecycle Workflow
            </h4>
            <p className="text-xs text-slate-400 mb-4">Complete end-to-end adaptive machine learning pipeline</p>

            <div className="space-y-2.5">
              <div
                onClick={() => onNavigate('dataset')}
                className={`p-3 rounded-xl border flex items-center justify-between cursor-pointer transition ${
                  dataset ? 'bg-slate-900/80 border-slate-700 hover:border-slate-600' : 'bg-purple-950/20 border-purple-800/60'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className={`h-7 w-7 rounded-lg flex items-center justify-center ${dataset ? 'bg-emerald-950 text-emerald-400' : 'bg-purple-900 text-purple-300'}`}>
                    {dataset ? <CheckCircle2 className="h-4 w-4" /> : '1'}
                  </div>
                  <div>
                    <div className="text-xs font-medium text-slate-200">1. Data Ingestion</div>
                    <div className="text-[11px] text-slate-500">{dataset ? `${dataset.rowCount} rows ready` : 'Upload or load sample'}</div>
                  </div>
                </div>
                <ArrowRight className="h-3.5 w-3.5 text-slate-500" />
              </div>

              <div
                onClick={() => onNavigate('training')}
                className={`p-3 rounded-xl border flex items-center justify-between cursor-pointer transition ${
                  models.length > 0 ? 'bg-slate-900/80 border-slate-700 hover:border-slate-600' : 'bg-slate-900/30 border-slate-800'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className={`h-7 w-7 rounded-lg flex items-center justify-center ${models.length > 0 ? 'bg-emerald-950 text-emerald-400' : 'bg-slate-800 text-slate-400'}`}>
                    {models.length > 0 ? <CheckCircle2 className="h-4 w-4" /> : '2'}
                  </div>
                  <div>
                    <div className="text-xs font-medium text-slate-200">2. Model Training</div>
                    <div className="text-[11px] text-slate-500">{models.length} model(s) trained</div>
                  </div>
                </div>
                <ArrowRight className="h-3.5 w-3.5 text-slate-500" />
              </div>

              <div
                onClick={() => onNavigate('optimization')}
                className="p-3 rounded-xl border border-slate-800 bg-slate-900/40 hover:bg-slate-900/70 hover:border-slate-700 flex items-center justify-between cursor-pointer transition"
              >
                <div className="flex items-center gap-3">
                  <div className="h-7 w-7 rounded-lg bg-slate-800 text-slate-400 flex items-center justify-center">
                    3
                  </div>
                  <div>
                    <div className="text-xs font-medium text-slate-200">3. Optuna Hyperparameter Tuning</div>
                    <div className="text-[11px] text-slate-500">Multi-trial Bayesian search</div>
                  </div>
                </div>
                <ArrowRight className="h-3.5 w-3.5 text-slate-500" />
              </div>

              <div
                onClick={() => onNavigate('drift')}
                className="p-3 rounded-xl border border-slate-800 bg-slate-900/40 hover:bg-slate-900/70 hover:border-slate-700 flex items-center justify-between cursor-pointer transition"
              >
                <div className="flex items-center gap-3">
                  <div className="h-7 w-7 rounded-lg bg-slate-800 text-slate-400 flex items-center justify-center">
                    4
                  </div>
                  <div>
                    <div className="text-xs font-medium text-slate-200">4. Drift & Robustness Shift</div>
                    <div className="text-[11px] text-slate-500">KS-test, Wasserstein & PSI</div>
                  </div>
                </div>
                <ArrowRight className="h-3.5 w-3.5 text-slate-500" />
              </div>
            </div>
          </div>

          <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400">
            <span>Core Version: <span className="font-mono text-cyan-400">v1.4</span></span>
            <span>Real Scikit/Optuna Engine</span>
          </div>
        </div>
      </div>

      {/* Recent Experiment History Table */}
      <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-sm font-semibold text-white">Recent Experiment History</h4>
            <p className="text-xs text-slate-400">Audit trail of trained models, hyperparameter evaluations, and performance</p>
          </div>
          <span className="text-xs font-mono text-slate-400">{history.length} runs recorded</span>
        </div>

        {history.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/60 text-slate-400 font-mono border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-4 font-normal">Timestamp</th>
                  <th className="py-2.5 px-4 font-normal">Model Architecture</th>
                  <th className="py-2.5 px-4 font-normal">Dataset</th>
                  <th className="py-2.5 px-4 font-normal">Optimized</th>
                  <th className="py-2.5 px-4 font-normal">Accuracy</th>
                  <th className="py-2.5 px-4 font-normal">F1-Score</th>
                  <th className="py-2.5 px-4 font-normal">Latency</th>
                  <th className="py-2.5 px-4 font-normal">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {history.map(exp => (
                  <tr key={exp.id} className="hover:bg-slate-800/30 transition">
                    <td className="py-2.5 px-4 font-mono text-slate-400">
                      {new Date(exp.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-2.5 px-4 font-medium text-white flex items-center gap-1.5">
                      <span>{exp.name}</span>
                      {exp.isOptimized && (
                        <span className="px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800 text-[10px]">
                          OPT
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-4 text-slate-400 truncate max-w-[140px]">{exp.datasetName}</td>
                    <td className="py-2.5 px-4">
                      {exp.isOptimized ? (
                        <span className="text-cyan-400 font-mono">Optuna TPE</span>
                      ) : (
                        <span className="text-slate-500 font-mono">Baseline</span>
                      )}
                    </td>
                    <td className="py-2.5 px-4 font-mono text-emerald-400">
                      {(exp.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-4 font-mono font-semibold text-cyan-400">
                      {(exp.f1Score * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-4 font-mono text-slate-400">
                      {exp.latencyMs} ms
                    </td>
                    <td className="py-2.5 px-4">
                      <button
                        onClick={() => onNavigate('comparison')}
                        className="text-xs text-indigo-400 hover:text-indigo-300 transition cursor-pointer"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="py-8 text-center text-xs text-slate-500 italic">
            No training runs registered yet. Start training in the Model Training tab to populate experiment tracking.
          </div>
        )}
      </div>
    </div>
  );
};
