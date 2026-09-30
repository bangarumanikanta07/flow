import React from 'react';
import {
  TrainedModelResult,
  DriftAnalysisResult
} from '../lib/ml-engine/types';
import {
  GitCompare,
  Download,
  Zap,
  Clock,
  Layers,
  ShieldCheck,
  TrendingUp,
  BarChart2,
  CheckCircle2
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ScatterChart,
  Scatter,
  ZAxis
} from 'recharts';

// Drop percentages are positive when a metric fell; show the signed change instead of prefixing '-'
const formatMetricChange = (dropPct: number) => (dropPct > 0 ? `-${dropPct}%` : `+${Math.abs(dropPct)}%`);
const metricChangeClass = (dropPct: number) => (dropPct > 0 ? 'text-rose-400' : 'text-emerald-400');

interface ComparisonViewProps {
  models: TrainedModelResult[];
  activeDrift: DriftAnalysisResult | null;
}

export const ComparisonView: React.FC<ComparisonViewProps> = ({
  models,
  activeDrift
}) => {
  const exportReport = () => {
    const reportData = {
      exportedAt: new Date().toISOString(),
      modelsCount: models.length,
      models: models.map(m => ({
        name: m.modelName,
        type: m.modelType,
        isOptimized: m.isOptimized,
        metrics: m.metrics,
        hyperparameters: m.hyperparameters,
        topFeatures: m.featureImportances.slice(0, 5)
      })),
      driftAnalysis: activeDrift ? {
        overallPsi: activeDrift.overallDriftIndex,
        driftedFeaturesCount: activeDrift.driftedFeaturesCount,
        robustness: activeDrift.modelRobustness
      } : null
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `driftforge_model_comparison_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const chartData = models.map(m => ({
    name: m.modelName.replace(' Classifier', '').replace(' (Optuna Optimized)', ' (Opt)'),
    Accuracy: Math.round(m.metrics.accuracy * 1000) / 10,
    Precision: Math.round(m.metrics.precision * 1000) / 10,
    Recall: Math.round(m.metrics.recall * 1000) / 10,
    F1: Math.round(m.metrics.f1Score * 1000) / 10,
    latency: m.metrics.latencyMs
  }));

  const efficiencyData = models.map(m => ({
    name: m.modelName,
    latency: m.metrics.latencyMs,
    f1: Math.round(m.metrics.f1Score * 1000) / 10,
    params: m.metrics.modelComplexity.parameterCount
  }));

  return (
    <div className="space-y-6">
      {/* Header and Export Action */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <GitCompare className="h-4 w-4 text-cyan-400" />
            Model Efficiency, Robustness & Tradeoff Matrix
          </h3>
          <p className="text-xs text-slate-400">
            Side-by-side benchmark comparing baseline vs Optuna-tuned architectures across speed, accuracy, and drift retention
          </p>
        </div>

        <button
          onClick={exportReport}
          disabled={models.length === 0}
          className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-xs font-medium flex items-center gap-2 transition cursor-pointer disabled:opacity-50"
        >
          <Download className="h-3.5 w-3.5 text-cyan-400" />
          <span>Export Comparison (JSON)</span>
        </button>
      </div>

      {models.length > 0 ? (
        <>
          {/* Main Comparison Metrics Chart */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-semibold text-white">Multi-Metric Evaluation Benchmark</h4>
                <p className="text-xs text-slate-400">Accuracy, Precision, Recall, and F1 across trained architectures</p>
              </div>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="name" stroke="#64748b" fontSize={11} tickLine={false} />
                  <YAxis domain={[40, 100]} stroke="#64748b" fontSize={11} tickLine={false} unit="%" />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                  <Bar dataKey="Accuracy" fill="#818cf8" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Precision" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Recall" fill="#a855f7" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="F1" fill="#10b981" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Efficiency Tradeoff: Latency (ms) vs F1 Score */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
              <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                <Clock className="h-4 w-4 text-cyan-400" />
                Inference Latency vs F1-Score Tradeoff
              </h4>
              <p className="text-xs text-slate-400">
                Pareto efficiency curve: models in the upper-left offer maximum predictive score with minimum latency
              </p>

              <div className="h-56 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: -20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis type="number" dataKey="latency" name="Latency" unit="ms" stroke="#64748b" fontSize={10} tickLine={false} label={{ value: 'Inference Latency (ms)', position: 'insideBottom', offset: -5, fontSize: 10, fill: '#64748b' }} />
                    <YAxis type="number" dataKey="f1" name="F1-Score" unit="%" stroke="#64748b" fontSize={10} tickLine={false} domain={[50, 100]} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                    <Scatter name="Models" data={efficiencyData} fill="#06b6d4" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Robustness Under Drift Performance */}
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4 flex flex-col justify-between">
              <div>
                <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                  <ShieldCheck className="h-4 w-4 text-purple-400" />
                  Robustness & Distribution Drift Resilience
                </h4>
                <p className="text-xs text-slate-400">
                  Model behavior under synthetic covariate shift perturbations
                </p>

                {activeDrift ? (
                  <div className="mt-4 space-y-3">
                    <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-400">Evaluated Model</span>
                        <span className="text-white font-medium">{activeDrift.modelRobustness.modelName}</span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-400">Baseline F1 → Shifted F1</span>
                        <span className="font-mono text-cyan-400">
                          {(activeDrift.modelRobustness.baselineF1 * 100).toFixed(1)}% → {(activeDrift.modelRobustness.driftedF1 * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-400">Performance Drop</span>
                        <span className={`font-mono ${metricChangeClass(activeDrift.modelRobustness.f1DropPct)}`}>
                          {formatMetricChange(activeDrift.modelRobustness.f1DropPct)}
                        </span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-400">Classification Flip Rate</span>
                        <span className="font-mono text-amber-400">
                          {activeDrift.modelRobustness.predictionFlipPct}% of predictions shifted
                        </span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="mt-8 text-center text-xs text-slate-500 italic p-6 border border-dashed border-slate-800 rounded-xl">
                    Run the Drift & Robustness tab to benchmark distribution resilience
                  </div>
                )}
              </div>

              <div className="pt-3 border-t border-slate-800 text-[11px] text-slate-500">
                Ensemble architectures typically retain higher drift robustness compared to unregularized linear baselines.
              </div>
            </div>
          </div>

          {/* Full Detailed Comparison Matrix Table */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
            <h4 className="text-sm font-semibold text-white">Full Architectural Comparison Table</h4>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 font-mono border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-4 font-normal">Model Architecture</th>
                    <th className="py-2.5 px-4 font-normal">Optimized</th>
                    <th className="py-2.5 px-4 font-normal">Accuracy</th>
                    <th className="py-2.5 px-4 font-normal">Precision</th>
                    <th className="py-2.5 px-4 font-normal">Recall</th>
                    <th className="py-2.5 px-4 font-normal">F1-Score</th>
                    <th className="py-2.5 px-4 font-normal">ROC-AUC</th>
                    <th className="py-2.5 px-4 font-normal">Latency</th>
                    <th className="py-2.5 px-4 font-normal">Memory Footprint</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {models.map(m => (
                    <tr key={m.id} className="hover:bg-slate-800/30 transition">
                      <td className="py-2.5 px-4 font-medium text-white flex items-center gap-1.5">
                        <span>{m.modelName}</span>
                        {m.isOptimized && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
                            Optuna
                          </span>
                        )}
                      </td>
                      <td className="py-2.5 px-4">
                        {m.isOptimized ? (
                          <span className="text-cyan-400 font-mono">TPE Tuned</span>
                        ) : (
                          <span className="text-slate-500 font-mono">Default</span>
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
                        {m.metrics.modelComplexity.memoryKb} KB
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : (
        <div className="py-16 text-center text-xs text-slate-500 italic p-8 border border-dashed border-slate-800 rounded-2xl">
          No trained models available for comparison. Train multiple models in the Model Training tab first.
        </div>
      )}
    </div>
  );
};
