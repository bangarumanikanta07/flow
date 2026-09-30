import React, { useState } from 'react';
import {
  DatasetSummary,
  ModelType,
  DriftAnalysisResult,
  TrainedModelResult
} from '../lib/ml-engine/types';
import {
  Wind,
  AlertTriangle,
  Play,
  RotateCcw,
  Sliders,
  CheckCircle2,
  ShieldAlert,
  BarChart3,
  TrendingDown,
  Layers,
  ArrowRight
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend
} from 'recharts';
import { analyzeDriftApi } from '../lib/api-client';

interface DriftViewProps {
  dataset: DatasetSummary | null;
  rawRows: Record<string, string | number>[];
  activeModel: TrainedModelResult | null;
  onDriftAnalyzed: (result: DriftAnalysisResult) => void;
}

export const DriftView: React.FC<DriftViewProps> = ({
  dataset,
  rawRows,
  activeModel,
  onDriftAnalyzed
}) => {
  const [meanShiftPct, setMeanShiftPct] = useState(30); // 30% shift
  const [varianceScale, setVarianceScale] = useState(1.4); // 40% variance increase
  const [noiseLevel, setNoiseLevel] = useState(0.2);
  const [selectedFeature, setSelectedFeature] = useState<string>('');
  const [isSimulating, setIsSimulating] = useState(false);
  const [driftResult, setDriftResult] = useState<DriftAnalysisResult | null>(null);

  const numericalFeatures = dataset?.columns.filter(c => c.type === 'numerical' && c.name !== dataset.targetColumn) || [];

  const handleSimulate = async () => {
    if (!dataset || rawRows.length === 0) return;
    setIsSimulating(true);

    try {
      const res = await analyzeDriftApi({
        rows: rawRows,
        targetCol: dataset.targetColumn,
        modelType: activeModel?.modelType || 'random_forest',
        shiftConfig: {
          meanShiftPct,
          varianceScale,
          noiseLevel,
          affectedFeatures: selectedFeature ? [selectedFeature] : []
        }
      });

      setDriftResult(res);
      onDriftAnalyzed(res);
    } catch (err) {
      console.error('Drift simulation failed', err);
    } finally {
      setIsSimulating(false);
    }
  };

  const chartFeature = selectedFeature || (numericalFeatures[0]?.name ?? '');
  const distributionData = (driftResult?.distributionComparison[chartFeature] || []).map(b => ({
    range: b.binLabel,
    Baseline: Math.round(b.baselineFrequency * 1000) / 10,
    Shifted: Math.round(b.driftedFrequency * 1000) / 10
  }));

  return (
    <div className="space-y-6">
      {/* Distinction Banner: Simulated vs Real-World */}
      <div className="p-4 rounded-xl bg-purple-950/30 border border-purple-800/50 flex items-start gap-3">
        <Wind className="h-5 w-5 text-purple-400 shrink-0 mt-0.5" />
        <div className="text-xs space-y-1">
          <span className="font-semibold text-purple-200">
            Simulated Covariate & Distribution Shift Engine
          </span>
          <p className="text-slate-400 leading-relaxed">
            This module executes controlled synthetic distribution perturbations (mean drift $\Delta \mu$, variance scaling $\sigma^2 \cdot s$, and noise injection) to stress-test model degradation. In production deployments, DriftForge monitors real-time streaming telemetry using the identical Kolmogorov-Smirnov, Wasserstein, and PSI mathematical estimators.
          </p>
        </div>
      </div>

      {/* Simulation Controls Panel */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <Sliders className="h-4 w-4 text-cyan-400" />
              Distribution Shift Simulator & Stress Tester
            </h3>
            <p className="text-xs text-slate-400">Configure synthetic drift perturbations across continuous features</p>
          </div>

          <div className="text-xs font-mono text-slate-400">
            Model Under Test: <span className="text-cyan-400">{activeModel?.modelName || 'Random Forest (Default)'}</span>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-2">
          {/* Mean Shift Slider */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 font-medium">Mean Shift ($\Delta \mu$)</span>
              <span className="font-mono text-cyan-400">+{meanShiftPct}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={meanShiftPct}
              onChange={e => setMeanShiftPct(Number(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer"
            />
            <p className="text-[11px] text-slate-500">Offsets feature mean in standard deviations</p>
          </div>

          {/* Variance Scaling Slider */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 font-medium">Variance Dispersion ($s$)</span>
              <span className="font-mono text-purple-400">{varianceScale}x</span>
            </div>
            <input
              type="range"
              min="0.8"
              max="2.5"
              step="0.1"
              value={varianceScale}
              onChange={e => setVarianceScale(Number(e.target.value))}
              className="w-full accent-purple-400 cursor-pointer"
            />
            <p className="text-[11px] text-slate-500">Broadens tail spread and kurtosis</p>
          </div>

          {/* Noise Injection */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 font-medium">Noise Contamination</span>
              <span className="font-mono text-indigo-400">{(noiseLevel * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="0.5"
              step="0.05"
              value={noiseLevel}
              onChange={e => setNoiseLevel(Number(e.target.value))}
              className="w-full accent-indigo-400 cursor-pointer"
            />
            <p className="text-[11px] text-slate-500">Simulates sensor noise & data corruption</p>
          </div>

          {/* Targeted Feature Filter */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-300">Feature Target</label>
            <select
              value={selectedFeature}
              onChange={e => setSelectedFeature(e.target.value)}
              className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 cursor-pointer"
            >
              <option value="">All Numerical Features</option>
              {numericalFeatures.map(f => (
                <option key={f.name} value={f.name}>
                  {f.name}
                </option>
              ))}
            </select>
            <p className="text-[11px] text-slate-500">Select specific column or perturb all</p>
          </div>
        </div>

        {/* Simulate Action */}
        <button
          onClick={handleSimulate}
          disabled={isSimulating || !dataset}
          className="w-full py-3 rounded-xl bg-gradient-to-r from-purple-600 via-indigo-600 to-cyan-500 hover:from-purple-500 hover:to-cyan-400 text-white text-xs font-semibold shadow-lg shadow-purple-500/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition cursor-pointer"
        >
          {isSimulating ? (
            <>
              <Wind className="h-4 w-4 animate-spin text-white" />
              <span>Simulating Perturbations & Computing Statistical Distances...</span>
            </>
          ) : (
            <>
              <Play className="h-4 w-4 fill-white" />
              <span>Inject Distribution Shift & Compute Drift Metrics</span>
            </>
          )}
        </button>
      </div>

      {/* Drift Results & Impact Summary */}
      {driftResult && (
        <div className="space-y-6">
          {/* Robustness Impact Scorecards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Robustness Status Badge */}
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-2">
              <span className="text-xs text-slate-400 font-medium">Model Resilience</span>
              <div className="flex items-center gap-2">
                <span
                  className={`text-xl font-bold font-mono ${
                    driftResult.modelRobustness.status === 'ROBUST'
                      ? 'text-emerald-400'
                      : driftResult.modelRobustness.status === 'MODERATE_DEGRADATION'
                        ? 'text-amber-400'
                        : 'text-rose-400'
                  }`}
                >
                  {driftResult.modelRobustness.status.replace('_', ' ')}
                </span>
              </div>
              <span className="text-[11px] text-slate-500">
                Prediction Flip Rate: {driftResult.modelRobustness.predictionFlipPct}%
              </span>
            </div>

            {/* Accuracy Drop */}
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-2">
              <span className="text-xs text-slate-400 font-medium">Accuracy Retention</span>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold font-mono text-white">
                  {(driftResult.modelRobustness.driftedAccuracy * 100).toFixed(1)}%
                </span>
                <span className="text-xs font-mono text-rose-400">
                  (-{driftResult.modelRobustness.accuracyDropPct}%)
                </span>
              </div>
              <span className="text-[11px] text-slate-500">
                Baseline was {(driftResult.modelRobustness.baselineAccuracy * 100).toFixed(1)}%
              </span>
            </div>

            {/* F1 Drop */}
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-2">
              <span className="text-xs text-slate-400 font-medium">F1-Score Degradation</span>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold font-mono text-cyan-300">
                  {(driftResult.modelRobustness.driftedF1 * 100).toFixed(1)}%
                </span>
                <span className="text-xs font-mono text-rose-400">
                  (-{driftResult.modelRobustness.f1DropPct}%)
                </span>
              </div>
              <span className="text-[11px] text-slate-500">
                Baseline was {(driftResult.modelRobustness.baselineF1 * 100).toFixed(1)}%
              </span>
            </div>

            {/* Overall PSI */}
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-2">
              <span className="text-xs text-slate-400 font-medium">Average PSI Index</span>
              <div className="flex items-baseline gap-2">
                <span
                  className={`text-2xl font-bold font-mono ${
                    driftResult.overallDriftIndex > 0.25
                      ? 'text-rose-400'
                      : driftResult.overallDriftIndex > 0.1
                        ? 'text-amber-400'
                        : 'text-emerald-400'
                  }`}
                >
                  {driftResult.overallDriftIndex}
                </span>
              </div>
              <span className="text-[11px] text-slate-500">
                {driftResult.driftedFeaturesCount} of {driftResult.featureDriftMetrics.length} features flagged
              </span>
            </div>
          </div>

          {/* Distribution Histogram Overlay Chart */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                  <BarChart3 className="h-4 w-4 text-cyan-400" />
                  Density Distribution Comparison: Baseline vs Shifted ({chartFeature || 'Selected Feature'})
                </h4>
                <p className="text-xs text-slate-400">10-quantile frequency histogram comparing original empirical data with drifted space</p>
              </div>

              {/* Feature Dropdown for Chart */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-400">Feature:</span>
                <select
                  value={chartFeature}
                  onChange={e => setSelectedFeature(e.target.value)}
                  className="px-2.5 py-1 rounded-lg bg-slate-950 border border-slate-700 text-xs text-slate-200"
                >
                  {numericalFeatures.map(f => (
                    <option key={f.name} value={f.name}>
                      {f.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={distributionData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="range" stroke="#64748b" fontSize={10} tickLine={false} />
                  <YAxis stroke="#64748b" fontSize={10} tickLine={false} unit="%" />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                  <Bar dataKey="Baseline" fill="#818cf8" radius={[4, 4, 0, 0]} name="Baseline Distribution (%)" />
                  <Bar dataKey="Shifted" fill="#ec4899" radius={[4, 4, 0, 0]} name="Shifted Distribution (%)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Statistical Drift Metrics Table */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-semibold text-white">Feature Drift Statistical Metrics</h4>
                <p className="text-xs text-slate-400">
                  Kolmogorov-Smirnov 2-sample tests, Wasserstein distance, and Population Stability Index (PSI)
                </p>
              </div>
              <span className="text-xs font-mono text-slate-400">Alpha: $\alpha=0.05$</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 font-mono border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-3 font-normal">Feature Name</th>
                    <th className="py-2.5 px-3 font-normal">Baseline $\mu$</th>
                    <th className="py-2.5 px-3 font-normal">Shifted $\mu$</th>
                    <th className="py-2.5 px-3 font-normal">$\Delta \mu$ (%)</th>
                    <th className="py-2.5 px-3 font-normal">KS Statistic ($D$)</th>
                    <th className="py-2.5 px-3 font-normal">KS p-value</th>
                    <th className="py-2.5 px-3 font-normal">Wasserstein</th>
                    <th className="py-2.5 px-3 font-normal">PSI</th>
                    <th className="py-2.5 px-3 font-normal">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {driftResult.featureDriftMetrics.map(m => (
                    <tr key={m.feature} className="hover:bg-slate-800/30 transition">
                      <td className="py-2.5 px-3 font-mono font-medium text-white">{m.feature}</td>
                      <td className="py-2.5 px-3 font-mono text-slate-400">{m.baselineMean}</td>
                      <td className="py-2.5 px-3 font-mono text-slate-300">{m.driftedMean}</td>
                      <td className="py-2.5 px-3 font-mono text-cyan-400">
                        {m.meanShiftPct >= 0 ? `+${m.meanShiftPct}%` : `${m.meanShiftPct}%`}
                      </td>
                      <td className="py-2.5 px-3 font-mono">{m.ksStatistic}</td>
                      <td className="py-2.5 px-3 font-mono text-slate-400">
                        {m.ksPValue < 0.001 ? '< 0.001' : m.ksPValue}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-purple-300">{m.wassersteinDistance}</td>
                      <td
                        className={`py-2.5 px-3 font-mono font-semibold ${
                          m.psi > 0.25 ? 'text-rose-400' : m.psi > 0.1 ? 'text-amber-400' : 'text-emerald-400'
                        }`}
                      >
                        {m.psi}
                      </td>
                      <td className="py-2.5 px-3">
                        <span
                          className={`text-[10px] px-2 py-0.5 rounded font-mono ${
                            m.severity === 'high'
                              ? 'bg-rose-950 text-rose-300 border border-rose-800'
                              : m.severity === 'moderate'
                                ? 'bg-amber-950 text-amber-300 border border-amber-800'
                                : 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                          }`}
                        >
                          {m.severity.toUpperCase()}
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
