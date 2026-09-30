import React, { useState, useEffect } from 'react';
import {
  DatasetSummary,
  ModelType,
  ExplainabilityResult,
  TrainedModelResult
} from '../lib/ml-engine/types';
import {
  Search,
  Layers,
  Sparkles,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  Info,
  ChevronLeft,
  ChevronRight,
  BarChart3,
  Activity
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell
} from 'recharts';
import { explainPredictionsApi } from '../lib/api-client';

interface ExplainViewProps {
  dataset: DatasetSummary | null;
  rawRows: Record<string, string | number>[];
  activeModel: TrainedModelResult | null;
}

export const ExplainView: React.FC<ExplainViewProps> = ({
  dataset,
  rawRows,
  activeModel
}) => {
  const [sampleIdx, setSampleIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const [explainResult, setExplainResult] = useState<ExplainabilityResult | null>(null);

  const fetchExplanation = async (idx = sampleIdx) => {
    if (!dataset || rawRows.length === 0) return;
    setLoading(true);
    try {
      const res = await explainPredictionsApi({
        rows: rawRows,
        targetCol: dataset.targetColumn,
        modelType: activeModel?.modelType || 'random_forest',
        hyperparameters: activeModel?.hyperparameters,
        sampleIndex: idx
      });
      setExplainResult(res);
    } catch (err) {
      console.error('Explainability failed', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchExplanation(sampleIdx);
  }, [activeModel?.id, sampleIdx]);

  const globalChartData = (explainResult?.globalImportances || []).slice(0, 8).map(item => ({
    feature: item.feature,
    importance: Math.round(item.importance * 1000) / 10,
    pct: item.relativePct
  }));

  const waterfallData = (explainResult?.samplePrediction?.localContributions || []).map(item => ({
    feature: item.feature,
    attribution: item.attribution,
    prob: Math.round(item.runningProbability * 1000) / 10,
    value: String(item.featureValue)
  }));

  return (
    <div className="space-y-6">
      {/* Method Header & Base Value Banner */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base font-semibold text-white flex items-center gap-2">
                <Search className="h-4 w-4 text-cyan-400" />
                Explainable AI (XAI) Attribution Suite
              </h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">
                {explainResult?.method || 'TreeSHAP Engine'}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Deconstruct model decisions into global feature importances and local Shapley attribution contributions
            </p>
          </div>

          {explainResult && (
            <div className="flex items-center gap-4 bg-slate-950/80 px-4 py-2 rounded-xl border border-slate-800 text-xs">
              <div>
                <span className="text-slate-400 block text-[11px]">Base Expected Value $E[f(x)]$</span>
                <span className="font-mono font-bold text-cyan-400">
                  {(explainResult.baseValue * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-6 w-[1px] bg-slate-800" />
              <div>
                <span className="text-slate-400 block text-[11px]">Active Architecture</span>
                <span className="font-medium text-slate-200">
                  {activeModel?.modelName || 'Random Forest'}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Global Feature Importance Chart */}
      <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-sm font-semibold text-white flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-purple-400" />
              Global Feature Importance Ranking
            </h4>
            <p className="text-xs text-slate-400">
              Aggregated influence of each feature across the full dataset distribution
            </p>
          </div>
          <span className="text-xs font-mono text-slate-400">Mean Absolute Attribution</span>
        </div>

        {globalChartData.length > 0 ? (
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={globalChartData}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 60, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                <XAxis type="number" stroke="#64748b" fontSize={11} tickLine={false} unit="%" />
                <YAxis dataKey="feature" type="category" stroke="#cbd5e1" fontSize={11} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                />
                <Bar dataKey="importance" fill="#818cf8" radius={[0, 4, 4, 0]} name="Importance Weight (%)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-48 flex items-center justify-center text-xs text-slate-500 italic">
            Computing global feature weights...
          </div>
        )}
      </div>

      {/* Local Prediction Explanation (Sample Level) */}
      <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h4 className="text-sm font-semibold text-white flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-cyan-400" />
              Local Prediction Decomposition (Waterfall Attribution)
            </h4>
            <p className="text-xs text-slate-400">
              How individual features for sample #{sampleIdx + 1} pushed probability away from base expected value
            </p>
          </div>

          {/* Sample Navigator */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Sample Row:</span>
            <button
              disabled={sampleIdx <= 0}
              onClick={() => setSampleIdx(i => Math.max(0, i - 1))}
              className="p-1.5 rounded-lg bg-slate-800 border border-slate-700 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-700 text-slate-200 cursor-pointer"
            >
              <ChevronLeft className="h-3.5 w-3.5" />
            </button>
            <span className="text-xs font-mono px-2 text-slate-200">#{sampleIdx + 1}</span>
            <button
              disabled={sampleIdx >= 25}
              onClick={() => setSampleIdx(i => Math.min(25, i + 1))}
              className="p-1.5 rounded-lg bg-slate-800 border border-slate-700 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-700 text-slate-200 cursor-pointer"
            >
              <ChevronRight className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {explainResult?.samplePrediction ? (
          <div className="space-y-5">
            {/* Outcome KPI Badges */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-[11px] text-slate-400 block">Actual Ground Truth</span>
                <span className="text-sm font-semibold font-mono text-white">
                  {explainResult.samplePrediction.actualLabel}
                </span>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-[11px] text-slate-400 block">Model Predicted Class</span>
                <span className="text-sm font-semibold font-mono text-cyan-400">
                  {explainResult.samplePrediction.predictedLabel}
                </span>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-[11px] text-slate-400 block">Predicted Probability</span>
                <span className="text-sm font-semibold font-mono text-emerald-400">
                  {(explainResult.samplePrediction.predictedProbability * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* Waterfall Attributions Chart */}
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={waterfallData} margin={{ top: 10, right: 20, left: -20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="feature" stroke="#64748b" fontSize={10} tickLine={false} interval={0} angle={-25} textAnchor="end" />
                  <YAxis stroke="#64748b" fontSize={10} tickLine={false} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                    formatter={(val: any, name: any, item: any) => [
                      `${val >= 0 ? '+' : ''}${val} (Value: ${item.payload.value})`,
                      'Shapley Attribution'
                    ]}
                  />
                  <Bar dataKey="attribution" name="Feature Contribution" radius={[4, 4, 0, 0]}>
                    {waterfallData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.attribution >= 0 ? '#10b981' : '#f43f5e'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Local Contribution Breakdown Table */}
            <div className="overflow-x-auto border border-slate-800 rounded-xl">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 font-mono border-b border-slate-800">
                  <tr>
                    <th className="py-2 px-3 font-normal">Feature</th>
                    <th className="py-2 px-3 font-normal">Sample Value</th>
                    <th className="py-2 px-3 font-normal">Attribution ($\phi_i$)</th>
                    <th className="py-2 px-3 font-normal">Direction</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {explainResult.samplePrediction.localContributions.map(c => (
                    <tr key={c.feature} className="hover:bg-slate-800/30 transition">
                      <td className="py-2 px-3 font-medium text-white">{c.feature}</td>
                      <td className="py-2 px-3 font-mono text-slate-400">{String(c.featureValue)}</td>
                      <td
                        className={`py-2 px-3 font-mono font-semibold ${
                          c.attribution >= 0 ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {c.attribution >= 0 ? `+${c.attribution}` : c.attribution}
                      </td>
                      <td className="py-2 px-3">
                        <span
                          className={`text-[10px] px-2 py-0.5 rounded font-mono ${
                            c.attribution >= 0
                              ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                              : 'bg-rose-950 text-rose-300 border border-rose-800'
                          }`}
                        >
                          {c.attribution >= 0 ? 'Increases Risk (↑)' : 'Decreases Risk (↓)'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="py-8 text-center text-xs text-slate-500 italic">
            {loading ? 'Evaluating Shapley feature attributions...' : 'Select a sample to inspect local attributions'}
          </div>
        )}
      </div>
    </div>
  );
};
