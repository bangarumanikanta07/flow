import React, { useState, useEffect } from 'react';
import { Activity, Cpu, ShieldCheck, Zap, Database, Server } from 'lucide-react';
import { DatasetSummary } from '../lib/ml-engine/types';
import { getEngineSettings, checkFastApiHealth } from '../lib/api-client';

interface HeaderProps {
  dataset: DatasetSummary | null;
  bestModelName?: string;
  bestF1?: number;
  onNavigateToSettings: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  dataset,
  bestModelName,
  bestF1,
  onNavigateToSettings
}) => {
  const [engineStatus, setEngineStatus] = useState<'integrated' | 'fastapi_connected' | 'fastapi_offline'>('integrated');
  const [fastApiPing, setFastApiPing] = useState<number | null>(null);

  useEffect(() => {
    const checkStatus = async () => {
      const settings = getEngineSettings();
      if (settings.mode === 'fastapi') {
        const start = performance.now();
        const res = await checkFastApiHealth(settings.fastApiUrl);
        const elapsed = Math.round(performance.now() - start);
        if (res.healthy) {
          setEngineStatus('fastapi_connected');
          setFastApiPing(elapsed);
        } else {
          setEngineStatus('fastapi_offline');
          setFastApiPing(null);
        }
      } else {
        setEngineStatus('integrated');
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md sticky top-0 z-40 px-6 py-3.5">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-600 to-purple-600 p-[1px] shadow-lg shadow-cyan-500/10">
            <div className="h-full w-full bg-slate-950 rounded-[11px] flex items-center justify-center">
              <Activity className="h-5 w-5 text-cyan-400 animate-pulse" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
                DriftForge AI
              </span>
              <span className="text-[10px] font-mono tracking-widest uppercase px-1.5 py-0.5 rounded bg-cyan-950/60 border border-cyan-800/50 text-cyan-300">
                v1.4 MLOps
              </span>
            </div>
            <p className="text-xs text-slate-400">Adaptive ML Optimization & Data Drift Platform</p>
          </div>
        </div>

        {/* Global Operational State */}
        <div className="flex items-center gap-4 text-xs">
          {/* Active Dataset */}
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800">
            <Database className="h-3.5 w-3.5 text-purple-400" />
            <span className="text-slate-400">Dataset:</span>
            <span className="font-medium text-slate-200 truncate max-w-[160px]">
              {dataset ? dataset.name : 'No dataset loaded'}
            </span>
            {dataset && (
              <span className="font-mono text-[11px] text-slate-500">
                ({dataset.rowCount} rows)
              </span>
            )}
          </div>

          {/* Best Model Banner */}
          {bestModelName && (
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800">
              <Zap className="h-3.5 w-3.5 text-cyan-400" />
              <span className="text-slate-400">Leader:</span>
              <span className="font-medium text-slate-200">{bestModelName}</span>
              {bestF1 !== undefined && (
                <span className="font-mono font-semibold text-cyan-400">
                  F1: {(bestF1 * 100).toFixed(1)}%
                </span>
              )}
            </div>
          )}

          {/* Engine Selector / Status Badge */}
          <button
            onClick={onNavigateToSettings}
            title="Click to configure ML Backend Engine in Settings"
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800 hover:border-slate-700 transition cursor-pointer"
          >
            <Server className="h-3.5 w-3.5 text-indigo-400" />
            <span className="text-slate-400">Engine:</span>
            {engineStatus === 'integrated' && (
              <span className="flex items-center gap-1.5 font-medium text-emerald-400">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-ping" />
                Integrated Core
              </span>
            )}
            {engineStatus === 'fastapi_connected' && (
              <span className="flex items-center gap-1.5 font-medium text-cyan-400">
                <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
                FastAPI Py ({fastApiPing}ms)
              </span>
            )}
            {engineStatus === 'fastapi_offline' && (
              <span className="flex items-center gap-1.5 font-medium text-amber-400">
                <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
                FastAPI (Fallback Core)
              </span>
            )}
          </button>
        </div>
      </div>
    </header>
  );
};
