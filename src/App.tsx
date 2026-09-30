/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import { Header } from './components/Header';
import { Navigation, TabId } from './components/Navigation';
import { DashboardView } from './components/DashboardView';
import { DatasetView } from './components/DatasetView';
import { TrainingView } from './components/TrainingView';
import { OptimizationView } from './components/OptimizationView';
import { DriftView } from './components/DriftView';
import { ExplainView } from './components/ExplainView';
import { ComparisonView } from './components/ComparisonView';
import { TestPredictionView } from './components/TestPredictionView';
import { SettingsView } from './components/SettingsView';
import {
  DatasetSummary,
  TrainedModelResult,
  DriftAnalysisResult,
  ExperimentRecord
} from './lib/ml-engine/types';
import { fetchSampleDataset, parseCsvDataset, trainModelApi } from './lib/api-client';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('dashboard');
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [rawRows, setRawRows] = useState<Record<string, string | number>[]>([]);
  const [trainedModels, setTrainedModels] = useState<TrainedModelResult[]>([]);
  const [activeDrift, setActiveDrift] = useState<DriftAnalysisResult | null>(null);
  const [history, setHistory] = useState<ExperimentRecord[]>([]);

  // Initial auto-seed with credit card fraud data (1,000 transactions) & baseline training
  useEffect(() => {
    const initializeSampleData = async () => {
      try {
        const sample = await fetchSampleDataset('fraud');
        const parsed = await parseCsvDataset(sample.rawCsv, sample.summary.targetColumn);

        const pRes = Papa.parse<Record<string, any>>(sample.rawCsv, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: 'greedy'
        });

        const headers = (pRes.meta.fields || []).map(h => h.trim());
        const rows: Record<string, string | number>[] = [];

        for (const rawRow of pRes.data) {
          const row: Record<string, string | number> = {};
          for (const h of headers) {
            const val = rawRow[h];
            if (val === null || val === undefined || val === '') {
              row[h] = '';
            } else if (typeof val === 'number') {
              row[h] = val;
            } else {
              row[h] = String(val).trim();
            }
          }
          rows.push(row);
        }

        setDataset({
          ...parsed.summary,
          rowCount: rows.length
        });
        setRawRows(rows);

        // Train initial baseline Random Forest
        const initialModel = await trainModelApi({
          rows,
          targetCol: parsed.summary.targetColumn,
          modelType: 'random_forest'
        });

        setTrainedModels([initialModel]);
        setHistory([
          {
            id: initialModel.id,
            timestamp: initialModel.trainedAt,
            name: initialModel.modelName,
            datasetName: parsed.summary.name,
            modelType: initialModel.modelType,
            isOptimized: false,
            accuracy: initialModel.metrics.accuracy,
            f1Score: initialModel.metrics.f1Score,
            latencyMs: initialModel.metrics.latencyMs
          }
        ]);
      } catch (e) {
        console.error('Initialization error:', e);
      }
    };

    initializeSampleData();
  }, []);

  const handleDatasetLoaded = (summary: DatasetSummary, rows: Record<string, string | number>[]) => {
    setDataset(summary);
    setRawRows(rows);
    setTrainedModels([]);
    setActiveDrift(null);
  };

  const handleTargetChanged = (newTarget: string) => {
    if (!dataset) return;
    setDataset({
      ...dataset,
      targetColumn: newTarget
    });
    setTrainedModels([]);
    setActiveDrift(null);
  };

  const handleTaskTypeChanged = (type: 'classification' | 'regression') => {
    if (!dataset) return;
    setDataset({
      ...dataset,
      taskType: type
    });
  };

  const handleModelTrained = (newModel: TrainedModelResult) => {
    setTrainedModels(prev => [...prev.filter(m => m.id !== newModel.id), newModel]);
    setHistory(prev => [
      {
        id: newModel.id,
        timestamp: newModel.trainedAt,
        name: newModel.modelName,
        datasetName: dataset?.name || 'Dataset',
        modelType: newModel.modelType,
        isOptimized: newModel.isOptimized || false,
        accuracy: newModel.metrics.accuracy,
        f1Score: newModel.metrics.f1Score,
        latencyMs: newModel.metrics.latencyMs
      },
      ...prev
    ]);
  };

  const handleDriftAnalyzed = (res: DriftAnalysisResult) => {
    setActiveDrift(res);
  };

  const bestModel = trainedModels.length > 0
    ? [...trainedModels].sort((a, b) => b.metrics.f1Score - a.metrics.f1Score)[0]
    : null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Header */}
      <Header
        dataset={dataset}
        bestModelName={bestModel?.modelName}
        bestF1={bestModel?.metrics.f1Score}
        onNavigateToSettings={() => setActiveTab('settings')}
      />

      {/* Navigation Bar */}
      <Navigation
        activeTab={activeTab}
        onTabChange={setActiveTab}
        hasDataset={!!dataset}
        hasTrainedModel={trainedModels.length > 0}
        driftAlert={!!activeDrift && activeDrift.driftedFeaturesCount > 0}
      />

      {/* Main View Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 md:p-8">
        {activeTab === 'dashboard' && (
          <DashboardView
            dataset={dataset}
            models={trainedModels}
            activeDrift={activeDrift}
            history={history}
            onNavigate={setActiveTab}
          />
        )}

        {activeTab === 'dataset' && (
          <DatasetView
            dataset={dataset}
            rawRows={rawRows}
            onDatasetLoaded={handleDatasetLoaded}
            onTargetChanged={handleTargetChanged}
            onTaskTypeChanged={handleTaskTypeChanged}
          />
        )}

        {activeTab === 'training' && (
          <TrainingView
            dataset={dataset}
            rawRows={rawRows}
            trainedModels={trainedModels}
            onModelTrained={handleModelTrained}
          />
        )}

        {activeTab === 'optimization' && (
          <OptimizationView
            dataset={dataset}
            rawRows={rawRows}
            onOptimizedModelSaved={handleModelTrained}
          />
        )}

        {activeTab === 'drift' && (
          <DriftView
            dataset={dataset}
            rawRows={rawRows}
            activeModel={bestModel}
            onDriftAnalyzed={handleDriftAnalyzed}
          />
        )}

        {activeTab === 'explain' && (
          <ExplainView
            dataset={dataset}
            rawRows={rawRows}
            activeModel={bestModel}
          />
        )}

        {activeTab === 'comparison' && (
          <ComparisonView
            models={trainedModels}
            activeDrift={activeDrift}
          />
        )}

        {activeTab === 'test_prediction' && (
          <TestPredictionView
            dataset={dataset}
            rawRows={rawRows}
            activeModel={bestModel}
          />
        )}

        {activeTab === 'settings' && (
          <SettingsView />
        )}
      </main>

      {/* Global Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 px-6 py-4 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-400">DriftForge AI</span>
            <span>·</span>
            <span>Adaptive ML Optimization &amp; Data Drift Detection Platform</span>
          </div>
          <div className="flex items-center gap-4 text-[11px] font-mono">
            <span>Kolmogorov-Smirnov &amp; PSI Evaluators</span>
            <span>·</span>
            <span>Optuna Bayesian TPE</span>
            <span>·</span>
            <span>TreeSHAP XAI</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
