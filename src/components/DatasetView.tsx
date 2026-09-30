import React, { useState, useRef } from 'react';
import Papa from 'papaparse';
import { DatasetSummary } from '../lib/ml-engine/types';
import {
  Upload,
  Database,
  CheckCircle2,
  AlertCircle,
  FileText,
  Search,
  ChevronLeft,
  ChevronRight,
  Filter,
  Layers,
  Sparkles
} from 'lucide-react';
import { parseCsvDataset, fetchSampleDataset } from '../lib/api-client';

interface DatasetViewProps {
  dataset: DatasetSummary | null;
  rawRows: Record<string, string | number>[];
  onDatasetLoaded: (summary: DatasetSummary, rows: Record<string, string | number>[]) => void;
  onTargetChanged: (newTarget: string) => void;
  onTaskTypeChanged: (type: 'classification' | 'regression') => void;
}

export const DatasetView: React.FC<DatasetViewProps> = ({
  dataset,
  rawRows,
  onDatasetLoaded,
  onTargetChanged,
  onTaskTypeChanged
}) => {
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const pageSize = 10;

  // Handle preset sample loading
  const handleLoadSample = async (type: 'churn' | 'credit' | 'housing' | 'fraud') => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await fetchSampleDataset(type);
      const parsed = await parseCsvDataset(res.rawCsv, res.summary.targetColumn);

      // Parse with PapaParse
      const pRes = Papa.parse<Record<string, any>>(res.rawCsv, {
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

      onDatasetLoaded(parsed.summary, rows);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to load sample dataset');
    } finally {
      setLoading(false);
    }
  };

  // Handle user uploaded CSV
  const handleFileUpload = async (file: File) => {
    if (!file) return;
    if (!file.name.endsWith('.csv') && file.type !== 'text/csv') {
      setErrorMsg('Please upload a valid .csv file');
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setErrorMsg('File size exceeds the 15MB safety limit');
      return;
    }

    setLoading(true);
    setErrorMsg(null);
    try {
      const text = await file.text();
      const parsed = await parseCsvDataset(text);

      const pRes = Papa.parse<Record<string, any>>(text, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: 'greedy'
      });

      if (pRes.errors && pRes.errors.length > 0 && pRes.data.length === 0) {
        throw new Error(pRes.errors[0].message || 'Invalid CSV syntax');
      }

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

      onDatasetLoaded(
        {
          ...parsed.summary,
          name: file.name.replace('.csv', '')
        },
        rows
      );
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to process CSV file');
    } finally {
      setLoading(false);
    }
  };

  // Table pagination & search
  const filteredRows = rawRows.filter(row => {
    if (!searchTerm) return true;
    return Object.values(row).some(v =>
      String(v).toLowerCase().includes(searchTerm.toLowerCase())
    );
  });

  const totalPages = Math.ceil(filteredRows.length / pageSize) || 1;
  const currentRows = filteredRows.slice(page * pageSize, (page + 1) * pageSize);

  const numericalColumns = dataset?.columns.filter(c => c.type === 'numerical') || [];
  const categoricalColumns = dataset?.columns.filter(c => c.type === 'categorical') || [];

  return (
    <div className="space-y-6">
      {/* Upload and Sample Header Card */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <Database className="h-4 w-4 text-cyan-400" />
              Dataset Ingestion & Schema Inspector
            </h3>
            <p className="text-xs text-slate-400">
              Upload custom CSV files or load pre-built enterprise benchmark datasets
            </p>
          </div>

          {/* Preset Buttons */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-400 mr-1">Load Presets:</span>
            <button
              onClick={() => handleLoadSample('churn')}
              disabled={loading}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-medium text-slate-200 transition cursor-pointer flex items-center gap-1.5"
            >
              <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
              <span>Customer Churn (500)</span>
            </button>
            <button
              onClick={() => handleLoadSample('credit')}
              disabled={loading}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-medium text-slate-200 transition cursor-pointer flex items-center gap-1.5"
            >
              <Sparkles className="h-3.5 w-3.5 text-purple-400" />
              <span>Credit Risk (450)</span>
            </button>
            <button
              onClick={() => handleLoadSample('housing')}
              disabled={loading}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-medium text-slate-200 transition cursor-pointer flex items-center gap-1.5"
            >
              <Sparkles className="h-3.5 w-3.5 text-emerald-400" />
              <span>Housing Valuation (400)</span>
            </button>
            <button
              onClick={() => handleLoadSample('fraud')}
              disabled={loading}
              className="px-3 py-1.5 rounded-lg bg-amber-950/60 hover:bg-amber-900/60 border border-amber-700/60 text-xs font-medium text-amber-200 transition cursor-pointer flex items-center gap-1.5"
            >
              <Sparkles className="h-3.5 w-3.5 text-amber-400" />
              <span>Fraud Transactions (1,000)</span>
            </button>
          </div>
        </div>

        {/* Drag & Drop Upload Zone */}
        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={e => e.preventDefault()}
          onDrop={e => {
            e.preventDefault();
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
              handleFileUpload(e.dataTransfer.files[0]);
            }
          }}
          className="border-2 border-dashed border-slate-700 hover:border-cyan-500/60 bg-slate-950/40 hover:bg-slate-900/40 rounded-xl p-8 text-center cursor-pointer transition flex flex-col items-center justify-center space-y-3"
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={e => {
              if (e.target.files && e.target.files[0]) {
                handleFileUpload(e.target.files[0]);
              }
            }}
            accept=".csv,text/csv"
            className="hidden"
          />
          <div className="h-12 w-12 rounded-xl bg-slate-800/80 flex items-center justify-center text-cyan-400">
            <Upload className="h-6 w-6" />
          </div>
          <div>
            <div className="text-sm font-medium text-slate-200">
              {loading ? 'Processing dataset...' : 'Click to browse or drop your CSV dataset here'}
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Supports standard CSV format with comma delimiters. Max file size: 15MB.
            </p>
          </div>
        </div>

        {errorMsg && (
          <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs flex items-center gap-2">
            <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
            <span>{errorMsg}</span>
          </div>
        )}
      </div>

      {/* Dataset Summary & Configuration (Target & Task Type) */}
      {dataset && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Metadata & Validation Card */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
            <h4 className="text-sm font-semibold text-white flex items-center gap-2">
              <Layers className="h-4 w-4 text-purple-400" />
              Dataset Health & Metadata
            </h4>

            <div className="space-y-3 text-xs">
              <div className="flex items-center justify-between py-1.5 border-b border-slate-800/60">
                <span className="text-slate-400">Dataset Name</span>
                <span className="font-medium text-slate-200 truncate max-w-[180px]">{dataset.name}</span>
              </div>
              <div className="flex items-center justify-between py-1.5 border-b border-slate-800/60">
                <span className="text-slate-400">Total Records</span>
                <span className="font-mono text-cyan-400">{dataset.rowCount} rows</span>
              </div>
              <div className="flex items-center justify-between py-1.5 border-b border-slate-800/60">
                <span className="text-slate-400">Total Features</span>
                <span className="font-mono text-slate-200">{dataset.columnCount} columns</span>
              </div>
              <div className="flex items-center justify-between py-1.5 border-b border-slate-800/60">
                <span className="text-slate-400">Numerical Columns</span>
                <span className="font-mono text-slate-200">{numericalColumns.length}</span>
              </div>
              <div className="flex items-center justify-between py-1.5 border-b border-slate-800/60">
                <span className="text-slate-400">Categorical Columns</span>
                <span className="font-mono text-slate-200">{categoricalColumns.length}</span>
              </div>
              <div className="flex items-center justify-between py-1.5">
                <span className="text-slate-400">Missing Values</span>
                <span className={`font-mono ${dataset.missingValuesTotal === 0 ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {dataset.missingValuesTotal === 0 ? '0 (Clean)' : `${dataset.missingValuesTotal} nulls`}
                </span>
              </div>
            </div>
          </div>

          {/* Model Objective Configuration (Target Column & Task Type) */}
          <div className="lg:col-span-2 p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
            <h4 className="text-sm font-semibold text-white flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-cyan-400" />
              Target Objective & Task Selection
            </h4>
            <p className="text-xs text-slate-400">
              Select the dependent target column to predict and define the machine learning task formulation.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
              {/* Target Column Selector */}
              <div className="space-y-2">
                <label className="text-xs font-medium text-slate-300">Target Column ($y$)</label>
                <select
                  value={dataset.targetColumn}
                  onChange={e => onTargetChanged(e.target.value)}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-700 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 cursor-pointer"
                >
                  {dataset.columns.map(col => (
                    <option key={col.name} value={col.name}>
                      {col.name} ({col.type})
                    </option>
                  ))}
                </select>
                <p className="text-[11px] text-slate-500">
                  Label or outcome variable being predicted.
                </p>
              </div>

              {/* Task Type Selector */}
              <div className="space-y-2">
                <label className="text-xs font-medium text-slate-300">Machine Learning Task</label>
                <div className="flex gap-2">
                  <button
                    onClick={() => onTaskTypeChanged('classification')}
                    className={`flex-1 py-2.5 px-3 rounded-xl text-xs font-medium transition cursor-pointer border ${
                      dataset.taskType === 'classification'
                        ? 'bg-cyan-950/60 border-cyan-500 text-cyan-300'
                        : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700'
                    }`}
                  >
                    Classification
                  </button>
                  <button
                    onClick={() => onTaskTypeChanged('regression')}
                    className={`flex-1 py-2.5 px-3 rounded-xl text-xs font-medium transition cursor-pointer border ${
                      dataset.taskType === 'regression'
                        ? 'bg-purple-950/60 border-purple-500 text-purple-300'
                        : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700'
                    }`}
                  >
                    Regression
                  </button>
                </div>
                <p className="text-[11px] text-slate-500">
                  {dataset.taskType === 'classification' ? 'Discrete category or binary probability outcome' : 'Continuous real-valued prediction'}
                </p>
              </div>
            </div>

            {/* Column Schema Badges */}
            <div className="pt-3 border-t border-slate-800/80">
              <span className="text-xs text-slate-400 block mb-2">Feature Schema Summary:</span>
              <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto pr-1">
                {dataset.columns.map(col => (
                  <span
                    key={col.name}
                    className={`text-[11px] px-2 py-0.5 rounded-md font-mono ${
                      col.name === dataset.targetColumn
                        ? 'bg-cyan-900/60 text-cyan-300 border border-cyan-600/60 font-semibold'
                        : col.type === 'numerical'
                          ? 'bg-slate-800/80 text-slate-300 border border-slate-700/50'
                          : 'bg-indigo-950/40 text-indigo-300 border border-indigo-800/40'
                    }`}
                  >
                    {col.name} {col.name === dataset.targetColumn ? '★' : `(${col.type[0]})`}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Dataset Data Table Preview */}
      {dataset && rawRows.length > 0 && (
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h4 className="text-sm font-semibold text-white">Tabular Data Preview</h4>
              <p className="text-xs text-slate-400">
                Displaying {filteredRows.length} matching rows of {rawRows.length} total samples
              </p>
            </div>

            {/* Search Input */}
            <div className="relative">
              <Search className="h-3.5 w-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
              <input
                type="text"
                placeholder="Search across columns..."
                value={searchTerm}
                onChange={e => {
                  setSearchTerm(e.target.value);
                  setPage(0);
                }}
                className="pl-8 pr-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 w-56"
              />
            </div>
          </div>

          <div className="overflow-x-auto border border-slate-800 rounded-xl">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950 text-slate-400 font-mono border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3.5 font-normal text-slate-500">#</th>
                  {dataset.columns.map(col => (
                    <th
                      key={col.name}
                      className={`py-2.5 px-3.5 font-normal ${
                        col.name === dataset.targetColumn ? 'text-cyan-400 font-semibold' : ''
                      }`}
                    >
                      {col.name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {currentRows.map((row, i) => (
                  <tr key={i} className="hover:bg-slate-800/30 transition">
                    <td className="py-2 px-3.5 font-mono text-slate-500">{page * pageSize + i + 1}</td>
                    {dataset.columns.map(col => (
                      <td
                        key={col.name}
                        className={`py-2 px-3.5 ${
                          col.name === dataset.targetColumn ? 'font-mono font-semibold text-cyan-300 bg-cyan-950/20' : ''
                        }`}
                      >
                        {String(row[col.name] ?? '-')}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination Controls */}
          <div className="flex items-center justify-between text-xs text-slate-400 pt-1">
            <span>
              Page {page + 1} of {totalPages}
            </span>
            <div className="flex items-center gap-1">
              <button
                disabled={page === 0}
                onClick={() => setPage(p => Math.max(0, p - 1))}
                className="p-1.5 rounded-lg bg-slate-800 border border-slate-700 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-700 transition cursor-pointer text-slate-200"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <button
                disabled={page >= totalPages - 1}
                onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                className="p-1.5 rounded-lg bg-slate-800 border border-slate-700 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-700 transition cursor-pointer text-slate-200"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
