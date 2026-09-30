import React, { useState, useRef } from 'react';
import Papa from 'papaparse';
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  Upload,
  Download,
  Play,
  RotateCcw,
  Activity,
  Layers,
  CheckCircle2,
  FileText,
  Search,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  Info
} from 'lucide-react';
import {
  DatasetSummary,
  TrainedModelResult,
  PredictionResultItem,
  BatchPredictionResponse
} from '../lib/ml-engine/types';
import { predictTransactionsApi } from '../lib/api-client';

interface TestPredictionViewProps {
  dataset: DatasetSummary | null;
  rawRows: Record<string, string | number>[];
  activeModel: TrainedModelResult | null;
}

export const TestPredictionView: React.FC<TestPredictionViewProps> = ({
  dataset,
  rawRows,
  activeModel
}) => {
  // Features available for input (STRICTLY excluding target column 'is_fraud' and ID columns)
  const targetCol = dataset?.targetColumn || 'is_fraud';

  // Discover actual feature columns from the current dataset schema or rawRows
  const availableFeatures = React.useMemo(() => {
    if (dataset?.features && dataset.features.length > 0) {
      return dataset.features.filter(
        f => f !== targetCol && f !== 'is_fraud' && f !== 'id' && f !== 'transaction_id' && f !== 'customer_id' && f !== 'Unnamed: 0'
      );
    }
    if (rawRows.length > 0) {
      return Object.keys(rawRows[0]).filter(
        f => f !== targetCol && f !== 'is_fraud' && f !== 'id' && f !== 'transaction_id' && f !== 'customer_id' && f !== 'Unnamed: 0'
      );
    }
    return [
      'transaction_amount',
      'transaction_hour',
      'transactions_last_24h',
      'account_age_days',
      'distance_from_home_km',
      'device_risk_score',
      'failed_login_attempts',
      'merchant_category'
    ];
  }, [dataset, rawRows, targetCol]);

  // Discover distinct categories for categorical columns
  const categoricalOptionsMap = React.useMemo(() => {
    const map: Record<string, string[]> = {};
    for (const feat of availableFeatures) {
      const colInfo = dataset?.columns.find(c => c.name === feat);
      if (colInfo?.type === 'categorical') {
        const uniqueVals = Array.from(new Set(rawRows.map(r => String(r[feat])).filter(v => v !== '')));
        map[feat] = uniqueVals.length > 0 ? uniqueVals : (colInfo.sampleValues?.map(String) || ['grocery', 'electronics', 'travel', 'dining', 'online_retail', 'gambling']);
      }
    }
    return map;
  }, [dataset, rawRows, availableFeatures]);

  // Build clean default values dynamically matching current dataset schema
  const buildInitialForm = React.useCallback(() => {
    const initial: Record<string, string | number> = {};
    availableFeatures.forEach(feat => {
      const colInfo = dataset?.columns.find(c => c.name === feat);
      if (colInfo?.type === 'categorical') {
        const opts = categoricalOptionsMap[feat] || colInfo.sampleValues?.map(String) || ['grocery'];
        initial[feat] = opts[0] || 'grocery';
      } else {
        // Numerical mean or fallback
        if (colInfo?.mean !== undefined) {
          initial[feat] = colInfo.mean;
        } else {
          // Defaults for known financial features
          if (feat === 'transaction_amount') initial[feat] = 120.5;
          else if (feat === 'transaction_hour') initial[feat] = 14;
          else if (feat === 'transactions_last_24h') initial[feat] = 3;
          else if (feat === 'account_age_days') initial[feat] = 365;
          else if (feat === 'distance_from_home_km') initial[feat] = 12.4;
          else if (feat === 'device_risk_score') initial[feat] = 0.15;
          else if (feat === 'failed_login_attempts') initial[feat] = 0;
          else initial[feat] = 10;
        }
      }
    });
    return initial;
  }, [availableFeatures, dataset, categoricalOptionsMap]);

  // Manual transaction form state
  const [manualForm, setManualForm] = useState<Record<string, string | number>>(buildInitialForm);

  // Re-sync manual form if dataset changes
  React.useEffect(() => {
    setManualForm(buildInitialForm());
    setManualResult(null);
  }, [buildInitialForm]);

  const [manualResult, setManualResult] = useState<PredictionResultItem | null>(null);
  const [isPredictingSingle, setIsPredictingSingle] = useState(false);
  const [singleError, setSingleError] = useState<string | null>(null);

  // Batch CSV upload prediction state
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchRows, setBatchRows] = useState<Record<string, string | number>[]>([]);
  const [batchResponse, setBatchResponse] = useState<BatchPredictionResponse | null>(null);
  const [isPredictingBatch, setIsPredictingBatch] = useState(false);
  const [batchError, setBatchError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Table pagination & filtering
  const [searchTerm, setSearchTerm] = useState('');
  const [filterClass, setFilterClass] = useState<'all' | 'Fraud' | 'Legitimate'>('all');
  const [page, setPage] = useState(0);
  const pageSize = 10;

  // Handle single manual prediction
  const handleSinglePredict = async () => {
    if (!rawRows || rawRows.length === 0) {
      setSingleError('Please load or train a dataset first.');
      return;
    }
    setIsPredictingSingle(true);
    setSingleError(null);

    try {
      // Ensure target column is NOT in the test sample
      const sanitizedTestRow: Record<string, string | number> = {};
      availableFeatures.forEach(k => {
        const val = manualForm[k];
        const isNum = !categoricalOptionsMap[k] && !isNaN(Number(val)) && val !== '';
        sanitizedTestRow[k] = isNum ? Number(val) : val;
      });

      const response = await predictTransactionsApi({
        trainingRows: rawRows,
        testRows: [sanitizedTestRow],
        targetCol: targetCol,
        modelType: activeModel?.modelType || 'random_forest',
        hyperparameters: activeModel?.hyperparameters
      });

      if (response && response.predictions && response.predictions.length > 0) {
        setManualResult(response.predictions[0]);
      } else {
        setSingleError('Model inference returned an empty prediction vector.');
      }
    } catch (err: any) {
      setSingleError(err.message || 'Single transaction prediction failed.');
    } finally {
      setIsPredictingSingle(false);
    }
  };

  // Populate sample values for manual test
  const handleFillSample = (type: 'suspicious' | 'legitimate') => {
    setManualForm(prev => {
      const updated = { ...prev };
      if (type === 'suspicious') {
        if ('transaction_amount' in updated) updated['transaction_amount'] = 980.5;
        if ('amount' in updated) updated['amount'] = 980.5;
        if ('transaction_hour' in updated) updated['transaction_hour'] = 3;
        if ('hour_of_day' in updated) updated['hour_of_day'] = 3;
        if ('transactions_last_24h' in updated) updated['transactions_last_24h'] = 14;
        if ('account_age_days' in updated) updated['account_age_days'] = 12;
        if ('distance_from_home_km' in updated) updated['distance_from_home_km'] = 185.0;
        if ('distance_km' in updated) updated['distance_km'] = 185.0;
        if ('device_risk_score' in updated) updated['device_risk_score'] = 0.94;
        if ('failed_login_attempts' in updated) updated['failed_login_attempts'] = 4;
        if ('merchant_category' in updated) updated['merchant_category'] = 'gambling';
      } else {
        if ('transaction_amount' in updated) updated['transaction_amount'] = 34.5;
        if ('amount' in updated) updated['amount'] = 34.5;
        if ('transaction_hour' in updated) updated['transaction_hour'] = 14;
        if ('hour_of_day' in updated) updated['hour_of_day'] = 14;
        if ('transactions_last_24h' in updated) updated['transactions_last_24h'] = 2;
        if ('account_age_days' in updated) updated['account_age_days'] = 740;
        if ('distance_from_home_km' in updated) updated['distance_from_home_km'] = 4.2;
        if ('distance_km' in updated) updated['distance_km'] = 4.2;
        if ('device_risk_score' in updated) updated['device_risk_score'] = 0.08;
        if ('failed_login_attempts' in updated) updated['failed_login_attempts'] = 0;
        if ('merchant_category' in updated) updated['merchant_category'] = 'grocery';
      }
      return updated;
    });
  };

  // Handle batch CSV upload with strict column and type validation
  const handleBatchFileUpload = async (file: File) => {
    if (!file) return;
    if (!file.name.endsWith('.csv') && file.type !== 'text/csv') {
      setBatchError('Please upload a valid .csv file.');
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setBatchError('File exceeds 15MB limit.');
      return;
    }

    setBatchFile(file);
    setBatchError(null);
    setBatchResponse(null);

    try {
      const text = await file.text();
      const parsed = Papa.parse<Record<string, any>>(text, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: 'greedy'
      });

      if (parsed.errors && parsed.errors.length > 0 && parsed.data.length === 0) {
        throw new Error(parsed.errors[0].message || 'CSV syntax error');
      }

      const rows: Record<string, string | number>[] = [];
      const uploadedHeaders = (parsed.meta.fields || []).map(h => h.trim());

      // Validate that all expected features are present (excluding target column)
      const missing = availableFeatures.filter(f => !uploadedHeaders.includes(f));
      if (missing.length > 0) {
        throw new Error(
          `Incompatible CSV columns! Missing required feature(s): [${missing.join(', ')}]. Uploaded columns: [${uploadedHeaders.join(', ')}]. Expected columns: [${availableFeatures.join(', ')}].`
        );
      }

      for (const rawRow of parsed.data) {
        const cleanRow: Record<string, string | number> = {};
        for (const h of uploadedHeaders) {
          const val = rawRow[h];
          if (val === null || val === undefined || val === '') {
            cleanRow[h] = '';
          } else if (typeof val === 'number') {
            cleanRow[h] = val;
          } else {
            cleanRow[h] = String(val).trim();
          }
        }
        rows.push(cleanRow);
      }

      if (rows.length === 0) {
        throw new Error('The uploaded CSV contains 0 data rows.');
      }

      // Check numeric column types
      for (const feat of availableFeatures) {
        const isCategorical = !!categoricalOptionsMap[feat];
        if (!isCategorical) {
          const allInvalid = rows.every(r => {
            const v = r[feat];
            return v === '' || isNaN(Number(v));
          });
          if (allInvalid) {
            throw new Error(`Incompatible data types! Column '${feat}' expects numeric values, but non-numeric data was found.`);
          }
        }
      }

      setBatchRows(rows);
    } catch (err: any) {
      setBatchError(err.message || 'Failed to read uploaded batch CSV');
    }
  };

  // Run predictions on all rows of the uploaded CSV
  const handleRunBatchPrediction = async () => {
    if (batchRows.length === 0) return;
    setIsPredictingBatch(true);
    setBatchError(null);

    try {
      const response = await predictTransactionsApi({
        trainingRows: rawRows,
        testRows: batchRows,
        targetCol: targetCol,
        modelType: activeModel?.modelType || 'random_forest',
        hyperparameters: activeModel?.hyperparameters
      });

      setBatchResponse(response);
      setPage(0);
    } catch (err: any) {
      setBatchError(err.message || 'Failed to execute batch model predictions.');
    } finally {
      setIsPredictingBatch(false);
    }
  };

  // Download prediction results as CSV file
  const handleDownloadResultsCsv = () => {
    if (!batchResponse || batchResponse.predictions.length === 0) return;

    // Build CSV rows
    const dataForExport = batchResponse.predictions.map(item => {
      const row: Record<string, any> = {
        row_id: item.rowIndex,
        ...item.inputFeatures,
        PREDICTED_CLASS: item.predictedClass,
        FRAUD_PROBABILITY: item.fraudProbability,
        CONFIDENCE_PERCENT: `${item.confidenceScore}%`,
        VERIFIED_OUTCOME: item.verifiedOutcome,
        STATUS: item.isVerifiedOutcome ? 'VERIFIED' : 'UNVERIFIED_MODEL_PREDICTION'
      };
      return row;
    });

    const csvString = Papa.unparse(dataForExport);
    const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute(
      'download',
      `DriftForge_Predictions_${batchFile?.name.replace('.csv', '') || 'transactions'}_${Date.now()}.csv`
    );
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Table filtering
  const allPredictions = batchResponse?.predictions || [];
  const filteredPredictions = allPredictions.filter(item => {
    if (filterClass !== 'all' && item.predictedClass !== filterClass) {
      return false;
    }
    if (!searchTerm) return true;
    const matchesSearch =
      item.predictedClass.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.verifiedOutcome.toLowerCase().includes(searchTerm.toLowerCase()) ||
      Object.values(item.inputFeatures).some(v => String(v).toLowerCase().includes(searchTerm.toLowerCase()));
    return matchesSearch;
  });

  const totalPages = Math.ceil(filteredPredictions.length / pageSize) || 1;
  const paginatedPredictions = filteredPredictions.slice(page * pageSize, (page + 1) * pageSize);

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <ShieldAlert className="h-5 w-5 text-amber-400" />
              Test Prediction &amp; Unseen Fraud Detection
            </h3>
            <p className="text-xs text-slate-400">
              Run real inference on unseen transaction records using the currently trained{' '}
              <span className="text-cyan-400 font-semibold font-mono">
                {activeModel?.modelName || 'Random Forest Classifier'}
              </span>
              . Features exclude the target column{' '}
              <code className="text-amber-400 font-mono bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800">
                {targetCol}
              </code>
              .
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs px-2.5 py-1 rounded-lg bg-amber-950/60 border border-amber-800/80 text-amber-300 font-mono flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-amber-400 animate-pulse"></span>
              Live Model Serving
            </span>
          </div>
        </div>

        {/* Verification Distinction Notice */}
        <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 flex items-start gap-3">
          <Info className="h-4 w-4 text-cyan-400 shrink-0 mt-0.5" />
          <div className="text-xs text-slate-300 leading-relaxed">
            <span className="font-semibold text-white">Auditing &amp; Integrity Policy:</span> Predictions represent
            model inference outputs based on learned decision boundaries and feature vectors. They are explicitly
            tagged as <span className="text-amber-400 font-medium font-mono">UNVERIFIED PREDICTIONS</span> to distinguish
            them from historically verified ground-truth chargebacks or confirmed audits.
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* LEFT COLUMN: Manual Single Transaction Prediction */}
        <div className="lg:col-span-5 space-y-6">
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                  <Play className="h-4 w-4 text-cyan-400" />
                  Manual Transaction Input
                </h4>
                <p className="text-[11px] text-slate-400">Enter transaction parameters to evaluate live fraud probability</p>
              </div>

              {/* Quick Fill Buttons */}
              <div className="flex items-center gap-1.5">
                <button
                  onClick={() => handleFillSample('suspicious')}
                  className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[10px] font-medium text-amber-300 border border-slate-700 transition cursor-pointer"
                  title="Fill with suspicious/high-risk transaction pattern"
                >
                  High Risk
                </button>
                <button
                  onClick={() => handleFillSample('legitimate')}
                  className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[10px] font-medium text-emerald-300 border border-slate-700 transition cursor-pointer"
                  title="Fill with normal/low-risk transaction pattern"
                >
                  Low Risk
                </button>
              </div>
            </div>

            {/* Feature Inputs */}
            <div className="space-y-3 max-h-[380px] overflow-y-auto pr-1">
              {availableFeatures.length === 0 ? (
                <div className="text-xs text-slate-400 py-4 text-center">
                  No input features available. Please load a dataset first.
                </div>
              ) : (
                availableFeatures.map(feat => {
                  const val = manualForm[feat] !== undefined ? manualForm[feat] : '';
                  const categoricalOptions = categoricalOptionsMap[feat];
                  const isCategorical = !!categoricalOptions && categoricalOptions.length > 0;

                  return (
                    <div key={feat} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <label className="text-slate-300 font-mono text-[11px]">
                          {feat}
                        </label>
                        <span className="text-[10px] text-slate-500 font-mono">
                          {isCategorical ? 'categorical' : 'numerical'}
                        </span>
                      </div>

                      {isCategorical ? (
                        <select
                          value={String(val)}
                          onChange={e => {
                            setManualForm(prev => ({
                              ...prev,
                              [feat]: e.target.value
                            }));
                          }}
                          className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 font-mono focus:outline-none focus:border-cyan-500 transition"
                        >
                          {categoricalOptions.map(opt => (
                            <option key={opt} value={opt}>
                              {opt}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <input
                          type="number"
                          step="any"
                          value={val}
                          onChange={e => {
                            const rawVal = e.target.value;
                            setManualForm(prev => ({
                              ...prev,
                              [feat]: rawVal === '' ? '' : isNaN(Number(rawVal)) ? rawVal : Number(rawVal)
                            }));
                          }}
                          className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 font-mono focus:outline-none focus:border-cyan-500 transition"
                        />
                      )}
                    </div>
                  );
                })
              )}
            </div>

            {singleError && (
              <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 shrink-0 text-rose-400" />
                <span>{singleError}</span>
              </div>
            )}

            <button
              onClick={handleSinglePredict}
              disabled={isPredictingSingle || availableFeatures.length === 0}
              className="w-full py-2.5 rounded-xl bg-gradient-to-r from-amber-600 to-amber-500 hover:from-amber-500 hover:to-amber-400 text-white font-medium text-xs shadow-lg shadow-amber-950/40 flex items-center justify-center gap-2 transition cursor-pointer disabled:opacity-50"
            >
              {isPredictingSingle ? (
                <>
                  <Activity className="h-4 w-4 animate-spin" />
                  <span>Evaluating Decision Trees...</span>
                </>
              ) : (
                <>
                  <ShieldAlert className="h-4 w-4" />
                  <span>Predict Transaction Fraud</span>
                </>
              )}
            </button>

            {/* Prediction Result Display */}
            {manualResult && (
              <div
                className={`p-4 rounded-xl border transition-all ${
                  manualResult.predictedClass === 'Fraud'
                    ? 'bg-rose-950/40 border-rose-500/70 text-rose-200'
                    : 'bg-emerald-950/40 border-emerald-500/70 text-emerald-200'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold uppercase tracking-wider flex items-center gap-1.5">
                    {manualResult.predictedClass === 'Fraud' ? (
                      <AlertTriangle className="h-4 w-4 text-rose-400" />
                    ) : (
                      <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                    )}
                    Predicted Outcome
                  </span>
                  <span
                    className={`text-[10px] font-mono px-2 py-0.5 rounded font-semibold border ${
                      manualResult.predictedClass === 'Fraud'
                        ? 'bg-rose-900/80 text-rose-200 border-rose-700'
                        : 'bg-emerald-900/80 text-emerald-200 border-emerald-700'
                    }`}
                  >
                    UNVERIFIED PREDICTION
                  </span>
                </div>

                <div className="text-2xl font-bold font-mono tracking-tight mb-3">
                  {manualResult.predictedClass === 'Fraud' ? '🚨 FRAUD DETECTED' : '✅ LEGITIMATE TRANSACTION'}
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs pt-1 border-t border-slate-800/60">
                  <div>
                    <div className="text-[10px] text-slate-400">Fraud Probability</div>
                    <div className="font-mono text-sm font-semibold text-white">
                      {(manualResult.fraudProbability * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-400">Model Confidence</div>
                    <div className="font-mono text-sm font-semibold text-white">
                      {manualResult.confidenceScore.toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Probability Bar */}
                <div className="mt-3 space-y-1">
                  <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                    <span>Legitimate (0.0)</span>
                    <span>Fraud Threshold (0.5)</span>
                    <span>High Risk (1.0)</span>
                  </div>
                  <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                    <div
                      className={`h-full transition-all duration-500 ${
                        manualResult.fraudProbability >= 0.5 ? 'bg-rose-500' : 'bg-emerald-500'
                      }`}
                      style={{ width: `${Math.max(3, manualResult.fraudProbability * 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* RIGHT COLUMN: Batch Unseen CSV Upload & Predictions */}
        <div className="lg:col-span-7 space-y-6">
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                  <Upload className="h-4 w-4 text-purple-400" />
                  Batch Unseen CSV Inference
                </h4>
                <p className="text-[11px] text-slate-400">
                  Upload an unseen transactions dataset to classify all records simultaneously
                </p>
              </div>

              <div className="flex items-center gap-2 flex-wrap">
                <button
                  onClick={() => {
                    // Generate 50 realistic unseen test transactions matching model schema
                    const cats = ['grocery', 'electronics', 'travel', 'dining', 'online_retail', 'gambling'];
                    const mockTestRows: Record<string, string | number>[] = [];
                    for (let i = 1; i <= 50; i++) {
                      const isHigh = i % 4 === 0;
                      mockTestRows.push({
                        transaction_amount: isHigh ? Math.round((600 + Math.random() * 800) * 100) / 100 : Math.round((15 + Math.random() * 120) * 100) / 100,
                        transaction_hour: isHigh ? (Math.random() > 0.5 ? 2 : 4) : Math.floor(8 + Math.random() * 12),
                        transactions_last_24h: isHigh ? Math.floor(7 + Math.random() * 9) : Math.floor(1 + Math.random() * 4),
                        account_age_days: isHigh ? Math.floor(5 + Math.random() * 30) : Math.floor(180 + Math.random() * 600),
                        distance_from_home_km: isHigh ? Math.round((120 + Math.random() * 250) * 10) / 10 : Math.round((1 + Math.random() * 20) * 10) / 10,
                        device_risk_score: isHigh ? Math.round((0.75 + Math.random() * 0.24) * 100) / 100 : Math.round((0.02 + Math.random() * 0.25) * 100) / 100,
                        failed_login_attempts: isHigh ? Math.floor(2 + Math.random() * 4) : 0,
                        merchant_category: isHigh ? (Math.random() > 0.5 ? 'gambling' : 'travel') : cats[Math.floor(Math.random() * 4)]
                      });
                    }
                    setBatchFile(new File([''], 'unseen_transactions_benchmark_50.csv', { type: 'text/csv' }));
                    setBatchRows(mockTestRows);
                    setBatchError(null);
                    setBatchResponse(null);
                  }}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-purple-300 text-xs font-medium flex items-center gap-1.5 transition cursor-pointer"
                  title="Load 50 sample unseen transactions with model features"
                >
                  <Sparkles className="h-3.5 w-3.5 text-purple-400" />
                  <span>Load Sample Batch (50 rows)</span>
                </button>

                {batchResponse && (
                  <button
                    onClick={handleDownloadResultsCsv}
                    className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-medium flex items-center gap-1.5 transition cursor-pointer shadow"
                  >
                    <Download className="h-3.5 w-3.5" />
                    <span>Download Prediction CSV</span>
                  </button>
                )}
              </div>
            </div>

            {/* Drop Zone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={e => e.preventDefault()}
              onDrop={e => {
                e.preventDefault();
                if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                  handleBatchFileUpload(e.dataTransfer.files[0]);
                }
              }}
              className="border-2 border-dashed border-slate-700 hover:border-purple-500/70 bg-slate-950/40 hover:bg-slate-900/40 rounded-xl p-6 text-center cursor-pointer transition flex flex-col items-center justify-center space-y-2"
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={e => {
                  if (e.target.files && e.target.files[0]) {
                    handleBatchFileUpload(e.target.files[0]);
                  }
                }}
                accept=".csv,text/csv"
                className="hidden"
              />
              <div className="h-10 w-10 rounded-xl bg-purple-950/60 border border-purple-800/60 flex items-center justify-center text-purple-400">
                <FileText className="h-5 w-5" />
              </div>
              <div>
                <div className="text-xs font-medium text-slate-200">
                  {batchFile ? `Loaded: ${batchFile.name} (${batchRows.length} rows)` : 'Click to select or drop unseen transactions CSV'}
                </div>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  Target column {targetCol} is automatically omitted from model inputs.
                </p>
              </div>
            </div>

            {batchError && (
              <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 shrink-0 text-rose-400" />
                <span>{batchError}</span>
              </div>
            )}

            {/* Run Button */}
            {batchRows.length > 0 && (
              <div className="flex items-center justify-between p-3 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-xs text-slate-300">
                  <span className="font-semibold text-white">{batchRows.length}</span> unseen records ready for inference
                </div>
                <button
                  onClick={handleRunBatchPrediction}
                  disabled={isPredictingBatch}
                  className="px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 text-white font-medium text-xs flex items-center gap-2 transition cursor-pointer disabled:opacity-50"
                >
                  {isPredictingBatch ? (
                    <>
                      <Activity className="h-3.5 w-3.5 animate-spin" />
                      <span>Classifying Rows...</span>
                    </>
                  ) : (
                    <>
                      <Play className="h-3.5 w-3.5" />
                      <span>Generate Batch Predictions</span>
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Batch Telemetry Metrics Card */}
            {batchResponse && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase font-mono">Total Evaluated</div>
                  <div className="text-lg font-bold font-mono text-white mt-0.5">
                    {batchResponse.totalSamples}
                  </div>
                  <div className="text-[10px] text-slate-400">Transactions</div>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-rose-900/50">
                  <div className="text-[10px] text-rose-400 uppercase font-mono">Predicted Fraud</div>
                  <div className="text-lg font-bold font-mono text-rose-300 mt-0.5">
                    {batchResponse.fraudCount}
                  </div>
                  <div className="text-[10px] text-rose-400">{batchResponse.fraudRatePct}% of batch</div>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-emerald-900/50">
                  <div className="text-[10px] text-emerald-400 uppercase font-mono">Predicted Legitimate</div>
                  <div className="text-lg font-bold font-mono text-emerald-300 mt-0.5">
                    {batchResponse.legitimateCount}
                  </div>
                  <div className="text-[10px] text-emerald-400">
                    {(100 - batchResponse.fraudRatePct).toFixed(1)}% of batch
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase font-mono">Inference Latency</div>
                  <div className="text-lg font-bold font-mono text-cyan-300 mt-0.5">
                    {batchResponse.inferenceLatencyMs} ms
                  </div>
                  <div className="text-[10px] text-cyan-400">
                    {(batchResponse.inferenceLatencyMs / (batchResponse.totalSamples || 1)).toFixed(2)} ms / sample
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Predictions Table Card */}
          {batchResponse && (
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <h4 className="text-sm font-semibold text-white">Prediction Results Feed</h4>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
                    {filteredPredictions.length} matching
                  </span>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-300 border border-emerald-800/60 font-mono">
                    {batchResponse.totalSamples} total processed
                  </span>
                </div>

                {/* Filter, Search, and Download */}
                <div className="flex items-center gap-2 flex-wrap">
                  <div className="relative">
                    <Search className="h-3.5 w-3.5 absolute left-2.5 top-2.5 text-slate-500" />
                    <input
                      type="text"
                      placeholder="Search row or value..."
                      value={searchTerm}
                      onChange={e => {
                        setSearchTerm(e.target.value);
                        setPage(0);
                      }}
                      className="pl-8 pr-3 py-1 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 w-36 sm:w-44"
                    />
                  </div>

                  <select
                    value={filterClass}
                    onChange={e => {
                      setFilterClass(e.target.value as any);
                      setPage(0);
                    }}
                    className="px-2.5 py-1 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-300 focus:outline-none"
                  >
                    <option value="all">All Classes</option>
                    <option value="Fraud">Fraud Only</option>
                    <option value="Legitimate">Legitimate Only</option>
                  </select>

                  <button
                    onClick={handleDownloadResultsCsv}
                    className="px-3 py-1 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-medium flex items-center gap-1.5 transition cursor-pointer shadow"
                    title="Export all rows with predicted classes and probabilities as CSV"
                  >
                    <Download className="h-3.5 w-3.5" />
                    <span>Download CSV ({batchResponse.totalSamples})</span>
                  </button>
                </div>
              </div>

              {/* Table */}
              <div className="overflow-x-auto rounded-xl border border-slate-800">
                <table className="w-full text-left text-xs text-slate-300">
                  <thead className="bg-slate-950/80 text-[11px] font-mono text-slate-400 border-b border-slate-800 uppercase">
                    <tr>
                      <th className="px-3 py-2.5">Row #</th>
                      <th className="px-3 py-2.5">Predicted Class</th>
                      <th className="px-3 py-2.5">Fraud Probability</th>
                      <th className="px-3 py-2.5">Confidence</th>
                      <th className="px-3 py-2.5">Verification Status</th>
                      <th className="px-3 py-2.5">Sample Feature Values</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {paginatedPredictions.length === 0 ? (
                      <tr>
                        <td colSpan={6} className="text-center py-6 text-slate-500 text-xs">
                          No transaction records match the current filter.
                        </td>
                      </tr>
                    ) : (
                      paginatedPredictions.map(item => (
                        <tr key={item.rowIndex} className="hover:bg-slate-800/30 transition">
                          <td className="px-3 py-2.5 text-slate-400">#{item.rowIndex}</td>
                          <td className="px-3 py-2.5">
                            <span
                              className={`px-2 py-0.5 rounded text-[11px] font-semibold border ${
                                item.predictedClass === 'Fraud'
                                  ? 'bg-rose-950/60 text-rose-300 border-rose-800/80'
                                  : 'bg-emerald-950/60 text-emerald-300 border-emerald-800/80'
                              }`}
                            >
                              {item.predictedClass === 'Fraud' ? '🚨 Fraud' : '✅ Legitimate'}
                            </span>
                          </td>
                          <td className="px-3 py-2.5">
                            <div className="flex items-center gap-2">
                              <span>{(item.fraudProbability * 100).toFixed(1)}%</span>
                              <div className="w-12 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                <div
                                  className={`h-full ${
                                    item.fraudProbability >= 0.5 ? 'bg-rose-500' : 'bg-emerald-500'
                                  }`}
                                  style={{ width: `${item.fraudProbability * 100}%` }}
                                />
                              </div>
                            </div>
                          </td>
                          <td className="px-3 py-2.5 text-slate-300">{item.confidenceScore.toFixed(1)}%</td>
                          <td className="px-3 py-2.5">
                            <span className="text-[10px] px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-400">
                              {item.verifiedOutcome}
                            </span>
                          </td>
                          <td className="px-3 py-2.5 text-[10px] text-slate-400 max-w-[200px] truncate">
                            {Object.entries(item.inputFeatures)
                              .slice(0, 3)
                              .map(([k, v]) => `${k}:${v}`)
                              .join(' | ')}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between text-xs text-slate-400 pt-2">
                  <span>
                    Showing {page * pageSize + 1} to{' '}
                    {Math.min(filteredPredictions.length, (page + 1) * pageSize)} of {filteredPredictions.length}
                  </span>
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => setPage(p => Math.max(0, p - 1))}
                      disabled={page === 0}
                      className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 transition cursor-pointer"
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </button>
                    <span className="px-2 font-mono text-[11px]">
                      {page + 1} / {totalPages}
                    </span>
                    <button
                      onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                      disabled={page >= totalPages - 1}
                      className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 transition cursor-pointer"
                    >
                      <ChevronRight className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
