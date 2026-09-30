import React from 'react';
import {
  LayoutDashboard,
  Database,
  Cpu,
  Sliders,
  Wind,
  Search,
  GitCompare,
  Settings,
  ShieldAlert
} from 'lucide-react';

export type TabId =
  | 'dashboard'
  | 'dataset'
  | 'training'
  | 'optimization'
  | 'drift'
  | 'explain'
  | 'comparison'
  | 'test_prediction'
  | 'settings';

interface NavigationProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  hasDataset: boolean;
  hasTrainedModel: boolean;
  driftAlert?: boolean;
}

export const Navigation: React.FC<NavigationProps> = ({
  activeTab,
  onTabChange,
  hasDataset,
  hasTrainedModel,
  driftAlert
}) => {
  const tabs = [
    { id: 'dashboard' as TabId, label: 'Dashboard', icon: LayoutDashboard },
    { id: 'dataset' as TabId, label: 'Dataset', icon: Database, badge: !hasDataset ? 'Upload' : undefined },
    { id: 'training' as TabId, label: 'Model Training', icon: Cpu, disabled: !hasDataset },
    { id: 'optimization' as TabId, label: 'Optuna Tuning', icon: Sliders, disabled: !hasDataset },
    { id: 'drift' as TabId, label: 'Drift & Robustness', icon: Wind, disabled: !hasTrainedModel, alert: driftAlert },
    { id: 'explain' as TabId, label: 'Explainable AI', icon: Search, disabled: !hasTrainedModel },
    { id: 'comparison' as TabId, label: 'Model Comparison', icon: GitCompare, disabled: !hasTrainedModel },
    { id: 'test_prediction' as TabId, label: 'Test Prediction', icon: ShieldAlert, disabled: !hasTrainedModel, badge: 'Live' },
    { id: 'settings' as TabId, label: 'Settings', icon: Settings },
  ];

  return (
    <nav className="border-b border-slate-800 bg-slate-950/60 sticky top-[61px] z-30 px-6 backdrop-blur-sm">
      <div className="max-w-7xl mx-auto flex items-center gap-1 overflow-x-auto py-2 no-scrollbar">
        {tabs.map(tab => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;

          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-medium transition-all whitespace-nowrap cursor-pointer ${
                isActive
                  ? 'bg-slate-800 text-cyan-300 shadow-sm border border-slate-700/60'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <Icon className={`h-4 w-4 ${isActive ? 'text-cyan-400' : 'text-slate-500'}`} />
              <span>{tab.label}</span>

              {tab.badge && (
                <span className="text-[10px] px-1.5 py-0.2 rounded bg-indigo-900/50 text-indigo-300 border border-indigo-700/40">
                  {tab.badge}
                </span>
              )}

              {tab.alert && (
                <span className="h-2 w-2 rounded-full bg-rose-500 animate-ping" />
              )}
            </button>
          );
        })}
      </div>
    </nav>
  );
};
