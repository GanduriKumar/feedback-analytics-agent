import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { AnalysisReport, DataSource, LLMConfig, PipelineProgress, UserType } from '../types';

interface AppState {
  // selections
  userType: UserType | null;
  setUserType: (t: UserType) => void;

  searchQueries: string[];
  setSearchQueries: (q: string[]) => void;

  selectedSources: DataSource[];
  setSelectedSources: (s: DataSource[]) => void;

  llmConfig: LLMConfig;
  setLLMConfig: (c: LLMConfig) => void;

  // pipeline
  isRunning: boolean;
  progress: PipelineProgress;
  setProgress: (p: PipelineProgress) => void;

  // results
  lastRun: AnalysisReport | null;
  analysisHistory: AnalysisReport[];
  setLastRun: (r: AnalysisReport) => void;
  addToHistory: (r: AnalysisReport) => void;

  resetRunState: () => void;
}

const defaultLLM: LLMConfig = {
  provider: 'ollama',
  model: 'mistral',
  baseUrl: 'http://localhost:11434',
};

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      userType: null,
      setUserType: (t) => set({ userType: t }),

      searchQueries: [],
      setSearchQueries: (q) => set({ searchQueries: q }),

      selectedSources: ['reddit'],
      setSelectedSources: (s) => set({ selectedSources: s }),

      llmConfig: defaultLLM,
      setLLMConfig: (c) => set({ llmConfig: c }),

      isRunning: false,
      progress: { stage: 'idle', message: '', progress: 0 },
      setProgress: (p) =>
        set({
          progress: p,
          isRunning: p.stage !== 'idle' && p.stage !== 'complete' && p.stage !== 'error',
        }),

      lastRun: null,
      analysisHistory: [],
      setLastRun: (r) => set({ lastRun: r }),
      addToHistory: (r) =>
        set((state) => ({
          analysisHistory: [r, ...state.analysisHistory].slice(0, 20),
        })),

      resetRunState: () =>
        set({
          isRunning: false,
          progress: { stage: 'idle', message: '', progress: 0 },
        }),
    }),
    {
      name: 'feedback-analytics-ui',
      partialize: (state) => ({
        lastRun: state.lastRun,
        analysisHistory: state.analysisHistory,
        llmConfig: state.llmConfig,
      }),
    }
  )
);
