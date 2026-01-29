import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { AnalysisReport, DataSource, LLMConfig, PipelineProgress, UserType } from '../types';

const STORE_NAME = 'feedback-analytics-ui';
const STORE_VERSION = 2;
const MAX_HISTORY = 20;

function normalizeQuery(q: string): string {
  return q.replace(/[\r\n]+/g, ' ').replace(/\s+/g, ' ').trim();
}

interface AppState {
  // selections
  userType: UserType | null;
  setUserType: (t: UserType) => void;

  searchQueries: string[];
  setSearchQueries: (q: string[]) => void;
  addSearchQuery: (q: string) => void;
  removeSearchQuery: (q: string) => void;
  clearSearchQueries: () => void;

  selectedSources: DataSource[];
  setSelectedSources: (s: DataSource[]) => void;
  toggleSource: (s: DataSource) => void;

  llmConfig: LLMConfig;
  setLLMConfig: (c: LLMConfig) => void;

  // pipeline
  isRunning: boolean;
  progress: PipelineProgress;
  setProgress: (p: PipelineProgress) => void;
  lastError: string | null;

  // results
  lastRun: AnalysisReport | null;
  analysisHistory: AnalysisReport[];
  setLastRun: (r: AnalysisReport) => void;
  addToHistory: (r: AnalysisReport) => void;

  clearHistory: () => void;
  deleteFromHistory: (id: string) => void;

  resetRunState: () => void;
  resetSelections: () => void;
}

const defaultLLM: LLMConfig = {
  provider: 'ollama',
  model: 'mistral',
  baseUrl: 'http://localhost:11434',
};

const defaultProgress: PipelineProgress = { stage: 'idle', message: '', progress: 0 };
const defaultSources: DataSource[] = ['reddit'];

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      userType: null,
      setUserType: (t) => set({ userType: t }),

      searchQueries: [],
      setSearchQueries: (q) => set({ searchQueries: q }),
      addSearchQuery: (q) =>
        set((state) => {
          const next = normalizeQuery(q);
          if (!next) return state;
          if (state.searchQueries.includes(next)) return state;
          return { searchQueries: [...state.searchQueries, next] };
        }),
      removeSearchQuery: (q) =>
        set((state) => ({ searchQueries: state.searchQueries.filter((x) => x !== q) })),
      clearSearchQueries: () => set({ searchQueries: [] }),

      selectedSources: defaultSources,
      setSelectedSources: (s) => set({ selectedSources: s }),
      toggleSource: (s) =>
        set((state) => {
          const has = state.selectedSources.includes(s);
          const next = has ? state.selectedSources.filter((x) => x !== s) : [...state.selectedSources, s];
          return { selectedSources: next };
        }),

      llmConfig: defaultLLM,
      setLLMConfig: (c) => set({ llmConfig: c }),

      isRunning: false,
      progress: defaultProgress,
      lastError: null,
      setProgress: (p) =>
        set({
          progress: p,
          isRunning: p.stage !== 'idle' && p.stage !== 'complete' && p.stage !== 'error',
          lastError: p.stage === 'error' ? p.message : null,
        }),

      lastRun: null,
      analysisHistory: [],
      setLastRun: (r) => set({ lastRun: r }),
      addToHistory: (r) =>
        set((state) => ({
          analysisHistory: [r, ...state.analysisHistory].slice(0, MAX_HISTORY),
        })),

      clearHistory: () => set({ analysisHistory: [] }),
      deleteFromHistory: (id) =>
        set((state) => ({
          analysisHistory: state.analysisHistory.filter((r) => r.id !== id),
          lastRun: state.lastRun?.id === id ? null : state.lastRun,
        })),

      resetRunState: () =>
        set({
          isRunning: false,
          progress: defaultProgress,
          lastError: null,
        }),

      resetSelections: () =>
        set({
          userType: null,
          searchQueries: [],
          selectedSources: defaultSources,
          llmConfig: defaultLLM,
        }),
    }),
    {
      name: STORE_NAME,
      version: STORE_VERSION,
      migrate: (persisted: any, version) => {
        // v1 persisted only {lastRun, analysisHistory, llmConfig}
        if (version === 1) {
          return {
            ...persisted,
            userType: null,
            searchQueries: [],
            selectedSources: defaultSources,
          };
        }
        return persisted;
      },
      partialize: (state) => ({
        lastRun: state.lastRun,
        analysisHistory: state.analysisHistory,
        llmConfig: state.llmConfig,
        userType: state.userType,
        searchQueries: state.searchQueries,
        selectedSources: state.selectedSources,
      }),
    }
  )
);
