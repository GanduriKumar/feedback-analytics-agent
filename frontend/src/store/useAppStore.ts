import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { AnalysisReport, DataSource, LLMConfig, PipelineProgress, UserType, TimeFilter } from '../types';

const STORE_NAME = 'feedback-analytics-ui';
const STORE_VERSION = 5;
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

  timeFilter: TimeFilter;
  setTimeFilter: (t: TimeFilter) => void;

  llmConfig: LLMConfig;
  setLLMConfig: (c: LLMConfig) => void;

  // pipeline
  isRunning: boolean;
  isPaused: boolean;
  progress: PipelineProgress;
  setProgress: (p: PipelineProgress) => void;
  lastError: string | null;
  setPaused: (v: boolean) => void;

  // results
  lastRun: AnalysisReport | null;
  analysisHistory: AnalysisReport[];
  selectedReportId: string | null;
  setLastRun: (r: AnalysisReport) => void;
  addToHistory: (r: AnalysisReport) => void;
  setSelectedReport: (id: string | null) => void;

  purgeReports: () => void;

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
const defaultTimeFilter: TimeFilter = 'all';

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

      timeFilter: defaultTimeFilter,
      setTimeFilter: (t) => set({ timeFilter: t }),

      llmConfig: defaultLLM,
      setLLMConfig: (c) => set({ llmConfig: c }),

      isRunning: false,
      isPaused: false,
      progress: defaultProgress,
      lastError: null,
      setProgress: (p) =>
        set({
          progress: p,
          isRunning: !['idle', 'complete', 'error', 'aborted'].includes(p.stage),
          lastError: p.stage === 'error' ? p.message : null,
        }),
      setPaused: (v) => set({ isPaused: v }),

      lastRun: null,
      analysisHistory: [],
      selectedReportId: null,
      setLastRun: (r) => set({ lastRun: r, selectedReportId: r.id }),
      addToHistory: (r) =>
        set((state) => ({
          analysisHistory: [r, ...state.analysisHistory].slice(0, MAX_HISTORY),
        })),
      setSelectedReport: (id) => set({ selectedReportId: id }),

      purgeReports: () =>
        set({
          lastRun: null,
          analysisHistory: [],
          selectedReportId: null,
        }),

      clearHistory: () => set({ analysisHistory: [] }),
      deleteFromHistory: (id) =>
        set((state) => ({
          analysisHistory: state.analysisHistory.filter((r) => r.id !== id),
          lastRun: state.lastRun?.id === id ? null : state.lastRun,
        })),

      resetRunState: () =>
        set({
          isRunning: false,
          isPaused: false,
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
        const next = { ...persisted } as any;

        if (version <= 1) {
          next.userType = null;
          next.searchQueries = [];
          next.selectedSources = defaultSources;
        }
        if (version <= 2) {
          next.selectedReportId = persisted?.lastRun?.id ?? null;
        }
        if (version <= 3 || next.timeFilter === undefined) {
          next.timeFilter = persisted?.timeFilter || defaultTimeFilter;
        }
        if (version <= 4 || next.isPaused === undefined) {
          next.isPaused = false;
        }

        return next;
      },
      partialize: (state) => ({
        lastRun: state.lastRun,
        analysisHistory: state.analysisHistory,
        llmConfig: state.llmConfig,
        userType: state.userType,
        searchQueries: state.searchQueries,
        selectedSources: state.selectedSources,
        timeFilter: state.timeFilter,
        selectedReportId: state.selectedReportId,
        isPaused: state.isPaused,
      }),
    }
  )
);
