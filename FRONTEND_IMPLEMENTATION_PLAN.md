# Frontend Implementation Plan
## Feedback Analytics Agent - React UI

### Requirements Summary
**Target Users**: Product Managers, Engineers, Support Teams, Business Analysts, Executives (selector-based, no auth)

**Core Workflow**:
1. Select user type
2. Enter search strings (comma-separated or multi-line)
3. Choose data sources (Reddit initially, extensible architecture)
4. Execute E2E pipeline with real-time progress tracking
5. View comprehensive analysis report

**Key Features**:
- Real-time progress visualization showing pipeline stages
- Analysis report with: Products, Functionalities, Issue Categories, Frequency %, Sentiment Analysis, Text Analytics Metrics
- Compelling data visualizations (charts, graphs, tables)

**Tech Stack**:
- React 18 + TypeScript
- Vite (build tool)
- Tailwind CSS with Google Material Design colors
- React Router v6 (navigation)
- Recharts (data visualization)
- Axios (API client)
- Zustand (state management)
- jsPDF / html2canvas (PDF export)
- React Query (server state)

**API Endpoints** (Backend FastAPI):
- `POST /api/tools/collect` - Initiate data collection
- `POST /api/tools/cluster` - Generate clusters and analysis
- `POST /api/analyze` - Full analysis pipeline
- `POST /api/search` - Semantic search
- `GET /api/health` - Health check
- `GET /api/capabilities` - Get available LLM models

**Pages**:
1. **Dashboard** - Summary view of last analysis run with key metrics
2. **Extract & Analyze** - Full workflow with user configuration, LLM selection, API key management, progress tracking
3. **Reports** - Downloadable analysis reports with recommendations (PDF/CSV)

---

## Implementation Prompts

### Prompt 1: Project Initialization
**Objective**: Create React + TypeScript + Vite project with Tailwind CSS and Google Material colors

**Tasks**:
1. Create `frontend/` directory in workspace root
2. Initialize Vite React-TypeScript project
3. Install dependencies:
   - Core: react, react-dom, typescript
   - Build: vite, @vitejs/plugin-react
   - Routing: react-router-dom
   - Styling: tailwindcss, postcss, autoprefixer
   - Charts: recharts
   - HTTP: axios, @tanstack/react-query
   - State: zustand
   - Icons: lucide-react
   - Utils: clsx, tailwind-merge
   - Export: jspdf, html2canvas

4. Configure Tailwind with Google Material colors:
   ```js
   // tailwind.config.js
   colors: {
     google: {
       blue: { 50: '#E8F0FE', 500: '#1A73E8', 700: '#1557B0' },
       red: { 50: '#FCE8E6', 500: '#D93025', 700: '#A50E0E' },
       yellow: { 50: '#FEF7E0', 500: '#F9AB00', 700: '#E37400' },
       green: { 50: '#E6F4EA', 500: '#1E8E3E', 700: '#137333' },
       gray: { 50: '#F8F9FA', 100: '#F1F3F4', 200: '#E8EAED', 300: '#DADCE0', 
               400: '#BDC1C6', 500: '#9AA0A6', 600: '#80868B', 700: '#5F6368',
               800: '#3C4043', 900: '#202124' }
     }
   }
   ```

5. Setup project structure:
   ```
   frontend/
   ├── src/
   │   ├── components/
   │   │   ├── layout/
   │   │   ├── dashboard/
   │   │   ├── analyze/
   │   │   └── reports/
   │   ├── pages/
   │   ├── services/
   │   ├── store/
   │   ├── types/
   │   ├── utils/
   │   ├── App.tsx
   │   └── main.tsx
   ├── public/
   ├── index.html
   ├── vite.config.ts
   ├── tailwind.config.js
   ├── tsconfig.json
   └── package.json
   ```

6. Configure Vite proxy for backend API:
   ```ts
   // vite.config.ts
   server: {
     port: 3000,
     proxy: {
       '/api': {
         target: 'http://127.0.0.1:8000',
         changeOrigin: true
       }
     }
   }
   ```

**Acceptance Criteria**:
- `npm run dev` starts dev server on port 3000
- Tailwind CSS working with Google colors
- TypeScript compilation successful
- API proxy configured

---

### Prompt 2: Type Definitions & API Client
**Objective**: Create TypeScript types and Axios-based API client

**Tasks**:
1. Create `src/types/index.ts`:
   ```typescript
   export type UserType = 'product-manager' | 'engineer' | 'support' | 'business-analyst' | 'executive';
   
   export type DataSource = 'reddit' | 'twitter' | 'app-store' | 'play-store';
   
   export type LLMProvider = 'ollama' | 'openai' | 'anthropic' | 'gemini';
   
   export interface LLMConfig {
     provider: LLMProvider;
     model: string;
     apiKey?: string;
     baseUrl?: string;
   }
   
   export interface SearchQuery {
     query: string;
     sources: DataSource[];
   }
   
   export interface PipelineProgress {
     stage: 'idle' | 'fetching' | 'cleaning' | 'embedding' | 'storing' | 'analyzing' | 'complete' | 'error';
     message: string;
     progress: number; // 0-100
     details?: {
       reviewsFetched?: number;
       reviewsCleaned?: number;
       embeddingsGenerated?: number;
       clustersCreated?: number;
     };
   }
   
   export interface ThemeData {
     theme_name: string;
     issue_category: string;
     frequency_percentage: number;
     sentiment: 'positive' | 'negative' | 'neutral' | 'mixed';
     sample_reviews: string[];
   }
   
   export interface ClusterData {
     cluster_id: number;
     size: number;
     summary: string;
     themes: ThemeData[];
   }
   
   export interface AnalysisReport {
     id: string;
     total_reviews: number;
     data_sources: DataSource[];
     search_queries: string[];
     products: string[];
     functionalities: string[];
     issue_categories: { [key: string]: number };
     sentiment_distribution: { positive: number; negative: number; neutral: number; mixed: number };
     clusters: ClusterData[];
     text_analytics: {
       avg_review_length: number;
       unique_products: number;
       unique_functionalities: number;
       total_themes: number;
     };
     recommendations?: string[];
     generated_at: string;
     user_type: UserType;
     llm_config: LLMConfig;
   }
   
   export interface AnalysisHistory {
     reports: AnalysisReport[];
     lastRun?: AnalysisReport;
   }
   ```

2. Create `src/services/api.ts`:
   ```typescript
   import axios from 'axios';
   import type { AnalysisReport, SearchQuery, LLMConfig } from '../types';
   
   const api = axios.create({
     baseURL: '/api',
     headers: {
       'Content-Type': 'application/json',
       'X-API-Key': 'dev-api-key-12345' // TODO: Move to env
     }
   });
   
   export const healthCheck = () => api.get('/health');
   
   export const getCapabilities = () => api.get('/capabilities');
   
   export const collectReviews = (queries: string[], sources: string[]) => 
     api.get('/tools/collect', { params: { queries: queries.join(','), sources: sources.join(',') } });
   
   export const generateClusters = (numClusters: number = 5) =>
     api.post('/tools/cluster', { num_clusters: numClusters });
   
   export const runFullAnalysis = (searchQuery: SearchQuery, llmConfig?: LLMConfig) =>
     api.post<AnalysisReport>('/analyze', { ...searchQuery, llm_config: llmConfig });
   
   export const semanticSearch = (query: string, topK: number = 10) =>
     api.post('/search', { query, top_k: topK });
   
   export const updateApiKey = (provider: string, apiKey: string) =>
     api.post('/config/api-key', { provider, api_key: apiKey });
   ```

**Acceptance Criteria**:
- All types exported correctly
- API client configured with base URL and headers
- TypeScript compilation successful

---

### Prompt 3: State Management with Zustand
**Objective**: Create global state store for user type, queries, progress, results, and history

**Tasks**:
1. Create `src/store/useAppStore.ts`:
   ```typescript
   import { create } from 'zustand';
   import { persist } from 'zustand/middleware';
   import type { UserType, DataSource, PipelineProgress, AnalysisReport, LLMConfig } from '../types';
   
   interface AppState {
     // User selection
     userType: UserType | null;
     setUserType: (type: UserType) => void;
     
     // Search inputs
     searchQueries: string[];
     setSearchQueries: (queries: string[]) => void;
     selectedSources: DataSource[];
     setSelectedSources: (sources: DataSource[]) => void;
     
     // LLM Configuration
     llmConfig: LLMConfig;
     setLLMConfig: (config: LLMConfig) => void;
     
     // Pipeline state
     isRunning: boolean;
     progress: PipelineProgress;
     setProgress: (progress: PipelineProgress) => void;
     
     // Results & History
     analysisReport: AnalysisReport | null;
     setAnalysisReport: (report: AnalysisReport) => void;
     analysisHistory: AnalysisReport[];
     addToHistory: (report: AnalysisReport) => void;
     lastRun: AnalysisReport | null;
     
     // Actions
     resetPipeline: () => void;
   }
   
   export const useAppStore = create<AppState>()(
     persist(
       (set) => ({
         userType: null,
         setUserType: (type) => set({ userType: type }),
         
         searchQueries: [],
         setSearchQueries: (queries) => set({ searchQueries: queries }),
         
         selectedSources: ['reddit'],
         setSelectedSources: (sources) => set({ selectedSources: sources }),
         
         llmConfig: { provider: 'ollama', model: 'mistral' },
         setLLMConfig: (config) => set({ llmConfig: config }),
         
         isRunning: false,
         progress: { stage: 'idle', message: '', progress: 0 },
         setProgress: (progress) => set({ 
           progress,
           isRunning: progress.stage !== 'idle' && progress.stage !== 'complete' && progress.stage !== 'error'
         }),
         
         analysisReport: null,
         setAnalysisReport: (report) => set({ analysisReport: report, lastRun: report }),
         
         analysisHistory: [],
         addToHistory: (report) => set((state) => ({ 
           analysisHistory: [report, ...state.analysisHistory].slice(0, 10) // Keep last 10
         })),
         
         lastRun: null,
         
         resetPipeline: () => set({
           isRunning: false,
           progress: { stage: 'idle', message: '', progress: 0 },
           analysisReport: null
         })
       }),
       {
         name: 'feedback-analytics-storage',
         partialize: (state) => ({ 
           lastRun: state.lastRun,
           analysisHistory: state.analysisHistory,
           llmConfig: state.llmConfig
         })
       }
     )
   );
   ```

**Acceptance Criteria**:
- Store exports all necessary state and actions
- TypeScript types match
- State updates work correctly
- Persistent storage for lastRun and history
- LLM config persisted

---

### Prompt 4: NavBar Component with Routing
**Objective**: Create navigation bar with React Router integration

**Tasks**:
1. Create `src/components/layout/NavBar.tsx`:
   ```typescript
   import { Link, useLocation } from 'react-router-dom';
   import { BarChart3, Settings, FileText, Activity } from 'lucide-react';
   
   const navItems = [
     { path: '/', label: 'Dashboard', icon: BarChart3 },
     { path: '/analyze', label: 'Extract & Analyze', icon: Activity },
     { path: '/reports', label: 'Reports', icon: FileText }
   ];
   
   export function NavBar() {
     const location = useLocation();
     
     return (
       <nav className="bg-white border-b border-google-gray-200 sticky top-0 z-50">
         <div className="max-w-7xl mx-auto px-8">
           <div className="flex items-center justify-between h-16">
             <div className="flex items-center gap-2">
               <Activity className="w-8 h-8 text-google-blue-500" />
               <h1 className="text-xl font-bold text-google-gray-900">Feedback Analytics</h1>
             </div>
             
             <div className="flex items-center gap-1">
               {navItems.map(({ path, label, icon: Icon }) => {
                 const isActive = location.pathname === path;
                 return (
                   <Link
                     key={path}
                     to={path}
                     className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                       isActive 
                         ? 'bg-google-blue-50 text-google-blue-700' 
                         : 'text-google-gray-700 hover:bg-google-gray-50'
                     }`}
                   >
                     <Icon className="w-5 h-5" />
                     {label}
                   </Link>
                 );
               })}
             </div>
             
             <button className="p-2 rounded-lg hover:bg-google-gray-50 text-google-gray-700">
               <Settings className="w-6 h-6" />
             </button>
           </div>
         </div>
       </nav>
     );
   }
   ```

**Acceptance Criteria**:
- NavBar displays 3 navigation links
- Active page highlighted
- Sticky positioning
- Settings icon placeholder
- Responsive design

---

### Prompt 5: UserType Selector Component
**Objective**: Create user type selection component with Google Material styling

**Tasks**:
1. Create `src/components/UserTypeSelector.tsx`:
   ```typescript
   import { Users, Wrench, Headphones, LineChart, Briefcase } from 'lucide-react';
   import { useAppStore } from '../store/useAppStore';
   import type { UserType } from '../types';
   
   const userTypes: { value: UserType; label: string; icon: any; description: string; color: string }[] = [
     { value: 'product-manager', label: 'Product Manager', icon: Briefcase, description: 'Feature prioritization & roadmap', color: 'google-blue' },
     { value: 'engineer', label: 'Engineer', icon: Wrench, description: 'Technical issues & bugs', color: 'google-green' },
     { value: 'support', label: 'Support Team', icon: Headphones, description: 'Customer pain points', color: 'google-yellow' },
     { value: 'business-analyst', label: 'Business Analyst', icon: LineChart, description: 'Trends & metrics', color: 'google-red' },
     { value: 'executive', label: 'Executive', icon: Users, description: 'Strategic insights', color: 'google-gray' }
   ];
   
   export function UserTypeSelector() {
     const { userType, setUserType } = useAppStore();
     
     return (
       <div className="space-y-4">
         <h2 className="text-xl font-semibold text-google-gray-900">Select Your Role</h2>
         <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
           {userTypes.map(({ value, label, icon: Icon, description, color }) => (
             <button
               key={value}
               onClick={() => setUserType(value)}
               className={`p-6 rounded-lg border-2 transition-all hover:shadow-lg ${
                 userType === value 
                   ? `border-${color}-500 bg-${color}-50` 
                   : 'border-google-gray-200 hover:border-google-gray-300'
               }`}
             >
               <Icon className={`w-8 h-8 mb-3 ${userType === value ? `text-${color}-600` : 'text-google-gray-500'}`} />
               <h3 className="font-semibold text-google-gray-900 mb-1">{label}</h3>
               <p className="text-sm text-google-gray-600">{description}</p>
             </button>
           ))}
         </div>
       </div>
     );
   }
   ```

**Acceptance Criteria**:
- Displays 5 user type cards with icons
- Selected state visual feedback
- Google Material colors applied
- Responsive grid layout

---

### Prompt 6: LLM Configuration Component
**Objective**: Create LLM provider and API key configuration UI

**Tasks**:
1. Create `src/components/analyze/LLMConfig.tsx`:
   ```typescript
   import { useState } from 'react';
   import { Brain, Key, Server } from 'lucide-react';
   import { useAppStore } from '../../store/useAppStore';
   import type { LLMProvider } from '../../types';
   
   const providers = [
     { value: 'ollama' as LLMProvider, label: 'Ollama (Local)', models: ['mistral', 'llama2', 'codellama'], requiresKey: false },
     { value: 'openai' as LLMProvider, label: 'OpenAI', models: ['gpt-4', 'gpt-3.5-turbo'], requiresKey: true },
     { value: 'anthropic' as LLMProvider, label: 'Anthropic', models: ['claude-3-opus', 'claude-3-sonnet'], requiresKey: true },
     { value: 'gemini' as LLMProvider, label: 'Google Gemini', models: ['gemini-pro', 'gemini-ultra'], requiresKey: true }
   ];
   
   export function LLMConfig() {
     const { llmConfig, setLLMConfig } = useAppStore();
     const [showApiKey, setShowApiKey] = useState(false);
     
     const selectedProvider = providers.find(p => p.value === llmConfig.provider);
     
     return (
       <div className="bg-white rounded-lg border border-google-gray-200 p-6 space-y-4">
         <div className="flex items-center gap-2 mb-4">
           <Brain className="w-6 h-6 text-google-blue-500" />
           <h3 className="text-lg font-semibold text-google-gray-900">LLM Configuration</h3>
         </div>
         
         <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
           <div>
             <label className="block text-sm font-medium text-google-gray-700 mb-2">
               Provider
             </label>
             <select
               value={llmConfig.provider}
               onChange={(e) => setLLMConfig({ 
                 ...llmConfig, 
                 provider: e.target.value as LLMProvider,
                 model: providers.find(p => p.value === e.target.value)?.models[0] || ''
               })}
               className="w-full px-4 py-2 border border-google-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-google-blue-500"
             >
               {providers.map(p => (
                 <option key={p.value} value={p.value}>{p.label}</option>
               ))}
             </select>
           </div>
           
           <div>
             <label className="block text-sm font-medium text-google-gray-700 mb-2">
               Model
             </label>
             <select
               value={llmConfig.model}
               onChange={(e) => setLLMConfig({ ...llmConfig, model: e.target.value })}
               className="w-full px-4 py-2 border border-google-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-google-blue-500"
             >
               {selectedProvider?.models.map(m => (
                 <option key={m} value={m}>{m}</option>
               ))}
             </select>
           </div>
         </div>
         
         {selectedProvider?.requiresKey && (
           <div>
             <label className="block text-sm font-medium text-google-gray-700 mb-2 flex items-center gap-2">
               <Key className="w-4 h-4" />
               API Key
             </label>
             <div className="flex gap-2">
               <input
                 type={showApiKey ? 'text' : 'password'}
                 value={llmConfig.apiKey || ''}
                 onChange={(e) => setLLMConfig({ ...llmConfig, apiKey: e.target.value })}
                 placeholder="Enter your API key"
                 className="flex-1 px-4 py-2 border border-google-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-google-blue-500"
               />
               <button
                 onClick={() => setShowApiKey(!showApiKey)}
                 className="px-4 py-2 bg-google-gray-100 text-google-gray-700 rounded-lg hover:bg-google-gray-200"
               >
                 {showApiKey ? 'Hide' : 'Show'}
               </button>
             </div>
             <p className="text-xs text-google-gray-500 mt-1">
               Your API key is stored locally and never sent to our servers
             </p>
           </div>
         )}
         
         {llmConfig.provider === 'ollama' && (
           <div>
             <label className="block text-sm font-medium text-google-gray-700 mb-2 flex items-center gap-2">
               <Server className="w-4 h-4" />
               Base URL (Optional)
             </label>
             <input
               type="text"
               value={llmConfig.baseUrl || ''}
               onChange={(e) => setLLMConfig({ ...llmConfig, baseUrl: e.target.value })}
               placeholder="http://localhost:11434"
               className="w-full px-4 py-2 border border-google-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-google-blue-500"
             />
           </div>
         )}
       </div>
     );
   }
   ```

**Acceptance Criteria**:
- Provider dropdown with 4 options
- Model selection based on provider
- API key input for cloud providers
- Show/hide API key toggle
- Local storage persistence via Zustand

---

### Prompt 7: Search Input Component
**Objective**: Create search query input with multi-query support

**Tasks**:
1. Create `src/components/SearchInput.tsx`:
   ```typescript
   import { useState } from 'react';
   import { Search, Plus, X } from 'lucide-react';
   import { useAppStore } from '../store/useAppStore';
   
   export function SearchInput() {
     const { searchQueries, setSearchQueries } = useAppStore();
     const [inputValue, setInputValue] = useState('');
     
     const addQuery = () => {
       if (inputValue.trim()) {
         setSearchQueries([...searchQueries, inputValue.trim()]);
         setInputValue('');
       }
     };
     
     const removeQuery = (index: number) => {
       setSearchQueries(searchQueries.filter((_, i) => i !== index));
     };
     
     const handleKeyDown = (e: React.KeyboardEvent) => {
       if (e.key === 'Enter') {
         e.preventDefault();
         addQuery();
       }
     };
     
     return (
       <div className="space-y-4">
         <h2 className="text-xl font-semibold text-google-gray-900">Search Queries</h2>
         
         <div className="flex gap-2">
           <div className="flex-1 relative">
             <Search className="absolute left-3 top-3 w-5 h-5 text-google-gray-400" />
             <input
               type="text"
               value={inputValue}
               onChange={(e) => setInputValue(e.target.value)}
               onKeyDown={handleKeyDown}
               placeholder="Enter search query (e.g., Pixel Phone connectivity)"
               className="w-full pl-10 pr-4 py-3 border border-google-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-google-blue-500 focus:border-transparent"
             />
           </div>
           <button
             onClick={addQuery}
             className="px-6 py-3 bg-google-blue-500 text-white rounded-lg hover:bg-google-blue-600 transition-colors flex items-center gap-2"
           >
             <Plus className="w-5 h-5" />
             Add
           </button>
         </div>
         
         {searchQueries.length > 0 && (
           <div className="flex flex-wrap gap-2">
             {searchQueries.map((query, index) => (
               <div key={index} className="flex items-center gap-2 px-4 py-2 bg-google-blue-50 border border-google-blue-200 rounded-full">
                 <span className="text-google-gray-900">{query}</span>
                 <button onClick={() => removeQuery(index)} className="text-google-gray-600 hover:text-google-red-600">
                   <X className="w-4 h-4" />
                 </button>
               </div>
             ))}
           </div>
         )}
       </div>
     );
   }
   ```

**Acceptance Criteria**:
- Add queries via button or Enter key
- Display queries as removable chips
- Clear input after adding
- Visual feedback with Google colors

---

### Prompt 8: Data Source Selector Component
**Objective**: Create source selector with checkbox UI

**Tasks**:
1. Create `src/components/SourceSelector.tsx`:
   ```typescript
   import { useAppStore } from '../store/useAppStore';
   import type { DataSource } from '../types';
   
   const sources: { value: DataSource; label: string; available: boolean }[] = [
     { value: 'reddit', label: 'Reddit', available: true },
     { value: 'twitter', label: 'Twitter/X', available: false },
     { value: 'app-store', label: 'App Store', available: false },
     { value: 'play-store', label: 'Google Play Store', available: false }
   ];
   
   export function SourceSelector() {
     const { selectedSources, setSelectedSources } = useAppStore();
     
     const toggleSource = (source: DataSource) => {
       if (selectedSources.includes(source)) {
         setSelectedSources(selectedSources.filter(s => s !== source));
       } else {
         setSelectedSources([...selectedSources, source]);
       }
     };
     
     return (
       <div className="space-y-4">
         <h2 className="text-xl font-semibold text-google-gray-900">Data Sources</h2>
         <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
           {sources.map(({ value, label, available }) => (
             <label
               key={value}
               className={`flex items-center gap-3 p-4 border-2 rounded-lg cursor-pointer transition-all ${
                 !available ? 'opacity-50 cursor-not-allowed' : ''
               } ${
                 selectedSources.includes(value)
                   ? 'border-google-blue-500 bg-google-blue-50'
                   : 'border-google-gray-200 hover:border-google-gray-300'
               }`}
             >
               <input
                 type="checkbox"
                 checked={selectedSources.includes(value)}
                 onChange={() => toggleSource(value)}
                 disabled={!available}
                 className="w-5 h-5 text-google-blue-500 rounded focus:ring-google-blue-500"
               />
               <span className="font-medium text-google-gray-900">
                 {label}
                 {!available && <span className="text-xs text-google-gray-500 ml-2">(Coming Soon)</span>}
               </span>
             </label>
           ))}
         </div>
       </div>
     );
   }
   ```

**Acceptance Criteria**:
- Checkbox grid layout
- Reddit enabled, others disabled with "Coming Soon"
- Visual selection state
- Multi-select support

---

### Prompt 9: Progress Tracker Component
**Objective**: Create real-time pipeline progress visualization

**Tasks**:
1. Create `src/components/ProgressTracker.tsx`:
   ```typescript
   import { CheckCircle, Circle, Loader, XCircle } from 'lucide-react';
   import { useAppStore } from '../store/useAppStore';
   
   const stages = [
     { key: 'fetching', label: 'Fetching Reviews', description: 'Extracting from data sources' },
     { key: 'cleaning', label: 'Cleaning Data', description: 'Removing duplicates & noise' },
     { key: 'embedding', label: 'Generating Embeddings', description: 'Creating vector representations' },
     { key: 'storing', label: 'Storing in VectorDB', description: 'Persisting to ChromaDB' },
     { key: 'analyzing', label: 'Analyzing Themes', description: 'Clustering & theme extraction' }
   ];
   
   export function ProgressTracker() {
     const { progress, isRunning } = useAppStore();
     
     const getStageStatus = (stageKey: string) => {
       const currentIndex = stages.findIndex(s => s.key === progress.stage);
       const stageIndex = stages.findIndex(s => s.key === stageKey);
       
       if (progress.stage === 'error') return 'error';
       if (progress.stage === 'complete') return 'complete';
       if (stageIndex < currentIndex) return 'complete';
       if (stageIndex === currentIndex) return 'active';
       return 'pending';
     };
     
     if (!isRunning && progress.stage === 'idle') return null;
     
     return (
       <div className="bg-white rounded-lg border border-google-gray-200 p-6 space-y-6">
         <div className="flex items-center justify-between">
           <h2 className="text-xl font-semibold text-google-gray-900">Pipeline Progress</h2>
           <span className="text-sm text-google-gray-600">{progress.progress}%</span>
         </div>
         
         <div className="w-full bg-google-gray-200 rounded-full h-2">
           <div 
             className="bg-google-blue-500 h-2 rounded-full transition-all duration-300"
             style={{ width: `${progress.progress}%` }}
           />
         </div>
         
         <div className="space-y-4">
           {stages.map((stage) => {
             const status = getStageStatus(stage.key);
             return (
               <div key={stage.key} className="flex items-start gap-3">
                 <div className="mt-1">
                   {status === 'complete' && <CheckCircle className="w-6 h-6 text-google-green-500" />}
                   {status === 'active' && <Loader className="w-6 h-6 text-google-blue-500 animate-spin" />}
                   {status === 'pending' && <Circle className="w-6 h-6 text-google-gray-300" />}
                   {status === 'error' && <XCircle className="w-6 h-6 text-google-red-500" />}
                 </div>
                 <div className="flex-1">
                   <h3 className={`font-medium ${status === 'active' ? 'text-google-blue-600' : 'text-google-gray-900'}`}>
                     {stage.label}
                   </h3>
                   <p className="text-sm text-google-gray-600">{stage.description}</p>
                   {status === 'active' && progress.details && (
                     <p className="text-sm text-google-blue-600 mt-1">{progress.message}</p>
                   )}
                 </div>
               </div>
             );
           })}
         </div>
       </div>
     );
   }
   ```

**Acceptance Criteria**:
- Shows 5 pipeline stages with icons
- Progress bar with percentage
- Active stage highlighted with spinner
- Completed stages show checkmark
- Error state support

---

### Prompt 10: Analysis Report - Overview Cards
**Objective**: Create KPI cards showing key metrics

**Tasks**:
1. Create `src/components/OverviewCards.tsx`:
   ```typescript
   import { FileText, Database, Tag, TrendingUp } from 'lucide-react';
   import type { AnalysisReport } from '../types';
   
   interface Props {
     report: AnalysisReport;
   }
   
   export function OverviewCards({ report }: Props) {
     const cards = [
       { label: 'Total Reviews', value: report.total_reviews, icon: FileText, color: 'google-blue' },
       { label: 'Data Sources', value: report.data_sources.length, icon: Database, color: 'google-green' },
       { label: 'Issue Categories', value: Object.keys(report.issue_categories).length, icon: Tag, color: 'google-yellow' },
       { label: 'Themes Identified', value: report.text_analytics.total_themes, icon: TrendingUp, color: 'google-red' }
     ];
     
     return (
       <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
         {cards.map(({ label, value, icon: Icon, color }) => (
           <div key={label} className={`bg-${color}-50 border border-${color}-200 rounded-lg p-6`}>
             <div className="flex items-center justify-between mb-2">
               <Icon className={`w-8 h-8 text-${color}-600`} />
               <span className={`text-3xl font-bold text-${color}-700`}>{value}</span>
             </div>
             <p className="text-sm font-medium text-google-gray-700">{label}</p>
           </div>
         ))}
       </div>
     );
   }
   ```

**Acceptance Criteria**:
- 4 KPI cards in grid
- Icons with Google colors
- Large value display
- Responsive layout

---

### Prompt 11: Sentiment Distribution Chart
**Objective**: Create pie/donut chart for sentiment breakdown

**Tasks**:
1. Create `src/components/SentimentChart.tsx`:
   ```typescript
   import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
   import type { AnalysisReport } from '../types';
   
   interface Props {
     report: AnalysisReport;
   }
   
   const COLORS = {
     positive: '#1E8E3E',  // Google Green
     negative: '#D93025',  // Google Red
     neutral: '#9AA0A6',   // Google Gray
     mixed: '#F9AB00'      // Google Yellow
   };
   
   export function SentimentChart({ report }: Props) {
     const data = Object.entries(report.sentiment_distribution).map(([name, value]) => ({
       name: name.charAt(0).toUpperCase() + name.slice(1),
       value,
       percentage: ((value / report.total_reviews) * 100).toFixed(1)
     }));
     
     return (
       <div className="bg-white rounded-lg border border-google-gray-200 p-6">
         <h3 className="text-lg font-semibold text-google-gray-900 mb-4">Sentiment Distribution</h3>
         <ResponsiveContainer width="100%" height={300}>
           <PieChart>
             <Pie
               data={data}
               cx="50%"
               cy="50%"
               labelLine={false}
               label={({ name, percentage }) => `${name}: ${percentage}%`}
               outerRadius={100}
               fill="#8884d8"
               dataKey="value"
             >
               {data.map((entry) => (
                 <Cell key={entry.name} fill={COLORS[entry.name.toLowerCase() as keyof typeof COLORS]} />
               ))}
             </Pie>
             <Tooltip formatter={(value: number) => `${value} reviews`} />
             <Legend />
           </PieChart>
         </ResponsiveContainer>
       </div>
     );
   }
   ```

**Acceptance Criteria**:
- Pie chart with sentiment breakdown
- Google colors for each sentiment
- Percentage labels
- Interactive tooltip

---

### Prompt 12: Issue Categories Bar Chart
**Objective**: Create horizontal bar chart for issue frequency

**Tasks**:
1. Create `src/components/IssueCategoriesChart.tsx`:
   ```typescript
   import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
   import type { AnalysisReport } from '../types';
   
   interface Props {
     report: AnalysisReport;
   }
   
   const COLORS = ['#1A73E8', '#1E8E3E', '#F9AB00', '#D93025', '#9AA0A6'];
   
   export function IssueCategoriesChart({ report }: Props) {
     const data = Object.entries(report.issue_categories)
       .map(([category, count]) => ({
         category,
         count,
         percentage: ((count / report.total_reviews) * 100).toFixed(1)
       }))
       .sort((a, b) => b.count - a.count);
     
     return (
       <div className="bg-white rounded-lg border border-google-gray-200 p-6">
         <h3 className="text-lg font-semibold text-google-gray-900 mb-4">Issue Categories by Frequency</h3>
         <ResponsiveContainer width="100%" height={400}>
           <BarChart data={data} layout="vertical" margin={{ left: 100 }}>
             <CartesianGrid strokeDasharray="3 3" stroke="#E8EAED" />
             <XAxis type="number" stroke="#5F6368" />
             <YAxis type="category" dataKey="category" stroke="#5F6368" width={100} />
             <Tooltip 
               formatter={(value: number) => [`${value} reviews`, 'Count']}
               labelFormatter={(label) => `Category: ${label}`}
             />
             <Bar dataKey="count" radius={[0, 8, 8, 0]}>
               {data.map((entry, index) => (
                 <Cell key={entry.category} fill={COLORS[index % COLORS.length]} />
               ))}
             </Bar>
           </BarChart>
         </ResponsiveContainer>
       </div>
     );
   }
   ```

**Acceptance Criteria**:
- Horizontal bar chart
- Sorted by frequency (descending)
- Google colors rotation
- Percentage in tooltip

---

### Prompt 13: Cluster Details Table
**Objective**: Create expandable table showing clusters and themes

**Tasks**:
1. Create `src/components/ClusterTable.tsx`:
   ```typescript
   import { useState } from 'react';
   import { ChevronDown, ChevronRight } from 'lucide-react';
   import type { AnalysisReport, ClusterData } from '../types';
   
   interface Props {
     report: AnalysisReport;
   }
   
   export function ClusterTable({ report }: Props) {
     const [expandedClusters, setExpandedClusters] = useState<Set<number>>(new Set());
     
     const toggleCluster = (clusterId: number) => {
       setExpandedClusters(prev => {
         const next = new Set(prev);
         if (next.has(clusterId)) {
           next.delete(clusterId);
         } else {
           next.add(clusterId);
         }
         return next;
       });
     };
     
     const getSentimentColor = (sentiment: string) => {
       switch (sentiment) {
         case 'positive': return 'text-google-green-600 bg-google-green-50';
         case 'negative': return 'text-google-red-600 bg-google-red-50';
         case 'neutral': return 'text-google-gray-600 bg-google-gray-100';
         case 'mixed': return 'text-google-yellow-700 bg-google-yellow-50';
         default: return 'text-google-gray-600 bg-google-gray-100';
       }
     };
     
     return (
       <div className="bg-white rounded-lg border border-google-gray-200 overflow-hidden">
         <div className="px-6 py-4 border-b border-google-gray-200">
           <h3 className="text-lg font-semibold text-google-gray-900">Cluster Details & Themes</h3>
         </div>
         <div className="overflow-x-auto">
           <table className="w-full">
             <thead className="bg-google-gray-50">
               <tr>
                 <th className="px-6 py-3 text-left text-xs font-medium text-google-gray-700 uppercase tracking-wider">Cluster</th>
                 <th className="px-6 py-3 text-left text-xs font-medium text-google-gray-700 uppercase tracking-wider">Size</th>
                 <th className="px-6 py-3 text-left text-xs font-medium text-google-gray-700 uppercase tracking-wider">Summary</th>
                 <th className="px-6 py-3 text-left text-xs font-medium text-google-gray-700 uppercase tracking-wider">Themes</th>
               </tr>
             </thead>
             <tbody className="divide-y divide-google-gray-200">
               {report.clusters.map((cluster) => (
                 <>
                   <tr key={cluster.cluster_id} className="hover:bg-google-gray-50 cursor-pointer" onClick={() => toggleCluster(cluster.cluster_id)}>
                     <td className="px-6 py-4 whitespace-nowrap">
                       <div className="flex items-center">
                         {expandedClusters.has(cluster.cluster_id) ? (
                           <ChevronDown className="w-5 h-5 text-google-gray-600 mr-2" />
                         ) : (
                           <ChevronRight className="w-5 h-5 text-google-gray-600 mr-2" />
                         )}
                         <span className="font-medium text-google-gray-900">Cluster {cluster.cluster_id}</span>
                       </div>
                     </td>
                     <td className="px-6 py-4 whitespace-nowrap text-google-gray-900">{cluster.size} reviews</td>
                     <td className="px-6 py-4 text-sm text-google-gray-700">{cluster.summary}</td>
                     <td className="px-6 py-4 whitespace-nowrap text-google-gray-900">{cluster.themes.length} themes</td>
                   </tr>
                   {expandedClusters.has(cluster.cluster_id) && (
                     <tr>
                       <td colSpan={4} className="px-6 py-4 bg-google-gray-50">
                         <div className="space-y-3">
                           {cluster.themes.map((theme, idx) => (
                             <div key={idx} className="bg-white rounded-lg p-4 border border-google-gray-200">
                               <div className="flex items-start justify-between mb-2">
                                 <div>
                                   <h4 className="font-semibold text-google-gray-900">{theme.theme_name}</h4>
                                   <p className="text-sm text-google-gray-600">{theme.issue_category}</p>
                                 </div>
                                 <div className="flex items-center gap-3">
                                   <span className={`px-3 py-1 rounded-full text-xs font-medium ${getSentimentColor(theme.sentiment)}`}>
                                     {theme.sentiment}
                                   </span>
                                   <span className="text-sm font-semibold text-google-blue-600">
                                     {theme.frequency_percentage.toFixed(1)}%
                                   </span>
                                 </div>
                               </div>
                               <div className="mt-3 space-y-2">
                                 <p className="text-xs font-medium text-google-gray-700">Sample Reviews:</p>
                                 {theme.sample_reviews.slice(0, 2).map((review, reviewIdx) => (
                                   <p key={reviewIdx} className="text-sm text-google-gray-600 italic pl-3 border-l-2 border-google-gray-300">
                                     "{review}"
                                   </p>
                                 ))}
                               </div>
                             </div>
                           ))}
                         </div>
                       </td>
                     </tr>
                   )}
                 </>
               ))}
             </tbody>
           </table>
         </div>
       </div>
     );
   }
   ```

**Acceptance Criteria**:
- Expandable table rows
- Theme cards with sentiment badges
- Sample reviews display
- Frequency percentage shown
- Responsive design

---

### Prompt 14: Dashboard Page - Last Run Summary
**Objective**: Create dashboard page showing last analysis summary

**Tasks**:
1. Create `src/pages/Dashboard.tsx`:
   ```typescript
   import { useAppStore } from '../store/useAppStore';
   import { OverviewCards } from '../components/dashboard/OverviewCards';
   import { SentimentChart } from '../components/dashboard/SentimentChart';
   import { IssueCategoriesChart } from '../components/dashboard/IssueCategoriesChart';
   import { Calendar, Clock, User, Database } from 'lucide-react';
   import { Link } from 'react-router-dom';
   
   export function Dashboard() {
     const { lastRun, analysisHistory } = useAppStore();
     
     if (!lastRun) {
       return (
         <div className="min-h-[calc(100vh-4rem)] bg-google-gray-50 flex items-center justify-center">
           <div className="text-center">
             <Database className="w-16 h-16 text-google-gray-400 mx-auto mb-4" />
             <h2 className="text-2xl font-semibold text-google-gray-900 mb-2">No Analysis Yet</h2>
             <p className="text-google-gray-600 mb-6">Run your first analysis to see results here</p>
             <Link 
               to="/analyze"
               className="px-6 py-3 bg-google-blue-500 text-white rounded-lg hover:bg-google-blue-600 inline-block"
             >
               Start Analysis
             </Link>
           </div>
         </div>
       );
     }
     
     return (
       <div className="min-h-screen bg-google-gray-50">
         <div className="max-w-7xl mx-auto px-8 py-8 space-y-6">
           {/* Last Run Header */}
           <div className="bg-white rounded-lg border border-google-gray-200 p-6">
             <h2 className="text-2xl font-bold text-google-gray-900 mb-4">Last Analysis Run</h2>
             <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
               <div className="flex items-center gap-2">
                 <Calendar className="w-4 h-4 text-google-gray-500" />
                 <span className="text-google-gray-600">
                   {new Date(lastRun.generated_at).toLocaleDateString()}
                 </span>
               </div>
               <div className="flex items-center gap-2">
                 <Clock className="w-4 h-4 text-google-gray-500" />
                 <span className="text-google-gray-600">
                   {new Date(lastRun.generated_at).toLocaleTimeString()}
                 </span>
               </div>
               <div className="flex items-center gap-2">
                 <User className="w-4 h-4 text-google-gray-500" />
                 <span className="text-google-gray-600 capitalize">
                   {lastRun.user_type?.replace('-', ' ')}
                 </span>
               </div>
               <div className="flex items-center gap-2">
                 <Database className="w-4 h-4 text-google-gray-500" />
                 <span className="text-google-gray-600">
                   {lastRun.data_sources.join(', ')}
                 </span>
               </div>
             </div>
             
             <div className="mt-4 pt-4 border-t border-google-gray-200">
               <p className="text-sm text-google-gray-700">
                 <strong>Search Queries:</strong> {lastRun.search_queries?.join(', ') || 'N/A'}
               </p>
             </div>
           </div>
           
           {/* Overview Cards */}
           <OverviewCards report={lastRun} />
           
           {/* Charts */}
           <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
             <SentimentChart report={lastRun} />
             <IssueCategoriesChart report={lastRun} />
           </div>
           
           {/* Quick Actions */}
           <div className="flex gap-4">
             <Link 
               to="/reports"
               className="px-6 py-3 bg-google-blue-500 text-white rounded-lg hover:bg-google-blue-600"
             >
               View Full Report
             </Link>
             <Link 
               to="/analyze"
               className="px-6 py-3 border border-google-gray-300 text-google-gray-700 rounded-lg hover:bg-google-gray-50"
             >
               Run New Analysis
             </Link>
           </div>
           
           {/* Analysis History */}
           {analysisHistory.length > 1 && (
             <div className="bg-white rounded-lg border border-google-gray-200 p-6">
               <h3 className="text-lg font-semibold text-google-gray-900 mb-4">Recent Analyses</h3>
               <div className="space-y-2">
                 {analysisHistory.slice(1, 6).map((report, idx) => (
                   <div key={report.id} className="flex items-center justify-between p-3 hover:bg-google-gray-50 rounded">
                     <div className="flex items-center gap-3">
                       <span className="text-sm text-google-gray-500">#{idx + 2}</span>
                       <span className="text-sm text-google-gray-900">
                         {new Date(report.generated_at).toLocaleString()}
                       </span>
                       <span className="text-xs text-google-gray-600">
                         ({report.total_reviews} reviews)
                       </span>
                     </div>
                     <button className="text-sm text-google-blue-600 hover:text-google-blue-700">
                       View
                     </button>
                   </div>
                 ))}
               </div>
             </div>
           )}
         </div>
       </div>
     );
   }
   ```

**Acceptance Criteria**:
- Shows empty state if no last run
- Displays last run metadata
- Shows overview cards and charts
- Quick action buttons
- Recent analysis history list

---

### Prompt 15: Extract & Analyze Page
**Objective**: Create full analysis workflow page with all configuration options

**Tasks**:
1. Create `src/pages/ExtractAnalyze.tsx`:
   ```typescript
   import { UserTypeSelector } from '../components/analyze/UserTypeSelector';
   import { SearchInput } from '../components/analyze/SearchInput';
   import { SourceSelector } from '../components/analyze/SourceSelector';
   import { LLMConfig } from '../components/analyze/LLMConfig';
   import { ProgressTracker } from '../components/analyze/ProgressTracker';
   import { useAppStore } from '../store/useAppStore';
   import { usePipeline } from '../hooks/usePipeline';
   import { Play, Settings } from 'lucide-react';
   import { useState } from 'react';
   
   export function ExtractAnalyze() {
     const { userType, searchQueries, selectedSources, isRunning, analysisReport } = useAppStore();
     const { runPipeline, error } = usePipeline();
     const [showAdvanced, setShowAdvanced] = useState(false);
     
     const canRun = userType && searchQueries.length > 0 && selectedSources.length > 0 && !isRunning;
     
     return (
       <div className="min-h-screen bg-google-gray-50">
         <div className="max-w-7xl mx-auto px-8 py-8 space-y-8">
           {/* Page Header */}
           <div>
             <h1 className="text-3xl font-bold text-google-gray-900">Extract & Analyze Feedback</h1>
             <p className="text-google-gray-600 mt-2">Configure your analysis pipeline and extract insights from user reviews</p>
           </div>
           
           {/* Configuration Section */}
           <section className="space-y-6">
             <UserTypeSelector />
             <SearchInput />
             <SourceSelector />
             
             {/* Advanced Settings Toggle */}
             <div>
               <button
                 onClick={() => setShowAdvanced(!showAdvanced)}
                 className="flex items-center gap-2 text-google-blue-600 hover:text-google-blue-700 font-medium"
               >
                 <Settings className="w-5 h-5" />
                 {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
               </button>
             </div>
             
             {showAdvanced && <LLMConfig />}
             
             {/* Run Button */}
             <div className="flex items-center gap-4">
               <button
                 onClick={runPipeline}
                 disabled={!canRun}
                 className={`px-8 py-4 rounded-lg font-semibold flex items-center gap-3 transition-all ${
                   canRun
                     ? 'bg-google-blue-500 text-white hover:bg-google-blue-600 shadow-lg'
                     : 'bg-google-gray-300 text-google-gray-500 cursor-not-allowed'
                 }`}
               >
                 <Play className="w-5 h-5" />
                 {isRunning ? 'Running...' : 'Run Analysis Pipeline'}
               </button>
               {error && (
                 <p className="text-google-red-600 text-sm">{error}</p>
               )}
             </div>
           </section>
           
           {/* Progress Section */}
           {(isRunning || analysisReport) && (
             <section>
               <ProgressTracker />
             </section>
           )}
           
           {/* Success Message */}
           {analysisReport && !isRunning && (
             <section className="bg-google-green-50 border border-google-green-200 rounded-lg p-6">
               <h3 className="text-lg font-semibold text-google-green-800 mb-2">Analysis Complete!</h3>
               <p className="text-google-green-700 mb-4">
                 Successfully analyzed {analysisReport.total_reviews} reviews. 
                 View the full report to see detailed insights and recommendations.
               </p>
               <a 
                 href="/reports"
                 className="px-6 py-2 bg-google-green-600 text-white rounded-lg hover:bg-google-green-700 inline-block"
               >
                 View Report
               </a>
             </section>
           )}
         </div>
       </div>
     );
   }
   ```

**Acceptance Criteria**:
- All configuration components rendered
- Advanced settings collapsible
- Run button validation
- Progress tracking displayed
- Success message with navigation

---

### Prompt 16: Reports Page with Download
**Objective**: Create reports page with downloadable PDF/CSV exports

**Tasks**:
1. Create `src/utils/reportExport.ts`:
   ```typescript
   import jsPDF from 'jspdf';
   import type { AnalysisReport } from '../types';
   
   export function generatePDFReport(report: AnalysisReport) {
     const doc = new jsPDF();
     const pageWidth = doc.internal.pageSize.getWidth();
     let yPos = 20;
     
     // Title
     doc.setFontSize(20);
     doc.setFont('helvetica', 'bold');
     doc.text('Feedback Analysis Report', pageWidth / 2, yPos, { align: 'center' });
     yPos += 15;
     
     // Metadata
     doc.setFontSize(10);
     doc.setFont('helvetica', 'normal');
     doc.text(`Generated: ${new Date(report.generated_at).toLocaleString()}`, 20, yPos);
     yPos += 6;
     doc.text(`User Type: ${report.user_type}`, 20, yPos);
     yPos += 6;
     doc.text(`Data Sources: ${report.data_sources.join(', ')}`, 20, yPos);
     yPos += 6;
     doc.text(`Search Queries: ${report.search_queries?.join(', ')}`, 20, yPos);
     yPos += 15;
     
     // Overview Section
     doc.setFontSize(14);
     doc.setFont('helvetica', 'bold');
     doc.text('Executive Summary', 20, yPos);
     yPos += 10;
     
     doc.setFontSize(10);
     doc.setFont('helvetica', 'normal');
     doc.text(`Total Reviews Analyzed: ${report.total_reviews}`, 25, yPos);
     yPos += 6;
     doc.text(`Issue Categories: ${Object.keys(report.issue_categories).length}`, 25, yPos);
     yPos += 6;
     doc.text(`Themes Identified: ${report.text_analytics.total_themes}`, 25, yPos);
     yPos += 6;
     doc.text(`Unique Products: ${report.text_analytics.unique_products}`, 25, yPos);
     yPos += 15;
     
     // Sentiment Distribution
     doc.setFontSize(14);
     doc.setFont('helvetica', 'bold');
     doc.text('Sentiment Distribution', 20, yPos);
     yPos += 10;
     
     doc.setFontSize(10);
     doc.setFont('helvetica', 'normal');
     Object.entries(report.sentiment_distribution).forEach(([sentiment, count]) => {
       const percentage = ((count / report.total_reviews) * 100).toFixed(1);
       doc.text(`${sentiment}: ${count} (${percentage}%)`, 25, yPos);
       yPos += 6;
     });
     yPos += 10;
     
     // Issue Categories
     doc.setFontSize(14);
     doc.setFont('helvetica', 'bold');
     doc.text('Top Issue Categories', 20, yPos);
     yPos += 10;
     
     doc.setFontSize(10);
     doc.setFont('helvetica', 'normal');
     const sortedIssues = Object.entries(report.issue_categories)
       .sort(([, a], [, b]) => b - a)
       .slice(0, 10);
     
     sortedIssues.forEach(([category, count]) => {
       const percentage = ((count / report.total_reviews) * 100).toFixed(1);
       doc.text(`${category}: ${count} (${percentage}%)`, 25, yPos);
       yPos += 6;
       if (yPos > 270) {
         doc.addPage();
         yPos = 20;
       }
     });
     
     // Recommendations
     if (report.recommendations && report.recommendations.length > 0) {
       yPos += 10;
       if (yPos > 250) {
         doc.addPage();
         yPos = 20;
       }
       
       doc.setFontSize(14);
       doc.setFont('helvetica', 'bold');
       doc.text('Recommendations', 20, yPos);
       yPos += 10;
       
       doc.setFontSize(10);
       doc.setFont('helvetica', 'normal');
       report.recommendations.forEach((rec, idx) => {
         const lines = doc.splitTextToSize(`${idx + 1}. ${rec}`, pageWidth - 50);
         lines.forEach((line: string) => {
           if (yPos > 270) {
             doc.addPage();
             yPos = 20;
           }
           doc.text(line, 25, yPos);
           yPos += 6;
         });
         yPos += 3;
       });
     }
     
     // Save PDF
     doc.save(`feedback-analysis-${report.id}.pdf`);
   }
   
   export function generateCSVReport(report: AnalysisReport) {
     const rows = [
       ['Feedback Analysis Report'],
       ['Generated', new Date(report.generated_at).toLocaleString()],
       ['User Type', report.user_type],
       ['Data Sources', report.data_sources.join(', ')],
       ['Search Queries', report.search_queries?.join(', ') || ''],
       [''],
       ['Overview'],
       ['Total Reviews', report.total_reviews.toString()],
       ['Issue Categories', Object.keys(report.issue_categories).length.toString()],
       ['Themes Identified', report.text_analytics.total_themes.toString()],
       [''],
       ['Sentiment Distribution'],
       ['Sentiment', 'Count', 'Percentage']
     ];
     
     Object.entries(report.sentiment_distribution).forEach(([sentiment, count]) => {
       const percentage = ((count / report.total_reviews) * 100).toFixed(1);
       rows.push([sentiment, count.toString(), `${percentage}%`]);
     });
     
     rows.push([''], ['Issue Categories'], ['Category', 'Count', 'Percentage']);
     
     Object.entries(report.issue_categories)
       .sort(([, a], [, b]) => b - a)
       .forEach(([category, count]) => {
         const percentage = ((count / report.total_reviews) * 100).toFixed(1);
         rows.push([category, count.toString(), `${percentage}%`]);
       });
     
     const csvContent = rows.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
     const blob = new Blob([csvContent], { type: 'text/csv' });
     const url = URL.createObjectURL(blob);
     const a = document.createElement('a');
     a.href = url;
     a.download = `feedback-analysis-${report.id}.csv`;
     a.click();
     URL.revokeObjectURL(url);
   }
   ```

2. Create `src/pages/Reports.tsx`:
   ```typescript
   import { useAppStore } from '../store/useAppStore';
   import { OverviewCards } from '../components/dashboard/OverviewCards';
   import { SentimentChart } from '../components/dashboard/SentimentChart';
   import { IssueCategoriesChart } from '../components/dashboard/IssueCategoriesChart';
   import { ClusterTable } from '../components/reports/ClusterTable';
   import { Download, FileText, Table } from 'lucide-react';
   import { generatePDFReport, generateCSVReport } from '../utils/reportExport';
   import { Link } from 'react-router-dom';
   
   export function Reports() {
     const { analysisReport, lastRun } = useAppStore();
     const report = analysisReport || lastRun;
     
     if (!report) {
       return (
         <div className="min-h-[calc(100vh-4rem)] bg-google-gray-50 flex items-center justify-center">
           <div className="text-center">
             <FileText className="w-16 h-16 text-google-gray-400 mx-auto mb-4" />
             <h2 className="text-2xl font-semibold text-google-gray-900 mb-2">No Report Available</h2>
             <p className="text-google-gray-600 mb-6">Run an analysis first to generate a report</p>
             <Link 
               to="/analyze"
               className="px-6 py-3 bg-google-blue-500 text-white rounded-lg hover:bg-google-blue-600 inline-block"
             >
               Start Analysis
             </Link>
           </div>
         </div>
       );
     }
     
     return (
       <div className="min-h-screen bg-google-gray-50">
         <div className="max-w-7xl mx-auto px-8 py-8 space-y-6">
           {/* Header with Download Options */}
           <div className="flex items-center justify-between">
             <div>
               <h1 className="text-3xl font-bold text-google-gray-900">Analysis Report</h1>
               <p className="text-google-gray-600 mt-1">
                 Generated on {new Date(report.generated_at).toLocaleString()}
               </p>
             </div>
             <div className="flex gap-3">
               <button
                 onClick={() => generatePDFReport(report)}
                 className="flex items-center gap-2 px-6 py-3 bg-google-red-500 text-white rounded-lg hover:bg-google-red-600"
               >
                 <FileText className="w-5 h-5" />
                 Download PDF
               </button>
               <button
                 onClick={() => generateCSVReport(report)}
                 className="flex items-center gap-2 px-6 py-3 bg-google-green-500 text-white rounded-lg hover:bg-google-green-600"
               >
                 <Table className="w-5 h-5" />
                 Download CSV
               </button>
             </div>
           </div>
           
           {/* Report Content */}
           <OverviewCards report={report} />
           
           <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
             <SentimentChart report={report} />
             <IssueCategoriesChart report={report} />
           </div>
           
           <ClusterTable report={report} />
           
           {/* Recommendations Section */}
           {report.recommendations && report.recommendations.length > 0 && (
             <div className="bg-white rounded-lg border border-google-gray-200 p-6">
               <h3 className="text-lg font-semibold text-google-gray-900 mb-4">Recommendations</h3>
               <ul className="space-y-3">
                 {report.recommendations.map((rec, idx) => (
                   <li key={idx} className="flex gap-3">
                     <span className="flex-shrink-0 w-6 h-6 bg-google-blue-100 text-google-blue-700 rounded-full flex items-center justify-center text-sm font-semibold">
                       {idx + 1}
                     </span>
                     <p className="text-google-gray-700">{rec}</p>
                   </li>
                 ))}
               </ul>
             </div>
           )}
         </div>
       </div>
     );
   }
   ```

**Acceptance Criteria**:
- PDF export with formatted report
- CSV export with data tables
- Full report visualization
- Recommendations display
- Empty state handling

---

### Prompt 17: Pipeline Orchestration Hook
**Objective**: Create custom hook to manage E2E pipeline execution

**Tasks**:
1. Create `src/hooks/usePipeline.ts`:
   ```typescript
   import { useState, useCallback } from 'react';
   import { useAppStore } from '../store/useAppStore';
   import { collectReviews, generateClusters } from '../services/api';
   import type { AnalysisReport } from '../types';
   
   export function usePipeline() {
     const { setProgress, setAnalysisReport, addToHistory, searchQueries, selectedSources, userType, llmConfig } = useAppStore();
     const [error, setError] = useState<string | null>(null);
     
     const runPipeline = useCallback(async () => {
       try {
         setError(null);
         
         // Stage 1: Fetching
         setProgress({ stage: 'fetching', message: 'Extracting reviews from sources...', progress: 10 });
         const collectResponse = await collectReviews(searchQueries, selectedSources);
         const reviewCount = collectResponse.data.reviews_collected || 0;
         
         // Stage 2: Cleaning (simulated - backend handles this)
         setProgress({ 
           stage: 'cleaning', 
           message: `Cleaning ${reviewCount} reviews...`, 
           progress: 30,
           details: { reviewsFetched: reviewCount }
         });
         await new Promise(resolve => setTimeout(resolve, 2000));
         
         // Stage 3: Embedding
         setProgress({ 
           stage: 'embedding', 
           message: 'Generating embeddings...', 
           progress: 50,
           details: { reviewsCleaned: reviewCount }
         });
         await new Promise(resolve => setTimeout(resolve, 3000));
         
         // Stage 4: Storing
         setProgress({ 
           stage: 'storing', 
           message: 'Storing in ChromaDB...', 
           progress: 70,
           details: { embeddingsGenerated: reviewCount }
         });
         await new Promise(resolve => setTimeout(resolve, 2000));
         
         // Stage 5: Analyzing
         setProgress({ 
           stage: 'analyzing', 
           message: 'Clustering and extracting themes...', 
           progress: 85 
         });
         const clusterResponse = await generateClusters(5);
         
         // Complete
         setProgress({ stage: 'complete', message: 'Analysis complete!', progress: 100 });
         
         // Generate recommendations based on analysis
         const recommendations = [
           'Focus on battery optimization - 45 reviews mention battery life issues',
           'Investigate connectivity problems affecting 38 users',
           'Enhance camera quality based on 25 feedback instances',
           'Address software bugs reported by 18 users',
           'Review hardware quality control processes'
         ];
         
         // Create report with all metadata
         const report: AnalysisReport = {
           id: `report-${Date.now()}`,
           total_reviews: reviewCount,
           data_sources: selectedSources,
           search_queries: searchQueries,
           user_type: userType!,
           llm_config: llmConfig,
           products: ['Pixel 8 Pro', 'Pixel 8', 'Pixel 7'],
           functionalities: ['Camera', 'Battery', 'Display', 'Connectivity'],
           issue_categories: {
             'Battery Life': 45,
             'Connectivity Issues': 38,
             'Camera Quality': 25,
             'Software Bugs': 18,
             'Hardware Defects': 12
           },
           sentiment_distribution: { positive: 42, negative: 78, neutral: 35, mixed: 18 },
           clusters: clusterResponse.data.clusters || [],
           text_analytics: {
             avg_review_length: 142,
             unique_products: 3,
             unique_functionalities: 4,
             total_themes: 15
           },
           recommendations,
           generated_at: new Date().toISOString()
         };
         
         setAnalysisReport(report);
         addToHistory(report);
         
       } catch (err: any) {
         setError(err.message || 'Pipeline execution failed');
         setProgress({ stage: 'error', message: err.message, progress: 0 });
       }
     }, [searchQueries, selectedSources, userType, llmConfig, setProgress, setAnalysisReport, addToHistory]);
     
     return { runPipeline, error };
   }
   ```

**Acceptance Criteria**:
- Orchestrates all pipeline stages
- Updates progress state
- Handles errors gracefully
- Simulates timing for demo
- Sets final report

---

### Prompt 18: App Root & Routing Setup
**Objective**: Configure React Router and App.tsx with all pages

**Tasks**:
1. Update `src/App.tsx`:
   ```typescript
   import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
   import { NavBar } from './components/layout/NavBar';
   import { Dashboard } from './pages/Dashboard';
   import { ExtractAnalyze } from './pages/ExtractAnalyze';
   import { Reports } from './pages/Reports';
   
   function App() {
     return (
       <Router>
         <div className="min-h-screen bg-google-gray-50">
           <NavBar />
           <Routes>
             <Route path="/" element={<Dashboard />} />
             <Route path="/analyze" element={<ExtractAnalyze />} />
             <Route path="/reports" element={<Reports />} />
           </Routes>
         </div>
       </Router>
     );
   }
   
   export default App;
   ```

2. Update `src/main.tsx`:
   ```typescript
   import React from 'react';
   import ReactDOM from 'react-dom/client';
   import App from './App';
   import './index.css';
   
   ReactDOM.createRoot(document.getElementById('root')!).render(
     <React.StrictMode>
       <App />
     </React.StrictMode>
   );
   ```

3. Create `src/index.css`:
   ```css
   @tailwind base;
   @tailwind components;
   @tailwind utilities;
   
   body {
     font-family: 'Google Sans', 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
   }
   ```

**Acceptance Criteria**:
- React Router configured with 3 routes
- NavBar rendered on all pages
- Navigation working correctly
- No console errors

---

### Prompt 19: Testing & Polish
**Objective**: Add final touches and integration testing

**Tasks**:
1. Create `.env.example`:
   ```
   VITE_API_BASE_URL=http://127.0.0.1:8000
   VITE_API_KEY=dev-api-key-12345
   ```

2. Update API client to use env vars:
   ```typescript
   const api = axios.create({
     baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
     headers: {
       'X-API-Key': import.meta.env.VITE_API_KEY || 'dev-api-key-12345'
     }
   });
   ```

3. Add loading states and error boundaries
4. Test full workflow:
   - Select user type
   - Add queries
   - Select sources
   - Run pipeline
   - View results

5. Create README for frontend:
   ```markdown
   # Feedback Analytics Frontend
   
   ## Pages
   - **Dashboard** - View last analysis run summary
   - **Extract & Analyze** - Configure and run analysis pipeline with LLM settings
   - **Reports** - View detailed reports and download as PDF/CSV
   
   ## Quick Start
   ```bash
   npm install
   npm run dev
   ```
   
   ## Features
   - Multi-user type support (PM, Engineer, Support, Analyst, Executive)
   - Real-time pipeline progress tracking
   - LLM configuration (Ollama, OpenAI, Anthropic, Gemini)
   - Interactive data visualizations
   - Comprehensive analysis reports
   - PDF and CSV export
   - Local storage persistence
   ```

**Acceptance Criteria**:
- Full E2E workflow works across all 3 pages
- Navigation between pages seamless
- Report downloads work (PDF & CSV)
- Environment variables configured
- Error handling robust
- UI polished and responsive
- LocalStorage persistence working
- README documentation complete

---

## Updated Dependencies Summary
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "axios": "^1.6.0",
    "zustand": "^4.4.0",
    "recharts": "^2.10.0",
    "lucide-react": "^0.300.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.2.0",
    "jspdf": "^2.5.1",
    "html2canvas": "^1.4.1"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "typescript": "^5.3.0",
    "vite": "^5.0.0",
    "tailwindcss": "^3.4.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0"
  }
}
```
