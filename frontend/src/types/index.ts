export type UserType =
  | 'product-manager'
  | 'engineer'
  | 'support'
  | 'business-analyst'
  | 'executive';

export type DataSource = 'reddit' | 'twitter' | 'app-store' | 'play-store';

export type PipelineStage =
  | 'idle'
  | 'fetching'
  | 'cleaning'
  | 'embedding'
  | 'storing'
  | 'analyzing'
  | 'complete'
  | 'error';

export interface PipelineProgress {
  stage: PipelineStage;
  message: string;
  progress: number; // 0-100
  details?: {
    reviewsFetched?: number;
    reviewsCleaned?: number;
    clustersCreated?: number;
    themesExtracted?: number;
  };
}

export type LLMProvider = 'ollama' | 'openai' | 'anthropic' | 'gemini';

export interface LLMConfig {
  provider: LLMProvider;
  model: string;
  apiKey?: string;
  baseUrl?: string;
}

// Backend API shapes
export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
  capabilities: string[];
}

export interface RedditPost {
  post_title: string;
  self_text: string;
}

export type CollectedReview = RedditPost | string;

export interface CollectResponse {
  count: number;
  total: number;
  timestamp: string;
  reviews: CollectedReview[];
  // optional metadata (if backend provides)
  queries_used?: string[];
  sources_used?: string[];
  warnings?: string[];
}

export interface AnalyzeRequest {
  query: string;
  n_results?: number;
  user_type?: UserType;
  llm_config?: LLMConfig;
}

export interface SearchRequest {
  query: string;
  n_results?: number;
}

export interface ThemeData {
  product?: string | null;
  sentiment?: string | null;
  theme?: string | null;
  classification?: string | null;
  issue_description?: string | null;
}

export interface AnalyzeResponse {
  query: string;
  themes: ThemeData[];
  total_themes: number;
  timestamp: string;
  processing_time?: number;
}

export interface SearchResponse {
  query: string;
  results: string[];
  count: number;
  timestamp: string;
}

export interface ClusterResponse {
  // Backend returns `Dict[int, List[str]]`, which becomes `Record<string, string[]>` over JSON.
  clusters: Record<string, string[]>;
  count: number;
  time_taken: number;
  timestamp: string;
}

export interface CapabilitiesResponse {
  agent_name: string;
  version: string;
  capabilities: string[];
  endpoints: Record<
    string,
    {
      method: string;
      description: string;
      requires_auth: boolean;
    }
  >;
  authentication: {
    type: string;
    methods: string[];
  };
}

// Frontend report model (derived)
export interface AnalysisReport {
  id: string;
  generated_at: string;
  user_type: UserType;
  search_queries: string[];
  data_sources: DataSource[];
  llm_config: LLMConfig;

  total_reviews: number;
  total_themes: number;

  products: string[];
  functionalities: string[];

  // counts
  issue_categories: Record<string, number>;
  sentiment_distribution: Record<string, number>;

  // raw
  themes: ThemeData[];
  clusters?: ClusterResponse | null;

  recommendations: string[];
  text_analytics: {
    avg_review_length: number;
    unique_products: number;
    unique_functionalities: number;
    unique_issue_categories: number;
  };
}
