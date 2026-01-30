import axios from 'axios';
import type {
  AnalyzeRequest,
  AnalyzeResponse,
  CapabilitiesResponse,
  ClusterResponse,
  ClusterRequest,
  CollectResponse,
  HealthResponse,
  SearchRequest,
  SearchResponse,
} from '../types';

const baseURL = import.meta.env.VITE_API_BASE_URL || '/api';
// Internal API key for frontend-backend communication (not a secret)
const apiKey = 'feedback-analytics-internal-key-2026';

export const api = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json',
    ...(apiKey ? { 'X-API-Key': apiKey } : {}),
  },
  timeout: 10 * 60 * 1000, // 10 minutes (collection/analysis can be slow)
});

// Request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    console.log('API Request:', {
      method: config.method,
      url: config.url,
      baseURL: config.baseURL,
      headers: config.headers,
    });
    return config;
  },
  (error) => {
    console.error('Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for debugging
api.interceptors.response.use(
  (response) => {
    console.log('API Response:', {
      status: response.status,
      url: response.config.url,
      data: response.data,
    });
    return response;
  },
  (error) => {
    console.error('API Error:', {
      message: error.message,
      response: error.response?.data,
      status: error.response?.status,
      url: error.config?.url,
    });
    return Promise.reject(error);
  }
);

function getDetailFromBody(body: unknown): string | undefined {
  if (!body || typeof body !== 'object') return undefined;
  const maybeDetail = (body as any).detail;
  if (typeof maybeDetail === 'string' && maybeDetail.trim()) return maybeDetail;
  return undefined;
}

export function getApiErrorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const detail = getDetailFromBody(err.response?.data);
    if (detail) return detail;

    const status = err.response?.status;
    if (status === 401) return 'Unauthorized — missing or invalid API key.';
    if (status === 429) return 'Rate limit exceeded — please try again in a minute.';
    if (status && status >= 500) return 'Server error — please try again shortly.';

    return err.message || 'Request failed.';
  }

  if (err instanceof Error) return err.message;
  return 'Unexpected error.';
}

export async function healthCheck() {
  const { data } = await api.get<HealthResponse>('/health');
  return data;
}

export async function getCapabilities() {
  const { data } = await api.get<CapabilitiesResponse>('/capabilities');
  return data;
}

export async function collectReviews(params?: { queries?: string[]; sources?: string[]; time_filter?: string; signal?: AbortSignal }) {
  const { data } = await api.get<CollectResponse>('/tools/collect', {
    params: {
      queries: params?.queries?.join(',') || undefined,
      sources: params?.sources?.join(',') || undefined,
      time_filter: params?.time_filter || undefined,
    },
    signal: params?.signal,
  });
  return data;
}

export async function clusterReviews(reviews: string[], llmConfig?: ClusterRequest['llm_config'], signal?: AbortSignal) {
  const payload: ClusterRequest = { reviews, llm_config: llmConfig };
  const { data } = await api.post<ClusterResponse>('/tools/cluster', payload, { signal });
  return data;
}

export async function analyzeFeedback(body: AnalyzeRequest, signal?: AbortSignal) {
  const { data } = await api.post<AnalyzeResponse>('/analyze', body, { signal });
  return data;
}

export async function searchReviews(body: SearchRequest) {
  const { data } = await api.post<SearchResponse>('/search', body);
  return data;
}

export async function purgeStorage() {
  const { data } = await api.post<{ status: string; message: string }>('/tools/purge');
  return data;
}
