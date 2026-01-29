import axios from 'axios';
import type {
  AnalyzeRequest,
  AnalyzeResponse,
  CapabilitiesResponse,
  ClusterResponse,
  CollectResponse,
  HealthResponse,
  SearchRequest,
  SearchResponse,
} from '../types';

const baseURL = import.meta.env.VITE_API_BASE_URL || '/api';
const apiKey = import.meta.env.VITE_API_KEY || 'dev-api-key-12345';

export const api = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json',
    ...(apiKey ? { 'X-API-Key': apiKey } : {}),
  },
  timeout: 10 * 60 * 1000, // 10 minutes (collection/analysis can be slow)
});

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

export async function collectReviews(params?: { queries?: string[]; sources?: string[] }) {
  // Backend currently ignores query params unless implemented; safe to send.
  const { data } = await api.get<CollectResponse>('/tools/collect', {
    params: {
      queries: params?.queries?.join(',') || undefined,
      sources: params?.sources?.join(',') || undefined,
    },
  });
  return data;
}

export async function clusterReviews(reviews: string[]) {
  const { data } = await api.post<ClusterResponse>('/tools/cluster', reviews);
  return data;
}

export async function analyzeFeedback(body: AnalyzeRequest) {
  const { data } = await api.post<AnalyzeResponse>('/analyze', body);
  return data;
}

export async function searchReviews(body: SearchRequest) {
  const { data } = await api.post<SearchResponse>('/search', body);
  return data;
}
