import axios from 'axios';
import type {
  AnalyzeRequest,
  AnalyzeResponse,
  ClusterResponse,
  CollectResponse,
  HealthResponse,
} from '../types';

const baseURL = import.meta.env.VITE_API_BASE_URL || '/api';
const apiKey = import.meta.env.VITE_API_KEY;

export const api = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json',
    ...(apiKey ? { 'X-API-Key': apiKey } : {}),
  },
  timeout: 10 * 60 * 1000, // 10 minutes (collection/analysis can be slow)
});

export async function healthCheck() {
  const { data } = await api.get<HealthResponse>('/health');
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
