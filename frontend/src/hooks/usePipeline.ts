import { useState, useCallback } from 'react';
import { useAppStore } from '../store/useAppStore';
import { collectReviews, clusterReviews, analyzeFeedback } from '../services/api';
import type { AnalysisReport } from '../types';
import { deriveReport } from '../utils/report';

function normalizeCollectedReviews(reviews: Array<{ post_title: string; self_text: string } | string>): string[] {
  const texts: string[] = [];
  for (const r of reviews) {
    if (typeof r === 'string') {
      texts.push(r);
    } else if (r && typeof r === 'object') {
      const title = (r.post_title || '').trim();
      const body = (r.self_text || '').trim();
      const combined = [title, body].filter(Boolean).join('. ');
      if (combined) texts.push(combined);
    }
  }
  return texts;
}

function cleanText(s: string): string {
  return s.replace(/[\r\n]+/g, ' ').replace(/\s+/g, ' ').trim();
}

export function usePipeline() {
  const { 
    setProgress, 
    setLastRun, 
    addToHistory, 
    searchQueries, 
    selectedSources, 
    userType, 
    llmConfig 
  } = useAppStore();
  const [error, setError] = useState<string | null>(null);

  const runPipeline = useCallback(async () => {
    try {
      setError(null);

      // Stage 1: Fetching
      setProgress({ stage: 'fetching', message: 'Extracting reviews from sources...', progress: 10 });
      const collectResponse = await collectReviews({
        queries: searchQueries,
        sources: selectedSources,
      });

      const rawTexts = normalizeCollectedReviews(collectResponse.reviews);
      const reviewCount = rawTexts.length;

      // Stage 2: Cleaning
      setProgress({
        stage: 'cleaning',
        message: `Cleaning ${reviewCount} reviews...`,
        progress: 30,
        details: { reviewsFetched: reviewCount }
      });
      
      const cleaned = rawTexts.map(cleanText).filter(Boolean);
      const unique = Array.from(new Set(cleaned));
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Stage 3: Embedding
      setProgress({
        stage: 'embedding',
        message: 'Generating embeddings...',
        progress: 50,
        details: { reviewsCleaned: unique.length }
      });
      await new Promise(resolve => setTimeout(resolve, 1500));

      // Stage 4: Storing
      setProgress({
        stage: 'storing',
        message: 'Storing in ChromaDB...',
        progress: 70,
        details: { reviewsCleaned: unique.length }
      });
      
      const clusterResponse = await clusterReviews(unique, llmConfig);

      // Stage 5: Analyzing
      setProgress({
        stage: 'analyzing',
        message: 'Clustering and extracting themes...',
        progress: 85,
        details: { clustersCreated: clusterResponse.count }
      });

      const query = searchQueries.join(' | ');
      const analysisResponse = await analyzeFeedback({
        query,
        n_results: 50,
        user_type: userType,
        llm_config: llmConfig,
      });

      // Complete
      setProgress({ stage: 'complete', message: 'Analysis complete!', progress: 100 });

      // Create report with all metadata
      const report = deriveReport({
        userType: userType!,
        searchQueries,
        sources: selectedSources,
        llmConfig,
        totalReviews: unique.length,
        themes: analysisResponse.themes,
        clusters: clusterResponse,
      });

      // Calculate average review length
      const avgLen = unique.length 
        ? Math.round(unique.reduce((sum, t) => sum + t.length, 0) / unique.length) 
        : 0;
      report.text_analytics.avg_review_length = avgLen;

      setLastRun(report);
      addToHistory(report);

    } catch (err: any) {
      setError(err.message || 'Pipeline execution failed');
      setProgress({ stage: 'error', message: err.message || 'Pipeline execution failed', progress: 0 });
    }
  }, [searchQueries, selectedSources, userType, llmConfig, setProgress, setLastRun, addToHistory]);

  return { runPipeline, error };
}
