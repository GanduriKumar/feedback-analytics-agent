import { useCallback } from 'react';
import { analyzeFeedback, clusterReviews, collectReviews } from '../services/api';
import { useAppStore } from '../store/useAppStore';
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
    userType,
    searchQueries,
    selectedSources,
    llmConfig,
    setProgress,
    setLastRun,
    addToHistory,
  } = useAppStore();

  const runPipeline = useCallback(async () => {
    if (!userType) {
      setProgress({ stage: 'error', message: 'Please select a user type.', progress: 0 });
      return;
    }
    if (!searchQueries.length) {
      setProgress({ stage: 'error', message: 'Please add at least one search query.', progress: 0 });
      return;
    }
    if (!selectedSources.length) {
      setProgress({ stage: 'error', message: 'Please select at least one source.', progress: 0 });
      return;
    }

    try {
      // 1) Collect
      setProgress({ stage: 'fetching', message: 'Collecting reviews from selected sources…', progress: 10 });
      const collected = await collectReviews({
        queries: searchQueries,
        sources: selectedSources,
      });

      const rawTexts = normalizeCollectedReviews(collected.reviews);
      setProgress({
        stage: 'cleaning',
        message: `Cleaning ${rawTexts.length} reviews…`,
        progress: 30,
        details: { reviewsFetched: rawTexts.length },
      });

      // 2) Clean + dedupe (client-side for now)
      const cleaned = rawTexts.map(cleanText).filter(Boolean);
      const unique = Array.from(new Set(cleaned));

      setProgress({
        stage: 'clustering',
        message: `Clustering ${unique.length} cleaned reviews…`,
        progress: 55,
        details: { reviewsFetched: rawTexts.length, reviewsCleaned: unique.length },
      });

      // 3) Cluster
      const clusters = await clusterReviews(unique);

      // 4) Analyze (vector-db based)
      setProgress({
        stage: 'analyzing',
        message: 'Running theme extraction and sentiment analysis…',
        progress: 80,
        details: { reviewsFetched: rawTexts.length, reviewsCleaned: unique.length, clustersCreated: clusters.count },
      });

      const query = searchQueries.join(' | ');
      const analysis = await analyzeFeedback({ query, n_results: 50 });

      // 5) Derive report
      const report = deriveReport({
        userType,
        searchQueries,
        sources: selectedSources,
        llmConfig,
        totalReviews: unique.length,
        themes: analysis.themes,
        clusters,
      });

      // patch avg length
      const avgLen = unique.length ? Math.round(unique.reduce((sum, t) => sum + t.length, 0) / unique.length) : 0;
      report.text_analytics.avg_review_length = avgLen;

      setLastRun(report);
      addToHistory(report);

      setProgress({
        stage: 'complete',
        message: 'Complete — report is ready.',
        progress: 100,
        details: {
          reviewsFetched: rawTexts.length,
          reviewsCleaned: unique.length,
          clustersCreated: clusters.count,
          themesExtracted: analysis.total_themes,
        },
      });
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e?.message || 'Pipeline failed.';
      setProgress({ stage: 'error', message: msg, progress: 0 });
    }
  }, [
    userType,
    searchQueries,
    selectedSources,
    llmConfig,
    setProgress,
    setLastRun,
    addToHistory,
  ]);

  return { runPipeline };
}
