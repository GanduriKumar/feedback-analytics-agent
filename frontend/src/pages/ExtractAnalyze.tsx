import { useEffect, useState } from 'react';
import { Play, Settings } from 'lucide-react';
import { UserTypeSelector } from '../components/analyze/UserTypeSelector';
import { SearchInput } from '../components/analyze/SearchInput';
import { SourceSelector } from '../components/analyze/SourceSelector';
import { TimeFilterSelector } from '../components/analyze/TimeFilterSelector';
import { LLMConfig } from '../components/analyze/LLMConfig';
import { ProgressTracker } from '../components/analyze/ProgressTracker';
import { useAppStore } from '../store/useAppStore';
import { usePipeline } from '../hooks/usePipeline';
import { healthCheck } from '../services/api';

export function ExtractAnalyze() {
  const { userType, searchQueries, selectedSources, isRunning, lastRun } = useAppStore();
  const { runPipeline, pausePipeline, resumePipeline, abortPipeline, error, isPaused } = usePipeline();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'starting' | 'up'>('checking');
  const [backendMessage, setBackendMessage] = useState('Checking backend status...');

  useEffect(() => {
    let isActive = true;
    let timeoutId: number | undefined;

    const checkBackend = async () => {
      try {
        const health = await healthCheck();
        if (!isActive) return;
        setBackendStatus('up');
        setBackendMessage(`Backend online (v${health.version})`);
      } catch (err) {
        if (!isActive) return;
        setBackendStatus('starting');
        setBackendMessage('Backend starting up...');
        timeoutId = window.setTimeout(checkBackend, 3000);
      }
    };

    checkBackend();

    return () => {
      isActive = false;
      if (timeoutId) window.clearTimeout(timeoutId);
    };
  }, []);
  const canRun = userType && searchQueries.length > 0 && selectedSources.length > 0 && !isRunning && backendStatus === 'up';

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
          <TimeFilterSelector />

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
            {isRunning && (
              <div className="flex items-center gap-2">
                <button
                  onClick={isPaused ? resumePipeline : pausePipeline}
                  className={`px-4 py-3 rounded-lg text-sm font-semibold border ${
                    isPaused
                      ? 'border-google-green-500 text-google-green-700 bg-google-green-50 hover:bg-google-green-100'
                      : 'border-google-amber-500 text-google-amber-700 bg-google-amber-50 hover:bg-google-amber-100'
                  }`}
                >
                  {isPaused ? 'Resume' : 'Pause'}
                </button>
                <button
                  onClick={abortPipeline}
                  className="px-4 py-3 rounded-lg text-sm font-semibold border border-google-red-500 text-google-red-700 bg-google-red-50 hover:bg-google-red-100"
                >
                  Abort
                </button>
              </div>
            )}
            <div className="flex items-center gap-2 text-sm">
              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  backendStatus === 'up' ? 'bg-google-green-600' : 'bg-google-yellow-600'
                }`}
                aria-hidden
              />
              <span className={backendStatus === 'up' ? 'text-google-green-700' : 'text-google-gray-600'}>
                {backendMessage}
              </span>
            </div>
            {error && (
              <p className="text-google-red-600 text-sm">{error}</p>
            )}
          </div>
        </section>

        {/* Progress Section */}
        {(isRunning || lastRun) && (
          <section>
            <ProgressTracker />
          </section>
        )}

        {/* Success Message */}
        {lastRun && !isRunning && (
          <section className="bg-google-green-50 border border-google-green-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-google-green-800 mb-2">Analysis Complete!</h3>
            <p className="text-google-green-700 mb-4">
              Successfully analyzed {lastRun.total_reviews} reviews.
              View the full report to see detailed insights and recommendations.
            </p>
            <a
              href="/"
              className="px-6 py-2 bg-google-green-600 text-white rounded-lg hover:bg-google-green-700 inline-block"
            >
              View Dashboard
            </a>
          </section>
        )}
      </div>
    </div>
  );
}
