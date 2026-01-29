import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Play, Settings2 } from 'lucide-react';
import { UserTypeSelector } from '../components/analyze/UserTypeSelector';
import { SearchInput } from '../components/analyze/SearchInput';
import { SourceSelector } from '../components/analyze/SourceSelector';
import { LLMConfig } from '../components/analyze/LLMConfig';
import { ProgressTracker } from '../components/analyze/ProgressTracker';
import { usePipeline } from '../hooks/usePipeline';
import { useAppStore } from '../store/useAppStore';

export function ExtractAnalyze() {
  const { isRunning, progress, userType, searchQueries, selectedSources, lastRun } = useAppStore();
  const { runPipeline } = usePipeline();
  const [advanced, setAdvanced] = useState(false);

  const canRun = !!userType && searchQueries.length > 0 && selectedSources.length > 0 && !isRunning;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      <div>
        <h1 className="text-3xl font-semibold text-google-gray-900">Extract & Analyze</h1>
        <p className="text-google-gray-600 mt-1">Configure inputs, run the E2E workflow, and watch progress in real time.</p>
      </div>

      <UserTypeSelector />
      <SearchInput />
      <SourceSelector />

      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={() => setAdvanced((v) => !v)}
          className="inline-flex items-center gap-2 text-google-blue-700 hover:text-google-blue-800"
        >
          <Settings2 className="w-4 h-4" />
          {advanced ? 'Hide' : 'Show'} advanced settings
        </button>

        <button
          type="button"
          onClick={runPipeline}
          disabled={!canRun}
          className={[
            'inline-flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium',
            canRun ? 'bg-google-blue-600 text-white hover:bg-google-blue-700' : 'bg-google-gray-200 text-google-gray-500 cursor-not-allowed',
          ].join(' ')}
        >
          <Play className="w-4 h-4" />
          {isRunning ? 'Running…' : 'Run workflow'}
        </button>
      </div>

      {advanced && <LLMConfig />}

      <ProgressTracker />

      {progress.stage === 'complete' && lastRun && (
        <div className="rounded-xl border border-google-green-200 bg-google-green-50 p-5">
          <div className="font-semibold text-google-green-900">Complete</div>
          <div className="text-sm text-google-green-800 mt-1">Your report is ready.</div>
          <div className="mt-4 flex gap-2">
            <Link to="/reports" className="px-4 py-2 rounded-lg bg-google-green-700 text-white hover:bg-google-green-800">
              Open report
            </Link>
            <Link to="/" className="px-4 py-2 rounded-lg border border-google-green-300 text-google-green-900 hover:bg-google-green-100">
              Back to dashboard
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
