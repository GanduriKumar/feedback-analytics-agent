import { CheckCircle2, Circle, Loader2, XCircle } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';

const stages: Array<{ key: string; label: string; desc: string }> = [
  { key: 'fetching', label: 'Extract', desc: 'Collect reviews from selected sources' },
  { key: 'cleaning', label: 'Clean', desc: 'Normalize, dedupe, prep text' },
  { key: 'clustering', label: 'Cluster', desc: 'Group similar feedback' },
  { key: 'analyzing', label: 'Analyze', desc: 'Themes, sentiment, categories' },
];

function rank(stage: string) {
  const idx = stages.findIndex((s) => s.key === stage);
  return idx < 0 ? -1 : idx;
}

export function ProgressTracker() {
  const { progress, isRunning } = useAppStore();

  if (!isRunning && progress.stage === 'idle') return null;

  return (
    <section className="rounded-xl border border-google-gray-200 bg-white p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-base font-semibold text-google-gray-900">Workflow Progress</h3>
          <p className="text-sm text-google-gray-600">{progress.message || '—'}</p>
        </div>
        <div className="text-sm font-medium text-google-gray-700">{progress.progress}%</div>
      </div>

      <div className="w-full h-2 rounded-full bg-google-gray-200 overflow-hidden">
        <div className="h-2 bg-google-blue-600 transition-all" style={{ width: `${Math.min(100, Math.max(0, progress.progress))}%` }} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {stages.map((s) => {
          const current = rank(progress.stage);
          const me = rank(s.key);

          let icon = <Circle className="w-5 h-5 text-google-gray-300" />;
          if (progress.stage === 'error') icon = <XCircle className="w-5 h-5 text-google-red-600" />;
          else if (progress.stage === 'complete' || me < current) icon = <CheckCircle2 className="w-5 h-5 text-google-green-600" />;
          else if (me === current) icon = <Loader2 className="w-5 h-5 text-google-blue-600 animate-spin" />;

          return (
            <div key={s.key} className="flex items-start gap-3 rounded-lg border border-google-gray-200 p-4">
              <div className="mt-0.5">{icon}</div>
              <div>
                <div className="font-medium text-google-gray-900">{s.label}</div>
                <div className="text-xs text-google-gray-600">{s.desc}</div>
                {progress.details && me === current && (
                  <div className="mt-2 text-xs text-google-gray-700">
                    {progress.details.reviewsFetched != null && <div>Fetched: {progress.details.reviewsFetched}</div>}
                    {progress.details.reviewsCleaned != null && <div>Cleaned: {progress.details.reviewsCleaned}</div>}
                    {progress.details.clustersCreated != null && <div>Clusters: {progress.details.clustersCreated}</div>}
                    {progress.details.themesExtracted != null && <div>Themes: {progress.details.themesExtracted}</div>}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {progress.stage === 'error' && (
        <div className="rounded-lg border border-google-red-200 bg-google-red-50 p-3 text-sm text-google-red-700">
          {progress.message || 'Something went wrong.'}
        </div>
      )}
    </section>
  );
}
