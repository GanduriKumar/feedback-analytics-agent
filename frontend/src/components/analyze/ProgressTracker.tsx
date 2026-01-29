import { CheckCircle, Circle, Loader, XCircle } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';

const stages: Array<{ key: string; label: string; description: string }> = [
  { key: 'fetching', label: 'Fetching Reviews', description: 'Extracting from data sources' },
  { key: 'cleaning', label: 'Cleaning Data', description: 'Removing duplicates & noise' },
  { key: 'embedding', label: 'Generating Embeddings', description: 'Creating vector representations' },
  { key: 'storing', label: 'Storing in VectorDB', description: 'Persisting to ChromaDB' },
  { key: 'analyzing', label: 'Analyzing Themes', description: 'Clustering & theme extraction' },
];

function getStageStatus(stageKey: string, currentStage: string) {
  const currentIndex = stages.findIndex((s) => s.key === currentStage);
  const stageIndex = stages.findIndex((s) => s.key === stageKey);

  if (currentStage === 'error') return 'error';
  if (currentStage === 'complete') return 'complete';
  if (stageIndex < currentIndex) return 'complete';
  if (stageIndex === currentIndex) return 'active';
  return 'pending';
}

export function ProgressTracker() {
  const { progress, isRunning } = useAppStore();

  if (!isRunning && progress.stage === 'idle') return null;

  const currentStage = progress.stage;

  return (
    <section className="bg-white rounded-lg border border-google-gray-200 p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-google-gray-900">Pipeline Progress</h2>
        <span className="text-sm text-google-gray-600">{progress.progress}%</span>
      </div>

      <div className="w-full bg-google-gray-200 rounded-full h-2">
        <div
          className="bg-google-blue-500 h-2 rounded-full transition-all duration-300"
          style={{ width: `${Math.min(100, Math.max(0, progress.progress))}%` }}
        />
      </div>

      <div className="space-y-4">
        {stages.map((stage) => {
          const status = getStageStatus(stage.key, currentStage);
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
                {status === 'active' && progress.message && (
                  <p className="text-sm text-google-blue-600 mt-1">{progress.message}</p>
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
