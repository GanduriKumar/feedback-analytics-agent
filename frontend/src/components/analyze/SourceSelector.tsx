import { useAppStore } from '../../store/useAppStore';
import type { DataSource } from '../../types';

const sources: Array<{ value: DataSource; label: string; enabled: boolean }> = [
  { value: 'reddit', label: 'Reddit', enabled: true },
  { value: 'twitter', label: 'Twitter/X (coming soon)', enabled: false },
  { value: 'app-store', label: 'App Store (coming soon)', enabled: false },
  { value: 'play-store', label: 'Google Play (coming soon)', enabled: false },
];

export function SourceSelector() {
  const selectedSources = useAppStore((s) => s.selectedSources);
  const toggleSource = useAppStore((s) => s.toggleSource);

  return (
    <section className="space-y-3">
      <div>
        <h2 className="text-lg font-semibold text-google-gray-900">Sources</h2>
        <p className="text-sm text-google-gray-600">Select where reviews should be extracted from.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
        {sources.map((s) => (
          <label
            key={s.value}
            className={[
              'flex items-center gap-3 rounded-xl border p-4 bg-white',
              s.enabled ? 'cursor-pointer hover:border-google-gray-300' : 'opacity-60 cursor-not-allowed',
              selectedSources.includes(s.value) ? 'border-google-blue-300 bg-google-blue-50' : 'border-google-gray-200',
            ].join(' ')}
          >
            <input
              type="checkbox"
              checked={selectedSources.includes(s.value)}
              onChange={() => toggleSource(s.value)}
              disabled={!s.enabled}
              className="h-4 w-4"
            />
            <span className="text-sm font-medium text-google-gray-900">{s.label}</span>
          </label>
        ))}
      </div>
    </section>
  );
}
