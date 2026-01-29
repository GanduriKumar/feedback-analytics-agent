import { useAppStore } from '../../store/useAppStore';
import type { DataSource } from '../../types';

const sources: Array<{ value: DataSource; label: string; available: boolean }> = [
  { value: 'reddit', label: 'Reddit', available: true },
  { value: 'twitter', label: 'Twitter/X', available: false },
  { value: 'app-store', label: 'App Store', available: false },
  { value: 'play-store', label: 'Google Play Store', available: false },
];

export function SourceSelector() {
  const selectedSources = useAppStore((s) => s.selectedSources);
  const toggleSource = useAppStore((s) => s.toggleSource);

  return (
    <section className="space-y-3">
      <div>
        <h2 className="text-lg font-semibold text-google-gray-900">Data Sources</h2>
        <p className="text-sm text-google-gray-600">Select where reviews should be extracted from.</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {sources.map((s) => (
          <label
            key={s.value}
            className={[
              'flex items-center gap-3 p-4 border-2 rounded-lg transition-all',
              s.available ? 'cursor-pointer' : 'opacity-50 cursor-not-allowed',
              selectedSources.includes(s.value)
                ? 'border-google-blue-500 bg-google-blue-50'
                : 'border-google-gray-200 hover:border-google-gray-300',
            ].join(' ')}
          >
            <input
              type="checkbox"
              checked={selectedSources.includes(s.value)}
              onChange={() => toggleSource(s.value)}
              disabled={!s.available}
              className="w-5 h-5 text-google-blue-500 rounded focus:ring-google-blue-500"
            />
            <span className="font-medium text-google-gray-900">
              {s.label}
              {!s.available && <span className="text-xs text-google-gray-500 ml-2">(Coming Soon)</span>}
            </span>
          </label>
        ))}
      </div>
    </section>
  );
}
