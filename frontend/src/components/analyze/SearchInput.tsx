import { useMemo, useState } from 'react';
import { Plus, X, Search } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';

export function SearchInput() {
  const searchQueries = useAppStore((s) => s.searchQueries);
  const addSearchQuery = useAppStore((s) => s.addSearchQuery);
  const removeSearchQuery = useAppStore((s) => s.removeSearchQuery);
  const clearSearchQueries = useAppStore((s) => s.clearSearchQueries);
  const isRunning = useAppStore((s) => s.isRunning);
  const [value, setValue] = useState('');

  const parsedQueries = useMemo(() => {
    // Support comma-separated and/or multi-line input.
    // Example: "pixel battery, pixel overheating\nandroid auto disconnects"
    return value
      .split(/[\n,]/g)
      .map((s) => s.trim())
      .filter(Boolean);
  }, [value]);

  const canAdd = useMemo(() => parsedQueries.length > 0 && !isRunning, [parsedQueries.length, isRunning]);

  const add = () => {
    if (!parsedQueries.length) return;
    for (const q of parsedQueries) addSearchQuery(q);
    setValue('');
  };

  const remove = (q: string) => removeSearchQuery(q);

  return (
    <section className="space-y-3">
      <div>
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-google-gray-900">Search Queries</h2>
            <p className="text-sm text-google-gray-600">
              Enter one or more queries to extract and analyze reviews. Separate multiple queries with commas or new lines.
            </p>
          </div>

          {searchQueries.length > 0 && (
            <button
              type="button"
              onClick={clearSearchQueries}
              disabled={isRunning}
              className={[
                'text-sm font-medium',
                isRunning ? 'text-google-gray-400 cursor-not-allowed' : 'text-google-red-700 hover:text-google-red-800',
              ].join(' ')}
            >
              Clear all
            </button>
          )}
        </div>
      </div>

      <div className="flex gap-2">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-3 w-5 h-5 text-google-gray-500" />
          <textarea
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              // Enter adds; Shift+Enter inserts a newline.
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                add();
              }
            }}
            placeholder="e.g., Pixel connectivity issues, Android Auto disconnects\nBattery drain after update"
            rows={2}
            disabled={isRunning}
            className={[
              'w-full pl-10 pr-3 py-2.5 rounded-lg border bg-white focus:outline-none focus:ring-2 focus:ring-google-blue-500',
              isRunning ? 'border-google-gray-200 text-google-gray-500 bg-google-gray-50 cursor-not-allowed' : 'border-google-gray-300',
            ].join(' ')}
          />
          <div className="mt-1 text-xs text-google-gray-500">
            Press <span className="font-medium">Enter</span> to add. Use <span className="font-medium">Shift+Enter</span> for a new line.
          </div>
        </div>
        <button
          type="button"
          onClick={add}
          disabled={!canAdd}
          className={[
            'inline-flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium',
            canAdd ? 'bg-google-blue-600 text-white hover:bg-google-blue-700' : 'bg-google-gray-200 text-google-gray-500 cursor-not-allowed',
          ].join(' ')}
        >
          <Plus className="w-4 h-4" />
          Add
        </button>
      </div>

      {searchQueries.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {searchQueries.map((q) => (
            <div key={q} className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-google-blue-50 border border-google-blue-200">
              <span className="text-sm text-google-gray-900">{q}</span>
              <button
                type="button"
                onClick={() => remove(q)}
                disabled={isRunning}
                className={[
                  'text-google-gray-600',
                  isRunning ? 'cursor-not-allowed opacity-50' : 'hover:text-google-red-600',
                ].join(' ')}
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
