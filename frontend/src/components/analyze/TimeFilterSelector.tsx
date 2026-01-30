import { Clock } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import type { TimeFilter } from '../../types';

const options: { value: TimeFilter; label: string; helper: string }[] = [
  { value: 'hour', label: 'Past Hour', helper: 'Most recent posts' },
  { value: 'day', label: 'Past Day', helper: 'Last 24 hours' },
  { value: 'week', label: 'Past Week', helper: 'Trailing 7 days' },
  { value: 'month', label: 'Past Month', helper: 'Approx. 30 days' },
  { value: 'year', label: 'Past Year', helper: 'Last 12 months' },
  { value: 'all', label: 'All Time', helper: 'No time filter' },
];

export function TimeFilterSelector() {
  const { timeFilter, setTimeFilter } = useAppStore();

  return (
    <div className="bg-white border border-google-gray-200 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-3">
        <Clock className="w-5 h-5 text-google-gray-600" />
        <div>
          <h3 className="font-semibold text-google-gray-900">Timeline</h3>
          <p className="text-sm text-google-gray-600">Choose the Reddit time window to search.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
        {options.map((opt) => {
          const active = timeFilter === opt.value;
          return (
            <button
              key={opt.value}
              type="button"
              onClick={() => setTimeFilter(opt.value)}
              className={`text-left border rounded-lg p-3 transition-all focus:outline-none focus:ring-2 focus:ring-google-blue-500 focus:ring-offset-2 ${
                active
                  ? 'border-google-blue-400 bg-google-blue-50 text-google-blue-700 shadow-sm'
                  : 'border-google-gray-200 hover:border-google-blue-200 text-google-gray-800'
              }`}
            >
              <div className="font-medium">{opt.label}</div>
              <div className="text-xs text-google-gray-600">{opt.helper}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
