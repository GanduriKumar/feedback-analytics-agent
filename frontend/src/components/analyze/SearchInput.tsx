import { useMemo, useState } from 'react';
import { Plus, X, Search } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';

export function SearchInput() {
  const { searchQueries, setSearchQueries } = useAppStore();
  const [value, setValue] = useState('');

  const canAdd = useMemo(() => value.trim().length > 0, [value]);

  const add = () => {
    const v = value.trim();
    if (!v) return;
    setSearchQueries([...searchQueries, v]);
    setValue('');
  };

  const remove = (idx: number) => {
    setSearchQueries(searchQueries.filter((_, i) => i !== idx));
  };

  return (
    <section className="space-y-3">
      <div>
        <h2 className="text-lg font-semibold text-google-gray-900">Search Strings</h2>
        <p className="text-sm text-google-gray-600">Enter one or more search strings for review extraction and analysis.</p>
      </div>

      <div className="flex gap-2">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-3 w-5 h-5 text-google-gray-500" />
          <input
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                add();
              }
            }}
            placeholder="e.g., Pixel connectivity issues"
            className="w-full pl-10 pr-3 py-2.5 rounded-lg border border-google-gray-300 bg-white focus:outline-none focus:ring-2 focus:ring-google-blue-500"
          />
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
          {searchQueries.map((q, idx) => (
            <div key={`${q}-${idx}`} className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-google-blue-50 border border-google-blue-200">
              <span className="text-sm text-google-gray-900">{q}</span>
              <button type="button" onClick={() => remove(idx)} className="text-google-gray-600 hover:text-google-red-600">
                <X className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
