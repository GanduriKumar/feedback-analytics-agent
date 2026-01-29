import type { ThemeData } from '../../types';
import { labelColorClass } from '../../utils/labelColor';

function pillColor(sentiment?: string | null) {
  switch ((sentiment || '').toLowerCase()) {
    case 'positive':
      return 'bg-google-green-50 border-google-green-200 text-google-green-800';
    case 'negative':
      return 'bg-google-red-50 border-google-red-200 text-google-red-800';
    case 'neutral':
      return 'bg-google-gray-100 border-google-gray-200 text-google-gray-800';
    case 'mixed':
      return 'bg-google-yellow-50 border-google-yellow-200 text-google-yellow-800';
    default:
      return 'bg-google-gray-100 border-google-gray-200 text-google-gray-800';
  }
}

export function ThemesTable({ themes }: { themes: ThemeData[] }) {
  return (
    <div className="rounded-xl border border-google-gray-200 bg-white overflow-hidden shadow-sm">
      <div className="px-5 py-4 border-b border-google-gray-200">
        <div className={['font-semibold', labelColorClass('Extracted Themes')].join(' ')}>Extracted Themes</div>
        <div className="text-sm text-google-gray-600">Themes extracted from clustered summaries.</div>
      </div>
      <div className="overflow-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-google-gray-50 text-google-gray-700">
            <tr>
              <th className="text-left px-4 py-3 font-medium">Product</th>
              <th className="text-left px-4 py-3 font-medium">Theme</th>
              <th className="text-left px-4 py-3 font-medium">Issue Category</th>
              <th className="text-left px-4 py-3 font-medium">Sentiment</th>
              <th className="text-left px-4 py-3 font-medium">Description</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-google-gray-200">
            {themes.map((t, idx) => (
              <tr key={idx} className="hover:bg-google-gray-50">
                <td className="px-4 py-3 text-google-gray-900">{t.product || '—'}</td>
                <td className="px-4 py-3 text-google-gray-900">{t.theme || '—'}</td>
                <td className="px-4 py-3 text-google-gray-900">{t.classification || 'Unclassified'}</td>
                <td className="px-4 py-3">
                  <span className={['inline-flex items-center px-2.5 py-1 rounded-full border text-xs font-medium', pillColor(t.sentiment)].join(' ')}>
                    {t.sentiment || 'unknown'}
                  </span>
                </td>
                <td className="px-4 py-3 text-google-gray-700 max-w-[520px]">
                  {t.issue_description || '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
