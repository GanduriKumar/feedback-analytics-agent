import { FileText, Database, Tag, TrendingUp } from 'lucide-react';
import type { AnalysisReport } from '../../types';

interface Props {
  report: AnalysisReport;
}

export function OverviewCards({ report }: Props) {
  const uniqueIssueCategories = Object.keys(report.issue_categories).length;
  const uniqueThemes = report.themes.length;

  const cards = [
    {
      label: 'Reviews Analyzed',
      value: report.total_reviews,
      icon: FileText,
      color: 'google-blue',
      helper: 'All reviews analyzed after cleaning/deduping',
    },
    {
      label: 'Data Sources',
      value: report.data_sources.length,
      icon: Database,
      color: 'google-green',
      helper: 'How many sources were used (e.g., Reddit)',
    },
    {
      label: 'Issue Categories',
      value: uniqueIssueCategories,
      icon: Tag,
      color: 'google-yellow',
      helper: 'Unique issue classification types',
    },
    {
      label: 'Themes Extracted',
      value: uniqueThemes,
      icon: TrendingUp,
      color: 'google-red',
      helper: 'Unique themes identified across reviews',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {cards.map(({ label, value, icon: Icon, color, helper }) => (
        <div
          key={label}
          className={`bg-${color}-50 border border-${color}-200 rounded-lg p-6`}
        >
          <div className="flex items-center justify-between mb-2">
            <Icon className={`w-8 h-8 text-${color}-600`} />
            <span className={`text-3xl font-bold text-${color}-700`}>{value}</span>
          </div>
          <p className="text-sm font-medium text-google-gray-700">{label}</p>
          {helper && <p className="text-xs text-google-gray-600 mt-1 leading-snug">{helper}</p>}
        </div>
      ))}
    </div>
  );
}
