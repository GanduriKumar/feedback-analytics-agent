import { FileText, Database, Tag, TrendingUp } from 'lucide-react';
import type { AnalysisReport } from '../../types';

interface Props {
  report: AnalysisReport;
}

export function OverviewCards({ report }: Props) {
  const totalIssueMentions = Object.values(report.issue_categories).reduce((sum, n) => sum + n, 0);
  const totalThemeMentions = report.themes.reduce((sum, t) => sum + (t.review_count || 1), 0);

  const cards = [
    {
      label: 'Total Reviews',
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
      label: 'Issue Mentions',
      value: totalIssueMentions,
      icon: Tag,
      color: 'google-yellow',
      helper: 'Count of issue-category assignments across reviews',
    },
    {
      label: 'Theme Mentions',
      value: totalThemeMentions,
      icon: TrendingUp,
      color: 'google-red',
      helper: 'Review-weighted themes (reflects #reviews, not clusters)',
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
