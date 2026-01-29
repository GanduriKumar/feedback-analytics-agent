import { FileText, Database, Tag, TrendingUp } from 'lucide-react';
import type { AnalysisReport } from '../../types';

interface Props {
  report: AnalysisReport;
}

export function OverviewCards({ report }: Props) {
  const cards = [
    {
      label: 'Total Reviews',
      value: report.total_reviews,
      icon: FileText,
      color: 'google-blue',
    },
    {
      label: 'Data Sources',
      value: report.data_sources.length,
      icon: Database,
      color: 'google-green',
    },
    {
      label: 'Issue Categories',
      value: Object.keys(report.issue_categories).length,
      icon: Tag,
      color: 'google-yellow',
    },
    {
      label: 'Themes Identified',
      value: report.total_themes,
      icon: TrendingUp,
      color: 'google-red',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {cards.map(({ label, value, icon: Icon, color }) => (
        <div
          key={label}
          className={`bg-${color}-50 border border-${color}-200 rounded-lg p-6`}
        >
          <div className="flex items-center justify-between mb-2">
            <Icon className={`w-8 h-8 text-${color}-600`} />
            <span className={`text-3xl font-bold text-${color}-700`}>{value}</span>
          </div>
          <p className="text-sm font-medium text-google-gray-700">{label}</p>
        </div>
      ))}
    </div>
  );
}
