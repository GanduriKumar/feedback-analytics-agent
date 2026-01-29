import { FileText, Layers, Tag, TrendingUp } from 'lucide-react';
import type { AnalysisReport } from '../../types';

export function OverviewCards({ report }: { report: AnalysisReport }) {
  const cards = [
    {
      label: 'Reviews',
      value: report.total_reviews,
      icon: FileText,
      bg: 'bg-google-blue-50',
      border: 'border-google-blue-200',
      iconColor: 'text-google-blue-700',
    },
    {
      label: 'Themes',
      value: report.total_themes,
      icon: Layers,
      bg: 'bg-google-green-50',
      border: 'border-google-green-200',
      iconColor: 'text-google-green-700',
    },
    {
      label: 'Issue Categories',
      value: report.text_analytics.unique_issue_categories,
      icon: Tag,
      bg: 'bg-google-yellow-50',
      border: 'border-google-yellow-200',
      iconColor: 'text-google-yellow-700',
    },
    {
      label: 'Products',
      value: report.text_analytics.unique_products,
      icon: TrendingUp,
      bg: 'bg-google-red-50',
      border: 'border-google-red-200',
      iconColor: 'text-google-red-700',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
      {cards.map((c) => {
        const Icon = c.icon;
        return (
          <div key={c.label} className={[c.bg, c.border, 'border rounded-xl p-5 shadow-sm'].join(' ')}>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-google-gray-700">{c.label}</div>
                <div className="text-3xl font-semibold text-google-gray-900 mt-1">{c.value}</div>
              </div>
              <Icon className={['w-8 h-8', c.iconColor].join(' ')} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
