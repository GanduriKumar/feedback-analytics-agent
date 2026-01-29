import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { AnalysisReport } from '../../types';
import { labelColorClass } from '../../utils/labelColor';

export function IssueCategoriesChart({ report }: { report: AnalysisReport }) {
  const data = Object.entries(report.issue_categories)
    .map(([category, count]) => ({ category, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 12);

  return (
    <div className="rounded-xl border border-google-gray-200 bg-white p-5 shadow-sm">
      <div className={['font-semibold', labelColorClass('Top Issue Categories')].join(' ')}>Top Issue Categories</div>
      <div className="mt-4 h-[280px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 90 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E8EAED" />
            <XAxis type="number" stroke="#5F6368" />
            <YAxis type="category" dataKey="category" stroke="#5F6368" width={90} />
            <Tooltip />
            <Bar dataKey="count" fill="#1A73E8" radius={[0, 8, 8, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
