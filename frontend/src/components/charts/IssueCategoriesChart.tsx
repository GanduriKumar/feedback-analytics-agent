import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { AnalysisReport } from '../../types';

interface Props {
  report: AnalysisReport;
}

const COLORS = ['#1A73E8', '#1E8E3E', '#F9AB00', '#D93025', '#9AA0A6'];

export function IssueCategoriesChart({ report }: Props) {
  const data = Object.entries(report.issue_categories)
    .map(([category, count]) => ({
      category,
      count,
      percentage: ((count / report.total_reviews) * 100).toFixed(1)
    }))
    .sort((a, b) => b.count - a.count);

  return (
    <div className="bg-white rounded-lg border border-google-gray-200 p-6">
      <h3 className="text-base font-semibold text-google-gray-900 mb-4">Issue Categories by Frequency</h3>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={data} layout="vertical" margin={{ left: 100 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E8EAED" />
          <XAxis type="number" stroke="#5F6368" tick={{ fontSize: 11 }} />
          <YAxis type="category" dataKey="category" stroke="#5F6368" width={100} tick={{ fontSize: 11 }} />
          <Tooltip 
            formatter={(value) => [`${value ?? 0} reviews`, 'Count']}
            labelFormatter={(label) => `Category: ${label}`}
          />
          <Bar dataKey="count" radius={[0, 8, 8, 0]}>
            {data.map((entry, index) => (
              <Cell key={entry.category} fill={COLORS[index % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
