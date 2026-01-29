import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import type { AnalysisReport } from '../../types';

interface Props {
  report: AnalysisReport;
}

const COLORS = {
  positive: '#1E8E3E',  // Google Green
  negative: '#D93025',  // Google Red
  neutral: '#9AA0A6',   // Google Gray
  mixed: '#F9AB00'      // Google Yellow
};

export function SentimentChart({ report }: Props) {
  const data = Object.entries(report.sentiment_distribution).map(([name, value]) => ({
    name: name.charAt(0).toUpperCase() + name.slice(1),
    value,
    percentage: ((value / report.total_reviews) * 100).toFixed(1)
  }));

  return (
    <div className="bg-white rounded-lg border border-google-gray-200 p-6">
      <h3 className="text-lg font-semibold text-google-gray-900 mb-4">Sentiment Distribution</h3>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percentage }) => `${name}: ${percentage}%`}
            outerRadius={100}
            fill="#8884d8"
            dataKey="value"
          >
            {data.map((entry) => (
              <Cell key={entry.name} fill={COLORS[entry.name.toLowerCase() as keyof typeof COLORS]} />
            ))}
          </Pie>
          <Tooltip formatter={(value: number) => `${value} reviews`} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
