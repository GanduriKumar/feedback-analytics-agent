import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import type { AnalysisReport } from '../../types';

const COLORS: Record<string, string> = {
  positive: '#1E8E3E',
  negative: '#D93025',
  neutral: '#9AA0A6',
  mixed: '#F9AB00',
  unknown: '#80868B',
};

export function SentimentChart({ report }: { report: AnalysisReport }) {
  const data = Object.entries(report.sentiment_distribution).map(([name, value]) => ({
    name,
    value,
  }));

  return (
    <div className="rounded-xl border border-google-gray-200 bg-white p-5 shadow-sm">
      <div className="font-semibold text-google-gray-900">Sentiment Distribution</div>
      <div className="mt-4 h-[280px]">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie data={data} dataKey="value" nameKey="name" outerRadius={90} label>
              {data.map((d) => (
                <Cell key={d.name} fill={COLORS[d.name] || '#5F6368'} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
