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

  const renderLabel = ({
    cx,
    cy,
    midAngle,
    innerRadius,
    outerRadius,
    percent,
    name,
  }: {
    cx?: number;
    cy?: number;
    midAngle?: number;
    innerRadius?: number;
    outerRadius?: number;
    percent?: number;
    name?: string;
  }) => {
    if (cx == null || cy == null || midAngle == null || innerRadius == null || outerRadius == null || percent == null) {
      return null;
    }
    const radius = innerRadius + (outerRadius - innerRadius) * 0.7;
    const radian = Math.PI / 180;
    const x = cx + radius * Math.cos(-midAngle * radian);
    const y = cy + radius * Math.sin(-midAngle * radian);
    return (
      <text
        x={x}
        y={y}
        fill="#5F6368"
        textAnchor={x > cx ? 'start' : 'end'}
        dominantBaseline="central"
        fontSize={11}
      >
        {`${name}: ${(percent * 100).toFixed(1)}%`}
      </text>
    );
  };

  return (
    <div className="bg-white rounded-lg border border-google-gray-200 p-6">
      <h3 className="text-base font-semibold text-google-gray-900 mb-4">Sentiment Distribution</h3>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={renderLabel}
            outerRadius={100}
            fill="#8884d8"
            dataKey="value"
          >
            {data.map((entry) => (
              <Cell key={entry.name} fill={COLORS[entry.name.toLowerCase() as keyof typeof COLORS]} />
            ))}
          </Pie>
          <Tooltip formatter={(value) => `${value ?? 0} reviews`} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
