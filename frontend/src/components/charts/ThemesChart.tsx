import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { AnalysisReport } from '../../types';

interface Props {
  report: AnalysisReport;
}

const COLORS = ['#1A73E8', '#1E8E3E', '#F9AB00', '#D93025', '#9AA0A6'];

function normalizeTheme(theme?: string | null) {
  const value = (theme || '').trim();
  return value ? value : 'Unknown';
}

export function ThemesChart({ report }: Props) {
  const themeCounts: Record<string, number> = {};
  for (const t of report.themes) {
    const key = normalizeTheme(t.theme);
    themeCounts[key] = (themeCounts[key] || 0) + 1;
  }

  const totalThemes = Math.max(report.total_themes || 0, 1);
  const data = Object.entries(themeCounts)
    .map(([theme, count]) => ({
      theme,
      count,
      percentage: ((count / totalThemes) * 100).toFixed(1),
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);

  return (
    <div className="bg-white rounded-lg border border-google-gray-200 p-6">
      <h3 className="text-lg font-semibold text-google-gray-900 mb-4">Themes by Frequency</h3>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={data} layout="vertical" margin={{ left: 100 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E8EAED" />
          <XAxis type="number" stroke="#5F6368" />
          <YAxis type="category" dataKey="theme" stroke="#5F6368" width={120} />
          <Tooltip
            formatter={(value) => [`${value ?? 0} themes`, 'Count']}
            labelFormatter={(label) => `Theme: ${label}`}
          />
          <Bar dataKey="count" radius={[0, 8, 8, 0]}>
            {data.map((entry, index) => (
              <Cell key={entry.theme} fill={COLORS[index % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
