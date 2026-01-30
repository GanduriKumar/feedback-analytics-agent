import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { AnalysisReport } from '../../types';

interface Props {
  report: AnalysisReport;
}

const COLORS = ['#1A73E8', '#1E8E3E', '#F9AB00', '#D93025', '#9AA0A6'];

function normalizeTheme(theme?: string | null) {
  const raw = (theme || '').trim();
  if (!raw) return 'General';
  const cleaned = raw
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();

  if (!cleaned || cleaned === 'unknown' || cleaned === 'unclassified') return 'General';

  const mapping: Record<string, string> = {
    connectivity: 'Connectivity',
    bluetooth: 'Connectivity',
    wifi: 'Connectivity',
    network: 'Connectivity',
    battery: 'Battery',
    charging: 'Battery',
    power: 'Battery',
    camera: 'Camera',
    display: 'Display',
    screen: 'Display',
    performance: 'Performance',
    stability: 'Stability',
    crash: 'Stability',
    freeze: 'Stability',
    audio: 'Audio',
    speaker: 'Audio',
    mic: 'Audio',
    update: 'Update',
    pricing: 'Pricing',
    price: 'Pricing',
    cost: 'Pricing',
    design: 'Design',
    ux: 'UX',
    ui: 'UX',
    usability: 'UX',
    support: 'Support',
    'customer service': 'Support',
  };

  for (const key of Object.keys(mapping)) {
    if (cleaned.includes(key)) return mapping[key];
  }

  return cleaned
    .split(' ')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
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
      <h3 className="text-base font-semibold text-google-gray-900 mb-4">Themes by Frequency</h3>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={data} layout="vertical" margin={{ left: 100 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E8EAED" />
          <XAxis type="number" stroke="#5F6368" tick={{ fontSize: 11 }} />
          <YAxis type="category" dataKey="theme" stroke="#5F6368" width={120} tick={{ fontSize: 11 }} />
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
