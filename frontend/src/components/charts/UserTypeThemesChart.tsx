import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { AnalysisReport, UserType } from '../../types';

interface Props {
  report: AnalysisReport;
}

const COLORS = ['#1A73E8', '#1E8E3E', '#F9AB00', '#D93025', '#9AA0A6'];

const USER_LABELS: Record<UserType, string> = {
  'product-manager': 'Product Manager',
  engineer: 'Engineer',
  support: 'Support',
  'business-analyst': 'Business Analyst',
  executive: 'Executive',
};

function mapThemeToPersona(classification?: string | null, theme?: string | null): UserType {
  const text = `${classification || ''} ${theme || ''}`.toLowerCase();

  if (/(bug|crash|freeze|lag|slow|performance|stability|connectivity|bluetooth|wifi|network)/.test(text)) {
    return 'engineer';
  }
  if (/(billing|pricing|subscription|cost)/.test(text)) {
    return 'business-analyst';
  }
  if (/(support|onboarding|help|service|refund)/.test(text)) {
    return 'support';
  }
  if (/(design|ux|ui|usability|feature|roadmap)/.test(text)) {
    return 'product-manager';
  }

  return 'executive';
}

export function UserTypeThemesChart({ report }: Props) {
  const counts: Record<UserType, number> = {
    'product-manager': 0,
    engineer: 0,
    support: 0,
    'business-analyst': 0,
    executive: 0,
  };

  report.themes.forEach((t) => {
    const persona = mapThemeToPersona(t.classification, t.theme);
    counts[persona] += 1;
  });

  const data = (Object.keys(counts) as UserType[]).map((key) => ({
    persona: USER_LABELS[key],
    count: counts[key],
  }));

  return (
    <div className="bg-white rounded-lg border border-google-gray-200 p-6">
      <h3 className="text-base font-semibold text-google-gray-900 mb-4">Themes by Persona</h3>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={data} margin={{ left: 10, right: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E8EAED" />
          <XAxis dataKey="persona" stroke="#5F6368" tick={{ fontSize: 11 }} />
          <YAxis stroke="#5F6368" tick={{ fontSize: 11 }} />
          <Tooltip formatter={(value) => [`${value ?? 0} themes`, 'Count']} />
          <Bar dataKey="count" radius={[8, 8, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={entry.persona} fill={COLORS[index % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
