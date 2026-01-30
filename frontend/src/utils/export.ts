import type { AnalysisReport, ThemeData } from '../types';
import { getUniqueThemeCount } from './report';

type UserType = AnalysisReport['user_type'];

function normalizeSentiment(value?: string | null) {
  const s = (value || '').trim().toLowerCase();
  if (['positive', 'negative', 'neutral', 'mixed'].includes(s)) {
    return s.charAt(0).toUpperCase() + s.slice(1);
  }
  return 'Mixed';
}

function normalizeCategory(value?: string | null) {
  const v = (value || '').trim();
  if (!v) return 'General';
  return v.charAt(0).toUpperCase() + v.slice(1);
}

function buildRecommendedAction(
  userType: UserType,
  classification?: string | null,
  theme?: string | null,
  issue?: string | null,
  reviewCount?: number
) {
  const base = normalizeCategory(classification || theme || 'Issue');
  const countHint = reviewCount && reviewCount > 1 ? ` (${reviewCount} similar reviews)` : '';
  const detail = issue ? ` focusing on "${issue.slice(0, 80)}"` : '';

  switch (userType) {
    case 'product-manager':
      return `Prioritize ${base.toLowerCase()}${countHint} in the next sprint${detail}, and validate with representative users.`;
    case 'engineer':
      return `Create and scope a fix for ${base.toLowerCase()}${countHint}${detail}; add regression tests.`;
    case 'support':
      return `Proactively message affected users about ${base.toLowerCase()}${countHint}${detail} and share a workaround.`;
    case 'business-analyst':
      return `Quantify impact of ${base.toLowerCase()}${countHint}${detail} on KPIs and track trend over time.`;
    case 'executive':
      return `Monitor ${base.toLowerCase()}${countHint}${detail} and include in the next leadership readout with owner + ETA.`;
    default:
      return `Track ${base.toLowerCase()}${countHint}${detail} and monitor impact on key users.`;
  }
}

function escapeHtml(text: string) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function renderTable(headers: string[], rows: string[][]) {
  const thead = `<thead><tr>${headers.map((h) => `<th>${escapeHtml(h)}</th>`).join('')}</tr></thead>`;
  const tbody = `<tbody>${rows
    .map((r) => `<tr>${r.map((c) => `<td>${escapeHtml(c)}</td>`).join('')}</tr>`)
    .join('')}</tbody>`;
  return `<table class="table">${thead}${tbody}</table>`;
}

export function generateHTMLReport(report: AnalysisReport) {
  const sentimentDenominator = Math.max(report.total_reviews || report.total_themes || 1, 1);
    const sentimentRows = Object.entries(report.sentiment_distribution).map(([sentiment, count]) => [
      sentiment,
      `${count}`,
      `${((count / sentimentDenominator) * 100).toFixed(1)}%`,
    ]);

    const issueDenominator = Math.max(report.total_reviews || report.total_themes || 1, 1);
      const issueEntries = Object.entries(report.issue_categories)
        .sort(([, a], [, b]) => b - a);
      const issueRows = issueEntries.map(([cat, count]) => [cat, `${count}`, `${((count / issueDenominator) * 100).toFixed(1)}%`]);

  const maxThemeRows = Math.max(1, Math.min(report.total_reviews || report.themes.length, 20));
  const themeRows = report.themes.slice(0, maxThemeRows).map((t, idx) => [
    `${idx + 1}. ${normalizeCategory(t.theme)}`,
    normalizeSentiment(t.sentiment),
    (t.issue_description || t.classification || '—').toString(),
  ]);

  const themes: ThemeData[] = report.themes || [];
  const maxDetailRows = Math.max(1, Math.min(report.total_reviews || themes.length, themes.length));
  const detailRows = themes.slice(0, maxDetailRows).map((theme) => {
    const sentiment = normalizeSentiment(theme.sentiment);
    const themeCategory = normalizeCategory(theme.theme || theme.classification || 'General');
    const issue = theme.issue_description || theme.theme || 'General feedback';
    const action = buildRecommendedAction(report.user_type, theme.classification, theme.theme, issue, theme.review_count);
    return [issue, sentiment, themeCategory, action];
  });

  const overviewRows = [
    ['Total Reviews', `${report.total_reviews}`],
    ['Issue Mentions (weighted)', `${Object.values(report.issue_categories).reduce((s, n) => s + n, 0)}`],
    ['Theme Mentions (weighted)', `${report.themes.reduce((s, t) => s + (t.review_count || 1), 0)}`],
    ['Unique Products Mentioned', `${report.text_analytics.unique_products}`],
    ['Unique Functionalities Mentioned', `${report.text_analytics.unique_functionalities}`],
    ['Unique Issue Categories', `${report.text_analytics.unique_issue_categories}`],
  ];

  const recommendationsHtml = (report.recommendations || [])
    .map((rec, idx) => `<li><strong>${idx + 1}.</strong> ${escapeHtml(rec)}</li>`)
    .join('');

  const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>Feedback Analysis Report</title>
  <style>
    :root {
      --gray-50: #f8f9fa;
      --gray-100: #f1f3f4;
      --gray-200: #e8eaed;
      --gray-700: #5f6368;
      --gray-900: #202124;
      --blue-500: #1a73e8;
      --red-500: #d93025;
      --green-500: #1e8e3e;
    }
    body {
      font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
      margin: 0;
      background: var(--gray-50);
      color: var(--gray-900);
      padding: 24px;
    }
    .container {
      max-width: 1024px;
      margin: 0 auto;
      background: white;
      padding: 24px;
      border-radius: 12px;
      border: 1px solid var(--gray-200);
      box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    }
    h1 { font-size: 22px; margin: 0 0 12px; }
    h2 { font-size: 16px; margin: 16px 0 8px; }
    p { margin: 4px 0; color: var(--gray-700); }
    .meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 8px; margin-bottom: 12px; }
    .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px; }
    .card { background: var(--gray-50); border: 1px solid var(--gray-200); padding: 10px 12px; border-radius: 8px; }
    .card strong { display: block; color: var(--gray-900); }
    .table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    .table th, .table td { border: 1px solid var(--gray-200); padding: 8px; font-size: 12px; text-align: left; vertical-align: top; }
    .table th { background: var(--gray-100); font-weight: 600; }
    .chart { margin-top: 8px; }
    .bar-row { display: grid; grid-template-columns: 1fr auto auto; gap: 8px; align-items: center; margin-bottom: 6px; }
    .bar-label { font-size: 12px; color: var(--gray-900); }
    .bar-track { background: var(--gray-100); border-radius: 999px; overflow: hidden; height: 10px; }
    .bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #1a73e8, #34a853); }
    .bar-value { font-size: 12px; color: var(--gray-700); text-align: right; min-width: 60px; }
    ul { padding-left: 18px; color: var(--gray-700); }
    .note { font-size: 12px; color: var(--gray-700); margin-top: 6px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Feedback Analysis Report</h1>
    <div class="meta">
      <p><strong>Generated:</strong> ${escapeHtml(new Date(report.generated_at).toLocaleString())}</p>
      <p><strong>User Type:</strong> ${escapeHtml(report.user_type)}</p>
      <p><strong>Data Sources:</strong> ${escapeHtml(report.data_sources.join(', '))}</p>
      <p><strong>Search Queries:</strong> ${escapeHtml(report.search_queries?.join(', ') || '')}</p>
    </div>

    <h2>Overview</h2>
    <div class="card-grid">
      ${overviewRows
        .map(
          ([label, value]) => `<div class="card"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(value)}</span></div>`
        )
        .join('')}
    </div>
    <p class="note">Issue/Theme mentions are weighted by how many reviews support them. "Unique Products/Functionalities" count distinct names mentioned across reviews (context only; not charted).</p>

    <h2>Sentiment Distribution</h2>
    <div class="chart">
      ${sentimentRows
        .map(([label, count, pct]) => {
          const pctNum = Number(pct.replace('%', '')) || 0;
          return `
            <div class="bar-row">
              <div class="bar-label">${escapeHtml(label)}</div>
              <div class="bar-track"><div class="bar-fill" style="width:${pctNum}%;"></div></div>
              <div class="bar-value">${escapeHtml(pct)}</div>
            </div>`;
        })
        .join('')}
    </div>

    <h2>Issue Categories</h2>
    <div class="chart">
      ${issueEntries
        .map(([label, count]) => {
          const pctNum = issueDenominator ? Math.min(100, (count / issueDenominator) * 100) : 0;
          return `
            <div class="bar-row">
              <div class="bar-label">${escapeHtml(label)}</div>
              <div class="bar-track"><div class="bar-fill" style="width:${pctNum.toFixed(1)}%;"></div></div>
              <div class="bar-value">${count} (${pctNum.toFixed(1)}%)</div>
            </div>`;
        })
        .join('')}
    </div>

    <h2>Themes Snapshot</h2>
    ${renderTable(['Theme', 'Sentiment', 'Representative Issue'], themeRows)}

    <h2>Recommendations</h2>
    <ul>${recommendationsHtml || '<li>No recommendations available.</li>'}</ul>

    <h2>Review Details & Actions</h2>
    ${renderTable(['Review / Issue', 'Sentiment', 'Theme', 'Recommended Action'], detailRows)}
  </div>
</body>
</html>`;

  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `feedback-analysis-${report.id}.html`;
  a.click();
  URL.revokeObjectURL(url);
}

export function downloadHTMLReport(report: AnalysisReport) {
  generateHTMLReport(report);
}

export function generateIssuesCSVReport(report: AnalysisReport) {
  const headers = ['Issue Description', 'Sentiment', 'Theme', 'Recommended Action'];
  const rows = report.themes.map((t) => {
    const sentiment = normalizeSentiment(t.sentiment);
    const theme = normalizeCategory(t.theme || t.classification || 'General');
    const issue = (t.issue_description || t.theme || 'General feedback').toString();
    const action = buildRecommendedAction(report.user_type, t.classification, t.theme, issue, t.review_count);
    return [issue, sentiment, theme, action];
  });

  const csv = [headers, ...rows]
    .map((row) => row.map((cell) => `"${cell.replace(/"/g, '""')}"`).join(','))
    .join('\n');

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `issues-${report.id}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}
