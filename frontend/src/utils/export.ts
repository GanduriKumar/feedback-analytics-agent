import jsPDF from 'jspdf';
import type { AnalysisReport, ThemeData } from '../types';
import { getUniqueThemeCount } from './report';

type UserType = AnalysisReport['user_type'];

function normalizeSentiment(value?: string | null) {
  const s = (value || '').trim().toLowerCase();
  if (s === 'positive' || s === 'negative' || s === 'neutral' || s === 'mixed') return s;
  return 'unknown';
}

function normalizeCategory(value?: string | null) {
  const v = (value || '').trim();
  return v ? v : 'General';
}

function resolveOwner(userType: UserType, classification?: string | null, theme?: string | null) {
  const text = `${classification || ''} ${theme || ''}`.toLowerCase();

  if (/(bug|crash|freeze|lag|slow|performance|stability|connectivity|bluetooth|wifi|network)/.test(text)) {
    return userType === 'executive' ? 'engineering leadership' : 'engineering';
  }
  if (/(billing|pricing|subscription|cost)/.test(text)) {
    return userType === 'executive' ? 'business leadership' : 'business analyst';
  }
  if (/(support|onboarding|help|service|refund)/.test(text)) {
    return userType === 'executive' ? 'customer operations' : 'support';
  }
  if (/(design|ux|ui|usability|feature|roadmap)/.test(text)) {
    return userType === 'executive' ? 'product leadership' : 'product manager';
  }

  return userType.replace('-', ' ');
}

function buildRecommendedAction(userType: UserType, classification?: string | null, theme?: string | null) {
  const owner = resolveOwner(userType, classification, theme);
  const cat = normalizeCategory(classification || theme || 'General');

  const actionByPersona: Record<UserType, string> = {
    'product-manager': `Prioritize ${cat} in the roadmap and define acceptance criteria with ${owner}.`,
    engineer: `Investigate ${cat} root cause, add telemetry, and ship a fix with regression tests.`,
    support: `Prepare a troubleshooting guide for ${cat} and update macros/FAQs.`,
    'business-analyst': `Quantify ${cat} impact and track KPI deltas post-fix.`,
    executive: `Align owners on ${cat} risk, timeline, and user impact mitigation.`,
  };

  return actionByPersona[userType] || `Coordinate on ${cat} remediation with ${owner}.`;
}

export function generateIssuesCSVReport(report: AnalysisReport) {
  const rows = [
    ['Issue Scenario / Description', 'Sentiment', 'Theme Category'],
  ];

  report.themes.forEach((theme) => {
    const sentiment = normalizeSentiment(theme.sentiment);
    const themeCategory = normalizeCategory(theme.theme);
    const issueDescription = (theme.issue_description || theme.theme || 'General').toString();

    rows.push([
      issueDescription,
      sentiment,
      themeCategory,
    ]);
  });

  const csvContent = rows.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `feedback-issues-${report.id}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export function downloadIssuesCSVReport(report: AnalysisReport) {
  generateIssuesCSVReport(report);
}

export function generatePDFReport(report: AnalysisReport) {
  const doc = new jsPDF({ unit: 'pt' });
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 32;
  const contentWidth = pageWidth - margin * 2;
  const lineHeight = 12;
  let y = margin;

  const truncate = (text: string, max = 140) => (text.length > max ? `${text.slice(0, max - 1)}…` : text);

  const addSectionTitle = (title: string) => {
    doc.setFontSize(12);
    doc.setFont('helvetica', 'bold');
    doc.text(title, margin, y);
    y += lineHeight;
  };

  const ensureSpace = (needed = lineHeight) => {
    const pageHeight = doc.internal.pageSize.getHeight();
    if (y + needed > pageHeight - margin) {
      doc.addPage();
      y = margin;
    }
  };

  const addKvp = (label: string, value: string) => {
    ensureSpace();
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.text(`${label}:`, margin, y);
    doc.setFont('helvetica', 'normal');
    const lines = doc.splitTextToSize(value, contentWidth - 90);
    doc.text(lines, margin + 80, y);
    y += lines.length * (lineHeight - 6);
  };

  const addTable = (headers: string[], rows: string[][]) => {
    const colCount = headers.length;
    const colWidth = contentWidth / colCount;

    const drawRow = (cells: string[], isHeader = false) => {
      ensureSpace(lineHeight * 1.2);
      cells.forEach((cell, idx) => {
        const x = margin + idx * colWidth;
        const maxWidth = colWidth - 8;
        const lines = doc.splitTextToSize(cell, maxWidth);
        lines.forEach((line, lineIdx) => {
          ensureSpace(lineHeight);
          doc.setFontSize(isHeader ? 10 : 9);
          doc.setFont('helvetica', isHeader ? 'bold' : 'normal');
          doc.text(line, x + 4, y + lineIdx * (lineHeight - 4));
        });
      });
      y += lineHeight + 2;
    };

    drawRow(headers, true);
    rows.forEach((r) => drawRow(r, false));
    y += 4;
  };

  // Title
  doc.setFontSize(16);
  doc.setFont('helvetica', 'bold');
  doc.text('Feedback Analysis Report', pageWidth / 2, y, { align: 'center' });
  y += lineHeight * 1.5;

  // Metadata
  addKvp('Generated', new Date(report.generated_at).toLocaleString());
  addKvp('User Type', report.user_type);
  addKvp('Data Sources', report.data_sources.join(', '));
  addKvp('Search Queries', report.search_queries?.join(', ') || '');
  y += 6;

  // Overview / cards
  addSectionTitle('Overview');
  const overview = [
    ['Total Reviews', `${report.total_reviews}`],
    ['Issue Categories', `${Object.keys(report.issue_categories).length}`],
    ['Themes Identified', `${getUniqueThemeCount(report.themes)}`],
    ['Unique Products', `${report.text_analytics.unique_products}`],
    ['Unique Functionalities', `${report.text_analytics.unique_functionalities}`],
    ['Unique Issue Categories', `${report.text_analytics.unique_issue_categories}`],
  ];
  addTable(['Metric', 'Value'], overview);

  // Sentiment (visualization table)
  addSectionTitle('Sentiment Distribution');
  const sentimentDenominator = Math.max(report.total_themes || report.total_reviews || 1, 1);
  const sentimentRows = Object.entries(report.sentiment_distribution).map(([sentiment, count]) => [
    sentiment,
    `${count}`,
    `${((count / sentimentDenominator) * 100).toFixed(1)}%`,
  ]);
  addTable(['Sentiment', 'Count', 'Percentage'], sentimentRows);

  // Issue Categories (visualization table)
  addSectionTitle('Issue Categories');
  const issueDenominator = Math.max(report.total_themes || report.total_reviews || 1, 1);
  const issueRows = Object.entries(report.issue_categories)
    .sort(([, a], [, b]) => b - a)
    .map(([cat, count]) => [cat, `${count}`, `${((count / issueDenominator) * 100).toFixed(1)}%`]);
  addTable(['Category', 'Count', 'Percentage'], issueRows);

  // Themes summary table (visualization analog)
  addSectionTitle('Themes Snapshot');
  const themeRows = report.themes.slice(0, 20).map((t, idx) => [
    `${idx + 1}. ${normalizeCategory(t.theme)}`,
    normalizeSentiment(t.sentiment),
    (t.issue_description || t.classification || '—').toString(),
  ]);
  addTable(['Theme', 'Sentiment', 'Representative Issue'], themeRows);

  // Recommendations (text list)
  if (report.recommendations && report.recommendations.length) {
    addSectionTitle('Recommendations');
    doc.setFontSize(9);
    doc.setFont('helvetica', 'normal');
    report.recommendations.forEach((rec, idx) => {
      const lines = doc.splitTextToSize(`${idx + 1}. ${rec}`, contentWidth);
      lines.forEach((line) => {
        ensureSpace(lineHeight);
        doc.text(line, margin, y);
        y += lineHeight - 4;
      });
      y += 2;
    });
  }

  // Detailed review/action table per extracted theme
  addSectionTitle('Review Details & Actions');
  const detailRows: string[][] = [];
  const themes: ThemeData[] = report.themes || [];
  themes.forEach((theme) => {
    const sentiment = normalizeSentiment(theme.sentiment);
    const themeCategory = normalizeCategory(theme.theme || theme.classification || 'General');
    const issue = truncate(theme.issue_description || theme.theme || 'General feedback', 120);
    const action = truncate(buildRecommendedAction(report.user_type, theme.classification, theme.theme), 140);
    detailRows.push([
      issue,
      sentiment,
      themeCategory,
      action,
    ]);
  });
  addTable(['Review / Issue', 'Sentiment', 'Theme', 'Recommended Action'], detailRows);

  doc.save(`feedback-analysis-${report.id}.pdf`);
}

export function downloadPDFReport(report: AnalysisReport) {
  generatePDFReport(report);
}
