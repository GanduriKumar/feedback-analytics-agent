import jsPDF from 'jspdf';
import type { AnalysisReport } from '../types';
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

function buildTestcase(sentiment?: string | null, theme?: string | null, issue?: string | null) {
  const normalized = normalizeSentiment(sentiment);
  if (normalized === 'positive') return '';

  const text = `${theme || ''} ${issue || ''}`.toLowerCase();

  if (/(bluetooth|connectivity|wifi|network)/.test(text)) {
    return '1) Pair device with BT accessory. 2) Toggle BT off/on. 3) Start media playback. 4) Observe disconnects.';
  }
  if (/(battery|charging)/.test(text)) {
    return '1) Fully charge device. 2) Use for 1 hour with screen on. 3) Log battery drain rate vs baseline.';
  }
  if (/(camera)/.test(text)) {
    return '1) Open camera app. 2) Switch to low-light mode. 3) Capture 10 photos. 4) Observe focus/quality issues.';
  }
  if (/(performance|lag|slow|freeze|crash|stability)/.test(text)) {
    return '1) Open target feature. 2) Perform repeated navigation/actions. 3) Record response time and crashes.';
  }
  if (/(update)/.test(text)) {
    return '1) Update to latest build. 2) Reboot. 3) Re-run reported workflow. 4) Verify regression.';
  }

  return '1) Follow reported user steps. 2) Capture logs/telemetry. 3) Confirm reproducibility.';
}

export function generateCSVReport(report: AnalysisReport) {
  const rows = [
    ['Feedback Analysis Report'],
    ['Generated', new Date(report.generated_at).toLocaleString()],
    ['User Type', report.user_type],
    ['Data Sources', report.data_sources.join(', ')],
    ['Search Queries', report.search_queries?.join(', ') || ''],
    [''],
    ['Overview'],
    ['Total Reviews', report.total_reviews.toString()],
    ['Issue Categories', Object.keys(report.issue_categories).length.toString()],
    ['Themes Identified', getUniqueThemeCount(report.themes).toString()],
    [''],
    ['Sentiment Distribution'],
    ['Sentiment', 'Count', 'Percentage']
  ];

  const sentimentDenominator = Math.max(report.total_themes || 0, 1);
  Object.entries(report.sentiment_distribution).forEach(([sentiment, count]) => {
    const percentage = ((count / sentimentDenominator) * 100).toFixed(1);
    rows.push([sentiment, count.toString(), `${percentage}%`]);
  });

  rows.push([''], ['Issue Categories'], ['Category', 'Count', 'Percentage']);

  const issueDenominator = Math.max(report.total_themes || 0, 1);
  Object.entries(report.issue_categories)
    .sort(([, a], [, b]) => b - a)
    .forEach(([category, count]) => {
      const percentage = ((count / issueDenominator) * 100).toFixed(1);
      rows.push([category, count.toString(), `${percentage}%`]);
    });

  const csvContent = rows.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `feedback-analysis-${report.id}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export function downloadCSVReport(report: AnalysisReport) {
  generateCSVReport(report);
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
  const doc = new jsPDF();
  const pageWidth = doc.internal.pageSize.getWidth();
  let yPos = 20;

  // Title
  doc.setFontSize(20);
  doc.setFont('helvetica', 'bold');
  doc.text('Feedback Analysis Report', pageWidth / 2, yPos, { align: 'center' });
  yPos += 15;

  // Metadata
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.text(`Generated: ${new Date(report.generated_at).toLocaleString()}`, 20, yPos);
  yPos += 6;
  doc.text(`User Type: ${report.user_type}`, 20, yPos);
  yPos += 6;
  doc.text(`Data Sources: ${report.data_sources.join(', ')}`, 20, yPos);
  yPos += 6;
  doc.text(`Search Queries: ${report.search_queries?.join(', ')}`, 20, yPos);
  yPos += 15;

  // Overview Section
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text('Executive Summary', 20, yPos);
  yPos += 10;

  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.text(`Total Reviews Analyzed: ${report.total_reviews}`, 25, yPos);
  yPos += 6;
  doc.text(`Issue Categories: ${Object.keys(report.issue_categories).length}`, 25, yPos);
  yPos += 6;
  doc.text(`Themes Identified: ${report.total_themes}`, 25, yPos);
  yPos += 6;
  doc.text(`Unique Products: ${report.text_analytics.unique_products}`, 25, yPos);
  yPos += 15;

  // Sentiment Distribution
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text('Sentiment Distribution', 20, yPos);
  yPos += 10;

  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  Object.entries(report.sentiment_distribution).forEach(([sentiment, count]) => {
    const percentage = ((count / report.total_reviews) * 100).toFixed(1);
    doc.text(`${sentiment}: ${count} (${percentage}%)`, 25, yPos);
    yPos += 6;
  });
  yPos += 10;

  // Issue Categories
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text('Top Issue Categories', 20, yPos);
  yPos += 10;

  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  const sortedIssues = Object.entries(report.issue_categories)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10);

  sortedIssues.forEach(([category, count]) => {
    const percentage = ((count / report.total_reviews) * 100).toFixed(1);
    doc.text(`${category}: ${count} (${percentage}%)`, 25, yPos);
    yPos += 6;
    if (yPos > 270) {
      doc.addPage();
      yPos = 20;
    }
  });

  // Recommendations
  if (report.recommendations && report.recommendations.length > 0) {
    yPos += 10;
    if (yPos > 250) {
      doc.addPage();
      yPos = 20;
    }

    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('Recommendations', 20, yPos);
    yPos += 10;

    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    report.recommendations.forEach((rec, idx) => {
      const lines = doc.splitTextToSize(`${idx + 1}. ${rec}`, pageWidth - 50);
      lines.forEach((line: string) => {
        if (yPos > 270) {
          doc.addPage();
          yPos = 20;
        }
        doc.text(line, 25, yPos);
        yPos += 6;
      });
      yPos += 3;
    });
  }

  // Save PDF
  doc.save(`feedback-analysis-${report.id}.pdf`);
}

export function downloadPDFReport(report: AnalysisReport) {
  generatePDFReport(report);
}
