import jsPDF from 'jspdf';
import type { AnalysisReport } from '../types';

export function downloadCSVReport(report: AnalysisReport) {
  const rows: string[][] = [];

  rows.push(['Feedback Analysis Report']);
  rows.push(['Generated', new Date(report.generated_at).toLocaleString()]);
  rows.push(['User Type', report.user_type]);
  rows.push(['Sources', report.data_sources.join(', ')]);
  rows.push(['Queries', report.search_queries.join(', ')]);
  rows.push([]);

  rows.push(['Summary']);
  rows.push(['Total Reviews', String(report.total_reviews)]);
  rows.push(['Total Themes', String(report.total_themes)]);
  rows.push(['Unique Products', String(report.text_analytics.unique_products)]);
  rows.push(['Unique Functionalities', String(report.text_analytics.unique_functionalities)]);
  rows.push([]);

  rows.push(['Issue Categories']);
  rows.push(['Category', 'Count']);
  Object.entries(report.issue_categories)
    .sort((a, b) => b[1] - a[1])
    .forEach(([cat, count]) => rows.push([cat, String(count)]));

  rows.push([]);
  rows.push(['Sentiment Distribution']);
  rows.push(['Sentiment', 'Count']);
  Object.entries(report.sentiment_distribution)
    .sort((a, b) => b[1] - a[1])
    .forEach(([s, count]) => rows.push([s, String(count)]));

  rows.push([]);
  rows.push(['Recommendations']);
  report.recommendations.forEach((r) => rows.push([r]));

  const csv = rows
    .map((r) => r.map((cell) => `"${String(cell ?? '').replaceAll('"', '""')}"`).join(','))
    .join('\n');

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `feedback-analysis-${report.id}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function downloadPDFReport(report: AnalysisReport) {
  const doc = new jsPDF();
  const pageWidth = doc.internal.pageSize.getWidth();
  let y = 18;

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(18);
  doc.text('Feedback Analysis Report', pageWidth / 2, y, { align: 'center' });
  y += 10;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10);
  doc.text(`Generated: ${new Date(report.generated_at).toLocaleString()}`, 14, y);
  y += 6;
  doc.text(`User Type: ${report.user_type}`, 14, y);
  y += 6;
  doc.text(`Sources: ${report.data_sources.join(', ')}`, 14, y);
  y += 6;
  doc.text(`Queries: ${report.search_queries.join(', ')}`, 14, y);
  y += 10;

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(12);
  doc.text('Key Metrics', 14, y);
  y += 8;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10);
  doc.text(`Total Reviews: ${report.total_reviews}`, 18, y);
  y += 6;
  doc.text(`Total Themes: ${report.total_themes}`, 18, y);
  y += 6;
  doc.text(`Unique Issue Categories: ${report.text_analytics.unique_issue_categories}`, 18, y);
  y += 10;

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(12);
  doc.text('Top Issue Categories', 14, y);
  y += 8;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10);
  const topCats = Object.entries(report.issue_categories).sort((a, b) => b[1] - a[1]).slice(0, 10);
  for (const [cat, count] of topCats) {
    if (y > 280) {
      doc.addPage();
      y = 18;
    }
    doc.text(`${cat}: ${count}`, 18, y);
    y += 6;
  }

  y += 6;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(12);
  doc.text('Recommendations', 14, y);
  y += 8;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10);
  for (const rec of report.recommendations) {
    const lines = doc.splitTextToSize(`• ${rec}`, pageWidth - 28);
    for (const line of lines) {
      if (y > 280) {
        doc.addPage();
        y = 18;
      }
      doc.text(line, 14, y);
      y += 6;
    }
  }

  doc.save(`feedback-analysis-${report.id}.pdf`);
}
