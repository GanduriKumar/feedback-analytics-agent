import jsPDF from 'jspdf';
import type { AnalysisReport } from '../types';

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
    ['Themes Identified', report.total_themes.toString()],
    [''],
    ['Sentiment Distribution'],
    ['Sentiment', 'Count', 'Percentage']
  ];

  Object.entries(report.sentiment_distribution).forEach(([sentiment, count]) => {
    const percentage = ((count / report.total_reviews) * 100).toFixed(1);
    rows.push([sentiment, count.toString(), `${percentage}%`]);
  });

  rows.push([''], ['Issue Categories'], ['Category', 'Count', 'Percentage']);

  Object.entries(report.issue_categories)
    .sort(([, a], [, b]) => b - a)
    .forEach(([category, count]) => {
      const percentage = ((count / report.total_reviews) * 100).toFixed(1);
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
