import { Link } from 'react-router-dom';
import { FileText, Table } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { OverviewCards } from '../components/charts/OverviewCards';
import { SentimentChart } from '../components/charts/SentimentChart';
import { IssueCategoriesChart } from '../components/charts/IssueCategoriesChart';
import { ThemesChart } from '../components/charts/ThemesChart';
import { UserTypeThemesChart } from '../components/charts/UserTypeThemesChart';
import { ClusterTable } from '../components/reports/ClusterTable';
import { generatePDFReport, generateCSVReport, generateIssuesCSVReport } from '../utils/export';

export function Reports() {
  const { lastRun } = useAppStore();
  const report = lastRun;

  if (!report) {
    return (
      <div className="min-h-[calc(100vh-4rem)] bg-google-gray-50 flex items-center justify-center">
        <div className="text-center">
          <FileText className="w-16 h-16 text-google-gray-400 mx-auto mb-4" />
          <h2 className="text-2xl font-semibold text-google-gray-900 mb-2">No Report Available</h2>
          <p className="text-google-gray-600 mb-6">Run an analysis first to generate a report</p>
          <Link
            to="/analyze"
            className="px-6 py-3 bg-google-blue-500 text-white rounded-lg hover:bg-google-blue-600 inline-block"
          >
            Start Analysis
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-google-gray-50">
      <div className="max-w-7xl mx-auto px-8 py-8 space-y-6">
        {/* Header with Download Options */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-google-gray-900">Analysis Report</h1>
            <p className="text-google-gray-600 mt-1">
              Generated on {new Date(report.generated_at).toLocaleString()}
            </p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={() => generatePDFReport(report)}
              className="flex items-center gap-2 px-6 py-3 bg-google-red-500 text-white rounded-lg hover:bg-google-red-600"
            >
              <FileText className="w-5 h-5" />
              Download PDF
            </button>
            <button
              onClick={() => generateCSVReport(report)}
              className="flex items-center gap-2 px-6 py-3 bg-google-green-500 text-white rounded-lg hover:bg-google-green-600"
            >
              <Table className="w-5 h-5" />
              Download Summary CSV
            </button>
            <button
              onClick={() => generateIssuesCSVReport(report)}
              className="flex items-center gap-2 px-6 py-3 bg-google-blue-500 text-white rounded-lg hover:bg-google-blue-600"
            >
              <Table className="w-5 h-5" />
              Download Issues CSV
            </button>
          </div>
        </div>

        {/* Report Content */}
        <OverviewCards report={report} />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <SentimentChart report={report} />
          <IssueCategoriesChart report={report} />
          <ThemesChart report={report} />
        </div>

        <UserTypeThemesChart report={report} />

        <ClusterTable report={report} />

        {/* Recommendations Section */}
        {report.recommendations && report.recommendations.length > 0 && (
          <div className="bg-white rounded-lg border border-google-gray-200 p-6">
            <h3 className="text-lg font-semibold text-google-gray-900 mb-4">Recommendations</h3>
            <ul className="space-y-3">
              {report.recommendations.map((rec, idx) => (
                <li key={idx} className="flex gap-3">
                  <span className="flex-shrink-0 w-6 h-6 rounded-full bg-google-blue-100 text-google-blue-700 flex items-center justify-center text-sm font-semibold">
                    {idx + 1}
                  </span>
                  <p className="text-google-gray-700">{rec}</p>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
