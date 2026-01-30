import { Link } from 'react-router-dom';
import { Calendar, Clock, User, Database, FileText, Table, Check } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { OverviewCards } from '../components/charts/OverviewCards';
import { SentimentChart } from '../components/charts/SentimentChart';
import { IssueCategoriesChart } from '../components/charts/IssueCategoriesChart';
import { ThemesChart } from '../components/charts/ThemesChart';
import { generatePDFReport, generateIssuesCSVReport } from '../utils/export';

export function Dashboard() {
  const { lastRun, analysisHistory, selectedReportId, setSelectedReport } = useAppStore();

  const selectedReport = selectedReportId
    ? analysisHistory.find((r) => r.id === selectedReportId) || lastRun
    : lastRun;

  if (!selectedReport) {
    return (
      <div className="min-h-[calc(100vh-4rem)] bg-google-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Database className="w-16 h-16 text-google-gray-400 mx-auto mb-4" />
          <h2 className="text-2xl font-semibold text-google-gray-900 mb-2">No Analysis Yet</h2>
          <p className="text-google-gray-600 mb-6">Run your first analysis to see results here</p>
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
        {/* Selected Run Header */}
        <div className="bg-white rounded-lg border border-google-gray-200 p-6">
          <div className="flex items-start justify-between gap-3 mb-4">
            <div>
              <h2 className="text-2xl font-bold text-google-gray-900">Selected Analysis</h2>
              <p className="text-sm text-google-gray-600">Visualizations and downloads reflect this run.</p>
            </div>
            <div className="flex items-center gap-2 bg-google-gray-100 px-3 py-1.5 rounded text-sm text-google-gray-700">
              <span className="w-2 h-2 rounded-full bg-google-green-500" />
              <span>Viewing #{analysisHistory.findIndex((r) => r.id === selectedReport?.id) + 1 || 1}</span>
            </div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="flex items-center gap-2">
              <Calendar className="w-4 h-4 text-google-gray-500" />
              <span className="text-google-gray-600">
                {new Date(selectedReport.generated_at).toLocaleDateString()}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-google-gray-500" />
              <span className="text-google-gray-600">
                {new Date(selectedReport.generated_at).toLocaleTimeString()}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <User className="w-4 h-4 text-google-gray-500" />
              <span className="text-google-gray-600 capitalize">
                {selectedReport.user_type?.replace('-', ' ')}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-google-gray-500" />
              <span className="text-google-gray-600">
                {selectedReport.data_sources.join(', ')}
              </span>
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-google-gray-200">
            <p className="text-sm text-google-gray-700">
              <strong>Search Queries:</strong> {selectedReport.search_queries?.join(', ') || 'N/A'}
            </p>
          </div>
        </div>

        {/* Overview Cards */}
        <OverviewCards report={selectedReport} />

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <SentimentChart report={selectedReport} />
          <IssueCategoriesChart report={selectedReport} />
          <ThemesChart report={selectedReport} />
        </div>

        {/* Quick Actions */}
        <div className="flex flex-wrap gap-3">
          <button
            onClick={() => generatePDFReport(selectedReport)}
            className="flex items-center gap-2 px-6 py-3 bg-google-red-500 text-white rounded-lg hover:bg-google-red-600"
          >
            <FileText className="w-5 h-5" />
            Download PDF
          </button>
          <button
            onClick={() => generateIssuesCSVReport(selectedReport)}
            className="flex items-center gap-2 px-6 py-3 bg-google-blue-500 text-white rounded-lg hover:bg-google-blue-600"
          >
            <Table className="w-5 h-5" />
            Download Issues CSV
          </button>
          <Link
            to="/analyze"
            className="px-6 py-3 border border-google-gray-300 text-google-gray-700 rounded-lg hover:bg-google-gray-50"
          >
            Run New Analysis
          </Link>
        </div>

        {/* Analysis History */}
        {analysisHistory.length > 1 && (
          <div className="bg-white rounded-lg border border-google-gray-200 p-6">
            <h3 className="text-lg font-semibold text-google-gray-900 mb-4">Recent Analyses</h3>
            <div className="space-y-2">
              {analysisHistory.map((report, idx) => {
                const isSelected = report.id === selectedReport?.id;
                return (
                  <div
                    key={report.id}
                    className={`flex items-center justify-between p-3 rounded border ${
                      isSelected ? 'border-google-blue-200 bg-google-blue-50' : 'border-transparent hover:bg-google-gray-50'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-sm text-google-gray-500">#{idx + 1}</span>
                      <span className="text-sm text-google-gray-900">
                        {new Date(report.generated_at).toLocaleString()}
                      </span>
                      <span className="text-xs text-google-gray-600">
                        ({report.total_reviews} reviews)
                      </span>
                      {isSelected && (
                        <span className="flex items-center gap-1 text-xs text-google-blue-700 font-medium">
                          <Check className="w-4 h-4" /> Selected
                        </span>
                      )}
                    </div>
                    {!isSelected && (
                      <button
                        onClick={() => setSelectedReport(report.id)}
                        className="text-sm text-google-blue-600 hover:text-google-blue-700"
                      >
                        View
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
