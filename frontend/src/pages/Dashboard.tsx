import { Link } from 'react-router-dom';
import { Calendar, Clock, User, Database } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { OverviewCards } from '../components/charts/OverviewCards';
import { SentimentChart } from '../components/charts/SentimentChart';
import { IssueCategoriesChart } from '../components/charts/IssueCategoriesChart';

export function Dashboard() {
  const { lastRun, analysisHistory } = useAppStore();

  if (!lastRun) {
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
        {/* Last Run Header */}
        <div className="bg-white rounded-lg border border-google-gray-200 p-6">
          <h2 className="text-2xl font-bold text-google-gray-900 mb-4">Last Analysis Run</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="flex items-center gap-2">
              <Calendar className="w-4 h-4 text-google-gray-500" />
              <span className="text-google-gray-600">
                {new Date(lastRun.generated_at).toLocaleDateString()}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-google-gray-500" />
              <span className="text-google-gray-600">
                {new Date(lastRun.generated_at).toLocaleTimeString()}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <User className="w-4 h-4 text-google-gray-500" />
              <span className="text-google-gray-600 capitalize">
                {lastRun.user_type?.replace('-', ' ')}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-google-gray-500" />
              <span className="text-google-gray-600">
                {lastRun.data_sources.join(', ')}
              </span>
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-google-gray-200">
            <p className="text-sm text-google-gray-700">
              <strong>Search Queries:</strong> {lastRun.search_queries?.join(', ') || 'N/A'}
            </p>
          </div>
        </div>

        {/* Overview Cards */}
        <OverviewCards report={lastRun} />

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <SentimentChart report={lastRun} />
          <IssueCategoriesChart report={lastRun} />
        </div>

        {/* Quick Actions */}
        <div className="flex gap-4">
          <Link
            to="/reports"
            className="px-6 py-3 bg-google-blue-500 text-white rounded-lg hover:bg-google-blue-600"
          >
            View Full Report
          </Link>
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
              {analysisHistory.slice(1, 6).map((report, idx) => (
                <div key={report.id} className="flex items-center justify-between p-3 hover:bg-google-gray-50 rounded">
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-google-gray-500">#{idx + 2}</span>
                    <span className="text-sm text-google-gray-900">
                      {new Date(report.generated_at).toLocaleString()}
                    </span>
                    <span className="text-xs text-google-gray-600">
                      ({report.total_reviews} reviews)
                    </span>
                  </div>
                  <button className="text-sm text-google-blue-600 hover:text-google-blue-700">
                    View
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
