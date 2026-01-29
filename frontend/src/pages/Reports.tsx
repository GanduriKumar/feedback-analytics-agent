import { Link } from 'react-router-dom';
import { Download, FileText, Table2 } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { OverviewCards } from '../components/charts/OverviewCards';
import { SentimentChart } from '../components/charts/SentimentChart';
import { IssueCategoriesChart } from '../components/charts/IssueCategoriesChart';
import { ThemesTable } from '../components/reports/ThemesTable';
import { ClustersPanel } from '../components/reports/ClustersPanel';
import { downloadCSVReport, downloadPDFReport } from '../utils/export';

export function Reports() {
  const { lastRun } = useAppStore();

  if (!lastRun) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-10">
        <div className="rounded-2xl border border-google-gray-200 bg-white p-10 text-center shadow-sm">
          <FileText className="w-14 h-14 text-google-gray-400 mx-auto" />
          <h2 className="text-2xl font-semibold text-google-gray-900 mt-4">No report available</h2>
          <p className="text-google-gray-600 mt-2">Run Extract & Analyze first to generate a report.</p>
          <Link
            to="/analyze"
            className="inline-flex items-center gap-2 mt-6 px-5 py-2.5 rounded-lg bg-google-blue-600 text-white hover:bg-google-blue-700"
          >
            Start Extract & Analyze
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold text-google-gray-900">Reports</h1>
          <p className="text-google-gray-600 mt-1">Generated: {new Date(lastRun.generated_at).toLocaleString()}</p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => downloadPDFReport(lastRun)}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-google-red-600 text-white hover:bg-google-red-700"
          >
            <FileText className="w-4 h-4" /> PDF
          </button>
          <button
            type="button"
            onClick={() => downloadCSVReport(lastRun)}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-google-green-600 text-white hover:bg-google-green-700"
          >
            <Table2 className="w-4 h-4" /> CSV
          </button>
          <Link to="/analyze" className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-google-gray-300 hover:bg-google-gray-50">
            <Download className="w-4 h-4" /> Run new
          </Link>
        </div>
      </div>

      <OverviewCards report={lastRun} />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <SentimentChart report={lastRun} />
        <IssueCategoriesChart report={lastRun} />
      </div>

      <div className="rounded-xl border border-google-gray-200 bg-white p-5 shadow-sm">
        <div className="font-semibold text-google-gray-900">Recommendations</div>
        <ul className="mt-3 space-y-2">
          {lastRun.recommendations.map((r, idx) => (
            <li key={idx} className="text-sm text-google-gray-700">
              <span className="mr-2 text-google-blue-600 font-semibold">{idx + 1}.</span>
              {r}
            </li>
          ))}
        </ul>
      </div>

      {lastRun.clusters && <ClustersPanel clusters={lastRun.clusters} />}
      <ThemesTable themes={lastRun.themes} />
    </div>
  );
}
