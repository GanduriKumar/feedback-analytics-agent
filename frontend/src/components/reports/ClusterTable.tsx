import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { AnalysisReport } from '../../types';

interface Props {
  report: AnalysisReport;
}

export function ClusterTable({ report }: Props) {
  const [expandedClusters, setExpandedClusters] = useState<Set<string>>(new Set());

  const toggleCluster = (clusterId: string) => {
    setExpandedClusters((prev) => {
      const next = new Set(prev);
      if (next.has(clusterId)) {
        next.delete(clusterId);
      } else {
        next.add(clusterId);
      }
      return next;
    });
  };

  if (!report.clusters?.clusters) {
    return null;
  }

  const clusterEntries = Object.entries(report.clusters.clusters).sort(
    ([a], [b]) => Number(a) - Number(b)
  );

  return (
    <div className="bg-white rounded-lg border border-google-gray-200 overflow-hidden">
      <div className="px-6 py-4 border-b border-google-gray-200">
        <h3 className="text-lg font-semibold text-google-gray-900">Cluster Details</h3>
        <p className="text-sm text-google-gray-600 mt-1">
          Reviews grouped by semantic similarity
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-google-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-google-gray-700 uppercase tracking-wider">
                Cluster
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-google-gray-700 uppercase tracking-wider">
                Size
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-google-gray-700 uppercase tracking-wider">
                Reviews
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-google-gray-200">
            {clusterEntries.map(([clusterId, reviews]) => (
              <>
                <tr
                  key={clusterId}
                  className="hover:bg-google-gray-50 cursor-pointer"
                  onClick={() => toggleCluster(clusterId)}
                >
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      {expandedClusters.has(clusterId) ? (
                        <ChevronDown className="w-5 h-5 text-google-gray-600 mr-2" />
                      ) : (
                        <ChevronRight className="w-5 h-5 text-google-gray-600 mr-2" />
                      )}
                      <span className="font-medium text-google-gray-900">
                        Cluster {clusterId}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-google-gray-900">
                    {reviews.length} reviews
                  </td>
                  <td className="px-6 py-4 text-sm text-google-gray-700">
                    {expandedClusters.has(clusterId)
                      ? 'Click to collapse'
                      : 'Click to view reviews'}
                  </td>
                </tr>
                {expandedClusters.has(clusterId) && (
                  <tr>
                    <td colSpan={3} className="px-6 py-4 bg-google-gray-50">
                      <div className="space-y-2">
                        {reviews.slice(0, 10).map((review, idx) => (
                          <div
                            key={idx}
                            className="bg-white rounded-lg p-4 border border-google-gray-200 text-sm text-google-gray-700"
                          >
                            <span className="text-xs text-google-gray-500 font-medium mr-2">
                              #{idx + 1}
                            </span>
                            {review}
                          </div>
                        ))}
                        {reviews.length > 10 && (
                          <p className="text-xs text-google-gray-600 text-center py-2">
                            Showing 10 of {reviews.length} reviews
                          </p>
                        )}
                      </div>
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
