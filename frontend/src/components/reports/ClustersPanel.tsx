import { useMemo, useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { ClusterResponse } from '../../types';

export function ClustersPanel({ clusters }: { clusters: ClusterResponse }) {
  const [open, setOpen] = useState<Record<string, boolean>>({});

  const entries = useMemo(() => {
    return Object.entries(clusters.clusters).sort((a, b) => Number(a[0]) - Number(b[0]));
  }, [clusters]);

  return (
    <div className="rounded-xl border border-google-gray-200 bg-white overflow-hidden shadow-sm">
      <div className="px-5 py-4 border-b border-google-gray-200">
        <div className="font-semibold text-google-gray-900">Clusters</div>
        <div className="text-sm text-google-gray-600">Grouped reviews by semantic similarity.</div>
      </div>

      <div className="divide-y divide-google-gray-200">
        {entries.map(([clusterId, reviews]) => {
          const isOpen = !!open[clusterId];
          return (
            <div key={clusterId}>
              <button
                type="button"
                onClick={() => setOpen((s) => ({ ...s, [clusterId]: !s[clusterId] }))}
                className="w-full flex items-center justify-between px-5 py-3 hover:bg-google-gray-50"
              >
                <div className="flex items-center gap-2 text-google-gray-900 font-medium">
                  {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                  Cluster {clusterId}
                  <span className="text-xs text-google-gray-600 font-normal">({reviews.length} reviews)</span>
                </div>
              </button>

              {isOpen && (
                <div className="px-5 pb-4">
                  <div className="space-y-2">
                    {reviews.slice(0, 8).map((r, idx) => (
                      <div key={idx} className="text-sm text-google-gray-700 bg-google-gray-50 border border-google-gray-200 rounded-lg p-3">
                        {r || '—'}
                      </div>
                    ))}
                    {reviews.length > 8 && (
                      <div className="text-xs text-google-gray-600">Showing 8 of {reviews.length} reviews.</div>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
