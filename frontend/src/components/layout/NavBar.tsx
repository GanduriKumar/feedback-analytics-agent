import { Link, NavLink } from 'react-router-dom';
import { BarChart3, FileText, Activity, Settings } from 'lucide-react';

const nav = [
  { to: '/', label: 'Dashboard', icon: BarChart3 },
  { to: '/analyze', label: 'Extract & Analyze', icon: Activity },
  { to: '/reports', label: 'Reports', icon: FileText },
];

export function NavBar() {
  return (
    <header className="sticky top-0 z-50 bg-white border-b border-google-gray-200">
      <div className="max-w-7xl mx-auto px-6">
        <div className="h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <Activity className="w-7 h-7 text-google-blue-500" />
            <div className="leading-tight">
              <div className="font-semibold text-google-gray-900">Feedback Analytics</div>
              <div className="text-xs text-google-gray-600">Agent UI</div>
            </div>
          </Link>

          <nav className="flex items-center gap-1">
            {nav.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  [
                    'inline-flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-google-blue-50 text-google-blue-700'
                      : 'text-google-gray-700 hover:bg-google-gray-50',
                  ].join(' ')
                }
              >
                <Icon className="w-4 h-4" />
                {label}
              </NavLink>
            ))}
          </nav>

          <button
            type="button"
            className="p-2 rounded-lg text-google-gray-700 hover:bg-google-gray-50"
            title="Settings (coming soon)"
            aria-label="Settings"
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </div>
    </header>
  );
}
