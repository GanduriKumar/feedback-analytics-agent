import { useMemo, useState } from 'react';
import { Link, NavLink } from 'react-router-dom';
import { BarChart3, Activity, Settings, Menu, X } from 'lucide-react';

const nav = [
  { to: '/', label: 'Dashboard', icon: BarChart3 },
  { to: '/analyze', label: 'Extract & Analyze', icon: Activity },
];

export function NavBar() {
  const [mobileOpen, setMobileOpen] = useState(false);

  const navItems = useMemo(() => nav, []);

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

          {/* Desktop nav */}
          <nav className="hidden md:flex items-center gap-1" aria-label="Primary">
            {navItems.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  [
                    'inline-flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                    'focus:outline-none focus-visible:ring-2 focus-visible:ring-google-blue-500 focus-visible:ring-offset-2',
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

          <div className="flex items-center gap-1">
            <button
              type="button"
              className="hidden md:inline-flex p-2 rounded-lg text-google-gray-700 hover:bg-google-gray-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-google-blue-500 focus-visible:ring-offset-2"
              title="Settings (coming soon)"
              aria-label="Settings"
            >
              <Settings className="w-5 h-5" />
            </button>

            {/* Mobile menu toggle */}
            <button
              type="button"
              className="md:hidden inline-flex p-2 rounded-lg text-google-gray-700 hover:bg-google-gray-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-google-blue-500 focus-visible:ring-offset-2"
              aria-label={mobileOpen ? 'Close navigation menu' : 'Open navigation menu'}
              aria-expanded={mobileOpen}
              aria-controls="mobile-nav"
              onClick={() => setMobileOpen((v) => !v)}
            >
              {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile nav panel */}
        {mobileOpen && (
          <div id="mobile-nav" className="md:hidden pb-4" aria-label="Mobile primary navigation">
            <div className="mt-2 rounded-xl border border-google-gray-200 bg-white shadow-sm p-2">
              <div className="grid gap-1">
                {navItems.map(({ to, label, icon: Icon }) => (
                  <NavLink
                    key={to}
                    to={to}
                    end={to === '/'}
                    onClick={() => setMobileOpen(false)}
                    className={({ isActive }) =>
                      [
                        'inline-flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                        'focus:outline-none focus-visible:ring-2 focus-visible:ring-google-blue-500 focus-visible:ring-offset-2',
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

                <div className="border-t border-google-gray-200 my-1" />

                <button
                  type="button"
                  className="inline-flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium text-google-gray-500 cursor-not-allowed"
                  title="Settings (coming soon)"
                  aria-label="Settings (coming soon)"
                  disabled
                >
                  <Settings className="w-4 h-4" />
                  Settings (soon)
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}
