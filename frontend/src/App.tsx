import { Navigate, Route, Routes } from 'react-router-dom';
import { NavBar } from './components/layout/NavBar';
import { Dashboard } from './pages/Dashboard';
import { ExtractAnalyze } from './pages/ExtractAnalyze';
import { Reports } from './pages/Reports';

export default function App() {
  return (
    <div className="min-h-screen">
      <NavBar />
      <main>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyze" element={<ExtractAnalyze />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
