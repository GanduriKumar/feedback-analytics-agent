import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { NavBar } from './components/layout/NavBar';
import { Dashboard } from './pages/Dashboard';
import { ExtractAnalyze } from './pages/ExtractAnalyze';
import { Reports } from './pages/Reports';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-google-gray-50">
        <NavBar />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyze" element={<ExtractAnalyze />} />
          <Route path="/reports" element={<Reports />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
