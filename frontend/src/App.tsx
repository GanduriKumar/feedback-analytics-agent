import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { NavBar } from './components/layout/NavBar';
import { Dashboard } from './pages/Dashboard';
import { ExtractAnalyze } from './pages/ExtractAnalyze';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-google-gray-50">
        <NavBar />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyze" element={<ExtractAnalyze />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
