# Feedback Analytics Frontend

## Pages
- **Dashboard** - View last analysis run summary
- **Extract & Analyze** - Configure and run analysis pipeline with LLM settings
- **Reports** - View detailed reports and download as PDF/CSV

## Quick Start
```bash
npm install
npm run dev
```

## Features
- Multi-user type support (PM, Engineer, Support, Analyst, Executive)
- Real-time pipeline progress tracking
- LLM configuration (Ollama, OpenAI, Anthropic, Gemini)
- Interactive data visualizations
- Comprehensive analysis reports
- PDF and CSV export
- Local storage persistence

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_API_KEY=dev-api-key-12345
```

## Tech Stack

- React 18 + TypeScript
- Vite (build tool)
- Tailwind CSS with Google Material Design colors
- React Router v6 (navigation)
- Recharts (data visualization)
- Axios (API client)
- Zustand (state management)
- jsPDF (PDF export)
- Lucide React (icons)

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── analyze/      # Extract & Analyze page components
│   │   ├── charts/       # Visualization components
│   │   ├── layout/       # NavBar and layout components
│   │   └── reports/      # Reports page components
│   ├── pages/            # Page components (Dashboard, ExtractAnalyze, Reports)
│   ├── services/         # API client
│   ├── store/            # Zustand state management
│   ├── types/            # TypeScript type definitions
│   ├── utils/            # Utility functions (export, report, etc.)
│   ├── hooks/            # Custom React hooks
│   ├── App.tsx           # Root component with routing
│   └── main.tsx          # App entry point
├── public/               # Static assets
└── package.json          # Dependencies and scripts
```

## Development

1. Clone the repository
2. Install dependencies: `npm install`
3. Copy `.env.example` to `.env` and configure
4. Start backend server (see backend README)
5. Start frontend: `npm run dev`
6. Open http://localhost:3000

## Usage

1. **Select User Type** - Choose your role (PM, Engineer, etc.)
2. **Add Search Queries** - Enter keywords to search for
3. **Select Data Sources** - Choose Reddit or other sources
4. **Configure LLM** (Optional) - Set up your preferred LLM provider
5. **Run Pipeline** - Click "Run Analysis Pipeline"
6. **View Results** - Navigate to Reports page to see detailed analysis
7. **Download** - Export reports as PDF or CSV

## Workflow Testing Checklist

### Full E2E Workflow
- [ ] Navigate to Extract & Analyze page
- [ ] Select user type (e.g., Product Manager)
- [ ] Add search queries (comma-separated or multi-line)
- [ ] Select data sources (Reddit)
- [ ] Optionally configure advanced LLM settings
- [ ] Click "Run Analysis Pipeline"
- [ ] Verify progress tracker shows all stages:
  - Fetching (10%)
  - Cleaning (30%)
  - Embedding (50%)
  - Storing (70%)
  - Analyzing (85%)
  - Complete (100%)
- [ ] See success message with review count
- [ ] Click "View Report" to navigate to Reports page

### Reports Page
- [ ] Verify overview cards display correct metrics
- [ ] Check sentiment distribution chart renders
- [ ] Check issue categories chart renders
- [ ] Verify cluster table is expandable
- [ ] Verify recommendations section displays
- [ ] Click "Download PDF" - verify PDF downloads
- [ ] Click "Download CSV" - verify CSV downloads

### Dashboard Page
- [ ] Navigate to Dashboard
- [ ] Verify last run metadata displays
- [ ] Verify overview cards show correct data
- [ ] Verify charts render correctly
- [ ] Check recent analyses section (if multiple runs)
- [ ] Click "View Full Report" - navigates to Reports
- [ ] Click "Run New Analysis" - navigates to Extract & Analyze

### Navigation
- [ ] NavBar persists across all pages
- [ ] Active route is highlighted
- [ ] All navigation links work correctly
- [ ] Browser back/forward buttons work
- [ ] Direct URL navigation works

### State Persistence
- [ ] Run an analysis
- [ ] Refresh the page
- [ ] Verify last run data persists (localStorage)
- [ ] Verify analysis history persists
- [ ] Verify user selections persist

### Error Handling
- [ ] Try running pipeline without selecting user type
- [ ] Try running pipeline without search queries
- [ ] Try running pipeline without data sources
- [ ] Verify error messages display correctly
- [ ] Test with backend down - verify error handling
- [ ] Test API timeouts - verify graceful degradation

## Known Issues & Limitations

- Backend API must be running on http://127.0.0.1:8000
- Large datasets may take several minutes to process
- PDF export is limited to 10 top issue categories
- LocalStorage has 5-10MB limit depending on browser

## Browser Support

- Chrome/Edge (recommended)
- Firefox
- Safari
- Modern browsers with ES2020+ support
