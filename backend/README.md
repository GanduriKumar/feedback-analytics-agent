# Feedback Analytics Backend

Unified backend service providing feedback analysis, theme extraction, and semantic search capabilities through RESTful APIs.

## Architecture

```
backend/
├── app/
│   ├── main.py              # Main FastAPI application
│   ├── __init__.py
│   ├── api/                 # API route modules
│   │   └── __init__.py
│   ├── core/                # Core business logic
│   │   ├── __init__.py
│   │   ├── analyzer.py      # LangGraph pipeline
│   │   ├── vector_db.py     # ChromaDB operations
│   │   └── pipeline.py      # Data collection pipeline
│   ├── models/              # Pydantic models
│   │   ├── __init__.py
│   │   └── schemas.py       # Request/response schemas
│   ├── tools/               # Analysis tools
│   │   ├── __init__.py
│   │   ├── custom_llm.py
│   │   └── custom_tools.py
│   └── utilities/           # Helper modules
│       ├── __init__.py
│       ├── reddit_handler.py
│       ├── review_clustering.py
│       ├── review_summarizer.py
│       └── theme_issue_classifier.py
├── config/
│   └── search_queries.csv   # Reddit search configuration
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (not in git)
└── README.md               # This file
```

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file in the backend directory:

```env
# Reddit API
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=FeedbackAnalytics/1.0

# API Security
API_KEY=your_secure_api_key_here

# Ollama LLM
BASE_URL=http://localhost:11434
```

### 3. Start Ollama

```bash
ollama serve
ollama pull mistral
```

### 4. Run the Backend

```bash
cd backend/app
python main.py
```

Or with uvicorn (faster reload watching only code/config):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app --reload-dir config
```

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

## API Endpoints

### Public Endpoints

- `GET /` - Service information
- `GET /api/health` - Health check
- `GET /api/capabilities` - Capability discovery

### Authenticated Endpoints

All require `X-API-Key` header or `x_api_key` query parameter:

#### Analysis
- `POST /api/analyze` - Full pipeline analysis
  ```json
  {
    "query": "Pixel 9 battery issues",
    "n_results": 50
  }
  ```

#### Search
- `POST /api/search` - Semantic search
  ```json
  {
    "query": "camera problems",
    "n_results": 100
  }
  ```

#### Tools
- `GET /api/tools/collect` - Fetch Reddit reviews
- `POST /api/tools/cluster` - Cluster reviews by similarity

## Authentication

Include API key in requests:

**Header:**
```
X-API-Key: your_api_key_here
```

**Query Parameter:**
```
http://localhost:8000/api/analyze?x_api_key=your_api_key_here
```

## Rate Limiting

- 30 requests per minute per client IP
- 429 status code when exceeded

## CORS Configuration

Frontend origins allowed:
- `http://localhost:3000`
- `http://localhost:3001`
- `http://127.0.0.1:3000`
- `http://127.0.0.1:3001`

## Development

### Running in Development Mode

```bash
uvicorn app.main:app --reload --reload-dir app --reload-dir config --port 8000
```

### Testing Endpoints

```bash
# Health check
curl http://localhost:8000/api/health

# With authentication
curl -H "X-API-Key: your_key" http://localhost:8000/api/tools/collect

# POST request
curl -X POST http://localhost:8000/api/search \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "battery drain", "n_results": 50}'
```

## Logging

Logs are written to:
- `backend_api.log` - Application logs
- Console (stdout) - Real-time logging

## Security Features

- API key authentication
- Rate limiting
- Input validation and sanitization
- Path traversal prevention
- CORS restrictions
- Request/response logging
- Secure file operations

## Environment Variables

Required:
- `REDDIT_CLIENT_ID` - Reddit app client ID
- `REDDIT_CLIENT_SECRET` - Reddit app secret
- `REDDIT_USER_AGENT` - Reddit API user agent
- `API_KEY` - Backend API authentication key

Optional:
- `BASE_URL` - Ollama server URL (default: http://localhost:11434)
- `SUBREDDITS` - JSON array of subreddit names
- `TIME_FILTER` - Reddit time filter (day/week/month/year)
- `NUM_POSTS` - Posts per query (default: 100)
- `CHROMA_DB_PATH` - Vector DB path (default: ./chroma_db)

## Production Deployment

For production:

1. Use strong API keys
2. Configure proper CORS origins
3. Enable HTTPS
4. Set up reverse proxy (nginx/traefik)
5. Use production ASGI server (gunicorn + uvicorn)
6. Configure logging to external service
7. Set up monitoring and alerts

Example production command:

```bash
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

## Troubleshooting

### Import Errors

Ensure you're running from the correct directory:
```bash
cd backend
python -m app.main
```

### Ollama Connection Failed

Check Ollama is running:
```bash
ollama list
curl http://localhost:11434/api/version
```

### ChromaDB Not Found

Run the data collection pipeline first:
```bash
cd ..  # Return to project root
python custom_pipeline.py
```

## Support

For issues or questions:
- Check logs in `backend_api.log`
- Review API docs at `/api/docs`
- Verify environment variables in `.env`
