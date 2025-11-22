# Feedback Analytics Agent

An intelligent A2A-compatible agent system for extracting actionable intelligence from end-user reviews, surveys, and feedback using advanced NLP techniques, vector databases, and AI-powered analysis.

## Overview

This project leverages AI and machine learning to automatically analyze large volumes of user feedback from Reddit and other sources, identify themes, cluster similar reviews, and extract meaningful insights to inform product and business decisions. The system now includes an A2A (Agent-to-Agent) compatible interface for seamless integration with other AI agents.

## Features

- **A2A Compatible Interface**: Standardized REST API for agent-to-agent communication and discovery
- **Automated Data Collection**: Scrape and aggregate feedback from Reddit using PRAW
- **Vector Database Storage**: Store feedback embeddings in ChromaDB for efficient similarity search
- **Intelligent Clustering**: Group similar feedback using sentence transformers and ML algorithms
- **Theme Extraction**: Automatically identify recurring themes using LLM-based classification
- **Sentiment Analysis**: Analyze sentiment and emotional tone using VADER and TextBlob
- **AI-Powered Insights**: Generate actionable recommendations using LangChain and Ollama
- **Secure REST API**: FastAPI endpoints with authentication, rate limiting, and audit logging
- **Custom Pipelines**: Flexible LangGraph-based pipeline architecture for custom analysis workflows

## Project Structure

```
feedback-analytics-agent/
├── src/
│   ├── tools/
│   │   ├── custom_tools.py         # Core analysis tools
│   │   └── custom_llm.py           # Ollama LLM integration
│   └── utilities/
│       ├── reddit_handler.py       # Reddit data collection
│       ├── theme_issue_classifier.py # LLM theme extraction
│       ├── review_clustering.py    # ML-based clustering
│       └── review_summarizer.py    # Review summarization
├── chroma_db/                      # ChromaDB vector database storage
├── config/
│   └── search_queries.csv          # Reddit search queries
├── a2a_review_analyzer_agent.py    # A2A-compatible agent API
├── custom_apis.py                  # Secure FastAPI endpoints
├── custom_pipeline.py              # Custom analysis pipelines
├── feedback_analysis.py            # Simple analysis endpoint
├── feedback_analyzer.py            # LangGraph-based analyzer
├── query_vectorDB.py               # Vector database query interface
├── review_analyzer_agent.py        # LangChain agent for review analysis
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (API keys, etc.)
├── .gitignore                     # Git ignore patterns
└── Data files (generated):
    ├── all_posts.csv/json         # Raw collected posts
    ├── cleaned_reviews.csv/json   # Preprocessed reviews
    ├── clustered_reviews.csv/json # Clustered feedback data
    ├── clusters.csv/json          # Cluster metadata
    ├── curated_reviews.csv/json   # Curated/filtered reviews
    ├── themes.csv/json            # Extracted themes
    ├── feedback_analysis_results.csv/json # Final analysis results
    └── search_results.csv         # Vector DB query results
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Ollama (for local LLM inference)
- Reddit API credentials (for data collection)

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/feedback-analytics-agent.git
    cd feedback-analytics-agent
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Install and start Ollama:
    ```bash
    # Visit https://ollama.ai for installation instructions
    ollama pull mistral  # or your preferred model
    ollama serve
    ```

5. Configure environment variables in `.env`:
    ```env
    # Reddit API Configuration
    REDDIT_CLIENT_ID=your_client_id
    REDDIT_CLIENT_SECRET=your_client_secret
    REDDIT_USER_AGENT=FeedbackAnalyzer/1.0
    TIME_FILTER=month
    NUM_POSTS=100
    
    # LLM Configuration
    BASE_URL=http://localhost:11434
    INFERENCE_MODEL=mistral:latest
    VISION_MODEL=llava:latest
    EMBEDDING_MODEL=all-MiniLM-L6-v2
    MODEL_TEMPERATURE=0.7
    MODEL_MAX_TOKENS=2000
    MODEL_TOP_K=50
    
    # Vector Database
    CHROMA_DB_PATH=./chroma_db
    REVIEW_COLLECTION_NAME=reviews
    EMBEDDING_BATCH_SIZE=100
    UPSERT_BATCH_SIZE=500
    
    # API Security
    API_KEY=your_secure_api_key  # Auto-generated if not set
    
    # Optional: External LLM APIs
    OPENAI_API_KEY=your_openai_key
    ANTHROPIC_API_KEY=your_anthropic_key
    ```

6. Configure Reddit search queries in [`config/search_queries.csv`](config/search_queries.csv):
    ```csv
    queries
    Pixel Vs iPhone
    Android battery issues
    iOS camera problems
    ```

## Usage

### 1. A2A-Compatible Agent API (Recommended)

Start the A2A-compatible agent server:

```bash
python a2a_review_analyzer_agent.py
```

The agent will be available at `http://127.0.0.1:8001` with standardized A2A endpoints.

#### A2A Endpoints

**Health Check & Discovery** (No auth required)
```bash
curl http://127.0.0.1:8001/health
```

**Get Agent Capabilities**
```bash
curl http://127.0.0.1:8001/capabilities
```

**Analyze Feedback Themes (GET)**
```bash
# Browser-friendly GET endpoint
http://localhost:8001/themes?query=Pixel%20battery%20issues&n_results=50&x_api_key=YOUR_API_KEY
```

**Analyze Feedback (POST)**
```bash
curl -X POST http://127.0.0.1:8001/analyze \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Pixel battery issues",
    "n_results": 50
  }'
```

**Search Reviews Semantically**
```bash
curl -X POST http://127.0.0.1:8001/search \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "camera quality complaints",
    "n_results": 20
  }'
```

API Documentation: `http://127.0.0.1:8001/docs`

### 2. Secure REST API

Start the FastAPI server with built-in security:

```bash
python custom_apis.py
```

The API will be available at `http://127.0.0.1:8000` with the following endpoints:

#### Authentication
All endpoints (except `/health`) require API key authentication via:
- Header: `X-API-Key: your_api_key`
- Query parameter: `?x_api_key=your_api_key`

#### Endpoints

**Health Check** (No auth required)
```bash
curl http://127.0.0.1:8000/health
```

**Fetch Raw Reviews**
```bash
curl -H "X-API-Key: your_api_key" http://127.0.0.1:8000/reviews
```

**Get Summarized Reviews**
```bash
curl -H "X-API-Key: your_api_key" http://127.0.0.1:8000/summarized_reviews
```

**Compute Clusters**
```bash
curl -H "X-API-Key: your_api_key" http://127.0.0.1:8000/clusters
```

**Extract Themes**
```bash
curl -H "X-API-Key: your_api_key" http://127.0.0.1:8000/themes
```

API Documentation: `http://127.0.0.1:8000/docs`

### 3. LangGraph Pipeline

Execute the complete analysis pipeline with state management:

```python
from feedback_analyzer import execute_graph_pipeline

# Run the pipeline with your product query
user_query = "What are users saying about Pixel battery life?"
themes = execute_graph_pipeline(user_query)

# Results saved to feedback_analysis_results.json
print(f"Extracted {len(themes)} themes")
```

### 4. Simple Analysis Endpoint

Run a single-endpoint analysis:

```bash
python feedback_analysis.py
```

Then access: `http://127.0.0.1:8000/analyze_feedback`

### 5. LangChain Agent

Use the review analyzer agent for interactive analysis:

```python
from review_analyzer_agent import main

# Interactive agent-based analysis
main()
```

Interactive prompt example:
```
Enter your request for analyzing customer reviews:
> Analyze Pixel camera complaints

Analysis complete. Results saved to: review_analysis_output.json
Total themes extracted: 15
```

### 6. Vector Database Query

Search for similar feedback using semantic search:

```python
from query_vectorDB import query_vector_db

# Find similar reviews
results = query_vector_db(
    query_text="battery drain issues",
    n_results=10,
    output_file="search_results.csv"
)

print(f"Found {len(results)} similar reviews")
```

### 7. Custom Pipeline

Build and populate the vector database:

```bash
python custom_pipeline.py
```

This will:
1. Fetch Reddit reviews based on configured queries
2. Clean and preprocess the text
3. Generate embeddings using sentence transformers
4. Store embeddings in ChromaDB for similarity search

## Core Components

### 1. Data Collection & Preprocessing

- **[`src/utilities/reddit_handler.py`](src/utilities/reddit_handler.py)**: Reddit API integration using PRAW
  - Concurrent post fetching with thread pools
  - Automatic deduplication
  - CSV/JSON export with proper quoting
  - Configurable search queries and filters
  
- **[`src/tools/custom_tools.py`](src/tools/custom_tools.py)**: Core analysis utilities
  - `fetch_reddit_reviews()`: Collect posts from configured subreddits
  - `clean_reviews()`: Text preprocessing and normalization
  - `assess_clusters()`: ML-based review clustering
  - `summarize_clusters()`: LLM-based cluster summarization
  - `extract_themes()`: Theme extraction and classification

### 2. Vector Database (ChromaDB)

- **[`query_vectorDB.py`](query_vectorDB.py)**: Semantic similarity search
  - Custom embedding model integration (Ollama)
  - Input sanitization and validation
  - Secure file operations
  - Configurable result limits (max 1000)
  - Path traversal prevention
  
- **[`custom_pipeline.py`](custom_pipeline.py)**: Vector database population
  - Batch processing for efficiency
  - Duplicate detection using content hashing
  - Chunked upsertion for large datasets
  - Progress tracking and error handling

### 3. Clustering & Theme Extraction

- **Clustering**: Uses sentence transformers for semantic grouping
  - [`src/utilities/review_clustering.py`](src/utilities/review_clustering.py)
  - DBSCAN algorithm for automatic cluster detection
  - Semantic similarity-based grouping
  
- **Theme Classification**: [`src/utilities/theme_issue_classifier.py`](src/utilities/theme_issue_classifier.py)
  - LLM-based theme extraction using Ollama
  - Structured output using Pydantic models
  - Sentiment analysis integration
  - Product-specific categorization
  - Async processing support

### 4. AI-Powered Analysis

- **[`feedback_analyzer.py`](feedback_analyzer.py)**: LangGraph state machine
  - Multi-node analysis pipeline:
    1. Review Extraction (Vector DB query)
    2. Cluster Assessment (ML-based grouping)
    3. Cluster Summarization (LLM synthesis)
    4. Theme Extraction (LLM classification)
  - State persistence between steps
  - Secure file operations with restricted permissions
  - Visual graph generation
  
- **[`review_analyzer_agent.py`](review_analyzer_agent.py)**: LangChain agent
  - Tool-based architecture
  - JSON response validation
  - Environment variable validation
  - Interactive command-line interface
  - Comprehensive error handling

### 5. A2A-Compatible Agent Layer

- **[`a2a_review_analyzer_agent.py`](a2a_review_analyzer_agent.py)**: Agent-to-Agent API
  - **Standardized Interface**: RESTful API with structured request/response models
  - **Agent Discovery**: `/health` and `/capabilities` endpoints
  - **Security Features**:
    - API key authentication (header + query param)
    - Input validation with Pydantic V2
    - Path traversal prevention
    - Secure file permissions (0o600)
    - Comprehensive error handling
  - **Async Processing**: Thread pool execution for blocking operations
  - **Multiple Access Methods**: GET and POST endpoints for flexibility
  - **Structured Responses**: Type-safe models with metadata

### 6. Secure API Layer

- **[`custom_apis.py`](custom_apis.py)**: Production-ready FastAPI server
  - **Security Features**:
    - API key authentication (header + query param)
    - Rate limiting (10 requests/60s per IP)
    - Request logging and audit trail
    - Path traversal prevention
    - Secure file permissions (0o600)
    - Input sanitization
    - CORS protection
    - Trusted host middleware
  - **Async Processing**: Thread pool execution for blocking operations
  - **Error Handling**: Comprehensive exception handling with proper HTTP status codes
  - **Middleware Stack**:
    - Request logging
    - Rate limiting
    - CORS configuration
    - Trusted hosts validation

## Data Flow

```
User Query
    ↓
Reddit Data Collection (all_posts.csv/json)
    ↓
Text Cleaning & Preprocessing (cleaned_reviews.csv/json)
    ↓
Vector Embedding Generation (Sentence Transformers)
    ↓
ChromaDB Storage & Indexing
    ↓
Semantic Search & Review Extraction
    ↓
ML-Based Clustering (clustered_reviews.csv/json, clusters.csv/json)
    ↓
Cluster Summarization (curated_reviews.csv/json)
    ↓
LLM Theme Extraction (themes.csv/json)
    ↓
Final Analysis Results (feedback_analysis_results.csv/json)
    ↓
Structured JSON Output via A2A API
    ↓
Actionable Insights & Recommendations
```

## Output Files

All output files are generated in both CSV and JSON formats:

- **`all_posts.csv/json`**: Raw Reddit posts with metadata (title, text, subreddit, timestamp)
- **`cleaned_reviews.csv/json`**: Preprocessed and normalized feedback (special chars removed)
- **`clustered_reviews.csv/json`**: Feedback with cluster assignments and similarity scores
- **`clusters.csv/json`**: Cluster metadata, summaries, and representative reviews
- **`curated_reviews.csv/json`**: High-quality, representative reviews per cluster
- **`themes.csv/json`**: Extracted themes with categories, sentiment, and frequencies
- **`feedback_analysis_results.csv/json`**: Complete analysis with insights and recommendations
- **`search_results.csv`**: Vector database query results with similarity scores
- **`a2a_analysis_results.json`**: A2A-compatible analysis output
- **`a2a_themes_results.json`**: A2A-compatible themes output

**Security**: All files are written with restricted permissions (owner read/write only, chmod 0o600)

## Configuration

### Environment Variables

Configure in `.env` file:

```env
# Required
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=FeedbackAnalyzer/1.0
BASE_URL=http://localhost:11434
INFERENCE_MODEL=mistral:latest
REVIEW_COLLECTION_NAME=reviews

# Optional
API_KEY=custom_api_key  # Auto-generated if not set
TIME_FILTER=month
NUM_POSTS=100
EMBEDDING_MODEL=all-MiniLM-L6-v2
MODEL_TEMPERATURE=0.7
MODEL_MAX_TOKENS=2000
CHROMA_DB_PATH=./chroma_db
EMBEDDING_BATCH_SIZE=100
UPSERT_BATCH_SIZE=500
```

### Reddit Search Queries

Configure target subreddits and search terms in [`config/search_queries.csv`](config/search_queries.csv):

```csv
queries
Pixel Vs iPhone
Android battery problems
iOS camera quality
Smartphone performance issues
```

Default subreddits: `GooglePixel`, `Pixel`, `Google`, `pixel_phones`, `Smartphones`, `Android`, `apple`, `applesucks`, `iphone`

### Analysis Parameters

Customize in [`feedback_analyzer.py`](feedback_analyzer.py):

```python
# Vector DB query settings
n_results = 50  # Number of reviews to retrieve

# Clustering parameters (in review_clustering.py)
config = {
    'embedding_model': 'all-MiniLM-L6-v2',
    'cluster_min_size': 3,
    'similarity_threshold': 0.75,
    'max_clusters': 20,
}
```

### API Security

Configure in [`custom_apis.py`](custom_apis.py) and [`a2a_review_analyzer_agent.py`](a2a_review_analyzer_agent.py):

```python
# Rate limiting
RateLimiter(max_requests=10, window_seconds=60)

# Server configuration
uvicorn.run(
    app,
    host="127.0.0.1",  # Localhost only
    port=8000,  # or 8001 for A2A agent
    limit_concurrency=10,
    limit_max_requests=1000,
    timeout_keep_alive=5
)
```

## Security Features

1. **API Authentication**: Required API keys for all sensitive endpoints
2. **Rate Limiting**: Per-IP request throttling to prevent abuse (10 req/60s)
3. **Input Validation**: Sanitization of all user inputs using Pydantic V2
4. **Path Traversal Protection**: Validates all file paths within workspace
5. **Secure File Operations**: Restricted permissions (0o600) on all output files
6. **Audit Logging**: Comprehensive request/response logging with timestamps
7. **Error Handling**: No sensitive data exposed in error messages
8. **CORS Protection**: Localhost-only binding by default
9. **Query Sanitization**: Removes control characters and validates patterns
10. **A2A Security**: Dedicated authentication for agent-to-agent communication

## Dependencies

Key libraries (see [`requirements.txt`](requirements.txt)):

- **LLM & NLP**: `langchain`, `langchain-ollama`, `langgraph`, `sentence-transformers`, `dspy`
- **Vector Database**: `chromadb`, `langchain-chroma`
- **Data Collection**: `praw` (Reddit API)
- **API Framework**: `fastapi`, `uvicorn`, `pydantic`
- **ML & Analytics**: `scikit-learn`, `pandas`, `numpy`, `nltk`
- **Sentiment Analysis**: `vaderSentiment`, `textblob`
- **Summarization**: `sumy`
- **Documentation**: `sphinx`, `sphinx-autodoc-typehints`

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the existing code style and security practices
4. Add tests for new features
5. Update documentation as needed
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Quality

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Add docstrings for all functions and classes
- Implement proper error handling
- Include security considerations in all code
- Use Pydantic V2 validators for input validation
- Implement comprehensive logging

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **ChromaDB** for efficient vector database functionality
- **LangChain** and **LangGraph** for agent orchestration framework
- **Ollama** for local LLM inference
- **PRAW** for Reddit API access
- **Sentence Transformers** for semantic embeddings
- **FastAPI** for modern API framework
- Community contributors and feedback

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review API documentation at `/docs` endpoint
- Check A2A documentation at `http://localhost:8001/docs`

## Roadmap

### Planned Features
- [x] A2A-compatible agent interface
- [x] Pydantic V2 migration
- [ ] Multi-source data collection (Twitter, reviews, surveys)
- [ ] Real-time streaming analysis
- [ ] Interactive dashboard for visualization
- [ ] Advanced sentiment analysis with aspect-based detection
- [ ] Multi-language support
- [ ] Automated report generation
- [ ] Webhook notifications for critical insights
- [ ] Docker containerization
- [ ] Kubernetes deployment support

### Security Enhancements
- [x] API key authentication
- [x] Rate limiting
- [x] Input validation with Pydantic V2
- [x] Secure file operations
- [ ] OAuth2 authentication
- [ ] Role-based access control (RBAC)
- [ ] API key rotation
- [ ] Encrypted data storage
- [ ] Audit log retention policies

### A2A Integration
- [x] Standardized discovery endpoints
- [x] Structured request/response models
- [x] Multiple authentication methods
- [ ] Agent capability negotiation
- [ ] Inter-agent communication protocols
- [ ] Agent registry integration
- [ ] Distributed task coordination

---

**Note**: This is an educational/research project. Ensure compliance with platform terms of service and data privacy regulations when collecting user feedback.
