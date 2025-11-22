# Feedback Analytics Agent

An intelligent agent system for extracting actionable intelligence from end-user reviews, surveys, and feedback using advanced NLP techniques, vector databases, and AI-powered analysis.

## Overview

This project leverages AI and machine learning to automatically analyze large volumes of user feedback from Reddit and other sources, identify themes, cluster similar reviews, and extract meaningful insights to inform product and business decisions.

## Features

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
│   │   └── custom_tools.py         # Core analysis tools
│   └── utilities/
│       ├── reddit_handler.py       # Reddit data collection
│       └── theme_issue_classifier.py # LLM theme extraction
├── chroma_db/                      # ChromaDB vector database storage
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
    ollama pull llama3.2  # or your preferred model
    ```

5. Configure environment variables in `.env`:
    ```env
    # Reddit API Configuration
    REDDIT_CLIENT_ID=your_client_id
    REDDIT_CLIENT_SECRET=your_client_secret
    REDDIT_USER_AGENT=FeedbackAnalyzer/1.0
    
    # LLM Configuration
    OLLAMA_MODEL=llama3.2
    OLLAMA_BASE_URL=http://localhost:11434
    
    # Vector Database
    REVIEW_COLLECTION_NAME=reddit_reviews
    
    # API Security (auto-generated if not set)
    API_KEY=your_secure_api_key
    
    # Optional: External LLM APIs
    OPENAI_API_KEY=your_openai_key
    ANTHROPIC_API_KEY=your_anthropic_key
    ```

## Usage

### 1. Secure REST API (Recommended)

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

### 2. LangGraph Pipeline

Execute the complete analysis pipeline with state management:

```python
from feedback_analyzer import execute_graph_pipeline

# Run the pipeline with your product query
user_query = "What are users saying about our mobile app?"
themes = execute_graph_pipeline(user_query)

# Results saved to feedback_analysis_results.json
```

### 3. Simple Analysis Endpoint

Run a single-endpoint analysis:

```bash
python feedback_analysis.py
```

Then access: `http://127.0.0.1:8000/analyze_feedback`

### 4. LangChain Agent

Use the review analyzer agent for interactive analysis:

```python
from review_analyzer_agent import main

# Interactive agent-based analysis
main()
```

### 5. Vector Database Query

Search for similar feedback using semantic search:

```python
from query_vectorDB import query_vector_db

# Find similar reviews
results = query_vector_db(
    query_text="battery life complaints",
    n_results=10,
    output_file="search_results.csv"
)
```

### 6. Custom Pipeline

Build your own analysis workflow:

```python
from custom_pipeline import CustomPipeline

# Create custom pipeline
pipeline = CustomPipeline()
pipeline.add_step('collect', fetch_reddit_reviews)
pipeline.add_step('clean', clean_reviews)
pipeline.add_step('cluster', assess_clusters)
pipeline.add_step('analyze', extract_themes)

# Execute pipeline
results = pipeline.execute()
```

## Core Components

### 1. Data Collection & Preprocessing
- **[`src/utilities/reddit_handler.py`](src/utilities/reddit_handler.py)**: Reddit API integration using PRAW
  - Concurrent post fetching with thread pools
  - Automatic deduplication
  - CSV/JSON export with proper quoting
- **[`src/tools/custom_tools.py`](src/tools/custom_tools.py)**: Core analysis utilities
  - `fetch_reddit_reviews()`: Collect posts from configured subreddits
  - `clean_reviews()`: Text preprocessing and normalization
  - `assess_clusters()`: ML-based review clustering
  - `summarize_clusters()`: LLM-based cluster summarization

### 2. Vector Database (ChromaDB)
- **[`query_vectorDB.py`](query_vectorDB.py)**: Semantic similarity search
  - Custom embedding model integration
  - Input sanitization and validation
  - Secure file operations
  - Configurable result limits (max 1000)

### 3. Clustering & Theme Extraction
- **Clustering**: Uses sentence transformers for semantic grouping
- **Theme Classification**: [`src/utilities/theme_issue_classifier.py`](src/utilities/theme_issue_classifier.py)
  - LLM-based theme extraction
  - Structured output using Pydantic
  - Async processing support

### 4. AI-Powered Analysis
- **[`feedback_analyzer.py`](feedback_analyzer.py)**: LangGraph state machine
  - Multi-node analysis pipeline
  - State persistence between steps
  - Secure file operations with restricted permissions
- **[`review_analyzer_agent.py`](review_analyzer_agent.py)**: LangChain agent
  - Tool-based architecture
  - JSON response validation
  - Environment variable validation

### 5. Secure API Layer
- **[`custom_apis.py`](custom_apis.py)**: Production-ready FastAPI server
  - **Security Features**:
    - API key authentication (header + query param)
    - Rate limiting (10 requests/60s per IP)
    - Request logging and audit trail
    - Path traversal prevention
    - Secure file permissions (0o600)
    - Input sanitization
  - **Async Processing**: Thread pool execution for blocking operations
  - **Error Handling**: Comprehensive exception handling with proper HTTP status codes

## Data Flow

```
User Query
    ↓
Reddit Data Collection (all_posts.csv/json)
    ↓
Text Cleaning & Preprocessing (cleaned_reviews.csv/json)
    ↓
Vector Embedding & ChromaDB Storage
    ↓
ML-Based Clustering (clustered_reviews.csv/json, clusters.csv/json)
    ↓
Cluster Summarization (curated_reviews.csv/json)
    ↓
LLM Theme Extraction (themes.csv/json)
    ↓
Final Analysis Results (feedback_analysis_results.csv/json)
    ↓
Actionable Insights & Recommendations
```

## Output Files

All output files are generated in both CSV and JSON formats:

- **`all_posts.csv/json`**: Raw Reddit posts with metadata
- **`cleaned_reviews.csv/json`**: Preprocessed and normalized feedback
- **`clustered_reviews.csv/json`**: Feedback with cluster assignments
- **`clusters.csv/json`**: Cluster metadata and summaries
- **`curated_reviews.csv/json`**: High-quality, representative reviews per cluster
- **`themes.csv/json`**: Extracted themes with categories and frequencies
- **`feedback_analysis_results.csv/json`**: Complete analysis with insights
- **`search_results.csv`**: Vector database query results

**Security**: All files are written with restricted permissions (owner read/write only, chmod 0o600)

## Configuration

### Environment Variables

Configure in `.env` file:

```env
# Required
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=FeedbackAnalyzer/1.0
OLLAMA_MODEL=llama3.2
REVIEW_COLLECTION_NAME=reddit_reviews

# Optional
API_KEY=custom_api_key  # Auto-generated if not set
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Reddit Search Queries

Configure target subreddits and search terms in [`src/utilities/reddit_handler.py`](src/utilities/reddit_handler.py).

### Analysis Parameters

Customize in [`feedback_analyzer.py`](feedback_analyzer.py):

```python
config = {
    'embedding_model': 'all-MiniLM-L6-v2',
    'cluster_min_size': 3,
    'similarity_threshold': 0.75,
    'max_clusters': 20,
    'max_themes': 10
}
```

### API Security

Configure in [`custom_apis.py`](custom_apis.py):

```python
# Rate limiting
RateLimiter(max_requests=10, window_seconds=60)

# Server configuration
uvicorn.run(
    app,
    host="127.0.0.1",  # Localhost only
    port=8000,
    limit_concurrency=10,
    limit_max_requests=1000,
    timeout_keep_alive=5
)
```

## Security Features

1. **API Authentication**: Required API keys for all sensitive endpoints
2. **Rate Limiting**: Per-IP request throttling to prevent abuse
3. **Input Validation**: Sanitization of all user inputs
4. **Path Traversal Protection**: Validates all file paths within workspace
5. **Secure File Operations**: Restricted permissions on all output files
6. **Audit Logging**: Comprehensive request/response logging
7. **Error Handling**: No sensitive data in error messages
8. **CORS Protection**: Localhost-only binding by default

## Dependencies

Key libraries (see [`requirements.txt`](requirements.txt)):

- **LLM & NLP**: `langchain`, `langchain-ollama`, `sentence-transformers`
- **Vector Database**: `chromadb`
- **Data Collection**: `praw` (Reddit API)
- **API Framework**: `fastapi`, `uvicorn`, `pydantic`
- **ML & Analytics**: `scikit-learn`, `pandas`, `numpy`
- **Sentiment Analysis**: `vaderSentiment`, `textblob`
- **Summarization**: `sumy`

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **ChromaDB** for efficient vector database functionality
- **LangChain** for agent orchestration framework
- **Ollama** for local LLM inference
- **PRAW** for Reddit API access
- **Sentence Transformers** for semantic embeddings
- Community contributors and feedback

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review API documentation at `/docs` endpoint

## Roadmap

### Planned Features
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
- [ ] OAuth2 authentication
- [ ] Role-based access control (RBAC)
- [ ] API key rotation
- [ ] Encrypted data storage
- [ ] Audit log retention policies

---

**Note**: This is an educational/research project. Ensure compliance with platform terms of service and data privacy regulations when collecting user feedback.
