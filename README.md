# Feedback Analytics Agent

An intelligent agent system for extracting actionable intelligence from end-user reviews, surveys, and feedback using advanced NLP techniques, vector databases, and AI-powered analysis.

## Overview

This project leverages AI and machine learning to automatically analyze large volumes of user feedback, identify themes, cluster similar reviews, and extract meaningful insights to inform product and business decisions.

## Features

- **Automated Data Collection**: Scrape and aggregate feedback from multiple sources
- **Vector Database Storage**: Store feedback embeddings in ChromaDB for efficient similarity search
- **Intelligent Clustering**: Group similar feedback using advanced clustering algorithms
- **Theme Extraction**: Automatically identify recurring themes and topics
- **Sentiment Analysis**: Analyze sentiment and emotional tone of feedback
- **AI-Powered Insights**: Generate actionable recommendations using LLM analysis
- **Custom Pipelines**: Flexible pipeline architecture for custom analysis workflows

## Project Structure

```
feedback-analytics-agent/
├── chroma_db/                      # ChromaDB vector database storage
│   └── chroma.sqlite3
├── custom_apis.py                  # Custom API integrations
├── custom_pipeline.py              # Custom analysis pipelines
├── feedback_analysis.py            # Core feedback analysis logic
├── feedback_analyzer.py            # Main analyzer implementation
├── query_vectorDB.py               # Vector database query interface
├── review_analyzer_agent.py        # Review analysis agent
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (API keys, etc.)
└── Data files:
    ├── all_posts.csv/json         # Raw collected posts
    ├── cleaned_reviews.csv/json   # Preprocessed reviews
    ├── clustered_reviews.csv/json # Clustered feedback data
    ├── clusters.csv/json          # Cluster metadata
    ├── curated_reviews.csv/json   # Curated/filtered reviews
    ├── themes.csv/json            # Extracted themes
    └── feedback_analysis_results.csv/json  # Final analysis results
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- API keys for LLM services (OpenAI, Anthropic, etc.)

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

4. Configure environment variables:
    ```bash
    cp .env.example .env
    # Edit .env with your API keys and configuration
    ```

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key (if using Claude)
- Additional API keys as needed

## Usage

### Basic Analysis Pipeline

```python
from feedback_analyzer import FeedbackAnalyzer

# Initialize analyzer
analyzer = FeedbackAnalyzer()

# Load feedback data
analyzer.load_data('all_posts.csv')

# Run complete analysis pipeline
results = analyzer.analyze_all()

# Export results
analyzer.export_results('feedback_analysis_results.json')
```

### Query Vector Database

```python
from query_vectorDB import query_similar_feedback

# Find similar feedback
similar = query_similar_feedback(
    query="battery life issues",
    n_results=10
)
```

### Custom Analysis Pipeline

```python
from custom_pipeline import CustomPipeline

# Create custom pipeline
pipeline = CustomPipeline()
pipeline.add_step('clean', clean_data)
pipeline.add_step('cluster', cluster_feedback)
pipeline.add_step('analyze', analyze_themes)

# Run pipeline
results = pipeline.execute(data)
```

### Review Analyzer Agent

```python
from review_analyzer_agent import ReviewAnalyzerAgent

# Initialize agent
agent = ReviewAnalyzerAgent()

# Analyze specific reviews
insights = agent.analyze_reviews(review_ids=[1, 2, 3])
```

## Core Components

### 1. Data Collection & Preprocessing
- **[`custom_apis.py`](custom_apis.py)**: Custom API integrations for data sources
- Cleans and normalizes feedback text
- Removes duplicates and irrelevant content

### 2. Vector Database (ChromaDB)
- **[`query_vectorDB.py`](query_vectorDB.py)**: Interface for vector similarity search
- Stores feedback embeddings for efficient retrieval
- Enables semantic search across feedback

### 3. Clustering & Theme Extraction
- Groups similar feedback using clustering algorithms
- Identifies recurring themes and topics
- Generates cluster summaries

### 4. AI-Powered Analysis
- **[`feedback_analyzer.py`](feedback_analyzer.py)**: Main analysis engine
- **[`review_analyzer_agent.py`](review_analyzer_agent.py)**: Specialized review analysis
- Sentiment analysis
- Root cause identification
- Actionable insight generation

### 5. Custom Pipelines
- **[`custom_pipeline.py`](custom_pipeline.py)**: Extensible pipeline framework
- Build custom analysis workflows
- Chain multiple analysis steps

## Data Flow

```
Raw Feedback (all_posts.csv)
    ↓
Cleaning & Preprocessing (cleaned_reviews.csv)
    ↓
Vector Embedding & Storage (chroma_db/)
    ↓
Clustering (clustered_reviews.csv, clusters.csv)
    ↓
Theme Extraction (themes.csv)
    ↓
AI Analysis & Insights (feedback_analysis_results.csv)
    ↓
Actionable Recommendations
```

## Output Files

- **`cleaned_reviews.csv/json`**: Preprocessed and normalized feedback
- **`clustered_reviews.csv/json`**: Feedback with cluster assignments
- **`clusters.csv/json`**: Cluster metadata and summaries
- **`themes.csv/json`**: Extracted themes with frequencies
- **`curated_reviews.csv/json`**: High-quality, curated feedback
- **`feedback_analysis_results.csv/json`**: Complete analysis results with insights

## Configuration

Configuration options can be set in [`.env`](.env) or passed programmatically:

```python
config = {
    'embedding_model': 'text-embedding-3-small',
    'llm_model': 'gpt-4',
    'cluster_min_size': 3,
    'similarity_threshold': 0.75,
    'max_themes': 10
}
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ChromaDB for vector database functionality
- OpenAI and Anthropic for LLM capabilities
- Community contributors and feedback

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact: [your-email@example.com]

## Roadmap

- [ ] Support for additional data sources (Twitter, Reddit, etc.)
- [ ] Real-time feedback streaming
- [ ] Advanced visualization dashboard
- [ ] Multi-language support
- [ ] Automated reporting and alerts
- [ ] Integration with product management tools

---

**Note**: Remember to keep your API keys secure and never commit [`.env`](.env) files to version control.
