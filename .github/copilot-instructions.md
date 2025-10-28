# Feedback Analytics Agent Instructions

This is a Python-based system for analyzing user feedback, focusing on Reddit posts and comments, particularly comparing mobile device ecosystems (Pixel vs iPhone). Here's what you need to know to work effectively with this codebase:

## Architecture Overview

The system follows a pipeline architecture:
1. Data Collection (`src/tools/reddit_handler.py`) - Fetches Reddit posts using PRAW
2. Sentiment Analysis (`src/tools/sentiment_analyzer.py`) - Classifies posts using VADER
3. Classification (`src/tools/review_classification.py`) - Categorizes content using LLM
4. Theme Extraction (`src/tools/theme_issue_classifier.py`) - Identifies key themes and issues
5. Clustering (`src/tools/review_clustering.py`) - Groups similar feedback using sentence transformers

Core utilities:
- `src/utilities/custom_llm.py` - Provides Ollama integration for LLM operations
- `src/init/custom_tools.py` - Contains data pipeline initialization tools

## Key Development Patterns

1. Environment Configuration:
   - Uses `.env` file for sensitive credentials (Reddit API, Ollama settings)
   - All tools read from environment variables defined in `dotenv`

2. Data Processing Pattern:
   ```python
   # Standard flow in tools:
   input_data -> process -> DataFrame -> save_to_files(json, csv)
   ```

3. LLM Integration:
   - Always use `CustomLLMModel` from `src/utilities/custom_llm.py`
   - Model initialization pattern:
   ```python
   model = CustomLLMModel()
   client = model.getclientinterface()
   ```

## Development Workflow

1. Setup:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Configuration:
   - Copy `.env.example` to `.env` (if exists)
   - Required env vars: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

3. Running the pipeline:
   ```python
   python test_llm.py
   ```

## Integration Points

1. Reddit API:
   - Uses PRAW library with custom authentication
   - See `RedditHandler` in `src/tools/reddit_handler.py`

2. Ollama LLM:
   - Primary LLM interface through `CustomLLMModel`
   - Supports both completion and chat modes

3. File I/O:
   - Each component saves intermediate results as both JSON and CSV
   - Results stored in workspace root (see .gitignore for patterns)

## Project-Specific Conventions

1. Error Handling:
   - Each component handles its own exceptions
   - Failed operations should log errors but not crash pipeline

2. Data Formatting:
   - Uses pandas DataFrame as primary data structure
   - Always exports both JSON and CSV with consistent naming

3. Code Organization:
   - Tools in `src/tools/` for specific analysis tasks
   - Common utilities in `src/utilities/`
   - Pipeline initialization in `src/init/`

## Key Files

- `test_llm.py` - Main pipeline orchestration
- `src/tools/theme_issue_classifier.py` - Core classification logic
- `src/utilities/custom_llm.py` - LLM integration
- `requirements.txt` - Project dependencies