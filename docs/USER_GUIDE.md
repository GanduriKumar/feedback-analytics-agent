# Feedback Analytics Agent - User Guide

## Welcome!

Welcome to the Feedback Analytics Agent! This guide will help you understand how to use this powerful tool to automatically analyze customer feedback, reviews, and comments from Reddit. Think of it as your personal assistant that reads thousands of reviews and tells you what people are saying about products.

## What Does This Tool Do?

Imagine you want to know what people think about the Google Pixel 9 phone compared to the iPhone 15. Instead of reading thousands of Reddit posts yourself, this tool:

1. **Collects** all relevant Reddit posts and comments
2. **Organizes** them by finding similar opinions
3. **Summarizes** what each group of people is saying
4. **Identifies** common themes (like "battery problems" or "great camera")
5. **Gives you** easy-to-read reports

## Who Is This For?

- **Product managers** who want to understand customer pain points
- **Business analysts** researching market sentiment
- **Customer support teams** looking for common issues
- **Marketing teams** understanding customer preferences
- **Anyone** interested in analyzing large amounts of feedback automatically

## Key Concepts (Explained Simply)

### What is a "Vector Database"?
Think of it as a smart filing cabinet that doesn't just store your reviews, but also understands their *meaning*. When you ask "What do people say about battery life?", it finds reviews that talk about battery life, even if they use different words like "charge duration" or "power consumption."

### What is "Clustering"?
Imagine sorting a pile of customer feedback cards into stacks based on similarity. Reviews about "camera quality" go in one pile, reviews about "battery problems" in another. The tool does this automatically using smart algorithms.

### What is an "LLM"?
LLM stands for Large Language Model - it's like a very smart reading assistant that can understand text, summarize it, and extract key information. This tool uses a local LLM called Ollama, which runs on your computer (no data sent to the cloud).

### What is "A2A"?
A2A means "Agent-to-Agent" - it's a way for different software tools to talk to each other. This means other programs can use your Feedback Analytics Agent automatically without human intervention.

## How to Use the Tool

### Method 1: Simple Command-Line Analysis

This is the easiest way to analyze feedback:

1. **Open your terminal** (Command Prompt on Windows, Terminal on Mac/Linux)

2. **Navigate to the project folder**:
   ```
   cd c:\Users\kumar.gn\PycharmProjects\feedback-analytics-agent
   ```

3. **Run the analysis pipeline**:
   ```
   python custom_pipeline.py
   ```

4. **Wait for it to complete** - you'll see progress messages like:
   - "Fetching Reddit reviews..." (collecting data)
   - "Cleaning and preprocessing..." (removing noise)
   - "Generating embeddings..." (understanding meaning)
   - "Persisting to ChromaDB..." (saving for later)

5. **Check your results** - new files will appear in your project folder:
   - `all_posts.csv` - all the reviews collected
   - `cleaned_reviews.csv` - processed reviews ready for analysis

### Method 2: Ask Specific Questions (Interactive Mode)

Want to find specific information from all the feedback you've collected?

1. **Run the query tool**:
   ```
   python query_vectorDB.py
   ```

2. **Type your question** when prompted, for example:
   - "What do people say about battery life?"
   - "Are there complaints about the camera?"
   - "What are the main problems with the Pixel 9?"

3. **Get your answers** - the tool will find the most relevant reviews and show them to you, saved in `search_results.csv`

### Method 3: Full Analysis with Themes (Advanced)

This runs the complete analysis pipeline that groups reviews, summarizes them, and extracts themes:

1. **First, make sure you have data collected** by running `custom_pipeline.py` (Method 1)

2. **Run the analysis agent**:
   ```
   python review_analyzer_agent.py
   ```

3. **Enter your question** when prompted, such as:
   - "Compare Pixel and iPhone battery performance"
   - "What are the main issues with the Pixel 9?"

4. **Review the results**:
   - `feedback_analysis_results.csv` - detailed analysis with themes
   - `a2a_themes_results.json` - structured output for other programs

### Method 4: Use as a Web Service (For Developers)

If you want other programs to use your analysis tool:

#### Starting the A2A Compatible API:

```
python -m uvicorn a2acompatible_analyzer_agent:app --host 0.0.0.0 --port 8000
```

Then other programs can send questions to: `http://localhost:8000/analyze`

#### Starting the Custom API:

```
python -m uvicorn custom_apis:app --host 0.0.0.0 --port 8001
```

This provides direct access to individual tools like clustering and theme extraction.

## Understanding the Output

### CSV Files (Spreadsheet Format)

These files can be opened in Excel, Google Sheets, or any spreadsheet program:

- **all_posts.csv**: Raw data from Reddit
  - Columns: post title, text, author, subreddit, score, etc.

- **cleaned_reviews.csv**: Processed reviews
  - Easier to read, special characters removed

- **clustered_reviews.csv**: Reviews organized by similarity
  - Each review has a "cluster" number showing which group it belongs to

- **themes.csv**: Main themes identified
  - Columns: product, sentiment (positive/negative), theme (category), issue description

- **search_results.csv**: Results from your specific questions
  - The most relevant reviews matching what you asked

### JSON Files (Structured Data)

These are for technical users or other programs:

- **a2a_themes_results.json**: Complete analysis in standardized format
- **feedback_analysis_results.json**: Detailed breakdown of all findings

## Common Use Cases

### Use Case 1: Product Comparison
**Goal**: Compare Google Pixel vs iPhone

1. Edit `config/search_queries.csv` to include:
   - "Pixel 9 vs iPhone 15"
   - "Pixel battery vs iPhone battery"
   - "Pixel camera vs iPhone camera"

2. Run `python custom_pipeline.py`

3. Run `python review_analyzer_agent.py` and ask: "Compare Pixel and iPhone"

4. Open `themes.csv` to see organized comparisons

### Use Case 2: Identify Product Issues
**Goal**: Find common problems with your product

1. Collect feedback using `python custom_pipeline.py`

2. Query specific aspects:
   ```
   python query_vectorDB.py
   ```
   Ask: "What problems do people report?"

3. Review `search_results.csv` for the top complaints

### Use Case 3: Track Sentiment Over Time
**Goal**: Understand if sentiment is improving

1. Run collection and analysis regularly (weekly/monthly)

2. Compare `themes.csv` files from different time periods

3. Look at the "sentiment" column to track changes

### Use Case 4: Support Team Intelligence
**Goal**: Help customer support understand common issues

1. Set up automated daily collection in `custom_pipeline.py`

2. Run theme extraction: `python review_analyzer_agent.py`

3. Share `themes.csv` with your support team so they know what issues to expect

## Tips for Best Results

### Getting Quality Data

1. **Choose the right subreddits**: Edit the `SUBREDDITS` setting in your `.env` file
   - For phones: GooglePixel, iPhone, Android, Smartphones
   - For software: programming, webdev, SaaS
   - For games: gaming, specific game subreddits

2. **Craft good search queries**: In `config/search_queries.csv`, be specific:
   - Good: "Pixel 9 battery drain issue"
   - Better: "Pixel 9 battery drains fast after update"

3. **Adjust the time filter**: In `.env`, set `TIME_FILTER`:
   - "week" for recent trends
   - "month" for broader patterns
   - "year" for comprehensive history

### Improving Analysis Accuracy

1. **Use more reviews**: Increase `NUM_POSTS` in `.env` (default is 100)
   - More data = better patterns
   - But slower processing

2. **Adjust cluster count**: When using tools directly, try different numbers:
   - Fewer clusters (10-15): Broad categories
   - More clusters (20-30): Fine-grained topics

3. **Refine your questions**: When querying, be specific:
   - Vague: "Tell me about phones"
   - Better: "What battery problems do Pixel users report?"

## Troubleshooting Common Issues

### "No reviews found"
- Check your Reddit API credentials in `.env`
- Verify your search queries in `config/search_queries.csv`
- Try broader search terms

### "Database not found" error
- Run `custom_pipeline.py` first to create the database
- The database is stored in the `chroma_db` folder

### "API key invalid"
- Check your `.env` file has the correct `API_KEY`
- When using the web API, include the key in your request

### "Out of memory" error
- Reduce `BATCH_SIZE` in `.env` (try 50 instead of 100)
- Process fewer reviews by lowering `NUM_POSTS`

### Ollama connection errors
- Make sure Ollama is running: `ollama serve`
- Check the correct model is installed: `ollama list`
- Verify `BASE_URL` in `.env` points to `http://localhost:11434`

## Privacy and Security

### Your Data Stays Local
- All analysis happens on your computer
- No data is sent to external AI services (unless you configure OpenAI/Anthropic)
- Reddit data is public, but you control where it's stored

### API Keys and Credentials
- Never share your `.env` file
- Never commit `.env` to version control (it's in `.gitignore`)
- Change the default `API_KEY` to something secure

### Rate Limiting
- The tool respects Reddit's rate limits automatically
- Web APIs have built-in rate limiting (10 requests per minute)

## Getting Help

### Check the Logs
Most scripts show progress messages. If something fails, read the last few lines - they usually explain what went wrong.

### File Locations
- Configuration: `.env` and `config/search_queries.csv`
- Data storage: `chroma_db/` folder
- Results: Root folder (CSV and JSON files)
- Logs: `api_security.log` for API usage

### Common File Issues
- If CSV files won't open: Try opening with Notepad first, then import to Excel
- If JSON files look messy: Use an online JSON formatter (search "JSON formatter")

## Next Steps

Once you're comfortable with the basics:

1. **Automate regular collection**: Set up a scheduled task to run `custom_pipeline.py` daily
2. **Create dashboards**: Import CSV files into Power BI or Tableau for visualization
3. **Integrate with other tools**: Use the A2A API to connect with your existing systems
4. **Experiment with different LLM models**: Try different Ollama models for varied analysis styles

## Glossary

- **Embedding**: A numerical representation of text that captures its meaning
- **Semantic Search**: Finding similar content by meaning, not just keywords
- **Pipeline**: A series of automated steps that process data from start to finish
- **Cluster**: A group of similar items grouped together automatically
- **Theme**: A recurring topic or category found in feedback
- **Sentiment**: Whether the feedback is positive, negative, or neutral
- **API**: A way for programs to communicate (Application Programming Interface)
- **Endpoint**: A specific function or service available through an API
- **Query**: A question or search term you use to find information

Remember: Don't be intimidated by the technical terms. Think of this tool as a smart assistant that reads reviews for you and tells you what matters most!
