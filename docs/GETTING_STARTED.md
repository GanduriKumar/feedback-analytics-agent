# Getting Started with Feedback Analytics Agent

## Quick Start Guide

This guide will walk you through setting up and running the Feedback Analytics Agent from scratch. Follow these steps in order, and you'll be analyzing feedback in about 15 minutes!

## Prerequisites (What You Need First)

### Required Software

1. **Python 3.8 or higher**
   - Check if you have it: Open terminal and type `python --version`
   - If not installed: Download from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

2. **Git** (optional, for cloning the repository)
   - Check if you have it: Type `git --version` in terminal
   - If not installed: Download from [git-scm.com](https://git-scm.com/downloads)

3. **Ollama** (for AI analysis)
   - Download from [ollama.ai](https://ollama.ai)
   - Follow the installation instructions for your operating system

### Required Accounts

1. **Reddit API Account** (free)
   - You need this to collect feedback from Reddit
   - We'll show you how to set this up below

## Step 1: Install Python and Dependencies

### 1.1 Verify Python Installation

Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and check:

```powershell
python --version
```

You should see something like `Python 3.11.4`. If you see `Python 2.x`, try:

```powershell
python3 --version
```

If Python 3 isn't installed, download and install it from [python.org](https://www.python.org/downloads/).

### 1.2 Get the Project Files

**Option A: If you have Git**
```powershell
cd C:\Users\kumar.gn\PycharmProjects
git clone https://github.com/yourusername/feedback-analytics-agent.git
cd feedback-analytics-agent
```

**Option B: If you downloaded a ZIP file**
1. Extract the ZIP file to `C:\Users\kumar.gn\PycharmProjects\feedback-analytics-agent`
2. Open terminal and navigate there:
```powershell
cd C:\Users\kumar.gn\PycharmProjects\feedback-analytics-agent
```

### 1.3 Create a Virtual Environment (Recommended)

This keeps your project dependencies separate from other Python projects:

```powershell
python -m venv venv
```

Wait for it to complete, then activate it:

**On Windows:**
```powershell
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` at the start of your command prompt.

### 1.4 Install Required Packages

This will install all the necessary libraries (takes 2-5 minutes):

```powershell
pip install -r requirements.txt
```

You'll see lots of text scrolling by - this is normal! Wait for it to complete.

## Step 2: Set Up Reddit API Access

### 2.1 Create a Reddit Application

1. Go to [reddit.com](https://reddit.com) and log in (create an account if needed)

2. Visit [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)

3. Scroll down and click **"create another app..."** or **"are you a developer? create an app..."**

4. Fill out the form:
   - **name**: `Feedback Analytics` (or any name you like)
   - **App type**: Select **"script"**
   - **description**: `Analyzing product feedback` (optional)
   - **about url**: Leave blank
   - **redirect uri**: Type `http://localhost:8080` (required, but not used)

5. Click **"create app"**

6. You'll see your new app. Note these two values:
   - **client_id**: The string under "personal use script" (looks like: `a1b2c3d4e5f6g7h8i9`)
   - **client_secret**: The string labeled "secret" (looks like: `x1y2z3a4b5c6d7e8f9`)

### 2.2 Create Your Configuration File

1. In your project folder, create a file named `.env` (note the dot at the start)

2. Open `.env` with a text editor and add:

```env
# Reddit API Credentials
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_secret_here
REDDIT_USER_AGENT=FeedbackAnalytics/1.0 by YourRedditUsername

# Reddit Search Configuration
SUBREDDITS=["GooglePixel","Pixel","Google","pixel_phones","Smartphones","Android","apple","iphone"]
TIME_FILTER=month
NUM_POSTS=100

# Ollama LLM Configuration
BASE_URL=http://localhost:11434
API_KEY=Ollama

# Database Configuration
CHROMA_DB_PATH=./chroma_db
REVIEW_COLLECTION_NAME=reviews

# Performance Tuning
EMBEDDING_BATCH_SIZE=100
UPSERT_BATCH_SIZE=500

# API Security (generate a strong key for production)
API_KEY=your_secure_api_key_here_change_this
```

3. **Replace the placeholder values**:
   - Put your actual Reddit credentials where it says `your_client_id_here` and `your_secret_here`
   - Change `YourRedditUsername` to your Reddit username
   - For security, generate a strong `API_KEY` (or use the default for testing)

4. **Save the file**

**Important**: Never share or commit this `.env` file - it contains your secret credentials!

## Step 3: Set Up Ollama (Local AI)

### 3.1 Install Ollama

1. Download Ollama from [ollama.ai](https://ollama.ai)
2. Run the installer for your operating system
3. Follow the installation wizard

### 3.2 Download an AI Model

Open a new terminal window and run:

```powershell
ollama pull mistral
```

This downloads the Mistral AI model (about 4GB - takes 5-10 minutes depending on your internet speed).

Alternative models you can try:
- `ollama pull llama3` (larger, more powerful)
- `ollama pull phi` (smaller, faster)

### 3.3 Start Ollama Service

**On Windows:**
Ollama usually starts automatically. To verify it's running:

```powershell
ollama list
```

You should see the model you downloaded.

**On Mac/Linux:**
```bash
ollama serve
```

Keep this terminal window open while using the Feedback Analytics Agent.

## Step 4: Configure Your Search Queries

### 4.1 Edit Search Queries

1. Open the file `config/search_queries.csv` in a text editor or Excel

2. You'll see:
```csv
queries
Pixel 9
iPhone 15
Smartphone battery
```

3. Customize these queries based on what you want to analyze:
```csv
queries
Pixel 9 pro battery life
Pixel 9 camera quality
iPhone 15 vs Pixel 9
Pixel 9 overheating issue
Android 14 bugs
```

4. Save the file

**Tips**:
- Be specific (e.g., "Pixel 9 battery drain" vs just "battery")
- Include product names in queries
- One query per line
- No quotes needed

## Step 5: Run Your First Analysis

### 5.1 Collect and Process Feedback

In your terminal (with virtual environment activated), run:

```powershell
python custom_pipeline.py
```

**What you'll see:**
```
============================================================
Starting Custom Pipeline Execution
Batch size: 100, Upsert batch size: 500
============================================================

[1/4] Fetching Reddit reviews...
Fetched 150 reviews in 12.5s

[2/4] Cleaning and preprocessing reviews...
Cleaned 150 reviews in 2.3s

[3/4] Removing duplicate reviews...
Removed 23 duplicates, 127 unique reviews remain

[4/4] Initializing embedding model...

Generating embeddings in batches...
  Batch 1: Processed 100/127 (78.7%) in 15.2s
  Batch 2: Processed 127/127 (100.0%) in 4.8s

Completed embedding generation in 20.0s (0.157s per review)

Persisting embeddings to ChromaDB...
✓ Successfully stored 127 reviews in ChromaDB

Pipeline completed successfully in 35.8s
```

**First-time run**: Takes 1-3 minutes depending on your computer speed and number of reviews

**Result**: You now have:
- `all_posts.csv` - Raw Reddit data collected
- `cleaned_reviews.csv` - Processed reviews
- `chroma_db/` folder - Vector database with embedded reviews

### 5.2 Query Your Data

Now that you have data, ask questions:

```powershell
python query_vectorDB.py
```

When prompted, enter a question:
```
Enter your query: What do people say about Pixel 9 battery life?
```

**Output:**
```
Searching for: What do people say about Pixel 9 battery life?
Found 50 relevant reviews

Results saved to: search_results.csv
Top 5 results:
1. "Pixel 9 battery drains so fast after Android 14 update..."
2. "Amazing battery life! Easily lasts 2 days with moderate use..."
3. "Battery performance is decent but not as good as iPhone..."
...
```

Open `search_results.csv` to see all matching reviews.

### 5.3 Run Full Theme Analysis

For deep analysis with clustering and theme extraction:

```powershell
python review_analyzer_agent.py
```

Enter your analysis question:
```
Enter your query: What are the main issues with Pixel 9?
```

**Processing steps:**
1. Retrieves relevant reviews from database
2. Groups similar reviews together (clustering)
3. Summarizes each cluster
4. Extracts themes and sentiment
5. Saves results

**Output files:**
- `feedback_analysis_results.csv` - Complete analysis
- `a2a_themes_results.json` - Structured data format

**Sample output in CSV:**

| product | sentiment | theme | classification | issue_description |
|---------|-----------|-------|----------------|-------------------|
| Pixel 9 | negative | battery | hardware_issue | Battery drains quickly after update |
| Pixel 9 | positive | camera | feature | Excellent camera quality in low light |
| Pixel 9 | neutral | price | value | Expensive but worth it for features |

## Step 6: Start the API Services (Optional)

### 6.1 Start the A2A Compatible API

For agent-to-agent communication:

```powershell
python -m uvicorn a2acompatible_analyzer_agent:app --host 0.0.0.0 --port 8000
```

**You'll see:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Test it**: Open browser and go to `http://localhost:8000/docs`

You'll see the interactive API documentation!

### 6.2 Start the Custom Tools API

For direct access to analysis tools:

```powershell
python -m uvicorn custom_apis:app --host 0.0.0.0 --port 8001
```

**Test it**: Open browser and go to `http://localhost:8001/docs`

### 6.3 Make API Requests

**Example: Analyze feedback via API**

Using PowerShell:
```powershell
$headers = @{
    "X-API-Key" = "your_secure_api_key_here_change_this"
    "Content-Type" = "application/json"
}

$body = @{
    query = "What are Pixel 9 battery complaints?"
    n_results = 50
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/analyze" -Method POST -Headers $headers -Body $body
```

Using curl:
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "X-API-Key: your_secure_api_key_here_change_this" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are Pixel 9 battery complaints?", "n_results": 50}'
```

## Verification Checklist

Before you start real analysis, verify everything works:

- [ ] Python 3.8+ is installed and accessible
- [ ] Virtual environment is activated (you see `(venv)` in terminal)
- [ ] All packages installed successfully (`pip list` shows them)
- [ ] Reddit API credentials are in `.env` and valid
- [ ] Ollama is running (`ollama list` shows your model)
- [ ] `.env` file exists with all required variables
- [ ] `custom_pipeline.py` runs without errors
- [ ] `chroma_db/` folder was created
- [ ] CSV output files appear in project root
- [ ] `query_vectorDB.py` returns results

## Troubleshooting Installation Issues

### Python not found
```
'python' is not recognized as an internal or external command
```
**Solution**: Try `python3` instead, or reinstall Python with "Add to PATH" checked.

### pip not found
```
'pip' is not recognized...
```
**Solution**: Try `python -m pip` instead of just `pip`.

### Permission errors on Windows
```
Access denied...
```
**Solution**: Run terminal as Administrator, or install packages with `--user` flag:
```powershell
pip install --user -r requirements.txt
```

### SSL Certificate errors
```
SSL: CERTIFICATE_VERIFY_FAILED
```
**Solution**: The package `python-certifi-win32` should fix this. If not:
```powershell
pip install --upgrade certifi
```

### Ollama connection failed
```
Failed to connect to Ollama
```
**Solution**: 
1. Check Ollama is running: `ollama list`
2. Restart Ollama: Close it and run `ollama serve`
3. Verify BASE_URL in `.env` is `http://localhost:11434`

### Reddit API errors
```
401 Unauthorized
```
**Solution**: 
1. Double-check your credentials in `.env`
2. Ensure no extra spaces in the credential strings
3. Verify your Reddit app type is "script" not "web app"

### Out of memory
```
MemoryError
```
**Solution**: Reduce batch sizes in `.env`:
```env
EMBEDDING_BATCH_SIZE=50
UPSERT_BATCH_SIZE=250
NUM_POSTS=50
```

### ChromaDB errors
```
Failed to create collection
```
**Solution**: 
1. Delete the `chroma_db` folder
2. Run `custom_pipeline.py` again to recreate it

## Next Steps

Now that everything is set up:

1. **Customize your analysis**:
   - Edit `config/search_queries.csv` for your specific needs
   - Adjust subreddit list in `.env`
   - Modify time filters and result counts

2. **Schedule regular collection**:
   - Set up Windows Task Scheduler or cron job to run `custom_pipeline.py` daily
   - Keep your feedback database fresh

3. **Explore the APIs**:
   - Visit `http://localhost:8000/docs` to explore A2A API
   - Try different queries and parameters
   - Integrate with your own tools

4. **Read the User Guide**:
   - See `USER_GUIDE.md` for detailed usage examples
   - Learn about advanced features and best practices

## Common Workflows

### Daily Feedback Collection
```powershell
# Activate environment
venv\Scripts\activate

# Collect new data
python custom_pipeline.py

# Analyze specific topic
python review_analyzer_agent.py
# Enter query: "Recent complaints about Pixel 9"

# Review results
# Open: feedback_analysis_results.csv
```

### Quick Question Answer
```powershell
# Activate environment
venv\Scripts\activate

# Query existing data
python query_vectorDB.py
# Enter query: "Battery life comparisons"

# Review results
# Open: search_results.csv
```

### API Service Mode
```powershell
# Activate environment
venv\Scripts\activate

# Start API server
python -m uvicorn a2acompatible_analyzer_agent:app --host 0.0.0.0 --port 8000

# Keep terminal open
# Access API docs: http://localhost:8000/docs
```

## Getting Help

- **Check logs**: Most errors are explained in the terminal output
- **Verify configuration**: Ensure `.env` has correct values
- **Test components**: Run each script individually to isolate issues
- **Review documentation**: See `USER_GUIDE.md` and `README.md`

## Security Reminders

- Keep `.env` file private (never commit to Git)
- Change default `API_KEY` to a strong random value
- Use HTTPS in production deployments
- Regularly update dependencies: `pip install --upgrade -r requirements.txt`
- Review `api_security.log` for suspicious activity

Congratulations! You're ready to start analyzing feedback. Happy analyzing! 🎉
