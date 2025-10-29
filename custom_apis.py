"""
custom_apis.py

FastAPI endpoints exposing the core pipeline tools so external agents can discover and call them.

Each route has a clear docstring and FastAPI metadata (summary and description) to make the route
discoverable as a "tool" by automated agents. The routes wrap the project's utility functions:
- fetch_reddit_reviews: collect Reddit posts
- clean_reviews: perform summarization/cleanup
- assess_clusters: cluster summarized reviews
- ThemeClassifier.extract_themes: extract themes/issues via LLM

Side effects:
- Each route writes intermediate results to JSON and CSV files in the workspace root.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import csv
import time

from src.init.custom_tools import fetch_reddit_reviews, clean_reviews
from src.tools.theme_issue_classifier import ThemeClassifier
from src.tools.review_clustering import assess_clusters

app = FastAPI()

@app.get(
    "/reviews",
    summary="Fetch raw Reddit reviews",
    description="Tool: Fetch raw Reddit posts using the project's Reddit handler and return them as JSON. "
                "Also saves results to all_posts.json and all_posts.csv for later pipeline stages.",
    tags=["agent_tools", "data_collection"],
)
def get_reviews():
    """
    get_reviews() -> JSON list

    Tool purpose:
    - Collect raw Reddit posts using fetch_reddit_reviews()
    - Persist results to all_posts.json and all_posts.csv
    - Return a JSON array of the raw posts for agent consumption

    Side effects:
    - Writes all_posts.json and all_posts.csv to workspace root

    Returns:
    - JSONResponse with list of post records (dicts)
    """
    reviews = fetch_reddit_reviews()
    df = pd.DataFrame(reviews)
    df.to_json("all_posts.json", index=False)
    df.to_csv("all_posts.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
    return JSONResponse(content=df.to_dict(orient="records"))

@app.get(
    "/summarized_reviews",
    summary="Fetch summarized Reddit reviews",
    description="Tool: Fetch and clean/summarize Reddit posts using the project's summarization utilities. "
                "Saves summaries to summaries.json and summaries.csv and returns them as JSON.",
    tags=["agent_tools", "preprocessing"],
)
def get_summarized_reviews():
    """
    get_summarized_reviews() -> JSON list

    Tool purpose:
    - Fetch raw Reddit posts and run the project's clean_reviews() summarizer
    - Persist summarized results to summaries.json and summaries.csv
    - Return a JSON array of summarized review records for agent consumption

    Side effects:
    - Writes summaries.json and summaries.csv to workspace root

    Returns:
    - JSONResponse with list of summarized review records (dicts)
    """
    reviews = fetch_reddit_reviews()
    summarized_reviews = clean_reviews(reviews)
    df_summarized = pd.DataFrame(summarized_reviews)
    df_summarized.to_json("summaries.json", index=False)
    df_summarized.to_csv("summaries.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
    return JSONResponse(content=df_summarized.to_dict(orient="records"))

@app.get(
    "/clusters",
    summary="Compute clusters of summarized reviews",
    description="Tool: Cluster summarized reviews using sentence-transformer based clustering. "
                "Returns clusters and the time taken. Also persists clusters.json and clusters.csv.",
    tags=["agent_tools", "clustering"],
)
def get_clusters():
    """
    get_clusters() -> JSON object

    Tool purpose:
    - Fetch and summarize Reddit posts, then compute clusters with assess_clusters()
    - Persist cluster results to clusters.json and clusters.csv
    - Return a JSON object containing:
      - clusters: list of cluster records
      - time_taken: float seconds spent computing clusters

    Side effects:
    - Writes clusters.json and clusters.csv to workspace root

    Returns:
    - JSONResponse with {"clusters": [...], "time_taken": <seconds>}
    """
    reviews = fetch_reddit_reviews()
    summarized_reviews = clean_reviews(reviews)
    start = time.time()
    clusters = assess_clusters(summarized_reviews)
    df_clusters = pd.DataFrame(clusters)
    df_clusters.to_json("clusters.json", index=False)
    df_clusters.to_csv("clusters.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
    end = time.time()
    return JSONResponse(content={"clusters": df_clusters.to_dict(orient="records"), "time_taken": end - start})

@app.get(
    "/themes",
    summary="Extract themes and issues from summarized reviews",
    description="Tool: Use the ThemeClassifier LLM-based extractor to identify themes and issues per summarized review. "
                "Returns themes and time taken. Also persists themes.json and themes.csv.",
    tags=["agent_tools", "classification"],
)
def get_themes():
    """
    get_themes() -> JSON object

    Tool purpose:
    - Fetch and summarize Reddit posts, then extract themes/issues for each summary using ThemeClassifier
    - Persist theme results to themes.json and themes.csv
    - Return a JSON object containing:
      - themes: list of theme records for each summarized review
      - time_taken: float seconds spent extracting themes

    Side effects:
    - Writes themes.json and themes.csv to workspace root

    Returns:
    - JSONResponse with {"themes": [...], "time_taken": <seconds>}
    """
    reviews = fetch_reddit_reviews()
    summarized_reviews = clean_reviews(reviews)
    start = time.time()
    theme_classifier = ThemeClassifier()
    themes = [theme_classifier.extract_themes(review) for review in summarized_reviews]
    df_themes = pd.DataFrame(themes)
    df_themes.to_json("themes.json", index=False)
    df_themes.to_csv("themes.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
    end = time.time()
    return JSONResponse(content={"themes": df_themes.to_dict(orient="records"), "time_taken": end - start})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



