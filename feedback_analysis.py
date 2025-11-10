import pandas as pd, csv, time
from src.tools.custom_tools import fetch_reddit_reviews, clean_reviews, assess_clusters, summarize_clusters, extract_themes
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Initialize FastAPI application for feedback analysis pipeline
app = FastAPI()

@app.get(
    "/analyze_feedback",
    summary="Fetch raw Reddit reviews, group, cluster and then extract themes",
    description="Tool: Fetch raw Reddit posts , analyzes and extract themes and returns them as CSV and JSON. ",
    tags=["agent_tools", "analyze_reviews"],
)
def analyze_feedback():
    """
    Tool: Analyze Reddit Feedback

    Fetches Reddit posts based on configured queries, cleans and clusters the reviews, summarizes clusters,
    extracts key themes, and saves all intermediate and final results as JSON and CSV files.
    Returns the extracted themes as a JSON response.

    This endpoint is designed to be discoverable and callable by AI agents for automated feedback analysis.
    """
    # Step 1: Fetch raw Reddit posts using PRAW-based handler
    start = time.time()
    reviews = fetch_reddit_reviews()  # Returns list of dicts with post_title, self_text, etc.
    df = pd.DataFrame(reviews)
    
    # Save raw reviews for audit trail and debugging
    df.to_json("all_posts.json", index=False)
    df.to_csv("all_posts.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
    end = time.time()
    print(f"time taken for fetching posts", end - start)

    # Step 2: Clean reviews by combining title+text and removing special characters
    start = time.time()
    cleaned_reviews = clean_reviews(reviews)  # Returns list of sanitized strings
    df_cleaned = pd.DataFrame(cleaned_reviews)
    
    # Save cleaned reviews as intermediate pipeline output
    df_cleaned.to_json("cleaned_reviews.json", index=False)
    df_cleaned.to_csv("cleaned_reviews.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
    end = time.time()
    print(f"time taken for cleaning posts", end - start)

    # Step 3: Cluster reviews using sentence transformers for semantic grouping
    start = time.time()
    clusters = assess_clusters(cleaned_reviews)  # Returns dict with cluster assignments
    df_clusters = pd.DataFrame(clusters)
    
    # Save cluster results for analysis and visualization
    df_clusters.to_json("clusters.json", index=False)
    df_clusters.to_csv("clusters.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
    end = time.time()
    print(f"Time taken for creating the clusters of the reviews: {end - start:.2f} seconds")

    # Step 4: Summarize reviews within each cluster to reduce redundancy
    start = time.time()
    curated_reviews = summarize_clusters(clusters)  # Returns list of summarized review texts
    df_curated_reviews = pd.DataFrame(curated_reviews, columns=["Review"])
    
    # Save curated reviews before theme extraction
    df_curated_reviews.to_json("curated_reviews.json", index=False)
    df_curated_reviews.to_csv("curated_reviews.csv", index=False,quoting=csv.QUOTE_ALL, quotechar='"')
    end = time.time()
    print(f"Time taken for curating the reviews in clusters : {end - start:.2f} seconds")

    # Step 5: Extract themes using LLM-based classification (via Ollama)
    start = time.time()
    themes = extract_themes(curated_reviews)  # Returns list of theme objects/classifications
    df_themes = pd.DataFrame(themes)
    
    # Save final themes as both JSON and CSV for downstream consumption
    df_themes.to_json("themes.json", index=False)
    df_themes.to_csv("themes.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
    end = time.time()
    print(f"Time taken for classifying the posts: {end - start:.2f} seconds")
    
    # Return themes as JSON response for API consumers
    return JSONResponse(content=df_themes.to_dict(orient="records"))


# Run FastAPI server when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

