from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import csv
import time

from src.init.custom_tools import fetch_reddit_reviews, clean_reviews
from src.tools.theme_issue_classifier import ThemeClassifier
from src.tools.review_clustering import assess_clusters

app = FastAPI()

@app.get("/reviews")
def get_reviews():
    reviews = fetch_reddit_reviews()
    df = pd.DataFrame(reviews)
    df.to_json("all_posts.json", index=False)
    df.to_csv("all_posts.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
    return JSONResponse(content=df.to_dict(orient="records"))

@app.get("/summarized_reviews")
def get_summarized_reviews():
    reviews = fetch_reddit_reviews()
    summarized_reviews = clean_reviews(reviews)
    df_summarized = pd.DataFrame(summarized_reviews)
    df_summarized.to_json("summaries.json", index=False)
    df_summarized.to_csv("summaries.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
    return JSONResponse(content=df_summarized.to_dict(orient="records"))

@app.get("/clusters")
def get_clusters():
    reviews = fetch_reddit_reviews()
    summarized_reviews = clean_reviews(reviews)
    start = time.time()
    clusters = assess_clusters(summarized_reviews)
    df_clusters = pd.DataFrame(clusters)
    df_clusters.to_json("clusters.json", index=False)
    df_clusters.to_csv("clusters.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
    end = time.time()
    return JSONResponse(content={"clusters": df_clusters.to_dict(orient="records"), "time_taken": end - start})

@app.get("/themes")
def get_themes():
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



