from src.utilities.reddit_handler import RedditHandler
from src.utilities.review_summarizer import ReviewSummarizer
from src.utilities.review_clustering import AssessClusters
from src.utilities.theme_issue_classifier import ThemeClassifier
import pandas, time, re

def fetch_reddit_reviews()->list:
    print("Entered review fetcher")
    df = pandas.read_csv(f"./config/search_queries.csv")
    search_queries = [df['queries'][record] for record in range(0, df['queries'].size)]
    print(search_queries)
    reddit = RedditHandler(queries=search_queries)
    reviews= reddit.fetch_posts()
    return reviews

def clean_reviews(reviews:list) -> list:
    print("Entered the clean reviews")
    combined_reviews = [f"{review['post_title']}.{review['self_text']}" for review in reviews]
    cleaned_reviews = [re.sub(r'[^A-Za-z0-9 ]+', '',review) for review in combined_reviews]
    return cleaned_reviews

def summarize_clusters(clusters: dict)->list:
    print("Entered the cluster summarizer")
    summarizer = ReviewSummarizer()
    curated_reviews= summarizer.summarize_clusters(clusters)
    return curated_reviews

def assess_clusters(cleaned_reviews: list) -> dict:
    print("Entered the cluster assessor")
    cluster_assessor = AssessClusters(cleaned_reviews)
    clusters = cluster_assessor.assess_clusters()
    return clusters

def extract_themes(curated_reviews:list)->list:
    print("Entered the theme extractor")
    theme_classifier = ThemeClassifier()
    themes = [theme_classifier.extract_themes(review) for review in curated_reviews]
    return themes