import pandas as pd
import csv
import time

from src.init.custom_tools import fetch_reddit_reviews, clean_reviews
from src.tools.theme_issue_classifier import ThemeClassifier
from src.tools.review_clustering import assess_clusters

# Fetching Reddit posts and converting to DataFrame
reviews = fetch_reddit_reviews()
df = pd.DataFrame(reviews)

# Saving raw reviews to JSON and CSV
df.to_json("all_posts.json", index=False)
df.to_csv("all_posts.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')

# Cleaning and summarizing reviews
summarized_reviews = clean_reviews(reviews)
df_summarized = pd.DataFrame(summarized_reviews)

# Saving summarized reviews to JSON and CSV
df_summarized.to_json("summaries.json", index=False)
df_summarized.to_csv("summaries.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')

# Clustering summarized reviews
start = time.time()
clusters = assess_clusters(summarized_reviews)
df_clusters = pd.DataFrame(clusters)

# Saving clusters to JSON and CSV
df_clusters.to_json("clusters.json", index=False)
df_clusters.to_csv("clusters.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
end = time.time()
print(f"Time taken for creating the clusters of the reviews: {end - start:.2f} seconds")

# Theming classified reviews
start = time.time()
theme_classifier = ThemeClassifier()
themes = [theme_classifier.extract_themes(review) for review in summarized_reviews]
df_themes = pd.DataFrame(themes)

# Saving themes to JSON and CSV
df_themes.to_json("themes.json", index=False)
df_themes.to_csv("themes.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
end = time.time()
print(f"Time taken for classifying the posts: {end - start:.2f} seconds")



