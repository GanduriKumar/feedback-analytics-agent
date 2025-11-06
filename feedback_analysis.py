import pandas as pd, csv, time
from src.tools.custom_tools import fetch_reddit_reviews, clean_reviews, assess_clusters, summarize_clusters, extract_themes



# Fetching Reddit posts and converting to DataFrame
start = time.time()
reviews = fetch_reddit_reviews()
df = pd.DataFrame(reviews)
# Saving raw reviews to JSON and CSV
df.to_json("all_posts.json", index=False)
df.to_csv("all_posts.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
end = time.time()
print(f"time taken for fetching posts", end - start)

# Cleaning and summarizing reviews
start = time.time()
cleaned_reviews = clean_reviews(reviews)
df_cleaned = pd.DataFrame(cleaned_reviews)
# Saving cleaned and summarized reviews to JSON and CSV
df_cleaned.to_json("cleaned_reviews.json", index=False)
df_cleaned.to_csv("cleaned_reviews.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
end = time.time()
print(f"time taken for cleaning posts", end - start)

# Clustering summarized reviews
start = time.time()
clusters = assess_clusters(cleaned_reviews)
df_clusters = pd.DataFrame(clusters)
# Saving clusters to JSON and CSV
df_clusters.to_json("clusters.json", index=False)
df_clusters.to_csv("clusters.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
end = time.time()
print(f"Time taken for creating the clusters of the reviews: {end - start:.2f} seconds")

# create list by combining and summarizing reviews in a clusters
start = time.time()
curated_reviews = summarize_clusters(clusters)
df_curated_reviews = pd.DataFrame(curated_reviews, columns=["Review"])
df_curated_reviews.to_json("curated_reviews.json", index=False)
df_curated_reviews.to_csv("curated_reviews.csv", index=False,quoting=csv.QUOTE_ALL, quotechar='"')
end = time.time()
print(f"Time taken for curating the reviews in clusters : {end - start:.2f} seconds")

# Theming and classifying reviews
start = time.time()
themes = extract_themes(curated_reviews)
df_themes = pd.DataFrame(themes)
# Saving themes to JSON and CSV
df_themes.to_json("themes.json", index=False)
df_themes.to_csv("themes.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
end = time.time()
print(f"Time taken for classifying the posts: {end - start:.2f} seconds")



