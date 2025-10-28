import pandas,csv, time
import pandas as pd

from src.init.custom_tools import fetch_reddit_reviews, clean_reviews
from src.tools.theme_issue_classifier import ThemeClassifier
from src.tools.review_clustering import assess_clusters


#----fetching posts ------
reviews = fetch_reddit_reviews()
df = pandas.DataFrame(reviews)
json_filename = "all_posts.json"
csv_filename = "all_posts.csv"
df.to_json(json_filename, index=False, )
df.to_csv(csv_filename,index=False,quoting=csv.QUOTE_ALL,quotechar='"')

#----cleaning and summarizing ------
summarized_reviews = clean_reviews(reviews)
# summarized_reviews = summarize_reviews(detailed_reviews)

df = pandas.DataFrame(summarized_reviews)
json_filename = "summaries.json"
csv_filename = "summaries.csv"
df.to_json(json_filename, index=False, )
df.to_csv(csv_filename,index=False,quoting=csv.QUOTE_ALL,quotechar='"')


#---- clustering------
start = time.time()

clusters = assess_clusters(summarized_reviews)
df = pd.DataFrame(clusters)
json_filename = "clusters.json"
csv_filename = "clusters.csv"
df.to_json(json_filename, index=False, )
df.to_csv(csv_filename,index=False,quoting=csv.QUOTE_ALL,quotechar='"')

end = time.time()
print(f"time taken for creating the clusters of the reviews", end - start)


#----theming ------

start = time.time()
theme_classifier = ThemeClassifier()
themes = [theme_classifier.extract_themes(review) for review in summarized_reviews]
df = pandas.DataFrame(themes)
json_filename = "themes.json"
csv_filename = "themes.csv"
df.to_json(json_filename, index=False, )
df.to_csv(csv_filename,index=False,quoting=csv.QUOTE_ALL,quotechar='"')
end = time.time()
print(f"time taken for classifying the posts", end - start)



