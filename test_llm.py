import pandas,csv, nltk
from src.init.custom_tools import fetch_reddit_reviews, clean_reviews, summarize_reviews

# nltk.download('punkt')
# nltk.download('punkt_tab')


reviews = fetch_reddit_reviews()
summarized_reviews = clean_reviews(reviews)
# summarized_reviews = summarize_reviews(detailed_reviews)

df = pandas.DataFrame(summarized_reviews)
json_filename = "summary_reviews.json"
csv_filename = "summary_reviews.csv"
df.to_json(json_filename, index=False, )
df.to_csv(csv_filename,index=False,quoting=csv.QUOTE_ALL,quotechar='"')
