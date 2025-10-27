import pandas,csv, nltk
from src.custom_tools import fetch_reddit_reviews
from src.custom_tools import clean_reviews

nltk.download('punkt')
nltk.download('punkt_tab')


reviews = fetch_reddit_reviews()
detailed_reviews = clean_reviews(reviews)

df = pandas.DataFrame(detailed_reviews)
json_filename = "all_reviews.json"
csv_filename = "all_reviews.csv"
df.to_json(json_filename, index=False, )
df.to_csv(csv_filename,index=False,quoting=csv.QUOTE_ALL,quotechar='"')

