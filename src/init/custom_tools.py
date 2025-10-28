from src.tools.reddit_handler import RedditHandler
from src.tools.review_summarizer import getSummaries, getSummary
import pandas, time, re

def fetch_reddit_reviews()->list:
    """
    Fetch reddit posts based on the defined search queries
    Args:
        None:
    Return:
        list: List of extracted reviews from Reddit forums
    """

    start = time.time()
    # read the search query input file and create a list
    df = pandas.read_csv(f"./config/search_queries.csv")
    search_queries = [df['queries'][record] for record in range(0, df['queries'].size)]
    print(search_queries)

    # create Reddit handler and fetch reviews
    reddit = RedditHandler(queries=search_queries)
    reviews= reddit.fetch_posts()
    end = time.time()
    print(f"time taken for fetching posts", end - start)
    return reviews

def clean_reviews(reviews:list) -> list:
    """
    Fetch reddit posts based on the defined search queries
    Args:
        reviews: list of review dicts containing post title and post content
    Return:
        list: Combined list of independent sentences of post titles and post contents

    """
    start = time.time()
    # extract titles, contents from posts and clean them
    combined_reviews = [f"{getSummary(review['post_title'])}.{getSummary(review['self_text'])}" for review in reviews]

    cleaned_reviews = [re.sub(r'[^A-Za-z0-9 ]+', '',review) for review in combined_reviews]
    end = time.time()
    print(f"time taken for cleaning posts", end - start)

    return cleaned_reviews

def summarize_reviews(reviews: list)->list:
    """
        Summarize the given list of reviews
        Args:
            reviews: List of reviews
        Return:
            list: List of summarized reviews
        """
    start = time.time()
    # summarized_reviews = getSummaries(reviews)
    summarized_reviews = reviews
    end = time.time()
    print(f"time taken for summarizing posts", end - start)
    return summarized_reviews