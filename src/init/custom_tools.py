from src.tools.reddit_handler import RedditHandler
from src.tools.review_summarizer import getSummaries, getSummary
import pandas, time, re

def fetch_reddit_reviews()->list:
    """Tool for fetching Reddit reviews related to mobile devices.
    
    This tool searches Reddit and retrieves posts based on predefined search queries from a CSV file.
    Used to collect user feedback and discussions about mobile devices.

    Args:
        None

    Returns:
        list: Collection of Reddit posts containing user reviews and discussions
    """
    start = time.time()
    df = pandas.read_csv(f"./config/search_queries.csv")
    search_queries = [df['queries'][record] for record in range(0, df['queries'].size)]
    print(search_queries)

    reddit = RedditHandler(queries=search_queries)
    reviews= reddit.fetch_posts()
    end = time.time()
    print(f"time taken for fetching posts", end - start)
    return reviews

def clean_reviews(reviews:list) -> list:
    """Tool for cleaning and standardizing Reddit review text.
    
    This tool processes raw Reddit posts by combining titles and content, removing special characters,
    and standardizing the text format for analysis.

    Args:
        reviews (list): Raw Reddit reviews containing post titles and content

    Returns:
        list: Cleaned and standardized review sentences ready for analysis
    """
    start = time.time()
    combined_reviews = [f"{getSummary(review['post_title'])}.{getSummary(review['self_text'])}" for review in reviews]
    cleaned_reviews = [re.sub(r'[^A-Za-z0-9 ]+', '',review) for review in combined_reviews]
    end = time.time()
    print(f"time taken for cleaning posts", end - start)
    return cleaned_reviews

def summarize_reviews(reviews: list)->list:
    """Tool for generating concise summaries of Reddit reviews.
    
    This tool takes a collection of reviews and creates shorter, focused summaries
    while preserving key information and sentiment.

    Args:
        reviews (list): Collection of cleaned review texts to be summarized

    Returns:
        list: Condensed summaries of the input reviews maintaining core message
    """
    start = time.time()
    # summarized_reviews = getSummaries(reviews)
    summarized_reviews = reviews
    end = time.time()
    print(f"time taken for summarizing posts", end - start)
    return summarized_reviews