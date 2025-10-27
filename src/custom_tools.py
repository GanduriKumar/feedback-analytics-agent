from tools.reddit_handler import RedditHandler
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

def clean_reviews(reviews) -> list:
    """
    Fetch reddit posts based on the defined search queries
    Args:
        list: list of review dicts containing post title and post content
    Return:
        list: Combined list of independent sentences of post titles and post contents

    """
    # import tokenizer for breaking up sentences
    from nltk.tokenize import sent_tokenize

    # extract titles, contents from posts
    combined_reviews = [f"{review['post_title']}.{review['self_text']}" for review in reviews]
    reviews = [sent_tokenize(content) for content in combined_reviews]

    cleaned_reviews = [re.sub(r'[^A-Za-z0-9 ]+', '', item) for review in reviews for item in review]

    return cleaned_reviews