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
    review_titles   = [review['post_title'] for review in reviews]
    review_contents = [review['self_text'] for review in reviews]

    # break down the titles and contents into independent sentences
    titles_list = [sent_tokenize(title) for title in review_titles]
    contents_list = [sent_tokenize(content) for content in review_contents]
    titles = [[re.sub(r'[^A-Za-z0-9 ]+', '', item) for item in sub_list] for sub_list in titles_list ]
    contents = [[re.sub(r'[^A-Za-z0-9 ]+', '', item) for item in sub_list] for sub_list in contents_list ]

    print(f"titles: ", titles)
    print(f"contents: ", contents)
    # create a combined list
    combined_list = titles+contents
    print(combined_list)

    return combined_list