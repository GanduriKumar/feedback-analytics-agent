import praw, time, json, dotenv, os
import pandas as pd
import csv


class RedditHandler:
    """
    Class for authenticating and extracting posts from Reddit based on provided credentials
    and search queries.
    """

    # -----------------------------------------------------------------
    # Constructor
    def __init__(self, queries: list):
        """
        Initializes the RedditHandler with search queries and loads environment variables.

        :param queries: List of search queries to search for in Reddit posts.
        """
        dotenv.load_dotenv()
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.client_useragent = os.getenv('REDDIT_USER_AGENT')
        self.client_search_queries = queries
        # List of subreddits to search in
        self.subreddits = ["GooglePixel", "Pixel"]
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    def getRedditInstance(self):
        """
        Creates and returns an authenticated Reddit instance.

        :return: Instance of authenticated Reddit.
        """
        try:
            reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.client_useragent
            )
            print("Successfully authenticated with Reddit API")
            return reddit
        except Exception as e:
            print(f"Error authenticating with Reddit: {e}")
            exit()
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    def fetch_posts(self) -> list:
        """
        Fetches posts from specified subreddits based on search queries,
        saves the results to CSV and JSON files, and returns the list of posts.

        :return: List of extracted posts for the given search queries.
        """
        all_posts = []
        try:
            reddit = self.getRedditInstance()
            for subreddit in self.subreddits:
                for query in self.client_search_queries:
                    print(f"\nSearching in r/{subreddit} for posts related to: '{query}'")
                    subreddit_instance = reddit.subreddit(subreddit)
                    posts = subreddit_instance.search(
                        query=f"self_text:{query}",
                        time_filter=os.getenv('TIME_FILTER'),
                        limit=None,
                        sort="relevance",
                        syntax="lucene",
                    )

                    for post in posts:
                        post.comments.replace_more(limit=2)  # Avoid excessive API calls
                        all_posts.append({
                            "post_title": post.title,
                            "self_text": "".join(line for line in post.selftext.splitlines()),
                        })
                        time.sleep(1)  # Pause to prevent API rate limits

        except Exception as e:
            print(f"Error fetching reviews: {e}")

        # Save fetched posts to JSON and CSV files
        if all_posts:
            df = pd.DataFrame(all_posts)
            df = df.astype(str)
            json_filename = "all_posts.json"
            csv_filename = "all_posts.csv"
            df.to_json(json_filename, index=False)
            df.to_csv(csv_filename, index=False, quoting=csv.QUOTE_ALL, quotechar='"')

        return all_posts
    # -----------------------------------------------------------------
