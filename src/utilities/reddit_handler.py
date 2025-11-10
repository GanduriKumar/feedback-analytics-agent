import praw, time, json, dotenv, os
import pandas as pd
import csv


class RedditHandler:
    """
    Reddit data extraction tool for AI agents.
    
    Authenticates with Reddit API and extracts posts from specified subreddits based on search queries.
    Designed for feedback analysis pipelines focusing on mobile device ecosystems (Pixel vs iPhone).
    
    Capabilities:
    - Authenticates with Reddit using PRAW library
    - Searches multiple subreddits with configurable queries
    - Extracts post titles and self-text content
    - Exports results to both JSON and CSV formats
    - Implements rate limiting to comply with Reddit API guidelines
    
    Required Environment Variables:
    - REDDIT_CLIENT_ID: Reddit application client ID
    - REDDIT_CLIENT_SECRET: Reddit application secret key
    - REDDIT_USER_AGENT: User agent string for Reddit API
    - TIME_FILTER: Time filter for search (hour, day, week, month, year, all)
    - NUM_POSTS: Maximum number of posts to fetch per query
    
    Default Subreddits: GooglePixel, Pixel, Google, pixel_phones, Smartphones, Android, apple, applesucks, iphone
    """

# -----------------------------------------------------------------
# constructor
    def __init__(self, queries:list):
        """
        Initialize Reddit handler with search queries.
        
        Args:
            queries (list): List of search strings to query Reddit (e.g., ["Pixel 9", "iPhone 15"])
        
        Side Effects:
            - Loads environment variables from .env file
            - Initializes Reddit API credentials
            - Sets up default subreddit list for mobile device discussions
        """
        # Load environment variables from .env file
        dotenv.load_dotenv()
        
        # Initialize Reddit API credentials from environment
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.client_useragent = os.getenv('REDDIT_USER_AGENT')
        
        # Store search queries for later use
        self.client_search_queries = queries
        
        # Define default subreddits for mobile device feedback analysis
        self.subreddits = ["GooglePixel","Pixel","Google","pixel_phones","Smartphones","Android","apple","applesucks","iphone"]

# -----------------------------------------------------------------

# -----------------------------------------------------------------
    def getRedditInstance(self):
        """
        Create and authenticate a Reddit API client instance.
        
        Returns:
            praw.Reddit: Authenticated Reddit instance for API operations
        
        Raises:
            SystemExit: If authentication fails, exits the program after logging error
        
        Notes:
            - Uses credentials from environment variables
            - Validates authentication before returning instance
            - Logs success/failure messages to console
        """
        try:
            # Create Reddit instance with OAuth credentials
            reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT')
            )
            print("Successfully authenticated with Reddit API")
            return reddit
        except Exception as e:
            # Log authentication failure and exit
            print(f"Error authenticating with Reddit: {e}")
            exit()

# -----------------------------------------------------------------

# -----------------------------------------------------------------
    def fetch_posts(self):
        """
        Extract Reddit posts matching search queries and save to files.
        
        Primary data extraction tool for AI agents. Searches configured subreddits for posts
        matching provided queries, extracts relevant content, and saves results in standardized formats.
        
        Returns:
            list: List of dictionaries containing extracted post data with keys:
                - post_title (str): Post title text
                - self_text (str): Post body text (cleaned, single-line)
        
        Side Effects:
            - Creates 'all_posts.json' with extracted post data
            - Creates 'all_posts.csv' with extracted post data (quoted fields)
            - Prints progress messages to console
            - Implements 1-second delay between posts to respect API rate limits
        
        Search Configuration:
            - Uses Lucene syntax for advanced querying
            - Filters by time period (from TIME_FILTER env var)
            - Limits results per query (from NUM_POSTS env var)
            - Sorts by relevance for quality results
            - Expands comment threads (limit=2) for context
        
        Error Handling:
            - Catches and logs exceptions during fetch operation
            - Returns empty list on failure (does not crash pipeline)
        
        Example Output Format:
            [
                {
                    "post_title": "Pixel 9 Pro review after 3 months",
                    "self_text": "After using the Pixel 9 Pro for 3 months, here are my thoughts..."
                }
            ]
        """
        # Initialize empty list to store all extracted posts
        all_posts = []
        try:
            # Authenticate with Reddit API
            reddit = self.getRedditInstance()
            
            # Iterate through each configured subreddit
            for subreddit in self.subreddits:
                # Iterate through each search query
                for query in self.client_search_queries:
                    print(f"\nSearching in r/{subreddit} for posts related to: '{query}'")
                    
                    # Get subreddit instance for searching
                    subreddit_instance = reddit.subreddit(subreddit)
                    
                    # Search posts using Lucene syntax for self-text matching
                    posts = subreddit_instance.search(
                        query=f"self_text:{query}",  # Search within post body text
                        time_filter=os.getenv('TIME_FILTER'),  # Filter by time period (e.g., 'week')
                        limit=int(os.getenv('NUM_POSTS')),  # Limit number of results
                        sort="relevance",  # Sort by relevance to query
                        syntax="lucene"  # Use Lucene query syntax
                    )

                    # Process each post in search results
                    for post in posts:
                        # Expand comment threads (limit=2 to avoid excessive API calls)
                        post.comments.replace_more(limit=2)
                        
                        # Extract and store post data
                        all_posts.append({
                             "post_title": post.title,
                             # Clean self-text by removing newlines for single-line format
                             "self_text": "".join(line for line in post.selftext.splitlines()),
                        })
                        
                        # Rate limiting: pause 1 second between posts
                        time.sleep(1)

        except Exception as e:
            # Log error but don't crash pipeline
            print(f"Error fetching reviews: {e}")
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
        # Save extracted posts to JSON and CSV files
        if all_posts:
            # Convert to DataFrame for easy export
            df = pd.DataFrame(all_posts)
            # Convert all columns to string type
            df = df.astype(str)
            
            # Define output filenames
            json_filename = "all_posts.json"
            csv_filename = "all_posts.csv"
            
            # Export to JSON (without row indices)
            df.to_json(json_filename, index=False)
            # Export to CSV with all fields quoted for proper parsing
            df.to_csv(csv_filename, index=False, quoting=csv.QUOTE_ALL, quotechar='"')
        
        # Return list of extracted posts for pipeline processing
        return all_posts
#-----------------------------------------------------------------

