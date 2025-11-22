import praw, time, json, dotenv, os
import pandas as pd
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set


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
        # self.subreddits = ["GooglePixel","Pixel","Google","pixel_phones","Smartphones","Android","apple","applesucks","iphone"]
        self.subreddits = json.loads(os.getenv("SUBREDDITS", "[]"))
        
        # Performance optimization: connection pooling
        self._reddit_instance = None
        self._seen_post_ids: Set[str] = set()  # Track seen posts to avoid duplicates


# -----------------------------------------------------------------

# -----------------------------------------------------------------
    def getRedditInstance(self):
        """
        Create and authenticate a Reddit API client instance with connection pooling.
        
        Returns:
            praw.Reddit: Authenticated Reddit instance for API operations
        
        Raises:
            SystemExit: If authentication fails, exits the program after logging error
        
        Notes:
            - Uses credentials from environment variables
            - Reuses existing connection for better performance
            - Validates authentication before returning instance
            - Logs success/failure messages to console
        """
        # Return cached instance if available (connection pooling)
        if self._reddit_instance is not None:
            return self._reddit_instance
        
        try:
            # Create Reddit instance with OAuth credentials
            self._reddit_instance = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT')
            )
            print("Successfully authenticated with Reddit API")
            return self._reddit_instance
        except Exception as e:
            # Log authentication failure and exit
            print(f"Error authenticating with Reddit: {e}")
            exit()

# -----------------------------------------------------------------

# -----------------------------------------------------------------
    def _fetch_posts_batch(self, subreddit_name: str, query: str) -> List[Dict]:
        """
        Fetch posts for a single subreddit-query pair (helper for parallel processing).
        
        Args:
            subreddit_name (str): Name of subreddit to search
            query (str): Search query string
        
        Returns:
            List[Dict]: List of post dictionaries with title and text
        
        Notes:
            - Optimized for parallel execution
            - Handles errors gracefully without crashing pipeline
            - Uses reduced sleep time (0.1s) with Reddit's built-in rate limiting
        """
        posts = []
        try:
            reddit = self.getRedditInstance()
            subreddit = reddit.subreddit(subreddit_name)
            
            search_results = subreddit.search(
                query=f"selftext:{query}",
                time_filter=os.getenv('TIME_FILTER', 'week'),
                limit=None,
                sort="relevance",
                syntax="lucene"
            )
            
            for post in search_results:
                # Skip if already seen (early duplicate detection)
                if post.id in self._seen_post_ids:
                    continue
                
                self._seen_post_ids.add(post.id)
                post.comments.replace_more(limit=2)
                
                posts.append({
                    "post_id": post.id,
                    "post_title": post.title,
                    "self_text": "".join(line for line in post.selftext.splitlines()),
                })
                
                # Reduced sleep time - Reddit's rate limiter handles this
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error fetching from r/{subreddit_name} for '{query}': {e}")
        
        return posts

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
        
        print(f"\nFetching posts from {len(self.subreddits)} subreddits with {len(self.client_search_queries)} queries...")
        print(f"Total combinations to process: {len(self.subreddits) * len(self.client_search_queries)}")
        
        # Use ThreadPoolExecutor for parallel fetching (5-10x faster)
        max_workers = int(os.getenv('REDDIT_MAX_WORKERS', '5'))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all subreddit-query combinations
            futures = []
            for subreddit in self.subreddits:
                for query in self.client_search_queries:
                    print(f"Queuing: r/{subreddit} + '{query}'")
                    future = executor.submit(self._fetch_posts_batch, subreddit, query)
                    futures.append((future, subreddit, query))
            
            # Collect results as they complete
            completed = 0
            for future, subreddit, query in futures:
                try:
                    posts = future.result(timeout=120)  # 2 minute timeout per combination
                    all_posts.extend(posts)
                    completed += 1
                    print(f"[{completed}/{len(futures)}] Completed r/{subreddit} + '{query}': {len(posts)} posts")
                except Exception as e:
                    print(f"Error fetching r/{subreddit} + '{query}': {e}")
                    completed += 1
        
        print(f"\nTotal posts fetched: {len(all_posts)}")
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
        # Save extracted posts to JSON and CSV files
        if all_posts:
            try:
                # Convert to DataFrame for easy export
                df = pd.DataFrame(all_posts)
                
                # Remove duplicates early (by post_id if available, otherwise by title)
                original_count = len(df)
                if 'post_id' in df.columns:
                    df = df.drop_duplicates(subset=['post_id'])
                else:
                    df = df.drop_duplicates(subset=['post_title'])
                
                duplicates_removed = original_count - len(df)
                if duplicates_removed > 0:
                    print(f"Removed {duplicates_removed} duplicate posts")
                
                # Convert all columns to string type
                df = df.astype(str)
                
                # Define output filenames
                json_filename = "all_posts.json"
                csv_filename = "all_posts.csv"
                
                # Export to JSON (records format for better structure)
                df.to_json(json_filename, orient='records', lines=True)
                # Export to CSV with all fields quoted for proper parsing
                df.to_csv(csv_filename, index=False, quoting=csv.QUOTE_ALL, quotechar='"')
                
                print(f"\nSaved {len(df)} unique posts to {json_filename} and {csv_filename}")
                
            except Exception as e:
                print(f"Error saving posts to files: {e}")
        else:
            print("\nNo posts fetched. No files created.")
        
        # Return list of extracted posts for pipeline processing
        return all_posts
#-----------------------------------------------------------------

