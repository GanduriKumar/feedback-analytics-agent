from src.utilities.reddit_handler import RedditHandler
from src.utilities.review_summarizer import ReviewSummarizer
from src.utilities.review_clustering import AssessClusters
from src.utilities.theme_issue_classifier import ThemeClassifier
import pandas, time, re

def fetch_reddit_reviews()-> list:
    def fetch_reddit_reviews() -> list:
        """
        Fetch Reddit reviews based on search queries from a CSV file.

        This tool reads search queries from './config/search_queries.csv', initializes a RedditHandler
        with these queries, and fetches relevant Reddit posts. It is designed to be used as a tool
        by AI agents for retrieving user feedback or reviews from Reddit.

        Yields:
            str: Status messages or intermediate results for tracing execution.
            list: The list of search queries read from the CSV file.

        Returns:
            list: A list of Reddit posts or reviews fetched based on the search queries.

        Example:
            reviews = fetch_reddit_reviews()
        """
    # print("Entered review fetcher")
    yield "Entered review fetcher <br>"
    df = pandas.read_csv(f"./config/search_queries.csv")
    search_queries = [df['queries'][record] for record in range(0, df['queries'].size)]
    # print(search_queries)
    yield search_queries
    reddit = RedditHandler(queries=search_queries)
    reviews= reddit.fetch_posts()
    return reviews

def clean_reviews(reviews:list) -> list:
    """
    Clean and preprocess user reviews by combining title and text, then removing special characters.

    This function is designed to prepare raw review data for natural language processing
    and analysis by AI agents. It combines post titles with their self-text content and
    strips away special characters, leaving only alphanumeric characters and spaces.

    Args:
        reviews (list): A list of dictionaries where each dictionary represents a review
                       and must contain 'post_title' and 'self_text' keys.
                       Example: [{'post_title': 'Great!', 'self_text': 'Amazing product'}]

    Returns:
        list: A list of cleaned review strings with special characters removed,
              containing only letters, numbers, and spaces.
              Example: ['Great Amazing product']

    Tool Metadata:
        - tool_name: clean_reviews
        - tool_category: text_preprocessing
        - tool_purpose: Sanitize and normalize review text data for downstream analysis
        - input_format: List of review dictionaries with 'post_title' and 'self_text' fields
        - output_format: List of cleaned text strings
        - use_cases: sentiment_analysis, text_classification, topic_modeling, feedback_analysis

    Examples:
        >>> reviews = [
        ...     {'post_title': 'Love it!', 'self_text': 'Best purchase ever!!!'},
        ...     {'post_title': 'Not bad', 'self_text': 'Works as expected.'}
        ... ]
        >>> clean_reviews(reviews)
        ['Love it Best purchase ever', 'Not bad Works as expected']

    Note:
        - Special characters, punctuation, and symbols are removed
        - Multiple spaces may result from special character removal
        - Original review list is not modified (non-destructive operation)
    """
    print("Entered the clean reviews")
    combined_reviews = [f"{review['post_title']}.{review['self_text']}" for review in reviews]
    cleaned_reviews = [re.sub(r'[^A-Za-z0-9 ]+', '',review) for review in combined_reviews]
    return cleaned_reviews

def summarize_clusters(clusters: dict)->list:
    """
    Tool Name: summarize_clusters
    Purpose: Aggregate and summarize grouped (clustered) user reviews or text feedback.

    High-Level Description:
    Given a dictionary where each key represents a cluster label or identifier and the
    value is a collection (typically a list) of raw review / feedback texts belonging
    to that cluster, this function delegates summarization to an underlying
    ReviewSummarizer instance and returns a curated list of summaries. It is intended
    for post-clustering enrichment in feedback analytics pipelines.

    Parameters
    ----------
    clusters : dict
        Mapping of cluster identifiers to an iterable (usually list) of textual
        review strings.
        Expected structure example:
            {
                "pricing": ["The cost is high...", "Too expensive for..."],
                "ux": ["Interface feels intuitive", "Navigation could improve"]
            }

    Returns
    -------
    list
        A list of curated summary objects produced by ReviewSummarizer.summarize_clusters.
        The concrete item structure depends on ReviewSummarizer implementation, but
        commonly may include per-cluster fields such as:
            - cluster_id / name
            - summary (condensed textual synthesis)
            - key_themes (optional)
            - representative_reviews (optional)
            - sentiment (optional)

    Raises
    ------
    ValueError
        May be raised if the input dictionary is empty or not in the expected format
        (actual behavior depends on ReviewSummarizer internals).

    Side Effects
    ------------
    Prints a trace message ("Entered the cluster summarizer") for simple runtime logging.

    Dependencies
    ------------
    Relies on a ReviewSummarizer class available in the import scope. That class must
    implement a method: summarize_clusters(clusters: dict) -> list

    Usage Example
    -------------
    clusters = {
        "feature_requests": [
            "Would love dark mode.",
            "Please add multi-language support."
        ],
        "bugs": [
            "App crashes on startup.",
            "Login button unresponsive sometimes."
        ]
    }
    summaries = summarize_clusters(clusters)
    for entry in summaries:
        print(entry.get("cluster_id"), "=>", entry.get("summary"))

    AI Tool Metadata
    ----------------
    tool_name: summarize_clusters
    tool_type: function
    input_schema:
      clusters: dict[str, list[str]]
    output_schema:
      list[dict]  # semantic summaries per cluster
    capabilities: ["summarization", "nlp", "feedback-analysis"]
    version: "1.0.0"
    """
    print("Entered the cluster summarizer")
    summarizer = ReviewSummarizer()
    curated_reviews= summarizer.summarize_clusters(clusters)
    return curated_reviews

def assess_clusters(cleaned_reviews: list) -> dict:
    """
    Analyze cleaned customer reviews and return cluster assessments.

    This function acts as a thin tool wrapper around an internal `AssessClusters`
    component. It accepts a list of pre-cleaned review texts, delegates clustering
    to `AssessClusters.assess_clusters()`, and returns the resulting cluster
    structure. A simple log line is printed to stdout when invoked.

    AI Tool Metadata (for agent discovery/parsing):
        name: assess_clusters
        description: Assess semantic clusters from pre-cleaned text reviews.
        inputs:
            cleaned_reviews:
                type: array[string]
                required: true
                description: List of cleaned review texts. Each entry should be a single
                    review string that has already been normalized (e.g., lowercased,
                    punctuation removed, stopwords stripped) to the extent required by the
                    underlying model.
        outputs:
            type: object
            description: Dictionary containing clustering results as produced by the
                underlying AssessClusters implementation.
        side_effects:
            - Prints a diagnostic line to stdout ("Entered the cluster assessor").

    Args:
        cleaned_reviews (list[str]): Preprocessed review texts to cluster. Provide at
            least one non-empty string; behavior for empty input depends on the
            underlying clustering implementation.

    Returns:
        dict: Clustering results produced by `AssessClusters.assess_clusters()`. The
        exact schema depends on the implementation, but it commonly includes per-cluster
        assignments, summaries, and/or keywords.

    Raises:
        Any exception bubbled up from `AssessClusters` initialization or its
        `assess_clusters()` method.

    Notes:
        - This function logs to stdout for simple observability.
        - Ensure reviews are pre-cleaned as expected by the `AssessClusters` pipeline.

    Example:
        >>> reviews = ["great battery life", "poor camera quality", "amazing battery"]
        >>> result = assess_clusters(reviews)
        >>> isinstance(result, dict)
        True
    """
    print("Entered the cluster assessor")
    cluster_assessor = AssessClusters(cleaned_reviews)
    clusters = cluster_assessor.assess_clusters()
    return clusters

def extract_themes(curated_reviews:list)->list:
    """
    Extract themes from a list of curated reviews using AI-powered classification.

    This tool analyzes customer reviews and identifies key themes, topics, and sentiment patterns
    within the feedback. It processes each review individually and returns a comprehensive list
    of extracted themes that can be used for analytics and insights.

    Args:
        curated_reviews (list): A list of curated review texts to analyze. Each element should be
                               a string containing the review content.

    Returns:
        list: A list of extracted themes corresponding to each input review. The structure of each
              theme depends on the ThemeClassifier implementation, typically containing theme names,
              categories, sentiment scores, or related metadata.

    Tool Metadata:
        - Tool Name: extract_themes
        - Category: Text Analysis, Sentiment Analysis, Review Processing
        - Use Case: Feedback analytics, customer insight extraction, review categorization
        - Input Format: List of review strings
        - Output Format: List of theme objects/dictionaries
        
    Examples:
        >>> reviews = ["Great product, fast delivery!", "Poor quality, disappointed"]
        >>> themes = extract_themes(reviews)
        >>> # Returns themes for each review with sentiment and categories

    Note:
        - Requires ThemeClassifier to be properly initialized
        - Processing time scales linearly with the number of reviews
        - Each review is processed independently
    """
    print("Entered the theme extractor")
    theme_classifier = ThemeClassifier()
    themes = [theme_classifier.extract_themes(review) for review in curated_reviews]
    return themes