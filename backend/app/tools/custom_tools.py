from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import re

from app.utilities.reddit_handler import RedditHandler
from app.utilities.review_summarizer import ReviewSummarizer
from app.utilities.review_clustering import AssessClusters
from app.utilities.theme_issue_classifier import ThemeClassifier
from app.models.schemas import LLMConfig


def _default_search_queries_path() -> Path:
    # backend/app/tools/custom_tools.py -> backend/
    backend_root = Path(__file__).resolve().parents[2]
    return backend_root / "config" / "search_queries.csv"


def fetch_reddit_reviews(queries: Optional[List[str]] = None) -> list:
    """Fetch Reddit reviews for the provided queries.

    If `queries` is not provided (or empty), falls back to reading
    `backend/config/search_queries.csv`.
    """
    normalized_queries: List[str] = [q.strip() for q in (queries or []) if q and q.strip()]
    if not normalized_queries:
        df = pd.read_csv(_default_search_queries_path())
        normalized_queries = [str(v).strip() for v in df.get('queries', []) if str(v).strip()]

    reddit = RedditHandler(queries=normalized_queries)
    return reddit.fetch_posts()

def clean_reviews(reviews:list) -> list:
    """
    Clean and preprocess user reviews by combining title and text, then removing special characters.

    This function is designed to prepare raw review data for natural language processing
    and analysis by AI agents. It combines post titles with their self-text content and
    strips away special characters, leaving only alphanumeric characters and spaces.

    Args:
        reviews (list): A list where each dictionary represents a review
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

def summarize_clusters(clusters: dict, llm_config: LLMConfig | None = None)->list:
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
    summarizer = ReviewSummarizer(llm_config)
    curated_reviews= summarizer.summarize_clusters(clusters)
    return curated_reviews

def assess_clusters(cleaned_reviews: list, llm_config: LLMConfig | None = None) -> dict:
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
    # print(type(cleaned_reviews))
    # print(cleaned_reviews)
    cluster_assessor = AssessClusters(cleaned_reviews, llm_config=llm_config)
    clusters = cluster_assessor.assess_clusters()
    return clusters

def extract_themes(curated_reviews:list, llm_config: LLMConfig | None = None)->list:
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
    print("Entered the themes extractor")
    theme_classifier = ThemeClassifier(llm_config)
    themes = [theme_classifier.extract_themes(review) for review in curated_reviews]
    return themes