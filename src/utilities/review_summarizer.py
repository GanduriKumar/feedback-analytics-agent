import dspy, time, csv, pandas as pd
class ReviewSummarizer:
    """
    ReviewSummarizer: A utility class for summarizing customer reviews and review clusters.
    This class provides methods to generate concise summaries of individual reviews
    and groups of clustered reviews using DSPy's language model capabilities.
    Methods
    -------
    summarize_review(review: str) -> str
        Generates a summary of a single review document.
        Parameters:
            review (str): The review text to be summarized.
        Returns:
            str: A concise summary of the input review.
        Example:
            >>> summarizer = ReviewSummarizer()
            >>> summary = summarizer.summarize_review("This product is great...")
    summarize_clusters(clusters: dict) -> list
        Summarizes multiple clusters of reviews into a list of summaries.
        Parameters:
            clusters (dict): A dictionary containing clustered reviews where keys 
                            represent cluster identifiers and values contain review data.
        Returns:
            list: A list of summary strings, one for each cluster.
        Example:
            >>> summarizer = ReviewSummarizer()
            >>> clusters = {'cluster_1': ['review1', 'review2'], 'cluster_2': ['review3']}
            >>> summaries = summarizer.summarize_clusters(clusters)
    Notes
    -----
    - Uses DSPy with Ollama's Mistral model for natural language processing
    - Requires a local Ollama server running on http://localhost:11434
    - Combines multiple reviews within a cluster before summarization
    """
    def __init__(self):
        pass

    def summarize_review(self, review: str) -> str:
        dspy.configure(lm= dspy.LM('ollama_chat/mistral:latest',api_base='http://localhost:11434',api_key=''))
        summarize = dspy.ChainOfThought('document -> summary')
        response = summarize(document=review)
        print(response.summary)
        return response.summary

    def summarize_clusters(self, clusters:dict)->list:
        df_clusters = pd.DataFrame(clusters)
        columns_list = df_clusters.columns.to_list()
        # create list by combining and summarizing reviews in a clusters
        cluster_reviews = [df_clusters[item].tolist() for item in columns_list]
        curated_reviews = []
        for review in cluster_reviews:
            combined_review = " ".join([str(item) for item in review if item is not None])
            summary = self.summarize_review(combined_review)
            curated_reviews.append(summary)

        return curated_reviews
