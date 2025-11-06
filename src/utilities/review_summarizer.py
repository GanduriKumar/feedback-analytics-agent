import dspy, time, csv, pandas as pd
class ReviewSummarizer:
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
