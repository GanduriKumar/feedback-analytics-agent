from sklearn.cluster import KMeans
from src.tools.custom_llm import CustomLLMModel
import pandas as pd, csv

class AssessClusters:
    def __init__(self, reviews:list[str], num_clusters:int=20):
        self.reviews = reviews
        self.num_clusters = num_clusters
    def assess_clusters(self) -> dict:
        """
        Tool:
          name: assess_clusters
          description: >
            Cluster a list of textual reviews using sentence embeddings and KMeans.
            Produces a mapping of cluster id -> list of reviews assigned to that cluster.
          inputs:
            - name: reviews
              type: list[str]
              required: true
              description: List of textual review strings to cluster. Each element must be a str.
          outputs:
            - name: clusters
              type: dict[int, list[str]]
              description: Mapping from integer cluster labels to lists of reviews.
          implementation:
            embedding_model: sentence_transformers/all-MiniLM-L6-V2
            clustering_algorithm: sklearn.cluster.KMeans
          parameters:
            - name: num_clusters
              type: int
              default: 10
              description: Number of clusters used by KMeans (hard-coded by default).
            - name: random_state
              type: int
              default: 42
              description: Random seed for deterministic KMeans results.
          side_effects:
            - May download model weights when loading SentenceTransformer.
            - Prints cluster membership to stdout for human inspection.
            - Allocates memory for embeddings (size ~ len(reviews) x model_dim).
          errors:
            - TypeError: If `reviews` is not a list or contains non-str elements.
            - ValueError: If `reviews` is an empty list.
            - ImportError: If required packages are not installed.
            - RuntimeError: If the model cannot be downloaded/loaded.
          recommendations:
            - For large datasets, pass precomputed embeddings or batch model.encode.
            - Consider making the model and num_clusters injectable for reuse and testing.
            - Use MiniBatchKMeans or HDBSCAN for very large inputs.
        """
        embedding_model = CustomLLMModel().create_embedding()
        embeddings = embedding_model.embed_documents(self.reviews)
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        kmeans.fit(embeddings)
        labels = kmeans.labels_

        clusters = {}
        for label, review in zip(labels, self.reviews):
            clusters.setdefault(label, []).append(review)

        # for cluster_id, cluster_reviews in clusters.items():
        #     print(f"Cluster {cluster_id}")
        #     for r in cluster_reviews:
        #         print(f"- {r}")


        # Step 1: Check lengths of each list
        lengths = {key: len(value) for key, value in clusters.items()}
        # print("Lengths of each column:", lengths)
      # Step 2: Find the maximum length
        max_length = max(lengths.values())
      # Step 3: Pad shorter lists with None
        for key in clusters:
          while len(clusters[key]) < max_length:
            clusters[key].append(None)
        return clusters




