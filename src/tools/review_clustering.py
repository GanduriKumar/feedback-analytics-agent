from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd


def assess_clusters(reviews:list)->dict:

    model = SentenceTransformer("all-MiniLM-L6-V2")
    embeddings = model.encode(reviews)

    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

    clusters = {}
    for label , review in zip(labels,reviews):
        clusters.setdefault(label,[]).append(review)

    for cluster_id, cluster_reviews in clusters.items():
        print(f"Cluster {cluster_id}")
        for r in cluster_reviews:
            print(f"- {r}")
        # print()
    return clusters




