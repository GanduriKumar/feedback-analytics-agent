import os, pandas as pd, chromadb, csv, dotenv
from src.tools.custom_llm import CustomLLMModel


def query_vector_db(query_text: str, n_results: int = 50):
    """
    Query a ChromaDB vector database to find similar documents based on semantic similarity.
    This function performs a similarity search on a persisted ChromaDB collection using
    an embedding model. It converts the query text to a vector embedding and retrieves
    the most similar documents from the database.
    Args:
        query_text (str): The text query to search for similar documents in the vector database.
        n_results (int, optional): Maximum number of similar results to return. Defaults to 50.
    Returns:
        None: The function saves results to 'search_results.csv' but does not return a value.
    Side Effects:
        - Loads environment variables from .env file
        - Connects to a persistent ChromaDB instance at './chroma_db'
        - Writes search results to 'search_results.csv' in the current directory
    Environment Variables Required:
        REVIEW_COLLECTION_NAME: Name of the ChromaDB collection to query
    Output File:
        search_results.csv: CSV file containing the documents that match the query,
                           with proper quoting for all fields
    Example:
        >>> query_vector_db("What are customers saying about product quality?", n_results=10)
        # Creates search_results.csv with top 10 similar documents
    Notes:
        - Uses CustomLLMModel for embeddings generation (Ollama integration)
        - Performs semantic similarity search, not exact text matching
        - Results are ordered by similarity score (most similar first)
    """
    # Create a persistent Chroma client (data stored in ./chroma_db)
    client = chromadb.PersistentClient(path="./chroma_db")

    # Create or get a collection
    dotenv.load_dotenv()
    reviews_collection = client.get_or_create_collection(name=os.getenv("REVIEW_COLLECTION_NAME"))

    # Initialize the custom LLM model for Ollama integration
    model = CustomLLMModel()
    # Create embedding model for converting text to vectors
    embeddings_model = model.create_embedding()
    query_embedding = embeddings_model.embed_query(query_text)  # Example query

    # Perform similarity search
    results = reviews_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results # Number of results to return
    )

    df = pd.DataFrame(results['documents'][0])
    df.to_csv("search_results.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"')
    # print(type(results['documents'][0]))
    return results['documents'][0]

# if __name__ == "__main__":
#     user_query = input("Enter your search query: ")
#     print(query_vector_db(query_text=user_query, n_results=20))

