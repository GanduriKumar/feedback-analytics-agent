import os, pandas as pd, chromadb, csv, dotenv
from src.tools.custom_llm import CustomLLMModel

def query_vector_db(query_text: str, n_results: int = 50):
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