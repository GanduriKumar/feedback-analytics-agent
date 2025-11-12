from src.tools.custom_llm import CustomLLMModel
from src.tools.custom_tools import fetch_reddit_reviews, clean_reviews
from langchain_chroma import Chroma
import pandas as pd, chromadb, csv, dotenv, os

# Fetch raw Reddit reviews using PRAW integration
raw_reviews = fetch_reddit_reviews()
# raw_reviews = pd.read_csv("./all_posts.csv").to_dict(orient="records")  # Assuming reviews are stored in a CSV file

# Clean and preprocess the fetched reviews (remove noise, format text)
cleaned_reviews = clean_reviews(raw_reviews)

# Initialize the custom LLM model for Ollama integration
model = CustomLLMModel()

# Create embedding model for converting text to vectors
embeddings_model = model.create_embedding()

# Generate vector embeddings for all cleaned reviews
vectors = embeddings_model.embed_documents(cleaned_reviews)
# for vector in vectors:
#     print(vector)


# Create a persistent Chroma client (data stored in ./chroma_db)
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get a collection
dotenv.load_dotenv()
reviews_collection = client.get_or_create_collection(name=os.getenv("REVIEW_COLLECTION_NAME"))

# Add documents with precomputed embeddings
reviews_collection.add(
    ids=[f"review_{i}" for i, _ in enumerate(cleaned_reviews, start=1)],        # Unique IDs for each document
    embeddings=vectors,              # Precomputed embeddings
    documents=cleaned_reviews                     # Original text documents
)





