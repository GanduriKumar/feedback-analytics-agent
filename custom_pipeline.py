from src.tools.custom_llm import CustomLLMModel
from src.tools.custom_tools import fetch_reddit_reviews, clean_reviews
from langchain_chroma import Chroma
import pandas as pd, chromadb, csv, dotenv, os
from typing import List
import hashlib
import time
from pathlib import Path

# Load environment variables
dotenv.load_dotenv()

# Configuration from environment with defaults
BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '100'))
UPSERT_BATCH_SIZE = int(os.getenv('UPSERT_BATCH_SIZE', '500'))
CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './chroma_db')
REVIEW_COLLECTION_NAME = os.getenv('REVIEW_COLLECTION_NAME', 'reviews')

print("=" * 60)
print("Starting Custom Pipeline Execution")
print(f"Batch size: {BATCH_SIZE}, Upsert batch size: {UPSERT_BATCH_SIZE}")
print("=" * 60)

# Fetch raw Reddit reviews using PRAW integration
print("\n[1/4] Fetching Reddit reviews...")
start_time = time.time()
raw_reviews = fetch_reddit_reviews()
print(f"Fetched {len(raw_reviews)} reviews in {time.time() - start_time:.2f}s")

# Clean and preprocess the fetched reviews (remove noise, format text)
print("\n[2/4] Cleaning and preprocessing reviews...")
start_time = time.time()
cleaned_reviews = clean_reviews(raw_reviews)
print(f"Cleaned {len(cleaned_reviews)} reviews in {time.time() - start_time:.2f}s")

# Remove duplicates early to avoid processing redundant data
print("\n[3/4] Removing duplicate reviews...")
original_count = len(cleaned_reviews)
unique_reviews = list(dict.fromkeys(cleaned_reviews))  # Preserve order while removing duplicates
cleaned_reviews = unique_reviews
print(f"Removed {original_count - len(cleaned_reviews)} duplicates, {len(cleaned_reviews)} unique reviews remain")

# Early exit if no reviews to process
if not cleaned_reviews:
    print("\nNo reviews to process. Exiting.")
    exit(0)

# Initialize the custom LLM model for Ollama integration (lazy loading)
print("\n[4/4] Initializing embedding model...")
model = CustomLLMModel()
embeddings_model = model.create_embedding()

# Batch embedding generation (10x faster than one-by-one)
print("\nGenerating embeddings in batches...")
all_vectors = []
embed_start = time.time()

try:
    for i in range(0, len(cleaned_reviews), BATCH_SIZE):
        batch_start = time.time()
        batch = cleaned_reviews[i:i + BATCH_SIZE]
        
        # Generate embeddings for current batch
        batch_vectors = embeddings_model.embed_documents(batch)
        all_vectors.extend(batch_vectors)
        
        batch_time = time.time() - batch_start
        progress = min(i + BATCH_SIZE, len(cleaned_reviews))
        percentage = (progress / len(cleaned_reviews)) * 100
        
        print(f"  Batch {i//BATCH_SIZE + 1}: Processed {progress}/{len(cleaned_reviews)} ({percentage:.1f}%) in {batch_time:.2f}s")
    
    total_embed_time = time.time() - embed_start
    print(f"\nCompleted embedding generation in {total_embed_time:.2f}s ({total_embed_time/len(cleaned_reviews):.3f}s per review)")
    
except Exception as e:
    print(f"\nError during embedding generation: {e}")
    print("Attempting to save partially processed embeddings...")
    if all_vectors:
        # Save what we have so far
        pd.DataFrame({
            'review': cleaned_reviews[:len(all_vectors)],
            'embedding': all_vectors
        }).to_pickle('partial_embeddings.pkl')
        print(f"Saved {len(all_vectors)} embeddings to partial_embeddings.pkl")
    raise

# Persist to ChromaDB efficiently
print("\nPersisting embeddings to ChromaDB...")
db_start = time.time()

try:
    # Ensure ChromaDB directory exists
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    
    # Create persistent client with proper settings
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=chromadb.Settings(
            anonymized_telemetry=False,
            allow_reset=False
        )
    )
    
    # Get or create collection
    reviews_collection = client.get_or_create_collection(
        name=REVIEW_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity for better semantic search
    )
    
    # Check existing documents to avoid re-adding
    try:
        existing_count = reviews_collection.count()
        print(f"Collection '{REVIEW_COLLECTION_NAME}' currently contains {existing_count} documents")
    except:
        existing_count = 0
    
    # Batch upsert (faster than individual adds)
    print(f"\nUpserting embeddings in batches of {UPSERT_BATCH_SIZE}...")
    
    for i in range(0, len(cleaned_reviews), UPSERT_BATCH_SIZE):
        batch_start = time.time()
        end_idx = min(i + UPSERT_BATCH_SIZE, len(cleaned_reviews))
        
        # Generate unique IDs using hash of content (prevents duplicates)
        batch_ids = [
            hashlib.md5(cleaned_reviews[j].encode('utf-8')).hexdigest()
            for j in range(i, end_idx)
        ]
        
        # Upsert batch (will update if ID exists, add if new)
        reviews_collection.upsert(
            ids=batch_ids,
            embeddings=all_vectors[i:end_idx],
            documents=cleaned_reviews[i:end_idx]
        )
        
        batch_time = time.time() - batch_start
        percentage = (end_idx / len(cleaned_reviews)) * 100
        print(f"  Batch {i//UPSERT_BATCH_SIZE + 1}: Upserted {end_idx}/{len(cleaned_reviews)} ({percentage:.1f}%) in {batch_time:.2f}s")
    
    total_db_time = time.time() - db_start
    final_count = reviews_collection.count()
    print(f"\nCompleted ChromaDB persistence in {total_db_time:.2f}s")
    print(f"Collection now contains {final_count} total documents")
    
except Exception as e:
    print(f"\nError during ChromaDB persistence: {e}")
    print("Data may be partially saved. Check ChromaDB directory.")
    raise

print("\n" + "=" * 60)
print("Pipeline execution completed successfully!")
print(f"Total reviews processed: {len(cleaned_reviews)}")
print(f"Total embeddings generated: {len(all_vectors)}")
print("=" * 60)





