from app.tools.custom_llm import CustomLLMModel
from app.tools.custom_tools import fetch_reddit_reviews, clean_reviews
from langchain_chroma import Chroma
import pandas as pd, chromadb, csv, dotenv, os
from typing import List
import hashlib
import time
from pathlib import Path


def log(message: str):
    """Print immediately for real-time CLI feedback."""
    print(message, flush=True)

# Load environment variables
dotenv.load_dotenv()

# Configuration from environment with defaults
BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '100'))
UPSERT_BATCH_SIZE = int(os.getenv('UPSERT_BATCH_SIZE', '500'))
CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './chroma_db')
REVIEW_COLLECTION_NAME = os.getenv('REVIEW_COLLECTION_NAME', 'reviews')

log("=" * 60)
log("Starting Custom Pipeline Execution")
log(f"Batch size: {BATCH_SIZE}, Upsert batch size: {UPSERT_BATCH_SIZE}")
log("=" * 60)

# Fetch raw Reddit reviews using PRAW integration
log("\n[1/4] Fetching Reddit reviews...")
start_time = time.time()
raw_reviews = fetch_reddit_reviews()
log(f"Fetched {len(raw_reviews)} reviews in {time.time() - start_time:.2f}s")

def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# Clean and preprocess the fetched reviews (remove noise, format text)
log("\n[2/5] Cleaning and preprocessing reviews...")
start_time = time.time()
cleaned_reviews = clean_reviews(raw_reviews)
log(f"Cleaned {len(cleaned_reviews)} reviews in {time.time() - start_time:.2f}s")

# Remove duplicates early to avoid processing redundant data
log("\n[3/5] Removing duplicate reviews...")
original_count = len(cleaned_reviews)
unique_reviews = list(dict.fromkeys(cleaned_reviews))  # Preserve order while removing duplicates
cleaned_reviews = unique_reviews
log(f"Removed {original_count - len(cleaned_reviews)} duplicates, {len(cleaned_reviews)} unique reviews remain")

# Early exit if no reviews to process
if not cleaned_reviews:
    log("\nNo reviews to process. Exiting.")
    exit(0)

# Initialize the custom LLM model for Ollama integration (lazy loading)
log("\n[4/5] Initializing embedding model and collection...")
model = CustomLLMModel()
effective_collection_name = model.get_embedding_collection_name(REVIEW_COLLECTION_NAME)
embeddings_model = model.create_embedding()

# Prepare Chroma client/collection for delta detection
Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=chromadb.Settings(
        anonymized_telemetry=False,
        allow_reset=False
    )
)
reviews_collection = client.get_or_create_collection(
    name=effective_collection_name,
    metadata={"hnsw:space": "cosine"}
)

try:
    existing_count = reviews_collection.count()
    log(f"Collection '{effective_collection_name}' currently contains {existing_count} documents")
except Exception:
    existing_count = 0

# Delta detection: filter out already-stored reviews by content hash
log("\n[5/5] Filtering already-stored reviews...")
filtered_reviews: List[str] = []
filtered_ids: List[str] = []

CHECK_BATCH = max(100, UPSERT_BATCH_SIZE)
for i in range(0, len(cleaned_reviews), CHECK_BATCH):
    batch = cleaned_reviews[i:i + CHECK_BATCH]
    batch_ids = [compute_hash(text) for text in batch]

    existing = reviews_collection.get(ids=batch_ids, include=[])
    existing_ids = set(existing.get('ids', [])) if existing else set()

    for text, doc_id in zip(batch, batch_ids):
        if doc_id in existing_ids:
            continue
        filtered_reviews.append(text)
        filtered_ids.append(doc_id)

log(f"Found {len(filtered_reviews)} new reviews (skipped {len(cleaned_reviews) - len(filtered_reviews)} existing)")

# Early exit if no new reviews to process
if not filtered_reviews:
    log("No new reviews to process. Exiting without embedding.")
    exit(0)

# Batch embedding generation (10x faster than one-by-one)
log("\nGenerating embeddings for new reviews in batches...")
all_vectors = []
embed_start = time.time()

try:
    for i in range(0, len(filtered_reviews), BATCH_SIZE):
        batch_start = time.time()
        batch = filtered_reviews[i:i + BATCH_SIZE]
        
        # Generate embeddings for current batch
        batch_vectors = embeddings_model.embed_documents(batch)
        all_vectors.extend(batch_vectors)
        
        batch_time = time.time() - batch_start
        progress = min(i + BATCH_SIZE, len(filtered_reviews))
        percentage = (progress / len(filtered_reviews)) * 100
        
        log(f"  Batch {i//BATCH_SIZE + 1}: Processed {progress}/{len(filtered_reviews)} ({percentage:.1f}%) in {batch_time:.2f}s")
    
    total_embed_time = time.time() - embed_start
    log(f"\nCompleted embedding generation in {total_embed_time:.2f}s ({total_embed_time/len(filtered_reviews):.3f}s per review)")
    
except Exception as e:
    log(f"\nError during embedding generation: {e}")
    log("Attempting to save partially processed embeddings...")
    if all_vectors:
        # Save what we have so far
        pd.DataFrame({
            'review': filtered_reviews[:len(all_vectors)],
            'embedding': all_vectors
        }).to_pickle('partial_embeddings.pkl')
        log(f"Saved {len(all_vectors)} embeddings to partial_embeddings.pkl")
    raise

# Persist to ChromaDB efficiently
log("\nPersisting new embeddings to ChromaDB...")
db_start = time.time()

try:
    # Batch upsert (faster than individual adds)
    log(f"\nUpserting embeddings in batches of {UPSERT_BATCH_SIZE}...")
    
    for i in range(0, len(filtered_reviews), UPSERT_BATCH_SIZE):
        batch_start = time.time()
        end_idx = min(i + UPSERT_BATCH_SIZE, len(filtered_reviews))
        
        batch_ids = filtered_ids[i:end_idx]
        
        # Upsert batch (will update if ID exists, add if new)
        reviews_collection.upsert(
            ids=batch_ids,
            embeddings=all_vectors[i:end_idx],
            documents=filtered_reviews[i:end_idx]
        )
        
        batch_time = time.time() - batch_start
        percentage = (end_idx / len(filtered_reviews)) * 100
        log(f"  Batch {i//UPSERT_BATCH_SIZE + 1}: Upserted {end_idx}/{len(filtered_reviews)} ({percentage:.1f}%) in {batch_time:.2f}s")
    
    total_db_time = time.time() - db_start
    final_count = reviews_collection.count()
    log(f"\nCompleted ChromaDB persistence in {total_db_time:.2f}s")
    log(f"Collection now contains {final_count} total documents")
    
except Exception as e:
    log(f"\nError during ChromaDB persistence: {e}")
    log("Data may be partially saved. Check ChromaDB directory.")
    raise

log("\n" + "=" * 60)
log("Pipeline execution completed successfully!")
log(f"Total reviews processed this run: {len(filtered_reviews)} (new)")
log(f"Total embeddings generated: {len(all_vectors)}")
log("=" * 60)





