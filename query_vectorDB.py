import os
import pandas as pd
import chromadb
import csv
import dotenv
import re
from pathlib import Path
from typing import List, Optional
from src.tools.custom_llm import CustomLLMModel


def sanitize_query_text(query_text: str) -> str:
    """
    Sanitize query text to prevent injection attacks.
    
    Args:
        query_text: The raw query text to sanitize
        
    Returns:
        Sanitized query text
        
    Raises:
        ValueError: If query text is empty or contains only whitespace
    """
    if not query_text or not query_text.strip():
        raise ValueError("Query text cannot be empty")
    
    # Remove any potential control characters and limit length
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', query_text.strip())
    
    # Limit query length to prevent resource exhaustion
    max_length = 5000
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def validate_n_results(n_results: int) -> int:
    """
    Validate and constrain the number of results parameter.
    
    Args:
        n_results: Requested number of results
        
    Returns:
        Validated number of results
        
    Raises:
        ValueError: If n_results is invalid
    """
    if not isinstance(n_results, int):
        raise ValueError("n_results must be an integer")
    
    if n_results < 1:
        raise ValueError("n_results must be at least 1")
    
    # Limit maximum results to prevent resource exhaustion
    max_results = 1000
    if n_results > max_results:
        return max_results
    
    return n_results


def get_safe_db_path() -> Path:
    """
    Get a safe, validated path for the ChromaDB database.
    
    Returns:
        Validated Path object for the database
        
    Raises:
        ValueError: If DB path is invalid or outside workspace
    """
    db_path = Path(os.getcwd()) / "chroma_db"
    
    # Ensure path is within workspace (prevent path traversal)
    workspace_root = Path(os.getcwd()).resolve()
    try:
        db_path.resolve().relative_to(workspace_root)
    except ValueError:
        raise ValueError("Database path is outside workspace directory")
    
    return db_path


def get_safe_output_path(filename: str = "search_results.csv") -> Path:
    """
    Get a safe, validated path for output file.
    
    Args:
        filename: Name of the output file
        
    Returns:
        Validated Path object for output
        
    Raises:
        ValueError: If filename contains path traversal attempts
    """
    # Sanitize filename to prevent path traversal
    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename != filename:
        raise ValueError("Invalid filename: path traversal detected")
    
    output_path = Path(os.getcwd()) / safe_filename
    
    # Ensure path is within workspace
    workspace_root = Path(os.getcwd()).resolve()
    try:
        output_path.resolve().relative_to(workspace_root)
    except ValueError:
        raise ValueError("Output path is outside workspace directory")
    
    return output_path


def query_vector_db(query_text: str, n_results: int = 50, output_file: Optional[str] = None) -> List[str]:
    """
    Query a ChromaDB vector database to find similar documents based on semantic similarity.
    
    This function performs a similarity search on a persisted ChromaDB collection using
    an embedding model. It converts the query text to a vector embedding and retrieves
    the most similar documents from the database.
    
    Args:
        query_text (str): The text query to search for similar documents in the vector database.
        n_results (int, optional): Maximum number of similar results to return. Defaults to 50.
                                  Maximum allowed is 1000.
        output_file (str, optional): Name of output CSV file. Defaults to 'search_results.csv'.
    
    Returns:
        List[str]: List of documents that match the query
    
    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If database operations fail
        OSError: If file operations fail
    
    Side Effects:
        - Loads environment variables from .env file
        - Connects to a persistent ChromaDB instance at './chroma_db'
        - Writes search results to output CSV file in the current directory
    
    Environment Variables Required:
        REVIEW_COLLECTION_NAME: Name of the ChromaDB collection to query
    
    Output File:
        CSV file containing the documents that match the query,
        with proper quoting for all fields
    
    Example:
        >>> results = query_vector_db("What are customers saying about product quality?", n_results=10)
        >>> print(f"Found {len(results)} results")
    
    Notes:
        - Uses CustomLLMModel for embeddings generation (Ollama integration)
        - Performs semantic similarity search, not exact text matching
        - Results are ordered by similarity score (most similar first)
        - Input is sanitized to prevent injection attacks
        - Paths are validated to prevent directory traversal attacks
    """
    try:
        # Input validation and sanitization
        sanitized_query = sanitize_query_text(query_text)
        validated_n_results = validate_n_results(n_results)
        
        # Get safe paths
        db_path = get_safe_db_path()
        output_path = get_safe_output_path(output_file or "search_results.csv")
        
        # Load environment variables
        dotenv.load_dotenv()
        collection_name = os.getenv("REVIEW_COLLECTION_NAME")
        
        if not collection_name:
            raise ValueError("REVIEW_COLLECTION_NAME environment variable not set")
        
        # Create a persistent Chroma client with validated path
        client = chromadb.PersistentClient(path=str(db_path))
        
        # Get the collection (use get instead of get_or_create for read-only operation)
        try:
            reviews_collection = client.get_collection(name=collection_name)
        except Exception as e:
            raise RuntimeError(f"Failed to access collection '{collection_name}': {str(e)}")
        
        # Initialize the custom LLM model for Ollama integration
        model = CustomLLMModel()
        
        # Create embedding model for converting text to vectors
        embeddings_model = model.create_embedding()
        query_embedding = embeddings_model.embed_query(sanitized_query)
        
        # Perform similarity search
        results = reviews_collection.query(
            query_embeddings=[query_embedding],
            n_results=validated_n_results
        )
        
        # Validate results
        if not results or 'documents' not in results or not results['documents']:
            return []
        
        documents = results['documents'][0] if results['documents'] else []
        
        # Save results to CSV with error handling
        try:
            df = pd.DataFrame(documents, columns=['document'])
            df.to_csv(str(output_path), index=False, quoting=csv.QUOTE_ALL, quotechar='"')
        except Exception as e:
            raise OSError(f"Failed to write results to {output_path}: {str(e)}")
        
        return documents
        
    except ValueError as e:
        raise ValueError(f"Input validation error: {str(e)}")
    except RuntimeError as e:
        raise RuntimeError(f"Database operation error: {str(e)}")
    except OSError as e:
        raise OSError(f"File operation error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during query: {str(e)}")

