"""
custom_apis.py

Secure FastAPI endpoints exposing the core pipeline tools for external agent integration.

Security features:
- API key authentication
- Rate limiting
- Input validation
- Secure file operations
- Error handling
- Audit logging
"""

from fastapi import FastAPI, HTTPException, Depends, Security, Request, status, Query
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, APIKeyQuery
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import pandas as pd
import csv
import time
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from functools import wraps
import secrets
from collections import defaultdict
from threading import Lock
import asyncio

from src.tools.custom_tools import fetch_reddit_reviews, clean_reviews, assess_clusters
from src.utilities.theme_issue_classifier import ThemeClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_security.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Security configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_query = APIKeyQuery(name="x_api_key", auto_error=False)

# Load API key from environment variable (should be set securely)
VALID_API_KEY = os.getenv("API_KEY", None)
if not VALID_API_KEY:
    logger.warning("API_KEY not set in environment. Generating temporary key.")
    VALID_API_KEY = secrets.token_urlsafe(32)
    logger.warning(f"Temporary API Key: {VALID_API_KEY}")
    logger.warning(f"Example URL: http://127.0.0.1:8000/themes?x_api_key={VALID_API_KEY}")

# Rate limiting configuration
class RateLimiter:
    """Simple in-memory rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self.lock = Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        with self.lock:
            now = time.time()
            # Clean old requests
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if now - req_time < self.window_seconds
            ]
            
            # Check if under limit
            if len(self.requests[client_id]) < self.max_requests:
                self.requests[client_id].append(now)
                return True
            return False

rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

# Initialize FastAPI with security settings
app = FastAPI(
    title="Feedback Analytics API",
    description="Secure API for Reddit review analysis",
    version="1.0.0",
    docs_url="/docs",  # Can be disabled in production
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Restrict to known origins
    allow_credentials=True,
    allow_methods=["GET"],  # Only allow GET methods
    allow_headers=["*"],
    max_age=600,
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1"]
)


async def verify_api_key(
    api_key_header: Optional[str] = Security(api_key_header),
    api_key_query: Optional[str] = Security(api_key_query)
) -> str:
    """
    Verify API key for authentication from header or query parameter.
    
    Supports both methods:
    - Header: X-API-Key: your-api-key
    - Query: ?x_api_key=your-api-key
    
    Args:
        api_key_header: API key from X-API-Key header
        api_key_query: API key from x_api_key query parameter
        
    Returns:
        Validated API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    # Try header first, then query parameter
    api_key = api_key_header or api_key_query
    
    if not api_key:
        logger.warning("Request without API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide via header 'X-API-Key' or query parameter 'x_api_key'"
        )
    
    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, VALID_API_KEY):
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return api_key


async def check_rate_limit(request: Request):
    """
    Check rate limit for the client.
    
    Args:
        request: FastAPI request object
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    client_ip = request.client.host
    
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )


def get_safe_output_path(filename: str) -> Path:
    """
    Get validated output file path within workspace.
    
    Args:
        filename: Desired output filename
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid
    """
    # Sanitize filename
    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename != filename:
        raise ValueError("Invalid filename: path traversal detected")
    
    output_path = Path(os.getcwd()) / safe_filename
    
    # Verify path is within workspace
    workspace_root = Path(os.getcwd()).resolve()
    try:
        output_path.resolve().relative_to(workspace_root)
    except ValueError:
        raise ValueError("Output path is outside workspace directory")
    
    return output_path


def secure_file_write(filepath: Path, data: pd.DataFrame, format: str = 'json'):
    """
    Securely write data to file with proper permissions.
    
    Args:
        filepath: Path to write to
        data: DataFrame to write
        format: 'json' or 'csv'
    """
    try:
        if format == 'json':
            data.to_json(str(filepath), orient='records', indent=2)
        elif format == 'csv':
            data.to_csv(str(filepath), index=False, quoting=csv.QUOTE_ALL, quotechar='"')
        
        # Set secure permissions (owner read/write only)
        os.chmod(str(filepath), 0o600)
        logger.info(f"Securely wrote {format} to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to write file {filepath}: {e}")
        raise


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all API requests for audit trail."""
    start_time = time.time()
    client_ip = request.client.host
    
    logger.info(f"Request: {request.method} {request.url.path} from {client_ip}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} in {process_time:.2f}s")
    
    return response


@app.get("/health")
async def health_check():
    """Public health check endpoint (no auth required)."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get(
    "/reviews",
    summary="Fetch raw Reddit reviews",
    description="Fetch raw Reddit posts and save to secure storage.",
    tags=["agent_tools", "data_collection"],
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def get_reviews():
    """
    Fetch raw Reddit reviews with authentication and rate limiting.
    
    Returns:
        JSON array of review records
        
    Raises:
        HTTPException: On errors
    """
    try:
        logger.info("Fetching Reddit reviews")
        
        # Run blocking operation in thread pool
        reviews = await asyncio.to_thread(fetch_reddit_reviews)
        
        if not reviews:
            logger.warning("No reviews fetched")
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"detail": "No reviews found"}
            )
        
        # Validate and limit response size
        max_reviews = 1000
        if len(reviews) > max_reviews:
            logger.warning(f"Truncating {len(reviews)} reviews to {max_reviews}")
            reviews = reviews[:max_reviews]
        
        df = pd.DataFrame(reviews)
        
        # Secure file operations
        json_path = get_safe_output_path("all_posts.json")
        csv_path = get_safe_output_path("all_posts.csv")
        
        secure_file_write(json_path, df, 'json')
        secure_file_write(csv_path, df, 'csv')
        
        logger.info(f"Successfully fetched {len(reviews)} reviews")
        
        return JSONResponse(content={
            "reviews": df.to_dict(orient="records"),
            "count": len(reviews),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in get_reviews: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch reviews"
        )


@app.get(
    "/summarized_reviews",
    summary="Fetch summarized Reddit reviews",
    description="Fetch and summarize Reddit posts with LLM processing.",
    tags=["agent_tools", "preprocessing"],
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def get_summarized_reviews():
    """
    Fetch and summarize Reddit reviews with authentication.
    
    Returns:
        JSON array of summarized review records
        
    Raises:
        HTTPException: On errors
    """
    try:
        logger.info("Fetching and summarizing reviews")
        
        # Run blocking operations in thread pool
        reviews = await asyncio.to_thread(fetch_reddit_reviews)
        
        if not reviews:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"detail": "No reviews found"}
            )
        
        summarized_reviews = await asyncio.to_thread(clean_reviews, reviews)
        
        # Validate results
        if not summarized_reviews:
            logger.warning("Summarization produced no results")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Summarization failed"}
            )
        
        df_summarized = pd.DataFrame(summarized_reviews)
        
        # Secure file operations
        json_path = get_safe_output_path("summaries.json")
        csv_path = get_safe_output_path("summaries.csv")
        
        secure_file_write(json_path, df_summarized, 'json')
        secure_file_write(csv_path, df_summarized, 'csv')
        
        logger.info(f"Successfully summarized {len(summarized_reviews)} reviews")
        
        return JSONResponse(content={
            "summaries": df_summarized.to_dict(orient="records"),
            "count": len(summarized_reviews),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in get_summarized_reviews: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to summarize reviews"
        )


@app.get(
    "/clusters",
    summary="Compute clusters of summarized reviews",
    description="Cluster reviews using ML-based similarity analysis.",
    tags=["agent_tools", "clustering"],
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def get_clusters():
    """
    Compute review clusters with authentication.
    
    Returns:
        JSON object with clusters and metadata
        
    Raises:
        HTTPException: On errors
    """
    try:
        logger.info("Computing review clusters")
        
        # Run blocking operations in thread pool
        reviews = await asyncio.to_thread(fetch_reddit_reviews)
        
        if not reviews:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"detail": "No reviews found"}
            )
        
        summarized_reviews = await asyncio.to_thread(clean_reviews, reviews)
        
        start = time.time()
        clusters = await asyncio.to_thread(assess_clusters, summarized_reviews)
        end = time.time()
        
        if not clusters:
            logger.warning("Clustering produced no results")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Clustering failed"}
            )
        
        df_clusters = pd.DataFrame(clusters)
        
        # Secure file operations
        json_path = get_safe_output_path("clusters.json")
        csv_path = get_safe_output_path("clusters.csv")
        
        secure_file_write(json_path, df_clusters, 'json')
        secure_file_write(csv_path, df_clusters, 'csv')
        
        logger.info(f"Successfully computed {len(clusters)} clusters in {end-start:.2f}s")
        
        return JSONResponse(content={
            "clusters": df_clusters.to_dict(orient="records"),
            "count": len(clusters),
            "time_taken": round(end - start, 2),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in get_clusters: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute clusters"
        )


@app.get(
    "/themes",
    summary="Extract themes and issues from reviews",
    description="Extract themes using LLM-based classification.",
    tags=["agent_tools", "classification"],
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def get_themes():
    """
    Extract themes from reviews with authentication.
    
    Returns:
        JSON object with themes and metadata
        
    Raises:
        HTTPException: On errors
    """
    try:
        logger.info("Extracting themes from reviews")
        
        # Run blocking operations in thread pool
        reviews = await asyncio.to_thread(fetch_reddit_reviews)
        
        if not reviews:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"detail": "No reviews found"}
            )
        
        summarized_reviews = await asyncio.to_thread(clean_reviews, reviews)
        
        start = time.time()
        theme_classifier = ThemeClassifier()
        
        # Process themes with proper async handling
        themes = []
        for review in summarized_reviews:
            theme = await asyncio.to_thread(theme_classifier.extract_themes, review)
            themes.append(theme)
        
        end = time.time()
        
        if not themes:
            logger.warning("Theme extraction produced no results")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Theme extraction failed"}
            )
        
        df_themes = pd.DataFrame(themes)
        
        # Secure file operations
        json_path = get_safe_output_path("themes.json")
        csv_path = get_safe_output_path("themes.csv")
        
        secure_file_write(json_path, df_themes, 'json')
        secure_file_write(csv_path, df_themes, 'csv')
        
        logger.info(f"Successfully extracted {len(themes)} themes in {end-start:.2f}s")
        
        return JSONResponse(content={
            "themes": df_themes.to_dict(orient="records"),
            "count": len(themes),
            "time_taken": round(end - start, 2),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in get_themes: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract themes"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Secure server configuration
    uvicorn.run(
        app,
        host="127.0.0.1",  # Bind to localhost only
        port=8000,
        log_level="info",
        access_log=True,
        server_header=False,  # Don't expose server info
        limit_concurrency=10,  # Limit concurrent connections
        limit_max_requests=1000,  # Restart worker after N requests
        timeout_keep_alive=5,  # Keep-alive timeout
    )



