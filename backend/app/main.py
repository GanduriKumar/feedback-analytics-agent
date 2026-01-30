"""
Unified FastAPI Backend for Feedback Analytics Agent

Combines A2A-compatible API and custom tool endpoints into a single service.
"""

import sys
import os
from pathlib import Path

# Add backend directory to Python path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from fastapi import FastAPI, HTTPException, Depends, Security, Request, status, Query
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, APIKeyQuery
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from datetime import datetime, UTC
import logging
import json
import re
import dotenv
import secrets
import time
import asyncio
from functools import lru_cache
from collections import defaultdict
from threading import Lock
import pandas as pd

# Import models
from app.models.schemas import (
    AnalysisRequest, SearchRequest, ThemeData, AnalysisResponse,
    SearchResponse, HealthResponse, ClusterResponse, ClusterRequest
)

# Import core functionality
from app.core.analyzer import execute_graph_pipeline
from app.core.vector_db import query_vector_db
from app.tools.custom_tools import fetch_reddit_reviews, clean_reviews, assess_clusters
from app.utilities.theme_issue_classifier import ThemeClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

# Security configuration
API_KEY = os.getenv("API_KEY") or "default_dev_key_change_in_production"
logger.info(f"API initialized with key: {API_KEY[:8]}...")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="x_api_key", auto_error=False)


# Rate limiting
class RateLimiter:
    """Simple in-memory rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self.lock = Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        with self.lock:
            now = time.time()
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if now - req_time < self.window_seconds
            ]
            
            if len(self.requests[client_id]) < self.max_requests:
                self.requests[client_id].append(now)
                return True
            return False

rate_limiter = RateLimiter(max_requests=30, window_seconds=60)


# Initialize FastAPI app
app = FastAPI(
    title="Feedback Analytics Backend",
    description="Unified backend service for feedback analysis with A2A compatibility",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware - configured for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)


# Security middleware
async def verify_api_key(
    request: Request,
    api_key_header: Optional[str] = Security(api_key_header),
    api_key_query: Optional[str] = Security(api_key_query)
) -> str:
    """Verify API key from header or query parameter. Skip for OPTIONS requests."""
    # Allow OPTIONS requests to pass through for CORS preflight
    if request.method == "OPTIONS":
        return "options"
    
    key = api_key_header or api_key_query
    
    if not key or not secrets.compare_digest(key, API_KEY):
        logger.warning(f"Unauthorized access attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    
    return key


async def check_rate_limit(request: Request):
    """Check rate limit for the client. Skip for OPTIONS requests."""
    # Allow OPTIONS requests to pass through for CORS preflight
    if request.method == "OPTIONS":
        return
    
    client_ip = request.client.host
    
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )


def get_safe_output_path(filename: str) -> Path:
    """Get validated output file path within workspace."""
    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename != filename:
        raise ValueError("Invalid filename: path traversal detected")
    
    if not safe_filename.endswith(('.json', '.csv')):
        safe_filename += '.json'
    
    output_path = Path(os.getcwd()) / safe_filename
    workspace_root = Path(os.getcwd()).resolve()
    
    try:
        output_path.resolve().relative_to(workspace_root)
    except ValueError:
        raise ValueError("Output path is outside workspace directory")
    
    return output_path


def _parse_csv_query_param(value: Optional[str]) -> List[str]:
    """Parse a comma-separated query param into a list of non-empty strings."""
    if not value:
        return []
    parts = [p.strip() for p in value.split(',')]
    return [p for p in parts if p]


@lru_cache(maxsize=1)
def get_capabilities() -> List[str]:
    """Get list of available capabilities."""
    return [
        "feedback_analysis",
        "theme_extraction",
        "sentiment_analysis",
        "semantic_search",
        "cluster_analysis",
        "review_summarization",
        "data_collection"
    ]


# ============================================================================
# PUBLIC ENDPOINTS (No authentication required)
# ============================================================================

@app.get("/", tags=["root"])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Feedback Analytics Backend",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/health"
    }


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Public health check endpoint for monitoring and discovery."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(UTC).isoformat(),
        capabilities=get_capabilities()
    )


# ============================================================================
# A2A-COMPATIBLE ENDPOINTS (Analysis & Search)
# ============================================================================

@app.post(
    "/api/analyze",
    response_model=AnalysisResponse,
    tags=["analysis"],
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def analyze_feedback(request: AnalysisRequest):
    """
    Execute complete feedback analysis pipeline.
    
    Runs LangGraph-based pipeline:
    1. Query vector database for relevant reviews
    2. Cluster similar reviews
    3. Summarize clusters
    4. Extract themes using LLM
    """
    try:
        logger.info(f"Analysis request: {request.query[:100]}")
        start_time = datetime.now(UTC)
        
        # Execute pipeline in thread pool
        themes = await asyncio.to_thread(
            execute_graph_pipeline,
            request.query,
            request.llm_config
        )
        
        end_time = datetime.now(UTC)
        processing_time = (end_time - start_time).total_seconds()
        
        theme_data = [ThemeData(**theme) for theme in themes]
        
        response = AnalysisResponse(
            query=request.query,
            themes=theme_data,
            total_themes=len(theme_data),
            timestamp=end_time.isoformat(),
            processing_time=processing_time
        )
        
        logger.info(f"Analysis completed: {len(theme_data)} themes in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post(
    "/api/search",
    response_model=SearchResponse,
    tags=["search"],
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def search_reviews(request: SearchRequest):
    """Search for semantically similar reviews in vector database."""
    try:
        logger.info(f"Search request: {request.query[:100]}")
        
        results = await asyncio.to_thread(
            query_vector_db,
            query_text=request.query,
            n_results=request.n_results,
            output_file="api_search_results.csv"
        )
        
        response = SearchResponse(
            query=request.query,
            results=results,
            count=len(results),
            timestamp=datetime.now(UTC).isoformat()
        )
        
        logger.info(f"Search completed: {len(results)} results")
        return response
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


# ============================================================================
# TOOL ENDPOINTS (Data Collection & Processing)
# ============================================================================

@app.get(
    "/api/tools/collect",
    tags=["tools"],
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def collect_reviews(
    queries: Optional[str] = Query(None, description="Comma-separated search queries"),
    sources: Optional[str] = Query(None, description="Comma-separated sources (e.g., reddit)"),
):
    """Fetch raw reviews based on selected sources and queries.

    Currently supported sources:
    - reddit

    If queries are omitted, the backend will fall back to its configured CSV.
    """
    try:
        logger.info("Collecting reviews from Reddit")

        requested_queries = _parse_csv_query_param(queries)
        requested_sources = _parse_csv_query_param(sources)

        # Default behavior: if sources not specified, collect from reddit (legacy behavior)
        if not requested_sources:
            requested_sources = ["reddit"]

        supported_sources = {"reddit"}
        unsupported = [s for s in requested_sources if s not in supported_sources]

        if "reddit" not in requested_sources:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No supported sources selected. Supported: {sorted(supported_sources)}",
            )

        reviews = await asyncio.to_thread(fetch_reddit_reviews, requested_queries)
        
        if not reviews:
            return {
                "count": 0,
                "reviews": [],
                "total": 0,
                "timestamp": datetime.now(UTC).isoformat(),
                "queries_used": requested_queries,
                "sources_used": ["reddit"],
                "warnings": ["No reviews found"],
            }
        
        resp: Dict[str, Any] = {
            "count": len(reviews),
            "reviews": reviews[:100],  # Limit response size
            "total": len(reviews),
            "timestamp": datetime.now(UTC).isoformat()
        }

        # Optional metadata for UI
        resp["queries_used"] = requested_queries
        resp["sources_used"] = ["reddit"]
        if unsupported:
            resp["warnings"] = [f"Unsupported sources ignored: {', '.join(unsupported)}"]

        return resp
        
    except Exception as e:
        logger.error(f"Collection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Review collection failed: {str(e)}"
        )


@app.post(
    "/api/tools/cluster",
    response_model=ClusterResponse,
    tags=["tools"],
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def cluster_reviews(payload: ClusterRequest):
    """Cluster provided reviews by semantic similarity."""
    try:
        logger.info(f"Clustering {len(payload.reviews)} reviews")
        start_time = time.time()
        
        clusters = await asyncio.to_thread(assess_clusters, payload.reviews, payload.llm_config)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        response = ClusterResponse(
            clusters=clusters,
            count=len(clusters),
            time_taken=round(processing_time, 2),
            timestamp=datetime.now(UTC).isoformat()
        )
        
        logger.info(f"Clustering completed: {len(clusters)} clusters in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Clustering failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clustering failed: {str(e)}"
        )


@app.get(
    "/api/capabilities",
    tags=["discovery"]
)
async def get_agent_capabilities():
    """Return structured information about agent capabilities for A2A discovery."""
    return {
        "agent_name": "Feedback Analytics Agent",
        "version": "1.0.0",
        "capabilities": get_capabilities(),
        "endpoints": {
            "/api/analyze": {
                "method": "POST",
                "description": "Execute complete feedback analysis pipeline",
                "requires_auth": True
            },
            "/api/search": {
                "method": "POST",
                "description": "Semantic search for similar reviews",
                "requires_auth": True
            },
            "/api/tools/collect": {
                "method": "GET",
                "description": "Collect reviews from Reddit",
                "requires_auth": True
            },
            "/api/tools/cluster": {
                "method": "POST",
                "description": "Cluster reviews by similarity",
                "requires_auth": True
            }
        },
        "authentication": {
            "type": "API Key",
            "methods": ["Header: X-API-Key", "Query: x_api_key"]
        }
    }


# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all API requests for audit trail."""
    start_time = time.time()
    
    logger.info(f"{request.method} {request.url.path} - Client: {request.client.host}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} in {process_time:.2f}s")
    
    return response


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Feedback Analytics Backend")
    logger.info(f"API Documentation: http://localhost:8000/api/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
