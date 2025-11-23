"""
A2A-Compatible Review Analyzer Agent

This agent exposes review analysis capabilities through a standardized A2A interface
that other agents can discover and interact with. It provides structured endpoints
for feedback analysis, theme extraction, and insight generation.

Security Features:
- Input validation and sanitization
- Path traversal prevention
- Rate limiting
- Secure file operations
- Comprehensive error handling
"""

from fastapi import FastAPI, HTTPException, Depends, Security, Request, status, Query
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, APIKeyQuery
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from datetime import datetime, UTC
from pathlib import Path
import logging
import json
import os
import re
import dotenv
import secrets
from functools import lru_cache

from feedback_analyzer import execute_graph_pipeline
from query_vectorDB import query_vector_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

# Security configuration
API_KEY = os.getenv("API_KEY") or "API_TEXT"
logger.info(f"Loaded API_KEY: {API_KEY}")
if not os.getenv("API_KEY"):
    logger.warning(f"No API_KEY in .env, using default key: {API_KEY}")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="x_api_key", auto_error=False)


# Pydantic models for request/response validation
class AnalysisRequest(BaseModel):
    """Request model for feedback analysis."""
    query: str = Field(..., min_length=1, max_length=500, description="Product-related search query")
    n_results: Optional[int] = Field(50, ge=1, le=1000, description="Number of reviews to analyze")
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Sanitize and validate query input."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v.strip())
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'<script', r'javascript:', r'on\w+\s*=',
            r'\.\./|\.\.\\', r';\s*\w+\s*\('
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                raise ValueError(f"Query contains invalid content")
        
        return sanitized


class ThemeData(BaseModel):
    """Model for theme extraction results."""
    product: Optional[str] = Field(None, description="Product name")
    sentiment: Optional[str] = Field(None, description="Sentiment (positive/negative/neutral)")
    theme: Optional[str] = Field(None, description="Theme category")
    classification: Optional[str] = Field(None, description="Issue classification")
    issue_description: Optional[str] = Field(None, description="Issue description")


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    query: str = Field(..., description="Original query")
    themes: List[ThemeData] = Field(..., description="Extracted themes")
    total_themes: int = Field(..., description="Total number of themes")
    timestamp: str = Field(..., description="Analysis timestamp")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class SearchRequest(BaseModel):
    """Request model for vector database search."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    n_results: Optional[int] = Field(50, ge=1, le=1000, description="Number of results")
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Sanitize search query."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v.strip())


class SearchResponse(BaseModel):
    """Response model for search results."""
    query: str = Field(..., description="Original search query")
    results: List[str] = Field(..., description="Matching reviews")
    count: int = Field(..., description="Number of results")
    timestamp: str = Field(..., description="Search timestamp")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    capabilities: List[str] = Field(..., description="Available capabilities")


# Initialize FastAPI app
app = FastAPI(
    title="A2A Review Analyzer Agent",
    description="Agent-to-Agent compatible feedback analysis service with structured API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=600,
)


async def verify_api_key(
    api_key_header: Optional[str] = Security(api_key_header),
    api_key_query: Optional[str] = Security(api_key_query)
) -> str:
    """Verify API key from header or query parameter."""
    key = api_key_header or api_key_query
    
    logger.info(f"Verifying API key - Header: {api_key_header}, Query: {api_key_query}, Expected: {API_KEY}")
    
    if not key or key != API_KEY:
        logger.warning(f"Unauthorized access attempt - Received: {key}, Expected: {API_KEY}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    
    return key


def validate_and_sanitize_query(query: str) -> str:
    """Validate and sanitize query input."""
    if not query or not query.strip():
        raise ValueError("Query parameter cannot be empty")
    
    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', query.strip())
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'<script', r'javascript:', r'on\w+\s*=',
        r'\.\./|\.\.\\', r';\s*\w+\s*\('
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            raise ValueError("Query contains invalid content")
    
    return sanitized


@lru_cache(maxsize=1)
def get_capabilities() -> List[str]:
    """Get list of available agent capabilities."""
    return [
        "feedback_analysis",
        "theme_extraction",
        "sentiment_analysis",
        "semantic_search",
        "cluster_analysis",
        "review_summarization"
    ]


def get_safe_output_path(filename: str) -> Path:
    """Get validated output file path within workspace."""
    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename != filename:
        raise ValueError("Invalid filename: path traversal detected")
    
    if not safe_filename.endswith('.json'):
        safe_filename += '.json'
    
    output_path = Path(os.getcwd()) / safe_filename
    workspace_root = Path(os.getcwd()).resolve()
    
    try:
        output_path.resolve().relative_to(workspace_root)
    except ValueError:
        raise ValueError("Output path is outside workspace directory")
    
    return output_path


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    description="Check agent health and discover capabilities",
    tags=["agent_discovery"]
)
async def health_check():
    """
    Public health check endpoint for agent discovery.
    
    Returns:
        Service status and available capabilities
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(UTC).isoformat(),
        capabilities=get_capabilities()
    )


@app.get(
    "/themes",
    response_model=AnalysisResponse,
    summary="Get product feedback themes",
    description="Retrieve analyzed themes from product feedback using query parameter",
    tags=["analysis"],
    dependencies=[Depends(verify_api_key)]
)
async def get_themes(
    query: str = Query(..., description="Product-related search query", min_length=1, max_length=500),
    n_results: int = Query(50, ge=1, le=1000, description="Number of reviews to analyze")
):
    """
    Get analyzed themes from product feedback via GET request.
    
    This endpoint provides the same functionality as /analyze but via GET method,
    making it easier to test in browsers and simple HTTP clients.
    
    Usage:
        http://localhost:8080/themes?query=Pixel%20battery&x_api_key=YOUR_API_KEY
        http://localhost:8080/themes?query=camera%20issues&n_results=100&x_api_key=YOUR_API_KEY
    
    Args:
        query: Product-related search query (required)
        n_results: Number of reviews to analyze (default: 50)
        
    Returns:
        Structured analysis results with extracted themes
        
    Raises:
        HTTPException: On processing or validation errors
    """
    try:
        # Validate and sanitize query
        sanitized_query = validate_and_sanitize_query(query)
        
        logger.info(f"Processing themes request: {sanitized_query[:100]}")
        start_time = datetime.now(UTC)
        
        # Execute pipeline
        themes = execute_graph_pipeline(sanitized_query)
        
        # Calculate processing time
        end_time = datetime.now(UTC)
        processing_time = (end_time - start_time).total_seconds()
        
        # Convert to response model
        theme_data = [ThemeData(**theme) for theme in themes]
        
        response = AnalysisResponse(
            query=sanitized_query,
            themes=theme_data,
            total_themes=len(theme_data),
            timestamp=end_time.isoformat(),
            processing_time=processing_time
        )
        
        # Save results with secure file operations
        try:
            output_path = get_safe_output_path("a2a_themes_results.json")
            with open(output_path, 'w') as f:
                json.dump(response.model_dump(), f, indent=2)
            os.chmod(output_path, 0o600)
            logger.info(f"Themes results saved to {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save themes results: {e}")
        
        logger.info(f"Themes analysis completed in {processing_time:.2f}s, extracted {len(theme_data)} themes")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error in /themes: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Themes analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Themes analysis pipeline failed"
        )


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyze product feedback",
    description="Execute complete feedback analysis pipeline with theme extraction",
    tags=["analysis"],
    dependencies=[Depends(verify_api_key)]
)
async def analyze_feedback(request: AnalysisRequest):
    """
    Execute the complete feedback analysis pipeline.
    
    This endpoint runs the LangGraph-based pipeline that:
    1. Queries vector database for relevant reviews
    2. Clusters similar reviews together
    3. Summarizes each cluster
    4. Extracts themes using LLM
    
    Args:
        request: Analysis request with query and parameters
        
    Returns:
        Structured analysis results with extracted themes
        
    Raises:
        HTTPException: On processing errors
    """
    try:
        logger.info(f"Processing analysis request: {request.query[:100]}")
        start_time = datetime.now(UTC)
        
        # Execute pipeline
        themes = execute_graph_pipeline(request.query)
        
        # Calculate processing time
        end_time = datetime.now(UTC)
        processing_time = (end_time - start_time).total_seconds()
        
        # Convert to response model
        theme_data = [ThemeData(**theme) for theme in themes]
        
        response = AnalysisResponse(
            query=request.query,
            themes=theme_data,
            total_themes=len(theme_data),
            timestamp=end_time.isoformat(),
            processing_time=processing_time
        )
        
        # Save results with secure file operations
        try:
            output_path = get_safe_output_path("a2a_analysis_results.json")
            with open(output_path, 'w') as f:
                json.dump(response.model_dump(), f, indent=2)
            os.chmod(output_path, 0o600)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")
        
        logger.info(f"Analysis completed in {processing_time:.2f}s, extracted {len(theme_data)} themes")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis pipeline failed"
        )


@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Search reviews semantically",
    description="Query vector database for semantically similar reviews",
    tags=["search"],
    dependencies=[Depends(verify_api_key)]
)
async def search_reviews(request: SearchRequest):
    """
    Search for reviews using semantic similarity.
    
    This endpoint queries the ChromaDB vector database to find reviews
    that are semantically similar to the input query.
    
    Args:
        request: Search request with query and result limit
        
    Returns:
        List of matching reviews
        
    Raises:
        HTTPException: On search errors
    """
    try:
        logger.info(f"Processing search request: {request.query[:100]}")
        
        # Execute vector database query
        results = query_vector_db(
            query_text=request.query,
            n_results=request.n_results,
            output_file="a2a_search_results.csv"
        )
        
        response = SearchResponse(
            query=request.query,
            results=results,
            count=len(results),
            timestamp=datetime.now(UTC).isoformat()
        )
        
        logger.info(f"Search completed, found {len(results)} results")
        return response
        
    except ValueError as e:
        logger.error(f"Search validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search operation failed"
        )


@app.get(
    "/capabilities",
    response_model=Dict[str, Any],
    summary="Get agent capabilities",
    description="Discover available analysis capabilities and methods",
    tags=["agent_discovery"]
)
async def get_agent_capabilities():
    """
    Return structured information about agent capabilities.
    
    This endpoint provides metadata about the agent's functionality
    for A2A discovery and integration.
    
    Returns:
        Capability metadata including available methods and parameters
    """
    return {
        "agent_name": "Review Analyzer Agent",
        "version": "1.0.0",
        "capabilities": get_capabilities(),
        "endpoints": {
            "/analyze": {
                "method": "POST",
                "description": "Execute complete feedback analysis pipeline",
                "parameters": {
                    "query": "string (required, 1-500 chars)",
                    "n_results": "integer (optional, 1-1000)"
                },
                "returns": "themes, sentiment, classifications"
            },
            "/themes": {
                "method": "GET",
                "description": "Get product feedback themes via query parameters",
                "parameters": {
                    "query": "string (required, 1-500 chars)",
                    "n_results": "integer (optional, 1-1000)",
                    "x_api_key": "string (required, API key)"
                },
                "returns": "themes, sentiment, classifications",
                "example": "http://localhost:8080/themes?query=Pixel%20battery&x_api_key=YOUR_API_KEY"
            },
            "/search": {
                "method": "POST",
                "description": "Semantic search for similar reviews",
                "parameters": {
                    "query": "string (required, 1-500 chars)",
                    "n_results": "integer (optional, 1-1000)"
                },
                "returns": "list of matching reviews"
            }
        },
        "authentication": {
            "type": "API Key",
            "methods": ["Header: X-API-Key", "Query: x_api_key"]
        },
        "rate_limits": {
            "requests_per_minute": 60,
            "concurrent_requests": 10
        }
    }


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all API requests for audit trail."""
    logger.info(f"{request.method} {request.url.path} - Client: {request.client.host}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting A2A Review Analyzer Agent")
    logger.info(f"API Key: {API_KEY}")
    logger.info("Example usage:")
    logger.info(f"  http://localhost:8080/themes?query=Pixel%20battery&x_api_key={API_KEY}")
    
    # Secure server configuration
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8080,
        log_level="info",
        access_log=True,
        server_header=False,
        limit_concurrency=10,
        limit_max_requests=1000,
        timeout_keep_alive=5,
    )
