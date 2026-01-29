"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import re


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


class ClusterResponse(BaseModel):
    """Response model for clustering results."""
    clusters: Dict[int, List[str]] = Field(..., description="Clustered reviews")
    count: int = Field(..., description="Number of clusters")
    time_taken: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Timestamp")
