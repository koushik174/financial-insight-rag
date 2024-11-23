from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

class AnalysisType(str, Enum):
    """Enumeration for different types of analysis."""
    FULL = "full"
    METRICS = "metrics"
    SENTIMENT = "sentiment"
    TRENDS = "trends"

class AnalysisRequest(BaseModel):
    """Request model for earnings analysis."""
    company: str = Field(..., description="Company ticker symbol")
    quarter: str = Field(..., description="Quarter identifier (e.g., Q3_2023)")
    analysis_type: AnalysisType = Field(
        default=AnalysisType.FULL,
        description="Type of analysis to perform"
    )
    
    @validator('company')
    def validate_company(cls, v):
        if not v.isalpha():
            raise ValueError("Company ticker must contain only letters")
        return v.upper()
    
    @validator('quarter')
    def validate_quarter(cls, v):
        if not v.startswith('Q'):
            raise ValueError("Quarter must start with Q (e.g., Q3_2023)")
        return v

class AnalysisResponse(BaseModel):
    """Response model for earnings analysis."""
    company: str
    quarter: str
    analysis: Dict = Field(..., description="Analysis results")
    timestamp: str
    
    class Config:
        schema_extra = {
            "example": {
                "company": "AAPL",
                "quarter": "Q3_2023",
                "analysis": {
                    "metrics": {
                        "revenue_growth": "+12.5%",
                        "profit_margin": "28.3%"
                    },
                    "sentiment": {
                        "overall": "positive",
                        "confidence": 0.85
                    }
                },
                "timestamp": "2024-03-23T10:30:00Z"
            }
        }

class ComparisonRequest(BaseModel):
    """Request model for quarter comparison."""
    company: str = Field(..., description="Company ticker symbol")
    quarters: List[str] = Field(..., description="List of quarters to compare")
    metrics: Optional[List[str]] = Field(
        default=None,
        description="Specific metrics to compare"
    )

class ComparisonResponse(BaseModel):
    """Response model for quarter comparison."""
    company: str
    quarters: List[str]
    comparison: Dict
    timestamp: str

class Document(BaseModel):
    """Model for financial document."""
    content: str
    metadata: Dict
    type: str = Field(..., description="Document type (e.g., earnings_call, sec_filing)")

class DocumentRequest(BaseModel):
    """Request model for document processing."""
    documents: List[Document]
    analysis_type: Optional[AnalysisType] = Field(
        default=AnalysisType.FULL,
        description="Type of analysis to perform"
    )

class DocumentResponse(BaseModel):
    """Response model for document processing."""
    status: str
    document_count: int
    timestamp: str

class InsightRequest(BaseModel):
    """Request model for insight extraction."""
    query: str = Field(..., description="Question or query about the documents")
    documents: List[Document]
    max_results: Optional[int] = Field(
        default=5,
        description="Maximum number of insights to return"
    )

class InsightResponse(BaseModel):
    """Response model for insight extraction."""
    query: str
    insights: str
    context_used: List[Dict]
    timestamp: str

class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    status_code: int
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class Config:
        schema_extra = {
            "example": {
                "error": "Invalid company ticker symbol",
                "status_code": 400,
                "timestamp": "2024-03-23T10:30:00Z"
            }
        }

class HealthCheck(BaseModel):
    """Model for health check response."""
    status: str
    timestamp: str
    version: str
