from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import logging
from datetime import datetime

from .schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ComparisonRequest,
    ComparisonResponse,
    DocumentRequest,
    DocumentResponse,
    ErrorResponse,
    InsightRequest,
    InsightResponse
)
from ..core.analyzer import FinancialAnalyzer
from ..models.rag_model import FinancialRAGModel

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Insight RAG API",
    description="API for analyzing financial documents using RAG technology",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
analyzer = FinancialAnalyzer()
rag_model = FinancialRAGModel()

@app.post("/api/v1/analyze",
          response_model=AnalysisResponse,
          responses={400: {"model": ErrorResponse},
                    500: {"model": ErrorResponse}})
async def analyze_earnings(request: AnalysisRequest):
    """
    Analyze earnings call and financial documents for a company.
    """
    try:
        logger.info(f"Processing analysis request for {request.company} {request.quarter}")
        
        analysis_result = await analyzer.analyze_earnings(
            company=request.company,
            quarter=request.quarter
        )
        
        return AnalysisResponse(
            company=request.company,
            quarter=request.quarter,
            analysis=analysis_result,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/v1/compare",
          response_model=ComparisonResponse,
          responses={400: {"model": ErrorResponse},
                    500: {"model": ErrorResponse}})
async def compare_quarters(request: ComparisonRequest):
    """
    Compare financial performance across multiple quarters.
    """
    try:
        comparison_result = await analyzer.get_comparative_analysis(
            company=request.company,
            quarters=request.quarters
        )
        
        return ComparisonResponse(
            company=request.company,
            quarters=request.quarters,
            comparison=comparison_result,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/v1/insights",
          response_model=InsightResponse,
          responses={400: {"model": ErrorResponse},
                    500: {"model": ErrorResponse}})
async def get_insights(request: InsightRequest):
    """
    Extract key insights from financial documents using RAG model.
    """
    try:
        insights = await rag_model.process_query(
            query=request.query,
            context_docs=request.documents
        )
        
        return InsightResponse(
            query=request.query,
            insights=insights.response,
            context_used=insights.retrieved_contexts,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Insight extraction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/v1/documents",
          response_model=DocumentResponse,
          responses={400: {"model": ErrorResponse},
                    500: {"model": ErrorResponse}})
async def process_documents(
    request: DocumentRequest,
    background_tasks: BackgroundTasks
):
    """
    Process and analyze financial documents asynchronously.
    """
    try:
        # Start async processing
        background_tasks.add_task(
            analyzer.process_documents,
            documents=request.documents
        )
        
        return DocumentResponse(
            status="processing",
            document_count=len(request.documents),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/v1/health")
async def health_check():
    """
    Check API health status.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return ErrorResponse(
        error=str(exc.detail),
        status_code=exc.status_code
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return ErrorResponse(
        error="Internal server error",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )
