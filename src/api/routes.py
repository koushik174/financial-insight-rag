from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from .schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ComparisonRequest,
    ComparisonResponse,
    DocumentRequest,
    DocumentResponse,
    ErrorResponse,
    InsightRequest,
    InsightResponse,
)
from ..core.analyzer import FinancialAnalyzer
from ..models.rag_model import FinancialRAGModel

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Insight RAG API",
    description="API for analyzing financial documents using RAG technology",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory (contains the web UI)
_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# Initialize models lazily (populated on first request to avoid startup crash)
_analyzer: Optional[FinancialAnalyzer] = None
_rag_model: Optional[FinancialRAGModel] = None


def get_analyzer() -> FinancialAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = FinancialAnalyzer()
    return _analyzer


def get_rag_model() -> FinancialRAGModel:
    global _rag_model
    if _rag_model is None:
        _rag_model = FinancialRAGModel()
    return _rag_model


# -----------------------------------------------------------------------
# Web UI – served at the root URL so the RAG is accessible from any browser
# -----------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serve the Financial Insight RAG web interface."""
    static_index = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
    if os.path.isfile(static_index):
        with open(static_index, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Financial Insight RAG API</h1><p>Visit <a href='/docs'>/docs</a> to use the API.</p>")


# -----------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------

@app.post(
    "/api/v1/analyze",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def analyze_earnings(request: AnalysisRequest):
    """Analyze earnings call and financial documents for a company."""
    try:
        logger.info(f"Processing analysis request for {request.company} {request.quarter}")

        analysis_result = await get_analyzer().analyze_earnings(
            company=request.company,
            quarter=request.quarter,
        )

        return AnalysisResponse(
            company=request.company,
            quarter=request.quarter,
            analysis=analysis_result,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post(
    "/api/v1/compare",
    response_model=ComparisonResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def compare_quarters(request: ComparisonRequest):
    """Compare financial performance across multiple quarters."""
    try:
        comparison_result = await get_analyzer().get_comparative_analysis(
            company=request.company,
            quarters=request.quarters,
        )

        return ComparisonResponse(
            company=request.company,
            quarters=request.quarters,
            comparison=comparison_result,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post(
    "/api/v1/insights",
    response_model=InsightResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_insights(request: InsightRequest):
    """Extract key insights from financial documents using the RAG model."""
    try:
        insights = await get_rag_model().process_query(
            query=request.query,
            context_docs=[doc.model_dump() for doc in request.documents],
        )

        return InsightResponse(
            query=request.query,
            insights=insights.response,
            context_used=insights.retrieved_contexts,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.error(f"Insight extraction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post(
    "/api/v1/documents",
    response_model=DocumentResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def process_documents(
    request: DocumentRequest,
    background_tasks: BackgroundTasks,
):
    """Process and analyze financial documents asynchronously."""
    try:
        background_tasks.add_task(
            get_analyzer().process_documents,
            documents=[doc.model_dump() for doc in request.documents],
        )

        return DocumentResponse(
            status="processing",
            document_count=len(request.documents),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/api/v1/health")
async def health_check():
    """Check API health status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


# -----------------------------------------------------------------------
# Error handlers
# -----------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=str(exc.detail),
            status_code=exc.status_code,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ).model_dump(),
    )
