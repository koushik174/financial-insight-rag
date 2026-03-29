"""
Entry point for the Financial Insight RAG application.

Run with:
    uvicorn src.main:app --reload
or:
    python -m src.main
"""

import uvicorn
from src.api.routes import app  # noqa: F401  re-exported for uvicorn

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
