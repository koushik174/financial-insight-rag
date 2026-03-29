from typing import Dict, List, Optional
import logging
from datetime import datetime, timezone

from ..models.rag_model import FinancialRAGModel
from ..utils.financial import FinancialMetrics
from ..data.collector import DataCollector

logger = logging.getLogger(__name__)


class FinancialAnalyzer:
    """
    Core analyzer class for processing financial documents and generating insights.
    """

    def __init__(self, model_config: Optional[Dict] = None):
        self.model = FinancialRAGModel()
        self.metrics = FinancialMetrics()
        self.collector = DataCollector(config=model_config or {})

    async def analyze_earnings(
        self,
        company: str,
        quarter: str,
    ) -> Dict:
        """
        Analyze earnings call transcripts and related financial documents.

        Args:
            company: Company ticker or identifier
            quarter: Quarter identifier (e.g., 'Q3_2023')

        Returns:
            Dict containing analysis results including metrics, sentiment, and insights
        """
        try:
            documents = await self.collector.get_earnings_documents(company, quarter)
            analysis_results = await self._analyze_documents(documents)
            financial_metrics = self.metrics.extract_metrics(documents)

            return {
                "company": company,
                "quarter": quarter,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": financial_metrics,
                "analysis": analysis_results,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Analysis failed for {company} {quarter}: {str(e)}")
            raise

    async def _analyze_documents(self, documents: Dict) -> Dict:
        """
        Perform detailed analysis on documents using RAG model.

        Args:
            documents: Dictionary of collected documents

        Returns:
            Dict containing various analysis results
        """
        # Flatten documents dict into a list for the RAG model
        doc_list: List[Dict] = []
        for key, value in documents.items():
            if key == "metadata":
                continue
            if isinstance(value, dict) and "content" in value:
                doc_list.append(value)
            elif isinstance(value, list):
                doc_list.extend([d for d in value if isinstance(d, dict) and "content" in d])

        return await self.model.analyze(doc_list)

    async def get_comparative_analysis(
        self,
        company: str,
        quarters: List[str],
    ) -> Dict:
        """
        Perform comparative analysis across multiple quarters.

        Args:
            company: Company identifier
            quarters: List of quarters to compare

        Returns:
            Dict containing comparative analysis results
        """
        results = {}
        for quarter in quarters:
            results[quarter] = await self.analyze_earnings(company, quarter)

        return {
            "company": company,
            "comparative_analysis": self.metrics.compare_quarters(results),
            "trend_analysis": self.metrics.analyze_trends(results),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def get_key_insights(
        self,
        company: str,
        quarter: str,
    ) -> List[Dict]:
        """
        Extract key insights from earnings analysis.

        Args:
            company: Company identifier
            quarter: Quarter identifier

        Returns:
            List of key insights with supporting evidence
        """
        analysis = await self.analyze_earnings(company, quarter)
        return self.model.extract_insights(analysis)

    async def process_documents(self, documents: List[Dict]) -> None:
        """
        Process and index a list of financial documents asynchronously.

        Args:
            documents: List of document dicts with 'content', 'metadata', and 'type' fields
        """
        try:
            for doc in documents:
                content = doc.get("content", "")
                if content:
                    logger.info(
                        f"Processing document of type: {doc.get('type', 'unknown')}"
                    )
            logger.info(f"Finished processing {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error in process_documents: {str(e)}")
            raise
