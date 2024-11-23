from typing import Dict, List, Optional
import logging
from datetime import datetime

from ..models.rag_model import RAGModel
from ..utils.financial import FinancialMetrics
from ..data.collector import DataCollector

logger = logging.getLogger(__name__)

class FinancialAnalyzer:
    """
    Core analyzer class for processing financial documents and generating insights.
    """
    def __init__(self, model_config: Optional[Dict] = None):
        self.model = RAGModel(model_config or {})
        self.metrics = FinancialMetrics()
        self.collector = DataCollector()
        
    async def analyze_earnings(self, 
                             company: str, 
                             quarter: str) -> Dict:
        """
        Analyze earnings call transcripts and related financial documents.
        
        Args:
            company: Company ticker or identifier
            quarter: Quarter identifier (e.g., 'Q3_2023')
            
        Returns:
            Dict containing analysis results including metrics, sentiment, and insights
        """
        try:
            # Collect relevant documents
            documents = await self.collector.get_earnings_documents(company, quarter)
            
            # Process and analyze
            analysis_results = await self._analyze_documents(documents)
            
            # Extract metrics
            financial_metrics = self.metrics.extract_metrics(documents)
            
            return {
                "company": company,
                "quarter": quarter,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": financial_metrics,
                "analysis": analysis_results,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Analysis failed for {company} {quarter}: {str(e)}")
            raise
            
    async def _analyze_documents(self, documents: List[Dict]) -> Dict:
        """
        Perform detailed analysis on documents using RAG model.
        
        Args:
            documents: List of document dictionaries containing text and metadata
            
        Returns:
            Dict containing various analysis results
        """
        return await self.model.analyze(documents)
    
    async def get_comparative_analysis(self,
                                    company: str,
                                    quarters: List[str]) -> Dict:
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
            "timestamp": datetime.utcnow().isoformat()
        }

    async def get_key_insights(self,
                             company: str,
                             quarter: str) -> List[Dict]:
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
