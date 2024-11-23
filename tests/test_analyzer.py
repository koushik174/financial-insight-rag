import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.core.analyzer import FinancialAnalyzer

@pytest.fixture
def mock_collector():
    return Mock()

@pytest.fixture
def analyzer(mock_collector):
    analyzer = FinancialAnalyzer()
    analyzer.collector = mock_collector
    return analyzer

class TestFinancialAnalyzer:
    """Test suite for FinancialAnalyzer class."""
    
    async def test_analyze_earnings(self, analyzer, mock_collector):
        # Mock data
        mock_documents = {
            "earnings_call": {
                "content": "Q3 2023 earnings call transcript...",
                "metadata": {"type": "earnings_call"}
            }
        }
        mock_collector.get_earnings_documents.return_value = mock_documents
        
        # Execute
        result = await analyzer.analyze_earnings("AAPL", "Q3_2023")
        
        # Assert
        assert result["company"] == "AAPL"
        assert result["quarter"] == "Q3_2023"
        assert "metrics" in result
        assert "analysis" in result
        
    async def test_analyze_earnings_error(self, analyzer, mock_collector):
        # Mock error
        mock_collector.get_earnings_documents.side_effect = Exception("API Error")
        
        # Assert error handling
        with pytest.raises(Exception):
            await analyzer.analyze_earnings("AAPL", "Q3_2023")
            
    async def test_get_comparative_analysis(self, analyzer):
        quarters = ["Q1_2023", "Q2_2023", "Q3_2023"]
        result = await analyzer.get_comparative_analysis("AAPL", quarters)
        
        assert "comparative_analysis" in result
        assert "trend_analysis" in result
        assert result["company"] == "AAPL"

    @pytest.mark.parametrize("company,quarter", [
        ("AAPL", "Q3_2023"),
        ("GOOGL", "Q2_2023"),
        ("MSFT", "Q4_2023")
    ])
    async def test_multiple_companies(self, analyzer, company, quarter):
        result = await analyzer.analyze_earnings(company, quarter)
        assert result["company"] == company
        assert result["quarter"] == quarter
