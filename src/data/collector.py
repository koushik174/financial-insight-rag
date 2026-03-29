import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timezone

from ..core.processor import DocumentProcessor
from ..utils.financial import clean_financial_text

logger = logging.getLogger(__name__)

# Optional dependencies – imported lazily so the app starts even without them
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    logger.warning("yfinance not installed; financial metrics collection will be skipped.")

try:
    from sec_api import QueryApi as _SecQueryApi
    _SEC_AVAILABLE = True
except ImportError:
    _SEC_AVAILABLE = False
    logger.warning("sec_api not installed; SEC filing collection will be skipped.")

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


class DataCollector:
    """
    Handles collection of financial data from various sources including
    earnings calls, SEC filings, and financial metrics.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.processor = DocumentProcessor()

        self.sec_api = None
        if _SEC_AVAILABLE and self.config.get("SEC_API_KEY"):
            self.sec_api = _SecQueryApi(api_key=self.config["SEC_API_KEY"])

    async def get_earnings_documents(
        self,
        company: str,
        quarter: str,
    ) -> Dict:
        """
        Collect all relevant documents for earnings analysis.

        Args:
            company: Company ticker symbol
            quarter: Quarter identifier (e.g., 'Q3_2023')

        Returns:
            Dictionary containing all collected documents and metadata
        """
        try:
            earnings_call, sec_filings, financials = await asyncio.gather(
                self.get_earnings_transcript(company, quarter),
                self.get_sec_filings(company, quarter),
                self.get_financial_metrics(company, quarter),
                return_exceptions=True,
            )

            # Replace exceptions with empty defaults so partial data still works
            if isinstance(earnings_call, Exception):
                logger.warning(f"Earnings transcript unavailable: {earnings_call}")
                earnings_call = {"content": "", "source": "unavailable", "metadata": {}}
            if isinstance(sec_filings, Exception):
                logger.warning(f"SEC filings unavailable: {sec_filings}")
                sec_filings = []
            if isinstance(financials, Exception):
                logger.warning(f"Financial metrics unavailable: {financials}")
                financials = {"metrics": {}, "metadata": {}}

            return {
                "earnings_call": earnings_call,
                "sec_filings": sec_filings,
                "financial_metrics": financials,
                "metadata": {
                    "company": company,
                    "quarter": quarter,
                    "collection_time": datetime.now(timezone.utc).isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"Error collecting documents for {company} {quarter}: {str(e)}")
            raise

    async def get_earnings_transcript(
        self,
        company: str,
        quarter: str,
    ) -> Dict:
        """
        Fetch earnings call transcript from various sources.

        Args:
            company: Company ticker
            quarter: Quarter identifier

        Returns:
            Dictionary containing transcript and metadata
        """
        sources = [
            self._fetch_seeking_alpha_transcript,
            self._fetch_motley_fool_transcript,
            self._fetch_company_ir_transcript,
        ]

        for source in sources:
            try:
                transcript = await source(company, quarter)
                if transcript:
                    return {
                        "content": clean_financial_text(transcript),
                        "source": source.__name__,
                        "metadata": {
                            "company": company,
                            "quarter": quarter,
                            "type": "earnings_transcript",
                        },
                    }
            except Exception as e:
                logger.warning(
                    f"Failed to fetch transcript from {source.__name__}: {str(e)}"
                )
                continue

        logger.warning(f"No transcript found for {company} {quarter}; returning empty.")
        return {
            "content": "",
            "source": "none",
            "metadata": {
                "company": company,
                "quarter": quarter,
                "type": "earnings_transcript",
            },
        }

    async def get_sec_filings(
        self,
        company: str,
        quarter: str,
    ) -> List[Dict]:
        """
        Fetch relevant SEC filings for the specified quarter.

        Args:
            company: Company ticker
            quarter: Quarter identifier

        Returns:
            List of filing documents with metadata
        """
        if self.sec_api is None:
            logger.info("SEC API not configured; skipping SEC filings.")
            return []

        try:
            date_range = self._quarter_to_date_range(quarter)

            query = {
                "query": {
                    "query_string": {
                        "query": (
                            f"ticker:{company} AND "
                            f"filedAt:[{date_range['start']} TO {date_range['end']}]"
                        )
                    }
                }
            }

            filings = self.sec_api.get_filings(query)

            return [
                {
                    "content": self._extract_filing_content(filing),
                    "metadata": {
                        "company": company,
                        "quarter": quarter,
                        "filing_type": filing.get("type"),
                        "filing_date": filing.get("filedAt"),
                        "type": "sec_filing",
                    },
                }
                for filing in filings.get("filings", [])
            ]

        except Exception as e:
            logger.error(f"Error fetching SEC filings: {str(e)}")
            raise

    async def get_financial_metrics(
        self,
        company: str,
        quarter: str,
    ) -> Dict:
        """
        Collect financial metrics from Yahoo Finance.

        Args:
            company: Company ticker
            quarter: Quarter identifier

        Returns:
            Dictionary containing financial metrics
        """
        if not _YF_AVAILABLE:
            logger.info("yfinance not available; skipping financial metrics.")
            return {
                "metrics": {},
                "metadata": {
                    "company": company,
                    "quarter": quarter,
                    "type": "financial_metrics",
                    "source": "unavailable",
                },
            }

        try:
            stock = yf.Ticker(company)
            financials = stock.quarterly_financials
            quarter_data = self._extract_quarter_data(financials, quarter)

            return {
                "metrics": quarter_data,
                "metadata": {
                    "company": company,
                    "quarter": quarter,
                    "type": "financial_metrics",
                    "source": "yahoo_finance",
                },
            }

        except Exception as e:
            logger.error(f"Error fetching financial metrics: {str(e)}")
            raise

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _fetch_seeking_alpha_transcript(
        self, company: str, quarter: str
    ) -> Optional[str]:
        """Fetch transcript from Seeking Alpha (requires API key / subscription)."""
        # Placeholder – returns None so the next source is tried
        return None

    async def _fetch_motley_fool_transcript(
        self, company: str, quarter: str
    ) -> Optional[str]:
        """Fetch transcript from Motley Fool."""
        # Placeholder – returns None so the next source is tried
        return None

    async def _fetch_company_ir_transcript(
        self, company: str, quarter: str
    ) -> Optional[str]:
        """Fetch transcript from the company IR website."""
        # Placeholder – returns None so the next source is tried
        return None

    def _quarter_to_date_range(self, quarter: str) -> Dict[str, str]:
        """
        Convert a quarter identifier (e.g., 'Q3_2023') to a date range dict.

        Returns:
            {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}
        """
        try:
            parts = quarter.upper().split("_")
            q_num = int(parts[0][1])  # e.g. '3' from 'Q3'
            year = int(parts[1])       # e.g. 2023

            quarter_dates = {
                1: ("01-01", "03-31"),
                2: ("04-01", "06-30"),
                3: ("07-01", "09-30"),
                4: ("10-01", "12-31"),
            }
            start_suffix, end_suffix = quarter_dates.get(q_num, ("01-01", "03-31"))
            return {
                "start": f"{year}-{start_suffix}",
                "end": f"{year}-{end_suffix}",
            }
        except Exception as e:
            logger.error(f"Error parsing quarter '{quarter}': {str(e)}")
            return {"start": "1900-01-01", "end": "1900-12-31"}

    def _extract_filing_content(self, filing: Dict) -> str:
        """Extract and clean filing content from an SEC filing dict."""
        content_fields = ["documentFormatFiles", "linkToFilingDetails", "description"]
        for field in content_fields:
            value = filing.get(field)
            if value and isinstance(value, str):
                return clean_financial_text(value)
            if value and isinstance(value, list) and value:
                return clean_financial_text(str(value[0]))
        # Fall back to a string representation of the whole filing
        return clean_financial_text(str(filing))

    def _extract_quarter_data(self, financials, quarter: str) -> Dict:
        """
        Extract relevant quarter data from a yfinance quarterly_financials DataFrame.

        Args:
            financials: pandas DataFrame returned by yf.Ticker().quarterly_financials
            quarter: Quarter identifier (e.g., 'Q3_2023')

        Returns:
            Dictionary of metric name → value
        """
        try:
            if not _PANDAS_AVAILABLE or financials is None or financials.empty:
                return {}

            date_range = self._quarter_to_date_range(quarter)

            start = pd.Timestamp(date_range["start"])
            end = pd.Timestamp(date_range["end"])

            # quarterly_financials columns are period-end timestamps
            cols_in_range = [
                col
                for col in financials.columns
                if start <= pd.Timestamp(col) <= end
            ]

            if not cols_in_range:
                # Return the most recent column if nothing in range
                if len(financials.columns) > 0:
                    cols_in_range = [financials.columns[0]]
                else:
                    return {}

            col = cols_in_range[0]
            series = financials[col].dropna()
            return {str(k): float(v) for k, v in series.items()}

        except Exception as e:
            logger.error(f"Error extracting quarter data: {str(e)}")
            return {}
