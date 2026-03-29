import re
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def clean_financial_text(text: str) -> str:
    """
    Clean and normalize financial text content.

    Args:
        text: Raw financial text

    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""

    # Remove HTML tags if present
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Fix common encoding issues
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')

    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)

    return text.strip()


class FinancialMetrics:
    """
    Utility class for extracting and comparing financial metrics from documents.
    """

    # Regex patterns for common financial values
    CURRENCY_PATTERN = re.compile(
        r"\$\s*(\d+(?:\.\d+)?)\s*(billion|million|thousand|B|M|K)?",
        re.IGNORECASE,
    )
    PERCENTAGE_PATTERN = re.compile(r"(\-?\d+(?:\.\d+)?)\s*%")
    REVENUE_PATTERN = re.compile(
        r"revenue[s]?\s+(?:of|was|were|totaled?|reached?|grew|increased?|declined?)?\s*"
        r"\$?\s*(\d+(?:\.\d+)?)\s*(billion|million|thousand|B|M|K)?",
        re.IGNORECASE,
    )
    EPS_PATTERN = re.compile(
        r"(?:EPS|earnings per share)[^\d]*(\$?\s*\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    YOY_PATTERN = re.compile(
        r"(\-?\d+(?:\.\d+)?)\s*%\s*(?:year[- ]over[- ]year|YoY|YOY|y/y)",
        re.IGNORECASE,
    )

    UNIT_MULTIPLIERS: dict = {
        "billion": 1e9,
        "million": 1e6,
        "thousand": 1e3,
        "b": 1e9,
        "m": 1e6,
        "k": 1e3,
    }

    def extract_metrics(self, documents: Any) -> Dict:
        """
        Extract financial metrics from collected documents.

        Args:
            documents: Dictionary of collected documents or list of documents

        Returns:
            Dictionary containing extracted metrics
        """
        metrics: Dict = {
            "revenue": None,
            "eps": None,
            "yoy_growth": None,
            "margins": [],
            "key_figures": [],
        }

        try:
            # Handle different document formats
            texts: List[str] = []
            if isinstance(documents, dict):
                for key, doc in documents.items():
                    if isinstance(doc, dict) and "content" in doc:
                        texts.append(str(doc["content"]))
                    elif isinstance(doc, list):
                        for item in doc:
                            if isinstance(item, dict) and "content" in item:
                                texts.append(str(item["content"]))
            elif isinstance(documents, list):
                for doc in documents:
                    if isinstance(doc, dict) and "content" in doc:
                        texts.append(str(doc["content"]))

            combined_text = " ".join(texts)

            # Extract revenue
            revenue_match = self.REVENUE_PATTERN.search(combined_text)
            if revenue_match:
                value = float(revenue_match.group(1))
                unit = (revenue_match.group(2) or "").lower()
                multiplier = self.UNIT_MULTIPLIERS.get(unit, 1)
                metrics["revenue"] = {"value": value * multiplier, "raw": revenue_match.group(0).strip()}

            # Extract EPS
            eps_match = self.EPS_PATTERN.search(combined_text)
            if eps_match:
                metrics["eps"] = eps_match.group(1).strip()

            # Extract YoY growth
            yoy_matches = self.YOY_PATTERN.findall(combined_text)
            if yoy_matches:
                metrics["yoy_growth"] = [float(v) for v in yoy_matches]

            # Extract percentages (margins etc.)
            pct_matches = self.PERCENTAGE_PATTERN.findall(combined_text)
            if pct_matches:
                metrics["margins"] = [float(v) for v in pct_matches[:5]]

            # Extract key currency figures
            currency_matches = self.CURRENCY_PATTERN.findall(combined_text)
            if currency_matches:
                metrics["key_figures"] = [
                    {"value": m[0], "unit": m[1]} for m in currency_matches[:10]
                ]

        except Exception as e:
            logger.error(f"Error extracting metrics: {str(e)}")

        return metrics

    def compare_quarters(self, quarterly_results: Dict) -> Dict:
        """
        Compare financial metrics across multiple quarters.

        Args:
            quarterly_results: Dictionary mapping quarter → analysis results

        Returns:
            Dictionary containing comparative analysis
        """
        comparison: Dict = {
            "quarters": list(quarterly_results.keys()),
            "metrics_comparison": {},
            "trend": "stable",
        }

        try:
            revenues = {}
            for quarter, result in quarterly_results.items():
                metrics = result.get("metrics", {})
                if isinstance(metrics, dict) and metrics.get("revenue"):
                    rev = metrics["revenue"]
                    if isinstance(rev, dict):
                        revenues[quarter] = rev.get("value")
                    elif isinstance(rev, (int, float)):
                        revenues[quarter] = rev

            if revenues:
                comparison["metrics_comparison"]["revenue"] = revenues
                values = list(revenues.values())
                if len(values) >= 2 and values[-1] and values[0]:
                    pct_change = ((values[-1] - values[0]) / values[0]) * 100
                    comparison["trend"] = "growing" if pct_change > 0 else "declining"
                    comparison["metrics_comparison"]["revenue_trend_pct"] = round(pct_change, 2)

        except Exception as e:
            logger.error(f"Error comparing quarters: {str(e)}")

        return comparison

    def analyze_trends(self, quarterly_results: Dict) -> Dict:
        """
        Analyze financial trends across quarters.

        Args:
            quarterly_results: Dictionary mapping quarter → analysis results

        Returns:
            Dictionary containing trend analysis
        """
        trends: Dict = {
            "quarters_analyzed": len(quarterly_results),
            "overall_direction": "stable",
            "insights": [],
        }

        try:
            comparison = self.compare_quarters(quarterly_results)
            trends["overall_direction"] = comparison.get("trend", "stable")

            revenue_data = comparison.get("metrics_comparison", {}).get("revenue", {})
            if revenue_data:
                trends["insights"].append(
                    f"Revenue data tracked across {len(revenue_data)} quarters."
                )

            trend_pct = comparison.get("metrics_comparison", {}).get("revenue_trend_pct")
            if trend_pct is not None:
                direction = "increased" if trend_pct > 0 else "decreased"
                trends["insights"].append(
                    f"Revenue {direction} by {abs(trend_pct):.1f}% over the analyzed period."
                )

        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")

        return trends
