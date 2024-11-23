from typing import List, Dict, Union, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProcessedFinancialData:
    """Data class for processed financial information."""
    raw_text: str
    cleaned_text: str
    metrics: Dict
    metadata: Dict
    processed_at: datetime

class FinancialPreprocessor:
    """
    Handles preprocessing of financial documents and data,
    including text cleaning, metric standardization, and feature extraction.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._load_preprocessing_rules()
        
    def process_financial_text(self,
                             text: str,
                             document_type: str) -> ProcessedFinancialData:
        """
        Process financial text documents with type-specific handling.
        
        Args:
            text: Raw text content
            document_type: Type of financial document (e.g., 'earnings_call', 'sec_filing')
            
        Returns:
            ProcessedFinancialData object containing cleaned text and extracted information
        """
        # Basic cleaning
        cleaned_text = self._clean_text(text)
        
        # Document-specific processing
        if document_type == 'earnings_call':
            processed = self._process_earnings_call(cleaned_text)
        elif document_type == 'sec_filing':
            processed = self._process_sec_filing(cleaned_text)
        else:
            processed = self._process_generic_financial(cleaned_text)
            
        # Extract metrics
        metrics = self._extract_financial_metrics(processed)
        
        return ProcessedFinancialData(
            raw_text=text,
            cleaned_text=processed,
            metrics=metrics,
            metadata={
                "document_type": document_type,
                "preprocessing_steps": self._get_preprocessing_steps()
            },
            processed_at=datetime.utcnow()
        )
        
    def _clean_text(self, text: str) -> str:
        """Perform basic text cleaning."""
        # Remove special characters
        text = re.sub(r'[^\w\s.,?!-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common financial notation
        text = self._standardize_financial_notation(text)
        
        return text.strip()
        
    def _process_earnings_call(self, text: str) -> str:
        """Process earnings call specific text."""
        # Remove speaker annotations
        text = re.sub(r'\[.*?\]', '', text)
        
        # Separate Q&A section
        text = self._separate_qa_section(text)
        
        # Standardize speaker transitions
        text = self._standardize_speaker_formats(text)
        
        return text
        
    def _process_sec_filing(self, text: str) -> str:
        """Process SEC filing specific text."""
        # Remove headers and footers
        text = self._remove_sec_boilerplate(text)
        
        # Standardize section headers
        text = self._standardize_section_headers(text)
        
        # Extract relevant sections
        text = self._extract_relevant_sections(text)
        
        return text
        
    def _process_generic_financial(self, text: str) -> str:
        """Process generic financial document text."""
        # Basic financial text processing
        text = self._standardize_financial_notation(text)
        text = self._clean_numbers_and_dates(text)
        return text
        
    def _extract_financial_metrics(self, text: str) -> Dict:
        """Extract key financial metrics from text."""
        metrics = {
            "currency_values": self._extract_currency_values(text),
            "percentages": self._extract_percentages(text),
            "dates": self._extract_dates(text),
            "key_metrics": self._extract_key_performance_indicators(text)
        }
        return metrics
        
    def _standardize_financial_notation(self, text: str) -> str:
        """Standardize financial notation and abbreviations."""
        notation_rules = {
            r'\$(\d+)K\b': r'$\1,000',
            r'\$(\d+)M\b': r'$\1,000,000',
            r'\$(\d+)B\b': r'$\1,000,000,000',
            r'(\d+)%': r'\1 percent',
            # Add more rules as needed
        }
        
        for pattern, replacement in notation_rules.items():
            text = re.sub(pattern, replacement, text)
            
        return text
        
    def _clean_numbers_and_dates(self, text: str) -> str:
        """Standardize number and date formats."""
        # Standardize date formats
        text = self._standardize_dates(text)
        
        # Standardize number formats
        text = self._standardize_numbers(text)
        
        return text
        
    def _standardize_dates(self, text: str) -> str:
        """Convert dates to standard format."""
        # Date standardization logic
        return text
        
    def _standardize_numbers(self, text: str) -> str:
        """Convert numbers to standard format."""
        # Number standardization logic
        return text
        
    def _load_preprocessing_rules(self):
        """Load preprocessing rules from configuration."""
        self.rules = self.config.get('preprocessing_rules', {})
        
    def _get_preprocessing_steps(self) -> List[str]:
        """Get list of preprocessing steps applied."""
        return [
            "basic_cleaning",
            "notation_standardization",
            "number_standardization",
            "metric_extraction"
        ]
        
    def process_batch(self,
                     documents: List[Dict]) -> List[ProcessedFinancialData]:
        """
        Process multiple financial documents in batch.
        
        Args:
            documents: List of document dictionaries with text and metadata
            
        Returns:
            List of ProcessedFinancialData objects
        """
        processed_docs = []
        for doc in documents:
            try:
                processed = self.process_financial_text(
                    text=doc['text'],
                    document_type=doc.get('type', 'generic')
                )
                processed_docs.append(processed)
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                continue
                
        return processed_docs
