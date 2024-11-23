from typing import List, Dict, Optional
import re
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ProcessedDocument:
    """Data class for processed document information."""
    id: str
    content: str
    chunks: List[str]
    metadata: Dict
    processed_at: datetime

class DocumentProcessor:
    """
    Handles document processing and chunking for financial documents.
    """
    def __init__(self, 
                 chunk_size: int = 1000, 
                 overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def process_document(self, 
                        content: str, 
                        metadata: Optional[Dict] = None) -> ProcessedDocument:
        """
        Process a single document: clean, chunk, and prepare for analysis.
        
        Args:
            content: Raw document text
            metadata: Optional metadata about the document
            
        Returns:
            ProcessedDocument containing cleaned and chunked content
        """
        # Clean the text
        cleaned_text = self._clean_text(content)
        
        # Create chunks
        chunks = self._create_chunks(cleaned_text)
        
        return ProcessedDocument(
            id=self._generate_doc_id(metadata),
            content=cleaned_text,
            chunks=chunks,
            metadata=metadata or {},
            processed_at=datetime.utcnow()
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove special characters
        text = re.sub(r'[^\w\s.,?!-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common financial text issues
        text = self._fix_financial_text(text)
        
        return text.strip()
    
    def _create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk with specified size
            chunk = text[start:start + self.chunk_size]
            
            # Adjust chunk to end at sentence boundary
            if len(chunk) < len(text) - start:
                last_period = chunk.rfind('.')
                if last_period != -1:
                    chunk = chunk[:last_period + 1]
            
            chunks.append(chunk)
            start += len(chunk) - self.overlap
            
        return chunks
    
    def _fix_financial_text(self, text: str) -> str:
        """Fix common issues in financial text."""
        # Fix common financial abbreviations
        replacements = {
            'MM': 'Million',
            'B': 'Billion',
            'K': 'Thousand',
            'Q1': 'First Quarter',
            'Q2': 'Second Quarter',
            'Q3': 'Third Quarter',
            'Q4': 'Fourth Quarter'
        }
        
        for old, new in replacements.items():
            text = re.sub(rf'\b{old}\b', new, text)
            
        return text
    
    def _generate_doc_id(self, metadata: Optional[Dict]) -> str:
        """Generate unique document ID based on metadata."""
        if metadata and 'company' in metadata and 'date' in metadata:
            return f"{metadata['company']}_{metadata['date']}_{datetime.utcnow().timestamp()}"
        return f"doc_{datetime.utcnow().timestamp()}"
    
    def batch_process(self, 
                     documents: List[Dict]) -> List[ProcessedDocument]:
        """
        Process multiple documents in batch.
        
        Args:
            documents: List of document dictionaries with content and metadata
            
        Returns:
            List of ProcessedDocument objects
        """
        processed_docs = []
        for doc in documents:
            try:
                processed = self.process_document(
                    content=doc['content'],
                    metadata=doc.get('metadata')
                )
                processed_docs.append(processed)
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                continue
                
        return processed_docs
