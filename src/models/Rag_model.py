from typing import List, Dict, Optional, Union
import torch
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from .embeddings import FinancialEmbeddingModel

logger = logging.getLogger(__name__)

@dataclass
class RAGOutput:
    """Data class for RAG model outputs."""
    response: str
    retrieved_contexts: List[Dict]
    metadata: Dict
    created_at: datetime

class FinancialRAGModel:
    """
    Retrieval-Augmented Generation model specialized for financial analysis.
    Combines retrieval of relevant financial context with language model generation.
    """
    
    def __init__(self,
                 embedding_model: Optional[FinancialEmbeddingModel] = None,
                 llm_model_name: str = "anthropic/claude-3",
                 device: Optional[str] = None):
        """
        Initialize the RAG model.
        
        Args:
            embedding_model: Model for generating embeddings
            llm_model_name: Name of the language model to use
            device: Device to run the model on
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_model = embedding_model or FinancialEmbeddingModel()
        
        try:
            # Initialize language model and tokenizer
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_name
            ).to(self.device)
            logger.info(f"Initialized RAG model with {llm_model_name}")
        except Exception as e:
            logger.error(f"Error initializing RAG model: {str(e)}")
            raise
            
    async def process_query(self,
                          query: str,
                          context_docs: List[Dict],
                          max_context_length: int = 2048) -> RAGOutput:
        """
        Process a query using the RAG pipeline.
        
        Args:
            query: User query
            context_docs: List of context documents
            max_context_length: Maximum context length for generation
            
        Returns:
            RAGOutput containing response and metadata
        """
        try:
            # Retrieve relevant contexts
            relevant_contexts = await self._retrieve_contexts(query, context_docs)
            
            # Prepare prompt with retrieved contexts
            prompt = self._prepare_prompt(query, relevant_contexts)
            
            # Generate response
            response = await self._generate_response(prompt)
            
            return RAGOutput(
                response=response,
                retrieved_contexts=relevant_contexts,
                metadata={
                    "query": query,
                    "context_count": len(relevant_contexts),
                    "model_name": self.llm_model.config.name_or_path
                },
                created_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            raise
            
    async def _retrieve_contexts(self,
                               query: str,
                               context_docs: List[Dict],
                               top_k: int = 5) -> List[Dict]:
        """Retrieve relevant contexts for the query."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.generate_embeddings(query)
            
            # Get context texts and generate embeddings
            context_texts = [doc['content'] for doc in context_docs]
            context_embeddings = self.embedding_model.generate_embeddings(context_texts)
            
            # Compute similarities and get top matches
            similarities = [
                self.embedding_model.compute_similarity(
                    query_embedding.embeddings[0],
                    context_emb
                )
                for context_emb in context_embeddings.embeddings
            ]
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            return [
                {
                    **context_docs[idx],
                    'similarity_score': similarities[idx]
                }
                for idx in top_indices
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving contexts: {str(e)}")
            raise
            
    def _prepare_prompt(self,
                       query: str,
                       contexts: List[Dict]) -> str:
        """Prepare prompt with query and retrieved contexts."""
        try:
            # Format contexts
            context_text = "\n\n".join([
                f"Context {i+1}:\n{ctx['content']}"
                for i, ctx in enumerate(contexts)
            ])
            
            # Construct prompt
            prompt = (
                f"Based on the following financial information:\n\n"
                f"{context_text}\n\n"
                f"Please answer the following question:\n{query}"
            )
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error preparing prompt: {str(e)}")
            raise
            
    async def _generate_response(self, prompt: str) -> str:
        """Generate response using the language model."""
        try:
            # Tokenize input
            inputs = self.llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.llm_tokenizer.model_max_length
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9
                )
                
            # Decode response
            response = self.llm_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
            
    async def analyze_financial_document(self,
                                       document: Dict,
                                       analysis_type: str = "general") -> Dict:
        """
        Perform specific analysis on a financial document.
        
        Args:
            document: Document to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Prepare analysis-specific prompt
            if analysis_type == "earnings":
                prompt = self._prepare_earnings_analysis_prompt(document)
            elif analysis_type == "risk":
                prompt = self._prepare_risk_analysis_prompt(document)
            else:
                prompt = self._prepare_general_analysis_prompt(document)
                
            # Generate analysis
            analysis = await self._generate_response(prompt)
            
            return {
                "analysis_type": analysis_type,
                "analysis": analysis,
                "metadata": {
                    "document_type": document.get('type'),
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in document analysis: {str(e)}")
            raise
            
    def _prepare_earnings_analysis_prompt(self, document: Dict) -> str:
        """Prepare prompt for earnings call analysis."""
        return f"Analyze the following earnings call transcript for key financial insights:\n\n{document['content']}"
        
    def _prepare_risk_analysis_prompt(self, document: Dict) -> str:
        """Prepare prompt for risk analysis."""
        return f"Identify key risks and concerns in the following financial document:\n\n{document['content']}"
        
    def _prepare_general_analysis_prompt(self, document: Dict) -> str:
        """Prepare prompt for general analysis."""
        return f"Provide a comprehensive analysis of the following financial document:\n\n{document['content']}"
