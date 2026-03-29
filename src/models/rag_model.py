from typing import List, Dict, Optional
import torch
import numpy as np
from datetime import datetime, timezone
import logging
from dataclasses import dataclass
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
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

    DEFAULT_LLM_MODEL = "google/flan-t5-small"

    def __init__(
        self,
        embedding_model: Optional[FinancialEmbeddingModel] = None,
        llm_model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the RAG model.

        Args:
            embedding_model: Model for generating embeddings
            llm_model_name: Name of the HuggingFace model to use for generation.
                            Defaults to 'google/flan-t5-small'.
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = embedding_model or FinancialEmbeddingModel()
        self.llm_model_name = llm_model_name or self.DEFAULT_LLM_MODEL

        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.llm_model_name
            ).to(self.device)
            logger.info(f"Initialized RAG model with {self.llm_model_name}")
        except Exception as e:
            logger.error(f"Error initializing RAG model: {str(e)}")
            raise

    async def process_query(
        self,
        query: str,
        context_docs: List[Dict],
        max_context_length: int = 2048,
    ) -> RAGOutput:
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
            relevant_contexts = await self._retrieve_contexts(query, context_docs)
            prompt = self._prepare_prompt(query, relevant_contexts)
            response = await self._generate_response(prompt)

            return RAGOutput(
                response=response,
                retrieved_contexts=relevant_contexts,
                metadata={
                    "query": query,
                    "context_count": len(relevant_contexts),
                    "model_name": self.llm_model_name,
                },
                created_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            raise

    async def _retrieve_contexts(
        self,
        query: str,
        context_docs: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """Retrieve relevant contexts for the query."""
        if not context_docs:
            return []

        try:
            query_embedding = self.embedding_model.generate_embeddings(query)
            context_texts = [str(doc.get("content", "")) for doc in context_docs]
            context_embeddings = self.embedding_model.generate_embeddings(context_texts)

            similarities = [
                self.embedding_model.compute_similarity(
                    query_embedding.embeddings[0],
                    context_emb,
                )
                for context_emb in context_embeddings.embeddings
            ]

            actual_top_k = min(top_k, len(context_docs))
            top_indices = np.argsort(similarities)[-actual_top_k:][::-1]

            return [
                {**context_docs[idx], "similarity_score": float(similarities[idx])}
                for idx in top_indices
            ]

        except Exception as e:
            logger.error(f"Error retrieving contexts: {str(e)}")
            raise

    def _prepare_prompt(self, query: str, contexts: List[Dict]) -> str:
        """Prepare prompt with query and retrieved contexts."""
        try:
            context_text = "\n\n".join(
                [
                    f"Context {i + 1}:\n{ctx.get('content', '')}"
                    for i, ctx in enumerate(contexts)
                ]
            )

            if context_text:
                prompt = (
                    f"Based on the following financial information:\n\n"
                    f"{context_text}\n\n"
                    f"Answer the following question:\n{query}"
                )
            else:
                prompt = f"Answer the following financial question:\n{query}"

            return prompt

        except Exception as e:
            logger.error(f"Error preparing prompt: {str(e)}")
            raise

    async def _generate_response(self, prompt: str) -> str:
        """Generate response using the language model."""
        try:
            inputs = self.llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_return_sequences=1,
                )

            response = self.llm_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    async def analyze(self, documents: List[Dict]) -> Dict:
        """
        Perform analysis on a list of financial documents.

        Args:
            documents: List of document dictionaries with content and metadata

        Returns:
            Dictionary containing analysis results
        """
        try:
            if not documents:
                return {"summary": "No documents provided for analysis.", "insights": []}

            query = "Summarize the key financial highlights, metrics, and insights from these documents."
            result = await self.process_query(query, documents)
            return {
                "summary": result.response,
                "insights": [
                    ctx.get("content", "")[:200] for ctx in result.retrieved_contexts
                ],
            }
        except Exception as e:
            logger.error(f"Error in document analysis: {str(e)}")
            raise

    def extract_insights(self, analysis: Dict) -> List[Dict]:
        """
        Extract key insights from an analysis result.

        Args:
            analysis: Dictionary containing analysis data

        Returns:
            List of insight dictionaries
        """
        insights = []
        try:
            if "analysis" in analysis and isinstance(analysis["analysis"], dict):
                summary = analysis["analysis"].get("summary", "")
                if summary:
                    insights.append({"type": "summary", "content": summary})

            if "metrics" in analysis and isinstance(analysis["metrics"], dict):
                metrics = analysis["metrics"]
                if metrics.get("revenue"):
                    insights.append(
                        {"type": "metric", "content": f"Revenue: {metrics['revenue']}"}
                    )
                if metrics.get("yoy_growth"):
                    insights.append(
                        {
                            "type": "metric",
                            "content": f"YoY Growth: {metrics['yoy_growth']}",
                        }
                    )

        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")

        return insights

    async def analyze_financial_document(
        self, document: Dict, analysis_type: str = "general"
    ) -> Dict:
        """
        Perform specific analysis on a financial document.

        Args:
            document: Document to analyze
            analysis_type: Type of analysis to perform

        Returns:
            Dictionary containing analysis results
        """
        try:
            if analysis_type == "earnings":
                prompt = self._prepare_earnings_analysis_prompt(document)
            elif analysis_type == "risk":
                prompt = self._prepare_risk_analysis_prompt(document)
            else:
                prompt = self._prepare_general_analysis_prompt(document)

            analysis = await self._generate_response(prompt)

            return {
                "analysis_type": analysis_type,
                "analysis": analysis,
                "metadata": {
                    "document_type": document.get("type"),
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"Error in document analysis: {str(e)}")
            raise

    def _prepare_earnings_analysis_prompt(self, document: Dict) -> str:
        """Prepare prompt for earnings call analysis."""
        content = document.get("content", "")
        return f"Analyze the following earnings call transcript for key financial insights:\n\n{content}"

    def _prepare_risk_analysis_prompt(self, document: Dict) -> str:
        """Prepare prompt for risk analysis."""
        content = document.get("content", "")
        return f"Identify key risks and concerns in the following financial document:\n\n{content}"

    def _prepare_general_analysis_prompt(self, document: Dict) -> str:
        """Prepare prompt for general analysis."""
        content = document.get("content", "")
        return f"Provide a comprehensive analysis of the following financial document:\n\n{content}"
