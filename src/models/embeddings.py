import torch
from typing import List, Dict, Optional, Union
from transformers import AutoTokenizer, AutoModel
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingOutput:
    """Data class for embedding outputs."""
    embeddings: np.ndarray
    metadata: Dict
    created_at: datetime

class FinancialEmbeddingModel:
    """
    Handles generation of embeddings for financial texts using
    transformer models optimized for financial domain.
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 device: Optional[str] = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        try:
            self.model = SentenceTransformer(model_name).to(self.device)
            logger.info(f"Loaded embedding model {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = self.tokenizer.model_max_length
        
    def generate_embeddings(self,
                          texts: Union[str, List[str]],
                          batch_size: int = 32) -> EmbeddingOutput:
        """
        Generate embeddings for input texts.
        
        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            EmbeddingOutput containing embeddings and metadata
        """
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            # Process in batches
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self._generate_batch_embeddings(batch_texts)
                all_embeddings.append(batch_embeddings)
                
            # Concatenate all embeddings
            final_embeddings = np.vstack(all_embeddings)
            
            return EmbeddingOutput(
                embeddings=final_embeddings,
                metadata={
                    "model_name": self.model_name,
                    "device": self.device,
                    "input_count": len(texts),
                    "embedding_dim": final_embeddings.shape[1]
                },
                created_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
            
    def _generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        try:
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            return embeddings
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {str(e)}")
            raise
            
    def update_model(self, new_model_name: str):
        """
        Update the embedding model to use a different pre-trained model.
        
        Args:
            new_model_name: Name of the new pre-trained model to use
        """
        try:
            self.model = SentenceTransformer(new_model_name).to(self.device)
            self.model_name = new_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(new_model_name)
            self.max_length = self.tokenizer.model_max_length
            logger.info(f"Successfully updated to model {new_model_name}")
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            raise
            
    async def generate_embeddings_async(self,
                                      texts: Union[str, List[str]],
                                      batch_size: int = 32) -> EmbeddingOutput:
        """
        Asynchronously generate embeddings for input texts.
        
        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            EmbeddingOutput containing embeddings and metadata
        """
        # Note: The actual implementation would need to use proper async libraries
        return self.generate_embeddings(texts, batch_size)
    
    def compute_similarity(self,
                         embedding1: np.ndarray,
                         embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        try:
            return float(np.dot(embedding1, embedding2) / 
                        (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            raise
            
    def search_similar_texts(self,
                           query_text: str,
                           corpus_texts: List[str],
                           top_k: int = 5) -> List[Dict]:
        """
        Search for most similar texts in a corpus.
        
        Args:
            query_text: Text to search for
            corpus_texts: List of texts to search in
            top_k: Number of similar texts to return
            
        Returns:
            List of dictionaries containing similar texts and scores
        """
        try:
            # Generate embeddings
            query_embedding = self.generate_embeddings(query_text).embeddings[0]
            corpus_embeddings = self.generate_embeddings(corpus_texts).embeddings
            
            # Compute similarities
            similarities = [
                self.compute_similarity(query_embedding, corp_emb)
                for corp_emb in corpus_embeddings
            ]
            
            # Get top_k similar texts
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            return [
                {
                    "text": corpus_texts[idx],
                    "similarity_score": similarities[idx],
                    "index": idx
                }
                for idx in top_indices
            ]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
