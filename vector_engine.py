# vector_engine.py
"""
Advanced Vector Embedding Engine for semantic similarity
Implements multiple embedding techniques for optimal retrieval
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Any
import logging
from config import Config

logger = logging.getLogger(__name__)

class VectorEmbeddingEngine:
    """
    Advanced Vector Embedding Engine for semantic similarity
    Supports multiple embedding methods for optimal retrieval performance
    """
    
    def __init__(self, embedding_method: str = None):
        self.embedding_method = embedding_method or Config.EMBEDDING_METHOD
        self.vectorizer = None
        self.embeddings = None
        self.documents = []
        self._embedding_cache = {}
        
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Create embeddings for documents
        
        Args:
            documents: List of text documents to embed
            
        Returns:
            Document embeddings matrix
        """
        self.documents = documents
        logger.info(f"Creating embeddings for {len(documents)} documents using {self.embedding_method} method")
        
        if self.embedding_method == "tfidf":
            self.embeddings = self._create_tfidf_embeddings(documents)
        elif self.embedding_method == "hybrid":
            self.embeddings = self._create_hybrid_embeddings(documents)
        else:
            raise ValueError(f"Unknown embedding method: {self.embedding_method}")
            
        logger.info(f"Embeddings created with shape: {self.embeddings.shape}")
        return self.embeddings
    
    def _create_tfidf_embeddings(self, documents: List[str]) -> np.ndarray:
        """Create TF-IDF embeddings"""
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            max_features=Config.TFIDF_MAX_FEATURES,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            norm='l2'
        )
        return self.vectorizer.fit_transform(documents)
    
    def _create_hybrid_embeddings(self, documents: List[str]) -> np.ndarray:
        """
        Create hybrid embeddings combining TF-IDF and character n-grams
        This improves matching for misspellings and partial matches
        """
        # TF-IDF component for semantic similarity
        tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000,
            norm='l2'
        )
        tfidf_embeddings = tfidf_vectorizer.fit_transform(documents)
        
        # Character n-gram component for fuzzy matching
        char_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=Config.CHAR_NGRAM_RANGE,
            max_features=500,
            norm='l2'
        )
        char_embeddings = char_vectorizer.fit_transform(documents)
        
        # Combine embeddings (weighted combination)
        from scipy.sparse import hstack
        combined_embeddings = hstack([
            tfidf_embeddings * 0.7,  # Weight semantic similarity higher
            char_embeddings * 0.3    # Weight fuzzy matching lower
        ])
        
        # Store both vectorizers
        self.vectorizer = {
            'tfidf': tfidf_vectorizer,
            'char': char_vectorizer
        }
        
        return combined_embeddings
    
    def transform(self, query: str) -> np.ndarray:
        """
        Transform query to embedding space
        
        Args:
            query: Input query string
            
        Returns:
            Query embedding vector
        """
        # Check cache first
        query_hash = hash(query)
        if Config.ENABLE_CACHING and query_hash in self._embedding_cache:
            return self._embedding_cache[query_hash]
        
        if self.embedding_method == "tfidf":
            query_embedding = self.vectorizer.transform([query])
        elif self.embedding_method == "hybrid":
            tfidf_query = self.vectorizer['tfidf'].transform([query])
            char_query = self.vectorizer['char'].transform([query])
            from scipy.sparse import hstack
            query_embedding = hstack([tfidf_query * 0.7, char_query * 0.3])
        else:
            raise ValueError(f"Unknown embedding method: {self.embedding_method}")
        
        # Cache the result
        if Config.ENABLE_CACHING:
            self._embedding_cache[query_hash] = query_embedding
        
        return query_embedding
    
    def compute_similarity(self, query_embedding: np.ndarray, top_k: int = None) -> List[Tuple[int, float]]:
        """
        Compute cosine similarity and return top-k matches
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top matches to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        top_k = top_k or Config.TOP_K_RETRIEVAL
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k indices with similarity scores above threshold
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [
            (idx, similarities[idx]) 
            for idx in top_indices 
            if similarities[idx] > Config.MIN_SIMILARITY_THRESHOLD
        ]
        
        return results
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding space"""
        if self.embeddings is None:
            return {"error": "No embeddings created yet"}
        
        return {
            "embedding_method": self.embedding_method,
            "num_documents": len(self.documents),
            "embedding_dimension": self.embeddings.shape[1],
            "sparsity": 1.0 - (self.embeddings.nnz / (self.embeddings.shape[0] * self.embeddings.shape[1])),
            "cache_size": len(self._embedding_cache) if Config.ENABLE_CACHING else 0
        }
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")