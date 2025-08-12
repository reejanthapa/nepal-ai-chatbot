# context_manager.py
"""
Smart Context Management for RAG Pipeline
Handles context window optimization and relevance scoring
"""

import numpy as np
from typing import List, Dict, Any
import logging
from config import Config

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Intelligent context management system for RAG pipeline
    Optimizes context selection and formatting for LLM consumption
    """
    
    def __init__(self, max_context_length: int = None):
        self.max_context_length = max_context_length or Config.MAX_CONTEXT_LENGTH
        self.context_history = []
        self.relevance_weights = {
            'similarity': 0.4,
            'length_similarity': 0.2,
            'keyword_overlap': 0.2,
            'recency': 0.1,
            'completeness': 0.1
        }
        
    def select_relevant_examples(self, 
                               similar_examples: List[Dict], 
                               user_question: str,
                               max_examples: int = None) -> List[Dict]:
        """
        Select most relevant examples using multi-factor scoring
        
        Args:
            similar_examples: List of candidate examples with similarity scores
            user_question: Original user query
            max_examples: Maximum number of examples to select
            
        Returns:
            List of selected examples with relevance scores
        """
        max_examples = max_examples or Config.MAX_EXAMPLES
        
        if not similar_examples:
            logger.warning("No similar examples provided")
            return []
        
        # Score examples using multiple relevance factors
        scored_examples = []
        for example in similar_examples:
            relevance_score = self._compute_multi_factor_relevance(example, user_question)
            example['relevance_score'] = relevance_score
            example['context_rank'] = len(scored_examples) + 1
            scored_examples.append(example)
        
        # Sort by relevance score
        scored_examples.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Select examples that fit within context window
        selected_examples = self._fit_context_window(scored_examples, max_examples)
        
        logger.info(f"Selected {len(selected_examples)} examples from {len(similar_examples)} candidates")
        return selected_examples
    
    def _compute_multi_factor_relevance(self, example: Dict, user_question: str) -> float:
        """
        Compute relevance score based on multiple factors
        
        Args:
            example: Example with similarity score and content
            user_question: User's query
            
        Returns:
            Composite relevance score (0-1)
        """
        score = 0.0
        
        # Factor 1: Base similarity score from vector search
        similarity = example.get('similarity', 0.0)
        score += similarity * self.relevance_weights['similarity']
        
        # Factor 2: Question length similarity (prefer similar complexity)
        length_sim = self._compute_length_similarity(example.get('input', ''), user_question)
        score += length_sim * self.relevance_weights['length_similarity']
        
        # Factor 3: Keyword overlap
        keyword_overlap = self._compute_keyword_overlap(example.get('input', ''), user_question)
        score += keyword_overlap * self.relevance_weights['keyword_overlap']
        
        # Factor 4: Response completeness (longer, more detailed responses score higher)
        completeness = self._compute_completeness_score(example.get('output', ''))
        score += completeness * self.relevance_weights['completeness']
        
        # Factor 5: Recency bias (more recent interactions preferred)
        recency = self._compute_recency_score(example)
        score += recency * self.relevance_weights['recency']
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _compute_length_similarity(self, example_input: str, user_question: str) -> float:
        """Compute similarity based on question length"""
        if not example_input or not user_question:
            return 0.0
        
        input_len = len(example_input.split())
        query_len = len(user_question.split())
        
        if input_len == 0 or query_len == 0:
            return 0.0
        
        # Compute length similarity (closer lengths = higher score)
        length_ratio = min(input_len, query_len) / max(input_len, query_len)
        return length_ratio
    
    def _compute_keyword_overlap(self, example_input: str, user_question: str) -> float:
        """Compute keyword overlap between example and query"""
        if not example_input or not user_question:
            return 0.0
        
        # Extract keywords (filter out common stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'how', 'what', 'where', 'when', 'why', 'who'}
        
        input_words = set(word.lower().strip('.,!?;:"()[]{}') 
                         for word in example_input.split() 
                         if len(word) > 2 and word.lower() not in stop_words)
        query_words = set(word.lower().strip('.,!?;:"()[]{}') 
                         for word in user_question.split() 
                         if len(word) > 2 and word.lower() not in stop_words)
        
        if not input_words or not query_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(input_words & query_words)
        union = len(input_words | query_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_completeness_score(self, example_output: str) -> float:
        """Score based on response completeness and detail level"""
        if not example_output:
            return 0.0
        
        # Factors that indicate completeness
        word_count = len(example_output.split())
        sentence_count = len([s for s in example_output.split('.') if s.strip()])
        
        # Normalize scores
        word_score = min(word_count / 100, 1.0)  # Normalize to 100 words
        sentence_score = min(sentence_count / 5, 1.0)  # Normalize to 5 sentences
        
        # Check for additional tip (indicates comprehensive response)
        has_tip = 'tip:' in example_output.lower() or 'additional' in example_output.lower()
        tip_bonus = 0.2 if has_tip else 0.0
        
        return (word_score * 0.5 + sentence_score * 0.3 + tip_bonus)
    
    def _compute_recency_score(self, example: Dict) -> float:
        """Compute recency score (placeholder for future enhancement)"""
        # For now, return neutral score
        # In future, could track when examples were last used successfully
        return 0.5
    
    def _fit_context_window(self, scored_examples: List[Dict], max_examples: int) -> List[Dict]:
        """
        Select examples that fit within the context window
        
        Args:
            scored_examples: Examples sorted by relevance
            max_examples: Maximum number of examples
            
        Returns:
            Examples that fit within context constraints
        """
        selected_examples = []
        current_length = 0
        
        for example in scored_examples[:max_examples]:
            # Estimate token length (rough approximation: 4 chars per token)
            example_text = (example.get('input', '') + 
                          example.get('output', '') + 
                          example.get('nepali', ''))
            estimated_tokens = len(example_text) // 4
            
            if current_length + estimated_tokens < self.max_context_length:
                selected_examples.append(example)
                current_length += estimated_tokens
                logger.debug(f"Added example {len(selected_examples)}: {estimated_tokens} tokens")
            else:
                logger.debug(f"Skipping example due to context limit: {estimated_tokens} tokens")
                break
        
        return selected_examples
    
    def build_context_string(self, examples: List[Dict], user_question: str) -> str:
        """
        Build optimized context string for the LLM prompt
        
        Args:
            examples: Selected relevant examples
            user_question: Original user query
            
        Returns:
            Formatted context string
        """
        if not examples:
            return "No relevant examples found in knowledge base."
        
        context_parts = [
            "=== NEPAL KNOWLEDGE BASE ===",
            f"Query: {user_question}",
            f"Found {len(examples)} relevant examples:\n"
        ]
        
        for i, example in enumerate(examples, 1):
            example_context = [
                f"--- Example {i} (Relevance: {example.get('relevance_score', 0):.3f}) ---",
                f"Question: {example.get('input', 'N/A')}",
                f"Answer: {example.get('output', 'N/A')}"
            ]
            
            # Add Nepali translation if available
            if example.get('nepali'):
                example_context.append(f"Nepali: {example['nepali']}")
            
            # Add metadata for debugging
            if example.get('similarity'):
                example_context.append(f"Vector Similarity: {example['similarity']:.3f}")
            
            context_parts.extend(example_context)
            context_parts.append("")  # Empty line between examples
        
        context_parts.append("=== END KNOWLEDGE BASE ===\n")
        
        return "\n".join(context_parts)
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get context management statistics"""
        return {
            'max_context_length': self.max_context_length,
            'relevance_weights': self.relevance_weights,
            'context_history_size': len(self.context_history)
        }
    
    def update_relevance_weights(self, new_weights: Dict[str, float]):
        """Update relevance scoring weights"""
        # Validate weights sum to 1.0
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Relevance weights don't sum to 1.0: {total_weight}")
        
        self.relevance_weights.update(new_weights)
        logger.info(f"Updated relevance weights: {self.relevance_weights}")