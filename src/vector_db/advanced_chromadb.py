"""
Kapittel 5: Avansert ChromaDB
Advanced search techniques: hybrid search, multi-query, and reranking.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

try:
    from utils import config, logger, LoggerMixin
    from fundamentals.embeddings import EmbeddingService
    from vector_db.chromadb_basics import KnowledgeBase
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import config, logger, LoggerMixin
    from fundamentals.embeddings import EmbeddingService
    from vector_db.chromadb_basics import KnowledgeBase


class AdvancedSearch(LoggerMixin):
    """
    Advanced search techniques for ChromaDB.
    
    Features:
    - Multi-query search (query expansion)
    - Reciprocal rank fusion (RRF)
    - Query decomposition
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        from anthropic import Anthropic
        self.client = Anthropic(api_key=config.ai.api_key)
    
    def multi_query_search(
        self,
        query: str,
        n_results: int = 5,
        n_variations: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple query variations and combine results.
        
        Args:
            query: Original query
            n_results: Number of results per query
            n_variations: Number of query variations to generate
            
        Returns:
            Fused and ranked results
        """
        self.log_info(f"Multi-query search for: {query}")
        
        # Generate query variations
        prompt = f"""Generate {n_variations} different ways to ask this question.
Each variation should capture different aspects or phrasings.

Original question: {query}

Return only the {n_variations} variations, one per line, without numbering."""

        response = self.client.messages.create(
            model=config.ai.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        variations = [line.strip() for line in response.content[0].text.strip().split('\n') if line.strip()]
        variations = [query] + variations[:n_variations]  # Include original
        
        self.log_info(f"Generated {len(variations)} query variations")
        
        # Search with each variation
        all_results = []
        for var in variations:
            results = self.kb.query(var, n_results=n_results)
            all_results.append(results)
        
        # Fuse results using reciprocal rank fusion
        fused = self._reciprocal_rank_fusion(all_results)
        
        return fused[:n_results]
    
    def _reciprocal_rank_fusion(
        self,
        result_sets: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine multiple result sets using Reciprocal Rank Fusion.
        
        RRF formula: score = sum(1 / (k + rank))
        
        Args:
            result_sets: List of query results
            k: RRF constant (typically 60)
            
        Returns:
            Fused and ranked results
        """
        doc_scores = defaultdict(float)
        doc_data = {}
        
        for results in result_sets:
            for rank, (doc_id, doc, metadata, distance) in enumerate(
                zip(
                    results["ids"],
                    results["documents"],
                    results["metadatas"],
                    results["distances"]
                )
            ):
                # RRF score
                doc_scores[doc_id] += 1.0 / (k + rank + 1)
                
                # Store document data
                if doc_id not in doc_data:
                    doc_data[doc_id] = {
                        "id": doc_id,
                        "document": doc,
                        "metadata": metadata,
                        "original_distance": distance
                    }
        
        # Sort by RRF score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build result list
        fused_results = []
        for doc_id, score in sorted_docs:
            result = doc_data[doc_id].copy()
            result["rrf_score"] = score
            fused_results.append(result)
        
        self.log_info(f"Fused {len(result_sets)} result sets into {len(fused_results)} unique documents")
        
        return fused_results
    
    def decompose_and_search(
        self,
        complex_query: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Decompose complex query into sub-questions and search separately.
        
        Args:
            complex_query: Complex multi-part question
            n_results: Number of results per sub-question
            
        Returns:
            Dictionary with sub-questions and their results
        """
        self.log_info(f"Decomposing query: {complex_query}")
        
        # Decompose query
        prompt = f"""Break down this complex question into 2-4 simpler sub-questions.
Each sub-question should be self-contained and searchable.

Complex question: {complex_query}

Return only the sub-questions, one per line, without numbering."""

        response = self.client.messages.create(
            model=config.ai.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        sub_questions = [line.strip() for line in response.content[0].text.strip().split('\n') if line.strip()]
        
        self.log_info(f"Decomposed into {len(sub_questions)} sub-questions")
        
        # Search for each sub-question
        results = {}
        for sq in sub_questions:
            sq_results = self.kb.query(sq, n_results=n_results)
            results[sq] = sq_results
        
        return {
            "original_query": complex_query,
            "sub_questions": sub_questions,
            "results": results
        }


class HybridSearcher(LoggerMixin):
    """
    Hybrid search combining semantic and keyword search with reranking.
    """
    
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.kb = knowledge_base
        self.embedder = embedding_service or EmbeddingService()
        from anthropic import Anthropic
        self.client = Anthropic(api_key=config.ai.api_key)
    
    def hybrid_search(
        self,
        query: str,
        n_results: int = 10,
        semantic_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Combine semantic and keyword search with weighted fusion.
        
        Args:
            query: Search query
            n_results: Number of results
            semantic_weight: Weight for semantic search (0-1)
            
        Returns:
            Hybrid search results
        """
        keyword_weight = 1.0 - semantic_weight
        
        # Semantic search
        semantic_results = self.kb.query(query, n_results=n_results * 2)
        
        # Keyword search (simple text matching)
        keyword_results = self._keyword_search(query, n_results=n_results * 2)
        
        # Combine with weighted scores
        combined = self._weighted_fusion(
            semantic_results,
            keyword_results,
            semantic_weight,
            keyword_weight
        )
        
        return combined[:n_results]
    
    def _keyword_search(
        self,
        query: str,
        n_results: int
    ) -> Dict[str, Any]:
        """
        Simple keyword-based search.
        
        In a real implementation, this could use BM25 or full-text search.
        Here we use ChromaDB's semantic search with very specific terms.
        """
        # Extract keywords (simplified - in production use proper tokenization)
        keywords = query.lower().split()
        
        # Search with keywords
        keyword_query = " ".join(keywords)
        results = self.kb.query(keyword_query, n_results=n_results)
        
        # Score based on keyword matches
        scored_results = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "scores": []
        }
        
        for doc_id, doc, meta, dist in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
            results["distances"]
        ):
            # Count keyword matches
            doc_lower = doc.lower()
            matches = sum(1 for kw in keywords if kw in doc_lower)
            score = matches / len(keywords) if keywords else 0
            
            scored_results["ids"].append(doc_id)
            scored_results["documents"].append(doc)
            scored_results["metadatas"].append(meta)
            scored_results["scores"].append(score)
        
        return scored_results
    
    def _weighted_fusion(
        self,
        semantic_results: Dict[str, Any],
        keyword_results: Dict[str, Any],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """Combine semantic and keyword results with weights."""
        doc_scores = {}
        doc_data = {}
        
        # Process semantic results (convert distance to similarity)
        for doc_id, doc, meta, dist in zip(
            semantic_results["ids"],
            semantic_results["documents"],
            semantic_results["metadatas"],
            semantic_results["distances"]
        ):
            # Convert distance to similarity (assuming L2 distance)
            similarity = 1.0 / (1.0 + dist)
            score = similarity * semantic_weight
            
            doc_scores[doc_id] = score
            doc_data[doc_id] = {"id": doc_id, "document": doc, "metadata": meta}
        
        # Add keyword scores
        for doc_id, doc, meta, kw_score in zip(
            keyword_results["ids"],
            keyword_results["documents"],
            keyword_results["metadatas"],
            keyword_results["scores"]
        ):
            weighted_kw = kw_score * keyword_weight
            
            if doc_id in doc_scores:
                doc_scores[doc_id] += weighted_kw
            else:
                doc_scores[doc_id] = weighted_kw
                doc_data[doc_id] = {"id": doc_id, "document": doc, "metadata": meta}
        
        # Sort by combined score
        sorted_results = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build result list
        results = []
        for doc_id, score in sorted_results:
            result = doc_data[doc_id].copy()
            result["hybrid_score"] = score
            results.append(result)
        
        return results
    
    def rerank_with_ai(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using AI to assess relevance.
        
        Args:
            query: Original query
            results: Search results to rerank
            top_k: Number of top results to return
            
        Returns:
            Reranked results
        """
        if not results:
            return []
        
        self.log_info(f"Reranking {len(results)} results with AI")
        
        # Build prompt with query and documents
        docs_text = "\n\n".join([
            f"Document {i+1}:\n{r['document']}"
            for i, r in enumerate(results[:20])  # Limit to avoid token limits
        ])
        
        prompt = f"""Given this query: "{query}"

Rank the following documents by relevance (1 = most relevant).
Return only the document numbers in order, comma-separated.

{docs_text}

Ranking (e.g., "3,1,5,2,4"):"""

        response = self.client.messages.create(
            model=config.ai.fast_model,  # Use fast model for efficiency
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse ranking
        try:
            ranking_text = response.content[0].text.strip()
            ranking = [int(x.strip()) - 1 for x in ranking_text.split(',')]
            
            # Reorder results
            reranked = []
            for idx in ranking[:top_k]:
                if 0 <= idx < len(results):
                    result = results[idx].copy()
                    result["rerank_position"] = len(reranked) + 1
                    reranked.append(result)
            
            self.log_info(f"Reranked to {len(reranked)} results")
            return reranked
            
        except Exception as e:
            self.log_error(f"Error parsing reranking: {e}")
            # Return original order on error
            return results[:top_k]


# Example usage
def example_multi_query():
    """Example: Multi-query search"""
    kb = KnowledgeBase(collection_name="multi_query_test")
    
    # Add documents
    docs = [
        "Python er et programmeringsspråk som er lett å lære",
        "JavaScript brukes til webutvikling og frontendprogrammering",
        "Machine learning krever god forståelse av matematikk",
        "Deep learning er en delgren av maskinlæring",
        "ChromaDB lagrer vektorrepresentasjoner av dokumenter"
    ]
    kb.add_batch(docs)
    
    # Multi-query search
    searcher = AdvancedSearch(kb)
    results = searcher.multi_query_search(
        "Hva er det beste språket for nybegynnere?",
        n_results=3,
        n_variations=2
    )
    
    print("Multi-query results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['document'][:60]}... (RRF: {r['rrf_score']:.3f})")
    
    kb.clear()


def example_hybrid_search():
    """Example: Hybrid search"""
    kb = KnowledgeBase(collection_name="hybrid_test")
    
    # Add documents with keywords
    docs = [
        "Python Flask er et lightweight web framework",
        "Django er et fullverdig Python web framework",
        "FastAPI er moderne og rask for API-utvikling",
        "React er et JavaScript bibliotek for UI",
        "Vue.js er et progressivt JavaScript framework"
    ]
    kb.add_batch(docs)
    
    # Hybrid search
    searcher = HybridSearcher(kb)
    results = searcher.hybrid_search(
        "Python web framework",
        n_results=3,
        semantic_weight=0.6
    )
    
    print("\nHybrid search results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['document']} (score: {r['hybrid_score']:.3f})")
    
    kb.clear()


if __name__ == "__main__":
    print("=== Multi-Query Search ===")
    example_multi_query()
    
    print("\n=== Hybrid Search ===")
    example_hybrid_search()
