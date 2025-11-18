"""
Kapittel 8: RAG System (Retrieval-Augmented Generation)
Complete RAG system combining vector search with AI generation.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from anthropic import Anthropic

try:
    from utils import config, logger, LoggerMixin
    from vector_db.chromadb_basics import KnowledgeBase
    from vector_db.advanced_chromadb import AdvancedSearch, HybridSearcher
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import config, logger, LoggerMixin
    from vector_db.chromadb_basics import KnowledgeBase
    from vector_db.advanced_chromadb import AdvancedSearch, HybridSearcher


@dataclass
class QueryResult:
    """Result from RAG query"""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    confidence: Optional[float] = None


class RAGSystem(LoggerMixin):
    """
    Complete Retrieval-Augmented Generation system.
    
    Features:
    - Multi-query retrieval
    - Hybrid search
    - Source attribution
    - Confidence scoring
    """
    
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        model: Optional[str] = None,
        use_hybrid_search: bool = False
    ):
        self.kb = knowledge_base
        self.model = model or config.ai.model
        self.client = Anthropic(api_key=config.ai.api_key)
        
        if use_hybrid_search:
            self.searcher = HybridSearcher(knowledge_base)
            self.log_info("Initialized RAG with hybrid search")
        else:
            self.searcher = AdvancedSearch(knowledge_base)
            self.log_info("Initialized RAG with multi-query search")
    
    def query(
        self,
        question: str,
        n_results: int = 5,
        include_sources: bool = True
    ) -> QueryResult:
        """
        Query the RAG system.
        
        Args:
            question: User question
            n_results: Number of context documents
            include_sources: Include source attribution
            
        Returns:
            QueryResult with answer and sources
        """
        self.log_info(f"RAG query: {question}")
        
        # Retrieve relevant documents
        if hasattr(self.searcher, 'multi_query_search'):
            results = self.searcher.multi_query_search(
                question,
                n_results=n_results
            )
        else:
            results = self.searcher.hybrid_search(
                question,
                n_results=n_results
            )
        
        # Build context from results
        context_docs = []
        for i, result in enumerate(results[:n_results], 1):
            doc_text = result.get('document', '')
            context_docs.append(f"[{i}] {doc_text}")
        
        context = "\n\n".join(context_docs)
        
        # Generate answer with context
        prompt = f"""Answer the following question based on the provided context.
If the context doesn't contain enough information, say so.
Always cite sources using [1], [2], etc.

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer = response.content[0].text.strip()
        
        # Build sources list
        sources = []
        if include_sources:
            for i, result in enumerate(results[:n_results], 1):
                sources.append({
                    "index": i,
                    "text": result.get('document', ''),
                    "metadata": result.get('metadata', {}),
                    "score": result.get('rrf_score') or result.get('hybrid_score', 0)
                })
        
        self.log_info(f"Generated answer with {len(sources)} sources")
        
        return QueryResult(
            answer=answer,
            sources=sources,
            query=question
        )
    
    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add document to knowledge base.
        
        Args:
            text: Document text
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        doc_id = self.kb.add(text, metadata)
        self.log_info(f"Added document {doc_id} to RAG knowledge base")
        return doc_id
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add multiple documents in batch.
        
        Args:
            texts: List of document texts
            metadatas: Optional metadata list
            
        Returns:
            List of document IDs
        """
        doc_ids = self.kb.add_batch(texts, metadatas)
        self.log_info(f"Added {len(doc_ids)} documents to RAG knowledge base")
        return doc_ids


# Example usage
def example_rag_system():
    """Example: Complete RAG system"""
    # Create knowledge base
    kb = KnowledgeBase(collection_name="rag_demo")
    
    # Add documents about Norwegian AI
    documents = [
        "Norge har flere sterke AI-miljøer, særlig i Oslo, Trondheim og Bergen.",
        "Norske bedrifter bruker AI for automatisering, analyse og kundeservice.",
        "NTNU i Trondheim er ledende innen AI-forskning i Norge.",
        "Telenor og DNB er blant de største brukerne av AI i Norge.",
        "Schibsted bruker AI for anbefalingssystemer og innholdsmoderering.",
    ]
    
    metadatas = [
        {"topic": "geography", "source": "report"},
        {"topic": "business", "source": "report"},
        {"topic": "research", "source": "university"},
        {"topic": "business", "source": "news"},
        {"topic": "business", "source": "news"}
    ]
    
    # Initialize RAG
    rag = RAGSystem(kb, use_hybrid_search=False)
    rag.add_documents(documents, metadatas)
    
    # Query
    result = rag.query(
        "Hvilke norske bedrifter bruker AI?",
        n_results=3
    )
    
    print(f"Question: {result.query}\n")
    print(f"Answer: {result.answer}\n")
    print("Sources:")
    for source in result.sources:
        print(f"  [{source['index']}] {source['text'][:60]}...")
    
    kb.clear()


if __name__ == "__main__":
    print("=== RAG System Example ===")
    example_rag_system()
