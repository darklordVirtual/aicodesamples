"""
Kapittel 4: ChromaDB Grunnleggende
Basic ChromaDB operations for vector storage and retrieval.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import uuid

try:
    from utils import config, logger, LoggerMixin
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import config, logger, LoggerMixin


class KnowledgeBase(LoggerMixin):
    """
    Basic knowledge base using ChromaDB for vector storage.
    
    Features:
    - Add documents with automatic embeddings
    - Query by text with semantic search
    - Update and delete documents
    - Filter by metadata
    """
    
    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_directory: Optional[str] = None
    ):
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.log_info(f"Initialized persistent ChromaDB at {persist_directory}")
        else:
            self.client = chromadb.Client()
            self.log_info("Initialized in-memory ChromaDB")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Knowledge base collection"}
        )
        self.log_info(f"Using collection: {collection_name}")
    
    def add(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add a document to the knowledge base.
        
        Args:
            text: Document text
            metadata: Optional metadata dictionary
            doc_id: Optional custom document ID
            
        Returns:
            Document ID
        """
        if not doc_id:
            doc_id = str(uuid.uuid4())
        
        if metadata is None:
            metadata = {}
        
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        self.log_info(f"Added document {doc_id}")
        return doc_id
    
    def add_batch(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        doc_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add multiple documents in batch.
        
        Args:
            texts: List of document texts
            metadatas: Optional list of metadata dictionaries
            doc_ids: Optional list of custom document IDs
            
        Returns:
            List of document IDs
        """
        if not doc_ids:
            doc_ids = [str(uuid.uuid4()) for _ in texts]
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=doc_ids
        )
        
        self.log_info(f"Added {len(texts)} documents in batch")
        return doc_ids
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the knowledge base with semantic search.
        
        Args:
            query_text: Search query
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Query results with documents, distances, and metadata
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        self.log_info(f"Query returned {len(results['documents'][0])} results")
        
        # Format results
        return {
            "documents": results["documents"][0],
            "distances": results["distances"][0],
            "metadatas": results["metadatas"][0],
            "ids": results["ids"][0]
        }
    
    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data or None if not found
        """
        try:
            result = self.collection.get(ids=[doc_id])
            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "document": result["documents"][0],
                    "metadata": result["metadatas"][0]
                }
        except Exception as e:
            self.log_error(f"Error getting document {doc_id}: {e}")
        
        return None
    
    def update(
        self,
        doc_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update a document.
        
        Args:
            doc_id: Document ID
            text: New document text (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if successful
        """
        try:
            update_kwargs = {"ids": [doc_id]}
            if text is not None:
                update_kwargs["documents"] = [text]
            if metadata is not None:
                update_kwargs["metadatas"] = [metadata]
            
            self.collection.update(**update_kwargs)
            self.log_info(f"Updated document {doc_id}")
            return True
        except Exception as e:
            self.log_error(f"Error updating document {doc_id}: {e}")
            return False
    
    def delete(self, doc_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=[doc_id])
            self.log_info(f"Deleted document {doc_id}")
            return True
        except Exception as e:
            self.log_error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def count(self) -> int:
        """Get total document count."""
        return self.collection.count()
    
    def clear(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Knowledge base collection"}
            )
            self.log_info("Cleared all documents")
            return True
        except Exception as e:
            self.log_error(f"Error clearing collection: {e}")
            return False


# Example usage
def example_basic_operations():
    """Example: Basic CRUD operations"""
    kb = KnowledgeBase(collection_name="test_kb")
    
    # Add documents
    doc1_id = kb.add(
        "Python er et populært programmeringsspråk",
        metadata={"category": "programming", "language": "no"}
    )
    
    doc2_id = kb.add(
        "JavaScript brukes for webutvikling",
        metadata={"category": "programming", "language": "no"}
    )
    
    doc3_id = kb.add(
        "ChromaDB er en vektordatabase",
        metadata={"category": "database", "language": "no"}
    )
    
    print(f"Added 3 documents, total count: {kb.count()}")
    
    # Query
    results = kb.query("Hva er ChromaDB?", n_results=2)
    print("\nQuery results:")
    for doc, distance in zip(results["documents"], results["distances"]):
        print(f"- {doc} (distance: {distance:.3f})")
    
    # Filter by metadata
    prog_results = kb.query(
        "populært språk",
        n_results=5,
        where={"category": "programming"}
    )
    print(f"\nProgramming docs: {len(prog_results['documents'])}")
    
    # Update
    kb.update(doc1_id, metadata={"category": "programming", "language": "no", "difficulty": "easy"})
    
    # Get
    doc = kb.get(doc1_id)
    print(f"\nUpdated doc metadata: {doc['metadata']}")
    
    # Clean up
    kb.clear()


def example_batch_operations():
    """Example: Batch operations for efficiency"""
    kb = KnowledgeBase(collection_name="batch_test")
    
    # Prepare documents
    texts = [
        "Machine learning er en gren av kunstig intelligens",
        "Deep learning bruker nevrale nettverk",
        "Natural language processing analyserer tekst",
        "Computer vision forstår bilder",
        "Reinforcement learning lærer fra belønninger"
    ]
    
    metadatas = [
        {"topic": "ml", "complexity": "medium"},
        {"topic": "dl", "complexity": "high"},
        {"topic": "nlp", "complexity": "medium"},
        {"topic": "cv", "complexity": "high"},
        {"topic": "rl", "complexity": "high"}
    ]
    
    # Add in batch
    doc_ids = kb.add_batch(texts, metadatas)
    print(f"Added {len(doc_ids)} documents in batch")
    
    # Query for complex topics
    results = kb.query(
        "avanserte AI-teknikker",
        n_results=3,
        where={"complexity": "high"}
    )
    
    print("\nComplex AI topics:")
    for doc, meta in zip(results["documents"], results["metadatas"]):
        print(f"- [{meta['topic']}] {doc}")
    
    kb.clear()


if __name__ == "__main__":
    print("=== Basic Operations ===")
    example_basic_operations()
    
    print("\n=== Batch Operations ===")
    example_batch_operations()
