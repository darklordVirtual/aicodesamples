"""
Del II: Vector Database og Semantisk Søk
ChromaDB implementation for vector storage and semantic search.
"""

from .chromadb_basics import KnowledgeBase
from .advanced_chromadb import AdvancedSearch, HybridSearcher
from .chunking import intelligent_chunking, SemanticChunker
from .backup import backup_collection, restore_collection

__all__ = [
    "KnowledgeBase",
    "AdvancedSearch", 
    "HybridSearcher",
    "intelligent_chunking",
    "SemanticChunker",
    "backup_collection",
    "restore_collection",
]
