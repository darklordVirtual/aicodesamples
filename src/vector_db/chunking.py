"""
Kapittel 5: Intelligent Text Chunking
Smart strategies for splitting text into optimal chunks for embeddings.
"""
from typing import List, Dict, Any, Optional
import re

try:
    from utils import logger, LoggerMixin
    from fundamentals.embeddings import EmbeddingService
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import logger, LoggerMixin
    from fundamentals.embeddings import EmbeddingService


def intelligent_chunking(
    text: str,
    max_chunk_size: int = 512,
    overlap: int = 50,
    split_on: str = "sentence"
) -> List[str]:
    """
    Split text into intelligent chunks with overlap.
    
    Args:
        text: Text to split
        max_chunk_size: Maximum characters per chunk
        overlap: Characters to overlap between chunks
        split_on: 'sentence', 'paragraph', or 'word'
        
    Returns:
        List of text chunks
    """
    if split_on == "paragraph":
        segments = text.split('\n\n')
    elif split_on == "sentence":
        # Simple sentence splitting
        segments = re.split(r'(?<=[.!?])\s+', text)
    else:  # word
        segments = text.split()
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for segment in segments:
        segment_size = len(segment)
        
        if current_size + segment_size > max_chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_text = ' '.join(current_chunk)
            if len(overlap_text) > overlap:
                overlap_text = overlap_text[-overlap:]
                overlap_words = overlap_text.split()
                current_chunk = overlap_words
                current_size = len(overlap_text)
            else:
                current_chunk = []
                current_size = 0
        
        current_chunk.append(segment)
        current_size += segment_size + 1  # +1 for space
    
    # Add last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


class SemanticChunker(LoggerMixin):
    """
    Semantic-aware text chunking using embeddings.
    
    Splits text where semantic similarity drops below threshold.
    """
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        similarity_threshold: float = 0.7
    ):
        self.embedder = embedding_service or EmbeddingService()
        self.similarity_threshold = similarity_threshold
    
    def chunk_by_similarity(
        self,
        text: str,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Split text where semantic similarity drops.
        
        Args:
            text: Text to chunk
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of chunks with metadata
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if not sentences:
            return []
        
        # Get embeddings for each sentence
        self.log_info(f"Computing embeddings for {len(sentences)} sentences")
        embeddings = [self.embedder.get_embedding(s) for s in sentences]
        
        # Find split points where similarity drops
        chunks = []
        current_chunk = [sentences[0]]
        current_size = len(sentences[0])
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = self.embedder.cosine_similarity(
                embeddings[i-1],
                embeddings[i]
            )
            
            sentence_size = len(sentences[i])
            
            # Decide whether to split
            should_split = (
                similarity < self.similarity_threshold or
                current_size + sentence_size > max_chunk_size
            )
            
            if should_split and current_size >= min_chunk_size:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "start_sentence": len(chunks) * len(current_chunk),
                    "num_sentences": len(current_chunk),
                    "size": len(chunk_text)
                })
                
                # Start new chunk
                current_chunk = [sentences[i]]
                current_size = sentence_size
            else:
                current_chunk.append(sentences[i])
                current_size += sentence_size + 1
        
        # Add last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "start_sentence": len(chunks) * len(current_chunk),
                "num_sentences": len(current_chunk),
                "size": len(chunk_text)
            })
        
        self.log_info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def chunk_with_headers(
        self,
        text: str,
        max_chunk_size: int = 800
    ) -> List[Dict[str, Any]]:
        """
        Chunk text while preserving headers for context.
        
        Args:
            text: Text with markdown-style headers
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of chunks with header context
        """
        # Find headers (lines starting with #)
        lines = text.split('\n')
        
        chunks = []
        current_header = None
        current_chunk = []
        current_size = 0
        
        for line in lines:
            # Check if line is a header
            if line.strip().startswith('#'):
                # Save previous chunk if exists
                if current_chunk:
                    chunks.append({
                        "text": '\n'.join(current_chunk),
                        "header": current_header,
                        "size": current_size
                    })
                
                # Start new section
                current_header = line.strip()
                current_chunk = [line]
                current_size = len(line)
            else:
                line_size = len(line)
                
                # Split if chunk too large
                if current_size + line_size > max_chunk_size and current_chunk:
                    chunks.append({
                        "text": '\n'.join(current_chunk),
                        "header": current_header,
                        "size": current_size
                    })
                    
                    # Continue with header context
                    current_chunk = [current_header, line] if current_header else [line]
                    current_size = len(current_header) + line_size if current_header else line_size
                else:
                    current_chunk.append(line)
                    current_size += line_size + 1
        
        # Add last chunk
        if current_chunk:
            chunks.append({
                "text": '\n'.join(current_chunk),
                "header": current_header,
                "size": current_size
            })
        
        self.log_info(f"Created {len(chunks)} header-aware chunks")
        return chunks


# Example usage
def example_basic_chunking():
    """Example: Basic chunking strategies"""
    text = """
    Kunstig intelligens har revolusjonert måten vi arbeider på. AI kan analysere store mengder data.
    Det finnes mange typer AI. Machine learning er en type AI som lærer fra data.
    Deep learning bruker nevrale nettverk. Dette er inspirert av menneskehjernen.
    Natural language processing lar maskiner forstå tekst. Dette brukes i chatbots.
    Computer vision lar maskiner se. Dette brukes i selvkjørende biler.
    """
    
    # Sentence-based chunking
    sentence_chunks = intelligent_chunking(text, max_chunk_size=150, overlap=30, split_on="sentence")
    print("Sentence chunks:")
    for i, chunk in enumerate(sentence_chunks, 1):
        print(f"{i}. {chunk[:80]}...")
    
    # Word-based chunking
    word_chunks = intelligent_chunking(text, max_chunk_size=100, overlap=20, split_on="word")
    print(f"\nWord chunks: {len(word_chunks)} chunks")


def example_semantic_chunking():
    """Example: Semantic chunking"""
    text = """
    Python er et programmeringsspråk. Det er lett å lære. Mange bruker det for dataanalyse.
    
    JavaScript er annerledes. Det kjører i nettleseren. Det brukes for webutvikling.
    
    Machine learning handler om AI. Modeller lærer fra data. Dette krever mye beregning.
    """
    
    chunker = SemanticChunker(similarity_threshold=0.75)
    chunks = chunker.chunk_by_similarity(text, min_chunk_size=50, max_chunk_size=300)
    
    print("\nSemantic chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. [{chunk['num_sentences']} sentences, {chunk['size']} chars]")
        print(f"   {chunk['text'][:80]}...")


def example_header_chunking():
    """Example: Header-aware chunking"""
    text = """# Introduksjon til AI
Dette er en introduksjon til kunstig intelligens. AI er et spennende felt.

## Machine Learning
Machine learning er en delgren av AI. Modeller lærer fra data.

### Supervised Learning
Supervised learning bruker merkede data. Dette er den vanligste typen.

### Unsupervised Learning
Unsupervised learning finner mønstre selv. Dette brukes for clustering.

## Deep Learning
Deep learning bruker nevrale nettverk. Dette krever mye data og beregning."""
    
    chunker = SemanticChunker()
    chunks = chunker.chunk_with_headers(text, max_chunk_size=200)
    
    print("\nHeader-aware chunks:")
    for i, chunk in enumerate(chunks, 1):
        header = chunk['header'] or "No header"
        print(f"{i}. {header} ({chunk['size']} chars)")


if __name__ == "__main__":
    print("=== Basic Chunking ===")
    example_basic_chunking()
    
    print("\n=== Semantic Chunking ===")
    example_semantic_chunking()
    
    print("\n=== Header-Aware Chunking ===")
    example_header_chunking()
