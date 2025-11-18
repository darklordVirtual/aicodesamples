"""
Kapittel 3: Embeddings og Semantisk Søk
Implementation of embeddings and semantic search functionality.
"""
from openai import OpenAI
import numpy as np
from typing import List, Dict, Any, Tuple

try:
    from utils import config, logger, LoggerMixin
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import config, logger, LoggerMixin


class EmbeddingService(LoggerMixin):
    """
    Service for generating and working with embeddings.
    Uses OpenAI's text-embedding-3-large model.
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=config.ai.openai_api_key)
        self.model = "text-embedding-3-large"
        self.dimensions = 3072
        self.logger.info(f"Initialized EmbeddingService with model: {self.model}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector (list of floats)
        """
        self.logger.debug(f"Generating embedding for text: {text[:50]}...")
        
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        
        embedding = response.data[0].embedding
        self.logger.debug(f"Generated embedding with {len(embedding)} dimensions")
        
        return embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        embeddings = [item.embedding for item in response.data]
        return embeddings
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Similarity score (0-1, where 1 is identical)
        """
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def find_most_similar(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find most similar texts to query using cosine similarity.
        
        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            top_k: Number of top results to return
        
        Returns:
            List of (text, similarity_score) tuples, sorted by similarity
        """
        self.logger.info(f"Finding {top_k} most similar from {len(candidate_texts)} candidates")
        
        # Get embeddings
        query_emb = self.get_embedding(query_text)
        candidate_embs = self.get_embeddings_batch(candidate_texts)
        
        # Calculate similarities
        similarities = []
        for text, emb in zip(candidate_texts, candidate_embs):
            sim = self.cosine_similarity(query_emb, emb)
            similarities.append((text, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class SemanticDeduplicator(LoggerMixin):
    """
    Detect and remove semantic duplicates using embeddings.
    """
    
    def __init__(self, embedding_service: EmbeddingService = None):
        self.embedding_service = embedding_service or EmbeddingService()
    
    def find_duplicates(
        self,
        texts: List[str],
        threshold: float = 0.85
    ) -> List[List[int]]:
        """
        Find groups of semantically similar texts (likely duplicates).
        
        Args:
            texts: List of texts to check
            threshold: Similarity threshold (0-1)
        
        Returns:
            List of duplicate groups (each group is list of indices)
        """
        self.logger.info(f"Checking {len(texts)} texts for duplicates (threshold={threshold})")
        
        # Generate embeddings
        embeddings = self.embedding_service.get_embeddings_batch(texts)
        
        # Find duplicates
        duplicate_groups = []
        processed = set()
        
        for i in range(len(texts)):
            if i in processed:
                continue
            
            group = [i]
            
            for j in range(i + 1, len(texts)):
                if j in processed:
                    continue
                
                similarity = self.embedding_service.cosine_similarity(
                    embeddings[i],
                    embeddings[j]
                )
                
                if similarity >= threshold:
                    group.append(j)
                    processed.add(j)
            
            if len(group) > 1:
                duplicate_groups.append(group)
                processed.add(i)
        
        self.logger.info(f"Found {len(duplicate_groups)} duplicate groups")
        return duplicate_groups
    
    def deduplicate(
        self,
        texts: List[str],
        threshold: float = 0.85,
        keep_first: bool = True
    ) -> List[str]:
        """
        Remove semantic duplicates from list.
        
        Args:
            texts: List of texts
            threshold: Similarity threshold
            keep_first: Keep first occurrence in each group
        
        Returns:
            Deduplicated list
        """
        duplicate_groups = self.find_duplicates(texts, threshold)
        
        # Collect indices to remove
        to_remove = set()
        for group in duplicate_groups:
            if keep_first:
                to_remove.update(group[1:])  # Remove all except first
            else:
                to_remove.update(group[:-1])  # Remove all except last
        
        # Build deduplicated list
        deduplicated = [
            text for i, text in enumerate(texts)
            if i not in to_remove
        ]
        
        self.logger.info(f"Removed {len(texts) - len(deduplicated)} duplicates")
        return deduplicated


class SemanticClassifier(LoggerMixin):
    """
    Classify texts using semantic similarity to category examples.
    """
    
    def __init__(self, embedding_service: EmbeddingService = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.categories = {}
    
    def add_category(self, name: str, examples: List[str]):
        """
        Add a category with example texts.
        
        Args:
            name: Category name
            examples: Example texts for this category
        """
        self.logger.info(f"Adding category '{name}' with {len(examples)} examples")
        
        # Get embeddings for examples
        embeddings = self.embedding_service.get_embeddings_batch(examples)
        
        # Store average embedding as category representation
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        
        self.categories[name] = {
            "examples": examples,
            "embedding": avg_embedding
        }
    
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify text into one of the defined categories.
        
        Args:
            text: Text to classify
        
        Returns:
            Dict with category, confidence, and all scores
        """
        if not self.categories:
            raise ValueError("No categories defined. Use add_category() first.")
        
        # Get embedding for text
        text_emb = self.embedding_service.get_embedding(text)
        
        # Calculate similarity to each category
        scores = {}
        for name, cat_data in self.categories.items():
            similarity = self.embedding_service.cosine_similarity(
                text_emb,
                cat_data["embedding"]
            )
            scores[name] = similarity
        
        # Find best match
        best_category = max(scores, key=scores.get)
        confidence = scores[best_category]
        
        return {
            "category": best_category,
            "confidence": confidence,
            "all_scores": scores
        }


# Example usage functions
def example_basic_embeddings():
    """Example: Generate and compare embeddings"""
    service = EmbeddingService()
    
    # Generate embeddings
    text1 = "Internett fungerer ikke"
    text2 = "Nettverksproblemer"
    text3 = "Faktura spørsmål"
    
    emb1 = service.get_embedding(text1)
    emb2 = service.get_embedding(text2)
    emb3 = service.get_embedding(text3)
    
    # Calculate similarities
    sim_12 = service.cosine_similarity(emb1, emb2)
    sim_13 = service.cosine_similarity(emb1, emb3)
    
    print(f"Similarity '{text1}' vs '{text2}': {sim_12:.3f}")
    print(f"Similarity '{text1}' vs '{text3}': {sim_13:.3f}")


def example_semantic_search():
    """Example: Semantic search"""
    service = EmbeddingService()
    
    query = "hvordan bytte betaling"
    
    documents = [
        "Endre betalingsmetode i Min Side under Innstillinger",
        "Oppdatere fakturainnstillinger går enkelt online",
        "Bytte fra faktura til AvtaleGiro i kundeportalen",
        "Installasjon av fiber tar vanligvis 2-3 uker",
        "Teknisk support er tilgjengelig 24/7 på telefon"
    ]
    
    results = service.find_most_similar(query, documents, top_k=3)
    
    print(f"\nQuery: '{query}'")
    print("\nTopp 3 resultater:")
    for doc, score in results:
        print(f"  {score:.3f}: {doc}")


def example_duplicate_detection():
    """Example: Detect semantic duplicates"""
    deduplicator = SemanticDeduplicator()
    
    tickets = [
        "Internett fungerer ikke",
        "Nettverket mitt er nede",
        "Ingen internettforbindelse",  # Duplicate of above
        "Kan jeg få faktura på e-post?",
        "Send faktura til min e-postadresse",  # Duplicate
        "Fiber-boksen blinker rødt",
        "Rød lampe på fiber-enheten"  # Duplicate
    ]
    
    duplicate_groups = deduplicator.find_duplicates(tickets, threshold=0.80)
    
    print("\nFunnet duplikater:")
    for group in duplicate_groups:
        print("\nGruppe:")
        for idx in group:
            print(f"  - {tickets[idx]}")
    
    # Deduplicate
    unique = deduplicator.deduplicate(tickets, threshold=0.80)
    print(f"\nOriginalt: {len(tickets)} tickets")
    print(f"Etter deduplisering: {len(unique)} tickets")


def example_semantic_classification():
    """Example: Classify using semantic similarity"""
    classifier = SemanticClassifier()
    
    # Define categories with examples
    classifier.add_category("TEKNISK", [
        "internett problemer",
        "fiber-boks defekt",
        "tregt nettverk",
        "ingen forbindelse"
    ])
    
    classifier.add_category("FAKTURERING", [
        "feil beløp",
        "krediteringskrav",
        "betalingsproblem",
        "faktura spørsmål"
    ])
    
    classifier.add_category("GENERELT", [
        "åpningstider",
        "kontaktinformasjon",
        "generelle spørsmål",
        "om selskapet"
    ])
    
    # Classify new messages
    messages = [
        "Fiber-boksen blinker rødt",
        "Når blir fakturaen sendt?",
        "Hvor finner jeg kundeservicenummeret?"
    ]
    
    print("\nKlassifisering:")
    for msg in messages:
        result = classifier.classify(msg)
        print(f"\n'{msg}'")
        print(f"  → {result['category']} (confidence: {result['confidence']:.2f})")
        print(f"  Alle scores: {result['all_scores']}")


if __name__ == "__main__":
    print("=== Example 1: Basic Embeddings ===")
    example_basic_embeddings()
    
    print("\n=== Example 2: Semantic Search ===")
    example_semantic_search()
    
    print("\n=== Example 3: Duplicate Detection ===")
    example_duplicate_detection()
    
    print("\n=== Example 4: Semantic Classification ===")
    example_semantic_classification()
