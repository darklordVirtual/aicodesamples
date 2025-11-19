"""
Kapittel 4: Intelligent Support System
Example of a specialized knowledge base for support tickets using ChromaDB.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os

try:
    from vector_db.chromadb_basics import KnowledgeBase
    from utils import config, logger
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from vector_db.chromadb_basics import KnowledgeBase
    from utils import config, logger

class SupportKnowledgeBase(KnowledgeBase):
    """
    Specialized knowledge base for support tickets.
    Extends the basic KnowledgeBase with support-specific logic.
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        super().__init__(
            collection_name="support_tickets",
            persist_directory=persist_directory
        )
        from anthropic import Anthropic
        self.ai_client = Anthropic(api_key=config.ai.api_key)
    
    def add_ticket(
        self,
        ticket_id: str,
        description: str,
        category: str,
        priority: str,
        resolved: bool = False
    ) -> str:
        """Add a support ticket to the knowledge base."""
        metadata = {
            "category": category,
            "priority": priority,
            "resolved": resolved,
            "created_at": datetime.now().isoformat()
        }
        
        return self.add(
            text=description,
            metadata=metadata,
            doc_id=ticket_id
        )
    
    def find_similar_tickets(
        self,
        description: str,
        n_results: int = 5,
        only_resolved: bool = False
    ) -> List[Dict[str, Any]]:
        """Find similar tickets, optionally filtering for resolved ones."""
        where_filter = {"resolved": True} if only_resolved else None
        
        results = self.query(
            query_text=description,
            n_results=n_results,
            where=where_filter
        )
        
        similar_tickets = []
        for doc, meta, dist in zip(
            results['documents'],
            results['metadatas'],
            results['distances']
        ):
            similar_tickets.append({
                "description": doc,
                "metadata": meta,
                "similarity": 1.0 - dist  # Convert distance to similarity
            })
            
        return similar_tickets
    
    def get_resolution_suggestions(self, new_ticket_description: str) -> Optional[Dict[str, Any]]:
        """Generate resolution suggestions based on similar resolved tickets."""
        similar = self.find_similar_tickets(
            new_ticket_description,
            n_results=3,
            only_resolved=True
        )
        
        if not similar or similar[0]['similarity'] < 0.7:
            return None
        
        # Use AI to generate suggestions based on similar resolved cases
        context = "\n\n".join([
            f"Lignende sak {i+1} (likhet: {s['similarity']:.2f}):\n{s['description']}"
            for i, s in enumerate(similar)
        ])
        
        prompt = f"""
Basert på disse tidligere løste sakene:

{context}

Gi anbefalinger for hvordan løse denne nye saken:
{new_ticket_description}

Gi konkrete steg som support-medarbeider kan følge.
"""

        response = self.ai_client.messages.create(
            model=config.ai.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "suggestions": response.content[0].text,
            "based_on": similar
        }
    
    def analyze_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze trends in support tickets over the last N days."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Note: ChromaDB's get() with where clause is limited, 
        # in a real app we might want to use a separate SQL db for metadata
        # or fetch all and filter (inefficient for large datasets)
        
        # For this example, we'll fetch recent tickets (limit to 100)
        # In production, use a proper metadata store
        result = self.collection.get(
            where={"created_at": {"$gte": cutoff_date}},
            limit=100
        )
        
        if not result['ids']:
            return {"period_days": days, "total_tickets": 0, "by_category": {}}
            
        # Group by category
        categories = {}
        for meta in result['metadatas']:
            cat = meta.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            
        return {
            "period_days": days,
            "total_tickets": len(result['ids']),
            "by_category": categories
        }

# Example usage
if __name__ == "__main__":
    kb = SupportKnowledgeBase()
    
    # Add tickets
    kb.add_ticket(
        "TICKET-001",
        "Fiber-boksen blinker rødt, ingen internett",
        "teknisk",
        "høy"
    )
    
    kb.add_ticket(
        "TICKET-002",
        "Fiberboks rødt lys, koble fra og til fikset det",
        "teknisk",
        "høy",
        resolved=True
    )
    
    # New ticket
    new_ticket = "Fiber-enheten har rødt blinkende lys"
    
    # Get suggestions
    suggestions = kb.get_resolution_suggestions(new_ticket)
    
    if suggestions:
        print("Anbefalte løsningssteg:")
        print(suggestions['suggestions'])
        print("\nBasert på lignende saker:")
        for ticket in suggestions['based_on']:
            print(f"  - Likhet: {ticket['similarity']:.2f}")
    
    kb.clear()
