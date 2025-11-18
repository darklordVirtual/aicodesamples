"""
Kapittel 14: AI Kundesupport
AI-powered customer support system with RAG and conversation management.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

try:
    from utils import config, logger, LoggerMixin
    from fundamentals.ai_basics import AIClient
    from integrations.rag_system import RAGSystem
    from vector_db.chromadb_basics import KnowledgeBase
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import config, logger, LoggerMixin
    from fundamentals.ai_basics import AIClient
    from integrations.rag_system import RAGSystem
    from vector_db.chromadb_basics import KnowledgeBase


@dataclass
class SupportTicket:
    """Support ticket data"""
    ticket_id: str
    customer_name: str
    subject: str
    message: str
    category: Optional[str] = None
    priority: Optional[str] = None
    status: str = "open"
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class CustomerSupportBot(LoggerMixin):
    """
    AI-powered customer support bot.
    
    Features:
    - RAG-based question answering
    - Ticket classification
    - Priority detection
    - Multi-turn conversations
    """
    
    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        self.ai_client = AIClient()
        
        if knowledge_base:
            self.rag = RAGSystem(knowledge_base)
        else:
            kb = KnowledgeBase(collection_name="support_kb")
            self.rag = RAGSystem(kb)
        
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}
        self.log_info("Initialized customer support bot")
    
    def classify_ticket(self, ticket: SupportTicket) -> Dict[str, str]:
        """
        Classify ticket category and priority.
        
        Args:
            ticket: Support ticket
            
        Returns:
            Classification result
        """
        prompt = f"""Classify this support ticket:

Subject: {ticket.subject}
Message: {ticket.message}

Return JSON with:
- category: technical, billing, general, or feedback
- priority: low, medium, or high
- reasoning: brief explanation"""
        
        result = self.ai_client.query(prompt)
        
        # Parse result (simplified)
        category = "general"
        priority = "medium"
        
        if "technical" in result.lower():
            category = "technical"
        elif "billing" in result.lower():
            category = "billing"
        elif "feedback" in result.lower():
            category = "feedback"
        
        if "high" in result.lower() or "urgent" in result.lower():
            priority = "high"
        elif "low" in result.lower():
            priority = "low"
        
        self.log_info(f"Classified ticket {ticket.ticket_id}: {category}/{priority}")
        
        return {
            "category": category,
            "priority": priority,
            "reasoning": result
        }
    
    def answer_question(
        self,
        question: str,
        customer_id: Optional[str] = None
    ) -> str:
        """
        Answer customer question using RAG.
        
        Args:
            question: Customer question
            customer_id: Optional customer ID for conversation history
            
        Returns:
            Answer
        """
        # Get RAG answer
        result = self.rag.query(question, n_results=3)
        
        # Add to conversation history
        if customer_id:
            if customer_id not in self.conversation_history:
                self.conversation_history[customer_id] = []
            
            self.conversation_history[customer_id].append({
                "role": "user",
                "content": question
            })
            self.conversation_history[customer_id].append({
                "role": "assistant",
                "content": result.answer
            })
        
        return result.answer
    
    def load_knowledge_base(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """Load documents into knowledge base."""
        self.rag.add_documents(documents, metadatas)
        self.log_info(f"Loaded {len(documents)} documents into knowledge base")


# Example usage
def example_support_bot():
    """Example: Customer support bot"""
    # Create and load knowledge base
    bot = CustomerSupportBot()
    
    docs = [
        "For å tilbakestille passordet ditt, gå til innstillinger og klikk 'Glemt passord'.",
        "Våre åpningstider er mandag-fredag 09:00-17:00.",
        "Du kan endre betalingsmetode under 'Min konto' -> 'Betaling'.",
        "For teknisk support, kontakt support@firma.no eller ring 12345678.",
        "Refusjon behandles innen 5-7 virkedager etter mottatt retur."
    ]
    
    bot.load_knowledge_base(docs)
    
    # Create ticket
    ticket = SupportTicket(
        ticket_id="T-001",
        customer_name="Ola Nordmann",
        subject="Kan ikke logge inn",
        message="Jeg har glemt passordet mitt og får ikke logget inn."
    )
    
    # Classify ticket
    classification = bot.classify_ticket(ticket)
    print(f"Ticket {ticket.ticket_id}:")
    print(f"  Category: {classification['category']}")
    print(f"  Priority: {classification['priority']}")
    
    # Answer question
    answer = bot.answer_question(
        "Hvordan tilbakestiller jeg passordet?",
        customer_id="customer_123"
    )
    print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    print("=== Customer Support Bot ===")
    example_support_bot()
