"""
Kapittel 1: Hva er moderne AI?
Basic AI client implementations and fundamental concepts.
"""
from anthropic import Anthropic
from typing import Dict, Any, Optional, List

try:
    from utils import config, logger, LoggerMixin
except ImportError:
    # For direct execution
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import config, logger, LoggerMixin


class AIClient(LoggerMixin):
    """
    Basic AI client for interacting with Claude API.
    Demonstrates fundamental AI concepts from Chapter 1.
    """
    
    def __init__(self, model: str = None):
        """
        Initialize AI client.
        
        Args:
            model: Model to use (defaults to config.ai.default_model)
        """
        self.client = Anthropic(api_key=config.ai.anthropic_api_key)
        self.model = model or config.ai.default_model
        self.logger.info(f"Initialized AIClient with model: {self.model}")
    
    def query(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Send a simple query to the AI.
        
        Args:
            prompt: The question or instruction
            max_tokens: Maximum tokens in response
        
        Returns:
            AI's response text
        """
        self.logger.debug(f"Query: {prompt[:100]}...")
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        answer = response.content[0].text
        self.logger.debug(f"Response: {answer[:100]}...")
        
        return answer
    
    def analyze_with_context(self, data: str, question: str) -> str:
        """
        Analyze data with context (demonstrates context windows).
        
        Args:
            data: Context data (e.g., long report)
            question: Question about the data
        
        Returns:
            AI's analysis
        """
        prompt = f"""Analyser følgende data og svar på spørsmålet.

DATA:
{data}

SPØRSMÅL: {question}

Gi et presist og databasert svar."""
        
        return self.query(prompt)
    
    def extract_structured_data(self, text: str, schema: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract structured data from unstructured text.
        
        Args:
            text: Unstructured text
            schema: Description of fields to extract
        
        Returns:
            Extracted structured data
        """
        import json
        import re
        
        schema_description = "\n".join([
            f"- {field}: {description}"
            for field, description in schema.items()
        ])
        
        prompt = f"""Ekstraher følgende informasjon fra teksten:

{schema_description}

TEKST:
{text}

Returner som gyldig JSON (kun JSON, ingen markdown):"""
        
        response = self.query(prompt, max_tokens=1000)
        
        # Clean up markdown if present
        json_text = re.sub(r'```json\n?|\n?```', '', response)
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.debug(f"Response was: {response}")
            return {}


class TokenCounter:
    """Utility for estimating and counting tokens."""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate number of tokens in text.
        Rule of thumb: ~4 characters = 1 token for English/Norwegian
        
        Args:
            text: Text to estimate
        
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    @staticmethod
    def truncate_to_tokens(text: str, max_tokens: int) -> str:
        """
        Truncate text to approximate token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens
        
        Returns:
            Truncated text
        """
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."


# Example usage functions for demonstration
def example_basic_query():
    """Example: Basic AI query"""
    client = AIClient()
    
    response = client.query("Hva er en vektordatabase? Svar på norsk i 2-3 setninger.")
    print("Svar:", response)


def example_context_analysis():
    """Example: Analyze data with context"""
    client = AIClient()
    
    report = """
    Q4 2024 Resultat:
    - Omsetning: 15.5M NOK (+12% YoY)
    - Nye kunder: 234 (+45% YoY)
    - Churn rate: 3.2% (ned fra 4.1%)
    - Customer lifetime value: 45,000 NOK
    """
    
    analysis = client.analyze_with_context(
        data=report,
        question="Hva er de viktigste trekkene i Q4-resultatene?"
    )
    print("Analyse:", analysis)


def example_structured_extraction():
    """Example: Extract structured data"""
    import json
    
    client = AIClient()
    
    invoice_text = """
    Faktura #2024-001
    Dato: 15. januar 2025
    Fra: Acme Consulting AS
    Til: TechCorp Norge
    
    Tjenester:
    - Konsulenttime: 50 timer @ 1500 NOK = 75,000 NOK
    - Reiseutgifter: 5,000 NOK
    
    Subtotal: 80,000 NOK
    MVA (25%): 20,000 NOK
    Total: 100,000 NOK
    """
    
    schema = {
        "invoice_number": "Fakturanummer",
        "date": "Fakturadato (YYYY-MM-DD format)",
        "supplier": "Leverandørens navn",
        "customer": "Kundens navn",
        "subtotal": "Subtotal beløp (tall)",
        "vat": "MVA beløp (tall)",
        "total": "Totalt beløp (tall)"
    }
    
    data = client.extract_structured_data(invoice_text, schema)
    print("Ekstrahert data:", json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    print("=== Example 1: Basic Query ===")
    example_basic_query()
    
    print("\n=== Example 2: Context Analysis ===")
    example_context_analysis()
    
    print("\n=== Example 3: Structured Extraction ===")
    example_structured_extraction()
