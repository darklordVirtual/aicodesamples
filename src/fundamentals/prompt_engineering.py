"""
Kapittel 2: Prompt Engineering
Techniques and patterns for effective prompt engineering.
"""
from anthropic import Anthropic
from typing import List, Dict, Any, Optional

try:
    from utils import config, logger, LoggerMixin
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import config, logger, LoggerMixin


class PromptEngineer(LoggerMixin):
    """
    Advanced prompt engineering techniques from Chapter 2.
    """
    
    def __init__(self, model: str = None):
        self.client = Anthropic(api_key=config.ai.anthropic_api_key)
        self.model = model or config.ai.default_model
    
    def few_shot_classification(
        self,
        examples: List[Dict[str, str]],
        new_input: str
    ) -> Dict[str, Any]:
        """
        Few-shot learning classification.
        
        Args:
            examples: List of example inputs and outputs
            new_input: New input to classify
        
        Returns:
            Classification result
        """
        import json
        import re
        
        # Build few-shot prompt
        examples_text = "\n\n".join([
            f"Input: {ex['input']}\nOutput: {ex['output']}"
            for ex in examples
        ])
        
        prompt = f"""Klassifiser kundehenvendelser basert på disse eksemplene:

{examples_text}

Klassifiser denne nye henvendelsen:
Input: {new_input}
Output:"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        output = response.content[0].text.strip()
        
        # Try to parse as JSON if possible
        try:
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {"classification": output}
    
    def chain_of_thought(self, problem: str) -> Dict[str, str]:
        """
        Chain-of-thought reasoning for complex problems.
        
        Args:
            problem: Problem to solve
        
        Returns:
            Dict with reasoning and conclusion
        """
        prompt = f"""Løs dette problemet steg-for-steg:

{problem}

Tenk høyt gjennom problemet:
1. Hva er situasjonen?
2. Hvilke faktorer er relevante?
3. Hva er mulige løsninger?
4. Hva er den beste løsningen og hvorfor?

Gi deretter en konklusjon."""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        full_response = response.content[0].text
        
        # Try to separate reasoning from conclusion
        parts = full_response.split("Konklusjon", 1)
        if len(parts) == 2:
            return {
                "reasoning": parts[0].strip(),
                "conclusion": parts[1].strip()
            }
        
        return {
            "reasoning": full_response,
            "conclusion": ""
        }
    
    def role_play(self, role: str, context: str, task: str) -> str:
        """
        Role-based prompting for specialized expertise.
        
        Args:
            role: The role/expertise to assume
            context: Background context
            task: Task to perform
        
        Returns:
            Role-based response
        """
        prompt = f"""Du er en {role}.

KONTEKST:
{context}

OPPGAVE:
{task}

Utfør oppgaven basert på din ekspertise som {role}."""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def structured_output(
        self,
        data: str,
        output_format: str,
        instructions: str = ""
    ) -> str:
        """
        Generate structured output in specific format.
        
        Args:
            data: Input data
            output_format: Desired output format (JSON, XML, CSV, etc.)
            instructions: Additional instructions
        
        Returns:
            Structured output
        """
        prompt = f"""Prosesser følgende data og returner i {output_format} format.

{instructions}

DATA:
{data}

VIKTIG: Returner KUN {output_format}, ingen annen tekst eller forklaring."""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def iterative_refinement(
        self,
        initial_prompt: str,
        refinements: List[str],
        max_iterations: int = 3
    ) -> List[str]:
        """
        Iteratively refine output through multiple rounds.
        
        Args:
            initial_prompt: Starting prompt
            refinements: List of refinement requests
            max_iterations: Maximum iterations
        
        Returns:
            List of outputs at each iteration
        """
        outputs = []
        conversation = []
        
        # Initial generation
        conversation.append({
            "role": "user",
            "content": initial_prompt
        })
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=conversation
        )
        
        initial_output = response.content[0].text
        outputs.append(initial_output)
        conversation.append({
            "role": "assistant",
            "content": initial_output
        })
        
        # Refinement iterations
        for i, refinement in enumerate(refinements[:max_iterations]):
            conversation.append({
                "role": "user",
                "content": refinement
            })
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=conversation
            )
            
            refined_output = response.content[0].text
            outputs.append(refined_output)
            conversation.append({
                "role": "assistant",
                "content": refined_output
            })
        
        return outputs


# Example usage functions
def example_few_shot_classification():
    """Example: Few-shot classification"""
    engineer = PromptEngineer()
    
    examples = [
        {
            "input": "Internett fungerer ikke, helt nede siden i går",
            "output": '{"kategori": "TEKNISK", "prioritet": "HØY", "sentiment": "FRUSTRERT"}'
        },
        {
            "input": "Kan jeg få faktura på e-post?",
            "output": '{"kategori": "SERVICE", "prioritet": "LAV", "sentiment": "NØYTRAL"}'
        },
        {
            "input": "Dere har fakturert meg feil i 3 måneder og ingen svarer!",
            "output": '{"kategori": "KLAGE", "prioritet": "HØY", "sentiment": "SINT"}'
        }
    ]
    
    new_message = "Fiber-boksen blinker rødt, får ikke kontakt"
    result = engineer.few_shot_classification(examples, new_message)
    print("Klassifisering:", result)


def example_chain_of_thought():
    """Example: Chain-of-thought reasoning"""
    engineer = PromptEngineer()
    
    problem = """
    En kunde har følgende betalingsmønster de siste 6 månedene:
    - Måned 1: Betalt 2 dager for sent
    - Måned 2: Betalt 5 dager for sent
    - Måned 3: Betalt 8 dager for sent
    - Måned 4: Betalt 12 dager for sent
    - Måned 5: Betalt 15 dager for sent
    - Måned 6: Ikke betalt (30 dager over forfall)
    
    Vurder kredittrisikoen og anbefal handling.
    """
    
    result = engineer.chain_of_thought(problem)
    print("Resonnering:", result['reasoning'])
    print("\nKonklusjon:", result['conclusion'])


def example_role_play():
    """Example: Role-based prompting"""
    engineer = PromptEngineer()
    
    analysis = engineer.role_play(
        role="senior nettverksingeniør med 20 års erfaring i fiber-infrastruktur",
        context="Kunde rapporterer ustabil forbindelse om kvelden. Hastighet går fra 500 Mbit til 20 Mbit mellom 18:00-22:00.",
        task="Diagnose problemet og foreslå løsning."
    )
    print("Teknisk analyse:", analysis)


def example_structured_output():
    """Example: Structured output generation"""
    engineer = PromptEngineer()
    
    customer_data = """
    Kunde: TechCorp AS
    Kontakt: 98765432
    E-post: post@techcorp.no
    Adresse: Storgata 123, 0180 Oslo
    Org.nr: 987654321
    Kundenummer: C-12345
    Opprettet: 2023-05-15
    Abonnement: Fiber 1000/1000
    Månedspris: 599 kr
    """
    
    json_output = engineer.structured_output(
        data=customer_data,
        output_format="JSON",
        instructions="Ekstraher all relevant kundeinformasjon i strukturert format."
    )
    print("Strukturert output:", json_output)


def example_iterative_refinement():
    """Example: Iterative refinement"""
    engineer = PromptEngineer()
    
    outputs = engineer.iterative_refinement(
        initial_prompt="Skriv en kort e-post til kunde om planlagt vedlikehold.",
        refinements=[
            "Gjør tonen mer personlig og mindre formell.",
            "Legg til informasjon om kompensasjon hvis kunden opplever lengre nedetid enn varslet."
        ]
    )
    
    for i, output in enumerate(outputs):
        print(f"\n=== Versjon {i+1} ===")
        print(output)


if __name__ == "__main__":
    print("=== Example 1: Few-Shot Classification ===")
    example_few_shot_classification()
    
    print("\n=== Example 2: Chain-of-Thought ===")
    example_chain_of_thought()
    
    print("\n=== Example 3: Role Play ===")
    example_role_play()
    
    print("\n=== Example 4: Structured Output ===")
    example_structured_output()
    
    print("\n=== Example 5: Iterative Refinement ===")
    example_iterative_refinement()
