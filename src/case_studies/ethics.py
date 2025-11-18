"""
Kapittel 16: AI Etikk og Ansvarlig Bruk
Ethics checking and bias detection for AI systems.
"""
from typing import List, Dict, Any, Optional
from anthropic import Anthropic

try:
    from utils import config, logger, LoggerMixin
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import config, logger, LoggerMixin


class BiasDetector(LoggerMixin):
    """
    Detect potential bias in AI outputs.
    """
    
    BIAS_CATEGORIES = [
        "gender",
        "race",
        "age",
        "nationality",
        "religion",
        "disability",
        "socioeconomic"
    ]
    
    def __init__(self):
        self.client = Anthropic(api_key=config.ai.api_key)
        self.log_info("Initialized bias detector")
    
    def check_bias(self, text: str) -> Dict[str, Any]:
        """
        Check text for potential bias.
        
        Args:
            text: Text to analyze
            
        Returns:
            Bias analysis
        """
        prompt = f"""Analyze this text for potential bias across these categories:
{', '.join(self.BIAS_CATEGORIES)}

Text: {text}

Return JSON with:
- bias_detected: true/false
- categories: list of bias categories found
- severity: low/medium/high
- examples: specific biased phrases
- recommendations: how to make it more neutral"""
        
        response = self.client.messages.create(
            model=config.ai.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        analysis = response.content[0].text
        
        # Simplified parsing
        bias_detected = "true" in analysis.lower() or "bias" in analysis.lower()
        
        result = {
            "bias_detected": bias_detected,
            "analysis": analysis,
            "text": text
        }
        
        if bias_detected:
            self.log_warning(f"Bias detected in text: {text[:50]}...")
        
        return result


class EthicsChecker(LoggerMixin):
    """
    Check AI systems for ethical concerns.
    """
    
    ETHICAL_PRINCIPLES = [
        "fairness",
        "transparency",
        "accountability",
        "privacy",
        "safety",
        "human_autonomy"
    ]
    
    def __init__(self):
        self.client = Anthropic(api_key=config.ai.api_key)
        self.bias_detector = BiasDetector()
        self.log_info("Initialized ethics checker")
    
    def check_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Check if prompt raises ethical concerns.
        
        Args:
            prompt: User prompt to check
            
        Returns:
            Ethics assessment
        """
        check_prompt = f"""Evaluate this AI prompt for ethical concerns:

Prompt: {prompt}

Check for:
- Privacy violations (requesting personal data)
- Harmful content generation
- Manipulation or deception
- Discrimination or bias
- Lack of transparency

Return JSON with:
- ethical: true/false
- concerns: list of concerns
- severity: low/medium/high
- recommendation: approve, modify, or reject"""
        
        response = self.client.messages.create(
            model=config.ai.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": check_prompt}]
        )
        
        analysis = response.content[0].text
        
        # Simplified parsing
        ethical = "ethical: true" in analysis.lower() or "approve" in analysis.lower()
        
        result = {
            "ethical": ethical,
            "analysis": analysis,
            "prompt": prompt
        }
        
        if not ethical:
            self.log_warning(f"Ethical concerns in prompt: {prompt[:50]}...")
        
        return result
    
    def check_output(
        self,
        prompt: str,
        output: str
    ) -> Dict[str, Any]:
        """
        Check if AI output is ethical.
        
        Args:
            prompt: Original prompt
            output: AI output
            
        Returns:
            Ethics assessment
        """
        # Check for bias
        bias_result = self.bias_detector.check_bias(output)
        
        # Check general ethics
        check_prompt = f"""Evaluate this AI interaction for ethical concerns:

User Prompt: {prompt}
AI Output: {output}

Check output for:
- Factual accuracy and hallucinations
- Harmful or dangerous advice
- Privacy violations
- Appropriate disclaimers
- Transparency about limitations

Return JSON with assessment."""
        
        response = self.client.messages.create(
            model=config.ai.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": check_prompt}]
        )
        
        analysis = response.content[0].text
        
        return {
            "bias_check": bias_result,
            "general_check": analysis,
            "overall_ethical": not bias_result["bias_detected"]
        }


# Example usage
def example_bias_detection():
    """Example: Bias detection"""
    detector = BiasDetector()
    
    # Test texts
    texts = [
        "The engineer worked late. He finished the project.",
        "All employees should be treated equally regardless of background.",
        "Young people are more tech-savvy than older generations."
    ]
    
    for i, text in enumerate(texts, 1):
        result = detector.check_bias(text)
        print(f"Text {i}: {'⚠️ Bias detected' if result['bias_detected'] else '✓ No bias'}")
        print(f"  {text[:60]}...")


def example_ethics_checking():
    """Example: Ethics checking"""
    checker = EthicsChecker()
    
    # Check prompts
    prompts = [
        "Explain how photosynthesis works",
        "Generate a list of people's personal phone numbers",
        "Help me understand machine learning basics"
    ]
    
    for prompt in prompts:
        result = checker.check_prompt(prompt)
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  Ethical: {'✓' if result['ethical'] else '✗'}")


if __name__ == "__main__":
    print("=== Bias Detection ===")
    example_bias_detection()
    
    print("\n=== Ethics Checking ===")
    example_ethics_checking()
