"""
Kapittel 11: Kostnadsoptimalisering
Tools for optimizing AI costs and token usage.
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from anthropic import Anthropic

try:
    from utils import config, logger, LoggerMixin
    from fundamentals.ai_basics import TokenCounter
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import config, logger, LoggerMixin
    from fundamentals.ai_basics import TokenCounter


@dataclass
class CostEstimate:
    """Cost estimate for AI operation"""
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    model: str


class CostOptimizer(LoggerMixin):
    """
    Optimize AI costs through smart model selection and token management.
    """
    
    # Model costs per 1M tokens (approximate)
    MODEL_COSTS = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-haiku-3-5-20250514": {"input": 0.8, "output": 4.0},
        "claude-opus-4-20250514": {"input": 15.0, "output": 75.0}
    }
    
    def __init__(self):
        self.total_cost = 0.0
        self.operation_history: List[Dict[str, Any]] = []
        self.log_info("Initialized cost optimizer")
    
    def estimate_cost(
        self,
        input_text: str,
        expected_output_tokens: int,
        model: str
    ) -> CostEstimate:
        """
        Estimate cost for AI operation.
        
        Args:
            input_text: Input prompt
            expected_output_tokens: Expected response length
            model: Model name
            
        Returns:
            Cost estimate
        """
        counter = TokenCounter()
        input_tokens = counter.estimate_tokens(input_text)
        
        costs = self.MODEL_COSTS.get(model, {"input": 3.0, "output": 15.0})
        
        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (expected_output_tokens / 1_000_000) * costs["output"]
        total_cost = input_cost + output_cost
        
        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=expected_output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            model=model
        )
    
    def recommend_model(
        self,
        task_complexity: str,
        max_cost: Optional[float] = None
    ) -> str:
        """
        Recommend model based on task complexity and budget.
        
        Args:
            task_complexity: 'simple', 'medium', or 'complex'
            max_cost: Maximum acceptable cost per operation
            
        Returns:
            Recommended model name
        """
        if task_complexity == "simple":
            return config.ai.fast_model
        elif task_complexity == "complex":
            return config.ai.advanced_model
        else:
            return config.ai.model
    
    def track_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str
    ):
        """Track actual usage and cost."""
        costs = self.MODEL_COSTS.get(model, {"input": 3.0, "output": 15.0})
        
        cost = (input_tokens / 1_000_000) * costs["input"] + \
               (output_tokens / 1_000_000) * costs["output"]
        
        self.total_cost += cost
        self.operation_history.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "model": model
        })
        
        self.log_info(f"Operation cost: ${cost:.4f}, Total: ${self.total_cost:.4f}")


class TokenOptimizer(LoggerMixin):
    """
    Optimize token usage through compression and summarization.
    """
    
    def __init__(self):
        self.client = Anthropic(api_key=config.ai.api_key)
        self.counter = TokenCounter()
    
    def compress_prompt(
        self,
        prompt: str,
        max_tokens: int = 1000
    ) -> str:
        """
        Compress prompt to fit within token limit.
        
        Args:
            prompt: Original prompt
            max_tokens: Maximum tokens
            
        Returns:
            Compressed prompt
        """
        current_tokens = self.counter.estimate_tokens(prompt)
        
        if current_tokens <= max_tokens:
            return prompt
        
        self.log_info(f"Compressing prompt from {current_tokens} to {max_tokens} tokens")
        
        # Use AI to compress
        compression_prompt = f"""Compress the following text to approximately {max_tokens} tokens while preserving key information:

{prompt}

Compressed version:"""
        
        response = self.client.messages.create(
            model=config.ai.fast_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": compression_prompt}]
        )
        
        return response.content[0].text.strip()
    
    def summarize_conversation(
        self,
        messages: List[Dict[str, str]],
        max_summary_tokens: int = 500
    ) -> str:
        """
        Summarize conversation history to reduce tokens.
        
        Args:
            messages: List of conversation messages
            max_summary_tokens: Maximum tokens for summary
            
        Returns:
            Conversation summary
        """
        conversation = "\n\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])
        
        prompt = f"""Summarize this conversation in {max_summary_tokens} tokens or less:

{conversation}

Summary:"""
        
        response = self.client.messages.create(
            model=config.ai.fast_model,
            max_tokens=max_summary_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        summary = response.content[0].text.strip()
        self.log_info(f"Summarized {len(messages)} messages")
        
        return summary


# Example usage
def example_cost_optimization():
    """Example: Cost optimization"""
    optimizer = CostOptimizer()
    
    # Estimate costs for different models
    prompt = "Analyze this complex financial report..." * 100
    
    for model in ["claude-haiku-3-5-20250514", "claude-sonnet-4-20250514", "claude-opus-4-20250514"]:
        estimate = optimizer.estimate_cost(prompt, expected_output_tokens=500, model=model)
        print(f"{model}:")
        print(f"  Input: {estimate.input_tokens} tokens (${estimate.input_cost:.4f})")
        print(f"  Output: {estimate.output_tokens} tokens (${estimate.output_cost:.4f})")
        print(f"  Total: ${estimate.total_cost:.4f}\n")
    
    # Get recommendation
    model = optimizer.recommend_model("simple")
    print(f"Recommended model for simple task: {model}")


def example_token_optimization():
    """Example: Token optimization"""
    optimizer = TokenOptimizer()
    
    long_text = "This is a very long document. " * 200
    
    compressed = optimizer.compress_prompt(long_text, max_tokens=100)
    print(f"Original: ~{optimizer.counter.estimate_tokens(long_text)} tokens")
    print(f"Compressed: ~{optimizer.counter.estimate_tokens(compressed)} tokens")
    print(f"Text: {compressed[:100]}...")


if __name__ == "__main__":
    print("=== Cost Optimization ===")
    example_cost_optimization()
    
    print("\n=== Token Optimization ===")
    example_token_optimization()
