"""
Cost optimization strategies for AI applications.

Features:
- Smart model routing based on complexity
- Request batching to reduce overhead
- Cost-aware caching
"""
import time
import asyncio
import uuid
from typing import Any, Dict, List, Tuple
import threading


class CostOptimizedRouter:
    """
    Route requests to the cheapest model that can handle them.
    Analyzes prompt complexity and selects appropriate model.
    
    Example:
        router = CostOptimizedRouter()
        
        model = router.route(
            prompt="What is 2+2?",
            context_size=0
        )
        # Returns: "haiku" (cheapest sufficient model)
        
        model = router.route(
            prompt="Analyze this complex data and provide insights...",
            context_size=50000
        )
        # Returns: "opus" (most capable model needed)
    """
    
    MODELS = {
        "haiku": {
            "cost_per_token": 0.00025,
            "max_complexity": 3,
            "speed": "fast",
            "max_tokens": 4096
        },
        "sonnet": {
            "cost_per_token": 0.003,
            "max_complexity": 7,
            "speed": "medium",
            "max_tokens": 8192
        },
        "opus": {
            "cost_per_token": 0.015,
            "max_complexity": 10,
            "speed": "slow",
            "max_tokens": 16384
        }
    }
    
    COMPLEX_KEYWORDS = [
        "analyze", "compare", "synthesize", "design", "optimize",
        "evaluate", "critique", "reason", "explain in detail",
        "comprehensive", "thoroughly", "in depth"
    ]
    
    def estimate_complexity(self, prompt: str, context_size: int = 0) -> int:
        """
        Estimate query complexity (0-10).
        
        Factors:
        - Prompt length
        - Presence of complex keywords
        - Context size
        - Question complexity
        """
        score = 0
        prompt_lower = prompt.lower()
        
        # Length factor
        if len(prompt) > 1000:
            score += 2
        elif len(prompt) > 500:
            score += 1
        
        # Complex keywords
        keyword_count = sum(
            1 for kw in self.COMPLEX_KEYWORDS
            if kw in prompt_lower
        )
        score += min(keyword_count, 4)
        
        # Context size
        if context_size > 50000:
            score += 3
        elif context_size > 10000:
            score += 2
        elif context_size > 5000:
            score += 1
        
        # Multiple questions or steps
        if '?' in prompt and prompt.count('?') > 2:
            score += 1
        
        return min(score, 10)
    
    def route(
        self,
        prompt: str,
        context_size: int = 0,
        prefer_speed: bool = False
    ) -> str:
        """
        Determine best model for request.
        
        Args:
            prompt: User prompt
            context_size: Size of additional context
            prefer_speed: Prefer faster models over cost
            
        Returns:
            Model name (haiku, sonnet, opus)
        """
        complexity = self.estimate_complexity(prompt, context_size)
        
        # If speed preferred, use faster model
        if prefer_speed and complexity <= 5:
            return "haiku"
        
        # Choose cheapest model that can handle complexity
        for model_name in sorted(
            self.MODELS.keys(),
            key=lambda x: self.MODELS[x]["cost_per_token"]
        ):
            config = self.MODELS[model_name]
            if config["max_complexity"] >= complexity:
                return model_name
        
        # Fallback to most capable
        return "opus"
    
    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Calculate estimated cost"""
        if model not in self.MODELS:
            return 0.0
        
        cost_per_token = self.MODELS[model]["cost_per_token"]
        return (prompt_tokens + completion_tokens) * cost_per_token
    
    def suggest_optimization(
        self,
        prompt: str,
        current_model: str
    ) -> Dict[str, Any]:
        """Suggest cheaper model if possible"""
        optimal_model = self.route(prompt)
        
        if optimal_model == current_model:
            return {
                "can_optimize": False,
                "current_model": current_model
            }
        
        current_cost = self.MODELS[current_model]["cost_per_token"]
        optimal_cost = self.MODELS[optimal_model]["cost_per_token"]
        savings = ((current_cost - optimal_cost) / current_cost) * 100
        
        return {
            "can_optimize": True,
            "current_model": current_model,
            "suggested_model": optimal_model,
            "cost_savings_percent": savings
        }


class RequestBatcher:
    """
    Batch multiple requests to reduce API overhead.
    Particularly useful for embeddings generation.
    
    Example:
        batcher = RequestBatcher(batch_size=100, max_wait_time=1.0)
        
        # These requests will be batched together
        embedding1 = await batcher.add_request("text 1")
        embedding2 = await batcher.add_request("text 2")
        ...
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        max_wait_time: float = 1.0,
        process_func: callable = None
    ):
        """
        Initialize request batcher.
        
        Args:
            batch_size: Max requests per batch
            max_wait_time: Max time to wait before processing
            process_func: Function to process batch
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.process_func = process_func
        
        self.queue: List[Tuple[Any, str]] = []
        self.results: Dict[str, Any] = {}
        self.queue_lock = threading.Lock()
        self.processing = False
        
        print(f"Request batcher initialized: batch_size={batch_size}, max_wait={max_wait_time}s")
    
    async def add_request(self, data: Any) -> Any:
        """
        Add request to batch queue.
        
        Args:
            data: Request data
            
        Returns:
            Result from batch processing
        """
        request_id = str(uuid.uuid4())
        
        with self.queue_lock:
            self.queue.append((data, request_id))
            
            # Process if batch full
            if len(self.queue) >= self.batch_size:
                await self._process_batch()
        
        # Wait for result
        start_time = time.time()
        while request_id not in self.results:
            if time.time() - start_time > self.max_wait_time * 2:
                raise TimeoutError("Request timed out")
            await asyncio.sleep(0.01)
        
        # Get and remove result
        result = self.results.pop(request_id)
        return result
    
    async def _process_batch(self):
        """Process accumulated requests in batch"""
        if self.processing:
            return
        
        self.processing = True
        
        try:
            with self.queue_lock:
                if not self.queue:
                    return
                
                # Take batch
                batch = self.queue[:self.batch_size]
                self.queue = self.queue[self.batch_size:]
            
            # Extract data and IDs
            data_items = [item[0] for item in batch]
            request_ids = [item[1] for item in batch]
            
            # Process batch
            if self.process_func:
                results = await self.process_func(data_items)
            else:
                # Default: just return data as-is
                results = data_items
            
            # Store results
            for request_id, result in zip(request_ids, results):
                self.results[request_id] = result
        
        finally:
            self.processing = False
    
    async def flush(self):
        """Process remaining items in queue"""
        if self.queue:
            await self._process_batch()


class CostTracker:
    """
    Track AI API costs over time.
    
    Example:
        tracker = CostTracker()
        
        tracker.record_usage(
            provider="anthropic",
            model="claude-sonnet",
            prompt_tokens=100,
            completion_tokens=200
        )
        
        stats = tracker.get_stats()
        print(f"Total cost: ${stats['total_cost']:.2f}")
    """
    
    COSTS = {
        "anthropic": {
            "claude-opus": {"prompt": 0.015, "completion": 0.075},
            "claude-sonnet": {"prompt": 0.003, "completion": 0.015},
            "claude-haiku": {"prompt": 0.00025, "completion": 0.00125}
        },
        "openai": {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-3.5": {"prompt": 0.0015, "completion": 0.002}
        }
    }
    
    def __init__(self):
        self.usage_log: List[Dict[str, Any]] = []
        self.total_cost = 0.0
    
    def record_usage(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        timestamp: float = None
    ):
        """Record API usage"""
        if timestamp is None:
            timestamp = time.time()
        
        # Calculate cost
        costs = self.COSTS.get(provider, {}).get(model, {"prompt": 0, "completion": 0})
        cost = (
            prompt_tokens * costs["prompt"] +
            completion_tokens * costs["completion"]
        )
        
        # Store
        self.usage_log.append({
            "timestamp": timestamp,
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost
        })
        
        self.total_cost += cost
    
    def get_stats(
        self,
        time_window: int = None
    ) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Args:
            time_window: Only include last N seconds
            
        Returns:
            Statistics dict
        """
        logs = self.usage_log
        
        # Filter by time window
        if time_window:
            cutoff = time.time() - time_window
            logs = [log for log in logs if log["timestamp"] >= cutoff]
        
        if not logs:
            return {"total_cost": 0, "total_requests": 0}
        
        return {
            "total_cost": sum(log["cost"] for log in logs),
            "total_requests": len(logs),
            "total_prompt_tokens": sum(log["prompt_tokens"] for log in logs),
            "total_completion_tokens": sum(log["completion_tokens"] for log in logs),
            "by_provider": self._group_by(logs, "provider"),
            "by_model": self._group_by(logs, "model")
        }
    
    def _group_by(self, logs: List[Dict], key: str) -> Dict[str, Dict]:
        """Group logs by key"""
        grouped = {}
        for log in logs:
            value = log[key]
            if value not in grouped:
                grouped[value] = {"cost": 0, "requests": 0, "tokens": 0}
            
            grouped[value]["cost"] += log["cost"]
            grouped[value]["requests"] += 1
            grouped[value]["tokens"] += log["prompt_tokens"] + log["completion_tokens"]
        
        return grouped


if __name__ == "__main__":
    print("=== Cost Optimization Demo ===\n")
    
    # 1. Smart routing
    print("1. Smart Model Routing:")
    router = CostOptimizedRouter()
    
    test_cases = [
        ("What is 2+2?", 0),
        ("Explain quantum mechanics in simple terms", 100),
        ("Analyze this comprehensive dataset and provide detailed insights", 50000),
    ]
    
    for prompt, context in test_cases:
        model = router.route(prompt, context)
        complexity = router.estimate_complexity(prompt, context)
        cost = router.calculate_cost(model, 100, 200)
        
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  Complexity: {complexity}/10")
        print(f"  Selected: {model}")
        print(f"  Est. cost (300 tokens): ${cost:.4f}")
    
    print("\n" + "="*60)
    
    # 2. Cost tracking
    print("\n2. Cost Tracking:")
    tracker = CostTracker()
    
    # Simulate some usage
    tracker.record_usage("anthropic", "claude-sonnet", 100, 200)
    tracker.record_usage("anthropic", "claude-haiku", 50, 100)
    tracker.record_usage("openai", "gpt-4", 200, 300)
    
    stats = tracker.get_stats()
    print(f"\nTotal cost: ${stats['total_cost']:.4f}")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Total tokens: {stats['total_prompt_tokens'] + stats['total_completion_tokens']}")
    
    print("\nBy provider:")
    for provider, data in stats['by_provider'].items():
        print(f"  {provider}: ${data['cost']:.4f} ({data['requests']} requests)")
    
    print("\nBy model:")
    for model, data in stats['by_model'].items():
        print(f"  {model}: ${data['cost']:.4f} ({data['requests']} requests)")
