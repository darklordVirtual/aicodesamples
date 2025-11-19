"""
Kapittel 10 & 10.5: Production Utilities and Scaling Components
Retry logic, rate limiting, caching, monitoring, circuit breakers, and health checks
for production AI systems at scale.
"""
import time
import functools
from typing import Any, Callable, Optional, Dict
from collections import OrderedDict
from datetime import datetime, timedelta
import hashlib
import json
from enum import Enum
import threading

try:
    from utils import logger, LoggerMixin
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import logger, LoggerMixin


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for each retry
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries} retries failed")
            
            raise last_exception
        
        return wrapper
    return decorator


class RateLimiter(LoggerMixin):
    """
    Token bucket rate limiter for API calls.
    """
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.tokens = calls_per_minute
        self.last_update = time.time()
        self.log_info(f"Initialized rate limiter: {calls_per_minute} calls/min")
    
    def acquire(self, block: bool = True) -> bool:
        """
        Acquire permission to make API call.
        
        Args:
            block: If True, wait for permission. If False, return immediately.
            
        Returns:
            True if permission granted
        """
        while True:
            now = time.time()
            elapsed = now - self.last_update
            
            # Replenish tokens
            self.tokens = min(
                self.calls_per_minute,
                self.tokens + (elapsed * self.calls_per_minute / 60.0)
            )
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            
            if not block:
                return False
            
            # Wait for next token
            wait_time = (1 - self.tokens) * 60.0 / self.calls_per_minute
            self.log_debug(f"Rate limit reached, waiting {wait_time:.2f}s")
            time.sleep(wait_time)


class ResponseCache(LoggerMixin):
    """
    LRU cache for API responses.
    """
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.log_info(f"Initialized cache: max_size={max_size}, ttl={ttl_seconds}s")
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            
            # Check if expired
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.log_debug(f"Cache hit: {key[:8]}")
                return value
            else:
                # Expired
                del self.cache[key]
                self.log_debug(f"Cache expired: {key[:8]}")
        
        return None
    
    def set(self, key: str, value: Any):
        """Store value in cache."""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.log_debug(f"Cache eviction: {oldest_key[:8]}")
        
        self.cache[key] = (value, datetime.now())
        self.log_debug(f"Cache set: {key[:8]}")
    
    def clear(self):
        """Clear entire cache."""
        self.cache.clear()
        self.log_info("Cache cleared")


class MonitoredSystem(LoggerMixin):
    """
    Wrapper for monitoring AI system performance.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_latency": 0.0,
            "min_latency": float('inf'),
            "max_latency": 0.0
        }
        self.log_info(f"Initialized monitoring for: {name}")
    
    def track_call(self, func: Callable) -> Callable:
        """Decorator to track function calls."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            self.metrics["total_calls"] += 1
            
            try:
                result = func(*args, **kwargs)
                self.metrics["successful_calls"] += 1
                return result
            except Exception as e:
                self.metrics["failed_calls"] += 1
                raise e
            finally:
                latency = time.time() - start_time
                self.metrics["total_latency"] += latency
                self.metrics["min_latency"] = min(self.metrics["min_latency"], latency)
                self.metrics["max_latency"] = max(self.metrics["max_latency"], latency)
        
        return wrapper
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        total = self.metrics["total_calls"]
        if total == 0:
            return {"message": "No calls recorded"}
        
        return {
            "name": self.name,
            "total_calls": total,
            "successful_calls": self.metrics["successful_calls"],
            "failed_calls": self.metrics["failed_calls"],
            "success_rate": self.metrics["successful_calls"] / total,
            "avg_latency": self.metrics["total_latency"] / total,
            "min_latency": self.metrics["min_latency"],
            "max_latency": self.metrics["max_latency"]
        }


# Example usage
def example_retry():
    """Example: Retry with backoff"""
    @retry_with_backoff(max_retries=3, initial_delay=0.5)
    def unreliable_function():
        import random
        if random.random() < 0.7:
            raise Exception("Random failure")
        return "Success!"
    
    try:
        result = unreliable_function()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed after retries: {e}")


def example_rate_limiter():
    """Example: Rate limiting"""
    limiter = RateLimiter(calls_per_minute=10)
    
    print("Making 15 API calls with rate limiting...")
    for i in range(15):
        limiter.acquire()
        print(f"  Call {i+1} completed")


def example_cache():
    """Example: Response caching"""
    cache = ResponseCache(max_size=3, ttl_seconds=5)
    
    def expensive_function(x: int) -> int:
        print(f"  Computing {x}**2...")
        time.sleep(0.5)
        return x ** 2
    
    # First calls - cache misses
    for i in range(3):
        key = cache._make_key(i)
        result = cache.get(key)
        if result is None:
            result = expensive_function(i)
            cache.set(key, result)
        print(f"Result: {result}")
    
    # Second calls - cache hits
    print("\nSecond round (should use cache):")
    for i in range(3):
        key = cache._make_key(i)
        result = cache.get(key)
        if result is None:
            result = expensive_function(i)
            cache.set(key, result)
        print(f"Result: {result}")


def example_monitoring():
    """Example: System monitoring"""
    monitor = MonitoredSystem("test_system")
    
    @monitor.track_call
    def api_call(success: bool = True):
        time.sleep(0.1)
        if not success:
            raise Exception("API error")
        return "OK"
    
    # Make some calls
    for _ in range(5):
        try:
            api_call(success=True)
        except:
            pass
    
    for _ in range(2):
        try:
            api_call(success=False)
        except:
            pass
    
    # Print stats
    stats = monitor.get_stats()
    print("\nSystem statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    print("=== Retry with Backoff ===")
    example_retry()
    
    print("\n=== Rate Limiting ===")
    example_rate_limiter()
    
    print("\n=== Response Caching ===")
    example_cache()
    
    print("\n=== Monitoring ===")
    example_monitoring()


# ============================================================================
# SCALING COMPONENTS (Kapittel 10.5)
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker(LoggerMixin):
    """
    Circuit breaker for external services.
    Prevents cascading failures by stopping requests to failing services.
    
    Example:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        result = breaker.call(lambda: external_api_call())
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
        self.lock = threading.Lock()
        
        self.log_info(
            f"Circuit breaker initialized: threshold={failure_threshold}, "
            f"timeout={recovery_timeout}s"
        )
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.log_info("Circuit breaker: HALF_OPEN, attempting reset")
                else:
                    raise Exception(f"Circuit breaker is OPEN (since {self.last_failure_time})")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Reset on successful call"""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.log_info("Circuit breaker: CLOSED (recovered)")
            self.failure_count = 0
            self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failure"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.log_error(
                    f"Circuit breaker: OPEN (failures: {self.failure_count})"
                )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to try recovery"""
        if not self.last_failure_time:
            return False
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class HealthChecker(LoggerMixin):
    """
    Comprehensive health checker for microservices.
    Checks Redis, database, disk, memory, etc.
    
    Example:
        checker = HealthChecker()
        checker.add_check("redis", lambda: redis_client.ping())
        status = checker.check_health()
    """
    
    def __init__(self, cache_ttl: int = 10):
        self.checks: Dict[str, Callable] = {}
        self.cache_ttl = cache_ttl
        self.last_check_time: Optional[datetime] = None
        self.cached_status: Optional[Dict] = None
        self.lock = threading.Lock()
    
    def add_check(self, name: str, check_func: Callable[[], bool]):
        """Add a health check"""
        self.checks[name] = check_func
        self.log_info(f"Added health check: {name}")
    
    def check_health(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Perform all health checks.
        
        Args:
            use_cache: Use cached results if available
            
        Returns:
            Health status dict with individual check results
        """
        with self.lock:
            # Use cache if recent enough
            if use_cache and self.cached_status and self.last_check_time:
                elapsed = (datetime.now() - self.last_check_time).total_seconds()
                if elapsed < self.cache_ttl:
                    return self.cached_status
            
            # Run all checks
            results = {}
            for name, check_func in self.checks.items():
                try:
                    results[name] = check_func()
                except Exception as e:
                    self.log_error(f"Health check '{name}' failed: {e}")
                    results[name] = False
            
            # Determine overall status
            all_healthy = all(results.values())
            
            status = {
                "status": "healthy" if all_healthy else "unhealthy",
                "checks": results,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache results
            self.cached_status = status
            self.last_check_time = datetime.now()
            
            return status
    
    def is_healthy(self) -> bool:
        """Quick health check - returns boolean"""
        status = self.check_health()
        return status["status"] == "healthy"


class TokenAwareCache(ResponseCache):
    """
    Enhanced cache that adjusts TTL based on token costs.
    More expensive queries are cached longer.
    
    Example:
        cache = TokenAwareCache()
        cache.set_with_cost(key, result, model="claude-opus", tokens=1000)
    """
    
    COSTS_PER_TOKEN = {
        "claude-opus": 0.015,
        "claude-sonnet": 0.003,
        "claude-haiku": 0.00025,
        "gpt-4": 0.03,
        "gpt-3.5": 0.0015
    }
    
    def calculate_ttl(self, model: str, tokens: int) -> int:
        """Calculate TTL based on cost"""
        cost_per_token = self.COSTS_PER_TOKEN.get(model, 0.003)
        total_cost = cost_per_token * tokens
        
        # More expensive = longer cache
        if total_cost > 1.0:      # >$1
            return 86400 * 7        # 7 days
        elif total_cost > 0.1:    # >$0.10
            return 86400            # 1 day
        elif total_cost > 0.01:   # >$0.01
            return 3600             # 1 hour
        else:
            return 300              # 5 minutes
    
    def set_with_cost(self, key: str, value: Any, model: str, tokens: int):
        """Store value with cost-aware TTL"""
        ttl = self.calculate_ttl(model, tokens)
        
        # Temporarily override instance TTL
        original_ttl = self.ttl_seconds
        self.ttl_seconds = ttl
        
        self.set(key, value)
        
        # Restore original TTL
        self.ttl_seconds = original_ttl
        
        self.log_info(
            f"Cached with cost-aware TTL: {ttl}s (model={model}, tokens={tokens})"
        )


class CostOptimizedRouter(LoggerMixin):
    """
    Route requests to the cheapest model that can handle them.
    Estimates query complexity and selects appropriate model.
    
    Example:
        router = CostOptimizedRouter()
        model = router.route(prompt="Simple question", context_size=100)
    """
    
    MODELS = {
        "haiku": {
            "cost_per_token": 0.00025,
            "max_complexity": 3,
            "speed": "fast"
        },
        "sonnet": {
            "cost_per_token": 0.003,
            "max_complexity": 7,
            "speed": "medium"
        },
        "opus": {
            "cost_per_token": 0.015,
            "max_complexity": 10,
            "speed": "slow"
        }
    }
    
    COMPLEX_KEYWORDS = [
        "analyze", "compare", "synthesize", "design",
        "optimize", "evaluate", "critique", "reason",
        "explain in detail", "comprehensive"
    ]
    
    def estimate_complexity(self, prompt: str, context_size: int = 0) -> int:
        """
        Estimate query complexity on scale 0-10.
        
        Factors:
        - Prompt length
        - Presence of complex keywords
        - Context size
        """
        score = 0
        prompt_lower = prompt.lower()
        
        # Length factor
        if len(prompt) > 1000:
            score += 2
        elif len(prompt) > 500:
            score += 1
        
        # Complexity keywords
        keyword_count = sum(1 for kw in self.COMPLEX_KEYWORDS if kw in prompt_lower)
        score += min(keyword_count, 4)
        
        # Context size
        if context_size > 50000:
            score += 3
        elif context_size > 10000:
            score += 2
        elif context_size > 5000:
            score += 1
        
        return min(score, 10)
    
    def route(self, prompt: str, context_size: int = 0) -> str:
        """
        Determine best model for request.
        
        Returns:
            Model name (haiku, sonnet, opus)
        """
        complexity = self.estimate_complexity(prompt, context_size)
        
        # Choose cheapest model that can handle it
        for model_name, config in sorted(
            self.MODELS.items(),
            key=lambda x: x[1]["cost_per_token"]
        ):
            if config["max_complexity"] >= complexity:
                self.log_info(
                    f"Routed to {model_name} (complexity={complexity}, "
                    f"cost={config['cost_per_token']}/token)"
                )
                return model_name
        
        # Fallback to most capable
        self.log_warning(
            f"Using opus for high complexity query (complexity={complexity})"
        )
        return "opus"
    
    def calculate_cost(self, model: str, tokens: int) -> float:
        """Calculate estimated cost for request"""
        if model not in self.MODELS:
            return 0.0
        
        return self.MODELS[model]["cost_per_token"] * tokens


def example_circuit_breaker():
    """Example: Circuit breaker pattern"""
    import random
    
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5)
    
    def unreliable_service():
        """Simulates an unreliable external service"""
        if random.random() < 0.7:  # 70% failure rate
            raise Exception("Service unavailable")
        return "Success"
    
    print("Making 10 calls through circuit breaker...")
    for i in range(10):
        try:
            result = breaker.call(unreliable_service)
            print(f"  Call {i+1}: {result}")
        except Exception as e:
            print(f"  Call {i+1}: Failed - {e}")
        
        time.sleep(0.5)
    
    print(f"\nFinal state: {breaker.get_state()}")


def example_health_checker():
    """Example: Health checking"""
    import random
    
    checker = HealthChecker(cache_ttl=5)
    
    # Add various checks
    checker.add_check("redis", lambda: random.random() > 0.1)  # 90% success
    checker.add_check("database", lambda: random.random() > 0.05)  # 95% success
    checker.add_check("disk_space", lambda: True)  # Always pass
    
    # Check health multiple times
    for i in range(3):
        print(f"\nHealth check #{i+1}:")
        status = checker.check_health(use_cache=(i > 0))
        print(f"  Status: {status['status']}")
        print(f"  Checks: {status['checks']}")
        time.sleep(2)


def example_cost_optimization():
    """Example: Cost-optimized routing"""
    router = CostOptimizedRouter()
    
    test_cases = [
        ("What is 2+2?", 0),
        ("Explain quantum mechanics in simple terms", 100),
        ("Analyze this 50-page document and provide comprehensive insights", 50000),
    ]
    
    print("Cost-optimized routing:")
    for prompt, context_size in test_cases:
        model = router.route(prompt, context_size)
        complexity = router.estimate_complexity(prompt, context_size)
        cost = router.calculate_cost(model, 1000)  # Assume 1000 tokens
        
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  Complexity: {complexity}/10")
        print(f"  Selected model: {model}")
        print(f"  Est. cost (1000 tokens): ${cost:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("BASIC COMPONENTS")
    print("=" * 60)
    
    print("\n=== Retry with Backoff ===")
    example_retry()
    
    print("\n=== Rate Limiting ===")
    example_rate_limiter()
    
    print("\n=== Response Caching ===")
    example_cache()
    
    print("\n=== Monitoring ===")
    example_monitoring()
    
    print("\n" + "=" * 60)
    print("SCALING COMPONENTS (Kapittel 10.5)")
    print("=" * 60)
    
    print("\n=== Circuit Breaker ===")
    example_circuit_breaker()
    
    print("\n=== Health Checker ===")
    example_health_checker()
    
    print("\n=== Cost Optimization ===")
    example_cost_optimization()
