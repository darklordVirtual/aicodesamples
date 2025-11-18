"""
Kapittel 10: Production Utilities
Retry logic, rate limiting, caching, and monitoring for production AI systems.
"""
import time
import functools
from typing import Any, Callable, Optional, Dict
from collections import OrderedDict
from datetime import datetime, timedelta
import hashlib
import json

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
