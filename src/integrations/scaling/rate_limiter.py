"""
Sliding window rate limiter with Redis for distributed rate limiting.

Features:
- Accurate sliding window algorithm
- Distributed across multiple servers via Redis
- Per-user and per-endpoint rate limits
- Different tiers (free, pro, enterprise)
"""
import time
from datetime import datetime
from typing import Optional, Dict, Tuple
from functools import wraps


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter using Redis sorted sets.
    More accurate than fixed window, fair across time boundaries.
    
    Example:
        limiter = SlidingWindowRateLimiter(redis_client)
        
        allowed, info = limiter.allow(
            key="user:123:chat",
            max_requests=100,
            window_seconds=3600
        )
        
        if not allowed:
            raise RateLimitExceeded(f"Try again at {info['reset_at']}")
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def allow(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed under rate limit.
        
        Args:
            key: Unique identifier (e.g., "user:123:api")
            max_requests: Maximum requests in window
            window_seconds: Time window in seconds
            
        Returns:
            (allowed: bool, info: dict with limit details)
        """
        now = datetime.utcnow().timestamp()
        window_start = now - window_seconds
        
        # Redis sorted set: timestamp as score
        set_key = f"rate_limit:{key}"
        
        try:
            pipe = self.redis.pipeline()
            
            # Remove old entries outside window
            pipe.zremrangebyscore(set_key, 0, window_start)
            
            # Count requests in current window
            pipe.zcard(set_key)
            
            # Add current request timestamp
            pipe.zadd(set_key, {str(now): now})
            
            # Set expiry on the key
            pipe.expire(set_key, window_seconds)
            
            results = pipe.execute()
            request_count = results[1]
            
        except Exception as e:
            # Fail open on Redis errors
            print(f"Rate limiter error: {e}")
            return True, {
                "limit": max_requests,
                "remaining": max_requests,
                "reset_at": int(now + window_seconds)
            }
        
        allowed = request_count < max_requests
        
        info = {
            "limit": max_requests,
            "remaining": max(0, max_requests - request_count - 1),
            "reset_at": int(now + window_seconds),
            "current_usage": request_count
        }
        
        return allowed, info
    
    def reset(self, key: str):
        """Reset rate limit for key"""
        set_key = f"rate_limit:{key}"
        try:
            self.redis.delete(set_key)
        except Exception:
            pass


# Rate limit configurations per tier
RATE_LIMITS = {
    "free": {
        "chat": {"requests": 100, "window": 3600},        # 100/hour
        "embeddings": {"requests": 500, "window": 3600},  # 500/hour
        "rag": {"requests": 50, "window": 3600},          # 50/hour
    },
    "pro": {
        "chat": {"requests": 1000, "window": 3600},       # 1000/hour
        "embeddings": {"requests": 5000, "window": 3600}, # 5000/hour
        "rag": {"requests": 500, "window": 3600},         # 500/hour
    },
    "enterprise": {
        "chat": {"requests": 10000, "window": 3600},      # 10k/hour
        "embeddings": {"requests": 50000, "window": 3600},# 50k/hour
        "rag": {"requests": 5000, "window": 3600},        # 5k/hour
    }
}


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    
    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


def rate_limit(
    limiter: SlidingWindowRateLimiter,
    tier_func: callable,
    endpoint: str
):
    """
    Decorator for rate limiting endpoints.
    
    Args:
        limiter: SlidingWindowRateLimiter instance
        tier_func: Function that returns user tier (free/pro/enterprise)
        endpoint: Endpoint name (chat/embeddings/rag)
        
    Example:
        @rate_limit(limiter, get_user_tier, "chat")
        def chat_endpoint(user_id: str, message: str):
            return generate_response(message)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract user_id (assumes first arg or kwarg)
            user_id = kwargs.get('user_id') or (args[0] if args else None)
            if not user_id:
                raise ValueError("user_id required for rate limiting")
            
            # Get user tier
            tier = tier_func(user_id)
            
            # Get rate limit config
            limit_config = RATE_LIMITS.get(tier, RATE_LIMITS["free"]).get(endpoint)
            if not limit_config:
                raise ValueError(f"No rate limit config for {tier}/{endpoint}")
            
            # Check rate limit
            key = f"{user_id}:{endpoint}"
            allowed, info = limiter.allow(
                key=key,
                max_requests=limit_config["requests"],
                window_seconds=limit_config["window"]
            )
            
            if not allowed:
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {endpoint}. "
                    f"Limit: {info['limit']}/{limit_config['window']}s",
                    retry_after=info['reset_at']
                )
            
            # Add rate limit info to response (if dict)
            result = func(*args, **kwargs)
            
            if isinstance(result, dict):
                result['rate_limit'] = {
                    'limit': info['limit'],
                    'remaining': info['remaining'],
                    'reset_at': info['reset_at']
                }
            
            return result
        
        return wrapper
    return decorator


# Mock Redis for demo
class MockRedis:
    """Simple in-memory mock for Redis (for testing only)"""
    
    def __init__(self):
        self.data = {}
    
    def pipeline(self):
        return MockPipeline(self)
    
    def delete(self, key):
        self.data.pop(key, None)


class MockPipeline:
    """Mock Redis pipeline"""
    
    def __init__(self, redis):
        self.redis = redis
        self.commands = []
    
    def zremrangebyscore(self, key, min_score, max_score):
        self.commands.append(('zremrangebyscore', key, min_score, max_score))
        return self
    
    def zcard(self, key):
        self.commands.append(('zcard', key))
        return self
    
    def zadd(self, key, mapping):
        self.commands.append(('zadd', key, mapping))
        return self
    
    def expire(self, key, seconds):
        self.commands.append(('expire', key, seconds))
        return self
    
    def execute(self):
        results = []
        for cmd in self.commands:
            if cmd[0] == 'zremrangebyscore':
                results.append(0)
            elif cmd[0] == 'zcard':
                key = cmd[1]
                zset = self.redis.data.get(key, {})
                results.append(len(zset))
            elif cmd[0] == 'zadd':
                key, mapping = cmd[1], cmd[2]
                if key not in self.redis.data:
                    self.redis.data[key] = {}
                self.redis.data[key].update(mapping)
                results.append(len(mapping))
            elif cmd[0] == 'expire':
                results.append(1)
        return results


if __name__ == "__main__":
    print("=== Sliding Window Rate Limiter Demo ===\n")
    
    # Create mock Redis and limiter
    redis_client = MockRedis()
    limiter = SlidingWindowRateLimiter(redis_client)
    
    # Simulate user making requests
    user_id = "user_123"
    endpoint = "chat"
    max_requests = 5
    window_seconds = 10
    
    print(f"Rate limit: {max_requests} requests per {window_seconds} seconds\n")
    
    for i in range(8):
        allowed, info = limiter.allow(
            key=f"{user_id}:{endpoint}",
            max_requests=max_requests,
            window_seconds=window_seconds
        )
        
        print(f"Request {i+1}:")
        print(f"  Allowed: {allowed}")
        print(f"  Remaining: {info['remaining']}")
        print(f"  Current usage: {info['current_usage']}")
        
        if not allowed:
            print(f"  ❌ Rate limit exceeded! Reset at: {info['reset_at']}")
        else:
            print(f"  ✅ Request allowed")
        
        print()
        time.sleep(0.5)
    
    print("\n=== Rate Limit Decorator Demo ===\n")
    
    # Mock tier function
    def get_user_tier(user_id: str) -> str:
        return "free"
    
    # Example endpoint with rate limiting
    @rate_limit(limiter, get_user_tier, "chat")
    def chat_endpoint(user_id: str, message: str) -> dict:
        return {
            "response": f"Echo: {message}",
            "user_id": user_id
        }
    
    # Make some calls
    for i in range(7):
        try:
            result = chat_endpoint(user_id="user_456", message=f"Hello {i}")
            print(f"Request {i+1}: Success")
            print(f"  Remaining: {result.get('rate_limit', {}).get('remaining', 'N/A')}")
        except RateLimitExceeded as e:
            print(f"Request {i+1}: {e}")
        
        time.sleep(0.3)
