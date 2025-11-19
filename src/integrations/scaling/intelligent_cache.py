"""
Intelligent caching strategies for AI applications at scale.

Features:
- Multi-tier caching (CDN, Redis, in-memory)
- Semantic caching for similar queries
- Token-aware TTL for cost optimization
"""
import hashlib
import json
import time
from typing import Any, Dict, Optional, Callable
from functools import lru_cache
import numpy as np


class TimedLRUCache:
    """LRU cache with TTL for hot data (application-level caching)"""
    
    def __init__(self, maxsize: int = 1000, ttl: int = 300):
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.maxsize = maxsize
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache with eviction if at capacity"""
        if len(self.cache) >= self.maxsize:
            # Evict oldest
            oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
            del self.cache[oldest_key]
        
        self.cache[key] = (value, time.time())
    
    def clear(self):
        """Clear all cached items"""
        self.cache.clear()


class IntelligentCache:
    """
    Multi-tier intelligent caching system.
    
    Tiers:
    1. In-memory LRU (fastest, smallest)
    2. Redis cluster (fast, shared across instances)
    3. Database (slowest, most persistent)
    
    Example:
        cache = IntelligentCache()
        result = cache.get_or_compute(
            cache_type="embeddings",
            key_data={"text": "hello world"},
            compute_fn=lambda: generate_embedding("hello world")
        )
    """
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.memory_cache = TimedLRUCache(maxsize=10000, ttl=300)
        
        # Different TTLs for different data types
        self.ttl_config = {
            "embeddings": 86400 * 7,    # 7 days
            "chat_response": 3600,       # 1 hour
            "user_session": 86400,       # 24 hours
            "rate_limit": 60,            # 1 minute
            "rag_results": 1800,         # 30 minutes
        }
    
    def cache_key(self, prefix: str, data: dict) -> str:
        """Generate deterministic cache key"""
        data_str = json.dumps(data, sort_keys=True)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]
        return f"{prefix}:{data_hash}"
    
    def get_or_compute(
        self,
        cache_type: str,
        key_data: dict,
        compute_fn: Callable,
        force_refresh: bool = False
    ) -> Any:
        """
        Get from cache or compute and store.
        
        Args:
            cache_type: Type of cache (determines TTL)
            key_data: Data to generate cache key from
            compute_fn: Function to call if cache miss
            force_refresh: Skip cache and recompute
            
        Returns:
            Cached or computed result
        """
        cache_key = self.cache_key(cache_type, key_data)
        
        if not force_refresh:
            # Try memory cache first (L1)
            result = self.memory_cache.get(cache_key)
            if result is not None:
                return result
            
            # Try Redis (L2)
            if self.redis:
                try:
                    cached = self.redis.get(cache_key)
                    if cached:
                        result = json.loads(cached)
                        # Store in memory cache
                        self.memory_cache.set(cache_key, result)
                        return result
                except Exception:
                    pass  # Fallback to compute
        
        # Compute
        result = compute_fn()
        
        # Store in both caches
        self.memory_cache.set(cache_key, result)
        
        if self.redis:
            ttl = self.ttl_config.get(cache_type, 3600)
            try:
                self.redis.setex(
                    cache_key,
                    ttl,
                    json.dumps(result)
                )
            except Exception:
                pass  # Continue without Redis cache
        
        return result
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        if not self.redis:
            return
        
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match=pattern, count=1000)
            if keys:
                self.redis.delete(*keys)
            if cursor == 0:
                break
        
        # Clear memory cache (can't pattern match)
        self.memory_cache.clear()


class SemanticCache:
    """
    Cache LLM responses based on semantic similarity.
    If a query is 95% similar to a previous one, return cached response.
    
    Example:
        cache = SemanticCache(redis_client)
        
        # Check for similar cached response
        cached = cache.find_similar("What is Python?", context={})
        if cached:
            return cached["response"]
        
        # Store new response
        response = generate_ai_response("What is Python?")
        cache.store("What is Python?", response, context={})
    """
    
    def __init__(self, redis_client, similarity_threshold: float = 0.95):
        self.redis = redis_client
        self.threshold = similarity_threshold
        self.embedding_cache = TimedLRUCache(maxsize=5000)
    
    def get_embedding(self, text: str) -> list[float]:
        """
        Get or compute embedding for text.
        In production, use actual embedding API (OpenAI, Cohere, etc.)
        """
        cached = self.embedding_cache.get(text)
        if cached:
            return cached
        
        # Placeholder: In production, call embedding API
        # embedding = openai.embeddings.create(
        #     model="text-embedding-3-small",
        #     input=text
        # ).data[0].embedding
        
        # For demo: simple hash-based pseudo-embedding
        import hashlib
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        embedding = [b / 255.0 for b in hash_bytes[:128]]
        
        self.embedding_cache.set(text, embedding)
        return embedding
    
    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def find_similar(self, query: str, context: dict) -> Optional[Dict[str, Any]]:
        """Find semantically similar cached response"""
        query_embedding = self.get_embedding(query)
        
        # Get all cached queries for this context
        context_hash = hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()
        context_key = f"semantic_cache:{context_hash}"
        
        try:
            cached_queries = self.redis.hgetall(context_key)
        except Exception:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for cached_query, cached_data_str in cached_queries.items():
            try:
                cached_data = json.loads(cached_data_str)
                cached_embedding = cached_data["embedding"]
                
                similarity = self.cosine_similarity(query_embedding, cached_embedding)
                
                if similarity > best_similarity and similarity >= self.threshold:
                    best_similarity = similarity
                    best_match = cached_data
            except Exception:
                continue
        
        return best_match
    
    def store(self, query: str, response: str, context: dict, ttl: int = 86400):
        """Store query-response pair with embedding"""
        embedding = self.get_embedding(query)
        
        context_hash = hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()
        context_key = f"semantic_cache:{context_hash}"
        
        data = {
            "response": response,
            "embedding": embedding,
            "timestamp": time.time()
        }
        
        try:
            self.redis.hset(context_key, query, json.dumps(data))
            self.redis.expire(context_key, ttl)
        except Exception:
            pass


class TokenAwareCache:
    """
    Cache that adjusts TTL based on token costs.
    More expensive queries are cached longer.
    
    Example:
        cache = TokenAwareCache(redis_client)
        cache.cache_response(
            key="user_query_123",
            response="AI response here",
            model="claude-opus",
            tokens=1500
        )
    """
    
    COSTS = {
        "claude-opus": 0.015,
        "claude-sonnet": 0.003,
        "claude-haiku": 0.00025,
        "gpt-4": 0.03,
        "gpt-3.5": 0.0015
    }
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def calculate_ttl(self, model: str, tokens: int) -> int:
        """Calculate TTL based on cost"""
        cost = self.COSTS.get(model, 0.003) * tokens
        
        # More expensive = longer cache
        if cost > 1.0:      # >$1
            return 86400 * 7  # 7 days
        elif cost > 0.1:    # >$0.10
            return 86400      # 1 day
        elif cost > 0.01:   # >$0.01
            return 3600       # 1 hour
        else:
            return 300        # 5 minutes
    
    def cache_response(
        self,
        key: str,
        response: str,
        model: str,
        tokens: int
    ):
        """Cache response with dynamic TTL"""
        ttl = self.calculate_ttl(model, tokens)
        
        data = {
            "response": response,
            "model": model,
            "tokens": tokens,
            "cost": self.COSTS.get(model, 0) * tokens,
            "cached_at": time.time()
        }
        
        try:
            self.redis.setex(key, ttl, json.dumps(data))
        except Exception:
            pass
    
    def get_response(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        try:
            cached = self.redis.get(key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass
        
        return None


# Decorator for caching function results
@lru_cache(maxsize=1000)
def get_system_prompt(template_name: str) -> str:
    """
    Cache system prompts in memory - they rarely change.
    Uses Python's built-in LRU cache.
    """
    # In production, load from database or file
    prompts = {
        "default": "You are a helpful AI assistant.",
        "technical": "You are a technical expert helping with programming.",
        "customer_support": "You are a friendly customer support agent."
    }
    return prompts.get(template_name, prompts["default"])


if __name__ == "__main__":
    print("=== Intelligent Cache Demo ===\n")
    
    # Demo multi-tier cache
    cache = IntelligentCache()
    
    def expensive_computation(x: int) -> int:
        print(f"  Computing {x}**2...")
        time.sleep(0.1)
        return x ** 2
    
    print("First call (cache miss):")
    result1 = cache.get_or_compute(
        cache_type="embeddings",
        key_data={"value": 5},
        compute_fn=lambda: expensive_computation(5)
    )
    print(f"Result: {result1}\n")
    
    print("Second call (cache hit):")
    result2 = cache.get_or_compute(
        cache_type="embeddings",
        key_data={"value": 5},
        compute_fn=lambda: expensive_computation(5)
    )
    print(f"Result: {result2}\n")
    
    print("=== System Prompt Cache Demo ===\n")
    
    # Demo LRU cache decorator
    print("Loading prompts (first time):")
    prompt1 = get_system_prompt("technical")
    print(f"Prompt: {prompt1}\n")
    
    print("Loading same prompt (cached):")
    prompt2 = get_system_prompt("technical")
    print(f"Prompt: {prompt2}\n")
    
    # Show cache info
    print(f"Cache info: {get_system_prompt.cache_info()}")
