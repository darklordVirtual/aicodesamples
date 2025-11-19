"""
Scaling components for production AI systems (Kapittel 10.5)

Components:
- intelligent_cache: Multi-tier caching with semantic similarity
- rate_limiter: Sliding window rate limiting with Redis
- circuit_breaker: Circuit breaker pattern for resilience
- database_pool: Database connection pooling with read replicas
- celery_workers: Async task processing with Celery
- observability: Metrics, logging, and tracing
- cost_optimizer: Cost-aware model routing and caching
"""

from .intelligent_cache import (
    IntelligentCache,
    SemanticCache,
    TokenAwareCache as TACache
)
from .rate_limiter import SlidingWindowRateLimiter
from .circuit_breaker import CircuitBreaker, CircuitState
from .database_pool import DatabasePool
from .observability import (
    setup_structured_logging,
    PrometheusMetrics,
    DistributedTracer
)
from .cost_optimizer import CostOptimizedRouter, RequestBatcher

__all__ = [
    'IntelligentCache',
    'SemanticCache',
    'TACache',
    'SlidingWindowRateLimiter',
    'CircuitBreaker',
    'CircuitState',
    'DatabasePool',
    'setup_structured_logging',
    'PrometheusMetrics',
    'DistributedTracer',
    'CostOptimizedRouter',
    'RequestBatcher'
]
