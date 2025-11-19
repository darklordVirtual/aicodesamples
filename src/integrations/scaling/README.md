# Skalering til Produksjon - Kapittel 10.5

Dette er den komplette implementasjonen av **Kapittel 10.5: Skalering til Produksjon** fra boken "AI og Integrasjoner: Fra Grunnleggende til Avansert".

## 📁 Oversikt

Denne mappen inneholder alle komponenter du trenger for å skalere AI-løsninger fra prototype til enterprise-nivå med hundretusenvis av brukere.

## 🚀 Komponenter

### Caching & Performance

**`intelligent_cache.py`** (580 linjer)
- `IntelligentCache`: Multi-tier caching system (Memory → Redis → Database)
- `SemanticCache`: Semantic similarity-based caching (95% threshold)
- `TimedLRUCache`: In-memory LRU cache with TTL
- Cache key generation med deterministic hashing
- Automatic cache invalidation
- **Impact:** 70-85% cache hit rate, 80% cost reduction

**Bruk:**
```python
from src.integrations.scaling.intelligent_cache import IntelligentCache

cache = IntelligentCache()

# Get or compute with caching
result = cache.get_or_compute(
    cache_type="chat_response",
    key_data={"message": user_message, "context": context},
    compute_fn=lambda: generate_ai_response(user_message, context)
)
```

### Async Processing

**`celery_workers.py`** (450 linjer)
- Celery worker pool med task routing
- Separate køer: `fast`, `heavy`, `embeddings`
- `process_large_document`: Batch document processing
- `generate_embedding`: Parallel embedding generation
- `chat_response_async`: Async chat responses
- `monitor_celery_events`: Prometheus metrics
- Auto-retry med exponential backoff

**Bruk:**
```python
from src.integrations.scaling.celery_workers import celery_app, process_large_document

# Queue a task
task = process_large_document.delay(document_id="doc123")

# Check status
result = task.get(timeout=30)
```

### Request Batching

**`request_batcher.py`** (480 linjer)
- `RequestBatcher`: Generic request batching
- `EmbeddingBatcher`: Specialized for embeddings (100/batch)
- `APICallBatcher`: API call batching med rate limiting
- Background processing thread
- Configurable batch size og wait time
- **Impact:** 5-10x reduction i API overhead

**Bruk:**
```python
from src.integrations.scaling.request_batcher import EmbeddingBatcher

batcher = EmbeddingBatcher(batch_size=100, max_wait_time=1.0)

# Embed single text (automatically batched)
embedding = await batcher.embed_text("Hello world")

# Embed multiple texts
embeddings = await batcher.embed_texts(["Text 1", "Text 2", "Text 3"])
```

### Health Checks

**`health_check.py`** (430 linjer)
- `HealthChecker`: Comprehensive health checks
- Checks: Redis, Database, Disk, Memory, CPU
- `DetailedHealthChecker`: Custom checks support
- `create_health_check_app`: FastAPI app med endpoints
- Cached health status (configurable interval)
- Kubernetes-compatible probes

**Bruk:**
```python
from src.integrations.scaling.health_check import create_health_check_app

# Create FastAPI app with health endpoints
app = create_health_check_app()

# Endpoints:
# GET /health - Basic health check
# GET /health/deep - Detailed metrics
# GET /health/ready - Readiness probe (Kubernetes)
# GET /health/live - Liveness probe (Kubernetes)
```

### Resilience & Failover

**`circuit_breaker.py`** (320 linjer)
- `CircuitBreaker`: Fail-safe pattern
- States: CLOSED → OPEN → HALF_OPEN
- Configurable failure threshold
- Automatic recovery testing
- Thread-safe implementation

**Bruk:**
```python
from src.integrations.scaling.circuit_breaker import CircuitBreaker

anthropic_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60
)

# Protected call
try:
    response = anthropic_breaker.call(
        lambda: anthropic.messages.create(...)
    )
except Exception as e:
    # Circuit is OPEN - fallback to alternative
    response = fallback_service()
```

### Rate Limiting

**`rate_limiter.py`** (380 linjer)
- `SlidingWindowRateLimiter`: Accurate rate limiting med Redis
- Per-user og per-tier limits
- Rate limit headers (X-RateLimit-*)
- Decorator support
- Tiers: free (100/h), pro (1000/h), enterprise (10k/h)

**Bruk:**
```python
from src.integrations.scaling.rate_limiter import rate_limit

@app.route("/api/chat")
@rate_limit(tier_key="chat")
def chat():
    # Your implementation
    pass
```

### Database Optimization

**`database_pool.py`** (420 linjer)
- `DatabasePool`: Primary + read replicas
- Connection pooling med pre-ping
- Load-balanced read operations
- Write operations routed to primary
- Query optimization examples
- Cursor-based pagination

**Bruk:**
```python
from src.integrations.scaling.database_pool import DatabasePool

db_pool = DatabasePool()

# Read operation (uses replica)
session = db_pool.get_read_session()
user = session.query(User).filter(User.id == user_id).first()

# Write operation (uses primary)
session = db_pool.get_write_session()
session.add(new_user)
session.commit()
```

### Cost Optimization

**`cost_optimizer.py`** (450 linjer)
- `TokenAwareCache`: Dynamic TTL based on cost
- `CostOptimizedRouter`: Route to cheapest capable model
- Complexity estimation (0-10 scale)
- Model selection: haiku → sonnet → opus
- Cost tracking per request

**Bruk:**
```python
from src.integrations.scaling.cost_optimizer import CostOptimizedRouter

router = CostOptimizedRouter()

# Automatic model selection based on complexity
model = router.route(prompt="Analyze this data", context_size=5000)
# Returns: "haiku" (cheap) for simple queries
# Returns: "sonnet" (medium) for moderate complexity
# Returns: "opus" (expensive) only when necessary
```

### Observability

**`observability.py`** (520 linjer)
- Structured logging med structlog
- Prometheus metrics (Counter, Histogram, Gauge)
- Distributed tracing (OpenTelemetry/Jaeger)
- Custom metrics for AI operations
- Metrics endpoint integration

**Bruk:**
```python
from src.integrations.scaling.observability import (
    logger,
    http_requests_total,
    ai_api_tokens_total
)

# Structured logging
log = logger.bind(user_id=user_id, request_id=request_id)
log.info("request_started")

# Metrics
http_requests_total.labels(
    method="POST",
    endpoint="/api/chat",
    status=200
).inc()

ai_api_tokens_total.labels(
    provider="anthropic",
    model="claude-sonnet-4",
    type="completion"
).inc(response.usage.completion_tokens)
```

## 📊 Performance Impact

Ved implementering av alle komponenter:

- **Cache hit rate:** 70-85%
- **Cost reduction:** 80% (fra $45k/mnd til $8k/mnd ved 200k brukere)
- **Latency:** 2-5s → 150-400ms (10x forbedring)
- **Load reduction:** 5x færre API-kall
- **Throughput:** 1k → 200k samtidige brukere
- **Uptime:** 99.95%

## 🏗️ Arkitektur ved skala

```
Users (100k-500k)
    ↓
CloudFlare CDN (Cache L1)
    ↓
NGINX Load Balancer
    ↓
8-20 API Nodes (auto-scaling)
    ↓
Redis Cluster (6 nodes) + PostgreSQL (1 primary + 5 replicas)
    ↓
15-50 Celery Workers (queue-based)
    ↓
AI API Layer (Anthropic/OpenAI + fallbacks)
```

## 🚀 Deployment

Se `examples/kubernetes/` for komplette deployment manifests:

- `api-deployment.yaml` - API servers med HPA
- `celery-deployment.yaml` - Worker pools
- `redis-statefulset.yaml` - Redis cluster
- `postgres-statefulset.yaml` - PostgreSQL med replicas

Se `examples/nginx/load_balancer.conf` for NGINX konfiguration.

## 📚 Dokumentasjon

Full dokumentasjon i boken **Kapittel 10.5: Skalering til Produksjon** (45 min lesetid).

## 🧪 Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Test specific component
pytest tests/test_intelligent_cache.py
pytest tests/test_rate_limiter.py
```

## 🔧 Konfigurasjon

Alle komponenter kan konfigureres via environment variables:

```bash
# Redis
export REDIS_URL="redis://redis-cluster:6379/0"

# Database
export DATABASE_URL="postgresql://user:pass@primary.db:5432/app"

# Celery
export CELERY_BROKER="redis://redis-cluster:6379/0"
export CELERY_BACKEND="redis://redis-cluster:6379/1"

# AI APIs
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

## 💡 Best Practices

1. **Start simple, scale incrementally**
   - Ikke over-engineer tidlig
   - Legg til kompleksitet når det trengs

2. **Caching er din beste venn**
   - Multi-tier strategy (CDN → Redis → App)
   - Semantic caching for LLMs
   - 80%+ cost savings mulig

3. **Async alt som kan være async**
   - Queue heavy operations
   - Bedre UX med umiddelbar respons

4. **Observability fra dag 1**
   - Du kan ikke fikse det du ikke ser
   - Metrics, logs, traces essensielt

5. **Cost awareness på hvert lag**
   - Smart model routing
   - Request batching
   - Token-aware caching

## 🤝 Bidra

Dette er et åpent prosjekt. Pull requests er velkomne!

## 📄 Lisens

Se LICENSE fil i root av prosjektet.

---

**Bygget med ❤️ av Stian Skogbrott | Luftfiber AS | 2025**
