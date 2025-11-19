"""
Observability components: metrics, logging, and tracing.

Components:
- Structured logging with context
- Prometheus metrics
- Distributed tracing
"""
import time
from typing import Any, Dict, Optional
from contextlib import contextmanager


def setup_structured_logging():
    """
    Setup structured logging with JSON output.
    In production, use structlog or similar.
    
    Example:
        setup_structured_logging()
        logger = get_logger("my_service")
        logger.info("request_started", user_id=123, endpoint="/api/chat")
    """
    try:
        import structlog
        
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        print("✅ Structured logging configured (structlog)")
        return structlog.get_logger()
    
    except ImportError:
        print("⚠️  structlog not available, using standard logging")
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger()


class PrometheusMetrics:
    """
    Prometheus metrics collector.
    
    Metrics:
    - Counters: Total requests, errors, etc.
    - Histograms: Request duration, token usage
    - Gauges: Active users, queue size
    
    Example:
        metrics = PrometheusMetrics()
        
        metrics.http_requests_total.inc(method="POST", endpoint="/api/chat", status=200)
        metrics.request_duration.observe(0.5, method="POST", endpoint="/api/chat")
        metrics.active_users.set(1234)
    """
    
    def __init__(self):
        try:
            from prometheus_client import Counter, Histogram, Gauge, generate_latest
            
            # HTTP metrics
            self.http_requests_total = Counter(
                'http_requests_total',
                'Total HTTP requests',
                ['method', 'endpoint', 'status']
            )
            
            self.request_duration = Histogram(
                'http_request_duration_seconds',
                'HTTP request duration',
                ['method', 'endpoint']
            )
            
            # AI API metrics
            self.ai_api_calls_total = Counter(
                'ai_api_calls_total',
                'Total AI API calls',
                ['provider', 'model', 'status']
            )
            
            self.ai_api_tokens_total = Counter(
                'ai_api_tokens_total',
                'Total tokens used',
                ['provider', 'model', 'type']  # type: prompt/completion
            )
            
            self.ai_api_cost_total = Counter(
                'ai_api_cost_total',
                'Total AI API cost in USD',
                ['provider', 'model']
            )
            
            # Cache metrics
            self.cache_hits_total = Counter(
                'cache_hits_total',
                'Total cache hits',
                ['cache_type']
            )
            
            self.cache_misses_total = Counter(
                'cache_misses_total',
                'Total cache misses',
                ['cache_type']
            )
            
            # System metrics
            self.active_users = Gauge(
                'active_users',
                'Currently active users'
            )
            
            self.queue_size = Gauge(
                'queue_size',
                'Current queue size',
                ['queue_name']
            )
            
            self.circuit_breaker_state = Gauge(
                'circuit_breaker_state',
                'Circuit breaker state (0=closed, 1=open, 2=half_open)',
                ['service']
            )
            
            self.generate_latest = generate_latest
            
            print("✅ Prometheus metrics initialized")
        
        except ImportError:
            print("⚠️  prometheus_client not available")
            # Create mock objects
            self._create_mocks()
    
    def _create_mocks(self):
        """Create mock metric objects for when prometheus_client not available"""
        class MockMetric:
            def inc(self, *args, **kwargs):
                pass
            def observe(self, *args, **kwargs):
                pass
            def set(self, *args, **kwargs):
                pass
        
        self.http_requests_total = MockMetric()
        self.request_duration = MockMetric()
        self.ai_api_calls_total = MockMetric()
        self.ai_api_tokens_total = MockMetric()
        self.ai_api_cost_total = MockMetric()
        self.cache_hits_total = MockMetric()
        self.cache_misses_total = MockMetric()
        self.active_users = MockMetric()
        self.queue_size = MockMetric()
        self.circuit_breaker_state = MockMetric()
        self.generate_latest = lambda: b"# Mock metrics"
    
    @contextmanager
    def track_request(self, method: str, endpoint: str):
        """Context manager to track request duration"""
        start_time = time.time()
        status = 500  # Default to error
        
        try:
            yield
            status = 200  # Success
        except Exception:
            status = 500  # Error
            raise
        finally:
            duration = time.time() - start_time
            
            self.http_requests_total.inc(
                method=method,
                endpoint=endpoint,
                status=status
            )
            
            self.request_duration.observe(
                duration,
                method=method,
                endpoint=endpoint
            )


class DistributedTracer:
    """
    Distributed tracing for microservices.
    Integrates with OpenTelemetry/Jaeger.
    
    Example:
        tracer = DistributedTracer("my_service")
        
        with tracer.start_span("process_request") as span:
            span.set_attribute("user_id", user_id)
            
            with tracer.start_span("db_query") as db_span:
                db_span.set_attribute("query", "SELECT * FROM users")
                result = execute_query()
            
            return result
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            
            trace.set_tracer_provider(TracerProvider())
            self.tracer = trace.get_tracer(service_name)
            self.available = True
            
            print(f"✅ Distributed tracing initialized for: {service_name}")
        
        except ImportError:
            print("⚠️  opentelemetry not available")
            self.tracer = None
            self.available = False
    
    @contextmanager
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Start a new span"""
        if not self.available:
            # Mock span
            yield MockSpan()
            return
        
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span


class MockSpan:
    """Mock span for when tracing not available"""
    def set_attribute(self, key: str, value: Any):
        pass


# Flask/FastAPI middleware example
def create_metrics_middleware(metrics: PrometheusMetrics):
    """
    Create middleware for automatic metrics collection.
    
    For Flask:
        @app.before_request
        def before_request():
            request.start_time = time.time()
        
        @app.after_request
        def after_request(response):
            duration = time.time() - request.start_time
            metrics.request_duration.observe(duration, ...)
            return response
    """
    pass  # Implementation depends on framework


if __name__ == "__main__":
    print("=== Observability Demo ===\n")
    
    # Setup logging
    print("1. Structured Logging:")
    logger = setup_structured_logging()
    print()
    
    # Setup metrics
    print("2. Prometheus Metrics:")
    metrics = PrometheusMetrics()
    print()
    
    # Simulate some requests
    print("3. Simulating requests with metrics:\n")
    
    for i in range(5):
        with metrics.track_request("POST", "/api/chat"):
            time.sleep(0.1)  # Simulate work
            metrics.ai_api_calls_total.inc(
                provider="anthropic",
                model="claude-sonnet",
                status="success"
            )
            metrics.ai_api_tokens_total.inc(
                provider="anthropic",
                model="claude-sonnet",
                type="prompt"
            )
        print(f"  Request {i+1} tracked")
    
    print()
    
    # Cache metrics
    print("4. Cache metrics:")
    for _ in range(3):
        metrics.cache_hits_total.inc(cache_type="redis")
    for _ in range(1):
        metrics.cache_misses_total.inc(cache_type="redis")
    print("  3 cache hits, 1 cache miss")
    print()
    
    # System metrics
    print("5. System metrics:")
    metrics.active_users.set(1234)
    metrics.queue_size.set(56, queue_name="ai_tasks")
    print("  Active users: 1234")
    print("  Queue size: 56")
    print()
    
    # Distributed tracing
    print("6. Distributed Tracing:")
    tracer = DistributedTracer("demo_service")
    
    with tracer.start_span("demo_operation", {"user_id": "123"}) as span:
        time.sleep(0.05)
        with tracer.start_span("sub_operation") as sub_span:
            time.sleep(0.02)
    
    print("  Trace completed")
    print()
    
    print("✅ Observability demo complete!")
