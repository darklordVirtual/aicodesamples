"""
Kapittel 10.5: Health Check Module
Comprehensive health checking for load balancers and monitoring systems.
"""
import os
import time
import psutil
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum

try:
    from utils import logger
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from utils import logger


class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthChecker:
    """
    Comprehensive health checking system
    
    Performs various health checks and caches results to avoid
    excessive checking overhead.
    """
    
    def __init__(
        self,
        check_interval: int = 10,
        redis_client=None,
        database_client=None
    ):
        """
        Initialize health checker
        
        Args:
            check_interval: Seconds between full health checks
            redis_client: Redis client instance
            database_client: Database client instance
        """
        self.check_interval = check_interval
        self.redis = redis_client
        self.database = database_client
        
        self.last_check = 0
        self.cached_status = None
        
        # Thresholds
        self.disk_threshold = 90  # percent
        self.memory_threshold = 90  # percent
        self.cpu_threshold = 95  # percent
    
    def check_redis(self) -> bool:
        """Check Redis connectivity"""
        if not self.redis:
            return True  # Skip if no Redis configured
        
        try:
            self.redis.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def check_database(self) -> bool:
        """Check database connectivity"""
        if not self.database:
            return True  # Skip if no database configured
        
        try:
            # Attempt simple query
            self.database.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def check_disk_space(self) -> tuple[bool, float]:
        """
        Check available disk space
        
        Returns:
            (is_healthy, usage_percent)
        """
        try:
            usage = psutil.disk_usage('/')
            is_healthy = usage.percent < self.disk_threshold
            return is_healthy, usage.percent
        except Exception as e:
            logger.error(f"Disk check failed: {e}")
            return False, 0.0
    
    def check_memory(self) -> tuple[bool, float]:
        """
        Check available memory
        
        Returns:
            (is_healthy, usage_percent)
        """
        try:
            memory = psutil.virtual_memory()
            is_healthy = memory.percent < self.memory_threshold
            return is_healthy, memory.percent
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return False, 0.0
    
    def check_cpu(self) -> tuple[bool, float]:
        """
        Check CPU usage
        
        Returns:
            (is_healthy, usage_percent)
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            is_healthy = cpu_percent < self.cpu_threshold
            return is_healthy, cpu_percent
        except Exception as e:
            logger.error(f"CPU check failed: {e}")
            return False, 0.0
    
    def check_dependencies(self) -> Dict[str, bool]:
        """
        Check external dependencies
        
        Returns:
            Dictionary of dependency statuses
        """
        return {
            "redis": self.check_redis(),
            "database": self.check_database(),
        }
    
    def check_resources(self) -> Dict[str, Any]:
        """
        Check system resources
        
        Returns:
            Dictionary of resource checks
        """
        disk_healthy, disk_usage = self.check_disk_space()
        memory_healthy, memory_usage = self.check_memory()
        cpu_healthy, cpu_usage = self.check_cpu()
        
        return {
            "disk": {
                "healthy": disk_healthy,
                "usage_percent": disk_usage,
                "threshold": self.disk_threshold
            },
            "memory": {
                "healthy": memory_healthy,
                "usage_percent": memory_usage,
                "threshold": self.memory_threshold
            },
            "cpu": {
                "healthy": cpu_healthy,
                "usage_percent": cpu_usage,
                "threshold": self.cpu_threshold
            }
        }
    
    def get_status(self, force: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive health status
        
        Args:
            force: Force new check even if cached
            
        Returns:
            Health status dictionary
        """
        now = time.time()
        
        # Use cached status if recent
        if not force and self.cached_status and (now - self.last_check) < self.check_interval:
            return self.cached_status
        
        # Perform checks
        dependencies = self.check_dependencies()
        resources = self.check_resources()
        
        # Determine overall status
        all_deps_healthy = all(dependencies.values())
        all_resources_healthy = all(r["healthy"] for r in resources.values())
        
        if all_deps_healthy and all_resources_healthy:
            overall_status = HealthStatus.HEALTHY
        elif not all_deps_healthy:
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED
        
        status_response = {
            "status": overall_status.value,
            "timestamp": now,
            "timestamp_iso": datetime.utcnow().isoformat(),
            "checks": {
                "dependencies": dependencies,
                "resources": resources
            },
            "version": os.getenv("APP_VERSION", "unknown"),
            "uptime_seconds": time.time() - psutil.boot_time()
        }
        
        self.cached_status = status_response
        self.last_check = now
        
        return status_response
    
    def is_healthy(self) -> bool:
        """
        Quick health check
        
        Returns:
            True if system is healthy
        """
        status = self.get_status()
        return status["status"] == HealthStatus.HEALTHY.value


class DetailedHealthChecker(HealthChecker):
    """
    Extended health checker with custom checks
    """
    
    def __init__(self, *args, custom_checks: Optional[Dict[str, Callable]] = None, **kwargs):
        """
        Initialize with custom checks
        
        Args:
            custom_checks: Dictionary of name -> check_function
        """
        super().__init__(*args, **kwargs)
        self.custom_checks = custom_checks or {}
    
    def run_custom_checks(self) -> Dict[str, Any]:
        """
        Run all custom checks
        
        Returns:
            Dictionary of custom check results
        """
        results = {}
        
        for name, check_func in self.custom_checks.items():
            try:
                results[name] = {
                    "healthy": check_func(),
                    "error": None
                }
            except Exception as e:
                logger.error(f"Custom check '{name}' failed: {e}")
                results[name] = {
                    "healthy": False,
                    "error": str(e)
                }
        
        return results
    
    def get_status(self, force: bool = False) -> Dict[str, Any]:
        """Get status including custom checks"""
        status = super().get_status(force)
        
        if self.custom_checks:
            status["checks"]["custom"] = self.run_custom_checks()
            
            # Update overall status if custom checks failed
            if not all(c["healthy"] for c in status["checks"]["custom"].values()):
                if status["status"] == HealthStatus.HEALTHY.value:
                    status["status"] = HealthStatus.DEGRADED.value
        
        return status


def create_health_check_app():
    """
    Create a FastAPI app with health check endpoints
    
    Returns:
        FastAPI app instance
    """
    try:
        from fastapi import FastAPI, status
        from fastapi.responses import JSONResponse
    except ImportError:
        logger.error("FastAPI not installed. Install with: pip install fastapi")
        return None
    
    app = FastAPI(title="Health Check API")
    health_checker = HealthChecker()
    
    @app.get("/health")
    def health_check():
        """Lightweight health check for load balancer"""
        health_status = health_checker.get_status()
        
        if health_status["status"] == HealthStatus.HEALTHY.value:
            return JSONResponse(
                content=health_status,
                status_code=status.HTTP_200_OK
            )
        else:
            return JSONResponse(
                content=health_status,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )
    
    @app.get("/health/deep")
    def deep_health_check():
        """Detailed health check with all metrics"""
        health_status = health_checker.get_status(force=True)
        
        # Add detailed system metrics
        health_status["metrics"] = {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent_per_core": psutil.cpu_percent(interval=0.1, percpu=True),
            "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "disk_free_gb": psutil.disk_usage('/').free / 1024 / 1024 / 1024,
            "active_connections": len(psutil.net_connections()),
            "process_count": len(psutil.pids())
        }
        
        return health_status
    
    @app.get("/health/ready")
    def readiness_check():
        """Kubernetes readiness probe"""
        health_status = health_checker.get_status()
        
        # Ready if healthy or degraded (but not unhealthy)
        is_ready = health_status["status"] != HealthStatus.UNHEALTHY.value
        
        return JSONResponse(
            content={"ready": is_ready, "status": health_status["status"]},
            status_code=status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
        )
    
    @app.get("/health/live")
    def liveness_check():
        """Kubernetes liveness probe"""
        # Simple liveness check - just respond
        return {"alive": True, "timestamp": time.time()}
    
    return app


if __name__ == "__main__":
    # Test health checker
    checker = HealthChecker()
    status = checker.get_status()
    
    print("Health Status:")
    print(f"  Overall: {status['status']}")
    print(f"  Dependencies: {status['checks']['dependencies']}")
    print(f"  Resources: {status['checks']['resources']}")
