"""
Circuit breaker pattern for resilient microservices.

Prevents cascading failures by stopping requests to failing services,
giving them time to recover.
"""
import threading
import time
from enum import Enum
from datetime import datetime
from typing import Any, Callable, Dict, Optional


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation - requests flow through
    OPEN = "open"            # Failing - reject all requests
    HALF_OPEN = "half_open"  # Testing - allow limited requests to test recovery


class CircuitBreaker:
    """
    Circuit breaker for external service calls.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Too many failures, reject all requests
    - HALF_OPEN: Allow test request to check if service recovered
    
    Example:
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=ServiceException
        )
        
        try:
            result = breaker.call(external_api_call, arg1, arg2)
        except Exception as e:
            # Handle failure
            pass
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        name: str = "circuit_breaker"
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting reset
            expected_exception: Exception type to catch
            name: Name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
        self.lock = threading.Lock()
        
        print(f"[{self.name}] Initialized: threshold={failure_threshold}, timeout={recovery_timeout}s")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result from function call
            
        Raises:
            Exception: If circuit is OPEN or function fails
        """
        with self.lock:
            if self.state == CircuitState.OPEN:
                # Check if should attempt reset
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    print(f"[{self.name}] State: HALF_OPEN (attempting reset)")
                else:
                    elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                    wait_time = self.recovery_timeout - elapsed
                    raise Exception(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Retry in {wait_time:.0f}s"
                    )
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        with self.lock:
            self.success_count += 1
            self.last_success_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                # Recovered!
                self.failure_count = 0
                self.state = CircuitState.CLOSED
                print(f"[{self.name}] State: CLOSED (recovered)")
            
            # Always reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                print(
                    f"[{self.name}] State: OPEN "
                    f"(failures: {self.failure_count}/{self.failure_threshold})"
                )
            else:
                print(
                    f"[{self.name}] Failure {self.failure_count}/{self.failure_threshold}"
                )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to try recovery"""
        if not self.last_failure_time:
            return False
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        with self.lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
                "last_success": self.last_success_time.isoformat() if self.last_success_time else None
            }
    
    def reset(self):
        """Manually reset circuit breaker"""
        with self.lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            print(f"[{self.name}] Manually reset to CLOSED")


class MultiCircuitBreaker:
    """
    Manage multiple circuit breakers for different services.
    
    Example:
        breakers = MultiCircuitBreaker()
        breakers.add("anthropic", failure_threshold=5, recovery_timeout=60)
        breakers.add("openai", failure_threshold=3, recovery_timeout=30)
        
        # Use with fallback
        result = breakers.call_with_fallback(
            primary=("anthropic", anthropic_call),
            fallback=("openai", openai_call)
        )
    """
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def add(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        """Add a circuit breaker"""
        self.breakers[name] = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=name
        )
    
    def call(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Call through named circuit breaker"""
        if name not in self.breakers:
            raise ValueError(f"No circuit breaker named '{name}'")
        
        return self.breakers[name].call(func, *args, **kwargs)
    
    def call_with_fallback(
        self,
        primary: tuple[str, Callable],
        fallback: tuple[str, Callable],
        *args,
        **kwargs
    ) -> Any:
        """
        Call primary, fallback to secondary if it fails.
        
        Args:
            primary: (breaker_name, function)
            fallback: (breaker_name, function)
            
        Returns:
            Result from primary or fallback
        """
        primary_name, primary_func = primary
        fallback_name, fallback_func = fallback
        
        # Try primary
        try:
            return self.call(primary_name, primary_func, *args, **kwargs)
        except Exception as e:
            print(f"Primary '{primary_name}' failed: {e}")
            
            # Try fallback
            try:
                return self.call(fallback_name, fallback_func, *args, **kwargs)
            except Exception as e2:
                print(f"Fallback '{fallback_name}' also failed: {e2}")
                raise Exception("All services unavailable")
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers"""
        return {
            name: breaker.get_state()
            for name, breaker in self.breakers.items()
        }


if __name__ == "__main__":
    import random
    
    print("=== Circuit Breaker Demo ===\n")
    
    # Simulated unreliable service
    call_count = [0]
    
    def unreliable_service():
        """Simulates service that fails initially, then recovers"""
        call_count[0] += 1
        
        # Fail first 5 calls, then succeed
        if call_count[0] <= 5:
            raise Exception("Service unavailable")
        return f"Success! (call {call_count[0]})"
    
    # Create circuit breaker
    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=2,
        name="demo_service"
    )
    
    # Make calls
    print("Making 15 calls through circuit breaker...\n")
    
    for i in range(15):
        try:
            result = breaker.call(unreliable_service)
            print(f"Call {i+1}: ✅ {result}")
        except Exception as e:
            print(f"Call {i+1}: ❌ {e}")
        
        # Show state
        state = breaker.get_state()
        print(f"  State: {state['state']}, Failures: {state['failure_count']}\n")
        
        time.sleep(0.5)
    
    print("\n=== Multi-Circuit Breaker with Fallback ===\n")
    
    # Create multi-breaker
    breakers = MultiCircuitBreaker()
    breakers.add("primary", failure_threshold=2, recovery_timeout=3)
    breakers.add("fallback", failure_threshold=2, recovery_timeout=3)
    
    # Simulate primary failing, fallback succeeding
    primary_calls = [0]
    fallback_calls = [0]
    
    def primary_service():
        primary_calls[0] += 1
        if primary_calls[0] <= 4:
            raise Exception("Primary down")
        return "Primary success"
    
    def fallback_service():
        fallback_calls[0] += 1
        return "Fallback success"
    
    print("Testing primary with fallback...\n")
    
    for i in range(8):
        try:
            result = breakers.call_with_fallback(
                primary=("primary", primary_service),
                fallback=("fallback", fallback_service)
            )
            print(f"Call {i+1}: ✅ {result}")
        except Exception as e:
            print(f"Call {i+1}: ❌ {e}")
        
        time.sleep(0.5)
    
    # Show final states
    print("\nFinal states:")
    for name, state in breakers.get_all_states().items():
        print(f"  {name}: {state['state']} (failures: {state['failure_count']}, successes: {state['success_count']})")
