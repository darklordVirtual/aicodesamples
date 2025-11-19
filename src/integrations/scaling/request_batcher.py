"""
Kapittel 10.5: Request Batching
Batch multiple requests to reduce API overhead and costs.
"""
import asyncio
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional
from collections import deque
from datetime import datetime

try:
    from utils import logger
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from utils import logger


class RequestBatcher:
    """
    Batch multiple requests to reduce API overhead
    Useful for embeddings generation and other bulk operations
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        max_wait_time: float = 1.0,
        process_func: Optional[Callable] = None
    ):
        """
        Initialize request batcher
        
        Args:
            batch_size: Maximum number of requests per batch
            max_wait_time: Maximum time to wait before processing batch
            process_func: Function to process batched requests
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.process_func = process_func
        
        self.queue: deque = deque()
        self.queue_lock = threading.Lock()
        self.results: Dict[str, Any] = {}
        self.results_lock = threading.Lock()
        
        self.last_batch_time = time.time()
        self.batch_count = 0
        self.total_requests = 0
        
        # Start background processor
        self._running = True
        self._processor_thread = threading.Thread(target=self._background_processor, daemon=True)
        self._processor_thread.start()
    
    def add_request(self, data: Any, request_id: Optional[str] = None) -> str:
        """
        Add request to batch queue
        
        Args:
            data: Request data
            request_id: Optional request ID (generated if not provided)
            
        Returns:
            Request ID for retrieving result
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        with self.queue_lock:
            self.queue.append((data, request_id))
            self.total_requests += 1
            
            # Process immediately if batch is full
            if len(self.queue) >= self.batch_size:
                self._process_batch_sync()
        
        return request_id
    
    async def add_request_async(self, data: Any, request_id: Optional[str] = None) -> Any:
        """
        Add request and wait for result asynchronously
        
        Args:
            data: Request data
            request_id: Optional request ID
            
        Returns:
            Processing result
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        with self.queue_lock:
            self.queue.append((data, request_id))
            self.total_requests += 1
            
            # Process immediately if batch is full
            if len(self.queue) >= self.batch_size:
                await self._process_batch()
        
        # Wait for result
        max_wait = self.max_wait_time + 5  # Add buffer
        start_time = time.time()
        
        while request_id not in self.results:
            if time.time() - start_time > max_wait:
                raise TimeoutError(f"Request {request_id} timed out")
            await asyncio.sleep(0.01)
        
        with self.results_lock:
            result = self.results.pop(request_id)
        
        return result
    
    def get_result(self, request_id: str, timeout: float = 10.0) -> Optional[Any]:
        """
        Get result for request (blocking)
        
        Args:
            request_id: Request ID
            timeout: Maximum time to wait
            
        Returns:
            Result or None if timeout
        """
        start_time = time.time()
        
        while request_id not in self.results:
            if time.time() - start_time > timeout:
                return None
            time.sleep(0.01)
        
        with self.results_lock:
            return self.results.pop(request_id)
    
    async def _process_batch(self):
        """Process accumulated requests in batch (async)"""
        with self.queue_lock:
            if not self.queue:
                return
            
            # Take batch
            batch = []
            request_ids = []
            
            while self.queue and len(batch) < self.batch_size:
                data, req_id = self.queue.popleft()
                batch.append(data)
                request_ids.append(req_id)
            
            if not batch:
                return
        
        logger.info(f"Processing batch of {len(batch)} requests")
        
        try:
            # Process batch
            if self.process_func:
                if asyncio.iscoroutinefunction(self.process_func):
                    results = await self.process_func(batch)
                else:
                    results = self.process_func(batch)
            else:
                # Default: identity function
                results = batch
            
            # Store results
            with self.results_lock:
                for req_id, result in zip(request_ids, results):
                    self.results[req_id] = result
            
            self.batch_count += 1
            self.last_batch_time = time.time()
            
            logger.info(f"Batch processing completed: {len(results)} results")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            
            # Store error for all requests
            with self.results_lock:
                for req_id in request_ids:
                    self.results[req_id] = {"error": str(e)}
    
    def _process_batch_sync(self):
        """Process batch synchronously"""
        with self.queue_lock:
            if not self.queue:
                return
            
            batch = []
            request_ids = []
            
            while self.queue and len(batch) < self.batch_size:
                data, req_id = self.queue.popleft()
                batch.append(data)
                request_ids.append(req_id)
            
            if not batch:
                return
        
        logger.info(f"Processing batch of {len(batch)} requests (sync)")
        
        try:
            if self.process_func:
                results = self.process_func(batch)
            else:
                results = batch
            
            with self.results_lock:
                for req_id, result in zip(request_ids, results):
                    self.results[req_id] = result
            
            self.batch_count += 1
            self.last_batch_time = time.time()
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            with self.results_lock:
                for req_id in request_ids:
                    self.results[req_id] = {"error": str(e)}
    
    def _background_processor(self):
        """Background thread to process batches on timeout"""
        while self._running:
            time.sleep(0.1)
            
            # Check if we should process due to timeout
            with self.queue_lock:
                queue_size = len(self.queue)
                time_since_batch = time.time() - self.last_batch_time
            
            if queue_size > 0 and time_since_batch >= self.max_wait_time:
                logger.debug(f"Processing batch due to timeout ({queue_size} items waiting)")
                self._process_batch_sync()
    
    def stop(self):
        """Stop background processor"""
        self._running = False
        if self._processor_thread.is_alive():
            self._processor_thread.join(timeout=5)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics"""
        with self.queue_lock:
            queue_size = len(self.queue)
        
        with self.results_lock:
            results_pending = len(self.results)
        
        return {
            "queue_size": queue_size,
            "results_pending": results_pending,
            "total_requests": self.total_requests,
            "batch_count": self.batch_count,
            "avg_batch_size": self.total_requests / max(self.batch_count, 1),
            "time_since_last_batch": time.time() - self.last_batch_time
        }


class EmbeddingBatcher(RequestBatcher):
    """
    Specialized batcher for embedding generation
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        max_wait_time: float = 1.0,
        embedding_func: Optional[Callable] = None
    ):
        """
        Initialize embedding batcher
        
        Args:
            batch_size: Maximum embeddings per batch
            max_wait_time: Maximum wait time
            embedding_func: Function to generate embeddings
        """
        super().__init__(
            batch_size=batch_size,
            max_wait_time=max_wait_time,
            process_func=embedding_func or self._default_embedding_func
        )
    
    def _default_embedding_func(self, texts: List[str]) -> List[List[float]]:
        """
        Default embedding function (mock)
        Replace with actual API call
        """
        logger.warning("Using mock embeddings - replace with actual implementation")
        
        # Mock embeddings
        return [[0.1] * 1536 for _ in texts]
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Embed single text (batched)
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return await self.add_request_async(text)
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts (batched)
        
        Args:
            texts: List of texts
            
        Returns:
            List of embedding vectors
        """
        request_ids = []
        for text in texts:
            req_id = self.add_request(text)
            request_ids.append(req_id)
        
        # Wait for all results
        results = []
        for req_id in request_ids:
            result = await asyncio.wait_for(
                self._wait_for_result(req_id),
                timeout=self.max_wait_time + 5
            )
            results.append(result)
        
        return results
    
    async def _wait_for_result(self, request_id: str) -> Any:
        """Wait for result asynchronously"""
        while request_id not in self.results:
            await asyncio.sleep(0.01)
        
        with self.results_lock:
            return self.results.pop(request_id)


class APICallBatcher(RequestBatcher):
    """
    Generic batcher for API calls
    """
    
    def __init__(
        self,
        batch_size: int = 50,
        max_wait_time: float = 0.5,
        api_func: Optional[Callable] = None,
        rate_limit_per_second: Optional[int] = None
    ):
        """
        Initialize API call batcher
        
        Args:
            batch_size: Maximum calls per batch
            max_wait_time: Maximum wait time
            api_func: Function to call API
            rate_limit_per_second: Rate limit enforcement
        """
        self.rate_limit = rate_limit_per_second
        self.last_call_time = 0
        
        super().__init__(
            batch_size=batch_size,
            max_wait_time=max_wait_time,
            process_func=api_func or self._default_api_func
        )
    
    def _default_api_func(self, requests: List[Any]) -> List[Any]:
        """Default API function (mock)"""
        logger.warning("Using mock API calls - replace with actual implementation")
        return [{"result": f"Response for {r}"} for r in requests]
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting"""
        if not self.rate_limit:
            return
        
        min_interval = 1.0 / self.rate_limit
        time_since_last = time.time() - self.last_call_time
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()


if __name__ == "__main__":
    # Test batcher
    def process_batch(items):
        logger.info(f"Processing {len(items)} items")
        return [f"Result: {item}" for item in items]
    
    batcher = RequestBatcher(batch_size=5, max_wait_time=2.0, process_func=process_batch)
    
    # Add some requests
    req_ids = []
    for i in range(12):
        req_id = batcher.add_request(f"Request {i}")
        req_ids.append(req_id)
        logger.info(f"Added request {i}: {req_id}")
    
    # Get results
    for req_id in req_ids:
        result = batcher.get_result(req_id, timeout=5)
        logger.info(f"Result for {req_id}: {result}")
    
    # Get stats
    stats = batcher.get_stats()
    logger.info(f"Batcher stats: {stats}")
    
    batcher.stop()
