"""
Kapittel 10.5: Celery Worker Pool for Async Processing
Queue-based architecture for handling background tasks at scale.
"""
import os
from celery import Celery
from celery.result import AsyncResult
from celery.events import EventReceiver
from kombu import Connection
import redis
from typing import Dict, List, Any
from datetime import datetime

try:
    from utils import logger
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from utils import logger


# Celery app configuration
celery_app = Celery(
    'ai_workers',
    broker=os.getenv('CELERY_BROKER', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_BACKEND', 'redis://localhost:6379/1')
)

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='Europe/Oslo',
    enable_utc=True,
    
    # Worker configuration for scale
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
    
    # Task routing - different queues for different workload types
    task_routes={
        'scaling.celery_workers.generate_embedding': {'queue': 'embeddings'},
        'scaling.celery_workers.process_large_document': {'queue': 'heavy'},
        'scaling.celery_workers.chat_response_async': {'queue': 'fast'},
    },
    
    # Result expiry
    result_expires=3600,
    
    # Task execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    name='scaling.celery_workers.process_large_document'
)
def process_large_document(self, document_id: str) -> Dict[str, Any]:
    """
    Process large document in background
    
    Args:
        document_id: ID of document to process
        
    Returns:
        Processing result with status and metadata
    """
    try:
        logger.info(f"Processing document {document_id}")
        
        # Mock document retrieval - replace with actual implementation
        # doc = Document.get(document_id)
        doc_content = f"Document {document_id} content..."
        
        # Simulate chunking
        chunks = [doc_content[i:i+1000] for i in range(0, len(doc_content), 1000)]
        
        logger.info(f"Document split into {len(chunks)} chunks")
        
        # Generate embeddings for all chunks (parallel)
        embedding_tasks = [
            generate_embedding.delay(chunk, i)
            for i, chunk in enumerate(chunks)
        ]
        
        # Wait for all embeddings
        embeddings = []
        for task in embedding_tasks:
            try:
                result = task.get(timeout=30)
                embeddings.append(result)
            except Exception as e:
                logger.error(f"Failed to get embedding: {e}")
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Store in vector DB - mock implementation
        # vector_db.add_embeddings(document_id, embeddings)
        
        # Update document status - mock implementation
        # doc.status = "processed"
        # doc.save()
        
        return {
            "status": "success",
            "document_id": document_id,
            "chunks": len(chunks),
            "embeddings": len(embeddings),
            "processed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)


@celery_app.task(
    queue='embeddings',
    name='scaling.celery_workers.generate_embedding'
)
def generate_embedding(text: str, chunk_id: int) -> Dict[str, Any]:
    """
    Generate embedding for text chunk
    
    Args:
        text: Text to embed
        chunk_id: ID of chunk
        
    Returns:
        Embedding result
    """
    try:
        logger.debug(f"Generating embedding for chunk {chunk_id}")
        
        # Mock embedding generation - replace with actual API call
        # embedding = openai.embeddings.create(
        #     model="text-embedding-3-small",
        #     input=text
        # ).data[0].embedding
        
        # Mock embedding
        embedding = [0.1] * 1536  # Mock 1536-dimensional embedding
        
        return {
            "chunk_id": chunk_id,
            "embedding": embedding,
            "text_length": len(text)
        }
        
    except Exception as e:
        logger.error(f"Failed to generate embedding for chunk {chunk_id}: {e}")
        raise


@celery_app.task(
    queue='fast',
    time_limit=30,  # Kill if takes >30s
    soft_time_limit=25,  # Warning at 25s
    name='scaling.celery_workers.chat_response_async'
)
def chat_response_async(message: str, context: Dict, user_id: str) -> str:
    """
    Fast async chat response
    
    Args:
        message: User message
        context: Conversation context
        user_id: User ID
        
    Returns:
        AI response
    """
    try:
        logger.info(f"Processing async chat for user {user_id}")
        
        # Check rate limit - mock implementation
        # if not rate_limiter.allow(user_id):
        #     raise RateLimitExceeded()
        
        # Generate response - mock implementation
        # response = anthropic.messages.create(
        #     model="claude-sonnet-4-20250514",
        #     max_tokens=1000,
        #     messages=[{"role": "user", "content": message}]
        # )
        
        # Mock response
        response_text = f"Response to: {message[:50]}..."
        
        # Store conversation - mock implementation
        # store_conversation(user_id, message, response_text)
        
        logger.info(f"Chat response completed for user {user_id}")
        
        return response_text
        
    except Exception as e:
        logger.error(f"Failed to generate chat response for user {user_id}: {e}")
        raise


@celery_app.task(name='scaling.celery_workers.batch_process_users')
def batch_process_users(user_ids: List[str]) -> Dict[str, Any]:
    """
    Batch process multiple users
    
    Args:
        user_ids: List of user IDs to process
        
    Returns:
        Processing results
    """
    results = {
        "total": len(user_ids),
        "success": 0,
        "failed": 0,
        "errors": []
    }
    
    for user_id in user_ids:
        try:
            # Process user - mock implementation
            logger.info(f"Processing user {user_id}")
            results["success"] += 1
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({
                "user_id": user_id,
                "error": str(e)
            })
            logger.error(f"Failed to process user {user_id}: {e}")
    
    return results


# Monitoring functions

def monitor_celery_events(app: Celery):
    """
    Monitor Celery events and export metrics
    
    This function should be run in a separate process to monitor
    all Celery events and collect metrics.
    """
    try:
        # Mock Prometheus metrics - uncomment when prometheus_client is installed
        # from prometheus_client import Counter, Histogram
        
        # worker_tasks_total = Counter(
        #     'celery_tasks_total',
        #     'Total tasks processed',
        #     ['queue', 'status']
        # )
        
        # worker_task_duration = Histogram(
        #     'celery_task_duration_seconds',
        #     'Task duration',
        #     ['task_name', 'queue']
        # )
        
        def on_task_succeeded(event):
            logger.info(f"Task succeeded: {event['name']} in queue {event.get('queue', 'default')}")
            # worker_tasks_total.labels(
            #     queue=event.get('queue', 'default'),
            #     status='success'
            # ).inc()
            
            # worker_task_duration.labels(
            #     task_name=event['name'],
            #     queue=event.get('queue', 'default')
            # ).observe(event.get('runtime', 0))
        
        def on_task_failed(event):
            logger.error(f"Task failed: {event['name']} in queue {event.get('queue', 'default')}")
            # worker_tasks_total.labels(
            #     queue=event.get('queue', 'default'),
            #     status='failed'
            # ).inc()
        
        def on_task_retried(event):
            logger.warning(f"Task retried: {event['name']}")
        
        with Connection(app.broker_connection()) as conn:
            recv = EventReceiver(
                conn,
                handlers={
                    'task-succeeded': on_task_succeeded,
                    'task-failed': on_task_failed,
                    'task-retried': on_task_retried,
                }
            )
            logger.info("Starting Celery event monitor")
            recv.capture(limit=None, timeout=None)
            
    except Exception as e:
        logger.error(f"Error in event monitor: {e}")


def get_queue_stats() -> Dict[str, Any]:
    """
    Get statistics for all queues
    
    Returns:
        Queue statistics including pending tasks, active workers, etc.
    """
    try:
        inspect = celery_app.control.inspect()
        
        stats = {
            "active": inspect.active(),
            "scheduled": inspect.scheduled(),
            "reserved": inspect.reserved(),
            "stats": inspect.stats(),
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # For testing purposes
    logger.info("Celery workers module loaded")
    logger.info(f"Configured queues: {list(celery_app.conf.task_routes.keys())}")
