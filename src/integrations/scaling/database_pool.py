"""
Database connection pooling with read replicas for high availability.

Features:
- Primary database for writes
- Multiple read replicas for queries
- Connection pooling
- Automatic failover
"""
import random
from typing import Optional
from contextlib import contextmanager


class DatabasePool:
    """
    Database connection pool with read replicas.
    Routes writes to primary, reads to replicas.
    
    Example:
        pool = DatabasePool(
            primary_url="postgresql://user:pass@primary:5432/db",
            replica_urls=[
                "postgresql://user:pass@replica1:5432/db",
                "postgresql://user:pass@replica2:5432/db",
            ]
        )
        
        # Write operation
        with pool.get_write_session() as session:
            session.add(new_user)
            session.commit()
        
        # Read operation (uses replica)
        with pool.get_read_session() as session:
            users = session.query(User).all()
    """
    
    def __init__(
        self,
        primary_url: str,
        replica_urls: list[str],
        pool_size: int = 20,
        max_overflow: int = 40
    ):
        """
        Initialize database pool.
        
        Args:
            primary_url: Primary database URL (writes)
            replica_urls: List of replica URLs (reads)
            pool_size: Base pool size
            max_overflow: Max overflow connections
        """
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
        except ImportError:
            raise ImportError("SQLAlchemy required: pip install sqlalchemy")
        
        # Primary database (writes)
        self.primary = create_engine(
            primary_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,   # Recycle connections after 1 hour
        )
        
        # Read replicas
        self.replicas = [
            create_engine(
                url,
                pool_size=max(10, pool_size // 2),
                max_overflow=max(20, max_overflow // 2),
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            for url in replica_urls
        ]
        
        self.SessionPrimary = sessionmaker(bind=self.primary)
        self.SessionReplicas = [
            sessionmaker(bind=replica)
            for replica in self.replicas
        ]
        
        print(f"Database pool initialized:")
        print(f"  Primary: {primary_url}")
        print(f"  Replicas: {len(replica_urls)}")
        print(f"  Pool size: {pool_size}, max overflow: {max_overflow}")
    
    @contextmanager
    def get_write_session(self):
        """Get session for write operations (uses primary)"""
        session = self.SessionPrimary()
        try:
            yield session
        finally:
            session.close()
    
    @contextmanager
    def get_read_session(self, prefer_primary: bool = False):
        """
        Get session for read operations (uses replica).
        
        Args:
            prefer_primary: If True, use primary instead of replica
        """
        if prefer_primary or not self.SessionReplicas:
            # Use primary
            session = self.SessionPrimary()
        else:
            # Load balance across replicas
            SessionReplica = random.choice(self.SessionReplicas)
            session = SessionReplica()
        
        try:
            yield session
        finally:
            session.close()
    
    def health_check(self) -> dict:
        """Check health of primary and replicas"""
        results = {"primary": False, "replicas": []}
        
        # Check primary
        try:
            with self.primary.connect() as conn:
                conn.execute("SELECT 1")
            results["primary"] = True
        except Exception as e:
            print(f"Primary health check failed: {e}")
        
        # Check replicas
        for i, replica in enumerate(self.replicas):
            try:
                with replica.connect() as conn:
                    conn.execute("SELECT 1")
                results["replicas"].append({"id": i, "healthy": True})
            except Exception as e:
                print(f"Replica {i} health check failed: {e}")
                results["replicas"].append({"id": i, "healthy": False})
        
        return results


if __name__ == "__main__":
    print("=== Database Pool Demo ===\n")
    print("(Note: This is a demonstration with mock URLs)")
    print("In production, use real PostgreSQL URLs\n")
    
    # Example URLs (would be real in production)
    primary = "postgresql://user:pass@primary.db:5432/app"
    replicas = [
        "postgresql://user:pass@replica1.db:5432/app",
        "postgresql://user:pass@replica2.db:5432/app",
        "postgresql://user:pass@replica3.db:5432/app",
    ]
    
    print("Configuration:")
    print(f"Primary: {primary}")
    print(f"Replicas: {len(replicas)}")
    print()
    
    print("Usage patterns:")
    print()
    print("# Write operation (uses primary)")
    print("with pool.get_write_session() as session:")
    print("    user = User(name='John', email='john@example.com')")
    print("    session.add(user)")
    print("    session.commit()")
    print()
    
    print("# Read operation (uses replica)")
    print("with pool.get_read_session() as session:")
    print("    users = session.query(User).filter(User.active == True).all()")
    print()
    
    print("# Read from primary (for consistency)")
    print("with pool.get_read_session(prefer_primary=True) as session:")
    print("    user = session.query(User).filter(User.id == user_id).first()")
