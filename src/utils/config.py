"""
Configuration management for the AI integration system.
Handles loading and validation of environment variables.
"""
import os
from typing import Optional
from dotenv import load_dotenv
from dataclasses import dataclass

# Load environment variables
load_dotenv()


@dataclass
class AIConfig:
    """Configuration for AI APIs"""
    anthropic_api_key: str
    openai_api_key: str
    default_model: str = "claude-sonnet-4-20250514"
    fast_model: str = "claude-haiku-3-5-20250514"
    advanced_model: str = "claude-opus-4-20250514"


@dataclass
class TripletexConfig:
    """Configuration for Tripletex API"""
    employee_token: str
    consumer_token: str
    base_url: str = "https://api.tripletex.io/v2"


@dataclass
class DatabaseConfig:
    """Configuration for databases"""
    chroma_db_path: str = "./chroma_db"
    persistent_storage: bool = True


@dataclass
class SecurityConfig:
    """Security configuration"""
    max_input_length: int = 5000
    enable_prompt_injection_check: bool = True


@dataclass
class PerformanceConfig:
    """Performance and rate limiting configuration"""
    max_requests_per_minute: int = 50
    max_requests_per_hour: int = 1000
    cache_ttl_seconds: int = 1800
    cache_max_size: int = 1000


class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.ai = self._load_ai_config()
        self.tripletex = self._load_tripletex_config()
        self.database = self._load_database_config()
        self.security = self._load_security_config()
        self.performance = self._load_performance_config()
    
    def _load_ai_config(self) -> AIConfig:
        """Load AI configuration from environment"""
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        return AIConfig(
            anthropic_api_key=anthropic_key,
            openai_api_key=openai_key,
            default_model=os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514"),
            fast_model=os.getenv("FAST_MODEL", "claude-haiku-3-5-20250514"),
            advanced_model=os.getenv("ADVANCED_MODEL", "claude-opus-4-20250514"),
        )
    
    def _load_tripletex_config(self) -> Optional[TripletexConfig]:
        """Load Tripletex configuration (optional)"""
        employee_token = os.getenv("TRIPLETEX_EMPLOYEE_TOKEN")
        consumer_token = os.getenv("TRIPLETEX_CONSUMER_TOKEN")
        
        if not employee_token or not consumer_token:
            return None
        
        return TripletexConfig(
            employee_token=employee_token,
            consumer_token=consumer_token,
            base_url=os.getenv("TRIPLETEX_BASE_URL", "https://api.tripletex.io/v2")
        )
    
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration"""
        return DatabaseConfig(
            chroma_db_path=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
            persistent_storage=os.getenv("PERSISTENT_STORAGE", "true").lower() == "true"
        )
    
    def _load_security_config(self) -> SecurityConfig:
        """Load security configuration"""
        return SecurityConfig(
            max_input_length=int(os.getenv("MAX_INPUT_LENGTH", 5000)),
            enable_prompt_injection_check=os.getenv(
                "ENABLE_PROMPT_INJECTION_CHECK", "true"
            ).lower() == "true"
        )
    
    def _load_performance_config(self) -> PerformanceConfig:
        """Load performance configuration"""
        return PerformanceConfig(
            max_requests_per_minute=int(os.getenv("MAX_REQUESTS_PER_MINUTE", 50)),
            max_requests_per_hour=int(os.getenv("MAX_REQUESTS_PER_HOUR", 1000)),
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", 1800)),
            cache_max_size=int(os.getenv("CACHE_MAX_SIZE", 1000))
        )


# Global config instance
config = Config()
