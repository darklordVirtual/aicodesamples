"""
Security utilities for input validation and sanitization.
Implements protection against prompt injection and data validation.
"""
import re
from typing import Optional
from .logging_config import logger


class SecurityValidator:
    """Security validation and sanitization"""
    
    # Dangerous patterns that might indicate prompt injection
    DANGEROUS_PATTERNS = [
        r"ignore (previous|all) (instructions|prompts|rules)",
        r"you are now",
        r"new (instructions|system|role):",
        r"system:",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"assistant:",
        r"disregard",
        r"forget (everything|all|previous)",
    ]
    
    @staticmethod
    def sanitize_input(user_input: str, max_length: int = 5000) -> str:
        """
        Sanitize and validate user input.
        
        Args:
            user_input: Input string to sanitize
            max_length: Maximum allowed length
        
        Returns:
            Sanitized input
        
        Raises:
            ValueError: If input is invalid
        """
        # Check length
        if len(user_input) > max_length:
            raise ValueError(f"Input too long: {len(user_input)} > {max_length} characters")
        
        # Remove null bytes
        sanitized = user_input.replace('\x00', '')
        
        # Strip leading/trailing whitespace
        sanitized = sanitized.strip()
        
        return sanitized
    
    @staticmethod
    def check_prompt_injection(user_input: str) -> bool:
        """
        Check for potential prompt injection attempts.
        
        Args:
            user_input: Input to check
        
        Returns:
            True if injection detected, False otherwise
        """
        user_input_lower = user_input.lower()
        
        for pattern in SecurityValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, user_input_lower):
                logger.warning(f"Prompt injection attempt detected: {user_input[:100]}")
                return True
        
        return False
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format.
        
        Args:
            email: Email address to validate
        
        Returns:
            True if valid, False otherwise
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_org_number(org_nr: str) -> bool:
        """
        Validate Norwegian organization number (9 digits).
        
        Args:
            org_nr: Organization number to validate
        
        Returns:
            True if valid, False otherwise
        """
        # Remove spaces and common formatting
        org_nr = org_nr.replace(' ', '').replace('-', '')
        return bool(re.match(r'^\d{9}$', org_nr))
    
    @staticmethod
    def mask_secret(secret: str, show_chars: int = 4) -> str:
        """
        Mask a secret for safe logging.
        
        Args:
            secret: Secret to mask
            show_chars: Number of characters to show
        
        Returns:
            Masked secret
        """
        if len(secret) <= show_chars:
            return "*" * len(secret)
        return secret[:show_chars] + "*" * (len(secret) - show_chars)


def safe_ai_input(user_input: str, max_length: int = 5000) -> str:
    """
    Safely process user input for AI queries.
    
    Args:
        user_input: Raw user input
        max_length: Maximum allowed length
    
    Returns:
        Sanitized input
    
    Raises:
        ValueError: If input is invalid or contains injection attempts
    """
    # Sanitize
    sanitized = SecurityValidator.sanitize_input(user_input, max_length)
    
    # Check for injection
    if SecurityValidator.check_prompt_injection(sanitized):
        raise ValueError("Invalid input detected")
    
    return sanitized
