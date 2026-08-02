"""
Utilities Module
================
Shared utility functions for the Healthcare AI System.

This module provides:
1. Prompt utilities
2. Schema definitions
3. Validation helpers
4. Logging utilities
5. Common helper functions

Design Philosophy:
- Pure functions where possible
- No side effects
- Comprehensive error handling
- Type hints for all functions
"""

from src.utils.prompts import (
    build_system_prompt,
    format_conversation_history,
    format_patient_context,
    get_phase_instructions,
    PromptTemplate,
    SYSTEM_PROMPTS
)

from src.utils.schemas import (
    PatientInfo,
    SymptomReport,
    TriageResult,
    SOAPNoteSchema,
    ConversationMessage,
    InterviewSummary,
    EmergencyAlert
)

from src.utils.validators import (
    validate_phone_number,
    validate_language_code,
    validate_session_id,
    validate_patient_age,
    sanitize_input,
    is_valid_symptom,
    ValidationResult
)

__all__ = [
    # Prompts
    "build_system_prompt",
    "format_conversation_history",
    "format_patient_context",
    "get_phase_instructions",
    "PromptTemplate",
    "SYSTEM_PROMPTS",
    
    # Schemas
    "PatientInfo",
    "SymptomReport",
    "TriageResult",
    "SOAPNoteSchema",
    "ConversationMessage",
    "InterviewSummary",
    "EmergencyAlert",
    
    # Validators
    "validate_phone_number",
    "validate_language_code",
    "validate_session_id",
    "validate_patient_age",
    "sanitize_input",
    "is_valid_symptom",
    "ValidationResult"
]


# =============================================================================
# COMMON UTILITIES
# =============================================================================

import re
import hashlib
import uuid
from typing import Optional, List, Dict, Any, TypeVar, Callable
from datetime import datetime, timedelta
from functools import wraps
import logging

logger = logging.getLogger("utils")

T = TypeVar('T')


def generate_session_id(prefix: str = "session") -> str:
    """
    Generate a unique session ID.
    
    Args:
        prefix: Prefix for the session ID
        
    Returns:
        Unique session ID string
    """
    unique_part = uuid.uuid4().hex[:12]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{timestamp}_{unique_part}"


def generate_message_id() -> str:
    """Generate a unique message ID."""
    return f"msg_{uuid.uuid4().hex[:16]}"


def hash_phone_number(phone: str) -> str:
    """
    Hash a phone number for privacy-safe logging.
    
    Args:
        phone: Phone number
        
    Returns:
        Hashed representation
    """
    return hashlib.sha256(phone.encode()).hexdigest()[:12]


def mask_sensitive_data(data: str, visible_chars: int = 4) -> str:
    """
    Mask sensitive data for logging.
    
    Args:
        data: Sensitive data string
        visible_chars: Number of characters to keep visible at end
        
    Returns:
        Masked string
    """
    if len(data) <= visible_chars:
        return "*" * len(data)
    return "*" * (len(data) - visible_chars) + data[-visible_chars:]


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    return " ".join(text.split())


def remove_special_characters(text: str, keep_urdu: bool = True) -> str:
    """
    Remove special characters from text.
    
    Args:
        text: Input text
        keep_urdu: Whether to keep Urdu/Arabic script characters
        
    Returns:
        Cleaned text
    """
    if keep_urdu:
        # Keep alphanumeric, spaces, and Urdu script
        pattern = r'[^\w\s\u0600-\u06FF\u0750-\u077F]'
    else:
        pattern = r'[^\w\s]'
    
    return re.sub(pattern, '', text)


def calculate_age(birth_date: datetime) -> int:
    """
    Calculate age from birth date.
    
    Args:
        birth_date: Date of birth
        
    Returns:
        Age in years
    """
    today = datetime.now()
    age = today.year - birth_date.year
    
    # Adjust if birthday hasn't occurred this year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1
    
    return age


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime to string.
    
    Args:
        dt: Datetime object
        format_str: Format string
        
    Returns:
        Formatted datetime string
    """
    return dt.strftime(format_str)


def parse_datetime(dt_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
    """
    Parse datetime from string.
    
    Args:
        dt_str: Datetime string
        format_str: Format string
        
    Returns:
        Datetime object or None if parsing fails
    """
    try:
        return datetime.strptime(dt_str, format_str)
    except ValueError:
        return None


def time_ago(dt: datetime) -> str:
    """
    Get human-readable time ago string.
    
    Args:
        dt: Datetime to compare
        
    Returns:
        Human-readable string (e.g., "5 minutes ago")
    """
    now = datetime.now()
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        return dt.strftime("%Y-%m-%d")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import time
            
            last_exception = None
            delay = base_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * 2, max_delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
            
            raise last_exception
        
        return wrapper
    return decorator


async def retry_with_backoff_async(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum retries
        base_delay: Base delay
        max_delay: Maximum delay
        exceptions: Exceptions to catch
        
    Returns:
        Function result
    """
    import asyncio
    
    last_exception = None
    delay = base_delay
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    f"Async attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
            else:
                logger.error(f"All {max_retries + 1} async attempts failed")
    
    raise last_exception


def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split a list into chunks.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary to merge (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def safe_get(d: Dict, keys: str, default: Any = None, sep: str = '.') -> Any:
    """
    Safely get nested dictionary value.
    
    Args:
        d: Dictionary
        keys: Dot-separated key path
        default: Default value if not found
        sep: Separator for keys
        
    Returns:
        Value or default
    """
    try:
        for key in keys.split(sep):
            d = d[key]
        return d
    except (KeyError, TypeError):
        return default


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

class ContextLogger:
    """
    Logger with context support.
    
    Automatically includes context in all log messages.
    """
    
    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(name)
        self.context = context or {}
    
    def with_context(self, **kwargs) -> 'ContextLogger':
        """Create new logger with additional context."""
        new_context = {**self.context, **kwargs}
        new_logger = ContextLogger(self.logger.name, new_context)
        return new_logger
    
    def _format_message(self, message: str) -> str:
        """Format message with context."""
        if self.context:
            context_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{message} | {context_str}"
        return message
    
    def debug(self, message: str, **kwargs) -> None:
        self.logger.debug(self._format_message(message), **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        self.logger.info(self._format_message(message), **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        self.logger.warning(self._format_message(message), **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        self.logger.error(self._format_message(message), **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        self.logger.critical(self._format_message(message), **kwargs)


def get_context_logger(
    name: str,
    session_id: Optional[str] = None,
    patient_id: Optional[str] = None
) -> ContextLogger:
    """
    Get a context-aware logger.
    
    Args:
        name: Logger name
        session_id: Session ID for context
        patient_id: Patient ID for context
        
    Returns:
        ContextLogger instance
    """
    context = {}
    if session_id:
        context["session_id"] = session_id
    if patient_id:
        context["patient_id"] = mask_sensitive_data(patient_id)
    
    return ContextLogger(name, context)