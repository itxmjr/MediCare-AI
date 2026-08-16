"""
Middleware Module
=================
FastAPI middleware components for the Healthcare AI System.

Middleware provides:
1. Error handling and logging
2. Request/Response logging
3. Rate limiting
4. Authentication (optional)
5. Request validation
6. Performance monitoring

Middleware Order (execution from top to bottom on request, reverse on response):
1. CORS (handled by FastAPI)
2. Request logging
3. Rate limiting
4. Authentication
5. Error handling
6. Route handlers
"""

from src.api.middleware.error_handler import (
    setup_error_handlers,
    HealthcareAIException,
    EmergencyException,
    ValidationException,
    RateLimitException,
    AuthenticationException,
    ServiceUnavailableException,
    error_response
)

__all__ = [
    # Setup function
    "setup_error_handlers",
    
    # Custom exceptions
    "HealthcareAIException",
    "EmergencyException",
    "ValidationException",
    "RateLimitException",
    "AuthenticationException",
    "ServiceUnavailableException",
    
    # Utilities
    "error_response"
]


# =============================================================================
# MIDDLEWARE UTILITIES
# =============================================================================

from typing import Callable, Optional
from functools import wraps
import time
import logging
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger("middleware")


class RateLimiter:
    """
    Simple in-memory rate limiter.
    
    For production, use Redis-based rate limiting.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self._requests: dict = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        now = datetime.now()
        
        # Clean old entries
        self._requests[identifier] = [
            t for t in self._requests[identifier]
            if now - t < timedelta(hours=1)
        ]
        
        requests = self._requests[identifier]
        
        # Check per-minute limit
        minute_ago = now - timedelta(minutes=1)
        recent_requests = [t for t in requests if t > minute_ago]
        if len(recent_requests) >= self.rpm:
            return False
        
        # Check per-hour limit
        if len(requests) >= self.rph:
            return False
        
        # Record request
        self._requests[identifier].append(now)
        return True
    
    def get_retry_after(self, identifier: str) -> int:
        """Get seconds until next allowed request."""
        if identifier not in self._requests:
            return 0
        
        now = datetime.now()
        requests = self._requests[identifier]
        
        if not requests:
            return 0
        
        # Find oldest request in the minute window
        minute_ago = now - timedelta(minutes=1)
        recent = [t for t in requests if t > minute_ago]
        
        if len(recent) >= self.rpm and recent:
            oldest = min(recent)
            retry_after = 60 - (now - oldest).seconds
            return max(0, retry_after)
        
        return 0


class RequestLogger:
    """
    Request/Response logger for monitoring.
    """
    
    def __init__(self, log_body: bool = False):
        self.log_body = log_body
        self.logger = logging.getLogger("request_logger")
    
    def log_request(
        self,
        request_id: str,
        method: str,
        path: str,
        client_ip: str,
        user_agent: Optional[str] = None,
        body_size: int = 0
    ) -> None:
        """Log incoming request."""
        self.logger.info(
            f"[{request_id}] → {method} {path} | "
            f"IP: {client_ip} | Size: {body_size}B"
        )
    
    def log_response(
        self,
        request_id: str,
        status_code: int,
        duration_ms: float,
        body_size: int = 0
    ) -> None:
        """Log outgoing response."""
        level = logging.INFO if status_code < 400 else logging.WARNING
        self.logger.log(
            level,
            f"[{request_id}] ← Status: {status_code} | "
            f"Duration: {duration_ms:.1f}ms | Size: {body_size}B"
        )


class PerformanceMonitor:
    """
    Simple performance monitoring.
    """
    
    def __init__(self):
        self._metrics: dict = defaultdict(list)
    
    def record(self, endpoint: str, duration_ms: float, status_code: int) -> None:
        """Record a request metric."""
        self._metrics[endpoint].append({
            "duration_ms": duration_ms,
            "status_code": status_code,
            "timestamp": datetime.now()
        })
        
        # Keep only last 1000 records per endpoint
        if len(self._metrics[endpoint]) > 1000:
            self._metrics[endpoint] = self._metrics[endpoint][-1000:]
    
    def get_stats(self, endpoint: Optional[str] = None) -> dict:
        """Get performance statistics."""
        if endpoint:
            return self._calculate_stats(self._metrics.get(endpoint, []))
        
        return {
            ep: self._calculate_stats(metrics)
            for ep, metrics in self._metrics.items()
        }
    
    def _calculate_stats(self, metrics: list) -> dict:
        """Calculate statistics for metrics list."""
        if not metrics:
            return {"count": 0}
        
        durations = [m["duration_ms"] for m in metrics]
        status_codes = [m["status_code"] for m in metrics]
        
        return {
            "count": len(metrics),
            "avg_duration_ms": sum(durations) / len(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "success_rate": sum(1 for s in status_codes if s < 400) / len(status_codes),
            "error_count": sum(1 for s in status_codes if s >= 400)
        }


# Global instances
rate_limiter = RateLimiter()
request_logger = RequestLogger()
performance_monitor = PerformanceMonitor()


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter."""
    return rate_limiter


def get_request_logger() -> RequestLogger:
    """Get the global request logger."""
    return request_logger


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return performance_monitor