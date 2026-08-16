"""
Error Handler Middleware
========================
Centralized error handling for the Healthcare AI System.

Features:
1. Custom exception classes for different error types
2. Structured error responses
3. Error logging with context
4. Emergency escalation for critical errors
5. Graceful degradation

Error Categories:
- ValidationException: Invalid input data
- AuthenticationException: Auth failures
- RateLimitException: Rate limit exceeded
- EmergencyException: Emergency detection errors (critical)
- ServiceUnavailableException: External service failures
"""

import logging
import traceback
from typing import Optional, Dict, Any, Union
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.config import settings

logger = logging.getLogger("error_handler")


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class HealthcareAIException(Exception):
    """
    Base exception for Healthcare AI System.
    
    All custom exceptions should inherit from this.
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "HEALTHCARE_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        user_message_urdu: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.user_message = user_message or "An error occurred. Please try again."
        self.user_message_urdu = user_message_urdu or "ایک مسئلہ ہوا۔ براہ کرم دوبارہ کوشش کریں۔"
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for response."""
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.message if settings.DEBUG_MODE else self.user_message,
            "message_urdu": self.user_message_urdu,
            "details": self.details if settings.DEBUG_MODE else {},
            "timestamp": self.timestamp
        }


class EmergencyException(HealthcareAIException):
    """
    Exception for emergency-related errors.
    
    These are CRITICAL and must be logged/alerted immediately.
    System should never fail silently on emergency cases.
    """
    
    def __init__(
        self,
        message: str,
        patient_id: Optional[str] = None,
        symptoms: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="EMERGENCY_ERROR",
            status_code=500,
            details={
                "patient_id": patient_id,
                "symptoms": symptoms,
                **(details or {})
            },
            user_message="A system error occurred while processing your emergency. Please call 1122 immediately.",
            user_message_urdu="ایمرجنسی پراسیس کرتے وقت ایک مسئلہ ہوا۔ براہ کرم فوری طور پر 1122 کال کریں۔"
        )
        self.patient_id = patient_id
        self.symptoms = symptoms or []
        
        # Log critical error
        logger.critical(
            f"EMERGENCY EXCEPTION: {message} | "
            f"Patient: {patient_id} | Symptoms: {symptoms}"
        )


class ValidationException(HealthcareAIException):
    """Exception for input validation errors."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details={"field": field, **(details or {})},
            user_message=f"Invalid input: {message}",
            user_message_urdu=f"غلط ان پٹ: {message}"
        )
        self.field = field


class RateLimitException(HealthcareAIException):
    """Exception for rate limit exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = 60,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details={"retry_after_seconds": retry_after, **(details or {})},
            user_message=f"Too many requests. Please wait {retry_after} seconds.",
            user_message_urdu=f"بہت زیادہ درخواستیں۔ براہ کرم {retry_after} سیکنڈ انتظار کریں۔"
        )
        self.retry_after = retry_after


class AuthenticationException(HealthcareAIException):
    """Exception for authentication failures."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
            details=details,
            user_message="Authentication required.",
            user_message_urdu="تصدیق درکار ہے۔"
        )


class ServiceUnavailableException(HealthcareAIException):
    """Exception for external service failures."""
    
    def __init__(
        self,
        service_name: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message or f"Service unavailable: {service_name}",
            error_code="SERVICE_UNAVAILABLE",
            status_code=503,
            details={"service": service_name, **(details or {})},
            user_message="A service is temporarily unavailable. Please try again later.",
            user_message_urdu="ایک سروس عارضی طور پر دستیاب نہیں ہے۔ براہ کرم بعد میں کوشش کریں۔"
        )
        self.service_name = service_name


class SessionNotFoundException(HealthcareAIException):
    """Exception for session not found."""
    
    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session not found: {session_id}",
            error_code="SESSION_NOT_FOUND",
            status_code=404,
            details={"session_id": session_id},
            user_message="Session not found. Please start a new conversation.",
            user_message_urdu="سیشن نہیں ملا۔ براہ کرم نئی گفتگو شروع کریں۔"
        )


class TranslationException(HealthcareAIException):
    """Exception for translation failures."""
    
    def __init__(
        self,
        source_language: str,
        target_language: str,
        message: Optional[str] = None
    ):
        super().__init__(
            message=message or f"Translation failed: {source_language} to {target_language}",
            error_code="TRANSLATION_ERROR",
            status_code=500,
            details={
                "source_language": source_language,
                "target_language": target_language
            },
            user_message="Translation service error. Please try in a different language.",
            user_message_urdu="ترجمہ سروس میں خرابی۔ براہ کرم دوسری زبان میں کوشش کریں۔"
        )


class VoiceProcessingException(HealthcareAIException):
    """Exception for voice processing failures."""
    
    def __init__(
        self,
        operation: str,  # "transcription" or "synthesis"
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message or f"Voice {operation} failed",
            error_code="VOICE_PROCESSING_ERROR",
            status_code=500,
            details={"operation": operation, **(details or {})},
            user_message="Voice processing failed. Please try again or use text input.",
            user_message_urdu="وائس پراسیسنگ ناکام۔ براہ کرم دوبارہ کوشش کریں یا ٹیکسٹ استعمال کریں۔"
        )


# =============================================================================
# ERROR RESPONSE HELPERS
# =============================================================================

def error_response(
    status_code: int,
    error_code: str,
    message: str,
    message_urdu: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> JSONResponse:
    """
    Create a standardized error response.
    
    Args:
        status_code: HTTP status code
        error_code: Application error code
        message: Error message
        message_urdu: Error message in Urdu
        details: Additional error details
        request_id: Request ID for tracking
        
    Returns:
        JSONResponse with error information
    """
    content = {
        "error": True,
        "error_code": error_code,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    
    if message_urdu:
        content["message_urdu"] = message_urdu
    
    if details and settings.DEBUG_MODE:
        content["details"] = details
    
    if request_id:
        content["request_id"] = request_id
    
    return JSONResponse(
        status_code=status_code,
        content=content
    )


def log_error(
    error: Exception,
    request: Optional[Request] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log error with context information.
    
    Args:
        error: The exception
        request: FastAPI request object
        context: Additional context
    """
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.now().isoformat()
    }
    
    if request:
        error_info["request"] = {
            "method": request.method,
            "path": str(request.url.path),
            "client_ip": request.client.host if request.client else "unknown"
        }
        
        if hasattr(request.state, "request_id"):
            error_info["request_id"] = request.state.request_id
    
    if context:
        error_info["context"] = context
    
    # Log based on severity
    if isinstance(error, EmergencyException):
        logger.critical(f"EMERGENCY ERROR: {error_info}")
    elif isinstance(error, HealthcareAIException):
        logger.error(f"Application error: {error_info}")
    else:
        logger.error(f"Unhandled error: {error_info}", exc_info=True)


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

async def healthcare_exception_handler(
    request: Request,
    exc: HealthcareAIException
) -> JSONResponse:
    """Handle HealthcareAIException and subclasses."""
    log_error(exc, request)
    
    request_id = getattr(request.state, "request_id", None)
    
    response = error_response(
        status_code=exc.status_code,
        error_code=exc.error_code,
        message=exc.message if settings.DEBUG_MODE else exc.user_message,
        message_urdu=exc.user_message_urdu,
        details=exc.details,
        request_id=request_id
    )
    
    # Add retry-after header for rate limit errors
    if isinstance(exc, RateLimitException):
        response.headers["Retry-After"] = str(exc.retry_after)
    
    return response


async def emergency_exception_handler(
    request: Request,
    exc: EmergencyException
) -> JSONResponse:
    """
    Special handler for emergency exceptions.
    
    These are critical and require immediate attention.
    """
    # Log critical error
    logger.critical(
        f"EMERGENCY EXCEPTION | Patient: {exc.patient_id} | "
        f"Symptoms: {exc.symptoms} | Message: {exc.message}"
    )
    
    # In production, trigger alerts here
    # await send_emergency_alert(exc)
    
    request_id = getattr(request.state, "request_id", None)
    
    # Always include emergency action in response
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "error_code": "EMERGENCY_ERROR",
            "message": "System error during emergency processing",
            "message_urdu": "ایمرجنسی پراسیسنگ میں سسٹم ایرر",
            "emergency_action": "براہ کرم فوری طور پر 1122 کال کریں | Please call 1122 immediately",
            "call_emergency": True,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    errors = exc.errors()
    
    # Format validation errors
    formatted_errors = []
    for error in errors:
        field = ".".join(str(loc) for loc in error["loc"])
        formatted_errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })
    
    log_error(exc, request, {"validation_errors": formatted_errors})
    
    request_id = getattr(request.state, "request_id", None)
    
    return error_response(
        status_code=422,
        error_code="VALIDATION_ERROR",
        message="Request validation failed",
        message_urdu="درخواست کی توثیق ناکام",
        details={"errors": formatted_errors} if settings.DEBUG_MODE else None,
        request_id=request_id
    )


async def http_exception_handler(
    request: Request,
    exc: HTTPException
) -> JSONResponse:
    """Handle FastAPI HTTPException."""
    log_error(exc, request)
    
    request_id = getattr(request.state, "request_id", None)
    
    # Map status codes to error codes
    error_code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        408: "REQUEST_TIMEOUT",
        429: "RATE_LIMIT_EXCEEDED",
        500: "INTERNAL_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
        504: "GATEWAY_TIMEOUT"
    }
    
    error_code = error_code_map.get(exc.status_code, "HTTP_ERROR")
    
    return error_response(
        status_code=exc.status_code,
        error_code=error_code,
        message=str(exc.detail),
        request_id=request_id
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle all unhandled exceptions."""
    log_error(exc, request)
    
    # Log full traceback
    logger.error(f"Unhandled exception: {traceback.format_exc()}")
    
    request_id = getattr(request.state, "request_id", None)
    
    return error_response(
        status_code=500,
        error_code="INTERNAL_ERROR",
        message=str(exc) if settings.DEBUG_MODE else "An unexpected error occurred",
        message_urdu="ایک غیر متوقع خرابی ہوئی",
        details={"traceback": traceback.format_exc()} if settings.DEBUG_MODE else None,
        request_id=request_id
    )


# =============================================================================
# SETUP FUNCTION
# =============================================================================

def setup_error_handlers(app: FastAPI) -> None:
    """
    Set up all error handlers for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Custom exceptions
    app.add_exception_handler(EmergencyException, emergency_exception_handler)
    app.add_exception_handler(HealthcareAIException, healthcare_exception_handler)
    
    # FastAPI/Starlette exceptions
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # Catch-all for unhandled exceptions
    app.add_exception_handler(Exception, generic_exception_handler)
    
    logger.info("Error handlers configured")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_execute(
    func: callable,
    error_message: str = "Operation failed",
    default_return: Any = None,
    raise_on_error: bool = False
) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        error_message: Error message if function fails
        default_return: Value to return on error
        raise_on_error: Whether to raise exception on error
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func()
    except Exception as e:
        logger.error(f"{error_message}: {e}")
        if raise_on_error:
            raise HealthcareAIException(
                message=f"{error_message}: {str(e)}",
                error_code="OPERATION_FAILED"
            )
        return default_return


async def safe_execute_async(
    func: callable,
    error_message: str = "Operation failed",
    default_return: Any = None,
    raise_on_error: bool = False
) -> Any:
    """
    Safely execute an async function with error handling.
    
    Args:
        func: Async function to execute
        error_message: Error message if function fails
        default_return: Value to return on error
        raise_on_error: Whether to raise exception on error
        
    Returns:
        Function result or default_return on error
    """
    try:
        return await func()
    except Exception as e:
        logger.error(f"{error_message}: {e}")
        if raise_on_error:
            raise HealthcareAIException(
                message=f"{error_message}: {str(e)}",
                error_code="OPERATION_FAILED"
            )
        return default_return