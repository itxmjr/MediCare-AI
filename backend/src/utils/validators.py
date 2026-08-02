"""
Validation Utilities
====================
Input validation and sanitization functions.

This module provides:
1. Phone number validation
2. Language code validation
3. Session ID validation
4. Patient data validation
5. Input sanitization
6. Medical data validation

All validators return structured results with error details.
"""

import re
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.config import SupportedLanguage

logger = logging.getLogger("validators")


# =============================================================================
# VALIDATION RESULT
# =============================================================================

@dataclass
class ValidationResult:
    """
    Result of a validation operation.
    
    Provides structured information about validation success/failure.
    """
    
    is_valid: bool
    value: Any = None  # Cleaned/normalized value if valid
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    field_name: Optional[str] = None
    
    def add_error(self, message: str) -> 'ValidationResult':
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
        return self
    
    def add_warning(self, message: str) -> 'ValidationResult':
        """Add a warning message."""
        self.warnings.append(message)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "value": self.value,
            "errors": self.errors,
            "warnings": self.warnings,
            "field_name": self.field_name
        }
    
    @staticmethod
    def valid(value: Any, field_name: Optional[str] = None) -> 'ValidationResult':
        """Create a valid result."""
        return ValidationResult(is_valid=True, value=value, field_name=field_name)
    
    @staticmethod
    def invalid(error: str, field_name: Optional[str] = None) -> 'ValidationResult':
        """Create an invalid result."""
        return ValidationResult(is_valid=False, errors=[error], field_name=field_name)


# =============================================================================
# PHONE NUMBER VALIDATION
# =============================================================================

def validate_phone_number(
    phone: str,
    country_code: str = "92",
    require_country_code: bool = False
) -> ValidationResult:
    """
    Validate and normalize a phone number.
    
    Supports Pakistani phone numbers by default.
    
    Args:
        phone: Phone number string
        country_code: Expected country code (default: 92 for Pakistan)
        require_country_code: Whether country code is required
        
    Returns:
        ValidationResult with normalized phone number
    """
    result = ValidationResult(is_valid=True, field_name="phone_number")
    
    if not phone:
        return result.add_error("Phone number is required")
    
    # Remove all non-digit characters except leading +
    cleaned = phone.strip()
    has_plus = cleaned.startswith('+')
    digits = re.sub(r'\D', '', cleaned)
    
    # Check minimum length
    if len(digits) < 10:
        return result.add_error("Phone number too short (minimum 10 digits)")
    
    # Check maximum length
    if len(digits) > 15:
        return result.add_error("Phone number too long (maximum 15 digits)")
    
    # Normalize Pakistani numbers
    if country_code == "92":
        # Handle different formats
        if digits.startswith('92'):
            # Already has country code
            normalized = '+' + digits
        elif digits.startswith('0'):
            # Local format (03xx)
            normalized = '+92' + digits[1:]
        elif len(digits) == 10:
            # Without leading zero
            normalized = '+92' + digits
        else:
            if require_country_code:
                return result.add_error("Invalid Pakistani phone number format")
            normalized = '+' + digits
            result.add_warning("Country code could not be verified")
        
        # Validate Pakistani mobile number pattern
        pk_pattern = r'^\+92[3][0-9]{9}$'
        if not re.match(pk_pattern, normalized):
            result.add_warning("Number may not be a valid Pakistani mobile number")
    else:
        # Generic international format
        if has_plus or digits.startswith(country_code):
            normalized = '+' + digits.lstrip(country_code) if not digits.startswith(country_code) else '+' + digits
        else:
            normalized = '+' + country_code + digits
    
    result.value = normalized
    return result


# =============================================================================
# LANGUAGE VALIDATION
# =============================================================================

def validate_language_code(
    code: str,
    allow_auto_detect: bool = True
) -> ValidationResult:
    """
    Validate a language code.
    
    Args:
        code: Language code (e.g., 'en', 'ur', 'pa')
        allow_auto_detect: Whether to allow 'auto' as a value
        
    Returns:
        ValidationResult with SupportedLanguage value
    """
    result = ValidationResult(is_valid=True, field_name="language")
    
    if not code:
        return result.add_error("Language code is required")
    
    code = code.strip().lower()
    
    # Handle auto-detect
    if code == 'auto':
        if allow_auto_detect:
            result.value = None  # None means auto-detect
            return result
        else:
            return result.add_error("Auto-detect not allowed")
    
    # Check if valid language
    try:
        language = SupportedLanguage(code)
        result.value = language
        return result
    except ValueError:
        valid_codes = [lang.value for lang in SupportedLanguage]
        return result.add_error(
            f"Invalid language code '{code}'. Valid codes: {', '.join(valid_codes)}"
        )


# =============================================================================
# SESSION VALIDATION
# =============================================================================

def validate_session_id(
    session_id: str,
    prefix: Optional[str] = None
) -> ValidationResult:
    """
    Validate a session ID.
    
    Args:
        session_id: Session ID string
        prefix: Expected prefix (e.g., 'session_')
        
    Returns:
        ValidationResult with validated session ID
    """
    result = ValidationResult(is_valid=True, field_name="session_id")
    
    if not session_id:
        return result.add_error("Session ID is required")
    
    session_id = session_id.strip()
    
    # Check length
    if len(session_id) < 8:
        return result.add_error("Session ID too short (minimum 8 characters)")
    
    if len(session_id) > 100:
        return result.add_error("Session ID too long (maximum 100 characters)")
    
    # Check prefix if specified
    if prefix and not session_id.startswith(prefix):
        return result.add_error(f"Session ID must start with '{prefix}'")
    
    # Check for valid characters (alphanumeric, underscore, hyphen)
    if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        return result.add_error("Session ID contains invalid characters")
    
    result.value = session_id
    return result


# =============================================================================
# PATIENT DATA VALIDATION
# =============================================================================

def validate_patient_age(
    age: Union[int, str, None],
    required: bool = False
) -> ValidationResult:
    """
    Validate patient age.
    
    Args:
        age: Age value (int or string)
        required: Whether age is required
        
    Returns:
        ValidationResult with validated age
    """
    result = ValidationResult(is_valid=True, field_name="age")
    
    if age is None:
        if required:
            return result.add_error("Age is required")
        result.value = None
        return result
    
    # Convert to int if string
    if isinstance(age, str):
        age = age.strip()
        if not age:
            if required:
                return result.add_error("Age is required")
            result.value = None
            return result
        
        try:
            age = int(age)
        except ValueError:
            return result.add_error("Age must be a number")
    
    # Validate range
    if age < 0:
        return result.add_error("Age cannot be negative")
    
    if age > 150:
        return result.add_error("Age cannot exceed 150 years")
    
    # Warnings for edge cases
    if age == 0:
        result.add_warning("Age is 0 - verify if this is correct for infant")
    elif age > 120:
        result.add_warning("Age exceeds 120 - please verify")
    
    result.value = age
    return result


def validate_patient_name(
    name: Optional[str],
    required: bool = False,
    min_length: int = 2,
    max_length: int = 100
) -> ValidationResult:
    """
    Validate patient name.
    
    Args:
        name: Patient name
        required: Whether name is required
        min_length: Minimum name length
        max_length: Maximum name length
        
    Returns:
        ValidationResult with validated name
    """
    result = ValidationResult(is_valid=True, field_name="name")
    
    if not name or not name.strip():
        if required:
            return result.add_error("Name is required")
        result.value = None
        return result
    
    name = name.strip()
    
    # Check length
    if len(name) < min_length:
        return result.add_error(f"Name must be at least {min_length} characters")
    
    if len(name) > max_length:
        return result.add_error(f"Name cannot exceed {max_length} characters")
    
    # Basic sanitization - remove multiple spaces
    name = ' '.join(name.split())
    
    result.value = name
    return result


# =============================================================================
# INPUT SANITIZATION
# =============================================================================

def sanitize_input(
    text: str,
    max_length: int = 5000,
    strip_html: bool = True,
    normalize_whitespace: bool = True,
    remove_control_chars: bool = True
) -> str:
    """
    Sanitize user input text.
    
    Args:
        text: Input text
        max_length: Maximum allowed length
        strip_html: Whether to remove HTML tags
        normalize_whitespace: Whether to normalize whitespace
        remove_control_chars: Whether to remove control characters
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove control characters (except newlines and tabs)
    if remove_control_chars:
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Strip HTML tags
    if strip_html:
        text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace
    if normalize_whitespace:
        text = ' '.join(text.split())
    
    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length]
    
    return text.strip()


def sanitize_medical_input(text: str) -> Tuple[str, List[str]]:
    """
    Sanitize medical input with special handling.
    
    Preserves medical terminology and numbers while sanitizing.
    
    Args:
        text: Medical input text
        
    Returns:
        Tuple of (sanitized_text, warnings)
    """
    warnings = []
    
    if not text:
        return "", warnings
    
    # Basic sanitization
    text = sanitize_input(text, max_length=5000)
    
    # Check for potential prompt injection
    injection_patterns = [
        r'ignore\s+(?:previous|all)\s+instructions',
        r'you\s+are\s+now',
        r'act\s+as\s+(?:a|an)',
        r'pretend\s+to\s+be',
        r'system\s*prompt',
        r'<\|.*\|>',
        r'\[INST\]',
        r'\[\/INST\]'
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            warnings.append(f"Potential injection pattern detected: {pattern}")
            logger.warning(f"Potential injection in medical input: {pattern}")
    
    # Preserve medical measurements
    # These patterns should not be modified
    measurement_patterns = [
        r'\d+\s*(?:mg|ml|kg|cm|mm|g|mcg|IU)\b',  # Units
        r'\d+/\d+\s*(?:mmHg)?',  # Blood pressure
        r'\d+\.?\d*\s*(?:°[CF]|degrees?)',  # Temperature
        r'\d+\.?\d*\s*%',  # Percentages
    ]
    
    return text, warnings


# =============================================================================
# SYMPTOM VALIDATION
# =============================================================================

def is_valid_symptom(
    symptom: str,
    min_length: int = 2,
    max_length: int = 200
) -> ValidationResult:
    """
    Validate a symptom description.
    
    Args:
        symptom: Symptom text
        min_length: Minimum length
        max_length: Maximum length
        
    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True, field_name="symptom")
    
    if not symptom:
        return result.add_error("Symptom description is required")
    
    symptom = symptom.strip()
    
    if len(symptom) < min_length:
        return result.add_error(f"Symptom description too short (minimum {min_length} characters)")
    
    if len(symptom) > max_length:
        return result.add_error(f"Symptom description too long (maximum {max_length} characters)")
    
    # Check for gibberish (very basic)
    words = symptom.split()
    if len(words) > 3:
        avg_word_length = sum(len(w) for w in words) / len(words)
        if avg_word_length > 20:
            result.add_warning("Symptom may contain invalid text")
    
    result.value = symptom
    return result


def validate_symptom_list(
    symptoms: List[str],
    max_symptoms: int = 20,
    allow_empty: bool = False
) -> ValidationResult:
    """
    Validate a list of symptoms.
    
    Args:
        symptoms: List of symptom descriptions
        max_symptoms: Maximum number of symptoms
        allow_empty: Whether empty list is allowed
        
    Returns:
        ValidationResult with validated symptom list
    """
    result = ValidationResult(is_valid=True, field_name="symptoms")
    
    if not symptoms:
        if allow_empty:
            result.value = []
            return result
        return result.add_error("At least one symptom is required")
    
    if len(symptoms) > max_symptoms:
        return result.add_error(f"Too many symptoms (maximum {max_symptoms})")
    
    validated_symptoms = []
    for i, symptom in enumerate(symptoms):
        symptom_result = is_valid_symptom(symptom)
        if symptom_result.is_valid:
            validated_symptoms.append(symptom_result.value)
        else:
            for error in symptom_result.errors:
                result.add_error(f"Symptom {i+1}: {error}")
    
    if result.is_valid:
        result.value = validated_symptoms
    
    return result


# =============================================================================
# SEVERITY VALIDATION
# =============================================================================

VALID_SEVERITIES = ['mild', 'moderate', 'severe', 'critical']
SEVERITY_ALIASES = {
    'low': 'mild',
    'medium': 'moderate',
    'high': 'severe',
    'extreme': 'critical',
    'very bad': 'severe',
    'terrible': 'severe',
    'slightly': 'mild',
    'very': 'severe',
    # Urdu
    'ہلکا': 'mild',
    'درمیانہ': 'moderate',
    'شدید': 'severe',
    'بہت': 'severe',
}


def validate_severity(severity: str) -> ValidationResult:
    """
    Validate and normalize severity level.
    
    Args:
        severity: Severity description
        
    Returns:
        ValidationResult with normalized severity
    """
    result = ValidationResult(is_valid=True, field_name="severity")
    
    if not severity:
        result.value = None
        return result
    
    severity = severity.strip().lower()
    
    # Check direct match
    if severity in VALID_SEVERITIES:
        result.value = severity
        return result
    
    # Check aliases
    if severity in SEVERITY_ALIASES:
        result.value = SEVERITY_ALIASES[severity]
        result.add_warning(f"Severity '{severity}' normalized to '{result.value}'")
        return result
    
    # Try to extract from numeric scale
    scale_match = re.search(r'(\d+)\s*(?:out of|\/)\s*10', severity)
    if scale_match:
        score = int(scale_match.group(1))
        if score <= 3:
            result.value = 'mild'
        elif score <= 6:
            result.value = 'moderate'
        elif score <= 8:
            result.value = 'severe'
        else:
            result.value = 'critical'
        return result
    
    # Default to moderate with warning
    result.value = 'moderate'
    result.add_warning(f"Could not parse severity '{severity}', defaulting to 'moderate'")
    return result


# =============================================================================
# COMPOSITE VALIDATORS
# =============================================================================

def validate_chat_request(
    message: str,
    session_id: Optional[str] = None,
    language: Optional[str] = None
) -> Dict[str, ValidationResult]:
    """
    Validate a complete chat request.
    
    Args:
        message: User message
        session_id: Session ID (optional for new sessions)
        language: Language code
        
    Returns:
        Dictionary of field -> ValidationResult
    """
    results = {}
    
    # Validate message
    if not message or not message.strip():
        results['message'] = ValidationResult.invalid("Message is required", "message")
    else:
        sanitized = sanitize_input(message)
        if len(sanitized) < 1:
            results['message'] = ValidationResult.invalid("Message cannot be empty", "message")
        else:
            results['message'] = ValidationResult.valid(sanitized, "message")
    
    # Validate session_id if provided
    if session_id:
        results['session_id'] = validate_session_id(session_id)
    
    # Validate language if provided
    if language:
        results['language'] = validate_language_code(language)
    
    return results


def validate_patient_registration(
    name: Optional[str] = None,
    age: Optional[Union[int, str]] = None,
    phone: Optional[str] = None,
    language: str = "ur"
) -> Dict[str, ValidationResult]:
    """
    Validate patient registration data.
    
    Args:
        name: Patient name
        age: Patient age
        phone: Phone number
        language: Preferred language
        
    Returns:
        Dictionary of field -> ValidationResult
    """
    results = {}
    
    results['name'] = validate_patient_name(name, required=False)
    results['age'] = validate_patient_age(age, required=False)
    
    if phone:
        results['phone'] = validate_phone_number(phone)
    
    results['language'] = validate_language_code(language)
    
    return results


def get_validation_errors(results: Dict[str, ValidationResult]) -> List[Dict[str, Any]]:
    """
    Extract all errors from validation results.
    
    Args:
        results: Dictionary of validation results
        
    Returns:
        List of error dictionaries
    """
    errors = []
    for field_name, result in results.items():
        if not result.is_valid:
            for error in result.errors:
                errors.append({
                    "field": field_name,
                    "message": error
                })
    return errors


def is_all_valid(results: Dict[str, ValidationResult]) -> bool:
    """
    Check if all validations passed.
    
    Args:
        results: Dictionary of validation results
        
    Returns:
        True if all valid
    """
    return all(result.is_valid for result in results.values())