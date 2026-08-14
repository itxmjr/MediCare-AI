"""
Services Module
===============
External service integrations for the Healthcare AI System.

This module provides Gold Tier capabilities:
1. Voice Service - Speech-to-Text (STT) and Text-to-Speech (TTS)
2. Translator Service - Multilingual translation
3. WhatsApp Service - WhatsApp Business API integration

Architecture:
- Services are stateless and can be scaled independently
- Each service has built-in error handling and fallbacks
- Services log all operations for monitoring

Key Design Decisions:
- Voice processing happens at the edge (before/after core logic)
- Core medical reasoning stays in English for accuracy
- Translation preserves medical terminology precisely
"""

from src.services.voice_service import (
    VoiceService,
    transcribe_audio,
    synthesize_speech,
    process_voice_message,
    VoiceConfig,
    AudioFormat
)

from src.services.translator_service import (
    TranslatorService,
    translate_text,
    translate_medical_text,
    detect_language,
    TranslationResult
)

from src.services.whatsapp_service import (
    WhatsAppService,
    send_whatsapp_message,
    send_whatsapp_voice,
    process_whatsapp_webhook,
    WhatsAppMessage
)

__all__ = [
    # Voice Service
    "VoiceService",
    "transcribe_audio",
    "synthesize_speech",
    "process_voice_message",
    "VoiceConfig",
    "AudioFormat",
    
    # Translator Service
    "TranslatorService",
    "translate_text",
    "translate_medical_text",
    "detect_language",
    "TranslationResult",
    
    # WhatsApp Service
    "WhatsAppService",
    "send_whatsapp_message",
    "send_whatsapp_voice",
    "process_whatsapp_webhook",
    "WhatsAppMessage"
]


# =============================================================================
# SERVICE HEALTH CHECK
# =============================================================================

def check_services_health() -> dict:
    """
    Check health status of all external services.
    
    Returns:
        Dict with service status information
    """
    from src.config import settings
    
    health = {
        "voice_service": {
            "enabled": settings.is_gold_tier(),
            "status": "unknown",
            "provider": "groq_whisper_gtts"
        },
        "translator_service": {
            "enabled": True,
            "status": "unknown",
            "provider": "langchain_groq"
        },
        "whatsapp_service": {
            "enabled": settings.is_gold_tier() and bool(settings.WHATSAPP_CLOUD_API_TOKEN),
            "status": "unknown",
            "provider": "meta_cloud_api"
        }
    }
    
    # Check Voice Service
    try:
        if settings.GROQ_API_KEY:
            health["voice_service"]["status"] = "available"
        else:
            health["voice_service"]["status"] = "no_api_key"
    except Exception as e:
        health["voice_service"]["status"] = f"error: {str(e)}"
    
    # Check Translator Service
    try:
        if settings.GROQ_API_KEY:
            health["translator_service"]["status"] = "available"
        else:
            health["translator_service"]["status"] = "no_api_key"
    except Exception as e:
        health["translator_service"]["status"] = f"error: {str(e)}"
    
    # Check WhatsApp Service
    try:
        if settings.WHATSAPP_CLOUD_API_TOKEN and settings.WHATSAPP_PHONE_NUMBER_ID:
            health["whatsapp_service"]["status"] = "configured"
        else:
            health["whatsapp_service"]["status"] = "not_configured"
    except Exception as e:
        health["whatsapp_service"]["status"] = f"error: {str(e)}"
    
    return health


# =============================================================================
# SERVICE FACTORY
# =============================================================================

class ServiceFactory:
    """
    Factory for creating service instances.
    
    Provides singleton access to services and handles initialization.
    """
    
    _voice_service: 'VoiceService' = None
    _translator_service: 'TranslatorService' = None
    _whatsapp_service: 'WhatsAppService' = None
    
    @classmethod
    def get_voice_service(cls) -> 'VoiceService':
        """Get or create VoiceService instance."""
        if cls._voice_service is None:
            cls._voice_service = VoiceService()
        return cls._voice_service
    
    @classmethod
    def get_translator_service(cls) -> 'TranslatorService':
        """Get or create TranslatorService instance."""
        if cls._translator_service is None:
            cls._translator_service = TranslatorService()
        return cls._translator_service
    
    @classmethod
    def get_whatsapp_service(cls) -> 'WhatsAppService':
        """Get or create WhatsAppService instance."""
        if cls._whatsapp_service is None:
            cls._whatsapp_service = WhatsAppService()
        return cls._whatsapp_service
    
    @classmethod
    def reset_all(cls) -> None:
        """Reset all service instances (for testing)."""
        cls._voice_service = None
        cls._translator_service = None
        cls._whatsapp_service = None