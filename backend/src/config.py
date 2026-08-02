"""
Configuration Management
========================
Centralized configuration for the Healthcare AI System.
Supports environment variables and tier-based feature flags.
"""

import os
from typing import Literal, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from enum import Enum


class TierLevel(str, Enum):
    """Hackathon tier levels with feature gates."""
    BRONZE = "bronze"  # 1 scenario, 5-10 questions, SOAP
    SILVER = "silver"  # 3 scenarios, multi-agent, red flags
    GOLD = "gold"      # Voice, multilingual, WhatsApp, explainability


class SupportedLanguage(str, Enum):
    """Languages supported by the system."""
    ENGLISH = "en"
    URDU = "ur"
    PUNJABI = "pa"
    SINDHI = "sd"
    PASHTO = "ps"
    ROMAN_URDU = "ur-Latn"


class TriageLevel(str, Enum):
    """WHO Emergency Triage Assessment levels."""
    IMMEDIATE = "IMMEDIATE"      # Life-threatening, needs immediate care
    VERY_URGENT = "VERY_URGENT"  # Serious, needs care within 10 minutes
    URGENT = "URGENT"            # Needs care within 1 hour
    STANDARD = "STANDARD"        # Can wait, non-urgent
    NON_URGENT = "NON_URGENT"    # Minor issues


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Create a .env file with these values.
    """
    
    # ===================
    # TIER CONFIGURATION
    # ===================
    TIER_LEVEL: TierLevel = Field(
        default=TierLevel.GOLD,
        description="Feature tier level"
    )
    
    # ===================
    # LLM CONFIGURATION
    # ===================
    GROQ_API_KEY: str = Field(
        default="",
        description="Groq API key for Llama models"
    )
    LLM_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Primary LLM model for medical reasoning"
    )
    LLM_TEMPERATURE: float = Field(
        default=0.3,
        description="Lower temperature for medical accuracy"
    )
    LLM_MAX_TOKENS: int = Field(
        default=2048,
        description="Max tokens for responses"
    )
    
    # ===================
    # EMBEDDING & RAG
    # ===================
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace embedding model"
    )
    CHROMA_PERSIST_DIR: str = Field(
        default="src/data/vector_store",
        description="ChromaDB persistence directory"
    )
    RAG_CHUNK_SIZE: int = Field(
        default=500,
        description="Chunk size for document splitting"
    )
    RAG_CHUNK_OVERLAP: int = Field(
        default=50,
        description="Overlap between chunks"
    )
    RAG_TOP_K: int = Field(
        default=5,
        description="Number of documents to retrieve"
    )
    
    # ===================
    # VOICE CONFIGURATION (Gold Tier)
    # ===================
    DEEPGRAM_API_KEY: str = Field(
        default="",
        description="Deepgram API for STT/TTS (Optional)"
    )
    WHISPER_MODEL: str = Field(
        default="whisper-large-v3-turbo",
        description="Whisper model for Groq STT"
    )
    DEFAULT_LANGUAGE: SupportedLanguage = Field(
        default=SupportedLanguage.URDU,
        description="Default patient language"
    )
    
    # ===================
    # WHATSAPP CONFIGURATION (Gold Tier)
    # ===================
    # WhatsApp Cloud API (Meta)
    WHATSAPP_CLOUD_API_TOKEN: str = Field(
        default="",
        description="Meta Graph API User/System Token"
    )
    WHATSAPP_PHONE_NUMBER_ID: str = Field(
        default="",
        description="WhatsApp Phone Number ID"
    )
    WHATSAPP_BUSINESS_ACCOUNT_ID: str = Field(
        default="",
        description="WhatsApp Business Account ID"
    )
    WHATSAPP_NUMBER: str = Field(
        default="",
        description="The allocated WhatsApp number (for display/reference)"
    )
    WHATSAPP_VERIFY_TOKEN: str = Field(
        default="my_secure_token",
        description="Custom verification token for webhook"
    )
    
    # ===================
    # INTERVIEW CONFIGURATION
    # ===================
    MAX_INTERVIEW_QUESTIONS: int = Field(
        default=15,
        description="Maximum questions in an interview"
    )
    MIN_INTERVIEW_QUESTIONS: int = Field(
        default=5,
        description="Minimum questions before allowing summary"
    )
    INTERVIEW_TIMEOUT_MINUTES: int = Field(
        default=20,
        description="Max interview duration"
    )
    
    # ===================
    # RED FLAG THRESHOLDS
    # ===================
    RED_FLAG_CONFIDENCE_THRESHOLD: float = Field(
        default=0.7,
        description="Confidence threshold for red flag alerts"
    )
    EMERGENCY_KEYWORDS: list = Field(
        default=[
            "chest pain", "difficulty breathing", "unconscious",
            "severe bleeding", "stroke symptoms", "seizure",
            "crushing pain", "radiating pain", "cold sweat"
        ],
        description="Keywords triggering immediate triage"
    )
    
    # ===================
    # API CONFIGURATION
    # ===================
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    DEBUG_MODE: bool = Field(default=False)
    
    # ===================
    # LOGGING & EXPLAINABILITY
    # ===================
    LOG_LEVEL: str = Field(default="INFO")
    ENABLE_REASONING_LOGS: bool = Field(
        default=True,
        description="Log agent reasoning for explainability"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"
    
    def is_gold_tier(self) -> bool:
        """Check if Gold tier features are enabled."""
        return self.TIER_LEVEL == TierLevel.GOLD
    
    def is_silver_or_above(self) -> bool:
        """Check if Silver+ tier features are enabled."""
        return self.TIER_LEVEL in [TierLevel.SILVER, TierLevel.GOLD]
    
    def get_available_languages(self) -> list[SupportedLanguage]:
        """Get languages based on tier."""
        if self.is_gold_tier():
            return list(SupportedLanguage)
        return [SupportedLanguage.ENGLISH, SupportedLanguage.URDU]


# Global settings instance
settings = Settings()


# ===================
# RED FLAG PATTERNS
# ===================
RED_FLAG_PATTERNS = {
    "cardiac_emergency": {
        "symptoms": ["chest pain", "radiating pain", "arm pain", "jaw pain", "cold sweat", "diaphoresis"],
        "combination_required": 2,
        "triage_level": TriageLevel.IMMEDIATE,
        "action": "Call 1122 NOW. Possible heart attack (MI)."
    },
    "stroke": {
        "symptoms": ["sudden weakness", "facial drooping", "speech difficulty", "confusion", "severe headache"],
        "combination_required": 2,
        "triage_level": TriageLevel.IMMEDIATE,
        "action": "Call 1122 NOW. Possible stroke. Note time of symptom onset."
    },
    "subarachnoid_hemorrhage": {
        "symptoms": ["thunderclap headache", "worst headache", "sudden severe headache", "neck stiffness"],
        "combination_required": 2,
        "triage_level": TriageLevel.IMMEDIATE,
        "action": "Call 1122 NOW. Possible brain bleed (SAH)."
    },
    "pulmonary_embolism": {
        "symptoms": ["difficulty breathing", "shortness of breath", "leg swelling", "chest pain", "rapid heartbeat"],
        "combination_required": 2,
        "triage_level": TriageLevel.IMMEDIATE,
        "action": "Call 1122 NOW. Possible blood clot in lungs (PE)."
    },
    "diabetic_emergency": {
        "symptoms": ["excessive thirst", "frequent urination", "fatigue", "weight loss", "fruity breath", "confusion"],
        "combination_required": 3,
        "triage_level": TriageLevel.VERY_URGENT,
        "action": "Urgent blood glucose test required. Possible diabetic crisis."
    },
    "appendicitis": {
        "symptoms": ["abdominal pain", "right lower pain", "nausea", "fever", "loss of appetite"],
        "combination_required": 3,
        "triage_level": TriageLevel.URGENT,
        "action": "Visit ER within 2 hours. Possible appendicitis."
    }
}