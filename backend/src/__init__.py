"""
AI Healthcare Agentic System
============================
Gold Tier Implementation for AI Hackathon 2025

This system provides:
- Autonomous patient interviews (text/voice)
- Adaptive follow-up questions
- Emergency red flag detection
- Structured SOAP clinical summaries
- Multilingual support (Urdu/Punjabi/English)
- WhatsApp integration ready

IMPORTANT: This is a clinical decision-support tool, NOT a diagnostic system.
"""

__version__ = "1.0.0"
__author__ = "Healthcare AI Team"
__tier__ = "GOLD"

from src.config import settings

__all__ = ["settings", "__version__", "__tier__"]