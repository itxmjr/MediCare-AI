"""
Translator Service
==================
Multilingual translation for healthcare communication.

This service provides:
1. Text translation between supported languages
2. Medical terminology-aware translation
3. Language detection
4. Cultural adaptation of medical advice

Strategy:
- Patient speaks native language → Translate to English for processing
- AI responds in English → Translate to patient's language
- Medical terms are preserved precisely

Supported Languages:
- English (en) - Primary processing language
- Urdu (ur) - Most common patient language
- Punjabi (pa) - Common in Punjab region
- Sindhi (sd) - Common in Sindh region
- Pashto (ps) - Common in KPK/Balochistan
"""

import logging
import json
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.config import settings, SupportedLanguage
from src.agents.prompts.translator_prompts import (
    TRANSLATION_SYSTEM_PROMPT,
    MEDICAL_TRANSLATION_PROMPT,
    MEDICAL_TERMS_DICTIONARY,
    COMMON_PHRASES,
    detect_language as pattern_detect_language
)

logger = logging.getLogger("translator_service")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TranslationResult:
    """Result of a translation operation."""
    original_text: str
    translated_text: str
    source_language: SupportedLanguage
    target_language: SupportedLanguage
    confidence: float
    medical_terms: List[Dict[str, str]]
    cultural_notes: Optional[str] = None
    is_success: bool = True
    error_message: Optional[str] = None


class TranslationOutput(BaseModel):
    """Structured output from LLM translation."""
    translated_text: str = Field(description="The translated text")
    confidence: float = Field(ge=0, le=1, description="Translation confidence")
    medical_terms: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Medical terms with their translations"
    )
    cultural_notes: Optional[str] = Field(
        None,
        description="Any cultural adaptations made"
    )
    back_translation: Optional[str] = Field(
        None,
        description="Back-translation for verification"
    )


# =============================================================================
# TRANSLATOR SERVICE CLASS
# =============================================================================

class TranslatorService:
    """
    Translation service for multilingual healthcare communication.
    
    Features:
    - Medical terminology preservation
    - Cultural sensitivity
    - Bidirectional translation
    - Caching for common phrases
    """
    
    def __init__(self):
        """Initialize the translator service."""
        self.llm = ChatGroq(
            model=settings.LLM_MODEL,
            temperature=0.3,
            groq_api_key=settings.GROQ_API_KEY
        )
        self.parser = PydanticOutputParser(pydantic_object=TranslationOutput)
        self._setup_prompts()
        self._phrase_cache: Dict[str, str] = {}
    
    def _setup_prompts(self):
        """Set up translation prompts."""
        self.translation_prompt = ChatPromptTemplate.from_messages([
            ("system", TRANSLATION_SYSTEM_PROMPT),
            ("human", MEDICAL_TRANSLATION_PROMPT + "\n\n{format_instructions}")
        ])
    
    async def translate(
        self,
        text: str,
        target_language: SupportedLanguage,
        source_language: Optional[SupportedLanguage] = None,
        medical_context: str = ""
    ) -> TranslationResult:
        """
        Translate text to target language.
        
        Args:
            text: Text to translate
            target_language: Language to translate to
            source_language: Source language (auto-detect if None)
            medical_context: Additional medical context
            
        Returns:
            TranslationResult with translated text
        """
        # Detect source language if not provided
        if source_language is None:
            source_language = self.detect_language(text)
        
        # No translation needed if same language
        if source_language == target_language:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language,
                target_language=target_language,
                confidence=1.0,
                medical_terms=[]
            )
        
        # Check cache for common phrases
        cache_key = f"{text}:{source_language.value}:{target_language.value}"
        if cache_key in self._phrase_cache:
            return TranslationResult(
                original_text=text,
                translated_text=self._phrase_cache[cache_key],
                source_language=source_language,
                target_language=target_language,
                confidence=0.95,
                medical_terms=[]
            )
        
        # Check predefined phrases
        predefined = self._get_predefined_phrase(text, target_language)
        if predefined:
            return TranslationResult(
                original_text=text,
                translated_text=predefined,
                source_language=source_language,
                target_language=target_language,
                confidence=0.99,
                medical_terms=[]
            )
        
        # Use LLM for translation
        try:
            result = await self._llm_translate(
                text, source_language, target_language, medical_context
            )
            
            # Cache the result
            self._phrase_cache[cache_key] = result.translated_text
            
            return result
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return TranslationResult(
                original_text=text,
                translated_text=text,  # Return original on failure
                source_language=source_language,
                target_language=target_language,
                confidence=0.0,
                medical_terms=[],
                is_success=False,
                error_message=str(e)
            )
    
    async def _llm_translate(
        self,
        text: str,
        source_language: SupportedLanguage,
        target_language: SupportedLanguage,
        medical_context: str
    ) -> TranslationResult:
        """Perform LLM-based translation."""
        
        chain = self.translation_prompt | self.llm | self.parser
        
        result = chain.invoke({
            "source_language": self._get_language_name(source_language),
            "target_language": self._get_language_name(target_language),
            "original_text": text,
            "medical_context": medical_context or "General healthcare conversation",
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return TranslationResult(
            original_text=text,
            translated_text=result.translated_text,
            source_language=source_language,
            target_language=target_language,
            confidence=result.confidence,
            medical_terms=result.medical_terms,
            cultural_notes=result.cultural_notes
        )
    
    async def translate_to_english(
        self,
        text: str,
        source_language: Optional[SupportedLanguage] = None
    ) -> Tuple[str, SupportedLanguage]:
        """
        Translate patient input to English for processing.
        
        Args:
            text: Patient's message
            source_language: Source language (auto-detect if None)
            
        Returns:
            Tuple of (english_text, detected_language)
        """
        if source_language is None:
            source_language = self.detect_language(text)
        
        if source_language == SupportedLanguage.ENGLISH:
            return text, source_language
        
        result = await self.translate(
            text,
            target_language=SupportedLanguage.ENGLISH,
            source_language=source_language,
            medical_context="Patient describing symptoms"
        )
        
        return result.translated_text, source_language
    
    async def translate_from_english(
        self,
        text: str,
        target_language: SupportedLanguage
    ) -> str:
        """
        Translate AI response from English to patient's language.
        
        Args:
            text: English response
            target_language: Patient's language
            
        Returns:
            Translated text
        """
        if target_language == SupportedLanguage.ENGLISH:
            return text
        
        result = await self.translate(
            text,
            target_language=target_language,
            source_language=SupportedLanguage.ENGLISH,
            medical_context="Healthcare AI response to patient"
        )
        
        return result.translated_text
    
    def detect_language(self, text: str) -> SupportedLanguage:
        """
        Detect the language of input text.
        
        Args:
            text: Input text
            
        Returns:
            Detected language
        """
        # Use pattern-based detection first (fast)
        detected = pattern_detect_language(text)
        
        logger.debug(f"Detected language: {detected.value} for '{text[:30]}...'")
        return detected
    
    def translate_medical_term(
        self,
        term: str,
        target_language: SupportedLanguage
    ) -> str:
        """
        Translate a medical term precisely.
        
        Args:
            term: Medical term in English
            target_language: Target language
            
        Returns:
            Translated term
        """
        term_lower = term.lower()
        
        # Check dictionary
        if term_lower in MEDICAL_TERMS_DICTIONARY:
            translations = MEDICAL_TERMS_DICTIONARY[term_lower]
            if target_language in translations:
                return translations[target_language]
        
        # Return original if not found
        return term
    
    def get_phrase(
        self,
        phrase_key: str,
        language: SupportedLanguage
    ) -> str:
        """
        Get a common phrase in the specified language.
        
        Args:
            phrase_key: Key for the phrase
            language: Target language
            
        Returns:
            Phrase in specified language
        """
        if phrase_key in COMMON_PHRASES:
            phrases = COMMON_PHRASES[phrase_key]
            return phrases.get(language, phrases.get(SupportedLanguage.ENGLISH, ""))
        return ""
    
    def _get_predefined_phrase(
        self,
        text: str,
        target_language: SupportedLanguage
    ) -> Optional[str]:
        """Check if text matches a predefined phrase."""
        text_lower = text.lower().strip()
        
        # Check common phrases
        for key, phrases in COMMON_PHRASES.items():
            english_phrase = phrases.get(SupportedLanguage.ENGLISH, "").lower()
            if text_lower == english_phrase:
                return phrases.get(target_language)
        
        return None
    
    def _get_language_name(self, language: SupportedLanguage) -> str:
        """Get full language name for prompts."""
        names = {
            SupportedLanguage.ENGLISH: "English",
            SupportedLanguage.URDU: "Urdu (اردو)",
            SupportedLanguage.PUNJABI: "Punjabi (پنجابی)",
            SupportedLanguage.SINDHI: "Sindhi (سنڌي)",
            SupportedLanguage.PASHTO: "Pashto (پښتو)"
        }
        return names.get(language, language.value)
    
    async def batch_translate(
        self,
        texts: List[str],
        target_language: SupportedLanguage,
        source_language: Optional[SupportedLanguage] = None
    ) -> List[TranslationResult]:
        """
        Translate multiple texts efficiently.
        
        Args:
            texts: List of texts to translate
            target_language: Target language
            source_language: Source language
            
        Returns:
            List of TranslationResults
        """
        results = []
        for text in texts:
            result = await self.translate(text, target_language, source_language)
            results.append(result)
        return results
    
    def translate_symptoms_list(
        self,
        symptoms: List[str],
        target_language: SupportedLanguage
    ) -> List[Dict[str, str]]:
        """
        Translate a list of symptoms with their English equivalents.
        
        Args:
            symptoms: List of symptom names
            target_language: Target language
            
        Returns:
            List of dicts with 'english' and 'translated' keys
        """
        translations = []
        for symptom in symptoms:
            translated = self.translate_medical_term(symptom, target_language)
            translations.append({
                "english": symptom,
                "translated": translated,
                "found_in_dictionary": translated != symptom
            })
        return translations


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global service instance
_translator_service: Optional[TranslatorService] = None


def get_translator_service() -> TranslatorService:
    """Get the singleton TranslatorService instance."""
    global _translator_service
    if _translator_service is None:
        _translator_service = TranslatorService()
    return _translator_service


async def translate_text(
    text: str,
    target_language: SupportedLanguage,
    source_language: Optional[SupportedLanguage] = None
) -> TranslationResult:
    """
    Translate text to target language.
    
    Args:
        text: Text to translate
        target_language: Target language
        source_language: Source language (auto-detect if None)
        
    Returns:
        TranslationResult
    """
    service = get_translator_service()
    return await service.translate(text, target_language, source_language)


async def translate_medical_text(
    text: str,
    target_language: SupportedLanguage,
    medical_context: str = ""
) -> TranslationResult:
    """
    Translate medical text with context.
    
    Args:
        text: Medical text to translate
        target_language: Target language
        medical_context: Additional context
        
    Returns:
        TranslationResult
    """
    service = get_translator_service()
    return await service.translate(
        text,
        target_language,
        medical_context=medical_context
    )


def detect_language(text: str) -> SupportedLanguage:
    """
    Detect language of text.
    
    Args:
        text: Input text
        
    Returns:
        Detected language
    """
    service = get_translator_service()
    return service.detect_language(text)


# =============================================================================
# QUICK TRANSLATION HELPERS
# =============================================================================

async def patient_to_english(
    text: str,
    detected_language: Optional[SupportedLanguage] = None
) -> Tuple[str, SupportedLanguage]:
    """
    Convert patient message to English for processing.
    
    Args:
        text: Patient's message in any language
        detected_language: Already detected language (optional)
        
    Returns:
        Tuple of (english_text, source_language)
    """
    service = get_translator_service()
    return await service.translate_to_english(text, detected_language)


async def english_to_patient(
    text: str,
    patient_language: SupportedLanguage
) -> str:
    """
    Convert English response to patient's language.
    
    Args:
        text: English response
        patient_language: Patient's preferred language
        
    Returns:
        Translated response
    """
    service = get_translator_service()
    return await service.translate_from_english(text, patient_language)


# =============================================================================
# MEDICAL TERM QUICK ACCESS
# =============================================================================

def get_symptom_translation(
    symptom: str,
    language: SupportedLanguage
) -> str:
    """Quick lookup for symptom translation."""
    service = get_translator_service()
    return service.translate_medical_term(symptom, language)


def get_emergency_phrase(language: SupportedLanguage) -> str:
    """Get emergency call instruction in patient's language."""
    service = get_translator_service()
    return service.get_phrase("call_emergency", language)


def get_greeting(language: SupportedLanguage) -> str:
    """Get greeting in patient's language."""
    service = get_translator_service()
    return service.get_phrase("hello_greeting", language)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_translator_service():
        print("Testing Translator Service...")
        
        service = TranslatorService()
        
        # Test language detection
        print("\n=== Language Detection ===")
        test_texts = [
            "I have a severe headache",
            "مجھے شدید سر درد ہے",
            "مینوں بہت سر درد اے",
        ]
        
        for text in test_texts:
            detected = service.detect_language(text)
            print(f"'{text[:30]}...' -> {detected.value}")
        
        # Test translation to Urdu
        print("\n=== English to Urdu Translation ===")
        english_texts = [
            "How long have you been experiencing this symptom?",
            "Please call emergency services immediately.",
            "You may have high blood sugar. Please see a doctor."
        ]
        
        for text in english_texts:
            result = await service.translate(
                text,
                target_language=SupportedLanguage.URDU,
                source_language=SupportedLanguage.ENGLISH
            )
            print(f"\nEnglish: {text}")
            print(f"Urdu: {result.translated_text}")
            print(f"Confidence: {result.confidence:.0%}")
        
        # Test translation from Urdu
        print("\n=== Urdu to English Translation ===")
        urdu_texts = [
            "مجھے دو تین مہینوں سے بہت تھکاوٹ ہو رہی ہے",
            "مجھے بہت پیاس لگتی ہے",
        ]
        
        for text in urdu_texts:
            result = await service.translate(
                text,
                target_language=SupportedLanguage.ENGLISH,
                source_language=SupportedLanguage.URDU
            )
            print(f"\nUrdu: {text}")
            print(f"English: {result.translated_text}")
        
        # Test medical term translation
        print("\n=== Medical Term Translation ===")
        terms = ["chest pain", "diabetes", "blood pressure", "fever"]
        
        for term in terms:
            urdu = service.translate_medical_term(term, SupportedLanguage.URDU)
            punjabi = service.translate_medical_term(term, SupportedLanguage.PUNJABI)
            print(f"{term}: Urdu={urdu}, Punjabi={punjabi}")
        
        print("\n=== Translator Service Test Complete ===")
    
    asyncio.run(test_translator_service())