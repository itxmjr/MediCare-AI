"""
Voice Service (Groq + gTTS)
===========================
Free-tier compatible Speech-to-Text (STT) and Text-to-Speech (TTS).

This service enables Gold Tier voice interaction:
1. Transcribe patient voice messages using Groq Whisper (STT)
2. Generate voice responses using gTTS (Free Google TTS)

Providers:
- Primary STT: Groq Whisper (Large-v3-turbo)
- Primary TTS: gTTS (Standard Google TTS - Free)
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union, BinaryIO
from enum import Enum
from dataclasses import dataclass
import sys

# Patch audioop for Python 3.13 compatibility
try:
    import audioop
except ImportError:
    try:
        import audioop_lts as audioop
    except ImportError:
        audioop = None

if audioop:
    sys.modules["audioop"] = audioop
    sys.modules["pyaudioop"] = audioop

from groq import Groq
from gtts import gTTS
from pydub import AudioSegment

from src.config import settings, SupportedLanguage

logger = logging.getLogger("voice_service")


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class AudioFormat(str, Enum):
    """Supported audio formats."""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    WEBM = "webm"
    M4A = "m4a"
    OPUS = "opus"


class VoiceGender(str, Enum):
    """Voice gender options."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


@dataclass
class VoiceConfig:
    """Configuration for voice synthesis."""
    language: SupportedLanguage = SupportedLanguage.URDU
    gender: VoiceGender = VoiceGender.FEMALE
    speed: float = 1.0
    output_format: AudioFormat = AudioFormat.MP3


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription."""
    text: str
    language: SupportedLanguage
    confidence: float
    duration_seconds: float
    is_success: bool
    error_message: Optional[str] = None


@dataclass
class SynthesisResult:
    """Result of text-to-speech synthesis."""
    audio_data: bytes
    format: AudioFormat
    duration_seconds: float
    is_success: bool
    error_message: Optional[str] = None


# =============================================================================
# VOICE SERVICE CLASS
# =============================================================================

class VoiceService:
    """
    Voice Service for STT and TTS operations.
    """
    
    # Language code mapping for Groq Whisper
    WHISPER_LANGUAGE_MAP = {
        SupportedLanguage.ENGLISH: "en",
        SupportedLanguage.URDU: "ur",
        SupportedLanguage.PUNJABI: "pa",
        SupportedLanguage.SINDHI: "sd",
        SupportedLanguage.PASHTO: "ps",
    }
    
    # Language code mapping for gTTS
    GTTS_LANGUAGE_MAP = {
        SupportedLanguage.ENGLISH: "en",
        SupportedLanguage.URDU: "ur",
        SupportedLanguage.PUNJABI: "pa",
        SupportedLanguage.SINDHI: "sd", # Limited
        SupportedLanguage.PASHTO: "ps", # Limited
    }
    
    def __init__(self):
        """Initialize the voice service."""
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        self.whisper_model = settings.WHISPER_MODEL or "whisper-large-v3-turbo"
        self._temp_dir = Path(tempfile.gettempdir()) / "healthcare_ai_voice"
        self._temp_dir.mkdir(exist_ok=True)
    
    async def transcribe(
        self,
        audio_data: Union[bytes, BinaryIO, str],
        language: Optional[SupportedLanguage] = None,
        audio_format: AudioFormat = AudioFormat.OGG
    ) -> TranscriptionResult:
        """Transcribe audio to text via Groq."""
        try:
            audio_file = await self._prepare_audio_file(audio_data, audio_format)
            lang_code = self.WHISPER_LANGUAGE_MAP.get(language) if language else None
            
            logger.info(f"Transcribing audio with Groq Whisper ({lang_code or 'auto'})")
            
            with open(audio_file, "rb") as file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=(str(audio_file), file.read()),
                    model=self.whisper_model,
                    language=lang_code,
                    response_format="verbose_json",
                )
            
            self._cleanup_temp_file(audio_file)
            
            return TranscriptionResult(
                text=transcription.text,
                language=self._map_whisper_language(transcription.language) if hasattr(transcription, 'language') else (language or SupportedLanguage.URDU),
                confidence=1.0,
                duration_seconds=getattr(transcription, 'duration', 0.0),
                is_success=True
            )
        except Exception as e:
            logger.error(f"Groq Transcription failed: {e}")
            return TranscriptionResult("", language or SupportedLanguage.URDU, 0.0, 0.0, False, str(e))
    
    async def synthesize(
        self,
        text: str,
        config: Optional[VoiceConfig] = None
    ) -> SynthesisResult:
        """Synthesize speech via gTTS (Free)."""
        if config is None:
            config = VoiceConfig()
        
        try:
            lang = self.GTTS_LANGUAGE_MAP.get(config.language, "ur")
            logger.info(f"Synthesizing speech via gTTS ({lang}): '{text[:30]}...'")
            
            tts = gTTS(text=text, lang=lang, slow=(config.speed < 1.0))
            
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            audio_data = fp.read()
            
            # Convert to OGG if required for WhatsApp
            if config.output_format == AudioFormat.OGG:
                 audio_data = self.convert_audio_format(audio_data, AudioFormat.MP3, AudioFormat.OGG)
            
            return SynthesisResult(
                audio_data=audio_data,
                format=config.output_format,
                duration_seconds=len(text) / 15, # Rough estimate
                is_success=True
            )
        except Exception as e:
            logger.error(f"gTTS Synthesis failed: {e}")
            return SynthesisResult(b"", config.output_format, 0.0, False, str(e))

    async def process_voice_message(self, audio_data: bytes, expected_language: Optional[SupportedLanguage] = None, audio_format: AudioFormat = AudioFormat.OGG) -> Tuple[str, SupportedLanguage]:
        result = await self.transcribe(audio_data, expected_language, audio_format)
        return result.text, result.language if result.is_success else ("", expected_language or SupportedLanguage.URDU)

    async def create_voice_response(self, text: str, language: SupportedLanguage = SupportedLanguage.URDU, for_whatsapp: bool = True) -> Optional[bytes]:
        config = VoiceConfig(language=language, output_format=AudioFormat.OGG if for_whatsapp else AudioFormat.MP3)
        result = await self.synthesize(text, config)
        return result.audio_data if result.is_success else None

    # =========================================================================
    # HELPERS (REUSED)
    # =========================================================================
    
    async def _prepare_audio_file(self, audio_data: Union[bytes, BinaryIO, str], audio_format: AudioFormat) -> Path:
        if isinstance(audio_data, str): return Path(audio_data)
        if hasattr(audio_data, 'read'): audio_data = audio_data.read()
        
        temp_path = self._temp_dir / f"input_{id(audio_data)}.{audio_format.value}"
        with open(temp_path, 'wb') as f: f.write(audio_data)
        
        # Groq Whisper accepts mp3, mp4, mpeg, mpga, m4a, wav, webm.
        # If it's OGG, we MUST convert it using pydub.
        if audio_format.value in ["ogg", "opus"]:
            try:
                converted_path = self._temp_dir / f"input_{id(audio_data)}.mp3"
                audio = AudioSegment.from_file(str(temp_path))
                audio.export(str(converted_path), format="mp3")
                self._cleanup_temp_file(temp_path)
                return converted_path
            except Exception as e:
                logger.warning(f"Audio conversion failed: {e}")
        return temp_path

    def _cleanup_temp_file(self, file_path: Path):
        try:
            if file_path.exists(): file_path.unlink()
        except: pass

    def convert_audio_format(self, audio_data: bytes, from_fmt: AudioFormat, to_fmt: AudioFormat) -> bytes:
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=from_fmt.value)
            out = io.BytesIO()
            audio.export(out, format=to_fmt.value)
            return out.getvalue()
        except: return audio_data

    def _map_whisper_language(self, whisper_lang: str) -> SupportedLanguage:
        reverse_map = {v: k for k, v in self.WHISPER_LANGUAGE_MAP.items()}
        return reverse_map.get(whisper_lang, SupportedLanguage.URDU)


# Global Singleton
_voice_service: Optional[VoiceService] = None

def get_voice_service() -> VoiceService:
    global _voice_service
    if _voice_service is None: _voice_service = VoiceService()
    return _voice_service

async def transcribe_audio(audio_data: Union[bytes, str], language: Optional[SupportedLanguage] = None, audio_format: AudioFormat = AudioFormat.OGG) -> TranscriptionResult:
    return await get_voice_service().transcribe(audio_data, language, audio_format)

async def synthesize_speech(text: str, language: SupportedLanguage = SupportedLanguage.URDU) -> SynthesisResult:
    return await get_voice_service().synthesize(text, VoiceConfig(language=language))

async def process_voice_message(audio_data: bytes, expected_language: Optional[SupportedLanguage] = None) -> Tuple[str, SupportedLanguage]:
    return await get_voice_service().process_voice_message(audio_data, expected_language)