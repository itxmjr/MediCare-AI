"""
Voice Routes
============
REST API endpoints for voice-based interaction.

Endpoints:
- POST /voice/transcribe - Transcribe audio to text
- POST /voice/synthesize - Convert text to speech
- POST /voice/chat - Voice-based chat (audio in, audio out)

Features:
- Multi-format audio support (OGG, MP3, WAV, WebM)
- Multilingual transcription
- Natural voice synthesis
- WhatsApp voice note compatibility
"""

import logging
import base64
import io
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

from src.config import settings, SupportedLanguage
from src.services.voice_service import (
    VoiceService,
    VoiceConfig,
    AudioFormat,
    VoiceGender,
    transcribe_audio,
    synthesize_speech,
    process_voice_message
)
from src.services.translator_service import detect_language

logger = logging.getLogger("voice_routes")

router = APIRouter()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class TranscribeRequest(BaseModel):
    """Transcription request with base64 audio."""
    audio_base64: str = Field(..., description="Base64 encoded audio data")
    audio_format: str = Field("ogg", description="Audio format (ogg, mp3, wav, webm)")
    language_hint: Optional[str] = Field(None, description="Expected language code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "audio_base64": "T2dnUwACAAAAAAA...",
                "audio_format": "ogg",
                "language_hint": "ur"
            }
        }


class TranscribeResponse(BaseModel):
    """Transcription response."""
    text: str
    detected_language: str
    confidence: float
    duration_seconds: float
    processing_time_ms: int


class SynthesizeRequest(BaseModel):
    """Speech synthesis request."""
    text: str = Field(..., min_length=1, max_length=4000, description="Text to synthesize")
    language: str = Field("ur", description="Language code")
    voice_gender: str = Field("female", description="Voice gender (male/female)")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed")
    output_format: str = Field("mp3", description="Output format (mp3, ogg, wav)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "آپ کو سر درد کتنے عرصے سے ہے؟",
                "language": "ur",
                "voice_gender": "female",
                "speed": 1.0,
                "output_format": "mp3"
            }
        }


class SynthesizeResponse(BaseModel):
    """Synthesis response metadata."""
    audio_base64: str
    format: str
    duration_seconds: float
    processing_time_ms: int
    text_length: int


class VoiceChatRequest(BaseModel):
    """Voice chat request (audio in, expects audio out)."""
    audio_base64: str = Field(..., description="Base64 encoded audio")
    audio_format: str = Field("ogg", description="Input audio format")
    session_id: str = Field(..., description="Session ID")
    response_format: str = Field("mp3", description="Output audio format")
    include_text: bool = Field(True, description="Include text in response")


class VoiceChatResponse(BaseModel):
    """Voice chat response."""
    session_id: str
    
    # Transcription
    input_text: str
    input_language: str
    
    # Response
    response_text: str
    response_audio_base64: Optional[str] = None
    response_format: str
    
    # Interview status
    triage_level: str
    is_emergency: bool
    emergency_action: Optional[str] = None
    interview_complete: bool
    
    # Timing
    transcription_time_ms: int
    processing_time_ms: int
    synthesis_time_ms: int
    total_time_ms: int


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def decode_base64_audio(audio_base64: str) -> bytes:
    """Decode base64 audio data."""
    try:
        # Handle data URL format
        if "," in audio_base64:
            audio_base64 = audio_base64.split(",")[1]
        return base64.b64decode(audio_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {e}")


def encode_audio_base64(audio_data: bytes) -> str:
    """Encode audio data to base64."""
    return base64.b64encode(audio_data).decode('utf-8')


def get_audio_format(format_str: str) -> AudioFormat:
    """Convert string to AudioFormat enum."""
    format_map = {
        "ogg": AudioFormat.OGG,
        "mp3": AudioFormat.MP3,
        "wav": AudioFormat.WAV,
        "webm": AudioFormat.WEBM,
        "m4a": AudioFormat.M4A,
        "opus": AudioFormat.OPUS
    }
    return format_map.get(format_str.lower(), AudioFormat.OGG)


def get_voice_gender(gender_str: str) -> VoiceGender:
    """Convert string to VoiceGender enum."""
    if gender_str.lower() == "male":
        return VoiceGender.MALE
    return VoiceGender.FEMALE


# =============================================================================
# ROUTES
# =============================================================================

@router.post("/voice/transcribe", response_model=TranscribeResponse, tags=["Voice"])
async def transcribe_audio_endpoint(request: TranscribeRequest):
    """
    Transcribe audio to text.
    
    Supports:
    - Multiple audio formats (OGG, MP3, WAV, WebM)
    - Multiple languages (Urdu, Punjabi, English, etc.)
    - WhatsApp voice notes
    
    Returns transcribed text with detected language.
    """
    start_time = time.time()
    
    # Decode audio
    audio_data = decode_base64_audio(request.audio_base64)
    
    # Get format and language hint
    audio_format = get_audio_format(request.audio_format)
    language_hint = None
    if request.language_hint:
        try:
            language_hint = SupportedLanguage(request.language_hint)
        except ValueError:
            pass
    
    logger.info(
        f"Transcription request - Format: {audio_format.value}, "
        f"Size: {len(audio_data)} bytes, Language hint: {language_hint}"
    )
    
    try:
        # Transcribe
        result = await transcribe_audio(
            audio_data,
            language=language_hint,
            audio_format=audio_format
        )
        
        if not result.is_success:
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {result.error_message}"
            )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return TranscribeResponse(
            text=result.text,
            detected_language=result.language.value,
            confidence=result.confidence,
            duration_seconds=result.duration_seconds,
            processing_time_ms=processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice/transcribe/file", response_model=TranscribeResponse, tags=["Voice"])
async def transcribe_audio_file(
    file: UploadFile = File(..., description="Audio file"),
    language_hint: Optional[str] = Form(None, description="Expected language")
):
    """
    Transcribe uploaded audio file.
    
    Alternative to base64 endpoint for direct file upload.
    """
    start_time = time.time()
    
    # Read file
    audio_data = await file.read()
    
    # Detect format from filename or content type
    filename = file.filename or ""
    content_type = file.content_type or ""
    
    if "ogg" in filename.lower() or "ogg" in content_type:
        audio_format = AudioFormat.OGG
    elif "mp3" in filename.lower() or "mpeg" in content_type:
        audio_format = AudioFormat.MP3
    elif "wav" in filename.lower() or "wav" in content_type:
        audio_format = AudioFormat.WAV
    elif "webm" in filename.lower() or "webm" in content_type:
        audio_format = AudioFormat.WEBM
    else:
        audio_format = AudioFormat.OGG  # Default
    
    # Parse language hint
    lang_hint = None
    if language_hint:
        try:
            lang_hint = SupportedLanguage(language_hint)
        except ValueError:
            pass
    
    logger.info(
        f"File transcription - Filename: {filename}, "
        f"Format: {audio_format.value}, Size: {len(audio_data)} bytes"
    )
    
    try:
        result = await transcribe_audio(
            audio_data,
            language=lang_hint,
            audio_format=audio_format
        )
        
        if not result.is_success:
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {result.error_message}"
            )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return TranscribeResponse(
            text=result.text,
            detected_language=result.language.value,
            confidence=result.confidence,
            duration_seconds=result.duration_seconds,
            processing_time_ms=processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice/synthesize", response_model=SynthesizeResponse, tags=["Voice"])
async def synthesize_speech_endpoint(request: SynthesizeRequest):
    """
    Convert text to speech.
    
    Generates natural-sounding speech in the specified language.
    Returns base64 encoded audio.
    """
    start_time = time.time()
    
    # Parse parameters
    try:
        language = SupportedLanguage(request.language)
    except ValueError:
        language = SupportedLanguage.URDU
    
    gender = get_voice_gender(request.voice_gender)
    output_format = get_audio_format(request.output_format)
    
    logger.info(
        f"Synthesis request - Language: {language.value}, "
        f"Gender: {gender.value}, Text length: {len(request.text)}"
    )
    
    try:
        # Create voice config
        config = VoiceConfig(
            language=language,
            gender=gender,
            speed=request.speed,
            output_format=output_format
        )
        
        # Synthesize
        result = await synthesize_speech(
            request.text,
            language=language,
            gender=gender
        )
        
        if not result.is_success:
            raise HTTPException(
                status_code=500,
                detail=f"Synthesis failed: {result.error_message}"
            )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Encode audio
        audio_base64 = encode_audio_base64(result.audio_data)
        
        return SynthesizeResponse(
            audio_base64=audio_base64,
            format=output_format.value,
            duration_seconds=result.duration_seconds,
            processing_time_ms=processing_time_ms,
            text_length=len(request.text)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Synthesis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice/synthesize/stream", tags=["Voice"])
async def synthesize_speech_stream(request: SynthesizeRequest):
    """
    Convert text to speech and stream audio.
    
    Returns audio as a streaming response (no base64 encoding).
    More efficient for direct playback.
    """
    # Parse parameters
    try:
        language = SupportedLanguage(request.language)
    except ValueError:
        language = SupportedLanguage.URDU
    
    gender = get_voice_gender(request.voice_gender)
    output_format = get_audio_format(request.output_format)
    
    try:
        result = await synthesize_speech(
            request.text,
            language=language,
            gender=gender
        )
        
        if not result.is_success:
            raise HTTPException(
                status_code=500,
                detail=f"Synthesis failed: {result.error_message}"
            )
        
        # Content type mapping
        content_types = {
            AudioFormat.MP3: "audio/mpeg",
            AudioFormat.OGG: "audio/ogg",
            AudioFormat.WAV: "audio/wav",
            AudioFormat.WEBM: "audio/webm"
        }
        
        content_type = content_types.get(output_format, "audio/mpeg")
        
        return Response(
            content=result.audio_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=response.{output_format.value}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice/chat", response_model=VoiceChatResponse, tags=["Voice"])
async def voice_chat(
    request: VoiceChatRequest,
    req: Request
):
    """
    Voice-based chat endpoint.
    
    Complete voice interaction:
    1. Receive audio input
    2. Transcribe to text
    3. Process through healthcare AI
    4. Synthesize response to audio
    5. Return audio response
    
    Ideal for voice-first interfaces.
    """
    total_start = time.time()
    
    # Decode input audio
    audio_data = decode_base64_audio(request.audio_base64)
    audio_format = get_audio_format(request.audio_format)
    response_format = get_audio_format(request.response_format)
    
    logger.info(
        f"Voice chat - Session: {request.session_id}, "
        f"Audio size: {len(audio_data)} bytes"
    )
    
    try:
        # Step 1: Transcribe
        transcribe_start = time.time()
        text, detected_language = await process_voice_message(audio_data)
        transcription_time = int((time.time() - transcribe_start) * 1000)
        
        if not text:
            raise HTTPException(
                status_code=400,
                detail="Could not transcribe audio"
            )
        
        logger.info(f"Transcribed: '{text[:50]}...' ({detected_language.value})")
        
        # Step 2: Process through healthcare AI
        process_start = time.time()
        from src.graph.workflow import HealthcareGraph
        
        graph: HealthcareGraph = req.app.state.healthcare_graph
        result = await graph.process_message(
            session_id=request.session_id,
            message=text,
            language=detected_language.value
        )
        processing_time = int((time.time() - process_start) * 1000)
        
        response_text = result.get("response", "")
        
        # Step 3: Synthesize response audio
        synthesis_start = time.time()
        response_audio_base64 = None
        
        if response_text:
            synthesis_result = await synthesize_speech(
                response_text,
                language=detected_language,
                gender=VoiceGender.FEMALE
            )
            
            if synthesis_result.is_success:
                response_audio_base64 = encode_audio_base64(synthesis_result.audio_data)
        
        synthesis_time = int((time.time() - synthesis_start) * 1000)
        total_time = int((time.time() - total_start) * 1000)
        
        return VoiceChatResponse(
            session_id=request.session_id,
            
            # Input
            input_text=text,
            input_language=detected_language.value,
            
            # Response
            response_text=response_text,
            response_audio_base64=response_audio_base64,
            response_format=response_format.value,
            
            # Status
            triage_level=result.get("triage_level", "STANDARD"),
            is_emergency=result.get("is_emergency", False),
            emergency_action=result.get("emergency_action"),
            interview_complete=result.get("interview_complete", False),
            
            # Timing
            transcription_time_ms=transcription_time,
            processing_time_ms=processing_time,
            synthesis_time_ms=synthesis_time,
            total_time_ms=total_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voice/languages", tags=["Voice"])
async def get_voice_languages():
    """
    Get supported languages for voice processing.
    
    Returns languages with their TTS/STT support status.
    """
    return {
        "languages": [
            {
                "code": "en",
                "name": "English",
                "stt_support": "full",
                "tts_support": "full",
                "tts_voices": ["male", "female"]
            },
            {
                "code": "ur",
                "name": "Urdu",
                "native_name": "اردو",
                "stt_support": "full",
                "tts_support": "full",
                "tts_voices": ["male", "female"]
            },
            {
                "code": "pa",
                "name": "Punjabi",
                "native_name": "پنجابی",
                "stt_support": "partial",
                "tts_support": "partial",
                "tts_voices": ["female"],
                "note": "May use Urdu model as fallback"
            },
            {
                "code": "sd",
                "name": "Sindhi",
                "native_name": "سنڌي",
                "stt_support": "limited",
                "tts_support": "limited",
                "tts_voices": ["female"]
            },
            {
                "code": "ps",
                "name": "Pashto",
                "native_name": "پښتو",
                "stt_support": "limited",
                "tts_support": "limited",
                "tts_voices": ["female"]
            }
        ],
        "default": "ur",
        "supported_formats": {
            "input": ["ogg", "mp3", "wav", "webm", "m4a"],
            "output": ["mp3", "ogg", "wav"]
        }
    }


@router.get("/voice/test", tags=["Voice"])
async def test_voice_service():
    """
    Test voice service availability.
    
    Performs a quick test of STT/TTS capabilities.
    """
    from src.services.voice_service import get_voice_service
    
    service = get_voice_service()
    
    # Test TTS
    tts_working = False
    try:
        result = await service.synthesize(
            "Test",
            VoiceConfig(language=SupportedLanguage.ENGLISH)
        )
        tts_working = result.is_success
    except Exception as e:
        logger.warning(f"TTS test failed: {e}")
    
    return {
        "status": "available" if tts_working else "degraded",
        "tts": {
            "status": "working" if tts_working else "error",
            "provider": "gtts"
        },
        "stt": {
            "status": "available",
            "provider": "groq_whisper",
            "model": settings.WHISPER_MODEL
        }
    }