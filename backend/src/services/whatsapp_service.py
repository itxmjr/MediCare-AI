"""
WhatsApp Service (Cloud API)
============================
Meta WhatsApp Cloud API integration for patient communication.

This service enables Gold Tier WhatsApp integration:
1. Receive messages via webhook (Meta format)
2. Send text responses via Graph API
3. Handle media (images, documents)

Optimized for:
- 2G compatibility (< 50KB responses where possible)
- Low-bandwidth environments
- Async message processing

Providers:
- Primary: Meta WhatsApp Cloud API (v22.0)
- Legacy: Twilio (Removed)
"""

import logging
import json
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import httpx

from src.config import settings, SupportedLanguage

logger = logging.getLogger("whatsapp_service")


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class MessageType(str, Enum):
    """WhatsApp message types."""
    TEXT = "text"
    VOICE = "voice"  # audio/ogg
    IMAGE = "image"
    DOCUMENT = "document"
    LOCATION = "location"
    INTERACTIVE = "interactive"
    TEMPLATE = "template"
    UNKNOWN = "unknown"


class MessageStatus(str, Enum):
    """Message delivery status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"


@dataclass
class WhatsAppMessage:
    """Incoming WhatsApp message."""
    message_id: str
    from_number: str
    to_number: str
    message_type: MessageType
    content: str
    media_id: Optional[str] = None
    media_url: Optional[str] = None # Will be populated after fetching media URL
    media_mime_type: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    profile_name: Optional[str] = None
    language: Optional[SupportedLanguage] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def is_voice(self) -> bool:
        """Check if message is a voice note."""
        return self.message_type == MessageType.VOICE
    
    def is_text(self) -> bool:
        """Check if message is text."""
        return self.message_type == MessageType.TEXT


@dataclass
class SendResult:
    """Result of sending a message."""
    success: bool
    message_sid: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class ConversationSession:
    """WhatsApp conversation session."""
    phone_number: str
    session_id: str
    language: SupportedLanguage
    started_at: datetime
    last_activity: datetime
    message_count: int = 0
    is_active: bool = True


# =============================================================================
# WHATSAPP SERVICE CLASS
# =============================================================================

class WhatsAppService:
    """
    WhatsApp Cloud API Service.
    
    Features:
    - Meta Graph API integration
    - Webhook handling and validation
    - Text message sending
    - Session management
    - Rate limiting
    """
    
    # Rate limiting: max messages per minute per number
    RATE_LIMIT = 20
    MAX_TEXT_LENGTH = 4096
    
    def __init__(self):
        """Initialize WhatsApp service."""
        self._sessions: Dict[str, ConversationSession] = {}
        self._rate_limits: Dict[str, List[datetime]] = {}
        
        # Meta API Config
        self.api_token = settings.WHATSAPP_CLOUD_API_TOKEN
        self.phone_number_id = settings.WHATSAPP_PHONE_NUMBER_ID
        self.verify_token = settings.WHATSAPP_VERIFY_TOKEN
        self.base_url = f"https://graph.facebook.com/v22.0/{self.phone_number_id}/messages"
        
        if not self.api_token or not self.phone_number_id:
            logger.warning("WhatsApp Cloud API credentials not fully configured")

    def is_service_available(self) -> bool:
        """Check if WhatsApp service is available."""
        return bool(self.api_token and self.phone_number_id)

    # =========================================================================
    # WEBHOOK HANDLING
    # =========================================================================
    
    def validate_verify_token(self, token: str) -> bool:
        """
        Validate webhook verification token.
        Matched against 'hub.verify_token'.
        """
        if not token:
            logger.warning("No token provided in verification request")
            return False
            
        # Clean and compare tokens (case-insensitive and trimmed)
        token_clean = token.strip()
        expected_clean = self.verify_token.strip()
        
        is_valid = token_clean == expected_clean
        if not is_valid:
            logger.warning(
                f"Verification token mismatch! "
                f"Received: '{token_clean}', Expected: '{expected_clean}'"
            )
        return is_valid
    
    def parse_incoming_message(
        self,
        webhook_data: Dict[str, Any]
    ) -> Optional[WhatsAppMessage]:
        """
        Parse incoming payload from Meta Webhook.
        Structure: entry -> changes -> value -> messages
        """
        try:
            entry = webhook_data.get("entry", [])[0]
            changes = entry.get("changes", [])[0]
            value = changes.get("value", {})
            messages = value.get("messages", [])
            
            if not messages:
                # Might be a status update
                return None
                
            message_data = messages[0]
            
            # Extract basic info
            message_id = message_data.get("id")
            from_number = message_data.get("from") # e.g. "92312..."
            
            # Get profile name
            contacts = value.get("contacts", [])
            profile_name = contacts[0]["profile"]["name"] if contacts else None
            
            # Determine type
            msg_type_str = message_data.get("type")
            content = ""
            media_id = None
            media_mime_type = None
            
            if msg_type_str == "text":
                message_type = MessageType.TEXT
                content = message_data.get("text", {}).get("body", "")
            elif msg_type_str == "audio":
                message_type = MessageType.VOICE
                audio_data = message_data.get("audio", {})
                media_id = audio_data.get("id")
                media_mime_type = audio_data.get("mime_type")
                content = "[Voice Message]"
            elif msg_type_str == "image":
                message_type = MessageType.IMAGE
                media_id = message_data.get("image", {}).get("id")
                content = message_data.get("image", {}).get("caption", "[Image]")
            elif msg_type_str == "interactive":
                message_type = MessageType.INTERACTIVE
                interactive = message_data.get("interactive", {})
                # Handle button replies
                if interactive.get("type") == "button_reply":
                    content = interactive.get("button_reply", {}).get("title", "")
                elif interactive.get("type") == "list_reply":
                    content = interactive.get("list_reply", {}).get("title", "")
            else:
                message_type = MessageType.UNKNOWN
                content = f"[{msg_type_str}]"

            # Clean number (remove + if present, usually Meta sends without +)
            # But standard is +, so we will normalize later
            
            # Detect language
            from src.services.translator_service import detect_language
            language = detect_language(content) if content else SupportedLanguage.URDU
            
            return WhatsAppMessage(
                message_id=message_id,
                from_number=from_number,
                to_number=self.phone_number_id, # This is our ID, not number
                message_type=message_type,
                content=content,
                media_id=media_id,
                media_mime_type=media_mime_type,
                profile_name=profile_name,
                language=language,
                raw_data=webhook_data
            )
            
        except Exception as e:
            logger.error(f"Failed to parse incoming message: {e}")
            return None

    # =========================================================================
    # SENDING MESSAGES
    # =========================================================================
    
    async def send_text_message(
        self,
        to_number: str,
        message: str
    ) -> SendResult:
        """
        Send a text message via WhatsApp Cloud API.
        """
        if not self.is_service_available():
            return SendResult(False, error_message="Service not configured")

        # Check rate limit (simple in-memory)
        if not self._check_rate_limit(to_number):
             return SendResult(False, error_message="Rate limit exceeded")

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Format "to_number": Meta expects only digits. remove '+'
        formatted_to = to_number.replace("+", "")
        
        payload = {
            "messaging_product": "whatsapp",
            "to": formatted_to,
            "type": "text",
            "text": {
                "body": message
            }
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code in [200, 201]:
                    data = response.json()
                    msg_id = data.get("messages", [{}])[0].get("id")
                    logger.info(f"WhatsApp message sent successfully to {to_number}. ID: {msg_id}")
                    self._record_message(to_number)
                    return SendResult(success=True, message_sid=msg_id)
                else:
                    error_data = response.text
                    logger.error(f"WhatsApp API Error to {to_number}: {error_data}")
                    return SendResult(False, error_message=error_data)
                    
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return SendResult(False, error_message=str(e))

    async def send_voice_message(
        self,
        to_number: str,
        audio_data: bytes,
        audio_url: Optional[str] = None
    ) -> SendResult:
        """
        Send a voice message (audio).
        1. If bytes provided, upload to Meta first to get media_id.
        2. If URL provided, send via link.
        """
        if not self.is_service_available():
            return SendResult(False, error_message="Service not configured")

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        formatted_to = to_number.replace("+", "")
        media_object = {}

        if audio_data:
            # Upload media to get ID
            media_id = await self.upload_media(audio_data, "audio/ogg")
            if not media_id:
                return SendResult(False, error_message="Failed to upload audio media")
            media_object = {"id": media_id}
        elif audio_url:
            media_object = {"link": audio_url}
        else:
            return SendResult(False, error_message="No audio data or URL provided")

        payload = {
            "messaging_product": "whatsapp",
            "to": formatted_to,
            "type": "audio",
            "audio": media_object
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code in [200, 201]:
                    data = response.json()
                    msg_id = data.get("messages", [{}])[0].get("id")
                    self._record_message(to_number)
                    return SendResult(success=True, message_sid=msg_id)
                else:
                    logger.error(f"WhatsApp Voice API Error: {response.text}")
                    return SendResult(False, error_message=response.text)
                    
        except Exception as e:
            logger.error(f"Failed to send voice message: {e}")
            return SendResult(False, error_message=str(e))

    async def upload_media(self, media_data: bytes, mime_type: str) -> Optional[str]:
        """
        Upload media to Meta to get a media_id.
        """
        upload_url = f"https://graph.facebook.com/v22.0/{self.phone_number_id}/media"
        headers = {
            "Authorization": f"Bearer {self.api_token}"
        }
        
        # Determine extension from mime type
        ext = "ogg" if "ogg" in mime_type else "mp3"
        filename = f"voice_message.{ext}"
        
        files = {
            "file": (filename, media_data, mime_type),
            "type": (None, mime_type),
            "messaging_product": (None, "whatsapp")
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    upload_url,
                    headers=headers,
                    files=files
                )
                
                if response.status_code == 200:
                    data = response.json()
                    media_id = data.get("id")
                    logger.info(f"Media uploaded successfully. ID: {media_id}")
                    return media_id
                else:
                    logger.error(f"Media upload failed: {response.text}")
                    return None
        except Exception as e:
            logger.error(f"Error uploading media: {e}")
            return None

    # =========================================================================
    # MEDIA HANDLING (Retrieve Audio)
    # =========================================================================
    
    async def get_media_url(self, media_id: str) -> Optional[str]:
        """Convert media ID to download URL."""
        url = f"https://graph.facebook.com/v22.0/{media_id}"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("url")
        except Exception as e:
            logger.error(f"Failed to get media URL: {e}")
        return None

    async def download_media(self, media_url: str) -> Optional[bytes]:
        """Download binary media content."""
        headers = {"Authorization": f"Bearer {self.api_token}"}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(media_url, headers=headers)
                if response.status_code == 200:
                    return response.content
        except Exception as e:
             logger.error(f"Failed to download media: {e}")
        return None

    async def process_voice_message(self, message: WhatsAppMessage) -> Optional[str]:
        """
        1. Get Media URL from Media ID
        2. Download
        3. Transcribe
        """
        if not message.media_id:
            return None
        
        try:
            # Get actual URL
            media_url = await self.get_media_url(message.media_id)
            if not media_url:
                logger.error("Could not retrieve media URL from ID")
                return None
                
            # Download
            audio_data = await self.download_media(media_url)
            if not audio_data:
                 return None
            
            # Transcribe (existing service reuse)
            from src.services.voice_service import process_voice_message
            text, language = await process_voice_message(audio_data, message.language)
            message.language = language
            return text
            
        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            return None

    # =========================================================================
    # SESSION & UTILS
    # =========================================================================
    
    def get_or_create_session(
        self,
        phone_number: str,
        language: Optional[SupportedLanguage] = None
    ) -> ConversationSession:
        
        # Normalize number for session key (+92...)
        # Meta sends without plus, let's just make sure we are consistent.
        if not phone_number.startswith("+"):
            phone_number = "+" + phone_number
            
        if phone_number in self._sessions:
            session = self._sessions[phone_number]
            session.last_activity = datetime.now()
            session.message_count += 1
            if language:
                session.language = language
            return session
        
        import uuid
        session = ConversationSession(
            phone_number=phone_number,
            session_id=str(uuid.uuid4()),
            language=language or SupportedLanguage.URDU,
            started_at=datetime.now(),
            last_activity=datetime.now(),
            message_count=1
        )
        self._sessions[phone_number] = session
        return session
    
    def get_session(self, phone_number: str) -> Optional[ConversationSession]:
        if not phone_number.startswith("+"):
             phone_number = "+" + phone_number
        return self._sessions.get(phone_number)
        
    def end_session(self, phone_number: str):
        if not phone_number.startswith("+"):
             phone_number = "+" + phone_number
        if phone_number in self._sessions:
            self._sessions[phone_number].is_active = False

    def format_phone_number(self, number: str) -> str:
        """Util to add + if missing for internal consistency."""
        if not number.startswith('+'):
            return f"+{number}"
        return number

    def _check_rate_limit(self, phone_number: str) -> bool:
        now = datetime.now()
        if phone_number not in self._rate_limits:
            return True
        recent = [t for t in self._rate_limits[phone_number] if (now - t).seconds < 60]
        self._rate_limits[phone_number] = recent
        return len(recent) < self.RATE_LIMIT

    def _record_message(self, phone_number: str):
        if phone_number not in self._rate_limits:
            self._rate_limits[phone_number] = []
        self._rate_limits[phone_number].append(datetime.now())


# Global Instance
_whatsapp_service: Optional[WhatsAppService] = None

def get_whatsapp_service() -> WhatsAppService:
    global _whatsapp_service
    if _whatsapp_service is None:
        _whatsapp_service = WhatsAppService()
    return _whatsapp_service

async def send_whatsapp_message(to_number: str, message: str) -> SendResult:
    service = get_whatsapp_service()
    return await service.send_text_message(to_number, message)

def process_whatsapp_webhook(webhook_data: Dict[str, Any]) -> Optional[WhatsAppMessage]:
    service = get_whatsapp_service()
    return service.parse_incoming_message(webhook_data)


async def send_whatsapp_voice(to_number: str, audio_data: bytes) -> SendResult:
    """
    Send a WhatsApp voice message using the Cloud API.
    """
    service = get_whatsapp_service()
    return await service.send_voice_message(to_number, audio_data)