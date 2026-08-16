"""
WhatsApp Routes (Cloud API)
===========================
Webhook endpoints for Meta WhatsApp Cloud API integration.

Endpoints:
- GET /whatsapp - Webhook verification (Meta Challenge)
- POST /whatsapp - Incoming message handler (JSON)
- POST /whatsapp/send - Send outgoing message (admin)
- GET /whatsapp/status - Service status

Features:
- Meta webhook validation
- Voice message processing
- Session management per phone number
- Emergency escalation via WhatsApp
"""

import logging
import json
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Request, Form, Header, BackgroundTasks, Query
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field

from src.config import settings, SupportedLanguage
from src.services.whatsapp_service import (
    WhatsAppService,
    get_whatsapp_service,
    WhatsAppMessage,
    MessageType,
    send_whatsapp_message,
    process_whatsapp_webhook
)
from src.services.voice_service import process_voice_message, get_voice_service
from src.services.translator_service import detect_language
from src.graph.workflow import HealthcareGraph

logger = logging.getLogger("whatsapp_routes")

router = APIRouter()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class SendMessageRequest(BaseModel):
    """Request to send a WhatsApp message."""
    to_number: str = Field(..., description="Recipient phone number")
    message: str = Field(..., min_length=1, max_length=4096)
    
    class Config:
        json_schema_extra = {
            "example": {
                "to_number": "923123456789",
                "message": "آپ کا ٹیسٹ پیغام"
            }
        }


class SendMessageResponse(BaseModel):
    """Response for send message."""
    success: bool
    message_sid: Optional[str] = None
    error: Optional[str] = None


class WebhookStatusResponse(BaseModel):
    """Webhook status response."""
    status: str
    configured: bool
    phone_id: Optional[str] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def process_incoming_message(
    message: WhatsAppMessage,
    graph: HealthcareGraph,
    whatsapp_service: WhatsAppService
) -> None:
    """
    Process an incoming WhatsApp message.
    """
    logger.info(
        f"Processing message from {message.from_number}, "
        f"Type: {message.message_type.value}"
    )
    
    try:
        # Get or create session
        session = whatsapp_service.get_or_create_session(
            message.from_number,
            message.language
        )
        
        # Get message content
        text = ""
        if message.is_voice():
            # Process voice message
            text = await whatsapp_service.process_voice_message(message)
            if not text:
                # Send error response
                await whatsapp_service.send_text_message(
                    message.from_number,
                    "معذرت، میں آپ کا وائس میسج نہیں سمجھ سکا۔ براہ کرم دوبارہ کوشش کریں یا ٹیکسٹ میں لکھیں۔"
                )
                return
            logger.info(f"Transcribed voice: '{text[:50]}...'")
        else:
            text = message.content
        
        # Detect language if not already detected
        if not message.language:
            message.language = detect_language(text)
        
        # Process through healthcare AI
        logger.info(f"Sending message to Healthcare Graph for session {session.session_id}")
        result = await graph.process_message(
            session_id=session.session_id,
            message=text,
            language=message.language.value
        )
        logger.info(f"Healthcare Graph returned result for session {session.session_id}")
        
        # Get response
        response_text = result.get("response", "")
        is_emergency = result.get("is_emergency", False)
        
        # Format emergency response
        if is_emergency:
            emergency_action = result.get("emergency_action", "")
            response_text = f"⚠️ *ایمرجنسی الرٹ* ⚠️\n\n{emergency_action}\n\n{response_text}"
        
        # Check if interview is complete
        if result.get("interview_complete"):
            soap_note = result.get("soap_note", "")
            if soap_note:
                # Send main response first
                await whatsapp_service.send_text_message(
                    message.from_number,
                    response_text
                )
                
                # Send summary
                summary_message = (
                    "📋 *آپ کے انٹرویو کا خلاصہ*\n\n"
                    f"یہ خلاصہ آپ کے ڈاکٹر کو دکھائیں:\n\n{soap_note[:1500]}"
                )
                await whatsapp_service.send_text_message(
                    message.from_number,
                    summary_message
                )
                return
        
        # Send standard response
        if response_text:
            logger.info(f"Attempting to send response to {message.from_number}")
            if message.is_voice():
                # For voice input, respond with both text and voice
                await whatsapp_service.send_text_message(
                    message.from_number,
                    response_text
                )
                
                # Generate voice data
                voice_service = get_voice_service()
                voice_data = await voice_service.create_voice_response(
                    response_text, 
                    message.language or SupportedLanguage.URDU
                )
                if voice_data:
                    await whatsapp_service.send_voice_message(
                        message.from_number,
                        voice_data
                    )
            else:
                # Text only
                send_result = await whatsapp_service.send_text_message(
                    message.from_number,
                    response_text
                )
                logger.info(f"Text message send result: {send_result.success}")
        
        logger.info(f"Response processing complete for {message.from_number}")
        
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        try:
            await whatsapp_service.send_text_message(
                message.from_number,
                "معذرت، ایک مسئلہ ہوا۔ براہ کرم دوبارہ کوشش کریں۔"
            )
        except Exception:
            pass


# =============================================================================
# ROUTES
# =============================================================================

@router.get("/whatsapp", tags=["WhatsApp"])
async def verify_webhook(
    request: Request
):
    """
    WhatsApp webhook verification endpoint (Meta Challenge).
    Checks hub.mode, hub.verify_token, hub.challenge.
    """
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    
    whatsapp_service = get_whatsapp_service()
    
    if mode == "subscribe" and whatsapp_service.validate_verify_token(token):
        logger.info("Webhook verified successfully")
        return PlainTextResponse(content=challenge, status_code=200)
    else:
        logger.warning(f"Webhook verification failed. Mode: {mode}, Token: {token}")
        raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/whatsapp", tags=["WhatsApp"])
async def receive_message(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Handle incoming WhatsApp messages (JSON Payload).
    """
    try:
        webhook_data = await request.json()
        logger.info(f"WhatsApp Webhook Payload Received")
        logger.debug(f"Payload: {json.dumps(webhook_data)}")
    except Exception:
        logger.warning("Invalid JSON payload or failed to parse")
        return Response(status_code=400)
    
    # Check if this is a status update or valid message
    whatsapp_service = get_whatsapp_service()
    message = whatsapp_service.parse_incoming_message(webhook_data)
    
    if not message:
        # Likely a status update or unparseable, just ack
        return Response(status_code=200)
    
    logger.info(
        f"Incoming message - From: {message.from_number}, "
        f"Type: {message.message_type.value}, "
        f"Content: '{message.content[:50]}...'"
    )
    
    # Get healthcare graph
    try:
        graph: HealthcareGraph = request.app.state.healthcare_graph
    except AttributeError:
        logger.error("Healthcare graph not initialized")
        return Response(status_code=200)
    
    # Process in background
    background_tasks.add_task(
        process_incoming_message,
        message,
        graph,
        whatsapp_service
    )
    
    # Return 200 OK immediately
    return Response(status_code=200)


@router.post("/whatsapp/send", response_model=SendMessageResponse, tags=["WhatsApp"])
async def send_message(
    request: SendMessageRequest,
    x_api_key: Optional[str] = Header(None)
):
    """
    Send an outgoing WhatsApp message (Admin).
    """
    # Simple API key auth
    if not settings.DEBUG_MODE:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="Unauthorized")
    
    whatsapp_service = get_whatsapp_service()
    
    result = await whatsapp_service.send_text_message(
        request.to_number,
        request.message
    )
    
    return SendMessageResponse(
        success=result.success,
        message_sid=result.message_sid,
        error=result.error_message
    )


@router.get("/whatsapp/status", response_model=WebhookStatusResponse, tags=["WhatsApp"])
async def get_webhook_status():
    """
    Get WhatsApp webhook configuration status.
    """
    whatsapp_service = get_whatsapp_service()
    
    return WebhookStatusResponse(
        status="configured" if whatsapp_service.is_service_available() else "not_configured",
        configured=whatsapp_service.is_service_available(),
        phone_id=settings.WHATSAPP_PHONE_NUMBER_ID if settings.WHATSAPP_PHONE_NUMBER_ID else None
    )


@router.get("/whatsapp/session/{phone_number}", tags=["WhatsApp"])
async def get_session_info(
    phone_number: str,
    x_api_key: Optional[str] = Header(None)
):
    """
    Get session info.
    """
    if not settings.DEBUG_MODE and not x_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    whatsapp_service = get_whatsapp_service()
    session = whatsapp_service.get_session(phone_number)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "phone_number": session.phone_number,
        "session_id": session.session_id,
        "language": session.language.value,
        "started_at": session.started_at.isoformat(),
        "message_count": session.message_count
    }


@router.delete("/whatsapp/session/{phone_number}", tags=["WhatsApp"])
async def end_session(
    phone_number: str,
    x_api_key: Optional[str] = Header(None)
):
    """End session."""
    if not settings.DEBUG_MODE and not x_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    whatsapp_service = get_whatsapp_service()
    whatsapp_service.end_session(phone_number)
    
    return {"status": "session_ended"}