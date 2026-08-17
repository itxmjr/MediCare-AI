"""
FastAPI Application
===================
Main FastAPI application for SkillForge AI.

Features:
- Chat endpoint for text-based interaction
- Voice endpoints for audio processing
- WhatsApp webhook for messaging integration
- Health checks and monitoring
- CORS support for web clients
- Request logging and tracing

Gold Tier Requirements:
- Voice/Multilingual support ✓
- Real-time explainability ✓
- Integration-ready (WhatsApp/API) ✓
"""

import logging
import time
import uuid

# Patch audioop for Python 3.13 compatibility
import sys
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
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import settings, TriageLevel, SupportedLanguage
from src.graph.workflow import create_healthcare_graph, HealthcareGraph
from src.api.middleware.error_handler import setup_error_handlers

# Set up logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")


# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan.
    
    Startup: Initialize services
    Shutdown: Cleanup resources
    """
    # Startup
    logger.info("Starting Healthcare AI System...")
    logger.info(f"Tier Level: {settings.TIER_LEVEL.value}")
    logger.info(f"Debug Mode: {settings.DEBUG_MODE}")
    
    # Initialize the healthcare graph
    app.state.healthcare_graph = create_healthcare_graph()
    logger.info("Healthcare graph initialized")
    
    # Initialize RAG if available
    try:
        from src.agents.tools.rag_search import get_rag_tool
        rag_tool = get_rag_tool()
        logger.info("RAG tool initialized")
    except Exception as e:
        logger.warning(f"RAG initialization failed: {e}")
    
    # Check service availability
    from src.services import check_services_health
    health = check_services_health()
    logger.info(f"Services health: {health}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Healthcare AI System...")


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, max_length=5000, description="Patient message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    language: Optional[str] = Field("ur", description="Language code (en, ur, pa, sd, ps)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "مجھے سر میں درد ہے",
                "session_id": "patient_123",
                "language": "ur"
            }
        }


class ChatResponse(BaseModel):
    """Chat response model."""
    session_id: str
    response: str
    response_english: Optional[str] = None
    triage_level: str
    is_emergency: bool
    emergency_action: Optional[str] = None
    interview_complete: bool
    symptoms_identified: list = []
    red_flags: list = []
    soap_note: Optional[str] = None
    reasoning_trace: list = []
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "patient_123",
                "response": "آپ کو سر درد کتنے عرصے سے ہے؟",
                "response_english": "How long have you had this headache?",
                "triage_level": "STANDARD",
                "is_emergency": False,
                "interview_complete": False,
                "symptoms_identified": [{"name": "headache"}],
                "red_flags": [],
                "reasoning_trace": ["Interviewer: Asking about duration"]
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    tier: str
    services: Dict[str, Any]
    timestamp: str


class EmergencyAlertResponse(BaseModel):
    """Emergency alert response."""
    is_emergency: bool
    triage_level: str
    action_required: str
    action_urdu: str
    symptoms_detected: list
    call_emergency: bool


# =============================================================================
# APPLICATION FACTORY
# =============================================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="SkillForge AI",
        description="""
        # SkillForge AI - Healthcare Assistant
        
        Gold Tier implementation for AI Healthcare Hackathon 2025.
        
        ## Features
        - 🏥 Autonomous patient interviews
        - 🚨 Real-time emergency triage
        - 🌐 Multilingual support (Urdu, Punjabi, English)
        - 🎤 Voice message processing
        - 📱 WhatsApp integration ready
        
        ## Important Notice
        This is a **clinical decision-support tool**, NOT a diagnostic system.
        Always consult healthcare professionals for medical decisions.
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.DEBUG_MODE else None,
        redoc_url="/redoc" if settings.DEBUG_MODE else None
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # Add request ID to state
        request.state.request_id = request_id
        
        logger.info(f"[{request_id}] {request.method} {request.url.path}")
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        logger.info(f"[{request_id}] Completed in {duration:.3f}s - Status: {response.status_code}")
        
        response.headers["X-Request-ID"] = request_id
        return response
    
    # Setup error handlers
    setup_error_handlers(app)
    
    # Include routers
    from src.api.routes.chat import router as chat_router
    from src.api.routes.voice import router as voice_router
    from src.api.routes.whatsapp import router as whatsapp_router
    
    app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
    app.include_router(voice_router, prefix="/api/v1", tags=["Voice"])
    app.include_router(whatsapp_router, prefix="/webhook", tags=["WhatsApp"])
    
    return app


# =============================================================================
# MAIN ENDPOINTS
# =============================================================================

# Create app instance
app = create_app()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SkillForge AI",
        "version": "1.0.0",
        "tier": settings.TIER_LEVEL.value,
        "status": "operational",
        "docs": "/docs" if settings.DEBUG_MODE else "disabled",
        "endpoints": {
            "chat": "/api/v1/chat",
            "voice": "/api/v1/voice",
            "whatsapp": "/webhook/whatsapp",
            "health": "/health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and service availability.
    """
    from datetime import datetime
    from src.services import check_services_health
    
    services = check_services_health()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        tier=settings.TIER_LEVEL.value,
        services=services,
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    req: Request
):
    """
    Main chat endpoint for patient interaction.
    
    This endpoint:
    1. Receives patient message (any language)
    2. Processes through the healthcare graph
    3. Returns response with triage information
    
    Supports:
    - Text messages in English, Urdu, Punjabi
    - Session continuity via session_id
    - Real-time emergency detection
    """
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get language
    try:
        language = SupportedLanguage(request.language)
    except ValueError:
        language = SupportedLanguage.URDU
    
    logger.info(f"Chat request - Session: {session_id}, Language: {language.value}")
    
    try:
        # Get healthcare graph from app state
        graph: HealthcareGraph = req.app.state.healthcare_graph
        
        # Process message
        result = await graph.process_message(
            session_id=session_id,
            message=request.message,
            language=language.value
        )
        
        # Build response
        return ChatResponse(
            session_id=session_id,
            response=result.get("response", ""),
            response_english=result.get("response_english"),
            triage_level=result.get("triage_level", "STANDARD"),
            is_emergency=result.get("is_emergency", False),
            emergency_action=result.get("emergency_action"),
            interview_complete=result.get("interview_complete", False),
            symptoms_identified=result.get("symptoms", []),
            red_flags=result.get("red_flags", []),
            soap_note=result.get("soap_note"),
            reasoning_trace=result.get("reasoning_trace", []) if settings.ENABLE_REASONING_LOGS else []
        )
        
    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )


@app.post("/api/v1/emergency-check", response_model=EmergencyAlertResponse, tags=["Emergency"])
async def emergency_check(
    request: ChatRequest,
    req: Request
):
    """
    Quick emergency check endpoint.
    
    Performs rapid red flag detection without full interview.
    Use for initial screening or when patient reports acute symptoms.
    """
    from src.graph.nodes.guardian import GuardianNode
    from src.graph.state import PatientState, Symptom
    
    # Create temporary state
    state = PatientState(session_id=f"emergency_{uuid.uuid4()}")
    
    # Add the message as a symptom
    state.add_symptom(Symptom(name=request.message))
    state.add_conversation_turn(
        role="patient",
        content=request.message,
        original_language=SupportedLanguage(request.language or "ur")
    )
    
    # Run guardian node
    guardian = GuardianNode()
    result_state = guardian.process(state)
    
    # Get appropriate language for response
    try:
        language = SupportedLanguage(request.language)
    except ValueError:
        language = SupportedLanguage.URDU
    
    # Get action in Urdu
    from src.agents.prompts.triage_prompts import get_triage_action_message
    action_urdu = get_triage_action_message(result_state.current_triage_level, language)
    action_english = get_triage_action_message(result_state.current_triage_level, SupportedLanguage.ENGLISH)
    
    return EmergencyAlertResponse(
        is_emergency=result_state.is_emergency,
        triage_level=result_state.current_triage_level.value,
        action_required=action_english,
        action_urdu=action_urdu,
        symptoms_detected=[s.name for s in result_state.symptoms],
        call_emergency=result_state.current_triage_level == TriageLevel.IMMEDIATE
    )


@app.get("/api/v1/session/{session_id}", tags=["Session"])
async def get_session(
    session_id: str,
    req: Request
):
    """
    Get current session state.
    
    Returns the current state of a patient interview session.
    """
    try:
        graph: HealthcareGraph = req.app.state.healthcare_graph
        state = graph.get_session_state(session_id)
        
        if not state:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "state": state
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/session/{session_id}", tags=["Session"])
async def end_session(
    session_id: str,
    req: Request
):
    """
    End a session and get final summary.
    
    Forces generation of SOAP note even if interview is incomplete.
    """
    try:
        graph: HealthcareGraph = req.app.state.healthcare_graph
        
        # Process with end signal
        result = await graph.process_message(
            session_id=session_id,
            message="[END_SESSION]",
            language="en"
        )
        
        return {
            "session_id": session_id,
            "status": "ended",
            "soap_note": result.get("soap_note"),
            "summary": {
                "symptoms": result.get("symptoms", []),
                "red_flags": result.get("red_flags", []),
                "triage_level": result.get("triage_level")
            }
        }
        
    except Exception as e:
        logger.error(f"Error ending session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/languages", tags=["Config"])
async def get_supported_languages():
    """Get list of supported languages."""
    return {
        "languages": [
            {"code": "en", "name": "English", "native_name": "English"},
            {"code": "ur", "name": "Urdu", "native_name": "اردو"},
            {"code": "pa", "name": "Punjabi", "native_name": "پنجابی"},
            {"code": "sd", "name": "Sindhi", "native_name": "سنڌي"},
            {"code": "ps", "name": "Pashto", "native_name": "پښتو"},
            {"code": "ur-Latn", "name": "Roman Urdu", "native_name": "Roman Urdu"}
        ],
        "default": "ur"
    }


@app.get("/api/v1/triage-levels", tags=["Config"])
async def get_triage_levels():
    """Get triage level information."""
    return {
        "levels": [
            {
                "level": "IMMEDIATE",
                "color": "red",
                "description": "Life-threatening, needs care NOW",
                "action": "Call 1122 immediately"
            },
            {
                "level": "VERY_URGENT",
                "color": "orange", 
                "description": "Serious, needs care within 10 minutes",
                "action": "Go to Emergency Room immediately"
            },
            {
                "level": "URGENT",
                "color": "yellow",
                "description": "Needs care within 1 hour",
                "action": "Visit ER or Urgent Care within 2 hours"
            },
            {
                "level": "STANDARD",
                "color": "green",
                "description": "Can wait, but needs care",
                "action": "Schedule appointment within 24-48 hours"
            },
            {
                "level": "NON_URGENT",
                "color": "blue",
                "description": "Minor issues",
                "action": "See doctor at convenience"
            }
        ]
    }


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG_MODE else "An error occurred",
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG_MODE,
        log_level=settings.LOG_LEVEL.lower()
    )