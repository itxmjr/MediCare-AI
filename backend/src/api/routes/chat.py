"""
Chat Routes
===========
REST API endpoints for text-based patient interaction.

Endpoints:
- POST /chat - Main conversation endpoint
- POST /chat/start - Start new interview session
- POST /chat/continue - Continue existing session
- GET /chat/history/{session_id} - Get conversation history
- POST /chat/feedback - Submit feedback on response

Features:
- Session management
- Language detection
- Real-time triage updates
- Conversation history
- Response streaming (optional)
"""

import logging
import uuid
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import settings, SupportedLanguage, TriageLevel
from src.graph.workflow import HealthcareGraph
from src.graph.state import PatientState, InterviewPhase
from src.services.translator_service import (
    get_translator_service,
    detect_language,
    patient_to_english,
    english_to_patient
)

logger = logging.getLogger("chat_routes")

router = APIRouter()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class StartSessionRequest(BaseModel):
    """Request to start a new session."""
    patient_name: Optional[str] = Field(None, description="Patient's name")
    patient_age: Optional[int] = Field(None, ge=0, le=150, description="Patient's age")
    patient_gender: Optional[str] = Field(None, description="Patient's gender")
    language: str = Field("ur", description="Preferred language code")
    channel: str = Field("api", description="Communication channel (api/whatsapp/web)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_name": "Ahmed",
                "patient_age": 45,
                "patient_gender": "male",
                "language": "ur",
                "channel": "api"
            }
        }


class StartSessionResponse(BaseModel):
    """Response for session start."""
    session_id: str
    greeting: str
    greeting_english: str
    language: str
    instructions: str


class ChatMessageRequest(BaseModel):
    """Chat message request."""
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: str = Field(..., description="Session ID")
    include_reasoning: bool = Field(False, description="Include AI reasoning trace")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "مجھے دو دن سے سر میں درد ہے",
                "session_id": "abc123",
                "include_reasoning": False
            }
        }


class ChatMessageResponse(BaseModel):
    """Chat message response."""
    session_id: str
    response: str
    response_english: Optional[str] = None
    detected_language: str
    
    # Interview status
    interview_phase: str
    question_number: int
    interview_complete: bool
    
    # Clinical information
    triage_level: str
    triage_color: str
    is_emergency: bool
    emergency_action: Optional[str] = None
    
    # Identified information
    symptoms: List[dict] = []
    red_flags: List[dict] = []
    
    # Optional detailed info
    reasoning_trace: Optional[List[str]] = None
    soap_note: Optional[str] = None
    
    # Metadata
    response_time_ms: int
    timestamp: str


class ConversationHistoryResponse(BaseModel):
    """Conversation history response."""
    session_id: str
    language: str
    started_at: str
    message_count: int
    current_phase: str
    triage_level: str
    messages: List[dict]
    symptoms_identified: List[dict]
    is_complete: bool


class FeedbackRequest(BaseModel):
    """Feedback submission request."""
    session_id: str
    message_id: Optional[str] = None
    rating: int = Field(..., ge=1, le=5, description="Rating 1-5")
    feedback_type: str = Field(..., description="accuracy/helpfulness/clarity/other")
    comment: Optional[str] = Field(None, max_length=1000)


class QuickTriageRequest(BaseModel):
    """Quick triage check request."""
    symptoms: List[str] = Field(..., min_items=1, description="List of symptoms")
    duration: Optional[str] = Field(None, description="How long symptoms present")
    severity: Optional[str] = Field(None, description="mild/moderate/severe")
    language: str = Field("ur", description="Response language")


class QuickTriageResponse(BaseModel):
    """Quick triage response."""
    triage_level: str
    triage_color: str
    urgency: str
    action_required: str
    action_local_language: str
    matched_patterns: List[str]
    recommendation: str


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_triage_color(level: str) -> str:
    """Get color code for triage level."""
    colors = {
        "IMMEDIATE": "red",
        "VERY_URGENT": "orange",
        "URGENT": "yellow",
        "STANDARD": "green",
        "NON_URGENT": "blue"
    }
    return colors.get(level, "gray")


def format_symptoms_for_response(symptoms: list) -> List[dict]:
    """Format symptoms for API response."""
    formatted = []
    for symptom in symptoms:
        if hasattr(symptom, 'model_dump'):
            formatted.append(symptom.model_dump())
        elif isinstance(symptom, dict):
            formatted.append(symptom)
        else:
            formatted.append({"name": str(symptom)})
    return formatted


def format_red_flags_for_response(red_flags: list) -> List[dict]:
    """Format red flags for API response."""
    formatted = []
    for rf in red_flags:
        if hasattr(rf, 'model_dump'):
            data = rf.model_dump()
            # Convert datetime to string
            if 'detected_at' in data and hasattr(data['detected_at'], 'isoformat'):
                data['detected_at'] = data['detected_at'].isoformat()
            formatted.append(data)
        elif isinstance(rf, dict):
            formatted.append(rf)
    return formatted


# =============================================================================
# ROUTES
# =============================================================================

@router.post("/chat/start", response_model=StartSessionResponse, tags=["Chat"])
async def start_session(
    request: StartSessionRequest,
    req: Request
):
    """
    Start a new patient interview session.
    
    Creates a new session and returns:
    - Unique session ID
    - Greeting in patient's language
    - Initial instructions
    """
    # Generate session ID
    session_id = f"session_{uuid.uuid4().hex[:12]}"
    
    # Validate language
    try:
        language = SupportedLanguage(request.language)
    except ValueError:
        language = SupportedLanguage.URDU
    
    logger.info(f"Starting new session: {session_id}, language: {language.value}")
    
    # Get translator service for greeting
    translator = get_translator_service()
    
    # Generate greeting based on language
    greetings = {
        SupportedLanguage.ENGLISH: (
            "Hello! I'm your AI health assistant. I'll ask you some questions to better understand "
            "your health concerns. Please describe what's bothering you today."
        ),
        SupportedLanguage.URDU: (
            "السلام علیکم! میں آپ کا AI صحت معاون ہوں۔ میں آپ سے کچھ سوالات پوچھوں گا تاکہ آپ کی "
            "صحت کے مسائل کو بہتر طور پر سمجھ سکوں۔ براہ کرم بتائیں کہ آج آپ کو کیا تکلیف ہے؟"
        ),
        SupportedLanguage.PUNJABI: (
            "سلام! میں تہاڈا AI صحت معاون آں۔ میں تہانوں کجھ سوال پچھاں گا تاکہ تہاڈی صحت دے "
            "مسئلے نوں سمجھ سکاں۔ دسو اج تہانوں کی تکلیف اے؟"
        ),
        SupportedLanguage.SINDHI: (
            "السلام عليڪم! مان توهان جو AI صحت معاون آهيان۔ مان توهان کان ڪجھ سوال پڇندس "
            "ته جيئن توهان جي صحت جي مسئلن کي سمجهي سگهان۔ ٻڌايو اڄ توهان کي ڇا تڪليف آهي؟"
        ),
        SupportedLanguage.PASHTO: (
            "سلام! زه ستاسو د روغتیا AI مرستندوی یم۔ زه به تاسو ته ځینې پوښتنې وکړم چې ستاسو "
            "د روغتیا ستونزې ښه پوه شم۔ مهرباني وکړئ ووایاست چې نن تاسو ته څه تکلیف شته؟"
        ),
        SupportedLanguage.ROMAN_URDU: (
            "Salam! Mein aap ka AI sehat muawin hoon. Mein aap se kuch sawalat puchoon ga taakay aap ke "
            "sehat ke masail ko behtar tor par samajh sakoon. Barah-e-karam batayen ke aaj aap ko kya takleef hai?"
        )
    }
    
    greeting = greetings.get(language, greetings[SupportedLanguage.URDU])
    greeting_english = greetings[SupportedLanguage.ENGLISH]
    
    # Instructions
    instructions = (
        "You can type or speak in your preferred language. "
        "I'll ask follow-up questions based on your responses. "
        "If you experience any emergency symptoms, I'll guide you immediately."
    )
    
    # Initialize session in graph (pre-warm)
    try:
        graph: HealthcareGraph = req.app.state.healthcare_graph
        # We could pre-initialize state here if needed
    except Exception as e:
        logger.warning(f"Could not pre-initialize session: {e}")
    
    return StartSessionResponse(
        session_id=session_id,
        greeting=greeting,
        greeting_english=greeting_english,
        language=language.value,
        instructions=instructions
    )


@router.post("/chat", response_model=ChatMessageResponse, tags=["Chat"])
async def chat_message(
    request: ChatMessageRequest,
    req: Request,
    background_tasks: BackgroundTasks
):
    """
    Send a message in an ongoing conversation.
    
    Processes the patient message through the healthcare graph and returns:
    - AI response in patient's language
    - Updated triage level
    - Emergency alerts if detected
    - Interview progress
    """
    import time
    start_time = time.time()
    
    # Detect language
    detected_lang = detect_language(request.message)
    
    logger.info(
        f"Chat message - Session: {request.session_id}, "
        f"Language: {detected_lang.value}, "
        f"Message length: {len(request.message)}"
    )
    
    try:
        # Get healthcare graph
        graph: HealthcareGraph = req.app.state.healthcare_graph
        
        # Process message
        result = await graph.process_message(
            session_id=request.session_id,
            message=request.message,
            language=detected_lang.value
        )
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Get triage level
        triage_level = result.get("triage_level", "STANDARD")
        
        # Format response
        response = ChatMessageResponse(
            session_id=request.session_id,
            response=result.get("response", ""),
            response_english=result.get("response_english"),
            detected_language=detected_lang.value,
            
            # Interview status
            interview_phase=result.get("interview_phase", "chief_complaint"),
            question_number=result.get("question_count", 0),
            interview_complete=result.get("interview_complete", False),
            
            # Triage
            triage_level=triage_level,
            triage_color=get_triage_color(triage_level),
            is_emergency=result.get("is_emergency", False),
            emergency_action=result.get("emergency_action"),
            
            # Clinical info
            symptoms=format_symptoms_for_response(result.get("symptoms", [])),
            red_flags=format_red_flags_for_response(result.get("red_flags", [])),
            
            # Optional
            reasoning_trace=result.get("reasoning_trace") if request.include_reasoning else None,
            soap_note=result.get("soap_note"),
            
            # Metadata
            response_time_ms=response_time_ms,
            timestamp=datetime.now().isoformat()
        )
        
        # Log if emergency
        if response.is_emergency:
            logger.warning(
                f"EMERGENCY DETECTED - Session: {request.session_id}, "
                f"Level: {triage_level}, Action: {response.emergency_action}"
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Chat processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )


@router.post("/chat/continue", response_model=ChatMessageResponse, tags=["Chat"])
async def continue_chat(
    request: ChatMessageRequest,
    req: Request,
    background_tasks: BackgroundTasks
):
    """
    Continue an existing conversation.
    
    Alias for /chat endpoint with explicit continuation semantics.
    """
    return await chat_message(request, req, background_tasks)


@router.get("/chat/history/{session_id}", response_model=ConversationHistoryResponse, tags=["Chat"])
async def get_conversation_history(
    session_id: str,
    req: Request,
    include_translations: bool = Query(False, description="Include English translations")
):
    """
    Get conversation history for a session.
    
    Returns all messages exchanged in the session along with
    identified symptoms and current interview state.
    """
    try:
        graph: HealthcareGraph = req.app.state.healthcare_graph
        state_dict = graph.get_session_state(session_id)
        
        if not state_dict:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Format messages
        messages = []
        conversation_turns = state_dict.get("conversation_turns", [])
        
        for turn in conversation_turns:
            msg = {
                "role": turn.get("role", "unknown"),
                "content": turn.get("content", ""),
                "timestamp": turn.get("timestamp", ""),
                "phase": turn.get("phase", "")
            }
            
            if include_translations and turn.get("english_translation"):
                msg["english_translation"] = turn["english_translation"]
            
            if turn.get("reasoning"):
                msg["reasoning"] = turn["reasoning"]
            
            messages.append(msg)
        
        # Get language
        demographics = state_dict.get("demographics", {})
        language = demographics.get("preferred_language", "ur")
        
        return ConversationHistoryResponse(
            session_id=session_id,
            language=language,
            started_at=state_dict.get("created_at", datetime.now().isoformat()),
            message_count=len(messages),
            current_phase=state_dict.get("current_phase", "unknown"),
            triage_level=state_dict.get("current_triage_level", "STANDARD"),
            messages=messages,
            symptoms_identified=format_symptoms_for_response(state_dict.get("symptoms", [])),
            is_complete=state_dict.get("interview_complete", False)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/quick-triage", response_model=QuickTriageResponse, tags=["Chat"])
async def quick_triage(
    request: QuickTriageRequest,
    req: Request
):
    """
    Quick triage assessment without full interview.
    
    Useful for:
    - Initial screening
    - Emergency detection
    - Triage level estimation
    
    Does not create a session or maintain state.
    """
    from src.graph.nodes.guardian import GuardianNode
    from src.graph.nodes.triage import TriageNode
    from src.graph.state import PatientState, Symptom
    from src.agents.prompts.triage_prompts import get_triage_action_message
    
    # Create temporary state
    state = PatientState(session_id=f"quick_{uuid.uuid4().hex[:8]}")
    
    # Add symptoms
    for symptom_text in request.symptoms:
        state.add_symptom(Symptom(
            name=symptom_text,
            duration=request.duration,
            severity=request.severity
        ))
    
    # Add as conversation turn for context
    combined_symptoms = ", ".join(request.symptoms)
    state.add_conversation_turn(
        role="patient",
        content=combined_symptoms,
        original_language=SupportedLanguage(request.language)
    )
    
    # Run guardian for red flag detection
    guardian = GuardianNode()
    state = guardian.process(state)
    
    # Run triage assessment
    triage = TriageNode()
    state = triage.process(state)
    
    # Get response language
    try:
        language = SupportedLanguage(request.language)
    except ValueError:
        language = SupportedLanguage.URDU
    
    # Get action messages
    action_english = get_triage_action_message(state.current_triage_level, SupportedLanguage.ENGLISH)
    action_local = get_triage_action_message(state.current_triage_level, language)
    
    # Get matched patterns
    matched_patterns = [rf.pattern_name for rf in state.red_flags]
    
    # Urgency description
    urgency_map = {
        "IMMEDIATE": "Life-threatening - needs immediate care",
        "VERY_URGENT": "Serious - needs care within 10 minutes",
        "URGENT": "Needs care within 1 hour",
        "STANDARD": "Can wait, schedule appointment",
        "NON_URGENT": "Minor issue, routine care"
    }
    
    return QuickTriageResponse(
        triage_level=state.current_triage_level.value,
        triage_color=get_triage_color(state.current_triage_level.value),
        urgency=urgency_map.get(state.current_triage_level.value, "Unknown"),
        action_required=action_english,
        action_local_language=action_local,
        matched_patterns=matched_patterns,
        recommendation=state.emergency_action or action_english
    )


@router.post("/chat/feedback", tags=["Chat"])
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit feedback on AI response.
    
    Used for:
    - Quality monitoring
    - Model improvement
    - User satisfaction tracking
    """
    logger.info(
        f"Feedback received - Session: {request.session_id}, "
        f"Type: {request.feedback_type}, Rating: {request.rating}"
    )
    
    # In production, store feedback in database
    # For now, just log it
    feedback_data = {
        "session_id": request.session_id,
        "message_id": request.message_id,
        "rating": request.rating,
        "feedback_type": request.feedback_type,
        "comment": request.comment,
        "timestamp": datetime.now().isoformat()
    }
    
    # Background task to store feedback
    async def store_feedback(data: dict):
        # Would store to database here
        logger.info(f"Feedback stored: {data}")
    
    background_tasks.add_task(store_feedback, feedback_data)
    
    return {
        "status": "received",
        "message": "Thank you for your feedback"
    }


@router.get("/chat/phrases/{language}", tags=["Chat"])
async def get_common_phrases(
    language: str
):
    """
    Get common phrases in specified language.
    
    Useful for building UI with pre-defined options.
    """
    from src.agents.prompts.translator_prompts import COMMON_PHRASES
    
    try:
        lang = SupportedLanguage(language)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    
    phrases = {}
    for key, translations in COMMON_PHRASES.items():
        if lang in translations:
            phrases[key] = {
                "text": translations[lang],
                "english": translations.get(SupportedLanguage.ENGLISH, "")
            }
    
    return {
        "language": language,
        "phrases": phrases
    }