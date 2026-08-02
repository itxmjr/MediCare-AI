"""
Schema Definitions
==================
Pydantic schemas for data validation and serialization.

This module defines:
1. Patient information schemas
2. Clinical data schemas
3. API request/response schemas
4. Internal data transfer objects

All schemas include:
- Field validation
- Serialization/deserialization
- Documentation
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import re


# =============================================================================
# ENUMS
# =============================================================================

class GenderEnum(str, Enum):
    """Patient gender options."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


class SeverityLevel(str, Enum):
    """Symptom severity levels."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class TriageLevelEnum(str, Enum):
    """WHO Triage levels."""
    IMMEDIATE = "IMMEDIATE"
    VERY_URGENT = "VERY_URGENT"
    URGENT = "URGENT"
    STANDARD = "STANDARD"
    NON_URGENT = "NON_URGENT"


class InterviewPhaseEnum(str, Enum):
    """Interview phases."""
    GREETING = "greeting"
    CHIEF_COMPLAINT = "chief_complaint"
    HISTORY_PRESENT_ILLNESS = "history_present_illness"
    REVIEW_OF_SYSTEMS = "review_of_systems"
    PAST_MEDICAL_HISTORY = "past_medical_history"
    FAMILY_HISTORY = "family_history"
    SOCIAL_HISTORY = "social_history"
    CLOSING = "closing"
    COMPLETED = "completed"
    EMERGENCY = "emergency"


class MessageRoleEnum(str, Enum):
    """Message roles in conversation."""
    PATIENT = "patient"
    AGENT = "agent"
    SYSTEM = "system"


class ChannelEnum(str, Enum):
    """Communication channels."""
    API = "api"
    WHATSAPP = "whatsapp"
    VOICE = "voice"
    WEB = "web"


# =============================================================================
# PATIENT SCHEMAS
# =============================================================================

class PatientInfo(BaseModel):
    """Patient demographic information."""
    
    name: Optional[str] = Field(None, max_length=100, description="Patient's name")
    age: Optional[int] = Field(None, ge=0, le=150, description="Patient's age in years")
    gender: Optional[GenderEnum] = Field(None, description="Patient's gender")
    phone_number: Optional[str] = Field(None, description="Contact phone number")
    preferred_language: str = Field("ur", description="Preferred language code")
    
    @field_validator('phone_number')
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # Remove non-digits
        digits = re.sub(r'\D', '', v)
        if len(digits) < 10 or len(digits) > 15:
            raise ValueError("Phone number must be 10-15 digits")
        return digits
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Ahmed Khan",
                "age": 45,
                "gender": "male",
                "phone_number": "+923001234567",
                "preferred_language": "ur"
            }
        }


class MedicalHistoryInfo(BaseModel):
    """Patient medical history."""
    
    past_conditions: List[str] = Field(default_factory=list, description="Past medical conditions")
    current_medications: List[str] = Field(default_factory=list, description="Current medications")
    allergies: List[str] = Field(default_factory=list, description="Known allergies")
    surgeries: List[str] = Field(default_factory=list, description="Previous surgeries")
    family_history: List[str] = Field(default_factory=list, description="Family medical history")
    
    class Config:
        json_schema_extra = {
            "example": {
                "past_conditions": ["Hypertension"],
                "current_medications": ["Amlodipine 5mg"],
                "allergies": ["Penicillin"],
                "surgeries": [],
                "family_history": ["Father: Type 2 Diabetes"]
            }
        }


# =============================================================================
# SYMPTOM SCHEMAS
# =============================================================================

class SymptomReport(BaseModel):
    """Individual symptom report."""
    
    name: str = Field(..., min_length=1, max_length=200, description="Symptom name")
    duration: Optional[str] = Field(None, description="How long symptom has been present")
    severity: Optional[SeverityLevel] = Field(None, description="Symptom severity")
    onset: Optional[str] = Field(None, description="How symptom started (sudden/gradual)")
    location: Optional[str] = Field(None, description="Body location")
    character: Optional[str] = Field(None, description="Nature of symptom (sharp, dull, etc.)")
    frequency: Optional[str] = Field(None, description="How often it occurs")
    aggravating_factors: List[str] = Field(default_factory=list, description="What makes it worse")
    relieving_factors: List[str] = Field(default_factory=list, description="What helps")
    associated_symptoms: List[str] = Field(default_factory=list, description="Other symptoms")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    def to_clinical_string(self) -> str:
        """Convert to clinical summary string."""
        parts = [self.name]
        if self.duration:
            parts.append(f"for {self.duration}")
        if self.severity:
            parts.append(f"({self.severity.value})")
        if self.onset:
            parts.append(f"onset: {self.onset}")
        return " ".join(parts)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Chest pain",
                "duration": "2 hours",
                "severity": "severe",
                "onset": "sudden",
                "location": "left side of chest",
                "character": "pressure-like",
                "aggravating_factors": ["exertion"],
                "relieving_factors": ["rest"],
                "associated_symptoms": ["sweating", "shortness of breath"]
            }
        }


class RedFlagAlert(BaseModel):
    """Red flag detection alert."""
    
    pattern_name: str = Field(..., description="Name of the red flag pattern")
    matched_symptoms: List[str] = Field(default_factory=list, description="Symptoms that triggered alert")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    triage_level: TriageLevelEnum = Field(..., description="Recommended triage level")
    recommended_action: str = Field(..., description="Recommended action")
    reasoning: str = Field(..., description="Clinical reasoning")
    detected_at: datetime = Field(default_factory=datetime.now, description="Detection timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pattern_name": "cardiac_emergency",
                "matched_symptoms": ["chest pain", "radiating to arm", "sweating"],
                "confidence": 0.85,
                "triage_level": "IMMEDIATE",
                "recommended_action": "Call 1122 NOW",
                "reasoning": "Classic MI presentation with chest pain, radiation, and diaphoresis"
            }
        }


# =============================================================================
# TRIAGE SCHEMAS
# =============================================================================

class TriageResult(BaseModel):
    """Triage assessment result."""
    
    triage_level: TriageLevelEnum = Field(..., description="Assigned triage level")
    confidence: float = Field(..., ge=0, le=1, description="Assessment confidence")
    primary_concern: str = Field(..., description="Main clinical concern")
    differential_diagnoses: List[str] = Field(default_factory=list, description="Possible diagnoses")
    recommended_action: str = Field(..., description="Recommended action")
    recommended_action_urdu: Optional[str] = Field(None, description="Action in Urdu")
    time_sensitivity: str = Field(..., description="How quickly action needed")
    red_flags_present: List[str] = Field(default_factory=list, description="Red flags identified")
    reasoning: str = Field(..., description="Clinical reasoning")
    assessed_at: datetime = Field(default_factory=datetime.now, description="Assessment timestamp")
    
    def get_triage_color(self) -> str:
        """Get color code for triage level."""
        colors = {
            TriageLevelEnum.IMMEDIATE: "red",
            TriageLevelEnum.VERY_URGENT: "orange",
            TriageLevelEnum.URGENT: "yellow",
            TriageLevelEnum.STANDARD: "green",
            TriageLevelEnum.NON_URGENT: "blue"
        }
        return colors.get(self.triage_level, "gray")
    
    class Config:
        json_schema_extra = {
            "example": {
                "triage_level": "VERY_URGENT",
                "confidence": 0.85,
                "primary_concern": "Suspected Type 2 Diabetes",
                "differential_diagnoses": ["Type 2 Diabetes", "Prediabetes", "Thyroid disorder"],
                "recommended_action": "Urgent blood glucose test required",
                "time_sensitivity": "Within 24 hours",
                "red_flags_present": ["Polyuria", "Polydipsia"],
                "reasoning": "Classic triad of fatigue, thirst, and frequent urination"
            }
        }


# =============================================================================
# SOAP NOTE SCHEMA
# =============================================================================

class SOAPNoteSchema(BaseModel):
    """SOAP note clinical summary."""
    
    # Subjective
    chief_complaint: str = Field(..., description="Main reason for visit")
    history_of_present_illness: str = Field(..., description="HPI narrative")
    review_of_systems: Optional[str] = Field(None, description="ROS findings")
    past_medical_history: Optional[str] = Field(None, description="PMH")
    family_history: Optional[str] = Field(None, description="Family history")
    social_history: Optional[str] = Field(None, description="Social history")
    
    # Objective (limited pre-consultation)
    vital_signs: Optional[Dict[str, Any]] = Field(None, description="Vital signs if available")
    general_appearance: Optional[str] = Field(None, description="General observations")
    
    # Assessment
    primary_assessment: str = Field(..., description="Main clinical impression")
    differential_diagnoses: List[str] = Field(default_factory=list, description="Differential list")
    triage_level: TriageLevelEnum = Field(..., description="Triage level")
    red_flags_identified: List[str] = Field(default_factory=list, description="Red flags")
    confidence_score: float = Field(..., ge=0, le=1, description="Assessment confidence")
    
    # Plan
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    tests_suggested: List[str] = Field(default_factory=list, description="Suggested tests")
    urgency: str = Field(..., description="Urgency level")
    follow_up_instructions: Optional[str] = Field(None, description="Follow-up instructions")
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    interview_duration_minutes: Optional[int] = Field(None)
    questions_asked: Optional[int] = Field(None)
    
    def to_markdown(self) -> str:
        """Generate formatted markdown SOAP note."""
        md = "# CLINICAL SUMMARY (AI-GENERATED)\n\n"
        md += f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M')}\n"
        md += f"**Triage Level:** {self.triage_level.value}\n"
        md += f"**Confidence:** {self.confidence_score:.0%}\n\n"
        
        md += "---\n\n"
        
        md += "## SUBJECTIVE\n\n"
        md += f"**Chief Complaint:** {self.chief_complaint}\n\n"
        md += f"**History of Present Illness:**\n{self.history_of_present_illness}\n\n"
        
        if self.past_medical_history:
            md += f"**Past Medical History:** {self.past_medical_history}\n\n"
        if self.family_history:
            md += f"**Family History:** {self.family_history}\n\n"
        
        md += "## ASSESSMENT\n\n"
        md += f"{self.primary_assessment}\n\n"
        
        if self.red_flags_identified:
            md += "**⚠️ RED FLAGS:**\n"
            for flag in self.red_flags_identified:
                md += f"- {flag}\n"
            md += "\n"
        
        if self.differential_diagnoses:
            md += "**Differential Diagnoses:**\n"
            for i, dx in enumerate(self.differential_diagnoses, 1):
                md += f"{i}. {dx}\n"
            md += "\n"
        
        md += "## PLAN\n\n"
        for action in self.recommended_actions:
            md += f"- {action}\n"
        
        if self.tests_suggested:
            md += "\n**Suggested Tests:**\n"
            for test in self.tests_suggested:
                md += f"- {test}\n"
        
        md += f"\n**Urgency:** {self.urgency}\n"
        
        md += "\n---\n"
        md += "*This is an AI-generated summary for clinical support. "
        md += "Final diagnosis and treatment decisions must be made by a qualified healthcare provider.*\n"
        
        return md
    
    class Config:
        json_schema_extra = {
            "example": {
                "chief_complaint": "Fatigue and excessive thirst for 2-3 months",
                "history_of_present_illness": "45-year-old male presenting with progressive fatigue...",
                "primary_assessment": "Suspected Type 2 Diabetes Mellitus",
                "triage_level": "VERY_URGENT",
                "confidence_score": 0.85
            }
        }


# =============================================================================
# CONVERSATION SCHEMAS
# =============================================================================

class ConversationMessage(BaseModel):
    """Single message in conversation."""
    
    message_id: str = Field(..., description="Unique message ID")
    role: MessageRoleEnum = Field(..., description="Who sent the message")
    content: str = Field(..., description="Message content")
    content_english: Optional[str] = Field(None, description="English translation")
    language: str = Field("ur", description="Message language")
    timestamp: datetime = Field(default_factory=datetime.now)
    phase: Optional[InterviewPhaseEnum] = Field(None, description="Interview phase")
    reasoning: Optional[str] = Field(None, description="AI reasoning (for agent messages)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "msg_abc123",
                "role": "patient",
                "content": "مجھے دو دن سے سر میں درد ہے",
                "content_english": "I have had a headache for two days",
                "language": "ur",
                "phase": "chief_complaint"
            }
        }


class InterviewSummary(BaseModel):
    """Summary of completed interview."""
    
    session_id: str = Field(..., description="Session identifier")
    patient_info: Optional[PatientInfo] = Field(None, description="Patient demographics")
    
    # Interview metrics
    started_at: datetime = Field(..., description="Interview start time")
    completed_at: datetime = Field(default_factory=datetime.now, description="Completion time")
    duration_minutes: int = Field(..., description="Interview duration")
    questions_asked: int = Field(..., description="Number of questions")
    messages_exchanged: int = Field(..., description="Total messages")
    
    # Clinical data
    symptoms: List[SymptomReport] = Field(default_factory=list, description="Identified symptoms")
    red_flags: List[RedFlagAlert] = Field(default_factory=list, description="Red flags detected")
    triage_result: Optional[TriageResult] = Field(None, description="Triage assessment")
    soap_note: Optional[SOAPNoteSchema] = Field(None, description="Generated SOAP note")
    
    # Status
    is_complete: bool = Field(False, description="Whether interview completed normally")
    is_emergency: bool = Field(False, description="Whether emergency was detected")
    emergency_action_taken: Optional[str] = Field(None, description="Emergency action if applicable")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_abc123",
                "duration_minutes": 12,
                "questions_asked": 8,
                "messages_exchanged": 16,
                "is_complete": True,
                "is_emergency": False
            }
        }


# =============================================================================
# EMERGENCY SCHEMAS
# =============================================================================

class EmergencyAlert(BaseModel):
    """Emergency alert notification."""
    
    alert_id: str = Field(..., description="Unique alert ID")
    session_id: str = Field(..., description="Session that triggered alert")
    patient_phone: Optional[str] = Field(None, description="Patient phone (masked)")
    
    # Emergency details
    triage_level: TriageLevelEnum = Field(..., description="Emergency triage level")
    pattern_detected: str = Field(..., description="Emergency pattern detected")
    symptoms: List[str] = Field(..., description="Triggering symptoms")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    
    # Actions
    recommended_action: str = Field(..., description="Recommended action (English)")
    recommended_action_urdu: str = Field(..., description="Recommended action (Urdu)")
    call_emergency_services: bool = Field(False, description="Whether to call 1122")
    
    # Metadata
    detected_at: datetime = Field(default_factory=datetime.now)
    acknowledged: bool = Field(False, description="Whether alert was acknowledged")
    acknowledged_by: Optional[str] = Field(None, description="Who acknowledged")
    acknowledged_at: Optional[datetime] = Field(None)
    
    class Config:
        json_schema_extra = {
            "example": {
                "alert_id": "alert_xyz789",
                "session_id": "session_abc123",
                "triage_level": "IMMEDIATE",
                "pattern_detected": "cardiac_emergency",
                "symptoms": ["chest pain", "arm pain", "sweating"],
                "confidence": 0.9,
                "recommended_action": "Call 1122 immediately",
                "recommended_action_urdu": "فوری طور پر 1122 کال کریں",
                "call_emergency_services": True
            }
        }


# =============================================================================
# API SCHEMAS
# =============================================================================

class APIResponse(BaseModel):
    """Standard API response wrapper."""
    
    success: bool = Field(..., description="Whether request succeeded")
    message: Optional[str] = Field(None, description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = Field(None, description="Request tracking ID")


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    
    items: List[Any] = Field(..., description="Page items")
    total: int = Field(..., description="Total items")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    total_pages: int = Field(..., description="Total pages")
    has_next: bool = Field(..., description="Has next page")
    has_previous: bool = Field(..., description="Has previous page")