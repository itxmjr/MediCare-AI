"""
Patient State Management
========================
This is the "Secret Sauce" - the structured state that enables adaptive questioning.

Instead of passing raw chat history, we maintain a structured PatientState that:
- Tracks identified symptoms
- Records red flags detected
- Maintains conversation context
- Stores patient demographics
- Tracks the interview progress
- Holds the triage assessment
"""

from typing import Optional, Annotated
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

from src.config import TriageLevel, SupportedLanguage


class InterviewPhase(str, Enum):
    """Current phase of the patient interview."""
    GREETING = "greeting"
    CHIEF_COMPLAINT = "chief_complaint"
    HISTORY_PRESENT_ILLNESS = "history_present_illness"
    REVIEW_OF_SYSTEMS = "review_of_systems"
    PAST_MEDICAL_HISTORY = "past_medical_history"
    FAMILY_HISTORY = "family_history"
    SOCIAL_HISTORY = "social_history"
    RED_FLAG_ASSESSMENT = "red_flag_assessment"
    CLOSING = "closing"
    EMERGENCY_ESCALATION = "emergency_escalation"
    COMPLETED = "completed"


class Symptom(BaseModel):
    """Structured symptom with clinical attributes."""
    name: str = Field(..., description="Symptom name")
    duration: Optional[str] = Field(None, description="How long the symptom has been present")
    severity: Optional[str] = Field(None, description="Severity: mild/moderate/severe")
    frequency: Optional[str] = Field(None, description="How often it occurs")
    onset: Optional[str] = Field(None, description="Sudden or gradual onset")
    aggravating_factors: list[str] = Field(default_factory=list)
    relieving_factors: list[str] = Field(default_factory=list)
    associated_symptoms: list[str] = Field(default_factory=list)
    location: Optional[str] = Field(None, description="Body location if applicable")
    radiation: Optional[str] = Field(None, description="Does pain radiate?")
    
    def to_clinical_string(self) -> str:
        """Convert to clinical summary string."""
        parts = [self.name]
        if self.duration:
            parts.append(f"for {self.duration}")
        if self.severity:
            parts.append(f"({self.severity})")
        if self.onset:
            parts.append(f"onset: {self.onset}")
        return " ".join(parts)


class RedFlag(BaseModel):
    """Emergency red flag detection record."""
    pattern_name: str = Field(..., description="Name of the red flag pattern matched")
    matched_symptoms: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)
    triage_level: TriageLevel
    recommended_action: str
    detected_at: datetime = Field(default_factory=datetime.now)
    reasoning: str = Field(..., description="Why this red flag was triggered")
    
    def is_immediate(self) -> bool:
        """Check if this requires immediate action."""
        return self.triage_level == TriageLevel.IMMEDIATE


class ConversationTurn(BaseModel):
    """Single turn in the conversation."""
    role: str = Field(..., description="'patient' or 'agent'")
    content: str
    original_language: SupportedLanguage
    english_translation: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    phase: InterviewPhase
    reasoning: Optional[str] = Field(None, description="Agent's reasoning for explainability")


class VitalSigns(BaseModel):
    """Patient vital signs if available."""
    blood_pressure: Optional[str] = None
    heart_rate: Optional[int] = None
    temperature: Optional[float] = None
    respiratory_rate: Optional[int] = None
    oxygen_saturation: Optional[int] = None
    blood_glucose: Optional[float] = None


class PatientDemographics(BaseModel):
    """Basic patient information."""
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    phone_number: Optional[str] = None
    preferred_language: SupportedLanguage = SupportedLanguage.URDU


class MedicalHistory(BaseModel):
    """Patient's medical history."""
    past_conditions: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    surgeries: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)


class SOAPNote(BaseModel):
    """
    SOAP Note Format for Clinical Summary.
    This is the final output for doctors.
    """
    # SUBJECTIVE
    chief_complaint: str = Field(default="")
    history_of_present_illness: str = Field(default="")
    review_of_systems: str = Field(default="")
    past_medical_history: str = Field(default="")
    family_history: str = Field(default="")
    social_history: str = Field(default="")
    
    # OBJECTIVE (Limited - we're pre-consultation)
    vital_signs: Optional[VitalSigns] = None
    observed_distress: Optional[str] = None
    
    # ASSESSMENT
    primary_assessment: str = Field(default="")
    differential_diagnoses: list[str] = Field(default_factory=list)
    red_flags_identified: list[str] = Field(default_factory=list)
    triage_level: TriageLevel = TriageLevel.STANDARD
    confidence_score: float = Field(default=0.0, ge=0, le=1)
    
    # PLAN
    recommended_actions: list[str] = Field(default_factory=list)
    tests_suggested: list[str] = Field(default_factory=list)
    urgency: str = Field(default="")
    follow_up_instructions: str = Field(default="")
    
    def to_markdown(self) -> str:
        """Generate a formatted SOAP note for the doctor."""
        md = "# CLINICAL SUMMARY (AI-GENERATED)\n\n"
        md += f"**Triage Level:** {self.triage_level.value}\n"
        md += f"**Confidence:** {self.confidence_score:.0%}\n\n"
        
        md += "## SUBJECTIVE\n"
        md += f"**Chief Complaint:** {self.chief_complaint}\n\n"
        md += f"**History of Present Illness:**\n{self.history_of_present_illness}\n\n"
        
        if self.past_medical_history:
            md += f"**Past Medical History:** {self.past_medical_history}\n\n"
        if self.family_history:
            md += f"**Family History:** {self.family_history}\n\n"
        
        md += "## ASSESSMENT\n"
        md += f"{self.primary_assessment}\n\n"
        
        if self.red_flags_identified:
            md += "**⚠️ RED FLAGS:**\n"
            for flag in self.red_flags_identified:
                md += f"- {flag}\n"
            md += "\n"
        
        if self.differential_diagnoses:
            md += "**Differential Diagnoses:**\n"
            for dx in self.differential_diagnoses:
                md += f"- {dx}\n"
            md += "\n"
        
        md += "## PLAN\n"
        if self.recommended_actions:
            for action in self.recommended_actions:
                md += f"- {action}\n"
        
        if self.tests_suggested:
            md += "\n**Suggested Tests:**\n"
            for test in self.tests_suggested:
                md += f"- {test}\n"
        
        md += f"\n**Urgency:** {self.urgency}\n"
        
        return md


class PatientState(BaseModel):
    """
    The central state object that flows through the LangGraph.
    
    This is what makes the system "Agentic" - we maintain structured
    state that allows for adaptive questioning and intelligent routing.
    """
    
    # ===================
    # SESSION METADATA
    # ===================
    session_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    channel: str = Field(default="api", description="whatsapp/voice/api")
    
    # ===================
    # PATIENT INFO
    # ===================
    demographics: PatientDemographics = Field(default_factory=PatientDemographics)
    medical_history: MedicalHistory = Field(default_factory=MedicalHistory)
    
    # ===================
    # CURRENT INTERVIEW
    # ===================
    current_phase: InterviewPhase = Field(default=InterviewPhase.GREETING)
    question_count: int = Field(default=0)
    
    # Symptoms identified during interview
    symptoms: list[Symptom] = Field(default_factory=list)
    symptom_keywords: list[str] = Field(
        default_factory=list,
        description="Raw symptom keywords for red flag matching"
    )
    
    # ===================
    # CONVERSATION HISTORY
    # ===================
    # Using Annotated with add_messages for LangGraph compatibility
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    conversation_turns: list[ConversationTurn] = Field(default_factory=list)
    
    # ===================
    # RED FLAGS & TRIAGE
    # ===================
    red_flags: list[RedFlag] = Field(default_factory=list)
    current_triage_level: TriageLevel = Field(default=TriageLevel.STANDARD)
    is_emergency: bool = Field(default=False)
    emergency_action: Optional[str] = None
    
    # ===================
    # RAG CONTEXT
    # ===================
    retrieved_guidelines: list[str] = Field(
        default_factory=list,
        description="Guidelines retrieved from RAG for context"
    )
    
    # ===================
    # FINAL OUTPUT
    # ===================
    soap_note: Optional[SOAPNote] = None
    interview_complete: bool = Field(default=False)
    
    # ===================
    # ROUTING DECISIONS
    # ===================
    next_node: Optional[str] = Field(
        None,
        description="Next node to route to (set by router)"
    )
    should_escalate: bool = Field(default=False)
    escalation_reason: Optional[str] = None
    
    # ===================
    # EXPLAINABILITY
    # ===================
    reasoning_trace: list[str] = Field(
        default_factory=list,
        description="Agent reasoning for transparency"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_symptom(self, symptom: Symptom) -> None:
        """Add a symptom and update keywords."""
        self.symptoms.append(symptom)
        self.symptom_keywords.append(symptom.name.lower())
        for assoc in symptom.associated_symptoms:
            if assoc.lower() not in self.symptom_keywords:
                self.symptom_keywords.append(assoc.lower())
        self.last_updated = datetime.now()
    
    def add_red_flag(self, red_flag: RedFlag) -> None:
        """Add a red flag and potentially escalate."""
        self.red_flags.append(red_flag)
        if red_flag.is_immediate():
            self.is_emergency = True
            self.emergency_action = red_flag.recommended_action
            self.current_triage_level = TriageLevel.IMMEDIATE
            self.should_escalate = True
            self.escalation_reason = red_flag.pattern_name
        elif red_flag.triage_level.value < self.current_triage_level.value:
            self.current_triage_level = red_flag.triage_level
        self.last_updated = datetime.now()
    
    def add_reasoning(self, reasoning: str) -> None:
        """Add to reasoning trace for explainability."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.reasoning_trace.append(f"[{timestamp}] {reasoning}")
    
    def add_conversation_turn(
        self,
        role: str,
        content: str,
        original_language: SupportedLanguage,
        english_translation: Optional[str] = None,
        reasoning: Optional[str] = None
    ) -> None:
        """Add a conversation turn."""
        turn = ConversationTurn(
            role=role,
            content=content,
            original_language=original_language,
            english_translation=english_translation,
            phase=self.current_phase,
            reasoning=reasoning
        )
        self.conversation_turns.append(turn)
        self.last_updated = datetime.now()
    
    def get_symptom_summary(self) -> str:
        """Get a summary of all symptoms for prompts."""
        if not self.symptoms:
            return "No symptoms recorded yet."
        return "; ".join([s.to_clinical_string() for s in self.symptoms])
    
    def should_end_interview(self) -> bool:
        """Check if interview should end."""
        from src.config import settings
        if self.is_emergency:
            return True
        if self.question_count >= settings.MAX_INTERVIEW_QUESTIONS:
            return True
        if self.current_phase == InterviewPhase.COMPLETED:
            return True
        return False
    
    def can_generate_summary(self) -> bool:
        """Check if we have enough information for a summary."""
        from src.config import settings
        return (
            self.question_count >= settings.MIN_INTERVIEW_QUESTIONS
            or self.is_emergency
            or len(self.symptoms) >= 2
        )