"""
Triage Node - WHO Emergency Triage Assessment
==============================================
This implements Case Study 2: Real-Time Emergency Triage.

Assigns WHO Emergency Triage levels:
- IMMEDIATE: Life-threatening, needs care NOW
- VERY_URGENT: Serious, needs care within 10 minutes
- URGENT: Needs care within 1 hour
- STANDARD: Can wait, non-urgent
- NON_URGENT: Minor issues

Also generates specific action plans:
- "Call 1122 NOW" vs "Visit ER within 2 hours"
"""

import logging
from typing import Optional
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.graph.state import PatientState, RedFlag
from src.config import settings, TriageLevel, SupportedLanguage

logger = logging.getLogger("triage")


class TriageAssessment(BaseModel):
    """Structured triage assessment output."""
    triage_level: str = Field(
        description="IMMEDIATE, VERY_URGENT, URGENT, STANDARD, or NON_URGENT"
    )
    confidence: float = Field(ge=0, le=1, description="Confidence in assessment")
    primary_concern: str = Field(description="The main clinical concern")
    differential_diagnoses: list[str] = Field(
        default_factory=list,
        description="Possible conditions to consider"
    )
    action_plan: str = Field(description="Specific action the patient should take")
    action_plan_urdu: str = Field(description="Action plan in Urdu")
    time_sensitivity: str = Field(description="How quickly action is needed")
    reasoning: str = Field(description="Clinical reasoning for triage decision")
    red_flags_present: list[str] = Field(
        default_factory=list,
        description="Red flags identified in assessment"
    )
    vital_signs_needed: list[str] = Field(
        default_factory=list,
        description="Vital signs that should be checked"
    )


# Triage decision matrix
TRIAGE_CRITERIA = {
    TriageLevel.IMMEDIATE: {
        "description": "Life-threatening conditions requiring immediate intervention",
        "conditions": [
            "Cardiac arrest / Respiratory arrest",
            "Airway obstruction",
            "Severe respiratory distress",
            "Central cyanosis",
            "Signs of shock",
            "Currently seizing",
            "Unresponsive / Unconscious",
            "Active massive hemorrhage"
        ],
        "action": "Call 1122 NOW. Do not wait.",
        "action_urdu": "ابھی 1122 کال کریں۔ انتظار نہ کریں۔"
    },
    TriageLevel.VERY_URGENT: {
        "description": "Very sick, at risk of deterioration, needs care within 10 min",
        "conditions": [
            "Severe chest pain (possible MI)",
            "Stroke symptoms (FAST positive)",
            "Severe abdominal pain",
            "Signs of sepsis",
            "Active psychosis with risk",
            "Diabetic emergency (DKA/HHS)",
            "Severe allergic reaction",
            "High fever with confusion"
        ],
        "action": "Go to Emergency Room immediately. Do not drive yourself.",
        "action_urdu": "فوری طور پر ایمرجنسی روم جائیں۔ خود گاڑی نہ چلائیں۔"
    },
    TriageLevel.URGENT: {
        "description": "Needs attention soon, within 1 hour",
        "conditions": [
            "Moderate abdominal pain",
            "Moderate chest pain (low cardiac risk)",
            "Moderate breathing difficulty",
            "High fever > 39°C",
            "Possible fracture (stable)",
            "Moderate dehydration",
            "Acute psychiatric symptoms (stable)"
        ],
        "action": "Visit ER within 2 hours or urgent care clinic.",
        "action_urdu": "2 گھنٹے کے اندر ایمرجنسی یا ارجنٹ کیئر کلینک جائیں۔"
    },
    TriageLevel.STANDARD: {
        "description": "Can wait, needs care but not urgent",
        "conditions": [
            "Mild pain",
            "Chronic conditions with mild worsening",
            "Minor injuries",
            "Mild infections",
            "Routine follow-up needed"
        ],
        "action": "Schedule appointment with doctor within 24-48 hours.",
        "action_urdu": "24-48 گھنٹوں میں ڈاکٹر سے ملاقات کا وقت لیں۔"
    },
    TriageLevel.NON_URGENT: {
        "description": "Minor issues, can wait",
        "conditions": [
            "Minor cold symptoms",
            "Chronic stable conditions",
            "Health advice needed",
            "Prescription refills",
            "Administrative issues"
        ],
        "action": "See doctor at your convenience or use telemedicine.",
        "action_urdu": "اپنی سہولت کے مطابق ڈاکٹر سے ملیں یا ٹیلی میڈیسن استعمال کریں۔"
    }
}


class TriageNode:
    """
    WHO Emergency Triage Node - assigns triage levels and action plans.
    
    This node:
    1. Analyzes all gathered symptoms
    2. Applies WHO triage criteria
    3. Considers red flags from Guardian
    4. Generates specific action plans
    5. Provides reasoning for transparency
    """
    
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.LLM_MODEL,
            temperature=0.2,  # Low temperature for consistent triage
            api_key=settings.GROQ_API_KEY
        )
        self.parser = PydanticOutputParser(pydantic_object=TriageAssessment)
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Set up the triage assessment prompt."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a WHO Emergency Triage Assessment AI.

YOUR ROLE:
- Assign appropriate triage levels based on symptoms
- You are NOT diagnosing - you are prioritizing
- Err on the side of caution (higher urgency when uncertain)

TRIAGE LEVELS (WHO ETAT):

1. IMMEDIATE (Red):
   - Life-threatening, needs care NOW
   - Examples: Cardiac arrest, airway obstruction, shock, unresponsive
   - Action: "Call 1122 NOW"

2. VERY_URGENT (Orange):
   - Very sick, needs care within 10 minutes
   - Examples: Possible MI, stroke, severe abdominal pain, DKA
   - Action: "Go to ER immediately"

3. URGENT (Yellow):
   - Needs care within 1 hour
   - Examples: Moderate pain, high fever, possible fracture
   - Action: "Visit ER within 2 hours"

4. STANDARD (Green):
   - Can wait, but needs care
   - Examples: Mild pain, minor infections
   - Action: "Schedule appointment within 24-48 hours"

5. NON_URGENT (Blue):
   - Minor issues
   - Examples: Chronic stable, health advice
   - Action: "See doctor at convenience"

CRITICAL RED FLAG COMBINATIONS:
- Chest pain + arm/jaw radiation + sweating = IMMEDIATE (MI)
- Sudden severe headache + neck stiffness = IMMEDIATE (SAH)
- Difficulty breathing + leg swelling = IMMEDIATE (PE)
- Fatigue + thirst + frequent urination = VERY_URGENT (Diabetes)

PATIENT INFORMATION:
- Symptoms: {symptoms}
- Duration: {symptom_durations}
- Red Flags Already Detected: {red_flags}
- Age: {age}
- Medical History: {medical_history}

CONVERSATION SUMMARY:
{conversation_summary}

ASSESSMENT RULES:
1. If ANY immediate red flag → IMMEDIATE
2. If uncertain between levels → choose MORE urgent
3. Consider patient age (elderly/children = higher urgency)
4. Consider symptom duration (sudden onset = higher urgency)
5. Always provide specific, actionable instructions

{format_instructions}"""),
            ("human", "Perform triage assessment for this patient.")
        ])
    
    def process(self, state: PatientState) -> PatientState:
        """
        Process patient state and assign triage level.
        
        Args:
            state: Current PatientState
            
        Returns:
            Updated PatientState with triage assessment
        """
        logger.info(f"Triage processing session {state.session_id}")
        
        # If already marked as emergency by Guardian, use that
        if state.is_emergency and state.current_triage_level == TriageLevel.IMMEDIATE:
            logger.info("Patient already triaged as IMMEDIATE by Guardian")
            state = self._confirm_emergency_triage(state)
            return state
        
        # Perform full triage assessment
        try:
            assessment = self._perform_assessment(state)
            state = self._apply_assessment(state, assessment)
        except Exception as e:
            logger.error(f"Triage assessment failed: {e}")
            # Conservative fallback - if we can't assess, mark as urgent
            state = self._apply_fallback_triage(state)
        
        return state
    
    def _perform_assessment(self, state: PatientState) -> TriageAssessment:
        """Perform LLM-based triage assessment."""
        
        # Build symptom information
        symptoms = state.get_symptom_summary()
        
        symptom_durations = []
        for s in state.symptoms:
            if s.duration:
                symptom_durations.append(f"{s.name}: {s.duration}")
        
        # Get red flags
        red_flags = [rf.pattern_name for rf in state.red_flags]
        
        # Build conversation summary
        conversation_summary = self._build_conversation_summary(state)
        
        # Get patient info
        age = state.demographics.age or "Unknown"
        medical_history = ", ".join(state.medical_history.past_conditions) or "Not provided"
        
        chain = self.prompt | self.llm | self.parser
        
        result = chain.invoke({
            "symptoms": symptoms,
            "symptom_durations": "; ".join(symptom_durations) if symptom_durations else "Not specified",
            "red_flags": ", ".join(red_flags) if red_flags else "None detected",
            "age": age,
            "medical_history": medical_history,
            "conversation_summary": conversation_summary,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return result
    
    def _build_conversation_summary(self, state: PatientState) -> str:
        """Build a summary of the conversation for triage."""
        if not state.conversation_turns:
            return "No conversation recorded."
        
        # Extract key patient statements
        patient_statements = []
        for turn in state.conversation_turns:
            if turn.role == "patient":
                content = turn.english_translation or turn.content
                patient_statements.append(content)
        
        return " | ".join(patient_statements[-5:])  # Last 5 statements
    
    def _apply_assessment(self, state: PatientState, assessment: TriageAssessment) -> PatientState:
        """Apply triage assessment to patient state."""
        
        # Map string to enum
        triage_map = {
            "IMMEDIATE": TriageLevel.IMMEDIATE,
            "VERY_URGENT": TriageLevel.VERY_URGENT,
            "URGENT": TriageLevel.URGENT,
            "STANDARD": TriageLevel.STANDARD,
            "NON_URGENT": TriageLevel.NON_URGENT
        }
        
        triage_level = triage_map.get(assessment.triage_level, TriageLevel.STANDARD)
        
        # Update state
        state.current_triage_level = triage_level
        
        if triage_level == TriageLevel.IMMEDIATE:
            state.is_emergency = True
            state.emergency_action = assessment.action_plan
            state.should_escalate = True
            state.escalation_reason = assessment.primary_concern
        
        # Add triage response to conversation
        response_content = self._format_triage_response(assessment, state.demographics.preferred_language)
        
        state.add_conversation_turn(
            role="agent",
            content=response_content,
            original_language=state.demographics.preferred_language,
            english_translation=assessment.action_plan,
            reasoning=assessment.reasoning
        )
        
        # Add to reasoning trace
        state.add_reasoning(
            f"Triage: Level={triage_level.value}, Confidence={assessment.confidence:.0%}. "
            f"Primary concern: {assessment.primary_concern}. "
            f"Action: {assessment.action_plan}"
        )
        
        return state
    
    def _format_triage_response(self, assessment: TriageAssessment, language: SupportedLanguage) -> str:
        """Format the triage response in the appropriate language."""
        
        if language == SupportedLanguage.URDU:
            if assessment.triage_level == "IMMEDIATE":
                return f"⚠️ فوری توجہ درکار ہے!\n\n{assessment.action_plan_urdu}\n\nیہ ایک ایمرجنسی ہو سکتی ہے۔"
            elif assessment.triage_level == "VERY_URGENT":
                return f"⚠️ بہت اہم!\n\n{assessment.action_plan_urdu}"
            else:
                return assessment.action_plan_urdu
        else:
            if assessment.triage_level == "IMMEDIATE":
                return f"⚠️ IMMEDIATE ATTENTION REQUIRED!\n\n{assessment.action_plan}\n\nThis may be an emergency."
            elif assessment.triage_level == "VERY_URGENT":
                return f"⚠️ VERY URGENT!\n\n{assessment.action_plan}"
            else:
                return assessment.action_plan
    
    def _confirm_emergency_triage(self, state: PatientState) -> PatientState:
        """Confirm and enhance existing emergency triage."""
        
        # The Guardian already marked this as emergency
        # Just add a triage response to the conversation
        
        emergency_response = {
            SupportedLanguage.ENGLISH: (
                f"⚠️ EMERGENCY DETECTED!\n\n"
                f"{state.emergency_action}\n\n"
                f"Do not delay. Every minute matters."
            ),
            SupportedLanguage.URDU: (
                f"⚠️ ایمرجنسی کا پتہ چلا!\n\n"
                f"ابھی 1122 کال کریں۔\n\n"
                f"تاخیر نہ کریں۔ ہر لمحہ اہم ہے۔"
            )
        }
        
        lang = state.demographics.preferred_language
        response = emergency_response.get(lang, emergency_response[SupportedLanguage.ENGLISH])
        
        state.add_conversation_turn(
            role="agent",
            content=response,
            original_language=lang,
            reasoning="Emergency triage confirmation"
        )
        
        return state
    
    def _apply_fallback_triage(self, state: PatientState) -> PatientState:
        """Apply conservative fallback triage when assessment fails."""
        
        # Default to URGENT if we can't assess properly
        state.current_triage_level = TriageLevel.URGENT
        
        fallback_response = {
            SupportedLanguage.ENGLISH: (
                "Based on your symptoms, I recommend visiting an Emergency Room "
                "or Urgent Care clinic within the next few hours for proper evaluation."
            ),
            SupportedLanguage.URDU: (
                "آپ کی علامات کی بنیاد پر، میں تجویز کرتا ہوں کہ آپ اگلے چند گھنٹوں میں "
                "ایمرجنسی روم یا ارجنٹ کیئر کلینک جائیں۔"
            )
        }
        
        lang = state.demographics.preferred_language
        response = fallback_response.get(lang, fallback_response[SupportedLanguage.ENGLISH])
        
        state.add_conversation_turn(
            role="agent",
            content=response,
            original_language=lang,
            reasoning="Fallback triage - conservative approach"
        )
        
        state.add_reasoning(
            "Triage: Applied fallback (URGENT) due to assessment error. "
            "Conservative approach when uncertain."
        )
        
        return state


# ===================
# STANDALONE TESTING
# ===================

if __name__ == "__main__":
    # Test the triage node
    from src.graph.state import PatientState, Symptom
    from src.config import SupportedLanguage
    
    # Test Case 2: Chest pain scenario
    state = PatientState(session_id="test_triage")
    state.demographics.preferred_language = SupportedLanguage.URDU
    
    # Add concerning symptoms
    state.add_symptom(Symptom(name="chest pain", severity="severe", duration="1 hour"))
    state.add_symptom(Symptom(name="pain radiating to left arm"))
    state.add_symptom(Symptom(name="sweating"))
    
    state.add_conversation_turn(
        role="patient",
        content="مجھے شدید سینے میں درد ہے اور بازو میں درد ہے",
        original_language=SupportedLanguage.URDU,
        english_translation="I have severe chest pain and pain in my arm"
    )
    
    # Run triage
    triage = TriageNode()
    result_state = triage.process(state)
    
    print(f"Triage Level: {result_state.current_triage_level}")
    print(f"Is Emergency: {result_state.is_emergency}")
    print(f"\nResponse:")
    for turn in result_state.conversation_turns:
        if turn.role == "agent":
            print(f"  {turn.content}")
    
    print(f"\nReasoning Trace:")
    for reason in result_state.reasoning_trace:
        print(f"  {reason}")