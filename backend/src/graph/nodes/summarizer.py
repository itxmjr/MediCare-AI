"""
Summarizer Node - SOAP Note Generation
======================================
Generates structured clinical summaries for doctors.

The SOAP format:
- Subjective: Patient's reported symptoms and history
- Objective: Observable findings (limited in pre-consultation)
- Assessment: Clinical impression and differential diagnoses
- Plan: Recommended actions and follow-up

This is the final output that doctors will see.
"""

import logging
from typing import Optional
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.graph.state import PatientState, SOAPNote, InterviewPhase
from src.config import settings, TriageLevel, SupportedLanguage

logger = logging.getLogger("summarizer")


class SOAPGeneration(BaseModel):
    """Structured SOAP note generation output."""
    
    # Subjective
    chief_complaint: str = Field(description="Main reason for visit in one sentence")
    history_of_present_illness: str = Field(description="Detailed HPI narrative")
    review_of_systems: str = Field(default="", description="Other systems reviewed")
    past_medical_history: str = Field(default="", description="Relevant PMH")
    family_history: str = Field(default="", description="Relevant family history")
    social_history: str = Field(default="", description="Relevant social history")
    
    # Assessment
    primary_assessment: str = Field(description="Main clinical impression")
    differential_diagnoses: list[str] = Field(
        default_factory=list,
        description="List of possible diagnoses to consider"
    )
    confidence_score: float = Field(ge=0, le=1, description="Confidence in assessment")
    
    # Plan
    recommended_actions: list[str] = Field(
        default_factory=list,
        description="Recommended next steps"
    )
    tests_suggested: list[str] = Field(
        default_factory=list,
        description="Lab tests or imaging to consider"
    )
    urgency: str = Field(description="Urgency level explanation")
    follow_up_instructions: str = Field(description="Follow-up care instructions")
    
    # Meta
    reasoning: str = Field(description="Clinical reasoning explanation")
    interview_quality: str = Field(description="Assessment of interview completeness")


class SummarizerNode:
    """
    Summarizer Node - Generates SOAP clinical notes.
    
    This is the final output for doctors. It synthesizes:
    1. All symptoms collected during interview
    2. Red flags identified by Guardian
    3. Triage assessment
    4. RAG-retrieved guidelines
    
    Into a structured, professional clinical summary.
    """
    
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.LLM_MODEL,
            temperature=0.3,
            api_key=settings.GROQ_API_KEY
        )
        self.parser = PydanticOutputParser(pydantic_object=SOAPGeneration)
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Set up the SOAP note generation prompt."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical scribe AI generating a clinical summary for a doctor.

YOUR TASK:
Generate a professional SOAP note based on the patient interview.

IMPORTANT:
- This is a SUPPORT tool, not a diagnostic system
- Present information objectively
- Clearly state confidence levels
- Highlight red flags prominently
- Use professional medical terminology
- Include differential diagnoses for the doctor to consider

SOAP FORMAT:

**SUBJECTIVE:**
- Chief Complaint (CC): One sentence, main reason for visit
- History of Present Illness (HPI): OLDCARTS narrative
  * Onset, Location, Duration, Character, Aggravating/Relieving factors, Timing, Severity
- Review of Systems (ROS): Other symptoms reviewed
- Past Medical History (PMH): Relevant conditions
- Family History (FHx): Relevant family conditions
- Social History (SHx): Relevant lifestyle factors

**OBJECTIVE:**
(Limited - this is pre-consultation)
- Reported vital signs if any
- Patient's apparent distress level based on descriptions

**ASSESSMENT:**
- Primary clinical impression
- Differential diagnoses (list 3-5 possibilities)
- Red flags identified (HIGHLIGHT THESE)
- Triage level: {triage_level}

**PLAN:**
- Recommended actions
- Suggested tests/labs
- Urgency of care needed
- Follow-up instructions

PATIENT INFORMATION:
- Age: {age}
- Gender: {gender}
- Language: {language}

SYMPTOMS COLLECTED:
{symptoms}

RED FLAGS IDENTIFIED:
{red_flags}

CONVERSATION TRANSCRIPT:
{conversation}

INTERVIEW METADATA:
- Questions Asked: {question_count}
- Interview Duration: {duration}
- Phases Completed: {phases}

{format_instructions}"""),
            ("human", "Generate the SOAP note for this patient interview.")
        ])
    
    def process(self, state: PatientState) -> PatientState:
        """
        Generate the SOAP note from patient state.
        
        Args:
            state: Complete PatientState with all interview data
            
        Returns:
            Updated PatientState with SOAP note attached
        """
        logger.info(f"Summarizer generating SOAP for session {state.session_id}")
        
        try:
            # Generate SOAP note
            soap_data = self._generate_soap(state)
            
            # Create SOAPNote object
            soap_note = self._build_soap_note(state, soap_data)
            
            # Attach to state
            state.soap_note = soap_note
            state.interview_complete = True
            state.current_phase = InterviewPhase.COMPLETED
            
            # Add closing message to conversation
            state = self._add_closing_message(state, soap_note)
            
            # Add to reasoning trace
            state.add_reasoning(
                f"Summarizer: Generated SOAP note. "
                f"Primary assessment: {soap_data.primary_assessment}. "
                f"Confidence: {soap_data.confidence_score:.0%}"
            )
            
        except Exception as e:
            logger.error(f"SOAP generation failed: {e}")
            state = self._generate_fallback_soap(state)
        
        return state
    
    def _generate_soap(self, state: PatientState) -> SOAPGeneration:
        """Generate SOAP note using LLM."""
        
        # Build symptoms summary
        symptoms_text = self._format_symptoms(state)
        
        # Build red flags summary
        red_flags_text = self._format_red_flags(state)
        
        # Build conversation transcript
        conversation_text = self._format_conversation(state)
        
        # Calculate interview duration
        duration = "N/A"
        if state.conversation_turns:
            start = state.created_at
            end = state.last_updated
            duration_mins = (end - start).seconds // 60
            duration = f"{duration_mins} minutes"
        
        # Get completed phases
        phases = self._get_completed_phases(state)
        
        chain = self.prompt | self.llm | self.parser
        
        result = chain.invoke({
            "triage_level": state.current_triage_level.value,
            "age": state.demographics.age or "Not provided",
            "gender": state.demographics.gender or "Not provided",
            "language": state.demographics.preferred_language.value,
            "symptoms": symptoms_text,
            "red_flags": red_flags_text,
            "conversation": conversation_text,
            "question_count": state.question_count,
            "duration": duration,
            "phases": phases,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return result
    
    def _format_symptoms(self, state: PatientState) -> str:
        """Format symptoms for the SOAP note."""
        if not state.symptoms:
            return "No specific symptoms recorded during interview."
        
        symptom_lines = []
        for i, symptom in enumerate(state.symptoms, 1):
            line = f"{i}. {symptom.name}"
            details = []
            if symptom.duration:
                details.append(f"Duration: {symptom.duration}")
            if symptom.severity:
                details.append(f"Severity: {symptom.severity}")
            if symptom.onset:
                details.append(f"Onset: {symptom.onset}")
            if symptom.location:
                details.append(f"Location: {symptom.location}")
            if symptom.aggravating_factors:
                details.append(f"Worse with: {', '.join(symptom.aggravating_factors)}")
            if symptom.relieving_factors:
                details.append(f"Better with: {', '.join(symptom.relieving_factors)}")
            
            if details:
                line += f" ({'; '.join(details)})"
            symptom_lines.append(line)
        
        return "\n".join(symptom_lines)
    
    def _format_red_flags(self, state: PatientState) -> str:
        """Format red flags for the SOAP note."""
        if not state.red_flags:
            return "No red flags detected during interview."
        
        flag_lines = []
        for rf in state.red_flags:
            flag_lines.append(
                f"⚠️ {rf.pattern_name.upper()} (Confidence: {rf.confidence:.0%})\n"
                f"   Matched symptoms: {', '.join(rf.matched_symptoms)}\n"
                f"   Recommended: {rf.recommended_action}"
            )
        
        return "\n".join(flag_lines)
    
    def _format_conversation(self, state: PatientState) -> str:
        """Format conversation transcript."""
        if not state.conversation_turns:
            return "No conversation recorded."
        
        lines = []
        for turn in state.conversation_turns:
            role = "Patient" if turn.role == "patient" else "AI"
            content = turn.english_translation or turn.content
            # Truncate long messages
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def _get_completed_phases(self, state: PatientState) -> str:
        """Get list of completed interview phases."""
        phases_seen = set()
        for turn in state.conversation_turns:
            phases_seen.add(turn.phase.value)
        
        return ", ".join(sorted(phases_seen))
    
    def _build_soap_note(self, state: PatientState, soap_data: SOAPGeneration) -> SOAPNote:
        """Build the final SOAPNote object."""
        
        return SOAPNote(
            # Subjective
            chief_complaint=soap_data.chief_complaint,
            history_of_present_illness=soap_data.history_of_present_illness,
            review_of_systems=soap_data.review_of_systems,
            past_medical_history=soap_data.past_medical_history,
            family_history=soap_data.family_history,
            social_history=soap_data.social_history,
            
            # Objective
            vital_signs=state.medical_history if hasattr(state, 'vital_signs') else None,
            observed_distress=None,  # Would need more info
            
            # Assessment
            primary_assessment=soap_data.primary_assessment,
            differential_diagnoses=soap_data.differential_diagnoses,
            red_flags_identified=[rf.pattern_name for rf in state.red_flags],
            triage_level=state.current_triage_level,
            confidence_score=soap_data.confidence_score,
            
            # Plan
            recommended_actions=soap_data.recommended_actions,
            tests_suggested=soap_data.tests_suggested,
            urgency=soap_data.urgency,
            follow_up_instructions=soap_data.follow_up_instructions
        )
    
    def _add_closing_message(self, state: PatientState, soap_note: SOAPNote) -> PatientState:
        """Add a closing message to the patient."""
        
        closing_messages = {
            SupportedLanguage.ENGLISH: (
                f"Thank you for sharing your health information with me. "
                f"Based on our conversation, I've prepared a summary for your doctor.\n\n"
                f"**Urgency Level:** {state.current_triage_level.value}\n"
                f"**Recommendation:** {soap_note.urgency}\n\n"
                f"Please share this summary with your healthcare provider. "
                f"This AI assessment is a support tool and does not replace professional medical advice."
            ),
            SupportedLanguage.URDU: (
                f"اپنی صحت کی معلومات شیئر کرنے کا شکریہ۔ "
                f"ہماری بات چیت کی بنیاد پر، میں نے آپ کے ڈاکٹر کے لیے ایک خلاصہ تیار کیا ہے۔\n\n"
                f"**فوری سطح:** {state.current_triage_level.value}\n"
                f"**سفارش:** {soap_note.urgency}\n\n"
                f"براہ کرم یہ خلاصہ اپنے ہیلتھ کیئر فراہم کنندہ کے ساتھ شیئر کریں۔ "
                f"یہ AI تشخیص ایک معاون ٹول ہے اور پیشہ ورانہ طبی مشورے کی جگہ نہیں لیتا۔"
            )
        }
        
        lang = state.demographics.preferred_language
        closing = closing_messages.get(lang, closing_messages[SupportedLanguage.ENGLISH])
        
        state.add_conversation_turn(
            role="agent",
            content=closing,
            original_language=lang,
            reasoning="Interview closing and summary notification"
        )
        
        return state
    
    def _generate_fallback_soap(self, state: PatientState) -> PatientState:
        """Generate a basic SOAP note when LLM fails."""
        
        # Build basic SOAP from state data
        soap_note = SOAPNote(
            chief_complaint=state.symptoms[0].name if state.symptoms else "Symptoms discussed",
            history_of_present_illness=state.get_symptom_summary(),
            primary_assessment="Please review patient symptoms.",
            differential_diagnoses=["Requires clinical evaluation"],
            red_flags_identified=[rf.pattern_name for rf in state.red_flags],
            triage_level=state.current_triage_level,
            confidence_score=0.5,
            recommended_actions=["Clinical evaluation recommended"],
            tests_suggested=["As clinically indicated"],
            urgency=f"Triage level: {state.current_triage_level.value}",
            follow_up_instructions="Follow up as needed based on clinical evaluation"
        )
        
        state.soap_note = soap_note
        state.interview_complete = True
        
        state.add_reasoning(
            "Summarizer: Generated fallback SOAP due to processing error"
        )
        
        return state


# ===================
# STANDALONE TESTING
# ===================

if __name__ == "__main__":
    # Test the summarizer node
    from src.graph.state import PatientState, Symptom, RedFlag
    from src.config import SupportedLanguage, TriageLevel
    
    # Create test state with full interview data
    state = PatientState(session_id="test_summarizer")
    state.demographics.preferred_language = SupportedLanguage.URDU
    state.demographics.age = 45
    state.demographics.gender = "male"
    
    # Add symptoms (diabetic farmer scenario)
    state.add_symptom(Symptom(
        name="chronic fatigue",
        duration="2-3 months",
        severity="moderate"
    ))
    state.add_symptom(Symptom(
        name="excessive thirst",
        severity="severe"
    ))
    state.add_symptom(Symptom(
        name="frequent urination",
        duration="2 months",
        frequency="3-4 times per night"
    ))
    state.add_symptom(Symptom(
        name="unintentional weight loss",
        duration="recent"
    ))
    
    # Add red flag
    state.add_red_flag(RedFlag(
        pattern_name="diabetic_emergency",
        matched_symptoms=["excessive thirst", "frequent urination", "fatigue"],
        confidence=0.85,
        triage_level=TriageLevel.VERY_URGENT,
        recommended_action="Urgent blood glucose test required",
        reasoning="Classic triad of diabetes symptoms"
    ))
    
    # Add conversation turns
    state.add_conversation_turn(
        role="patient",
        content="مجھے بہت تھکاوٹ ہو رہی ہے",
        original_language=SupportedLanguage.URDU,
        english_translation="I am feeling very tired"
    )
    state.add_conversation_turn(
        role="agent",
        content="How long have you been feeling tired?",
        original_language=SupportedLanguage.ENGLISH
    )
    state.add_conversation_turn(
        role="patient",
        content="تقریباً 2-3 مہینے",
        original_language=SupportedLanguage.URDU,
        english_translation="About 2-3 months"
    )
    state.add_conversation_turn(
        role="agent",
        content="Do you feel very thirsty?",
        original_language=SupportedLanguage.ENGLISH
    )
    state.add_conversation_turn(
        role="patient",
        content="ہاں، بہت زیادہ",
        original_language=SupportedLanguage.URDU,
        english_translation="Yes, very much"
    )
    
    state.question_count = 5
    state.current_triage_level = TriageLevel.VERY_URGENT
    state.medical_history.family_history = ["Father with Type 2 Diabetes"]
    
    # Run summarizer
    summarizer = SummarizerNode()
    result_state = summarizer.process(state)
    
    print("=" * 60)
    print("GENERATED SOAP NOTE")
    print("=" * 60)
    if result_state.soap_note:
        print(result_state.soap_note.to_markdown())
    
    print("\nReasoning Trace:")
    for reason in result_state.reasoning_trace:
        print(f"  {reason}")