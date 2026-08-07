"""
Guardian Node - The Safety Layer
================================
This node runs on EVERY conversation turn to detect red flags.
It's the "parallel watcher" that can interrupt normal flow for emergencies.

Red Flag Detection Strategy:
1. Pattern Matching: Known symptom combinations (chest pain + radiating + sweating)
2. Keyword Detection: Emergency keywords in any language
3. LLM Analysis: For subtle patterns humans might miss

This implements Case Study 2's requirement for real-time emergency detection.
"""

import logging
from typing import Optional
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.graph.state import PatientState, RedFlag, Symptom
from src.config import settings, RED_FLAG_PATTERNS, TriageLevel

logger = logging.getLogger("guardian")


class RedFlagAnalysis(BaseModel):
    """Structured output from LLM red flag analysis."""
    is_emergency: bool = Field(description="Is this an emergency requiring immediate action?")
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    pattern_detected: Optional[str] = Field(description="Name of the emergency pattern if detected")
    matched_symptoms: list[str] = Field(default_factory=list, description="Symptoms that triggered the alert")
    reasoning: str = Field(description="Clinical reasoning for the assessment")
    recommended_action: str = Field(description="What the patient should do immediately")
    triage_level: str = Field(description="IMMEDIATE, VERY_URGENT, URGENT, STANDARD, or NON_URGENT")


# Emergency keywords in multiple languages
EMERGENCY_KEYWORDS = {
    "en": [
        "chest pain", "can't breathe", "difficulty breathing", "unconscious",
        "severe bleeding", "worst headache", "sudden weakness", "heart attack",
        "stroke", "seizure", "crushing pain", "radiating pain", "cold sweat",
        "face drooping", "arm weakness", "speech difficulty", "thunderclap headache"
    ],
    "ur": [
        "سینے میں درد", "سانس نہیں آ رہی", "سانس لینے میں دشواری", "بے ہوش",
        "شدید خون بہنا", "سب سے بری سر درد", "اچانک کمزوری", "دل کا دورہ",
        "فالج", "مرگی کا دورہ", "دبانے والا درد", "پھیلتا درد", "ٹھنڈا پسینہ"
    ],
    "pa": [
        "چھاتی وچ درد", "ساہ نئیں آ رہی", "ساہ لین وچ مشکل", "بے ہوش",
        "بوہت زیادہ خون", "سب توں بری سر درد", "اچانک کمزوری"
    ]
}


class GuardianNode:
    """
    The Guardian Node - Always watching for emergencies.
    
    This node implements a multi-layer detection strategy:
    1. Fast Pattern Matching (no LLM call - instant)
    2. Keyword Detection (multi-language)
    3. LLM Analysis (for complex cases)
    """
    
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.LLM_MODEL,
            temperature=0.1,  # Very low for safety-critical decisions
            api_key=settings.GROQ_API_KEY
        )
        self.parser = PydanticOutputParser(pydantic_object=RedFlagAnalysis)
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Set up the red flag detection prompt."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an EMERGENCY TRIAGE AI. Your ONLY job is to detect life-threatening conditions.

You are NOT diagnosing. You are flagging potential emergencies that need immediate medical attention.

CRITICAL RED FLAG PATTERNS:
1. CARDIAC (MI): Chest pain + radiating to arm/jaw + sweating/nausea → IMMEDIATE
2. STROKE: Sudden weakness + facial droop + speech difficulty → IMMEDIATE  
3. SAH (Brain Bleed): "Worst headache of life" + sudden onset + neck stiffness → IMMEDIATE
4. PE (Lung Clot): Difficulty breathing + leg swelling + chest pain → IMMEDIATE
5. DIABETIC CRISIS: Excessive thirst + frequent urination + confusion + fruity breath → VERY_URGENT
6. APPENDICITIS: Right lower abdominal pain + fever + nausea → URGENT
7. SEPSIS: Fever + confusion + rapid breathing + rapid heart rate → IMMEDIATE

BE CONSERVATIVE: When in doubt, escalate. A false alarm is better than a missed emergency.

Current symptoms detected: {symptoms}
Full conversation context: {conversation}

{format_instructions}"""),
            ("human", "Latest patient message: {latest_message}\n\nAnalyze for red flags.")
        ])
    
    def process(self, state: PatientState) -> PatientState:
        """
        Main processing function - detect red flags in patient state.
        
        Returns:
            Updated PatientState with any detected red flags
        """
        logger.info(f"Guardian scanning session {state.session_id}")
        
        # Layer 1: Fast Pattern Matching (No LLM)
        pattern_result = self._check_patterns(state)
        if pattern_result:
            state = self._apply_red_flag(state, pattern_result)
            if pattern_result.is_emergency:
                return state  # Immediate return for emergencies
        
        # Layer 2: Keyword Detection
        keyword_result = self._check_keywords(state)
        if keyword_result and keyword_result.is_emergency:
            state = self._apply_red_flag(state, keyword_result)
            return state
        
        # Layer 3: LLM Analysis (for complex patterns)
        # Only run if we have enough context (at least 2 symptoms or concerning message)
        if len(state.symptoms) >= 2 or self._has_concerning_message(state):
            llm_result = self._llm_analysis(state)
            if llm_result and llm_result.confidence >= settings.RED_FLAG_CONFIDENCE_THRESHOLD:
                state = self._apply_red_flag(state, llm_result)
        
        return state
    
    def _check_patterns(self, state: PatientState) -> Optional[RedFlagAnalysis]:
        """
        Fast pattern matching against known red flag combinations.
        This is instant - no API calls.
        """
        symptom_keywords = [s.lower() for s in state.symptom_keywords]
        
        for pattern_name, pattern in RED_FLAG_PATTERNS.items():
            matched = []
            for symptom in pattern["symptoms"]:
                # Check if symptom or similar term is in patient's symptoms
                if any(symptom.lower() in kw or kw in symptom.lower() 
                       for kw in symptom_keywords):
                    matched.append(symptom)
            
            # Check if enough symptoms match
            if len(matched) >= pattern["combination_required"]:
                logger.warning(f"Pattern match: {pattern_name} with {matched}")
                return RedFlagAnalysis(
                    is_emergency=pattern["triage_level"] == TriageLevel.IMMEDIATE,
                    confidence=0.85,
                    pattern_detected=pattern_name,
                    matched_symptoms=matched,
                    reasoning=f"Pattern '{pattern_name}' matched with symptoms: {', '.join(matched)}",
                    recommended_action=pattern["action"],
                    triage_level=pattern["triage_level"].value
                )
        
        return None
    
    def _check_keywords(self, state: PatientState) -> Optional[RedFlagAnalysis]:
        """
        Check for emergency keywords in patient messages.
        Supports multiple languages.
        """
        if not state.conversation_turns:
            return None
        
        # Get last patient message
        last_patient_message = ""
        for turn in reversed(state.conversation_turns):
            if turn.role == "patient":
                last_patient_message = turn.content.lower()
                if turn.english_translation:
                    last_patient_message += " " + turn.english_translation.lower()
                break
        
        if not last_patient_message:
            return None
        
        # Check all language keywords
        matched_keywords = []
        for lang, keywords in EMERGENCY_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in last_patient_message:
                    matched_keywords.append(keyword)
        
        if matched_keywords:
            # Determine severity based on keywords
            critical_keywords = ["chest pain", "can't breathe", "unconscious", "worst headache",
                                "سینے میں درد", "سانس نہیں آ رہی", "بے ہوش"]
            
            is_critical = any(kw in matched_keywords for kw in critical_keywords)
            
            return RedFlagAnalysis(
                is_emergency=is_critical,
                confidence=0.75 if is_critical else 0.6,
                pattern_detected="keyword_match",
                matched_symptoms=matched_keywords,
                reasoning=f"Emergency keywords detected: {', '.join(matched_keywords)}",
                recommended_action="Seek immediate medical attention" if is_critical else "Urgent medical evaluation recommended",
                triage_level="IMMEDIATE" if is_critical else "VERY_URGENT"
            )
        
        return None
    
    def _has_concerning_message(self, state: PatientState) -> bool:
        """Check if the latest message warrants LLM analysis."""
        if not state.conversation_turns:
            return False
        
        concerning_phrases = [
            "worst", "never felt", "can't", "unable", "scared", "dying",
            "سب سے", "نہیں ہو رہا", "ڈر", "مر",
            "very bad", "terrible", "extreme", "unbearable"
        ]
        
        last_message = ""
        for turn in reversed(state.conversation_turns):
            if turn.role == "patient":
                last_message = turn.content.lower()
                break
        
        return any(phrase in last_message for phrase in concerning_phrases)
    
    def _llm_analysis(self, state: PatientState) -> Optional[RedFlagAnalysis]:
        """
        Use LLM for complex pattern detection.
        Only called when pattern matching is insufficient.
        """
        try:
            # Build conversation context
            conversation_text = ""
            for turn in state.conversation_turns[-10:]:  # Last 10 turns
                role = "Patient" if turn.role == "patient" else "AI"
                content = turn.english_translation or turn.content
                conversation_text += f"{role}: {content}\n"
            
            # Get latest message
            latest_message = ""
            for turn in reversed(state.conversation_turns):
                if turn.role == "patient":
                    latest_message = turn.english_translation or turn.content
                    break
            
            # Format prompt
            chain = self.prompt | self.llm | self.parser
            
            result = chain.invoke({
                "symptoms": state.get_symptom_summary(),
                "conversation": conversation_text,
                "latest_message": latest_message,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            logger.info(f"LLM analysis: emergency={result.is_emergency}, confidence={result.confidence}")
            return result
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return None
    
    def _apply_red_flag(self, state: PatientState, analysis: RedFlagAnalysis) -> PatientState:
        """Apply the red flag analysis to the patient state."""
        
        # Map string triage level to enum
        triage_map = {
            "IMMEDIATE": TriageLevel.IMMEDIATE,
            "VERY_URGENT": TriageLevel.VERY_URGENT,
            "URGENT": TriageLevel.URGENT,
            "STANDARD": TriageLevel.STANDARD,
            "NON_URGENT": TriageLevel.NON_URGENT
        }
        
        triage_level = triage_map.get(analysis.triage_level, TriageLevel.STANDARD)
        
        red_flag = RedFlag(
            pattern_name=analysis.pattern_detected or "llm_detected",
            matched_symptoms=analysis.matched_symptoms,
            confidence=analysis.confidence,
            triage_level=triage_level,
            recommended_action=analysis.recommended_action,
            reasoning=analysis.reasoning
        )
        
        state.add_red_flag(red_flag)
        
        # Add to reasoning trace
        state.add_reasoning(
            f"Guardian: Detected '{red_flag.pattern_name}' with confidence {red_flag.confidence:.0%}. "
            f"Symptoms: {', '.join(red_flag.matched_symptoms)}. "
            f"Action: {red_flag.recommended_action}"
        )
        
        return state


# ===================
# STANDALONE TESTING
# ===================

if __name__ == "__main__":
    # Test the guardian node
    from src.graph.state import PatientState, Symptom
    from src.config import SupportedLanguage
    
    # Create test state with concerning symptoms
    state = PatientState(session_id="test_guardian")
    
    # Add symptoms that should trigger cardiac alert
    state.add_symptom(Symptom(name="chest pain", severity="severe", duration="30 minutes"))
    state.add_symptom(Symptom(name="radiating to left arm"))
    state.add_symptom(Symptom(name="sweating", severity="profuse"))
    
    state.add_conversation_turn(
        role="patient",
        content="I have severe chest pain radiating to my arm and I'm sweating a lot",
        original_language=SupportedLanguage.ENGLISH
    )
    
    # Run guardian
    guardian = GuardianNode()
    result_state = guardian.process(state)
    
    print(f"Is Emergency: {result_state.is_emergency}")
    print(f"Triage Level: {result_state.current_triage_level}")
    print(f"Emergency Action: {result_state.emergency_action}")
    print(f"\nReasoning Trace:")
    for reason in result_state.reasoning_trace:
        print(f"  {reason}")