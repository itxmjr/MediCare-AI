"""
Interviewer Node - Adaptive Patient Interview
==============================================
This implements Case Study 1: The Diabetic Farmer scenario.

The interviewer:
1. Asks adaptive follow-up questions based on symptoms
2. Uses RAG to retrieve relevant clinical guidelines
3. Follows SOAP interview structure
4. Works in patient's preferred language
5. Identifies symptoms for later analysis

Key Feature: The questions adapt based on previous answers.
Example: "Fatigue" → asks about thirst → "Yes" → asks about urination frequency
"""

import logging
from typing import Optional
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.graph.state import PatientState, Symptom, InterviewPhase, ConversationTurn
from src.config import settings, SupportedLanguage

logger = logging.getLogger("interviewer")


class InterviewResponse(BaseModel):
    """Structured output from the interviewer LLM."""
    response_to_patient: str = Field(description="The response/question to ask the patient")
    response_in_original_language: str = Field(description="Response translated to patient's language")
    identified_symptoms: list[dict] = Field(
        default_factory=list,
        description="New symptoms identified from patient's response"
    )
    next_phase: str = Field(description="Next interview phase to move to")
    reasoning: str = Field(description="Clinical reasoning for the question asked")
    should_end_interview: bool = Field(
        default=False,
        description="Whether enough information has been gathered"
    )
    follow_up_topics: list[str] = Field(
        default_factory=list,
        description="Topics to explore in follow-up questions"
    )


# Interview phase questions and focus areas
PHASE_GUIDANCE = {
    InterviewPhase.GREETING: {
        "focus": "Establish rapport, confirm language preference, get basic info",
        "example_questions": [
            "Hello, I'm your AI health assistant. How can I help you today?",
            "آداب، میں آپ کا AI صحت معاون ہوں۔ آج میں آپ کی کیا مدد کر سکتا ہوں؟"
        ]
    },
    InterviewPhase.CHIEF_COMPLAINT: {
        "focus": "Identify the main reason for seeking help",
        "example_questions": [
            "What is the main problem that brought you here today?",
            "What symptom is bothering you the most?"
        ]
    },
    InterviewPhase.HISTORY_PRESENT_ILLNESS: {
        "focus": "OLDCARTS: Onset, Location, Duration, Character, Aggravating factors, Relieving factors, Timing, Severity",
        "example_questions": [
            "When did this start?",
            "How severe is it on a scale of 1-10?",
            "Is it constant or does it come and go?",
            "Does anything make it better or worse?"
        ]
    },
    InterviewPhase.REVIEW_OF_SYSTEMS: {
        "focus": "Check for associated symptoms that might indicate the cause",
        "example_questions": [
            "Have you noticed any other symptoms?",
            "Any fever, weight changes, or changes in appetite?"
        ]
    },
    InterviewPhase.PAST_MEDICAL_HISTORY: {
        "focus": "Previous conditions, surgeries, medications, allergies",
        "example_questions": [
            "Do you have any ongoing medical conditions?",
            "Are you taking any medications?",
            "Do you have any allergies?"
        ]
    },
    InterviewPhase.FAMILY_HISTORY: {
        "focus": "Family history of relevant conditions",
        "example_questions": [
            "Does anyone in your family have similar problems?",
            "Any family history of diabetes, heart disease, or cancer?"
        ]
    },
    InterviewPhase.SOCIAL_HISTORY: {
        "focus": "Lifestyle factors, occupation, habits",
        "example_questions": [
            "What kind of work do you do?",
            "Do you smoke or drink alcohol?"
        ]
    },
    InterviewPhase.CLOSING: {
        "focus": "Summarize and confirm understanding",
        "example_questions": [
            "Is there anything else you'd like to mention?",
            "Let me summarize what you've told me..."
        ]
    }
}

# Symptom-specific follow-up questions
ADAPTIVE_FOLLOWUPS = {
    "fatigue": ["excessive thirst", "frequent urination", "weight changes", "sleep quality"],
    "chest pain": ["radiation", "sweating", "shortness of breath", "nausea", "exertion triggers"],
    "headache": ["onset speed", "severity", "neck stiffness", "vision changes", "nausea"],
    "abdominal pain": ["location", "nausea", "vomiting", "fever", "bowel changes"],
    "shortness of breath": ["leg swelling", "chest pain", "lying flat", "exertion"],
    "dizziness": ["spinning", "fainting", "hearing changes", "nausea"],
    "thirst": ["urination frequency", "weight loss", "fatigue", "blurred vision"],  # Diabetes screen
    "polydipsia": ["polyuria", "weight loss", "fatigue", "family diabetes"],  # Diabetes screen
}


class InterviewerNode:
    """
    The Adaptive Interviewer - conducts thorough patient interviews.
    
    Key Features:
    1. Follows SOAP interview structure
    2. Adapts questions based on symptoms
    3. Uses RAG for clinical context
    4. Supports multiple languages
    5. Tracks interview progress
    """
    
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.GROQ_API_KEY
        )
        self.parser = PydanticOutputParser(pydantic_object=InterviewResponse)
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Set up the interviewer prompt."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a skilled medical intake AI assistant conducting a patient interview.

YOUR ROLE:
- You are NOT a doctor and cannot diagnose
- You gather information to help doctors make decisions
- You ask thorough, adaptive questions
- You are empathetic and patient

INTERVIEW STRUCTURE (SOAP):
1. GREETING: Establish rapport, confirm language
2. CHIEF COMPLAINT: Main problem
3. HISTORY OF PRESENT ILLNESS (HPI): OLDCARTS method
   - Onset: When did it start?
   - Location: Where exactly?
   - Duration: How long does it last?
   - Character: What does it feel like?
   - Aggravating factors: What makes it worse?
   - Relieving factors: What helps?
   - Timing: Constant or intermittent?
   - Severity: 1-10 scale
4. REVIEW OF SYSTEMS: Associated symptoms
5. PAST MEDICAL HISTORY: Conditions, medications, allergies
6. FAMILY HISTORY: Relevant family conditions
7. SOCIAL HISTORY: Lifestyle, occupation
8. CLOSING: Confirm understanding

CURRENT STATE:
- Phase: {current_phase}
- Questions Asked: {question_count}
- Symptoms Identified: {symptoms}
- Patient Language: {language}
- Follow-up Topics to Explore: {followup_topics}

CLINICAL GUIDELINES (from RAG):
{rag_context}

ADAPTIVE QUESTIONING:
Based on the symptoms, ask relevant follow-up questions:
{adaptive_guidance}

IMPORTANT RULES:
1. Ask ONE clear question at a time
2. Use the patient's language ({language})
3. Be empathetic and reassuring
4. Extract and record specific symptoms
5. If patient mentions emergency symptoms, note them clearly
6. After {max_questions} questions, start closing the interview

{format_instructions}"""),
            ("human", """CONVERSATION HISTORY:
{conversation_history}

PATIENT'S LATEST MESSAGE: {latest_message}

Generate your response. Remember to ask in {language}.""")
        ])
    
    def process(self, state: PatientState) -> PatientState:
        """
        Process the current state and generate the next interview question.
        
        Args:
            state: Current PatientState
            
        Returns:
            Updated PatientState with new question added
        """
        logger.info(f"Interviewer processing session {state.session_id}, phase: {state.current_phase}")
        
        # Get RAG context for relevant guidelines
        rag_context = self._get_rag_context(state)
        
        # Get adaptive follow-up guidance
        adaptive_guidance = self._get_adaptive_guidance(state)
        
        # Build conversation history
        conversation_history = self._build_conversation_history(state)
        
        # Get latest patient message
        latest_message = self._get_latest_patient_message(state)
        
        # Get follow-up topics from previous symptoms
        followup_topics = self._get_followup_topics(state)
        
        # Generate response
        try:
            chain = self.prompt | self.llm | self.parser
            
            result = chain.invoke({
                "current_phase": state.current_phase.value,
                "question_count": state.question_count,
                "symptoms": state.get_symptom_summary(),
                "language": "Roman Urdu" if state.demographics.preferred_language == SupportedLanguage.ROMAN_URDU else state.demographics.preferred_language.value,
                "followup_topics": ", ".join(followup_topics) if followup_topics else "None identified yet",
                "rag_context": rag_context,
                "adaptive_guidance": adaptive_guidance,
                "max_questions": settings.MAX_INTERVIEW_QUESTIONS,
                "conversation_history": conversation_history,
                "latest_message": latest_message,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Update state with results
            state = self._update_state(state, result)
            
        except Exception as e:
            logger.error(f"Interviewer error: {e}")
            # Fallback response
            state = self._add_fallback_response(state)
        
        return state
    
    def _get_rag_context(self, state: PatientState) -> str:
        """Retrieve relevant clinical guidelines using RAG."""
        try:
            from src.agents.tools.rag_search import search_guidelines
            
            # Build search query from symptoms
            query_parts = []
            for symptom in state.symptoms:
                query_parts.append(symptom.name)
            
            if state.conversation_turns:
                last_turn = state.conversation_turns[-1]
                if last_turn.role == "patient":
                    query_parts.append(last_turn.english_translation or last_turn.content)
            
            if not query_parts:
                query_parts = ["general patient interview"]
            
            query = " ".join(query_parts)
            results = search_guidelines(query, k=3)
            
            if results:
                state.retrieved_guidelines = results
                return "\n---\n".join(results)
            
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
        
        return "No specific guidelines retrieved."
    
    def _get_adaptive_guidance(self, state: PatientState) -> str:
        """Get adaptive follow-up question guidance based on symptoms."""
        guidance_parts = []
        
        for symptom in state.symptoms:
            symptom_lower = symptom.name.lower()
            for key, followups in ADAPTIVE_FOLLOWUPS.items():
                if key in symptom_lower:
                    guidance_parts.append(
                        f"For '{symptom.name}', consider asking about: {', '.join(followups)}"
                    )
                    break
        
        if not guidance_parts:
            return "Follow standard interview flow based on chief complaint."
        
        return "\n".join(guidance_parts)
    
    def _get_followup_topics(self, state: PatientState) -> list[str]:
        """Get list of follow-up topics based on identified symptoms."""
        topics = []
        explored = set()
        
        for symptom in state.symptoms:
            symptom_lower = symptom.name.lower()
            for key, followups in ADAPTIVE_FOLLOWUPS.items():
                if key in symptom_lower:
                    for followup in followups:
                        if followup not in explored:
                            topics.append(followup)
                            explored.add(followup)
        
        return topics[:5]  # Limit to 5 topics
    
    def _build_conversation_history(self, state: PatientState) -> str:
        """Build formatted conversation history."""
        if not state.conversation_turns:
            return "No previous conversation."
        
        history_parts = []
        for turn in state.conversation_turns[-10:]:  # Last 10 turns
            role = "Patient" if turn.role == "patient" else "AI"
            content = turn.english_translation or turn.content
            history_parts.append(f"{role}: {content}")
        
        return "\n".join(history_parts)
    
    def _get_latest_patient_message(self, state: PatientState) -> str:
        """Get the latest patient message."""
        for turn in reversed(state.conversation_turns):
            if turn.role == "patient":
                return turn.english_translation or turn.content
        return ""
    
    def _update_state(self, state: PatientState, result: InterviewResponse) -> PatientState:
        """Update patient state with interview results."""
        
        # Add identified symptoms
        for symptom_dict in result.identified_symptoms:
            symptom = Symptom(
                name=symptom_dict.get("name", "unknown"),
                duration=symptom_dict.get("duration"),
                severity=symptom_dict.get("severity"),
                onset=symptom_dict.get("onset"),
                location=symptom_dict.get("location")
            )
            state.add_symptom(symptom)
        
        # Add agent response to conversation
        state.add_conversation_turn(
            role="agent",
            content=result.response_in_original_language,
            original_language=state.demographics.preferred_language,
            english_translation=result.response_to_patient,
            reasoning=result.reasoning
        )
        
        # Update interview phase
        phase_map = {
            "greeting": InterviewPhase.GREETING,
            "chief_complaint": InterviewPhase.CHIEF_COMPLAINT,
            "history_present_illness": InterviewPhase.HISTORY_PRESENT_ILLNESS,
            "review_of_systems": InterviewPhase.REVIEW_OF_SYSTEMS,
            "past_medical_history": InterviewPhase.PAST_MEDICAL_HISTORY,
            "family_history": InterviewPhase.FAMILY_HISTORY,
            "social_history": InterviewPhase.SOCIAL_HISTORY,
            "closing": InterviewPhase.CLOSING,
            "completed": InterviewPhase.COMPLETED
        }
        
        if result.next_phase in phase_map:
            state.current_phase = phase_map[result.next_phase]
        
        # Check if interview should end
        if result.should_end_interview:
            state.interview_complete = True
            state.current_phase = InterviewPhase.COMPLETED
        
        # Add reasoning
        state.add_reasoning(
            f"Interviewer: Phase={state.current_phase.value}. "
            f"Reasoning: {result.reasoning}"
        )
        
        return state
    
    def _add_fallback_response(self, state: PatientState) -> PatientState:
        """Add a fallback response if LLM fails."""
        fallback_responses = {
            SupportedLanguage.ENGLISH: "I understand. Can you tell me more about how you're feeling?",
            SupportedLanguage.URDU: "میں سمجھ گیا۔ کیا آپ مجھے بتا سکتے ہیں کہ آپ کو کیسا محسوس ہو رہا ہے؟",
            SupportedLanguage.PUNJABI: "میں سمجھ گیا۔ کی تسیں دس سکدے ہو کہ تہانوں کیہو جیہا محسوس ہو رہا اے؟",
            SupportedLanguage.ROMAN_URDU: "Mein samajh gaya. Kya aap mujhe bata saktay hain ke aap ko kaisa mehsoos ho raha hai?"
        }
        
        lang = state.demographics.preferred_language
        response = fallback_responses.get(lang, fallback_responses[SupportedLanguage.ENGLISH])
        
        state.add_conversation_turn(
            role="agent",
            content=response,
            original_language=lang,
            reasoning="Fallback response due to processing error"
        )
        
        return state


# ===================
# STANDALONE TESTING
# ===================

if __name__ == "__main__":
    # Test the interviewer node
    from src.graph.state import PatientState
    from src.config import SupportedLanguage
    
    # Create test state
    state = PatientState(session_id="test_interviewer")
    state.demographics.preferred_language = SupportedLanguage.URDU
    
    # Add initial patient message (like the diabetic farmer)
    state.add_conversation_turn(
        role="patient",
        content="مجھے بہت تھکاوٹ ہو رہی ہے",
        original_language=SupportedLanguage.URDU,
        english_translation="I am feeling very tired"
    )
    
    # Run interviewer
    interviewer = InterviewerNode()
    result_state = interviewer.process(state)
    
    print(f"Phase: {result_state.current_phase}")
    print(f"\nAgent Response:")
    for turn in result_state.conversation_turns:
        if turn.role == "agent":
            print(f"  {turn.content}")
    
    print(f"\nSymptoms Identified: {result_state.get_symptom_summary()}")
    print(f"\nReasoning Trace:")
    for reason in result_state.reasoning_trace:
        print(f"  {reason}")