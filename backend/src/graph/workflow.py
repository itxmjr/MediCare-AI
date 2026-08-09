"""
Healthcare Graph Workflow
=========================
The master LangGraph workflow that orchestrates:
1. Patient Interview (Case Study 1)
2. Emergency Triage (Case Study 2)
3. Red Flag Detection (Guardian - runs in parallel)
4. SOAP Summary Generation

This is where the "Agentic" magic happens - the graph routes
patients through different nodes based on their responses.
"""

import logging
from typing import Literal, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import PatientState, InterviewPhase
from src.config import settings, TriageLevel

# Set up logging for explainability
logger = logging.getLogger("healthcare_graph")


class GraphConfig(TypedDict):
    """Configuration passed to the graph."""
    thread_id: str
    language: str


class HealthcareGraph:
    """
    The main healthcare workflow graph.
    
    Flow:
    START -> router -> [interviewer | triage | guardian] -> summarizer -> END
    
    The guardian node runs in parallel to detect red flags.
    """
    
    def __init__(self):
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create the graph with PatientState
        workflow = StateGraph(PatientState)
        
        # ===================
        # ADD NODES
        # ===================
        workflow.add_node("router", self._router_node)
        workflow.add_node("interviewer", self._interviewer_node)
        workflow.add_node("triage", self._triage_node)
        workflow.add_node("guardian", self._guardian_node)
        workflow.add_node("summarizer", self._summarizer_node)
        workflow.add_node("emergency_handler", self._emergency_handler_node)
        
        # ===================
        # SET ENTRY POINT
        # ===================
        workflow.add_edge(START, "guardian")  # Always check for emergencies first
        
        # ===================
        # CONDITIONAL ROUTING
        # ===================
        
        # After guardian checks, route based on emergency status
        workflow.add_conditional_edges(
            "guardian",
            self._route_after_guardian,
            {
                "emergency": "emergency_handler",
                "triage": "triage",
                "interview": "interviewer",
                "summarize": "summarizer"
            }
        )
        
        # After interview, decide next step
        workflow.add_conditional_edges(
            "interviewer",
            self._route_after_interview,
            {
                "continue": END,
                "summarize": "summarizer",
                "emergency": "emergency_handler"
            }
        )
        
        # After triage, go to summarizer or emergency
        workflow.add_conditional_edges(
            "triage",
            self._route_after_triage,
            {
                "emergency": "emergency_handler",
                "summarize": "summarizer"
            }
        )
        
        # Emergency handler always ends (after giving immediate action)
        workflow.add_edge("emergency_handler", "summarizer")
        
        # Summarizer ends the flow
        workflow.add_edge("summarizer", END)
        
        return workflow
    
    # ===================
    # NODE IMPLEMENTATIONS
    # ===================
    
    def _router_node(self, state: PatientState) -> PatientState:
        """
        Initial routing based on conversation state.
        Determines which specialized node should handle the patient.
        """
        state.add_reasoning("Router: Analyzing conversation state...")
        
        # Check if this is a new conversation
        if state.question_count == 0:
            state.next_node = "interview"
            state.add_reasoning("Router: New patient, starting interview")
        elif state.is_emergency:
            state.next_node = "emergency"
            state.add_reasoning(f"Router: Emergency detected - {state.escalation_reason}")
        elif state.should_end_interview():
            state.next_node = "summarize"
            state.add_reasoning("Router: Interview complete, generating summary")
        else:
            state.next_node = "interview"
            state.add_reasoning("Router: Continuing interview")
        
        return state
    
    def _guardian_node(self, state: PatientState) -> PatientState:
        """
        The Safety Layer - runs on every turn to detect red flags.
        This is the "parallel watcher" that can interrupt the flow.
        """
        from src.graph.nodes.guardian import GuardianNode
        
        state.add_reasoning("Guardian: Scanning for red flags...")
        
        guardian = GuardianNode()
        state = guardian.process(state)
        
        if state.is_emergency:
            state.add_reasoning(
                f"Guardian: ⚠️ EMERGENCY DETECTED - {state.escalation_reason}"
            )
        else:
            state.add_reasoning("Guardian: No immediate red flags detected")
        
        return state
    
    def _interviewer_node(self, state: PatientState) -> PatientState:
        """
        Conducts the adaptive patient interview (Case Study 1).
        Asks follow-up questions based on symptoms and state.
        """
        from src.graph.nodes.interviewer import InterviewerNode
        
        state.add_reasoning(
            f"Interviewer: Phase={state.current_phase.value}, "
            f"Questions asked={state.question_count}"
        )
        
        interviewer = InterviewerNode()
        state = interviewer.process(state)
        
        state.question_count += 1
        state.last_updated = datetime.now()
        
        return state
    
    def _triage_node(self, state: PatientState) -> PatientState:
        """
        WHO Emergency Triage Assessment (Case Study 2).
        Assigns triage level and action plan.
        """
        from src.graph.nodes.triage import TriageNode
        
        state.add_reasoning(
            f"Triage: Assessing patient with symptoms: {state.get_symptom_summary()}"
        )
        
        triage = TriageNode()
        state = triage.process(state)
        
        state.add_reasoning(
            f"Triage: Assigned level {state.current_triage_level.value}"
        )
        
        return state
    
    def _emergency_handler_node(self, state: PatientState) -> PatientState:
        """
        Handles confirmed emergencies.
        Generates immediate action instructions.
        """
        state.add_reasoning("Emergency Handler: Processing critical case...")
        
        # Ensure we have an emergency action
        if not state.emergency_action:
            state.emergency_action = "Call 1122 immediately. Seek emergency medical care."
        
        # Force immediate triage level
        state.current_triage_level = TriageLevel.IMMEDIATE
        
        # Mark interview as complete - no more questions needed
        state.interview_complete = True
        state.current_phase = InterviewPhase.EMERGENCY_ESCALATION
        
        state.add_reasoning(
            f"Emergency Handler: Action = {state.emergency_action}"
        )
        
        return state
    
    def _summarizer_node(self, state: PatientState) -> PatientState:
        """
        Generates the final SOAP note for doctors.
        """
        from src.graph.nodes.summarizer import SummarizerNode
        
        state.add_reasoning("Summarizer: Generating SOAP note...")
        
        summarizer = SummarizerNode()
        state = summarizer.process(state)
        
        state.interview_complete = True
        state.current_phase = InterviewPhase.COMPLETED
        
        state.add_reasoning("Summarizer: SOAP note generated successfully")
        
        return state
    
    # ===================
    # ROUTING FUNCTIONS
    # ===================
    
    def _route_after_guardian(
        self, state: PatientState
    ) -> Literal["emergency", "triage", "interview", "summarize"]:
        """Decide where to go after guardian check."""
        
        if state.is_emergency:
            return "emergency"
        
        # If patient explicitly mentions emergency keywords, go to triage
        emergency_keywords = ["emergency", "urgent", "severe", "worst pain"]
        last_message = ""
        if state.conversation_turns:
            last_turn = state.conversation_turns[-1]
            if last_turn.role == "patient":
                last_message = (
                    last_turn.english_translation or last_turn.content
                ).lower()
        
        if any(kw in last_message for kw in emergency_keywords):
            return "triage"
        
        # If interview is complete, summarize
        if state.should_end_interview():
            return "summarize"
        
        # Default: continue interview
        return "interview"
    
    def _route_after_interview(
        self, state: PatientState
    ) -> Literal["continue", "summarize", "emergency"]:
        """Decide where to go after interview turn."""
        
        if state.is_emergency:
            return "emergency"
        
        if state.should_end_interview() or state.interview_complete:
            return "summarize"
        
        return "continue"
    
    def _route_after_triage(
        self, state: PatientState
    ) -> Literal["emergency", "summarize"]:
        """Decide where to go after triage assessment."""
        
        if state.current_triage_level == TriageLevel.IMMEDIATE:
            return "emergency"
        
        return "summarize"
    
    # ===================
    # PUBLIC INTERFACE
    # ===================
    
    async def process_message(
        self,
        session_id: str,
        message: str,
        language: str = "ur"
    ) -> dict:
        """
        Process a patient message through the graph.
        
        Args:
            session_id: Unique session ID for the patient
            message: The patient's message
            language: Language code (en, ur, pa, etc.)
        
        Returns:
            dict with 'response', 'triage_level', 'is_emergency', etc.
        """
        from src.config import SupportedLanguage
        
        # Create or retrieve state
        config = {"configurable": {"thread_id": session_id}}
        
        # Get existing state or create new
        try:
            current_state = self.app.get_state(config)
            if current_state.values:
                state = PatientState(**current_state.values)
            else:
                state = PatientState(
                    session_id=session_id,
                    demographics={"preferred_language": language}
                )
        except Exception:
            state = PatientState(
                session_id=session_id
            )
            state.demographics.preferred_language = SupportedLanguage(language)
        
        # Add patient message to conversation
        state.add_conversation_turn(
            role="patient",
            content=message,
            original_language=SupportedLanguage(language)
        )
        
        # Run through graph
        result = await self.app.ainvoke(
            state.model_dump(),
            config
        )
        
        # Extract response
        response_state = PatientState(**result)
        
        # Get the last agent message
        agent_response = ""
        if response_state.conversation_turns:
            for turn in reversed(response_state.conversation_turns):
                if turn.role == "agent":
                    agent_response = turn.content
                    break
        
        return {
            "response": agent_response,
            "triage_level": response_state.current_triage_level.value,
            "is_emergency": response_state.is_emergency,
            "emergency_action": response_state.emergency_action,
            "interview_complete": response_state.interview_complete,
            "soap_note": response_state.soap_note.to_markdown() if response_state.soap_note else None,
            "reasoning_trace": response_state.reasoning_trace,
            "symptoms": [s.model_dump() for s in response_state.symptoms],
            "red_flags": [rf.model_dump() for rf in response_state.red_flags]
        }
    
    def get_session_state(self, session_id: str) -> dict:
        """Get the current state of a session."""
        config = {"configurable": {"thread_id": session_id}}
        try:
            state = self.app.get_state(config)
            return state.values if state.values else {}
        except Exception:
            return {}


def create_healthcare_graph() -> HealthcareGraph:
    """Factory function to create the healthcare graph."""
    return HealthcareGraph()


# ===================
# CONVENIENCE RUNNER
# ===================

async def run_interview(
    patient_messages: list[str],
    session_id: str = "test_session",
    language: str = "ur"
) -> dict:
    """
    Run a complete interview with a list of patient messages.
    Useful for testing.
    """
    graph = create_healthcare_graph()
    
    final_result = None
    for message in patient_messages:
        result = await graph.process_message(session_id, message, language)
        final_result = result
        
        print(f"\n{'='*50}")
        print(f"Patient: {message}")
        print(f"Agent: {result['response']}")
        print(f"Triage: {result['triage_level']}")
        
        if result['is_emergency']:
            print(f"⚠️ EMERGENCY: {result['emergency_action']}")
        
        if result['interview_complete']:
            print("\n" + "="*50)
            print("INTERVIEW COMPLETE")
            if result['soap_note']:
                print(result['soap_note'])
            break
    
    return final_result