"""
Graph Module
============
LangGraph-based workflow engine for the Healthcare AI System.

This module contains:
- state.py: PatientState definition (the conversation memory)
- workflow.py: The master graph connecting all nodes
- nodes/: Individual processing nodes (interviewer, triage, guardian, summarizer)

The graph implements a state machine that:
1. Routes patients through appropriate interview flows
2. Monitors for red flags in parallel
3. Generates structured SOAP summaries
"""

from src.graph.state import (
    PatientState,
    Symptom,
    SOAPNote,
    RedFlag,
    ConversationTurn
)
from src.graph.workflow import create_healthcare_graph, HealthcareGraph

__all__ = [
    "PatientState",
    "Symptom", 
    "SOAPNote",
    "RedFlag",
    "ConversationTurn",
    "create_healthcare_graph",
    "HealthcareGraph"
]