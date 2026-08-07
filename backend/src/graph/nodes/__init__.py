"""
Graph Nodes Module
==================
Processing nodes for the Healthcare LangGraph workflow.

Each node is a specialized processor that:
1. Takes PatientState as input
2. Performs its specific task
3. Updates and returns PatientState

Nodes:
- GuardianNode: Red flag detection (safety layer)
- InterviewerNode: Adaptive patient interview (Case Study 1)
- TriageNode: WHO emergency triage (Case Study 2)
- SummarizerNode: SOAP note generation
"""

from src.graph.nodes.guardian import GuardianNode
from src.graph.nodes.interviewer import InterviewerNode
from src.graph.nodes.triage import TriageNode
from src.graph.nodes.summarizer import SummarizerNode

__all__ = [
    "GuardianNode",
    "InterviewerNode",
    "TriageNode",
    "SummarizerNode"
]