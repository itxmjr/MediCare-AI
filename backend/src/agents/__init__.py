"""
Agents Module
=============
Prompt engineering and agent persona wrappers.

This module separates prompts from code for:
1. Easy iteration and tuning
2. A/B testing different prompt strategies
3. Language-specific prompt variants
4. Clear separation of concerns

Philosophy:
- Nodes (in graph/nodes/) contain the LOGIC
- Agents (here) contain the PERSONAS and PROMPTS
- This separation allows non-developers to tune prompts

Agent Personas:
- Triage Agent: Emergency assessment specialist
- Interview Agent: Patient intake specialist  
- Translator Agent: Multilingual communication specialist
"""

from src.agents.prompts.triage_prompts import (
    TRIAGE_SYSTEM_PROMPT,
    TRIAGE_ASSESSMENT_PROMPT,
    EMERGENCY_DETECTION_PROMPT,
    get_triage_prompt
)

from src.agents.prompts.interview_prompts import (
    INTERVIEW_SYSTEM_PROMPT,
    GREETING_PROMPT,
    CHIEF_COMPLAINT_PROMPT,
    HPI_PROMPT,
    REVIEW_OF_SYSTEMS_PROMPT,
    CLOSING_PROMPT,
    get_interview_prompt_for_phase
)

from src.agents.prompts.translator_prompts import (
    TRANSLATION_SYSTEM_PROMPT,
    MEDICAL_TRANSLATION_PROMPT,
    get_translation_prompt
)

from src.agents.utils import (
    PromptBuilder,
    format_symptoms_for_prompt,
    format_conversation_for_prompt,
    sanitize_patient_input,
    extract_medical_entities
)

__all__ = [
    # Triage prompts
    "TRIAGE_SYSTEM_PROMPT",
    "TRIAGE_ASSESSMENT_PROMPT", 
    "EMERGENCY_DETECTION_PROMPT",
    "get_triage_prompt",
    
    # Interview prompts
    "INTERVIEW_SYSTEM_PROMPT",
    "GREETING_PROMPT",
    "CHIEF_COMPLAINT_PROMPT",
    "HPI_PROMPT",
    "REVIEW_OF_SYSTEMS_PROMPT",
    "CLOSING_PROMPT",
    "get_interview_prompt_for_phase",
    
    # Translator prompts
    "TRANSLATION_SYSTEM_PROMPT",
    "MEDICAL_TRANSLATION_PROMPT",
    "get_translation_prompt",
    
    # Utilities
    "PromptBuilder",
    "format_symptoms_for_prompt",
    "format_conversation_for_prompt",
    "sanitize_patient_input",
    "extract_medical_entities"
]