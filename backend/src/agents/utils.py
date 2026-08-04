"""
Agent Utilities
===============
Utility functions for agent operations including:
- Prompt building
- Text formatting
- Input sanitization
- Entity extraction
- Response parsing

These utilities are shared across all agents.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from src.config import SupportedLanguage, settings
from src.graph.state import PatientState, Symptom, ConversationTurn

logger = logging.getLogger("agent_utils")


# =============================================================================
# PROMPT BUILDER CLASS
# =============================================================================

class PromptBuilder:
    """
    Utility class for building and managing prompts.
    
    Features:
    - Template management
    - Variable substitution
    - Context injection
    - Length management
    """
    
    def __init__(self, max_context_length: int = 4000):
        """
        Initialize the prompt builder.
        
        Args:
            max_context_length: Maximum characters for context sections
        """
        self.max_context_length = max_context_length
        self._templates: Dict[str, str] = {}
    
    def register_template(self, name: str, template: str) -> None:
        """Register a prompt template."""
        self._templates[name] = template
    
    def build(
        self,
        template_name: str,
        variables: Dict[str, Any],
        truncate_fields: Optional[List[str]] = None
    ) -> str:
        """
        Build a prompt from a template.
        
        Args:
            template_name: Name of the registered template
            variables: Variables to substitute
            truncate_fields: Fields to truncate if too long
            
        Returns:
            Formatted prompt string
        """
        if template_name not in self._templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self._templates[template_name]
        
        # Truncate specified fields
        if truncate_fields:
            for field in truncate_fields:
                if field in variables and isinstance(variables[field], str):
                    variables[field] = self._truncate(
                        variables[field],
                        self.max_context_length // len(truncate_fields)
                    )
        
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing variable in template: {e}")
            raise
    
    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max length, preserving complete sentences."""
        if len(text) <= max_length:
            return text
        
        truncated = text[:max_length]
        # Try to end at a sentence boundary
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.7:
            truncated = truncated[:last_period + 1]
        
        return truncated + "..."
    
    @staticmethod
    def build_from_string(template: str, variables: Dict[str, Any]) -> str:
        """Build prompt directly from a template string."""
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing variable: {e}")
            # Return template with missing variables noted
            return template


# =============================================================================
# TEXT FORMATTING FUNCTIONS
# =============================================================================

def format_symptoms_for_prompt(symptoms: List[Symptom]) -> str:
    """
    Format symptoms for inclusion in prompts.
    
    Args:
        symptoms: List of Symptom objects
        
    Returns:
        Formatted string representation
    """
    if not symptoms:
        return "No symptoms recorded yet."
    
    formatted_parts = []
    for i, symptom in enumerate(symptoms, 1):
        parts = [f"{i}. **{symptom.name}**"]
        
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
            parts.append(f"   ({'; '.join(details)})")
        
        formatted_parts.append("".join(parts))
    
    return "\n".join(formatted_parts)


def format_conversation_for_prompt(
    conversation_turns: List[ConversationTurn],
    max_turns: int = 10,
    include_reasoning: bool = False
) -> str:
    """
    Format conversation history for prompts.
    
    Args:
        conversation_turns: List of conversation turns
        max_turns: Maximum number of turns to include
        include_reasoning: Whether to include agent reasoning
        
    Returns:
        Formatted conversation string
    """
    if not conversation_turns:
        return "No conversation history."
    
    # Take last N turns
    recent_turns = conversation_turns[-max_turns:]
    
    formatted_parts = []
    for turn in recent_turns:
        role = "Patient" if turn.role == "patient" else "AI"
        content = turn.english_translation or turn.content
        
        line = f"{role}: {content}"
        
        if include_reasoning and turn.reasoning and turn.role == "agent":
            line += f"\n   [Reasoning: {turn.reasoning}]"
        
        formatted_parts.append(line)
    
    return "\n".join(formatted_parts)


def format_red_flags_for_prompt(state: PatientState) -> str:
    """
    Format red flags for prompts.
    
    Args:
        state: PatientState with red flags
        
    Returns:
        Formatted red flags string
    """
    if not state.red_flags:
        return "No red flags detected."
    
    parts = []
    for rf in state.red_flags:
        parts.append(
            f"⚠️ **{rf.pattern_name.upper()}** (Confidence: {rf.confidence:.0%})\n"
            f"   Matched: {', '.join(rf.matched_symptoms)}\n"
            f"   Action: {rf.recommended_action}"
        )
    
    return "\n".join(parts)


def format_medical_history_for_prompt(state: PatientState) -> str:
    """
    Format medical history for prompts.
    
    Args:
        state: PatientState with medical history
        
    Returns:
        Formatted medical history string
    """
    history = state.medical_history
    parts = []
    
    if history.past_conditions:
        parts.append(f"Past Conditions: {', '.join(history.past_conditions)}")
    if history.current_medications:
        parts.append(f"Current Medications: {', '.join(history.current_medications)}")
    if history.allergies:
        parts.append(f"Allergies: {', '.join(history.allergies)}")
    if history.surgeries:
        parts.append(f"Previous Surgeries: {', '.join(history.surgeries)}")
    if history.family_history:
        parts.append(f"Family History: {', '.join(history.family_history)}")
    
    if not parts:
        return "No medical history recorded."
    
    return "\n".join(parts)


# =============================================================================
# INPUT SANITIZATION
# =============================================================================

def sanitize_patient_input(text: str) -> str:
    """
    Sanitize patient input text.
    
    - Removes potential injection attempts
    - Normalizes whitespace
    - Removes control characters
    
    Args:
        text: Raw patient input
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove control characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove potential prompt injection patterns
    injection_patterns = [
        r'ignore previous instructions',
        r'ignore all instructions',
        r'disregard.*instructions',
        r'you are now',
        r'act as',
        r'pretend to be',
        r'system prompt',
        r'\[INST\]',
        r'\[/INST\]',
        r'<\|.*\|>',
    ]
    
    text_lower = text.lower()
    for pattern in injection_patterns:
        if re.search(pattern, text_lower):
            logger.warning(f"Potential injection attempt detected: {pattern}")
            # Don't remove, but log for monitoring
    
    # Limit length
    max_length = 2000
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text


def validate_patient_response(text: str) -> Tuple[bool, str]:
    """
    Validate that patient response is usable.
    
    Args:
        text: Patient response text
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Empty response"
    
    if len(text.strip()) < 2:
        return False, "Response too short"
    
    if len(text) > 5000:
        return False, "Response too long"
    
    # Check for gibberish (very basic check)
    words = text.split()
    if len(words) > 5:
        avg_word_length = sum(len(w) for w in words) / len(words)
        if avg_word_length > 20:
            return False, "Response appears to be invalid"
    
    return True, ""


# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

def extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract medical entities from patient text.
    
    This is a simple pattern-based extractor. For production,
    consider using a medical NER model.
    
    Args:
        text: Patient text
        
    Returns:
        Dict with 'symptoms', 'durations', 'severities', 'body_parts'
    """
    text_lower = text.lower()
    
    # Symptom patterns
    symptom_patterns = [
        r'(chest pain|headache|fever|fatigue|tired|dizzy|nausea|vomiting)',
        r'(cough|sore throat|shortness of breath|breathing problem)',
        r'(abdominal pain|stomach ache|diarrhea|constipation)',
        r'(back pain|joint pain|muscle pain|weakness)',
        r'(thirsty|excessive thirst|frequent urination)',
        r'(weight loss|weight gain|appetite loss)',
        r'(sweating|chills|night sweats)',
        r'(rash|itching|swelling)',
        # Urdu symptom patterns (transliterated)
        r'(dard|bukhar|thakan|chakkar|ulti|khansi)'
    ]
    
    symptoms = []
    for pattern in symptom_patterns:
        matches = re.findall(pattern, text_lower)
        symptoms.extend(matches)
    
    # Duration patterns
    duration_patterns = [
        r'(\d+\s*(?:day|days|week|weeks|month|months|year|years|hour|hours))',
        r'(since\s+(?:yesterday|last\s+week|last\s+month))',
        r'(for\s+(?:a\s+)?(?:few|couple|several)\s+(?:days|weeks|months))',
        r'(\d+[-\s]?\d*\s*(?:din|hafte|mahine))',  # Urdu
    ]
    
    durations = []
    for pattern in duration_patterns:
        matches = re.findall(pattern, text_lower)
        durations.extend(matches)
    
    # Severity patterns
    severity_patterns = [
        r'(mild|moderate|severe|extreme|terrible|worst)',
        r'(little|slightly|very|really|extremely)',
        r'(\d+\s*(?:out of|\/)\s*10)',
        r'(halka|zyada|bahut|shadeed)',  # Urdu
    ]
    
    severities = []
    for pattern in severity_patterns:
        matches = re.findall(pattern, text_lower)
        severities.extend(matches)
    
    # Body part patterns
    body_part_patterns = [
        r'(head|chest|stomach|back|arm|leg|neck|throat|eye|ear)',
        r'(left|right)\s+(arm|leg|side|eye|ear)',
        r'(sar|seena|pet|kamar|bazu|tang|gala)',  # Urdu
    ]
    
    body_parts = []
    for pattern in body_part_patterns:
        matches = re.findall(pattern, text_lower)
        if isinstance(matches[0], tuple) if matches else False:
            body_parts.extend([' '.join(m) for m in matches])
        else:
            body_parts.extend(matches)
    
    return {
        "symptoms": list(set(symptoms)),
        "durations": list(set(durations)),
        "severities": list(set(severities)),
        "body_parts": list(set(body_parts))
    }


def extract_numbers(text: str) -> List[Dict[str, Any]]:
    """
    Extract numbers from text (for vitals, scales, etc.).
    
    Args:
        text: Input text
        
    Returns:
        List of dicts with 'value', 'unit', 'context'
    """
    numbers = []
    
    # Blood pressure pattern
    bp_pattern = r'(\d{2,3})\s*[/\\]\s*(\d{2,3})'
    bp_matches = re.findall(bp_pattern, text)
    for match in bp_matches:
        numbers.append({
            "value": f"{match[0]}/{match[1]}",
            "unit": "mmHg",
            "context": "blood_pressure"
        })
    
    # Temperature pattern
    temp_pattern = r'(\d{2,3}(?:\.\d)?)\s*(?:°|degrees?)?\s*(?:F|C|fahrenheit|celsius)?'
    temp_matches = re.findall(temp_pattern, text, re.IGNORECASE)
    for match in temp_matches:
        val = float(match)
        if 95 <= val <= 108 or 35 <= val <= 42:  # Valid temp range
            numbers.append({
                "value": val,
                "unit": "°F" if val > 50 else "°C",
                "context": "temperature"
            })
    
    # Scale pattern (1-10)
    scale_pattern = r'(\d{1,2})\s*(?:out of|\/)\s*10'
    scale_matches = re.findall(scale_pattern, text)
    for match in scale_matches:
        numbers.append({
            "value": int(match),
            "unit": "/10",
            "context": "pain_scale"
        })
    
    # Blood glucose pattern
    glucose_pattern = r'(\d{2,3})\s*(?:mg\s*/?\s*dl|mg/dl)'
    glucose_matches = re.findall(glucose_pattern, text, re.IGNORECASE)
    for match in glucose_matches:
        numbers.append({
            "value": int(match),
            "unit": "mg/dL",
            "context": "blood_glucose"
        })
    
    return numbers


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_json_response(response: str) -> Optional[Dict]:
    """
    Parse JSON from LLM response, handling common issues.
    
    Args:
        response: LLM response that may contain JSON
        
    Returns:
        Parsed dict or None if parsing fails
    """
    # Try to find JSON in the response
    json_patterns = [
        r'```json\s*([\s\S]*?)\s*```',  # Markdown code block
        r'```\s*([\s\S]*?)\s*```',       # Generic code block
        r'\{[\s\S]*\}',                   # Raw JSON object
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, response)
        if match:
            json_str = match.group(1) if '```' in pattern else match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    # Try parsing the whole response
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON from response: {response[:100]}...")
        return None


def extract_structured_response(
    response: str,
    expected_fields: List[str]
) -> Dict[str, Any]:
    """
    Extract structured data from a response even if not valid JSON.
    
    Args:
        response: LLM response
        expected_fields: Fields we expect to find
        
    Returns:
        Dict with extracted fields
    """
    result = {}
    
    # First try JSON parsing
    json_data = parse_json_response(response)
    if json_data:
        return json_data
    
    # Fall back to pattern matching
    for field in expected_fields:
        # Try various patterns
        patterns = [
            rf'{field}\s*[:=]\s*"([^"]*)"',
            rf'{field}\s*[:=]\s*\'([^\']*)\'',
            rf'{field}\s*[:=]\s*([^\n,}}]+)',
            rf'"{field}"\s*:\s*"([^"]*)"',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result[field] = match.group(1).strip()
                break
    
    return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_interview_progress(state: PatientState) -> Dict[str, Any]:
    """
    Calculate interview progress metrics.
    
    Args:
        state: Current patient state
        
    Returns:
        Dict with progress metrics
    """
    max_questions = settings.MAX_INTERVIEW_QUESTIONS
    min_questions = settings.MIN_INTERVIEW_QUESTIONS
    
    progress = {
        "questions_asked": state.question_count,
        "max_questions": max_questions,
        "min_questions": min_questions,
        "percentage": min(100, (state.question_count / max_questions) * 100),
        "can_complete": state.question_count >= min_questions,
        "should_complete": state.question_count >= max_questions,
        "symptoms_count": len(state.symptoms),
        "red_flags_count": len(state.red_flags),
        "current_phase": state.current_phase.value
    }
    
    return progress


def get_time_since_start(state: PatientState) -> str:
    """
    Get human-readable time since interview started.
    
    Args:
        state: Patient state
        
    Returns:
        Formatted duration string
    """
    duration = datetime.now() - state.created_at
    minutes = duration.seconds // 60
    seconds = duration.seconds % 60
    
    if minutes == 0:
        return f"{seconds} seconds"
    elif minutes == 1:
        return "1 minute"
    else:
        return f"{minutes} minutes"


def should_ask_about(
    topic: str,
    state: PatientState
) -> bool:
    """
    Determine if we should ask about a topic.
    
    Args:
        topic: Topic to potentially ask about
        state: Current state
        
    Returns:
        True if we should ask about this topic
    """
    # Check if already covered in symptoms
    topic_lower = topic.lower()
    for symptom in state.symptoms:
        if topic_lower in symptom.name.lower():
            return False
    
    # Check if already asked in conversation
    for turn in state.conversation_turns:
        if turn.role == "agent":
            content = (turn.english_translation or turn.content).lower()
            if topic_lower in content:
                return False
    
    return True