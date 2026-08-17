"""
Tests Module
============
Test suite for the Healthcare AI System.

Test Categories:
1. Unit Tests - Individual component testing
2. Integration Tests - Component interaction testing
3. End-to-End Tests - Full workflow testing
4. Performance Tests - Load and latency testing

Running Tests:
    pytest src/tests/ -v
    pytest src/tests/ -v --cov=src
    pytest src/tests/test_agents.py -v -k "test_guardian"

Test Fixtures:
- mock_patient_state: Pre-configured patient state
- mock_symptoms: Sample symptom data
- mock_conversation: Sample conversation history
"""

import pytest
import asyncio
from typing import Generator, Dict, Any, List
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

# Import test utilities
from src.config import settings, SupportedLanguage, TriageLevel
from src.graph.state import PatientState, Symptom, RedFlag, ConversationTurn, InterviewPhase


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Override settings for testing
TEST_CONFIG = {
    "GROQ_API_KEY": "test-key-for-testing",
    "DEBUG_MODE": True,
    "LOG_LEVEL": "DEBUG",
    "MAX_INTERVIEW_QUESTIONS": 10,
    "MIN_INTERVIEW_QUESTIONS": 3
}


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_settings():
    """Override settings for testing."""
    original_values = {}
    for key, value in TEST_CONFIG.items():
        original_values[key] = getattr(settings, key, None)
        setattr(settings, key, value)
    
    yield settings
    
    # Restore original values
    for key, value in original_values.items():
        if value is not None:
            setattr(settings, key, value)


@pytest.fixture
def mock_patient_state() -> PatientState:
    """Create a mock patient state for testing."""
    state = PatientState(
        session_id="test_session_123",
        channel="api"
    )
    
    # Set demographics
    state.demographics.preferred_language = SupportedLanguage.URDU
    state.demographics.age = 45
    state.demographics.gender = "male"
    
    return state


@pytest.fixture
def mock_patient_with_symptoms() -> PatientState:
    """Create a patient state with symptoms."""
    state = PatientState(session_id="test_symptoms_123")
    state.demographics.preferred_language = SupportedLanguage.URDU
    
    # Add symptoms
    state.add_symptom(Symptom(
        name="fatigue",
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
    
    # Add conversation
    state.add_conversation_turn(
        role="patient",
        content="مجھے بہت تھکاوٹ ہو رہی ہے",
        original_language=SupportedLanguage.URDU,
        english_translation="I am feeling very tired"
    )
    
    return state


@pytest.fixture
def mock_emergency_state() -> PatientState:
    """Create a patient state with emergency symptoms."""
    state = PatientState(session_id="test_emergency_123")
    state.demographics.preferred_language = SupportedLanguage.URDU
    
    # Add emergency symptoms
    state.add_symptom(Symptom(
        name="chest pain",
        duration="1 hour",
        severity="severe",
        onset="sudden"
    ))
    state.add_symptom(Symptom(
        name="pain radiating to left arm"
    ))
    state.add_symptom(Symptom(
        name="sweating",
        severity="profuse"
    ))
    
    state.add_conversation_turn(
        role="patient",
        content="I have severe chest pain and it's going to my arm",
        original_language=SupportedLanguage.ENGLISH
    )
    
    return state


@pytest.fixture
def mock_symptoms() -> List[Symptom]:
    """Create mock symptom list."""
    return [
        Symptom(name="headache", duration="2 days", severity="moderate"),
        Symptom(name="fever", severity="mild"),
        Symptom(name="fatigue", duration="1 week")
    ]


@pytest.fixture
def mock_conversation() -> List[Dict[str, Any]]:
    """Create mock conversation history."""
    return [
        {
            "role": "agent",
            "content": "How can I help you today?",
            "phase": "greeting"
        },
        {
            "role": "patient",
            "content": "I have a headache",
            "phase": "chief_complaint"
        },
        {
            "role": "agent",
            "content": "How long have you had this headache?",
            "phase": "history_present_illness"
        },
        {
            "role": "patient",
            "content": "About 2 days",
            "phase": "history_present_illness"
        }
    ]


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return MagicMock(
        content="This is a mock LLM response",
        response_metadata={"model": "llama-3.3-70b-versatile"}
    )


@pytest.fixture
def mock_groq_client():
    """Create a mock Groq client."""
    with patch('groq.Groq') as mock:
        client = MagicMock()
        mock.return_value = client
        
        # Mock chat completions
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="Mock response")
            )]
        )
        
        # Mock audio
        client.audio.transcriptions.create.return_value = MagicMock(
            text="Transcribed text",
            language="ur"
        )
        
        yield client


# =============================================================================
# ASYNC FIXTURES
# =============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_mock_graph():
    """Create mock healthcare graph for async testing."""
    from src.graph.workflow import HealthcareGraph
    
    with patch.object(HealthcareGraph, '__init__', lambda x: None):
        graph = HealthcareGraph()
        graph.process_message = AsyncMock(return_value={
            "response": "Test response",
            "triage_level": "STANDARD",
            "is_emergency": False,
            "interview_complete": False
        })
        yield graph


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_test_state(
    symptoms: List[str] = None,
    language: SupportedLanguage = SupportedLanguage.URDU,
    phase: InterviewPhase = InterviewPhase.GREETING
) -> PatientState:
    """Helper to create test patient state."""
    state = PatientState(session_id=f"test_{datetime.now().timestamp()}")
    state.demographics.preferred_language = language
    state.current_phase = phase
    
    if symptoms:
        for symptom_name in symptoms:
            state.add_symptom(Symptom(name=symptom_name))
    
    return state


def assert_valid_soap_note(soap_note) -> None:
    """Assert that a SOAP note has required fields."""
    assert soap_note is not None
    assert soap_note.chief_complaint
    assert soap_note.primary_assessment
    assert soap_note.triage_level is not None
    assert 0 <= soap_note.confidence_score <= 1


def assert_valid_triage_result(result: Dict[str, Any]) -> None:
    """Assert that a triage result is valid."""
    assert "triage_level" in result
    assert result["triage_level"] in [level.value for level in TriageLevel]
    assert "is_emergency" in result
    assert isinstance(result["is_emergency"], bool)


# =============================================================================
# TEST MARKERS
# =============================================================================

# Markers for categorizing tests
pytest_plugins = []

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring API access"
    )


# =============================================================================
# SAMPLE TEST DATA
# =============================================================================

SAMPLE_PATIENT_MESSAGES = {
    "diabetes_symptoms_urdu": "مجھے بہت تھکاوٹ ہے اور بہت پیاس لگتی ہے",
    "diabetes_symptoms_english": "I'm very tired and always thirsty",
    "chest_pain_urdu": "مجھے سینے میں شدید درد ہے اور بازو میں بھی درد ہے",
    "chest_pain_english": "I have severe chest pain and my arm hurts too",
    "headache_urdu": "مجھے دو دن سے سر میں درد ہے",
    "headache_english": "I've had a headache for two days",
    "fever_urdu": "مجھے بخار ہے اور سردی لگ رہی ہے",
    "fever_english": "I have a fever and chills"
}

SAMPLE_RED_FLAG_TRIGGERS = [
    {
        "message": "Severe chest pain radiating to my arm, sweating",
        "expected_pattern": "cardiac_emergency",
        "expected_level": TriageLevel.IMMEDIATE
    },
    {
        "message": "Worst headache of my life, came on suddenly",
        "expected_pattern": "subarachnoid_hemorrhage",
        "expected_level": TriageLevel.IMMEDIATE
    },
    {
        "message": "Very thirsty, urinating frequently, tired for months",
        "expected_pattern": "diabetic_emergency",
        "expected_level": TriageLevel.VERY_URGENT
    }
]


# =============================================================================
# TEST UTILITIES
# =============================================================================

class TestTimer:
    """Context manager for timing tests."""
    
    def __init__(self, name: str = "Test"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"\n{self.name} completed in {duration:.3f}s")
    
    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


def generate_random_session_id() -> str:
    """Generate a random session ID for testing."""
    import uuid
    return f"test_session_{uuid.uuid4().hex[:8]}"