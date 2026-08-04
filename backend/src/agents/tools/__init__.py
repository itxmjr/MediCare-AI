"""
Tools Module
============
Capabilities that agents can use during processing.

Tools provide:
1. RAG Search - Retrieve clinical guidelines
2. Calculator - Medical calculations (BMI, dosages)
3. Time Utils - Symptom duration calculations
4. Validators - Input validation

These tools ground the AI in factual information and prevent
hallucinations in critical calculations.

Tool Design Principles:
- Tools are stateless functions
- Tools handle their own errors gracefully
- Tools return structured data
- Tools log their operations for explainability
"""

from src.agents.tools.rag_search import (
    search_guidelines,
    search_emergency_protocols,
    search_drug_interactions,
    get_symptom_guidance,
    RAGSearchTool
)

from src.agents.tools.calculator import (
    calculate_bmi,
    calculate_bsa,
    interpret_blood_pressure,
    interpret_blood_glucose,
    interpret_temperature,
    assess_dehydration_risk,
    MedicalCalculator
)

from src.agents.tools.time_utils import (
    parse_duration,
    calculate_symptom_urgency,
    is_acute_onset,
    format_duration,
    TimeParser
)

__all__ = [
    # RAG Search
    "search_guidelines",
    "search_emergency_protocols", 
    "search_drug_interactions",
    "get_symptom_guidance",
    "RAGSearchTool",
    
    # Calculator
    "calculate_bmi",
    "calculate_bsa",
    "interpret_blood_pressure",
    "interpret_blood_glucose",
    "interpret_temperature",
    "assess_dehydration_risk",
    "MedicalCalculator",
    
    # Time Utils
    "parse_duration",
    "calculate_symptom_urgency",
    "is_acute_onset",
    "format_duration",
    "TimeParser"
]


# =============================================================================
# TOOL REGISTRY
# =============================================================================

class ToolRegistry:
    """
    Registry for all available tools.
    
    This allows dynamic tool discovery and provides
    a unified interface for tool execution.
    """
    
    _tools: dict = {}
    
    @classmethod
    def register(cls, name: str, func: callable, description: str) -> None:
        """Register a tool."""
        cls._tools[name] = {
            "function": func,
            "description": description
        }
    
    @classmethod
    def get(cls, name: str) -> callable:
        """Get a tool by name."""
        if name not in cls._tools:
            raise ValueError(f"Tool '{name}' not found")
        return cls._tools[name]["function"]
    
    @classmethod
    def list_tools(cls) -> list:
        """List all registered tools."""
        return [
            {"name": name, "description": info["description"]}
            for name, info in cls._tools.items()
        ]
    
    @classmethod
    def execute(cls, name: str, **kwargs) -> any:
        """Execute a tool by name."""
        func = cls.get(name)
        return func(**kwargs)


# =============================================================================
# REGISTER DEFAULT TOOLS
# =============================================================================

def _register_default_tools():
    """Register all default tools."""
    
    # RAG Tools
    ToolRegistry.register(
        "search_guidelines",
        search_guidelines,
        "Search clinical guidelines for relevant information"
    )
    ToolRegistry.register(
        "search_emergency",
        search_emergency_protocols,
        "Search emergency protocols for critical situations"
    )
    
    # Calculator Tools
    ToolRegistry.register(
        "calculate_bmi",
        calculate_bmi,
        "Calculate Body Mass Index from height and weight"
    )
    ToolRegistry.register(
        "interpret_bp",
        interpret_blood_pressure,
        "Interpret blood pressure readings"
    )
    ToolRegistry.register(
        "interpret_glucose",
        interpret_blood_glucose,
        "Interpret blood glucose levels"
    )
    
    # Time Tools
    ToolRegistry.register(
        "parse_duration",
        parse_duration,
        "Parse symptom duration from text"
    )
    ToolRegistry.register(
        "check_acute_onset",
        is_acute_onset,
        "Check if symptom onset is acute/sudden"
    )


# Initialize default tools
try:
    _register_default_tools()
except ImportError:
    # Tools not yet implemented, will be registered when available
    pass


# =============================================================================
# LANGCHAIN TOOL WRAPPERS
# =============================================================================

def create_langchain_tools() -> list:
    """
    Create LangChain-compatible tool objects.
    
    Returns:
        List of LangChain Tool objects
    """
    from langchain.tools import Tool
    
    tools = []
    
    # RAG Search Tool
    tools.append(Tool(
        name="search_medical_guidelines",
        func=lambda q: search_guidelines(q, k=3),
        description="""Search clinical guidelines and protocols. 
        Use this when you need medical reference information.
        Input: search query about symptoms or conditions
        Output: relevant guideline excerpts"""
    ))
    
    # Emergency Protocol Tool
    tools.append(Tool(
        name="check_emergency_protocol",
        func=lambda q: search_emergency_protocols(q),
        description="""Check emergency protocols for urgent symptoms.
        Use this when patient describes potentially serious symptoms.
        Input: symptom description
        Output: relevant emergency protocols and actions"""
    ))
    
    # BMI Calculator
    tools.append(Tool(
        name="calculate_bmi",
        func=lambda x: calculate_bmi(*map(float, x.split(','))),
        description="""Calculate BMI from height and weight.
        Input: 'height_cm,weight_kg' (e.g., '170,75')
        Output: BMI value and interpretation"""
    ))
    
    # Blood Pressure Interpreter
    tools.append(Tool(
        name="interpret_blood_pressure",
        func=lambda x: interpret_blood_pressure(*map(int, x.split('/'))),
        description="""Interpret blood pressure reading.
        Input: 'systolic/diastolic' (e.g., '140/90')
        Output: interpretation and category"""
    ))
    
    # Blood Glucose Interpreter
    tools.append(Tool(
        name="interpret_blood_glucose",
        func=lambda x: interpret_blood_glucose(float(x.split(',')[0]), x.split(',')[1] if ',' in x else 'random'),
        description="""Interpret blood glucose level.
        Input: 'value,timing' where timing is 'fasting' or 'random' (e.g., '126,fasting')
        Output: interpretation and category"""
    ))
    
    # Duration Parser
    tools.append(Tool(
        name="parse_symptom_duration",
        func=parse_duration,
        description="""Parse duration from patient's description.
        Input: duration text (e.g., '2-3 months', 'since last week')
        Output: structured duration information"""
    ))
    
    return tools


# =============================================================================
# TOOL EXECUTION WITH LOGGING
# =============================================================================

import logging
from functools import wraps
from datetime import datetime

tool_logger = logging.getLogger("tools")


def log_tool_execution(func):
    """Decorator to log tool executions for explainability."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        tool_name = func.__name__
        
        tool_logger.info(f"Tool '{tool_name}' called with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            tool_logger.info(f"Tool '{tool_name}' completed in {duration:.3f}s")
            return result
        except Exception as e:
            tool_logger.error(f"Tool '{tool_name}' failed: {e}")
            raise
    
    return wrapper


# =============================================================================
# TOOL ERROR HANDLING
# =============================================================================

class ToolError(Exception):
    """Base exception for tool errors."""
    pass


class RAGSearchError(ToolError):
    """Error in RAG search operation."""
    pass


class CalculationError(ToolError):
    """Error in medical calculation."""
    pass


class ValidationError(ToolError):
    """Error in input validation."""
    pass


def safe_tool_execution(func: callable, default: any = None):
    """
    Wrapper for safe tool execution with fallback.
    
    Args:
        func: Tool function to execute
        default: Default value if execution fails
        
    Returns:
        Wrapped function that handles errors gracefully
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            tool_logger.error(f"Tool execution failed: {e}")
            return default
    
    return wrapper