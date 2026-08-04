"""
Time Utilities Tool
===================
Utilities for parsing and analyzing time-related information.

This tool handles:
1. Duration parsing (e.g., "2-3 months", "since last week")
2. Symptom urgency based on duration
3. Acute vs. chronic onset detection
4. Time formatting for clinical notes

Supports multiple languages (English, Urdu, Punjabi).
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger("time_utils")


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class DurationUnit(str, Enum):
    """Time duration units."""
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"


class OnsetType(str, Enum):
    """Symptom onset classification."""
    SUDDEN = "sudden"           # Seconds to minutes
    ACUTE = "acute"             # Hours to a day
    SUBACUTE = "subacute"       # Days to weeks
    CHRONIC = "chronic"         # Weeks to months
    LONGSTANDING = "longstanding"  # Months to years


class UrgencyLevel(str, Enum):
    """Urgency based on symptom duration and type."""
    EMERGENT = "emergent"       # Immediate attention
    URGENT = "urgent"           # Within hours
    SEMI_URGENT = "semi_urgent" # Within a day
    ROUTINE = "routine"         # Can wait


@dataclass
class ParsedDuration:
    """Parsed duration result."""
    value: float
    unit: DurationUnit
    original_text: str
    confidence: float
    is_range: bool
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def to_hours(self) -> float:
        """Convert to hours for comparison."""
        multipliers = {
            DurationUnit.SECONDS: 1/3600,
            DurationUnit.MINUTES: 1/60,
            DurationUnit.HOURS: 1,
            DurationUnit.DAYS: 24,
            DurationUnit.WEEKS: 24 * 7,
            DurationUnit.MONTHS: 24 * 30,
            DurationUnit.YEARS: 24 * 365
        }
        return self.value * multipliers.get(self.unit, 1)
    
    def to_days(self) -> float:
        """Convert to days."""
        return self.to_hours() / 24


# =============================================================================
# TIME PARSER CLASS
# =============================================================================

class TimeParser:
    """
    Parser for medical time expressions.
    
    Handles:
    - Numeric durations: "2 weeks", "3-4 days"
    - Relative time: "since yesterday", "last month"
    - Vague time: "a few days", "several weeks"
    - Multilingual: English, Urdu, Punjabi
    """
    
    # English patterns
    ENGLISH_PATTERNS = {
        # Exact durations
        r'(\d+)\s*(second|sec|s)s?': (DurationUnit.SECONDS, 1),
        r'(\d+)\s*(minute|min|m)s?': (DurationUnit.MINUTES, 1),
        r'(\d+)\s*(hour|hr|h)s?': (DurationUnit.HOURS, 1),
        r'(\d+)\s*(day|d)s?': (DurationUnit.DAYS, 1),
        r'(\d+)\s*(week|wk|w)s?': (DurationUnit.WEEKS, 1),
        r'(\d+)\s*(month|mo)s?': (DurationUnit.MONTHS, 1),
        r'(\d+)\s*(year|yr|y)s?': (DurationUnit.YEARS, 1),
        
        # Ranges
        r'(\d+)\s*[-–to]+\s*(\d+)\s*(day|week|month|year)s?': None,  # Handled specially
        
        # Relative time
        r'since\s+yesterday': (DurationUnit.DAYS, 1),
        r'since\s+last\s+night': (DurationUnit.HOURS, 12),
        r'since\s+this\s+morning': (DurationUnit.HOURS, 6),
        r'since\s+last\s+week': (DurationUnit.WEEKS, 1),
        r'since\s+last\s+month': (DurationUnit.MONTHS, 1),
        
        # Vague durations
        r'a\s+few\s+(minute|hour|day|week|month)s?': None,  # 2-3 of unit
        r'several\s+(minute|hour|day|week|month)s?': None,  # 4-6 of unit
        r'couple\s+of\s+(minute|hour|day|week|month)s?': None,  # 2 of unit
    }
    
    # Urdu patterns (transliterated and native script)
    URDU_PATTERNS = {
        r'(\d+)\s*(سیکنڈ|سکینڈ)': (DurationUnit.SECONDS, 1),
        r'(\d+)\s*(منٹ|منٹس)': (DurationUnit.MINUTES, 1),
        r'(\d+)\s*(گھنٹہ|گھنٹے)': (DurationUnit.HOURS, 1),
        r'(\d+)\s*(دن|din)': (DurationUnit.DAYS, 1),
        r'(\d+)\s*(ہفتہ|ہفتے|hafte)': (DurationUnit.WEEKS, 1),
        r'(\d+)\s*(مہینہ|مہینے|mahine)': (DurationUnit.MONTHS, 1),
        r'(\d+)\s*(سال|saal)': (DurationUnit.YEARS, 1),
        
        # Relative (Urdu)
        r'کل\s+سے': (DurationUnit.DAYS, 1),  # since yesterday
        r'پچھلے\s+ہفتے\s+سے': (DurationUnit.WEEKS, 1),
        r'پچھلے\s+مہینے\s+سے': (DurationUnit.MONTHS, 1),
    }
    
    # Punjabi patterns
    PUNJABI_PATTERNS = {
        r'(\d+)\s*(دن|din)': (DurationUnit.DAYS, 1),
        r'(\d+)\s*(ہفتہ|ہفتے|hafte)': (DurationUnit.WEEKS, 1),
        r'(\d+)\s*(مہینہ|مہینے|mahine)': (DurationUnit.MONTHS, 1),
        r'کنے\s+چر\s+توں': None,  # "how long" marker
    }
    
    @classmethod
    def parse(cls, text: str) -> Optional[ParsedDuration]:
        """
        Parse a duration from text.
        
        Args:
            text: Text containing duration information
            
        Returns:
            ParsedDuration object or None if not parseable
        """
        if not text:
            return None
        
        text_lower = text.lower().strip()
        
        # Try range pattern first
        range_result = cls._parse_range(text_lower)
        if range_result:
            return range_result
        
        # Try exact duration patterns
        all_patterns = {
            **cls.ENGLISH_PATTERNS,
            **cls.URDU_PATTERNS,
            **cls.PUNJABI_PATTERNS
        }
        
        for pattern, info in all_patterns.items():
            if info is None:
                continue
                
            match = re.search(pattern, text_lower)
            if match:
                unit, multiplier = info
                
                # Check if there's a number in the match
                groups = match.groups()
                if groups and groups[0].isdigit():
                    value = float(groups[0]) * multiplier
                else:
                    value = multiplier
                
                return ParsedDuration(
                    value=value,
                    unit=unit,
                    original_text=text,
                    confidence=0.9,
                    is_range=False
                )
        
        # Try vague duration patterns
        vague_result = cls._parse_vague(text_lower)
        if vague_result:
            return vague_result
        
        return None
    
    @classmethod
    def _parse_range(cls, text: str) -> Optional[ParsedDuration]:
        """Parse a range duration like '2-3 months'."""
        range_patterns = [
            r'(\d+)\s*[-–to]+\s*(\d+)\s*(second|minute|hour|day|week|month|year)s?',
            r'(\d+)\s*[-–to]+\s*(\d+)\s*(سیکنڈ|منٹ|گھنٹہ|دن|ہفتہ|مہینہ|سال)',
        ]
        
        unit_map = {
            'second': DurationUnit.SECONDS, 'سیکنڈ': DurationUnit.SECONDS,
            'minute': DurationUnit.MINUTES, 'منٹ': DurationUnit.MINUTES,
            'hour': DurationUnit.HOURS, 'گھنٹہ': DurationUnit.HOURS,
            'day': DurationUnit.DAYS, 'دن': DurationUnit.DAYS,
            'week': DurationUnit.WEEKS, 'ہفتہ': DurationUnit.WEEKS,
            'month': DurationUnit.MONTHS, 'مہینہ': DurationUnit.MONTHS,
            'year': DurationUnit.YEARS, 'سال': DurationUnit.YEARS,
        }
        
        for pattern in range_patterns:
            match = re.search(pattern, text)
            if match:
                min_val = float(match.group(1))
                max_val = float(match.group(2))
                unit_str = match.group(3).lower().rstrip('s')
                
                unit = unit_map.get(unit_str, DurationUnit.DAYS)
                avg_val = (min_val + max_val) / 2
                
                return ParsedDuration(
                    value=avg_val,
                    unit=unit,
                    original_text=match.group(0),
                    confidence=0.85,
                    is_range=True,
                    min_value=min_val,
                    max_value=max_val
                )
        
        return None
    
    @classmethod
    def _parse_vague(cls, text: str) -> Optional[ParsedDuration]:
        """Parse vague durations like 'a few days'."""
        vague_map = {
            'few': (2, 3),
            'couple': (2, 2),
            'several': (4, 6),
            'some': (2, 4),
            'many': (5, 10),
        }
        
        unit_map = {
            'minute': DurationUnit.MINUTES,
            'hour': DurationUnit.HOURS,
            'day': DurationUnit.DAYS,
            'week': DurationUnit.WEEKS,
            'month': DurationUnit.MONTHS,
            'year': DurationUnit.YEARS,
        }
        
        for vague_word, (min_v, max_v) in vague_map.items():
            for unit_word, unit in unit_map.items():
                pattern = rf'{vague_word}\s+{unit_word}s?'
                if re.search(pattern, text):
                    avg = (min_v + max_v) / 2
                    return ParsedDuration(
                        value=avg,
                        unit=unit,
                        original_text=text,
                        confidence=0.7,
                        is_range=True,
                        min_value=min_v,
                        max_value=max_v
                    )
        
        return None


# =============================================================================
# URGENCY ASSESSMENT
# =============================================================================

def calculate_symptom_urgency(
    duration: ParsedDuration,
    symptom_type: str = "general"
) -> Dict[str, Any]:
    """
    Calculate urgency based on symptom duration and type.
    
    Args:
        duration: Parsed duration object
        symptom_type: Type of symptom (e.g., "chest_pain", "headache", "general")
        
    Returns:
        Dict with urgency level and recommendations
    """
    hours = duration.to_hours()
    
    # Symptom-specific urgency modifiers
    urgent_symptoms = ["chest_pain", "breathing", "stroke", "bleeding", "unconscious"]
    semi_urgent_symptoms = ["abdominal_pain", "fever", "headache", "injury"]
    
    # Determine base urgency from duration
    if hours < 1:
        onset = OnsetType.SUDDEN
        base_urgency = UrgencyLevel.EMERGENT
    elif hours < 24:
        onset = OnsetType.ACUTE
        base_urgency = UrgencyLevel.URGENT
    elif hours < 24 * 7:  # Less than a week
        onset = OnsetType.SUBACUTE
        base_urgency = UrgencyLevel.SEMI_URGENT
    elif hours < 24 * 30:  # Less than a month
        onset = OnsetType.CHRONIC
        base_urgency = UrgencyLevel.ROUTINE
    else:
        onset = OnsetType.LONGSTANDING
        base_urgency = UrgencyLevel.ROUTINE
    
    # Adjust for symptom type
    final_urgency = base_urgency
    if symptom_type.lower() in urgent_symptoms:
        # Upgrade urgency for dangerous symptoms
        if base_urgency == UrgencyLevel.ROUTINE:
            final_urgency = UrgencyLevel.SEMI_URGENT
        elif base_urgency == UrgencyLevel.SEMI_URGENT:
            final_urgency = UrgencyLevel.URGENT
    
    # Generate recommendation
    recommendations = {
        UrgencyLevel.EMERGENT: "Seek immediate emergency care. Call 1122.",
        UrgencyLevel.URGENT: "Visit emergency room or urgent care within a few hours.",
        UrgencyLevel.SEMI_URGENT: "See a doctor within 24 hours.",
        UrgencyLevel.ROUTINE: "Schedule an appointment with your doctor."
    }
    
    return {
        "onset_type": onset.value,
        "urgency_level": final_urgency.value,
        "duration_hours": hours,
        "duration_description": format_duration(duration),
        "recommendation": recommendations.get(final_urgency, "Consult a healthcare provider."),
        "is_acute": onset in [OnsetType.SUDDEN, OnsetType.ACUTE]
    }


def is_acute_onset(text: str) -> bool:
    """
    Determine if the described onset is acute/sudden.
    
    Args:
        text: Text describing symptom onset
        
    Returns:
        True if onset appears acute/sudden
    """
    text_lower = text.lower()
    
    # Acute/sudden onset indicators
    acute_indicators = [
        "sudden", "suddenly", "all of a sudden",
        "out of nowhere", "abrupt", "abruptly",
        "just started", "just now", "minutes ago",
        "woke up with", "started suddenly",
        "worst ever", "thunderclap",
        "اچانک", "یکدم",  # Urdu: suddenly
    ]
    
    for indicator in acute_indicators:
        if indicator in text_lower:
            return True
    
    # Check parsed duration
    duration = TimeParser.parse(text)
    if duration:
        hours = duration.to_hours()
        if hours < 24:  # Less than a day
            return True
    
    return False


def format_duration(duration: ParsedDuration) -> str:
    """
    Format a parsed duration as human-readable string.
    
    Args:
        duration: ParsedDuration object
        
    Returns:
        Formatted string
    """
    if duration.is_range and duration.min_value and duration.max_value:
        return f"{int(duration.min_value)}-{int(duration.max_value)} {duration.unit.value}"
    else:
        value = duration.value
        if value == int(value):
            value = int(value)
        return f"{value} {duration.unit.value}"


def parse_duration(text: str) -> Dict[str, Any]:
    """
    Parse duration and return as dict.
    
    Args:
        text: Text containing duration
        
    Returns:
        Dict with parsed duration info
    """
    result = TimeParser.parse(text)
    
    if result is None:
        return {
            "parsed": False,
            "original_text": text,
            "message": "Could not parse duration from text"
        }
    
    return {
        "parsed": True,
        "value": result.value,
        "unit": result.unit.value,
        "original_text": result.original_text,
        "confidence": result.confidence,
        "is_range": result.is_range,
        "min_value": result.min_value,
        "max_value": result.max_value,
        "formatted": format_duration(result),
        "total_hours": result.to_hours(),
        "total_days": result.to_days()
    }


# =============================================================================
# TIME SINCE SYMPTOM STARTED
# =============================================================================

def time_since_symptom_start(
    start_description: str,
    reference_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Calculate time elapsed since symptom started.
    
    Args:
        start_description: When symptom started (e.g., "yesterday", "3 days ago")
        reference_time: Reference time (default: now)
        
    Returns:
        Dict with elapsed time information
    """
    if reference_time is None:
        reference_time = datetime.now()
    
    duration = TimeParser.parse(start_description)
    
    if duration is None:
        # Try special cases
        start_lower = start_description.lower()
        
        if "yesterday" in start_lower or "کل" in start_lower:
            hours = 24
        elif "this morning" in start_lower or "آج صبح" in start_lower:
            hours = (reference_time - reference_time.replace(hour=6, minute=0)).seconds / 3600
        elif "last night" in start_lower or "کل رات" in start_lower:
            hours = 12
        elif "today" in start_lower or "آج" in start_lower:
            hours = 6  # Assume a few hours ago
        else:
            return {
                "parsed": False,
                "description": start_description
            }
        
        return {
            "parsed": True,
            "hours_elapsed": hours,
            "days_elapsed": hours / 24,
            "description": start_description,
            "onset_type": "acute" if hours < 24 else "subacute"
        }
    
    hours = duration.to_hours()
    
    return {
        "parsed": True,
        "hours_elapsed": hours,
        "days_elapsed": duration.to_days(),
        "description": start_description,
        "onset_type": (
            "sudden" if hours < 1 else
            "acute" if hours < 24 else
            "subacute" if hours < 24 * 7 else
            "chronic"
        )
    }


# =============================================================================
# SYMPTOM TIMING PATTERNS
# =============================================================================

def analyze_timing_pattern(descriptions: List[str]) -> Dict[str, Any]:
    """
    Analyze timing pattern from multiple descriptions.
    
    Args:
        descriptions: List of timing descriptions
        
    Returns:
        Dict with timing pattern analysis
    """
    patterns = {
        "constant": ["constant", "all the time", "continuous", "always", "ہمیشہ"],
        "intermittent": ["comes and goes", "on and off", "sometimes", "occasional", "کبھی کبھی"],
        "worsening": ["getting worse", "increasing", "more frequent", "بڑھ رہا ہے"],
        "improving": ["getting better", "less frequent", "improving", "بہتر ہو رہا ہے"],
        "cyclic": ["every few hours", "daily", "weekly", "روزانہ", "ہفتہ وار"]
    }
    
    combined_text = " ".join(descriptions).lower()
    
    detected_patterns = []
    for pattern_name, indicators in patterns.items():
        if any(ind in combined_text for ind in indicators):
            detected_patterns.append(pattern_name)
    
    # Determine primary pattern
    if "worsening" in detected_patterns:
        primary = "progressive"
        concern_level = "high"
    elif "constant" in detected_patterns:
        primary = "persistent"
        concern_level = "moderate"
    elif "intermittent" in detected_patterns:
        primary = "episodic"
        concern_level = "moderate" if "worsening" not in detected_patterns else "high"
    elif "improving" in detected_patterns:
        primary = "resolving"
        concern_level = "low"
    else:
        primary = "unclear"
        concern_level = "moderate"
    
    return {
        "primary_pattern": primary,
        "detected_patterns": detected_patterns,
        "concern_level": concern_level,
        "recommendation": (
            "Urgent evaluation recommended" if concern_level == "high" else
            "Medical evaluation advised" if concern_level == "moderate" else
            "Monitor and follow up if symptoms return"
        )
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Time Utilities...")
    
    # Test duration parsing
    print("\n=== Duration Parsing ===")
    test_cases = [
        "2-3 months",
        "about 3 days",
        "since yesterday",
        "a few hours",
        "several weeks",
        "تقریباً 2-3 مہینے",  # Urdu
        "پچھلے ہفتے سے",  # Urdu: since last week
    ]
    
    for test in test_cases:
        result = parse_duration(test)
        print(f"\n'{test}':")
        if result["parsed"]:
            print(f"  Parsed: {result['formatted']}")
            print(f"  Total hours: {result['total_hours']:.1f}")
            print(f"  Confidence: {result['confidence']:.0%}")
        else:
            print(f"  {result['message']}")
    
    # Test acute onset detection
    print("\n=== Acute Onset Detection ===")
    onset_tests = [
        "It started suddenly while I was walking",
        "I've had this for about a month",
        "The worst headache of my life came on instantly",
        "It gradually got worse over weeks",
    ]
    
    for test in onset_tests:
        is_acute = is_acute_onset(test)
        print(f"'{test[:50]}...'")
        print(f"  Is acute: {is_acute}")
    
    # Test urgency calculation
    print("\n=== Urgency Calculation ===")
    duration = TimeParser.parse("30 minutes")
    if duration:
        urgency = calculate_symptom_urgency(duration, "chest_pain")
        print(f"Chest pain for 30 minutes:")
        print(f"  Onset: {urgency['onset_type']}")
        print(f"  Urgency: {urgency['urgency_level']}")
        print(f"  Recommendation: {urgency['recommendation']}")