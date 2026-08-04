"""
Medical Calculator Tool
=======================
Medical calculations to prevent AI hallucination on numbers.

This tool provides:
1. BMI calculation and interpretation
2. Body Surface Area (BSA) calculation
3. Blood pressure interpretation
4. Blood glucose interpretation
5. Temperature interpretation
6. Dehydration risk assessment

IMPORTANT: Calculations are evidence-based. The AI should NEVER
perform medical calculations without using this tool.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger("calculator")


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class BMICategory(str, Enum):
    """BMI classification categories (WHO)."""
    UNDERWEIGHT_SEVERE = "Severely Underweight"
    UNDERWEIGHT = "Underweight"
    NORMAL = "Normal Weight"
    OVERWEIGHT = "Overweight"
    OBESE_CLASS_1 = "Obese Class I"
    OBESE_CLASS_2 = "Obese Class II"
    OBESE_CLASS_3 = "Obese Class III"


class BloodPressureCategory(str, Enum):
    """Blood pressure classification (AHA)."""
    NORMAL = "Normal"
    ELEVATED = "Elevated"
    HYPERTENSION_STAGE_1 = "Hypertension Stage 1"
    HYPERTENSION_STAGE_2 = "Hypertension Stage 2"
    HYPERTENSIVE_CRISIS = "Hypertensive Crisis"
    HYPOTENSION = "Hypotension"


class GlucoseCategory(str, Enum):
    """Blood glucose classification."""
    HYPOGLYCEMIA_SEVERE = "Severe Hypoglycemia"
    HYPOGLYCEMIA = "Hypoglycemia"
    NORMAL = "Normal"
    PREDIABETES = "Prediabetes"
    DIABETES = "Diabetes Range"
    DIABETIC_EMERGENCY = "Diabetic Emergency"


class TemperatureCategory(str, Enum):
    """Body temperature classification."""
    HYPOTHERMIA_SEVERE = "Severe Hypothermia"
    HYPOTHERMIA = "Hypothermia"
    NORMAL = "Normal"
    LOW_GRADE_FEVER = "Low-Grade Fever"
    FEVER = "Fever"
    HIGH_FEVER = "High Fever"
    HYPERPYREXIA = "Hyperpyrexia (Critical)"


@dataclass
class CalculationResult:
    """Standard result format for calculations."""
    value: float
    unit: str
    category: str
    interpretation: str
    is_critical: bool
    action: str


# =============================================================================
# MEDICAL CALCULATOR CLASS
# =============================================================================

class MedicalCalculator:
    """
    Medical calculation utility class.
    
    All calculations follow evidence-based guidelines.
    Results include interpretation and recommended actions.
    """
    
    @staticmethod
    def calculate_bmi(
        height_cm: float,
        weight_kg: float
    ) -> CalculationResult:
        """
        Calculate Body Mass Index.
        
        Args:
            height_cm: Height in centimeters
            weight_kg: Weight in kilograms
            
        Returns:
            CalculationResult with BMI value and interpretation
        """
        # Validate inputs
        if height_cm <= 0 or weight_kg <= 0:
            raise ValueError("Height and weight must be positive")
        if height_cm < 50 or height_cm > 300:
            raise ValueError("Height must be between 50 and 300 cm")
        if weight_kg < 10 or weight_kg > 500:
            raise ValueError("Weight must be between 10 and 500 kg")
        
        # Calculate BMI
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        bmi = round(bmi, 1)
        
        # Categorize
        if bmi < 16:
            category = BMICategory.UNDERWEIGHT_SEVERE
            interpretation = "Severely underweight - significant health risk"
            is_critical = True
            action = "Urgent medical evaluation for malnutrition"
        elif bmi < 18.5:
            category = BMICategory.UNDERWEIGHT
            interpretation = "Underweight - may indicate health issues"
            is_critical = False
            action = "Discuss nutrition with healthcare provider"
        elif bmi < 25:
            category = BMICategory.NORMAL
            interpretation = "Normal weight - healthy range"
            is_critical = False
            action = "Maintain healthy lifestyle"
        elif bmi < 30:
            category = BMICategory.OVERWEIGHT
            interpretation = "Overweight - increased health risk"
            is_critical = False
            action = "Consider lifestyle modifications"
        elif bmi < 35:
            category = BMICategory.OBESE_CLASS_1
            interpretation = "Obese Class I - moderate health risk"
            is_critical = False
            action = "Discuss weight management with doctor"
        elif bmi < 40:
            category = BMICategory.OBESE_CLASS_2
            interpretation = "Obese Class II - severe health risk"
            is_critical = True
            action = "Medical intervention recommended"
        else:
            category = BMICategory.OBESE_CLASS_3
            interpretation = "Obese Class III - very severe health risk"
            is_critical = True
            action = "Urgent medical evaluation needed"
        
        return CalculationResult(
            value=bmi,
            unit="kg/m²",
            category=category.value,
            interpretation=interpretation,
            is_critical=is_critical,
            action=action
        )
    
    @staticmethod
    def calculate_bsa(
        height_cm: float,
        weight_kg: float,
        formula: str = "mosteller"
    ) -> CalculationResult:
        """
        Calculate Body Surface Area.
        
        Args:
            height_cm: Height in centimeters
            weight_kg: Weight in kilograms
            formula: 'mosteller' or 'dubois'
            
        Returns:
            CalculationResult with BSA value
        """
        import math
        
        if formula == "mosteller":
            # Mosteller formula: BSA = sqrt((height_cm * weight_kg) / 3600)
            bsa = math.sqrt((height_cm * weight_kg) / 3600)
        else:
            # DuBois formula: BSA = 0.007184 * height^0.725 * weight^0.425
            bsa = 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425)
        
        bsa = round(bsa, 2)
        
        # Normal range for adults: 1.5-2.0 m²
        if bsa < 1.5:
            category = "Below average"
        elif bsa <= 2.0:
            category = "Normal range"
        else:
            category = "Above average"
        
        return CalculationResult(
            value=bsa,
            unit="m²",
            category=category,
            interpretation=f"Body Surface Area: {bsa} m²",
            is_critical=False,
            action="Use for medication dosing calculations"
        )
    
    @staticmethod
    def interpret_blood_pressure(
        systolic: int,
        diastolic: int
    ) -> CalculationResult:
        """
        Interpret blood pressure reading.
        
        Args:
            systolic: Systolic pressure (mmHg)
            diastolic: Diastolic pressure (mmHg)
            
        Returns:
            CalculationResult with interpretation
        """
        # Validate
        if systolic < 50 or systolic > 300:
            raise ValueError("Systolic pressure out of valid range")
        if diastolic < 30 or diastolic > 200:
            raise ValueError("Diastolic pressure out of valid range")
        if diastolic > systolic:
            raise ValueError("Diastolic cannot be higher than systolic")
        
        # Classify (AHA Guidelines)
        if systolic < 90 or diastolic < 60:
            category = BloodPressureCategory.HYPOTENSION
            interpretation = "Low blood pressure - may cause dizziness"
            is_critical = systolic < 80
            action = "Evaluate for dehydration or other causes" if not is_critical else "Urgent evaluation needed"
        elif systolic < 120 and diastolic < 80:
            category = BloodPressureCategory.NORMAL
            interpretation = "Normal blood pressure"
            is_critical = False
            action = "Maintain healthy lifestyle"
        elif systolic < 130 and diastolic < 80:
            category = BloodPressureCategory.ELEVATED
            interpretation = "Elevated blood pressure - at risk for hypertension"
            is_critical = False
            action = "Lifestyle modifications recommended"
        elif systolic < 140 or diastolic < 90:
            category = BloodPressureCategory.HYPERTENSION_STAGE_1
            interpretation = "Hypertension Stage 1"
            is_critical = False
            action = "Consult doctor, lifestyle changes, possible medication"
        elif systolic < 180 and diastolic < 120:
            category = BloodPressureCategory.HYPERTENSION_STAGE_2
            interpretation = "Hypertension Stage 2 - requires treatment"
            is_critical = False
            action = "Medical treatment likely needed"
        else:
            category = BloodPressureCategory.HYPERTENSIVE_CRISIS
            interpretation = "HYPERTENSIVE CRISIS - Extremely high blood pressure"
            is_critical = True
            action = "SEEK IMMEDIATE MEDICAL ATTENTION"
        
        return CalculationResult(
            value=systolic,  # Primary value
            unit=f"{systolic}/{diastolic} mmHg",
            category=category.value,
            interpretation=interpretation,
            is_critical=is_critical,
            action=action
        )
    
    @staticmethod
    def interpret_blood_glucose(
        value: float,
        timing: str = "random"
    ) -> CalculationResult:
        """
        Interpret blood glucose level.
        
        Args:
            value: Glucose value in mg/dL
            timing: 'fasting' (8+ hours) or 'random'
            
        Returns:
            CalculationResult with interpretation
        """
        timing = timing.lower()
        
        if timing == "fasting":
            # Fasting glucose interpretation
            if value < 54:
                category = GlucoseCategory.HYPOGLYCEMIA_SEVERE
                interpretation = "Severe hypoglycemia - dangerous"
                is_critical = True
                action = "IMMEDIATE glucose intake needed. Seek emergency care."
            elif value < 70:
                category = GlucoseCategory.HYPOGLYCEMIA
                interpretation = "Low blood sugar"
                is_critical = True
                action = "Take 15-20g fast-acting carbohydrates immediately"
            elif value < 100:
                category = GlucoseCategory.NORMAL
                interpretation = "Normal fasting glucose"
                is_critical = False
                action = "Normal range, maintain healthy diet"
            elif value < 126:
                category = GlucoseCategory.PREDIABETES
                interpretation = "Prediabetes range (impaired fasting glucose)"
                is_critical = False
                action = "Lifestyle modifications recommended, follow-up testing"
            elif value < 250:
                category = GlucoseCategory.DIABETES
                interpretation = "Diabetes range"
                is_critical = False
                action = "Medical evaluation for diabetes diagnosis and management"
            else:
                category = GlucoseCategory.DIABETIC_EMERGENCY
                interpretation = "Very high glucose - possible diabetic emergency"
                is_critical = True
                action = "SEEK IMMEDIATE MEDICAL ATTENTION"
        else:
            # Random glucose interpretation
            if value < 54:
                category = GlucoseCategory.HYPOGLYCEMIA_SEVERE
                interpretation = "Severe hypoglycemia"
                is_critical = True
                action = "IMMEDIATE glucose intake and emergency care"
            elif value < 70:
                category = GlucoseCategory.HYPOGLYCEMIA
                interpretation = "Low blood sugar"
                is_critical = True
                action = "Take fast-acting carbohydrates"
            elif value < 140:
                category = GlucoseCategory.NORMAL
                interpretation = "Normal random glucose"
                is_critical = False
                action = "Normal range"
            elif value < 200:
                category = GlucoseCategory.PREDIABETES
                interpretation = "Elevated glucose - needs fasting test"
                is_critical = False
                action = "Schedule fasting glucose test"
            elif value < 300:
                category = GlucoseCategory.DIABETES
                interpretation = "Diabetes range (if symptomatic)"
                is_critical = False
                action = "Medical evaluation needed"
            else:
                category = GlucoseCategory.DIABETIC_EMERGENCY
                interpretation = "Critically high - possible DKA or HHS"
                is_critical = True
                action = "SEEK IMMEDIATE EMERGENCY CARE"
        
        return CalculationResult(
            value=value,
            unit="mg/dL",
            category=category.value,
            interpretation=interpretation,
            is_critical=is_critical,
            action=action
        )
    
    @staticmethod
    def interpret_temperature(
        value: float,
        unit: str = "F"
    ) -> CalculationResult:
        """
        Interpret body temperature.
        
        Args:
            value: Temperature value
            unit: 'F' for Fahrenheit, 'C' for Celsius
            
        Returns:
            CalculationResult with interpretation
        """
        # Convert to Fahrenheit if Celsius
        if unit.upper() == "C":
            temp_f = (value * 9/5) + 32
            temp_c = value
        else:
            temp_f = value
            temp_c = (value - 32) * 5/9
        
        # Classify
        if temp_f < 95:
            category = TemperatureCategory.HYPOTHERMIA_SEVERE if temp_f < 90 else TemperatureCategory.HYPOTHERMIA
            interpretation = "Hypothermia - body temperature dangerously low"
            is_critical = temp_f < 90
            action = "Seek immediate medical attention" if is_critical else "Warm gradually, monitor closely"
        elif temp_f < 97.8:
            category = TemperatureCategory.NORMAL  # Low normal
            interpretation = "Low-normal temperature"
            is_critical = False
            action = "Monitor, usually normal variation"
        elif temp_f <= 99.5:
            category = TemperatureCategory.NORMAL
            interpretation = "Normal body temperature"
            is_critical = False
            action = "Normal range"
        elif temp_f < 100.4:
            category = TemperatureCategory.LOW_GRADE_FEVER
            interpretation = "Low-grade fever"
            is_critical = False
            action = "Monitor, rest, stay hydrated"
        elif temp_f < 102.2:
            category = TemperatureCategory.FEVER
            interpretation = "Fever present"
            is_critical = False
            action = "Consider fever reducers, rest, hydration, watch for worsening"
        elif temp_f < 104:
            category = TemperatureCategory.HIGH_FEVER
            interpretation = "High fever - needs attention"
            is_critical = False
            action = "Take fever reducers, cool compresses, seek medical advice"
        else:
            category = TemperatureCategory.HYPERPYREXIA
            interpretation = "HYPERPYREXIA - Dangerously high temperature"
            is_critical = True
            action = "SEEK IMMEDIATE EMERGENCY CARE"
        
        return CalculationResult(
            value=round(temp_f, 1),
            unit=f"{round(temp_f, 1)}°F ({round(temp_c, 1)}°C)",
            category=category.value,
            interpretation=interpretation,
            is_critical=is_critical,
            action=action
        )
    
    @staticmethod
    def assess_dehydration_risk(
        symptoms: list
    ) -> CalculationResult:
        """
        Assess dehydration risk based on symptoms.
        
        Args:
            symptoms: List of symptom strings
            
        Returns:
            CalculationResult with dehydration assessment
        """
        symptoms_lower = [s.lower() for s in symptoms]
        
        # Dehydration indicators
        mild_signs = [
            "thirst", "dry mouth", "dark urine", "decreased urination",
            "fatigue", "headache", "dry skin"
        ]
        moderate_signs = [
            "very dry mouth", "little urination", "sunken eyes",
            "rapid heartbeat", "dizziness", "muscle cramps"
        ]
        severe_signs = [
            "no urination", "very dark urine", "confusion", "rapid breathing",
            "fainting", "unconscious", "shock", "weak pulse"
        ]
        
        # Count matches
        mild_count = sum(1 for s in mild_signs if any(s in sym for sym in symptoms_lower))
        moderate_count = sum(1 for s in moderate_signs if any(s in sym for sym in symptoms_lower))
        severe_count = sum(1 for s in severe_signs if any(s in sym for sym in symptoms_lower))
        
        # Determine severity
        if severe_count > 0:
            category = "Severe Dehydration"
            interpretation = "Signs of severe dehydration present"
            is_critical = True
            action = "SEEK IMMEDIATE MEDICAL ATTENTION - IV fluids may be needed"
            risk_score = 9
        elif moderate_count >= 2:
            category = "Moderate Dehydration"
            interpretation = "Moderate dehydration signs present"
            is_critical = False
            action = "Increase fluid intake significantly, seek medical care if not improving"
            risk_score = 6
        elif mild_count >= 2 or moderate_count >= 1:
            category = "Mild Dehydration"
            interpretation = "Mild dehydration signs present"
            is_critical = False
            action = "Increase fluid intake, rest, monitor symptoms"
            risk_score = 3
        else:
            category = "No Significant Dehydration"
            interpretation = "No clear dehydration signs"
            is_critical = False
            action = "Maintain normal fluid intake"
            risk_score = 0
        
        return CalculationResult(
            value=risk_score,
            unit="risk score (0-10)",
            category=category,
            interpretation=interpretation,
            is_critical=is_critical,
            action=action
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_bmi(height_cm: float, weight_kg: float) -> Dict[str, Any]:
    """Calculate BMI and return as dict."""
    result = MedicalCalculator.calculate_bmi(height_cm, weight_kg)
    return {
        "bmi": result.value,
        "category": result.category,
        "interpretation": result.interpretation,
        "is_critical": result.is_critical,
        "action": result.action
    }


def calculate_bsa(height_cm: float, weight_kg: float) -> Dict[str, Any]:
    """Calculate BSA and return as dict."""
    result = MedicalCalculator.calculate_bsa(height_cm, weight_kg)
    return {
        "bsa": result.value,
        "unit": result.unit,
        "category": result.category
    }


def interpret_blood_pressure(systolic: int, diastolic: int) -> Dict[str, Any]:
    """Interpret blood pressure and return as dict."""
    result = MedicalCalculator.interpret_blood_pressure(systolic, diastolic)
    return {
        "reading": result.unit,
        "category": result.category,
        "interpretation": result.interpretation,
        "is_critical": result.is_critical,
        "action": result.action
    }


def interpret_blood_glucose(value: float, timing: str = "random") -> Dict[str, Any]:
    """Interpret blood glucose and return as dict."""
    result = MedicalCalculator.interpret_blood_glucose(value, timing)
    return {
        "value": result.value,
        "unit": result.unit,
        "category": result.category,
        "interpretation": result.interpretation,
        "is_critical": result.is_critical,
        "action": result.action
    }


def interpret_temperature(value: float, unit: str = "F") -> Dict[str, Any]:
    """Interpret temperature and return as dict."""
    result = MedicalCalculator.interpret_temperature(value, unit)
    return {
        "temperature": result.unit,
        "category": result.category,
        "interpretation": result.interpretation,
        "is_critical": result.is_critical,
        "action": result.action
    }


def assess_dehydration_risk(symptoms: list) -> Dict[str, Any]:
    """Assess dehydration risk and return as dict."""
    result = MedicalCalculator.assess_dehydration_risk(symptoms)
    return {
        "risk_score": result.value,
        "category": result.category,
        "interpretation": result.interpretation,
        "is_critical": result.is_critical,
        "action": result.action
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Medical Calculator...")
    
    # BMI Test
    print("\n=== BMI Test ===")
    result = calculate_bmi(170, 75)
    print(f"Height: 170cm, Weight: 75kg")
    print(f"BMI: {result['bmi']}, Category: {result['category']}")
    
    # Blood Pressure Test
    print("\n=== Blood Pressure Test ===")
    result = interpret_blood_pressure(140, 90)
    print(f"BP: {result['reading']}")
    print(f"Category: {result['category']}")
    print(f"Action: {result['action']}")
    
    # Blood Glucose Test
    print("\n=== Blood Glucose Test ===")
    result = interpret_blood_glucose(126, "fasting")
    print(f"Fasting glucose: {result['value']} {result['unit']}")
    print(f"Category: {result['category']}")
    print(f"Action: {result['action']}")
    
    # Temperature Test
    print("\n=== Temperature Test ===")
    result = interpret_temperature(101.5, "F")
    print(f"Temperature: {result['temperature']}")
    print(f"Category: {result['category']}")
    print(f"Action: {result['action']}")
    
    # Dehydration Test
    print("\n=== Dehydration Risk Test ===")
    result = assess_dehydration_risk(["thirst", "dry mouth", "dark urine", "dizziness"])
    print(f"Risk Score: {result['risk_score']}")
    print(f"Category: {result['category']}")
    print(f"Action: {result['action']}")