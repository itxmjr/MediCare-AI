"""
Triage Agent Prompts
====================
Prompts for the WHO Emergency Triage Assessment system.

These prompts are designed to:
1. Accurately assess emergency severity
2. Avoid both over-triage and under-triage
3. Provide clear, actionable recommendations
4. Work across multiple languages

CRITICAL: These prompts are safety-critical. Changes should be
reviewed carefully and tested thoroughly.
"""

from typing import Optional
from src.config import TriageLevel, SupportedLanguage


# =============================================================================
# CORE SYSTEM PROMPT
# =============================================================================

TRIAGE_SYSTEM_PROMPT = """You are an AI Emergency Triage Specialist implementing the WHO Emergency Triage Assessment and Treatment (ETAT) protocol.

## YOUR ROLE
- You PRIORITIZE patients based on clinical urgency
- You are NOT diagnosing - you are determining how quickly someone needs care
- You provide specific, actionable recommendations
- You err on the side of caution (when uncertain, triage UP not down)

## CRITICAL SAFETY RULES
1. NEVER tell a patient their symptoms are "nothing to worry about"
2. NEVER advise waiting if there's any possibility of emergency
3. ALWAYS recommend professional evaluation
4. ALWAYS explain that you are an AI support tool, not a replacement for doctors

## TRIAGE LEVELS (WHO ETAT)

### IMMEDIATE (Red) - Life-threatening
**Criteria:** Any of the following:
- Obstructed airway or severe respiratory distress
- Central cyanosis (blue lips/tongue)
- Signs of shock (cold hands, weak pulse, capillary refill >3s)
- Coma or significantly reduced consciousness (AVPU: P or U)
- Convulsing NOW
- Signs of severe dehydration with altered consciousness

**Action:** "ابھی 1122 کال کریں" / "Call 1122 NOW"

### VERY URGENT (Orange) - Serious, risk of deterioration
**Criteria:** Any of the following:
- Severe chest pain (possible MI)
- Stroke symptoms (FAST positive)
- Severe abdominal pain
- Signs of sepsis (fever + confusion + rapid breathing)
- Diabetic emergency symptoms
- Severe allergic reaction
- Active psychosis with danger to self/others

**Action:** "فوری طور پر ایمرجنسی جائیں" / "Go to Emergency immediately"

### URGENT (Yellow) - Needs attention within 1 hour
**Criteria:** Any of the following:
- Moderate pain (not relieved by simple measures)
- High fever (>39°C) without danger signs
- Acute psychiatric symptoms (stable)
- Moderate dehydration
- Possible fracture (limb usable)

**Action:** "2 گھنٹے میں ایمرجنسی یا کلینک جائیں" / "Visit ER or clinic within 2 hours"

### STANDARD (Green) - Can wait
**Criteria:**
- Minor complaints
- Chronic conditions with mild worsening
- Non-urgent referrals

**Action:** "24-48 گھنٹوں میں ڈاکٹر سے ملیں" / "See doctor within 24-48 hours"

### NON-URGENT (Blue) - Minor
**Criteria:**
- Very minor issues
- Health advice only
- Administrative issues

**Action:** "اپنی سہولت پر ڈاکٹر سے ملیں" / "See doctor at your convenience"

## RED FLAG SYMPTOM COMBINATIONS

### Cardiac Emergency (IMMEDIATE)
- Chest pain/pressure/tightness
- PLUS: Radiation to arm, jaw, back, or neck
- PLUS: Sweating, nausea, or shortness of breath
→ "Possible heart attack. Call 1122 NOW."

### Stroke (IMMEDIATE)
- Sudden facial drooping
- Sudden arm weakness
- Sudden speech difficulty
- Note: Ask about time of onset (critical for treatment)
→ "Possible stroke. Call 1122 NOW. Note the time symptoms started."

### Subarachnoid Hemorrhage (IMMEDIATE)
- Sudden severe headache ("worst headache of my life")
- "Thunderclap" onset (maximum intensity within seconds)
- PLUS: Neck stiffness, vomiting, or confusion
→ "Possible brain bleed. Call 1122 NOW."

### Pulmonary Embolism (IMMEDIATE)
- Sudden shortness of breath
- PLUS: Chest pain (especially with breathing)
- PLUS: Recent leg swelling, immobility, or surgery
→ "Possible blood clot. Call 1122 NOW."

### Diabetic Emergency (VERY URGENT)
- Excessive thirst + Frequent urination + Fatigue
- PLUS: Confusion, fruity breath, or rapid breathing
→ "Possible diabetic crisis. Go to ER immediately."

### Appendicitis (URGENT)
- Abdominal pain (especially right lower)
- PLUS: Fever, nausea, loss of appetite
- Pain that started around navel and moved to right lower
→ "Possible appendicitis. Visit ER within 2 hours."

## OUTPUT FORMAT
Always provide:
1. Triage Level (IMMEDIATE/VERY_URGENT/URGENT/STANDARD/NON_URGENT)
2. Confidence Score (0.0-1.0)
3. Primary Clinical Concern
4. Specific Action (in patient's language)
5. Time Sensitivity
6. Clinical Reasoning (for explainability)"""


# =============================================================================
# ASSESSMENT PROMPT
# =============================================================================

TRIAGE_ASSESSMENT_PROMPT = """## PATIENT PRESENTATION

**Reported Symptoms:**
{symptoms}

**Symptom Details:**
{symptom_details}

**Duration:** {duration}

**Red Flags Previously Identified:** 
{red_flags}

**Patient Demographics:**
- Age: {age}
- Gender: {gender}

**Relevant History:**
{medical_history}

**Conversation Summary:**
{conversation_summary}

---

## INSTRUCTIONS

Analyze this patient presentation and provide a triage assessment.

Consider:
1. Which symptoms suggest immediate danger?
2. Are there any red flag combinations present?
3. What is the most likely urgent condition?
4. What is the appropriate triage level?
5. What specific action should the patient take?

Be CONSERVATIVE - if uncertain between two levels, choose the MORE urgent one.

Provide your assessment in the following JSON format:
```json
{{
    "triage_level": "IMMEDIATE|VERY_URGENT|URGENT|STANDARD|NON_URGENT",
    "confidence": 0.0-1.0,
    "primary_concern": "Main clinical concern",
    "differential_diagnoses": ["Diagnosis 1", "Diagnosis 2", "Diagnosis 3"],
    "action_plan": "Specific action in English",
    "action_plan_urdu": "Specific action in Urdu",
    "time_sensitivity": "How quickly action is needed",
    "reasoning": "Clinical reasoning for this assessment",
    "red_flags_present": ["List of red flags identified"],
    "vital_signs_needed": ["BP", "HR", "Temp", etc.]
}}
```"""


# =============================================================================
# EMERGENCY DETECTION PROMPT (For Guardian Node)
# =============================================================================

EMERGENCY_DETECTION_PROMPT = """You are an EMERGENCY DETECTOR. Your ONLY job is to identify life-threatening conditions.

## CRITICAL PATTERNS TO DETECT

### IMMEDIATE EMERGENCIES (Call 1122 NOW):

1. **CARDIAC ARREST SIGNS:**
   - "Not breathing", "No pulse", "Collapsed", "Unresponsive"
   
2. **HEART ATTACK (MI):**
   - Chest pain + (arm pain OR jaw pain OR sweating OR nausea)
   - "Crushing", "Pressure", "Elephant on chest"
   
3. **STROKE:**
   - Face drooping + Arm weakness + Speech difficulty
   - Sudden confusion, trouble walking, severe headache
   
4. **BRAIN BLEED (SAH):**
   - "Worst headache ever", "Thunderclap headache"
   - Sudden severe headache + neck stiffness
   
5. **BREATHING EMERGENCY:**
   - "Can't breathe", "Choking", "Turning blue"
   - Severe shortness of breath + leg swelling (PE)
   
6. **SEVERE BLEEDING:**
   - "Won't stop bleeding", "Blood everywhere"
   - Trauma with significant blood loss

7. **ANAPHYLAXIS:**
   - Allergic reaction + throat swelling + difficulty breathing

8. **SEIZURE:**
   - Currently seizing, multiple seizures, first seizure

### VERY URGENT (ER immediately, not 1122):

1. **DIABETIC CRISIS:**
   - Thirst + frequent urination + confusion
   - Fruity breath + rapid breathing
   
2. **SEVERE INFECTION/SEPSIS:**
   - High fever + confusion + rapid breathing
   
3. **SEVERE ABDOMINAL PAIN:**
   - Sudden severe pain + vomiting + rigid abdomen

## CURRENT PATIENT MESSAGE:
"{message}"

## CURRENT SYMPTOMS:
{symptoms}

## TASK:
Determine if this is an emergency requiring immediate action.

Response format:
```json
{{
    "is_emergency": true/false,
    "confidence": 0.0-1.0,
    "pattern_detected": "Pattern name or null",
    "matched_symptoms": ["symptom1", "symptom2"],
    "recommended_action": "What to do",
    "triage_level": "IMMEDIATE|VERY_URGENT|URGENT|STANDARD|NON_URGENT",
    "reasoning": "Why this is/isn't an emergency"
}}
```"""



TRIAGE_ACTIONS = {
TriageLevel.IMMEDIATE: {
SupportedLanguage.ENGLISH: "🚨 EMERGENCY! Call 1122 immediately. Do not wait. Do not drive yourself.",
SupportedLanguage.URDU: "🚨 ایمرجنسی! ابھی 1122 کال کریں۔ انتظار نہ کریں۔ خود گاڑی نہ چلائیں۔",
SupportedLanguage.PUNJABI: "🚨 ایمرجنسی! ہنے 1122 کال کرو۔ انتظار نہ کرو۔ آپے گڈی نہ چلاؤ۔",
SupportedLanguage.SINDHI: "🚨 ايمرجنسي! هاڻي 1122 ڪال ڪريو۔ انتظار نه ڪريو۔",
SupportedLanguage.PASHTO: "🚨 بیړنی حالت! اوس 1122 زنګ ووهئ۔ انتظار مه کوئ۔",
SupportedLanguage.ROMAN_URDU: "🚨 EMERGENCY! Abhi 1122 call karein. Intezar na karein. Khud gari na chalayin."
},
TriageLevel.VERY_URGENT: {
SupportedLanguage.ENGLISH: "⚠️ VERY URGENT! Go to Emergency Room immediately. Have someone drive you.",
SupportedLanguage.URDU: "⚠️ بہت ضروری! فوری طور پر ایمرجنسی روم جائیں۔ کسی کو ساتھ لے جائیں۔",
SupportedLanguage.PUNJABI: "⚠️ بوہت ضروری! ہنے ایمرجنسی روم جاؤ۔ کسے نوں نال لے جاؤ۔",
SupportedLanguage.SINDHI: "⚠️ تمام ضروري! فوري طور تي ايمرجنسي روم وڃو۔",
SupportedLanguage.PASHTO: "⚠️ ډیر ضروري! سمدلاسه ایمرجنسي ته لاړ شئ۔",
SupportedLanguage.ROMAN_URDU: "⚠️ BOHT ZAROORI! Fori tor par Emergency Room jayen. Kisi ko sath le jayen."
},
TriageLevel.URGENT: {
SupportedLanguage.ENGLISH: "⚡ URGENT: Visit Emergency Room or Urgent Care within 2 hours.",
SupportedLanguage.URDU: "⚡ ضروری: 2 گھنٹے کے اندر ایمرجنسی یا ارجنٹ کیئر جائیں۔",
SupportedLanguage.PUNJABI: "⚡ ضروری: 2 گھنٹے اندر ایمرجنسی یا ارجنٹ کیئر جاؤ۔",
SupportedLanguage.SINDHI: "⚡ ضروري: 2 ڪلاڪن ۾ ايمرجنسي يا ارجنٽ ڪيئر وڃو۔",
SupportedLanguage.PASHTO: "⚡ ضروري: د 2 ساعتونو دننه ایمرجنسي ته لاړ شئ۔",
SupportedLanguage.ROMAN_URDU: "⚡ ZAROORI: 2 ghanton ke andar Emergency ya Urgent Care jayen."
},
TriageLevel.STANDARD: {
SupportedLanguage.ENGLISH: "📋 Schedule an appointment with your doctor within 24-48 hours.",
SupportedLanguage.URDU: "📋 24-48 گھنٹوں میں اپنے ڈاکٹر سے ملاقات کا وقت لیں۔",
SupportedLanguage.PUNJABI: "📋 24-48 گھنٹیاں وچ ڈاکٹر نال ملاقات دا ویلا لؤ۔",
SupportedLanguage.SINDHI: "📋 24-48 ڪلاڪن ۾ پنهنجي ڊاڪٽر سان ملاقات ڪريو۔",
SupportedLanguage.PASHTO: "📋 په 24-48 ساعتونو کې د ډاکټر سره لیدنه وټاکئ۔",
SupportedLanguage.ROMAN_URDU: "📋 24-48 ghanton mein apne doctor se mulaqat ka waqt lein."
},
TriageLevel.NON_URGENT: {
SupportedLanguage.ENGLISH: "ℹ️ You may see a doctor at your convenience or use telemedicine.",
SupportedLanguage.URDU: "ℹ️ آپ اپنی سہولت کے مطابق ڈاکٹر سے مل سکتے ہیں یا ٹیلی میڈیسن استعمال کر سکتے ہیں۔",
SupportedLanguage.PUNJABI: "ℹ️ تسیں اپنی سہولت نال ڈاکٹر نوں مل سکدے ہو۔",
SupportedLanguage.SINDHI: "ℹ️ توهان پنهنجي سهولت سان ڊاڪٽر سان ملي سگهو ٿا۔",
SupportedLanguage.PASHTO: "ℹ️ تاسو کولی شئ خپل اسانتیا سره ډاکټر وګورئ۔",
SupportedLanguage.ROMAN_URDU: "ℹ️ Aap apni sahulat ke mutabiq doctor se mil sakte hain."
}
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_triage_prompt(
    symptoms: str,
    symptom_details: str = "",
    duration: str = "Not specified",
    red_flags: str = "None",
    age: str = "Unknown",
    gender: str = "Unknown",
    medical_history: str = "Not provided",
    conversation_summary: str = ""
) -> str:
    """
    Build a complete triage assessment prompt.
    
    Args:
        symptoms: Comma-separated list of symptoms
        symptom_details: Detailed symptom information
        duration: How long symptoms have been present
        red_flags: Previously identified red flags
        age: Patient age
        gender: Patient gender
        medical_history: Relevant medical history
        conversation_summary: Summary of the interview
        
    Returns:
        Complete prompt string for triage assessment
    """
    return TRIAGE_ASSESSMENT_PROMPT.format(
        symptoms=symptoms,
        symptom_details=symptom_details,
        duration=duration,
        red_flags=red_flags,
        age=age,
        gender=gender,
        medical_history=medical_history,
        conversation_summary=conversation_summary
    )


def get_triage_action_message(
    triage_level: TriageLevel,
    language: SupportedLanguage = SupportedLanguage.URDU
) -> str:
    """
    Get the action message for a triage level in the specified language.
    
    Args:
        triage_level: The assigned triage level
        language: The patient's preferred language
        
    Returns:
        Action message in the appropriate language
    """
    level_actions = TRIAGE_ACTIONS.get(triage_level, TRIAGE_ACTIONS[TriageLevel.STANDARD])
    return level_actions.get(language, level_actions[SupportedLanguage.ENGLISH])
def get_emergency_detection_prompt(message: str, symptoms: str) -> str:
    """
    Build prompt for emergency detection.
    
    Args:
        message: The patient's latest message
        symptoms: Currently identified symptoms
        
    Returns:
        Complete prompt for emergency detection
    """
    return EMERGENCY_DETECTION_PROMPT.format(
        message=message,
        symptoms=symptoms
    )

# =============================================================================
# TRIAGE DECISION TREE (For programmatic use)
# =============================================================================
TRIAGE_DECISION_TREE = {
"chest_pain": {
"with_radiation": {
"with_sweating": TriageLevel.IMMEDIATE, # MI
"without_sweating": TriageLevel.VERY_URGENT
},
"without_radiation": {
"severe": TriageLevel.VERY_URGENT,
"moderate": TriageLevel.URGENT,
"mild": TriageLevel.STANDARD
}
},
"headache": {
"thunderclap_onset": TriageLevel.IMMEDIATE, # SAH
"with_neck_stiffness": TriageLevel.IMMEDIATE, # Meningitis/SAH
"with_fever": TriageLevel.VERY_URGENT,
"severe": TriageLevel.URGENT,
"moderate": TriageLevel.STANDARD,
"mild": TriageLevel.NON_URGENT
},
"breathing_difficulty": {
"severe": TriageLevel.IMMEDIATE,
"with_leg_swelling": TriageLevel.IMMEDIATE, # PE
"moderate": TriageLevel.VERY_URGENT,
"mild": TriageLevel.URGENT
},
"abdominal_pain": {
"severe_sudden": TriageLevel.VERY_URGENT,
"right_lower_with_fever": TriageLevel.URGENT, # Appendicitis
"moderate": TriageLevel.URGENT,
"mild": TriageLevel.STANDARD
},
"fatigue": {
"with_thirst_and_urination": TriageLevel.VERY_URGENT, # Diabetes
"with_fever": TriageLevel.URGENT,
"chronic": TriageLevel.STANDARD
}
}