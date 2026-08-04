"""
Translator Agent Prompts
========================
Prompts for multilingual translation in healthcare context.

This is CRITICAL for the Gold Tier requirement:
- Voice/Multilingual support
- Urdu, Punjabi, Sindhi, Pashto support

Translation Strategy:
1. Patient speaks in native language → Translate to English for processing
2. AI responds in English → Translate back to patient's language
3. Medical terminology must be accurate and culturally appropriate

IMPORTANT: Medical translation requires precision. Mistranslation can be dangerous.
"""

from typing import Dict, List, Optional
from src.config import SupportedLanguage


# =============================================================================
# CORE TRANSLATION SYSTEM PROMPT
# =============================================================================

TRANSLATION_SYSTEM_PROMPT = """You are a Medical Translation Specialist with expertise in healthcare communication.

## YOUR ROLE
- Translate medical conversations accurately between languages
- Preserve medical meaning precisely
- Adapt cultural context appropriately
- Maintain empathetic tone

## CRITICAL RULES

### 1. MEDICAL ACCURACY IS PARAMOUNT
- Never approximate medical terms
- When uncertain, keep the original term and add explanation
- Symptoms must be translated precisely
- Numbers (dosages, vitals, scales) must be exact

### 2. CULTURAL SENSITIVITY
- Use appropriate honorifics (آپ vs تم)
- Adapt formality level to the culture
- Be aware of health stigmas in the culture
- Use gender-appropriate language

### 3. CLARITY OVER ELEGANCE
- Simple, clear language is better than eloquent but confusing
- Use common terms patients will understand
- Avoid overly technical jargon unless necessary

### 4. PRESERVE EMOTIONAL TONE
- If patient sounds distressed, translation should reflect that
- If message is urgent, urgency should be clear
- Empathetic phrases should remain empathetic

## SUPPORTED LANGUAGES
- English (en)
- Urdu (ur) - اردو
- Punjabi (pa) - پنجابی
- Sindhi (sd) - سنڌي
- Pashto (ps) - پښتو

## OUTPUT FORMAT
Always provide:
1. Translated text
2. Confidence score (0-1)
3. Any terms that may need clarification
4. Cultural notes if relevant"""


# =============================================================================
# MEDICAL TRANSLATION PROMPT
# =============================================================================

MEDICAL_TRANSLATION_PROMPT = """## TRANSLATION TASK

**Source Language:** {source_language}
**Target Language:** {target_language}
**Context:** Medical patient interview

**Original Text:**
{original_text}

**Medical Context:**
{medical_context}

---

## INSTRUCTIONS

Translate the text while:
1. Preserving exact medical meaning
2. Using culturally appropriate phrasing
3. Maintaining the emotional tone
4. Keeping any numbers/measurements exact

**Output Format:**
```json
{{
    "translated_text": "The translation",
    "confidence": 0.0-1.0,
    "medical_terms": [
        {{"original": "term", "translated": "translation", "note": "any clarification"}}
    ],
    "cultural_notes": "Any cultural adaptations made",
    "back_translation": "Back-translation to verify accuracy (optional)"
}}
```"""


# =============================================================================
# SYMPTOM TRANSLATION DICTIONARY
# =============================================================================

MEDICAL_TERMS_DICTIONARY = {
    # Common Symptoms
    "chest pain": {
        SupportedLanguage.URDU: "سینے میں درد",
        SupportedLanguage.PUNJABI: "چھاتی وچ درد",
        SupportedLanguage.SINDHI: "سيني ۾ درد",
        SupportedLanguage.PASHTO: "د سینې درد"
    },
    "headache": {
        SupportedLanguage.URDU: "سر درد",
        SupportedLanguage.PUNJABI: "سر درد",
        SupportedLanguage.SINDHI: "مٿي جو درد",
        SupportedLanguage.PASHTO: "د سر درد"
    },
    "fever": {
        SupportedLanguage.URDU: "بخار",
        SupportedLanguage.PUNJABI: "بخار / تاپ",
        SupportedLanguage.SINDHI: "تاپ / بخار",
        SupportedLanguage.PASHTO: "تبه"
    },
    "fatigue": {
        SupportedLanguage.URDU: "تھکاوٹ / کمزوری",
        SupportedLanguage.PUNJABI: "تھکاوٹ / کمزوری",
        SupportedLanguage.SINDHI: "ٿڪجڻ",
        SupportedLanguage.PASHTO: "ستړیا"
    },
    "shortness of breath": {
        SupportedLanguage.URDU: "سانس لینے میں دشواری",
        SupportedLanguage.PUNJABI: "ساہ لین وچ مشکل",
        SupportedLanguage.SINDHI: "ساهه کڻڻ ۾ ڏکيائي",
        SupportedLanguage.PASHTO: "د ساه نیولو کې ستونزه"
    },
    "nausea": {
        SupportedLanguage.URDU: "متلی / جی متلانا",
        SupportedLanguage.PUNJABI: "متلی / اُبکائی",
        SupportedLanguage.SINDHI: "متلي",
        SupportedLanguage.PASHTO: "زړه بدیدل"
    },
    "vomiting": {
        SupportedLanguage.URDU: "الٹی / قے",
        SupportedLanguage.PUNJABI: "الٹی",
        SupportedLanguage.SINDHI: "اُلٽي",
        SupportedLanguage.PASHTO: "کانګل"
    },
    "diarrhea": {
        SupportedLanguage.URDU: "دست / پیچش",
        SupportedLanguage.PUNJABI: "دست / ٹٹیاں",
        SupportedLanguage.SINDHI: "جلاب",
        SupportedLanguage.PASHTO: "نرۍ"
    },
    "abdominal pain": {
        SupportedLanguage.URDU: "پیٹ میں درد",
        SupportedLanguage.PUNJABI: "ڈھڈ وچ درد / پیٹ درد",
        SupportedLanguage.SINDHI: "پيٽ ۾ درد",
        SupportedLanguage.PASHTO: "د ګېډې درد"
    },
    "dizziness": {
        SupportedLanguage.URDU: "چکر آنا",
        SupportedLanguage.PUNJABI: "چکر آنا / گھمیری",
        SupportedLanguage.SINDHI: "چڪر اچڻ",
        SupportedLanguage.PASHTO: "سر ګرځیدل"
    },
    "excessive thirst": {
        SupportedLanguage.URDU: "بہت زیادہ پیاس",
        SupportedLanguage.PUNJABI: "بوہت زیادہ پیاس",
        SupportedLanguage.SINDHI: "تمام گهڻي تير",
        SupportedLanguage.PASHTO: "ډیره تږی"
    },
    "frequent urination": {
        SupportedLanguage.URDU: "بار بار پیشاب آنا",
        SupportedLanguage.PUNJABI: "وار وار پیشاب آنا",
        SupportedLanguage.SINDHI: "بار بار پيشاب اچڻ",
        SupportedLanguage.PASHTO: "ډیر ځله تشناب ته تلل"
    },
    "weight loss": {
        SupportedLanguage.URDU: "وزن کم ہونا",
        SupportedLanguage.PUNJABI: "وزن گھٹنا",
        SupportedLanguage.SINDHI: "وزن گهٽجڻ",
        SupportedLanguage.PASHTO: "د وزن کمیدل"
    },
    "sweating": {
        SupportedLanguage.URDU: "پسینہ آنا",
        SupportedLanguage.PUNJABI: "پسینہ آنا / مُڑکا",
        SupportedLanguage.SINDHI: "پگھر اچڻ",
        SupportedLanguage.PASHTO: "خولې راتلل"
    },
    "cough": {
        SupportedLanguage.URDU: "کھانسی",
        SupportedLanguage.PUNJABI: "کھنگھی / کھانسی",
        SupportedLanguage.SINDHI: "ڪھنگھي",
        SupportedLanguage.PASHTO: "ټوخی"
    },
    "sore throat": {
        SupportedLanguage.URDU: "گلے میں خراش / گلا خراب",
        SupportedLanguage.PUNJABI: "گلے وچ خراش",
        SupportedLanguage.SINDHI: "ڳلي ۾ خراش",
        SupportedLanguage.PASHTO: "ستونی درد"
    },
    
    # Diabetes-related (Case Study 1)
    "diabetes": {
        SupportedLanguage.URDU: "ذیابیطس / شوگر",
        SupportedLanguage.PUNJABI: "شوگر / ذیابیطس",
        SupportedLanguage.SINDHI: "شوگر / ذيابيطس",
        SupportedLanguage.PASHTO: "د شکرې ناروغي"
    },
    "blood sugar": {
        SupportedLanguage.URDU: "خون میں شکر / بلڈ شوگر",
        SupportedLanguage.PUNJABI: "خون وچ شکر",
        SupportedLanguage.SINDHI: "رت ۾ شڪر",
        SupportedLanguage.PASHTO: "د وینې بورې"
    },
    "insulin": {
        SupportedLanguage.URDU: "انسولین",
        SupportedLanguage.PUNJABI: "انسولین",
        SupportedLanguage.SINDHI: "انسولين",
        SupportedLanguage.PASHTO: "انسولین"
    },
    
    # Cardiac-related (Case Study 2)
    "heart attack": {
        SupportedLanguage.URDU: "دل کا دورہ",
        SupportedLanguage.PUNJABI: "دل دا دورہ",
        SupportedLanguage.SINDHI: "دل جو دورو",
        SupportedLanguage.PASHTO: "د زړه حمله"
    },
    "stroke": {
        SupportedLanguage.URDU: "فالج",
        SupportedLanguage.PUNJABI: "فالج",
        SupportedLanguage.SINDHI: "فالج",
        SupportedLanguage.PASHTO: "فالج"
    },
    "blood pressure": {
        SupportedLanguage.URDU: "بلڈ پریشر",
        SupportedLanguage.PUNJABI: "بلڈ پریشر",
        SupportedLanguage.SINDHI: "بلڊ پريشر",
        SupportedLanguage.PASHTO: "د وینې فشار"
    },
    
    # Emergency terms
    "emergency": {
        SupportedLanguage.URDU: "ایمرجنسی / ہنگامی",
        SupportedLanguage.PUNJABI: "ایمرجنسی / فوری",
        SupportedLanguage.SINDHI: "ايمرجنسي",
        SupportedLanguage.PASHTO: "بیړنی حالت"
    },
    "call ambulance": {
        SupportedLanguage.URDU: "ایمبولینس بلائیں / 1122 کال کریں",
        SupportedLanguage.PUNJABI: "ایمبولینس بلاؤ / 1122 کال کرو",
        SupportedLanguage.SINDHI: "ايمبولنس سڏيو / 1122 ڪال ڪريو",
        SupportedLanguage.PASHTO: "امبولانس وغواړئ / 1122 زنګ ووهئ"
    }
}


# =============================================================================
# COMMON PHRASES DICTIONARY
# =============================================================================

COMMON_PHRASES = {
    # Greetings
    "hello_greeting": {
        SupportedLanguage.ENGLISH: "Hello! I'm your AI health assistant.",
        SupportedLanguage.URDU: "السلام علیکم! میں آپ کا AI صحت معاون ہوں۔",
        SupportedLanguage.PUNJABI: "سلام! میں تہاڈا AI صحت معاون آں۔",
        SupportedLanguage.SINDHI: "السلام عليڪم! مان توهان جو AI صحت معاون آهيان۔",
        SupportedLanguage.PASHTO: "سلام! زه ستاسو د روغتیا AI مرستندوی یم۔"
    },
    
    # Questions
    "how_can_i_help": {
        SupportedLanguage.ENGLISH: "How can I help you today?",
        SupportedLanguage.URDU: "آج میں آپ کی کیا مدد کر سکتا ہوں؟",
        SupportedLanguage.PUNJABI: "اج میں تہاڈی کی مدد کر سکدا آں؟",
        SupportedLanguage.SINDHI: "اڄ مان توهان جي ڪهڙي مدد ڪري سگهان ٿو؟",
        SupportedLanguage.PASHTO: "نن زه ستاسو څه مرسته کولی شم؟"
    },
    "how_long_symptom": {
        SupportedLanguage.ENGLISH: "How long have you been experiencing this?",
        SupportedLanguage.URDU: "آپ کو یہ کتنے عرصے سے ہو رہا ہے؟",
        SupportedLanguage.PUNJABI: "تہانوں ایہ کنے چر توں ہو رہا اے؟",
        SupportedLanguage.SINDHI: "توهان کي اهو ڪيتري دير کان ٿي رهيو آهي؟",
        SupportedLanguage.PASHTO: "تاسو له څو وخت راهیسې دا تجربه کوئ؟"
    },
    "scale_1_to_10": {
        SupportedLanguage.ENGLISH: "On a scale of 1 to 10, how severe is it?",
        SupportedLanguage.URDU: "1 سے 10 کے پیمانے پر، یہ کتنا شدید ہے؟",
        SupportedLanguage.PUNJABI: "1 توں 10 دے پیمانے تے، ایہ کنا شدید اے؟",
        SupportedLanguage.SINDHI: "1 کان 10 جي پيماني تي، اهو ڪيترو شديد آهي؟",
        SupportedLanguage.PASHTO: "د 1 څخه تر 10 پورې، دا څومره شدید دی؟"
    },
    "any_other_symptoms": {
        SupportedLanguage.ENGLISH: "Are you experiencing any other symptoms?",
        SupportedLanguage.URDU: "کیا آپ کو کوئی اور علامات ہیں؟",
        SupportedLanguage.PUNJABI: "کیا تہانوں ہور کوئی علامات نے؟",
        SupportedLanguage.SINDHI: "ڇا توهان کي ٻيا ڪي علامتون آهن؟",
        SupportedLanguage.PASHTO: "ایا تاسو نورې نښې لرئ؟"
    },
    
    # Responses
    "i_understand": {
        SupportedLanguage.ENGLISH: "I understand.",
        SupportedLanguage.URDU: "میں سمجھ گیا۔",
        SupportedLanguage.PUNJABI: "میں سمجھ گیا۔",
        SupportedLanguage.SINDHI: "مان سمجھي ويس۔",
        SupportedLanguage.PASHTO: "زه پوهیدم۔"
    },
    "thank_you": {
        SupportedLanguage.ENGLISH: "Thank you for sharing this information.",
        SupportedLanguage.URDU: "یہ معلومات شیئر کرنے کا شکریہ۔",
        SupportedLanguage.PUNJABI: "ایہ معلومات دسن دا شکریہ۔",
        SupportedLanguage.SINDHI: "هي معلومات ڏيڻ جو مهرباني۔",
        SupportedLanguage.PASHTO: "د دې معلوماتو شریکولو لپاره مننه۔"
    },
    
    # Emergency
    "call_emergency": {
        SupportedLanguage.ENGLISH: "Please call 1122 immediately!",
        SupportedLanguage.URDU: "براہ کرم فوری طور پر 1122 کال کریں!",
        SupportedLanguage.PUNJABI: "مہربانی نال ہنے ای 1122 کال کرو!",
        SupportedLanguage.SINDHI: "مهرباني ڪري فوري طور تي 1122 ڪال ڪريو!",
        SupportedLanguage.PASHTO: "مهرباني وکړئ سمدلاسه 1122 زنګ ووهئ!"
    },
    "go_to_er": {
        SupportedLanguage.ENGLISH: "Please go to the Emergency Room immediately.",
        SupportedLanguage.URDU: "براہ کرم فوری طور پر ایمرجنسی روم جائیں۔",
        SupportedLanguage.PUNJABI: "مہربانی نال ہنے ایمرجنسی روم جاؤ۔",
        SupportedLanguage.SINDHI: "مهرباني ڪري فوري طور تي ايمرجنسي روم وڃو۔",
        SupportedLanguage.PASHTO: "مهرباني وکړئ سمدلاسه ایمرجنسي ته لاړ شئ۔"
    },
    
    # Closing
    "summary_intro": {
        SupportedLanguage.ENGLISH: "Let me summarize what you've told me:",
        SupportedLanguage.URDU: "آپ نے جو بتایا اس کا خلاصہ:",
        SupportedLanguage.PUNJABI: "تسیں جو دسیا اوہدا خلاصہ:",
        SupportedLanguage.SINDHI: "توهان جيڪو ٻڌايو ان جو خلاصو:",
        SupportedLanguage.PASHTO: "تاسو څه وویل هغه لنډیز:"
    },
    "doctor_will_review": {
        SupportedLanguage.ENGLISH: "A doctor will review this information.",
        SupportedLanguage.URDU: "ڈاکٹر اس معلومات کا جائزہ لے گا۔",
        SupportedLanguage.PUNJABI: "ڈاکٹر ایہ معلومات دیکھے گا۔",
        SupportedLanguage.SINDHI: "ڊاڪٽر هي معلومات ڏسندو۔",
        SupportedLanguage.PASHTO: "ډاکټر به دا معلومات وګوري۔"
    },
    "ai_disclaimer": {
        SupportedLanguage.ENGLISH: "This AI assessment is a support tool and does not replace professional medical advice.",
        SupportedLanguage.URDU: "یہ AI تشخیص ایک معاون ٹول ہے اور پیشہ ورانہ طبی مشورے کی جگہ نہیں لیتا۔",
        SupportedLanguage.PUNJABI: "ایہ AI تشخیص اک مددگار آلہ اے تے ڈاکٹر دی سلاح دی تھاں نئیں لیندا۔",
        SupportedLanguage.SINDHI: "هي AI تشخيص هڪ مددگار اوزار آهي ۽ ڊاڪٽري صلاح جي جاءِ نٿو وٺي۔",
        SupportedLanguage.PASHTO: "دا AI ارزونه یو مرستندویه وسیله ده او د ډاکټر مشورې ځای نه نیسي۔"
    }
}


# =============================================================================
# LANGUAGE DETECTION PATTERNS
# =============================================================================

LANGUAGE_PATTERNS = {
    SupportedLanguage.URDU: [
        "میں", "ہے", "کو", "کا", "کی", "ہیں", "ہوں", "آپ", "یہ", "وہ",
        "ہو", "نہیں", "سے", "پر", "اور", "کے", "تھا", "تھی", "ہوا", "ہوئی"
    ],
    SupportedLanguage.PUNJABI: [
        "میں", "اے", "نوں", "دا", "دی", "نے", "آں", "تسیں", "ایہ", "اوہ",
        "ہو", "نئیں", "توں", "تے", "تے", "دے", "سی", "ہویا", "ہوئی", "وچ"
    ],
    SupportedLanguage.SINDHI: [
        "مان", "آهي", "کي", "جو", "جي", "آهن", "آهيان", "توهان", "هي", "اهو",
        "ٿي", "ناهي", "کان", "تي", "۽", "جا", "هو", "ٿي", "ٿيو", "۾"
    ],
    SupportedLanguage.PASHTO: [
        "زه", "دی", "ته", "د", "دې", "دي", "یم", "تاسو", "دا", "هغه",
        "شي", "نه", "له", "په", "او", "چې", "وو", "شوه", "شو", "کې"
    ]
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_translation_prompt(
    original_text: str,
    source_language: SupportedLanguage,
    target_language: SupportedLanguage,
    medical_context: str = ""
) -> str:
    """
    Build a translation prompt.
    
    Args:
        original_text: Text to translate
        source_language: Source language
        target_language: Target language
        medical_context: Additional medical context
        
    Returns:
        Formatted translation prompt
    """
    return MEDICAL_TRANSLATION_PROMPT.format(
        source_language=source_language.value,
        target_language=target_language.value,
        original_text=original_text,
        medical_context=medical_context if medical_context else "General patient interview"
    )


def get_medical_term(
    term: str,
    target_language: SupportedLanguage
) -> Optional[str]:
    """
    Get the translation of a medical term.
    
    Args:
        term: English medical term
        target_language: Target language
        
    Returns:
        Translated term or None if not found
    """
    term_lower = term.lower()
    if term_lower in MEDICAL_TERMS_DICTIONARY:
        return MEDICAL_TERMS_DICTIONARY[term_lower].get(target_language)
    return None


def get_phrase(
    phrase_key: str,
    language: SupportedLanguage
) -> str:
    """
    Get a common phrase in the specified language.
    
    Args:
        phrase_key: Key for the phrase
        language: Target language
        
    Returns:
        Phrase in the specified language, or English fallback
    """
    if phrase_key in COMMON_PHRASES:
        phrases = COMMON_PHRASES[phrase_key]
        return phrases.get(language, phrases.get(SupportedLanguage.ENGLISH, ""))
    return ""


def detect_language(text: str) -> SupportedLanguage:
    """
    Detect the language of input text.
    
    Args:
        text: Input text
        
    Returns:
        Detected language (defaults to English if uncertain)
    """
    # Check for script-based detection first
    for char in text:
        # Arabic script (used by Urdu, Punjabi, Sindhi, Pashto)
        if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F':
            # Need further analysis for specific language
            break
    else:
        # No Arabic script found, assume English
        return SupportedLanguage.ENGLISH
    
    # Count pattern matches for each language
    scores = {}
    text_words = text.split()
    
    for lang, patterns in LANGUAGE_PATTERNS.items():
        score = sum(1 for word in text_words if word in patterns)
        scores[lang] = score
    
    # Return language with highest score, or Urdu as default for Arabic script
    if scores:
        best_lang = max(scores, key=scores.get)
        if scores[best_lang] > 0:
            return best_lang
    
    return SupportedLanguage.URDU  # Default for Arabic script


def translate_symptom_list(
    symptoms: List[str],
    target_language: SupportedLanguage
) -> List[Dict[str, str]]:
    """
    Translate a list of symptoms.
    
    Args:
        symptoms: List of symptom names in English
        target_language: Target language
        
    Returns:
        List of dicts with 'english' and 'translated' keys
    """
    translations = []
    for symptom in symptoms:
        translated = get_medical_term(symptom, target_language)
        translations.append({
            "english": symptom,
            "translated": translated or symptom  # Keep original if no translation
        })
    return translations