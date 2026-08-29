---
title: MediCare AI
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🏥 MediCare AI

### Autonomous Clinical Intelligence That Saves Lives

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow.svg)](https://itxmjr-medicare-ai.hf.space)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Enabled-purple.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hackathon](https://img.shields.io/badge/AI%20Hackathon-2025-red.svg)](#)
[![Tier](https://img.shields.io/badge/Tier-🥇%20Gold-gold.svg)](#gold-tier-features)

<p align="center">
  <img src="docs/assets/banner.png" alt="MediCare AI Banner" width="800"/>
</p>

**🌍 Multilingual** • **🎤 Voice-Enabled** • **📱 WhatsApp Ready** • **🚨 Real-time Triage**

[Live Demo](https://itxmjr-medicare-ai.hf.space) • [Documentation](#-documentation) • [Installation](#-quick-start) • [API Reference](#-api-reference)

---

</div>

## 📋 Table of Contents

- [The Problem](#-the-problem)
- [Our Solution](#-our-solution)
- [Key Features](#-key-features)
- [Architecture](#-system-architecture)
- [Case Studies](#-case-studies)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Testing](#-testing)
- [Roadmap](#-roadmap)
- [Team](#-team)
- [License](#-license)

---

## 🔴 The Problem

<table>
<tr>
<td width="50%">

### The Story of Fatima (Lahore, 2:00 AM)

> Fatima wakes with crushing chest pain. She calls a helpline. An untrained operator says, *"Come in if you think it's serious."* She waits until 6 AM. Doctors diagnose **acute appendicitis hours before rupture**.

</td>
<td width="50%">

### The Story of Dr. Ahmed (Rural Sindh, 9:00 AM)

> Dr. Ahmed sees **100 patients daily** (3 minutes each). A farmer says, *"I'm tired and thirsty."* Dr. Ahmed prescribes vitamins. Three weeks later: **diabetic ketoacidosis, blood glucose 450 mg/dL**.

</td>
</tr>
</table>

### 📊 The Healthcare Gap in Pakistan

| Metric | Reality |
|--------|---------|
| 👨‍⚕️ **Doctor-to-Patient Ratio** | 1 doctor per 1,000 people |
| ⏱️ **Average Consultation Time** | 2-3 minutes |
| ❌ **Medical Errors from Communication** | 50-70% |
| 🌐 **Language Barriers** | English/Urdu vs. Punjabi/Pashto/Sindhi |

---

## 💡 Our Solution

<div align="center">

### What if AI could conduct thorough 15-minute interviews in the patient's language, identify red flags, and hand doctors complete, structured information before they even enter the room?

</div>
```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   🗣️ Patient            🤖 AI System              👨‍⚕️ Doctor               │
│                                                                             │
│  "مجھے تھکاوٹ ہے"  ──────►  ┌──────────────────┐  ──────►   SOAP Note      │
│   (I'm tired)             │  • Interviews    │           + Triage          │
│                           │  • Detects Flags │           + Red Flags       │
│   Voice/Text   ◄────────────  • Triages       │                             │
│    in Urdu                │  • Summarizes    │                             │
│                           └──────────────────┘                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```


### 🎯 System Boundaries

| ✅ What This System IS | ❌ What This System IS NOT |
|------------------------|---------------------------|
| Clinical decision-support tool | A diagnostic system |
| Intelligent medical intake assistant | A treatment advisor |
| Triage support system | A replacement for doctors |
| Information gatherer for physicians | A prescription generator |

---

## ✨ Key Features

### 🥉 Bronze Tier
- ✅ Handles **ONE clinical scenario**
- ✅ **5-10 adaptive questions**
- ✅ Generates **SOAP summary**

### 🥈 Silver Tier
- ✅ Handles **3+ scenarios**
- ✅ **Multi-agent architecture**
- ✅ **Red-flag detection** logic

### 🥇 Gold Tier *(Our Implementation)*
- ✅ **Voice/Multilingual** support (Urdu, Punjabi, Sindhi, Pashto, English)
- ✅ **Real-time explainability** (reasoning traces)
- ✅ **Integration-ready** (WhatsApp API, REST API, Voice)

---

## 🏗️ System Architecture
```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          HEALTHCARE AI SYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐  │
│  │  WhatsApp   │       │    Voice    │       │  REST API   │       │   Web UI    │  │
│  │   Webhook   │       │    Input    │       │   Clients   │       │  (Future)   │  │
│  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘       └──────┬──────┘  │
│         │                     │                     │                     │         │
│         └───────────────────┴───────────────────┴───────────────────┘         │
│                                       │                                         │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                       🔄 API GATEWAY (FastAPI)                          │  │
│  │              Authentication • Rate Limiting • Logging                   │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                       │                                         │
│            ┌───────────────────────────┼───────────────────────────┐            │
│            ▼                           ▼                           ▼            │
│      ┌─────────────┐             ┌─────────────┐             ┌─────────────┐      │
│      │   🎤 Voice  │             │   🌐 Trans- │             │   💬 Chat   │      │
│      │   Service   │             │   lator     │             │   Handler   │      │
│      │  (STT / TTS)│             │   Service   │             │             │      │
│      └──────┬──────┘             └──────┬──────┘             └──────┬──────┘      │
│             │                           │                           │             │
│             └─────────────────────────┴─────────────────────────┘             │
│                                       │                                         │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                   🧠 LANGGRAPH WORKFLOW ENGINE                           │  │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │   │                        📊 PATIENT STATE                         │   │  │
│  │   │  • Demographics • Symptoms • Red Flags • Conversation           │   │  │
│  │   │  • Triage Level • SOAP Note • Reasoning Trace                   │   │  │
│  │   └─────────────────────────────────────────────────────────────────┘   │  │
│  │               │                     │                     │             │  │
│  │     ┌──────────────────────┼──────────────────────┐     │  │
│  │     ▼                      ▼                      ▼     │  │
│  │  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      │  │
│  │  │  🛡️ GUARDIAN │      │ 🩺 INTERVIEW │      │  🚨 TRIAGE  │      │  │
│  │  │     NODE    │      │     NODE    │      │    NODE     │      │  │
│  │  │             │      │             │      │             │      │  │
│  │  │   Red Flag  │      │   Adaptive  │      │   WHO ETAT  │      │  │
│  │  │  Detection  │      │   Questions │      │  Assessment │      │  │
│  │  │  (Parallel) │      │   OLDCARTS  │      │             │      │  │
│  │  └─────────────┘      └─────────────┘      └─────────────┘      │  │
│  │               │                     │                     │             │  │
│  │     └──────────────────────┴──────────────────────┘     │  │
│  │                             │                                   │  │
│  │                             ▼                                   │  │
│  │                      ┌─────────────────┐                        │  │
│  │                      │  📋 SUMMARIZER  │                        │  │
│  │                      │      NODE       │                        │  │
│  │                      │                 │                        │  │
│  │                      │   SOAP Note     │                        │  │
│  │                      │   Generation    │                        │  │
│  │                      └─────────────────┘                        │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                       │                                         │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                           🔧 TOOLS LAYER                                │  │
│  │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      │  │
│  │  │    📚 RAG   │      │ 🧮 Medical  │      │   ⏱️ Time    │      │  │
│  │  │    Search   │      │  Calculator │      │   Parser    │      │  │
│  │  │             │      │             │      │             │      │  │
│  │  │   Clinical  │      │   BMI, BP,  │      │  Duration   │      │  │
│  │  │  Guidelines │      │   Glucose   │      │  Analysis   │      │  │
│  │  └─────────────┘      └─────────────┘      └─────────────┘      │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                       │                                         │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                            💾 DATA LAYER                                │  │
│  │   ┌─────────────────────────┐         ┌─────────────────────────┐       │  │
│  │   │         ChromaDB        │         │    Medical Documents    │       │  │
│  │   │      (Vector Store)     │         │  • WHO Guidelines       │       │  │
│  │   │                         │         │  • Emergency Protocols  │       │  │
│  │   │      Embeddings for     │         │  • Drug Interactions    │       │  │
│  │   │      Semantic Search    │         │  • Symptom Database     │       │  │
│  │   └─────────────────────────┘         └─────────────────────────┘       │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```


### 🔄 Workflow Flow Diagram

```text
                                START
                                  │
                                  ▼
                        ┌─────────────────┐
                        │  Patient Input  │
                        │ (Text or Voice) │
                        └────────┬────────┘
                                 │
                     ┌───────────┴───────────┐
                     ▼                       ▼
              ┌─────────────┐         ┌─────────────┐
              │  Translate  │         │  Transcribe │
              │  to English │         │   (Voice)   │
              └──────┬──────┘         └──────┬──────┘
                     └───────────┬───────────┘
                                 ▼
     ┌───────────────────────────────────────────────────┐
     │                  🛡️ GUARDIAN NODE                  │
     │           (Runs on EVERY message)                  │
     │                                                    │
     │  • Pattern matching for emergencies                │
     │  • Keyword detection (multilingual)                │
     │  • LLM analysis for subtle patterns                │
     └───────────────────────┬───────────────────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
        ┌─────────────┐           ┌─────────────┐
        │ 🚨 EMERGENCY │           │ ✅ SAFE TO  │
        │  DETECTED!  │           │  CONTINUE   │
        └──────┬──────┘           └──────┬──────┘
               │                         │
               ▼                         ▼
     ┌─────────────────┐       ┌─────────────────┐
     │   IMMEDIATE     │       │   INTERVIEWER   │
     │   ESCALATION    │       │      NODE       │
     │                 │       │                 │
     │  "Call 1122!"   │       │  • OLDCARTS     │
     │                 │       │  • Adaptive Qs  │
     └────────┬────────┘       │  • Phase-based  │
              │                └────────┬────────┘
              │                         │
              │            ┌────────────┴────────────┐
              │            ▼                         ▼
              │     ┌─────────────┐         ┌─────────────┐
              │     │  Continue   │         │  Interview  │
              │     │  Interview  │         │  Complete   │
              │     └──────┬──────┘         └──────┬──────┘
              │            │                       │
              │            └───────┬───────────────┘
              │                    ▼
              │          ┌─────────────────┐
              │          │   TRIAGE NODE   │
              │          │                 │
              │          │  WHO ETAT:      │
              │          │  • IMMEDIATE    │
              │          │  • VERY URGENT  │
              │          │  • URGENT       │
              │          │  • STANDARD     │
              │          │  • NON-URGENT   │
              │          └────────┬────────┘
              │                   │
              └───────────────────┴───────────────────┐
                                                      ▼
                                           ┌─────────────────┐
                                           │   SUMMARIZER    │
                                           │      NODE       │
                                           │                 │
                                           │  Generate SOAP  │
                                           │  Note for MD    │
                                           └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │    TRANSLATE    │
                                           │   to Patient    │
                                           │    Language     │
                                           └────────┬────────┘
                                                    │
                                                    ▼
                                                  END
```

---

## 📚 Case Studies

### Case Study 1: The Diabetic Farmer 🌾

**Problem:** Short consultations lead to missed metabolic crises.

<table>
<tr>
<th width="50%">❌ Without AI (Current Reality)</th>
<th width="50%">✅ With Our AI System</th>
</tr>
<tr>
<td>

- 2-minute rushed consultation
- Language barrier (Punjabi vs Urdu)
- Symptoms dismissed as "fatigue"
- **Result:** Emergency 3 weeks later
- Diabetic ketoacidosis (glucose: 450 mg/dL)

</td>
<td>

- 15-minute thorough interview
- Conducted in native Punjabi
- Identified classic diabetes triad
- **Result:** Same-day diagnosis
- Immediate treatment initiated

</td>
</tr>
</table>

#### 🎬 Interview Flow Demonstration
```text
┌──────────────────────────────────────────────────────────────────┐
│                        INTERVIEW SESSION                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🤖 AI: "آپ کو تھکاوٹ کتنے عرصے سے ہو رہی ہے؟"                   │
│      (How long have you been feeling tired?)                     │
│                                                                  │
│  👤 Patient: "تقریباً 2-3 مہینے"                                  │
│      (About 2-3 months)                                          │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 🧠 REASONING: Chronic fatigue → rule out metabolic causes   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  🤖 AI: "کیا آپ کو بہت زیادہ پیاس لگتی ہے؟"                     │
│      (Do you feel very thirsty?)                                 │
│                                                                  │
│  👤 Patient: "ہاں، بہت زیادہ"                                     │
│      (Yes, very much)                                            │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 🧠 REASONING: Fatigue + polydipsia → diabetes screening     │ │
│  │ 🚨 RED FLAG: Classic diabetes triad forming                │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  🤖 AI: "رات کو کتنی بار پیشاب کے لیے اٹھنا پڑتا ہے؟"           │
│      (How many times do you wake up at night to urinate?)        │
│                                                                  │
│  👤 Patient: "3-4 بار"                                          │
│      (3-4 times)                                                 │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ ⚠️ TRIAGE ESCALATION: VERY URGENT                           │ │
│  │ Pattern: Polyuria + Polydipsia + Fatigue = Diabetes Screen   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```


#### 📋 Generated SOAP Note
```text
╔══════════════════════════════════════════════════════════════════╗
║                CLINICAL SUMMARY (AI-GENERATED)                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Triage Level: 🟠 VERY URGENT                                    ║
║  Confidence: 87%                                                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  SUBJECTIVE                                                      ║
║  ──────────────────────────────────────────────────────────────  ║
║  Chief Complaint: Chronic fatigue for 2-3 months                 ║
║                                                                  ║
║  History of Present Illness:                                     ║
║  45-year-old male farmer presenting with progressive fatigue     ║
║  for 2-3 months. Reports excessive thirst (polydipsia) and       ║
║  frequent urination, waking 3-4 times nightly (nocturia).        ║
║  Notes unintentional weight loss of approximately 5 kg.          ║
║                                                                  ║
║  Family History: Father with Type 2 Diabetes                     ║
║                                                                  ║
║  ASSESSMENT                                                      ║
║  ──────────────────────────────────────────────────────────────  ║
║  ⚠️ HIGH PRIORITY: Suspected Type 2 Diabetes Mellitus             ║
║                                                                  ║
║  Clinical Reasoning:                                             ║
║  Classic triad present: Polyuria, Polydipsia, Fatigue            ║
║  Additional: Weight loss, positive family history                ║
║                                                                  ║
║  Differential Diagnoses:                                         ║
║  1. Type 2 Diabetes Mellitus (most likely)                       ║
║  2. Type 1 Diabetes (less likely given age)                      ║
║  3. Thyroid dysfunction                                          ║
║                                                                  ║
║  PLAN                                                            ║
║  ──────────────────────────────────────────────────────────────  ║
║  1. 🔴 URGENT: Random blood glucose or HbA1c TODAY               ║
║  2. If glucose > 250 mg/dL: Rule out DKA                         ║
║  3. Fasting lipid panel                                          ║
║  4. Renal function tests                                         ║
║  5. Diabetes education if confirmed                              ║
║                                                                  ║
║  ──────────────────────────────────────────────────────────────  ║
║  ⚡ Time Sensitivity: Test within 24 hours                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```


---

### Case Study 2: Emergency Triage 🚨

**Problem:** Delayed response to life-threatening symptoms.

#### 🔴 Red Flag Detection System
```text
┌─────────────────────────────────────────────────────────────────┐
│                       RED FLAG PATTERNS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🫀 CARDIAC EMERGENCY (MI)                                      │
│  ├── Chest pain/pressure/tightness                              │
│  ├── PLUS: Radiation to arm, jaw, back                          │
│  ├── PLUS: Sweating (diaphoresis)                               │
│  └── ACTION: "ابھی 1122 کال کریں" (Call 1122 NOW)               │
│                                                                 │
│  🧠 STROKE (CVA)                                                │
│  ├── Sudden facial drooping                                     │
│  ├── PLUS: Arm weakness (one side)                              │
│  ├── PLUS: Speech difficulty                                    │
│  └── ACTION: Note time of onset, Call 1122                      │
│                                                                 │
│  🩸 SUBARACHNOID HEMORRHAGE (SAH)                               │
│  ├── "Worst headache of my life"                                │
│  ├── PLUS: Thunderclap onset (seconds)                          │
│  ├── PLUS: Neck stiffness                                       │
│  └── ACTION: IMMEDIATE - Do not delay                          │
│                                                                 │
│  🫁 PULMONARY EMBOLISM (PE)                                     │
│  ├── Sudden shortness of breath                                 │
│  ├── PLUS: Chest pain                                           │
│  ├── PLUS: Leg swelling                                         │
│  └── ACTION: Emergency evaluation                               │
│                                                                 │
│  🩺 DIABETIC CRISIS (DKA/HHS)                                   │
│  ├── Excessive thirst + Frequent urination                      │
│  ├── PLUS: Confusion or altered consciousness                   │
│  ├── PLUS: Fruity breath odor                                   │
│  └── ACTION: ER immediately                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```


#### 🚨 WHO Emergency Triage Levels

| Level | Color | Description | Action | Response Time |
|-------|-------|-------------|--------|---------------|
| **IMMEDIATE** | 🔴 Red | Life-threatening | Call 1122 NOW | Seconds |
| **VERY URGENT** | 🟠 Orange | Serious, risk of deterioration | ER immediately | 10 minutes |
| **URGENT** | 🟡 Yellow | Needs attention soon | ER within 2 hours | 1 hour |
| **STANDARD** | 🟢 Green | Can wait | Schedule appointment | 24-48 hours |
| **NON-URGENT** | 🔵 Blue | Minor issues | Routine care | Convenience |

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/healthcare-ai-system.git
cd healthcare-ai-system
```

### 2️⃣ Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
# Required: OPENAI_API_KEY (or Groq), and WhatsApp Cloud API credentials
```

### 5️⃣ Initialize the System

```bash
# Ingest medical documents into vector store
python run.py ingest

# Run health check
python run.py health
```

### 6️⃣ Start the Server

```bash
# Start the API server
python run.py serve

# Or run interactive demo
python run.py demo
```
## 📖 Usage

### 🖥️ Interactive CLI Demo

```bash
python run.py demo
```

```text
════════════════════════════════════════════════════════════════
🏥 Healthcare AI System - Interactive Demo
════════════════════════════════════════════════════════════════

AI: السلام علیکم! میں آپ کا AI صحت معاون ہوں۔
    آج میں آپ کی کیا مدد کر سکتا ہوں؟

You: مجھے سینے میں درد ہے

AI: سینے میں درد کتنی دیر سے ہے؟
    کیا یہ درد کہیں اور بھی جاتا ہے، جیسے بازو یا جبڑے میں؟

[Processing...]
────────────────────────────────────────
⚠️ TRIAGE: VERY URGENT - Possible cardiac event
   ACTION: Seek emergency care immediately
────────────────────────────────────────
```

### 🌐 REST API Usage

#### Start a New Session

```bash
curl -X POST "http://localhost:8000/api/v1/chat/start" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_name": "Ahmed",
    "patient_age": 45,
    "language": "ur"
  }'
```

**Response:**

```json
{
  "session_id": "session_20250120_abc123",
  "greeting": "السلام علیکم! میں آپ کا AI صحت معاون ہوں۔",
  "greeting_english": "Hello! I'm your AI health assistant.",
  "language": "ur"
}
```

#### Send a Message

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "مجھے دو دن سے سر میں درد ہے",
    "session_id": "session_20250120_abc123",
    "include_reasoning": true
  }'
```

**Response:**

```json
{
  "session_id": "session_20250120_abc123",
  "response": "سر درد کتنا شدید ہے، 1 سے 10 کے پیمانے پر؟",
  "response_english": "How severe is the headache on a scale of 1-10?",
  "triage_level": "STANDARD",
  "is_emergency": false,
  "symptoms": [
    {"name": "headache", "duration": "2 days"}
  ],
  "reasoning_trace": [
    "Interviewer: Headache reported, asking about severity",
    "Guardian: No red flags detected"
  ]
}
```

#### Quick Triage Check

```bash
curl -X POST "http://localhost:8000/api/v1/chat/quick-triage" \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["chest pain", "sweating", "arm pain"],
    "severity": "severe",
    "language": "ur"
  }'
```

**Response:**

```json
{
  "triage_level": "IMMEDIATE",
  "triage_color": "red",
  "urgency": "Life-threatening - needs immediate care",
  "action_required": "Call 1122 immediately",
  "action_local_language": "ابھی 1122 کال کریں",
  "matched_patterns": ["cardiac_emergency"],
  "call_emergency": true
}
```
### 📡 API Reference

#### Base URL

```text
http://localhost:8000/api/v1
```

#### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat/start` | Start new interview session |
| POST | `/chat` | Send message in conversation |
| GET | `/chat/history/{session_id}` | Get conversation history |
| POST | `/chat/quick-triage` | Quick symptom triage |
| POST | `/chat/feedback` | Submit feedback |
| POST | `/voice/transcribe` | Transcribe audio to text |
| POST | `/voice/synthesize` | Convert text to speech |
| POST | `/voice/chat` | Full voice conversation |
| POST | `/webhook/whatsapp` | WhatsApp webhook |
| GET | `/health` | System health check |

#### Full API Documentation

Once the server is running, access:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
🎬 Demo
🎥 Video Demo
Healthcare AI Demo

Click to watch the full demo video

📱 Screenshots
<table> <tr> <td width="33%"> <img src="docs/assets/screenshot-interview.png" alt="Interview Screen"/> <p align="center"><em>Patient Interview</em></p> </td> <td width="33%"> <img src="docs/assets/screenshot-triage.png" alt="Triage Screen"/> <p align="center"><em>Emergency Triage</em></p> </td> <td width="33%"> <img src="docs/assets/screenshot-soap.png" alt="SOAP Note"/> <p align="center"><em>SOAP Note Output</em></p> </td> </tr> </table>
💬 WhatsApp Demo
text

┌─────────────────────────────────────┐
│         WhatsApp Chat               │
├─────────────────────────────────────┤
│                                     │
│  👤 You                      10:30  │
│  ┌─────────────────────────────┐   │
│  │ 🎤 Voice Message (0:05)     │   │
│  │ "مجھے بخار ہے"               │   │
│  └─────────────────────────────┘   │
│                                     │
│  🤖 Healthcare AI            10:30  │
│  ┌─────────────────────────────┐   │
│  │ بخار کتنے دن سے ہے؟          │   │
│  │ کیا ٹھنڈ لگ رہی ہے؟          │   │
│  └─────────────────────────────┘   │
│                                     │
│  👤 You                      10:31  │
│  ┌─────────────────────────────┐   │
│  │ 2 دن سے، ہاں ٹھنڈ بھی ہے     │   │
│  └─────────────────────────────┘   │
│                                     │
│  🤖 Healthcare AI            10:31  │
│  ┌─────────────────────────────┐   │
│  │ 🌡️ Triage: STANDARD         │   │
│  │ گھر پر آرام کریں، پانی پئیں   │   │
│  │ اگر بخار 3 دن سے زیادہ رہے   │   │
│  │ تو ڈاکٹر سے ملیں              │   │
│  └─────────────────────────────┘   │
│                                     │
└─────────────────────────────────────┘
🛠️ Tech Stack
<table> <tr> <td align="center" width="20%"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/> <br/><b>Python 3.10+</b> <br/><sub>Core Language</sub> </td> <td align="center" width="20%"> <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/> <br/><b>FastAPI</b> <br/><sub>API Framework</sub> </td> <td align="center" width="20%"> <img src="https://img.shields.io/badge/LangChain-121212?style=for-the-badge" alt="LangChain"/> <br/><b>LangChain</b> <br/><sub>LLM Framework</sub> </td> <td align="center" width="20%"> <img src="https://img.shields.io/badge/LangGraph-6B46C1?style=for-the-badge" alt="LangGraph"/> <br/><b>LangGraph</b> <br/><sub>State Machine</sub> </td> <td align="center" width="20%"> <img src="https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge" alt="ChromaDB"/> <br/><b>ChromaDB</b> <br/><sub>Vector Store</sub> </td> </tr> <tr> <td align="center"> <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI"/> <br/><b>OpenAI / Groq</b> <br/><sub>LLM Provider</sub> </td> <td align="center"> <img src="https://img.shields.io/badge/Whisper-74AA9C?style=for-the-badge" alt="Whisper"/> <br/><b>Whisper</b> <br/><sub>Speech-to-Text</sub> </td> <td align="center"> <img src="https://img.shields.io/badge/Meta-0668E1?style=for-the-badge&logo=meta&logoColor=white" alt="Meta"/> <br/><b>WhatsApp Cloud API</b> <br/><sub>Messaging</sub> </td> <td align="center"> <img src="https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white" alt="Pydantic"/> <br/><b>Pydantic</b> <br/><sub>Validation</sub> </td> <td align="center"> <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/> <br/><b>Docker</b> <br/><sub>Deployment</sub> </td> </tr> </table>
## 📁 Project Structure

```text
healthcare-ai-system/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 .env.example                 # Environment template
├── 📄 run.py                       # Application entry point
├── 📄 LICENSE                      # MIT License
│
├── 📁 src/
│   ├── 📄 __init__.py
│   ├── 📄 config.py                # Configuration management
│   │
│   ├── 📁 graph/                   # 🧠 LangGraph Workflow
│   │   ├── 📄 __init__.py
│   │   ├── 📄 state.py             # PatientState definition
│   │   ├── 📄 workflow.py          # Main graph orchestration
│   │   └── 📁 nodes/
│   │       ├── 📄 guardian.py      # Red flag detection
│   │       ├── 📄 interviewer.py   # Adaptive questioning
│   │       ├── 📄 triage.py        # WHO triage assessment
│   │       └── 📄 summarizer.py    # SOAP note generation
│   │
│   ├── 📁 agents/                  # 🤖 Agent Prompts
│   │   ├── 📄 __init__.py
│   │   ├── 📁 prompts/
│   │   │   ├── 📄 triage_prompts.py
│   │   │   ├── 📄 interview_prompts.py
│   │   │   └── 📄 translator_prompts.py
│   │   ├── 📁 tools/               # 🔧 Agent Tools
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 rag_search.py    # Clinical guidelines RAG
│   │   │   ├── 📄 calculator.py    # Medical calculations
│   │   │   └── 📄 time_utils.py    # Duration parsing
│   │   └── 📄 utils.py
│   │
│   ├── 📁 services/                # 🌐 External Services
│   │   ├── 📄 __init__.py
│   │   ├── 📄 voice_service.py     # STT/TTS
│   │   ├── 📄 translator_service.py
│   │   └── 📄 whatsapp_service.py
│   │
│   ├── 📁 api/                     # 🚀 FastAPI Application
│   │   ├── 📄 __init__.py
│   │   ├── 📄 main.py              # App initialization
│   │   ├── 📁 routes/
│   │   │   ├── 📄 chat.py
│   │   │   ├── 📄 voice.py
│   │   │   └── 📄 whatsapp.py
│   │   └── 📁 middleware/
│   │       └── 📄 error_handler.py
│   │
│   ├── 📁 data/                    # 📚 Data & Documents
│   │   ├── 📄 ingestion.py         # Document ingestion
│   │   ├── 📁 docs/                # Medical guidelines
│   │   └── 📁 vector_store/        # ChromaDB storage
│   │
│   ├── 📁 utils/                   # 🛠️ Utilities
│   │   ├── 📄 __init__.py
│   │   ├── 📄 prompts.py
│   │   ├── 📄 schemas.py
│   │   └── 📄 validators.py
│   │
│   └── 📁 tests/                   # 🧪 Test Suite
│       ├── 📄 __init__.py
│       └── 📄 test_agents.py
│
└── 📁 docs/                        # 📖 Documentation
    ├── 📁 assets/                  # Images, diagrams
    └── 📄 api-reference.md
```
## 🧪 Testing

### Run All Tests

```bash
# Run all tests
python run.py test

# Run with coverage
python run.py test --coverage

# Run specific tests
python run.py test --pattern "test_guardian"

# Run by markers
python run.py test --markers "unit"
```

### Test Categories

| Category | Description | Command |
|----------|-------------|---------|
| Unit | Individual component tests | `pytest -m unit` |
| Integration | Component interaction tests | `pytest -m integration` |
| E2E | Full workflow tests | `pytest -m e2e` |
| Performance | Load testing | `pytest -m slow` |

### Sample Test Output

```text

================================ test session starts ================================
platform linux -- Python 3.10.12, pytest-7.4.0
collected 42 items

src/tests/test_agents.py::test_guardian_detects_cardiac_emergency PASSED    [  2%]
src/tests/test_agents.py::test_guardian_detects_stroke PASSED               [  4%]
src/tests/test_agents.py::test_interviewer_asks_followup PASSED             [  7%]
src/tests/test_agents.py::test_triage_assigns_correct_level PASSED          [  9%]
src/tests/test_agents.py::test_summarizer_generates_soap PASSED             [ 11%]
...
src/tests/test_api.py::test_chat_endpoint PASSED                            [ 95%]
src/tests/test_api.py::test_voice_transcription PASSED                      [ 97%]
src/tests/test_api.py::test_whatsapp_webhook PASSED                         [100%]

================================ 42 passed in 12.34s ================================
## 🗺️ Roadmap

- [x] **Phase 1: MVP (Completed)**
  - Core interview workflow
  - Diabetes case study
  - Basic SOAP generation
  - Urdu language support
- [x] **Phase 2: Silver Tier (Completed)**
  - Multi-agent architecture
  - Red flag detection (Guardian)
  - Emergency triage
  - Multiple case studies
- [x] **Phase 3: Gold Tier (Completed)**
  - Voice processing (STT/TTS)
  - Multilingual support (5 languages)
  - WhatsApp integration
  - Real-time explainability
- [ ] **Phase 4: Future Enhancements**
  - Mobile application (React Native)
  - EHR/EMR integration (HL7 FHIR)
  - Offline mode for rural areas
  - Continuous learning from feedback
  - Multi-modal input (images for skin conditions)
  - Specialist referral network

## 👥 Team

| Member | Role | Links |
| :--- | :--- | :--- |
| **Team Member 1** | AI/ML Engineer | [GitHub](https://github.com/member1) • [LinkedIn](https://linkedin.com/in/member1) |
| **Team Member 2** | Backend Developer | [GitHub](https://github.com/member2) • [LinkedIn](https://linkedin.com/in/member2) |
| **Team Member 3** | Healthcare Domain Expert | [GitHub](https://github.com/member3) • [LinkedIn](https://linkedin.com/in/member3) |
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```text
MIT License

Copyright (c) 2025 Healthcare AI Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

- WHO for Emergency Triage Assessment and Treatment (ETAT) guidelines
- LangChain team for the excellent LLM framework
- OpenAI for GPT and Whisper models
- AI Hackathon 2025 organizers
- Healthcare professionals who provided domain expertise

<div align="center">
  <h3>🏥 Built with ❤️ for Pakistan's Healthcare</h3>
  <p>Saving lives, one conversation at a time.</p>
  
  <sub> ⚠️ <b>Disclaimer:</b> This is a clinical decision-support tool, NOT a diagnostic system. It does not replace professional medical judgment. Always consult qualified healthcare providers for medical decisions. </sub>
</div>
