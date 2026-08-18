# API Reference

This document provides a detailed reference for the Healthcare AI System API.

## Base URL
`http://localhost:8000/api/v1`

## Endpoints

### 1. Start Chat Session
`POST /chat/start`

Starts a new interview session for a patient.

**Request Body:**
```json
{
  "patient_name": "string",
  "patient_age": "integer",
  "language": "string (en, ur, sd, ps, pa)"
}
```

### 2. Send Message
`POST /chat`

Sends a message to an active session.

**Request Body:**
```json
{
  "message": "string",
  "session_id": "string",
  "include_reasoning": "boolean"
}
```

### 3. Quick Triage
`POST /chat/quick-triage`

Performs a quick triage assessment based on symptoms.

**Request Body:**
```json
{
  "symptoms": ["string"],
  "severity": "string",
  "language": "string"
}
```

## Voice API

### 1. Transcribe
`POST /voice/transcribe`

### 2. Synthesize
`POST /voice/synthesize`

### 3. Voice Chat
`POST /voice/chat`

## WhatsApp Integration
`POST /webhook/whatsapp`
