"use client";

import { useState, useCallback, useEffect } from 'react';

export interface Message {
    id: string;
    type: 'user' | 'bot';
    content: string;
    timestamp: Date;
    matchedQuestion?: string;
    confidenceScore?: number;
}

interface ApiResponse {
    session_id: string;
    response: string;
    response_english?: string;
    triage_level: string;
    is_emergency: boolean;
    emergency_action?: string;
    interview_complete: boolean;
    symptoms_identified: string[];
    red_flags: string[];
    soap_note?: string;
    reasoning_trace: string[];
}

const API_BASE_URL = "/api/v1";

// Simple language detection - checks for Urdu/Arabic script
function detectLanguage(text: string): string {
    // Check for Urdu/Arabic script characters
    const urduPattern = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/;
    if (urduPattern.test(text)) {
        return 'ur';
    }
    return 'en';
}

async function fetchBotResponse(message: string, sessionId: string, language: string): Promise<ApiResponse> {
    const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message, session_id: sessionId, language }),
    });

    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }

    return response.json();
}

export function useChat() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isThinking, setIsThinking] = useState(false);
    const [sessionId, setSessionId] = useState<string>('');

    // Initialize sessionId on client-side only to avoid hydration mismatch
    useEffect(() => {
        setSessionId(`session_${Date.now()}`);
    }, []);

    const sendMessage = useCallback(async (content: string) => {
        if (!content.trim()) return;

        const userMessage: Message = {
            id: `user-${Date.now()}`,
            type: 'user',
            content: content.trim(),
            timestamp: new Date(),
        };

        setMessages(prev => [...prev, userMessage]);
        setIsThinking(true);

        try {
            // Detect language from user input
            const detectedLanguage = detectLanguage(content.trim());
            const response = await fetchBotResponse(content.trim(), sessionId, detectedLanguage);

            // Update session ID from response
            if (response.session_id) {
                setSessionId(response.session_id);
            }

            const botMessage: Message = {
                id: `bot-${Date.now()}`,
                type: 'bot',
                content: response.response,
                timestamp: new Date(),
                matchedQuestion: response.triage_level || undefined,
                confidenceScore: response.is_emergency ? 100 : 80,
            };

            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            console.error('Failed to fetch response:', error);

            const errorMessage: Message = {
                id: `bot-${Date.now()}`,
                type: 'bot',
                content: 'Sorry, I encountered an error while processing your request. Please make sure the backend server is running and try again.',
                timestamp: new Date(),
                confidenceScore: 0,
            };

            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsThinking(false);
        }
    }, [sessionId]);

    const clearMessages = useCallback(() => {
        setMessages([]);
    }, []);

    return {
        messages,
        isThinking,
        sendMessage,
        clearMessages,
    };
}
