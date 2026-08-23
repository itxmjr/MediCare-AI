"use client";

import { useState, useRef, useCallback } from 'react';

const API_BASE_URL = "/api/v1";

interface TranscribeResponse {
    text: string;
    detected_language: string;
    confidence: number;
    duration_seconds: number;
    processing_time_ms: number;
}

interface SynthesizeResponse {
    audio_base64: string;
    format: string;
    duration_seconds: number;
    processing_time_ms: number;
    text_length: number;
}

export function useVoice() {
    const [isRecording, setIsRecording] = useState(false);
    const [isTranscribing, setIsTranscribing] = useState(false);
    const [isSynthesizing, setIsSynthesizing] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const audioRef = useRef<HTMLAudioElement | null>(null);

    // Start recording audio from microphone
    const startRecording = useCallback(async () => {
        try {
            setError(null);
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            const mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });

            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorderRef.current = mediaRecorder;
            mediaRecorder.start();
            setIsRecording(true);
        } catch (err) {
            console.error('Failed to start recording:', err);
            setError('Could not access microphone. Please check permissions.');
        }
    }, []);

    // Stop recording and return the audio blob
    const stopRecording = useCallback((): Promise<Blob | null> => {
        return new Promise((resolve) => {
            const mediaRecorder = mediaRecorderRef.current;

            if (!mediaRecorder || mediaRecorder.state === 'inactive') {
                setIsRecording(false);
                resolve(null);
                return;
            }

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });

                // Stop all tracks
                mediaRecorder.stream.getTracks().forEach(track => track.stop());

                setIsRecording(false);
                resolve(audioBlob);
            };

            mediaRecorder.stop();
        });
    }, []);

    // Convert audio blob to base64
    const blobToBase64 = (blob: Blob): Promise<string> => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                const base64 = reader.result as string;
                // Remove the data URL prefix
                const base64Data = base64.split(',')[1];
                resolve(base64Data);
            };
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    };

    // Transcribe audio to text
    const transcribeAudio = useCallback(async (audioBlob: Blob): Promise<string | null> => {
        setIsTranscribing(true);
        setError(null);

        try {
            const audioBase64 = await blobToBase64(audioBlob);

            const response = await fetch(`${API_BASE_URL}/voice/transcribe`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    audio_base64: audioBase64,
                    audio_format: 'webm',
                    language_hint: 'en'
                }),
            });

            if (!response.ok) {
                throw new Error(`Transcription failed: ${response.status}`);
            }

            const data: TranscribeResponse = await response.json();
            return data.text;
        } catch (err) {
            console.error('Transcription error:', err);
            setError('Failed to transcribe audio. Please try again.');
            return null;
        } finally {
            setIsTranscribing(false);
        }
    }, []);

    // Synthesize text to speech and play it
    const synthesizeAndPlay = useCallback(async (text: string, language: string = 'en'): Promise<void> => {
        setIsSynthesizing(true);
        setError(null);

        try {
            const response = await fetch(`${API_BASE_URL}/voice/synthesize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text,
                    language,
                    voice_gender: 'female',
                    speed: 1.0,
                    output_format: 'mp3'
                }),
            });

            if (!response.ok) {
                throw new Error(`Synthesis failed: ${response.status}`);
            }

            const data: SynthesizeResponse = await response.json();

            // Create audio element and play
            const audioSrc = `data:audio/mp3;base64,${data.audio_base64}`;

            if (audioRef.current) {
                audioRef.current.pause();
            }

            const audio = new Audio(audioSrc);
            audioRef.current = audio;

            audio.onplay = () => setIsPlaying(true);
            audio.onended = () => setIsPlaying(false);
            audio.onerror = () => {
                setIsPlaying(false);
                setError('Failed to play audio');
            };

            await audio.play();
        } catch (err) {
            console.error('Synthesis error:', err);
            setError('Failed to synthesize speech. Please try again.');
        } finally {
            setIsSynthesizing(false);
        }
    }, []);

    // Stop playing audio
    const stopPlaying = useCallback(() => {
        if (audioRef.current) {
            audioRef.current.pause();
            audioRef.current.currentTime = 0;
            setIsPlaying(false);
        }
    }, []);

    // Record, transcribe, and return text
    const recordAndTranscribe = useCallback(async (): Promise<string | null> => {
        const audioBlob = await stopRecording();
        if (!audioBlob) return null;
        return transcribeAudio(audioBlob);
    }, [stopRecording, transcribeAudio]);

    return {
        // Recording
        isRecording,
        startRecording,
        stopRecording,
        recordAndTranscribe,

        // Transcription
        isTranscribing,
        transcribeAudio,

        // Synthesis
        isSynthesizing,
        synthesizeAndPlay,

        // Playback
        isPlaying,
        stopPlaying,

        // Error
        error,
        clearError: () => setError(null),
    };
}
