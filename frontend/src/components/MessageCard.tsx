"use client";

import { useState, useEffect } from 'react';

import { motion } from 'framer-motion';
import { Message } from '@/hooks/useChat';
import { ConfidenceMeter } from './ConfidenceMeter';
import { User, Sparkles, Tag, Volume2, VolumeX, Loader2 } from 'lucide-react';
import { useVoice } from '@/hooks/useVoice';

interface MessageCardProps {
  message: Message;
  index: number;
}

export function MessageCard({ message, index }: MessageCardProps) {
  const isUser = message.type === 'user';
  const isWarning = message.confidenceScore !== undefined && message.confidenceScore < 0.3;
  const { synthesizeAndPlay, isPlaying, isSynthesizing, stopPlaying } = useVoice();

  const handlePlayAudio = () => {
    if (isPlaying) {
      stopPlaying();
    } else {
      synthesizeAndPlay(message.content, 'en');
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 30, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{
        duration: 0.5,
        delay: index * 0.03,
        ease: [0.4, 0, 0.2, 1]
      }}
      className={`w-full flex ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      {isUser ? (
        <motion.div
          className="max-w-lg"
          whileHover={{ scale: 1.01 }}
          transition={{ duration: 0.2 }}
        >
          <div className="card-user rounded-2xl px-4 py-3 md:px-5 md:py-4">
            <div className="flex items-start gap-3 md:gap-4">
              <div className="flex-1">
                <p className="text-foreground/95 text-sm leading-relaxed">
                  {message.content}
                </p>
              </div>
              <div className="w-7 h-7 md:w-8 md:h-8 rounded-lg md:rounded-xl bg-surface-elevated flex items-center justify-center flex-shrink-0">
                <User className="w-3.5 h-3.5 md:w-4 md:h-4 text-muted-foreground" />
              </div>
            </div>
          </div>
          <div className="text-right mt-2 pr-2">
            <span className="text-[10px] text-muted-foreground/50 font-medium" suppressHydrationWarning>
              {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          </div>
        </motion.div>
      ) : (
        <motion.div
          className="w-full max-w-3xl"
          whileHover={{ scale: 1.005 }}
          transition={{ duration: 0.2 }}
        >
          <div className={`rounded-2xl overflow-hidden ${isWarning ? 'card-bot-warning' : 'card-bot'}`}>
            <div className="px-5 py-3 md:px-6 md:py-4 border-b border-border/30 flex items-center justify-between">
              <div className="flex items-center gap-2 md:gap-3">
                <div className="w-7 h-7 md:w-8 md:h-8 rounded-lg md:rounded-xl bg-gradient-to-br from-gold/20 to-gold/5 flex items-center justify-center">
                  <Sparkles className="w-3.5 h-3.5 md:w-4 md:h-4 text-gold" />
                </div>
                <div>
                  <span className="text-[10px] md:text-xs font-semibold text-foreground/90 uppercase tracking-wide">
                    AI Response
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-3">
                {/* Play Audio Button */}
                <motion.button
                  onClick={handlePlayAudio}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  className={`
                    w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-200
                    ${isPlaying
                      ? 'bg-gold/20 text-gold'
                      : 'bg-surface-elevated hover:bg-surface-2 text-muted-foreground hover:text-foreground'
                    }
                  `}
                  title={isPlaying ? 'Stop audio' : 'Play audio'}
                >
                  {isSynthesizing ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : isPlaying ? (
                    <VolumeX className="w-4 h-4" />
                  ) : (
                    <Volume2 className="w-4 h-4" />
                  )}
                </motion.button>
                {message.confidenceScore !== undefined && (
                  <ConfidenceMeter score={message.confidenceScore} />
                )}
              </div>
            </div>

            <div className="px-5 py-4 md:px-6 md:py-5">
              <p className="text-foreground/90 text-sm leading-relaxed tracking-wide">
                {message.content}
              </p>
            </div>

            {message.matchedQuestion && (
              <div className="px-5 py-3 md:px-6 md:py-4 bg-surface-1/50 border-t border-border/20">
                <div className="flex items-center gap-2 flex-wrap">
                  <Tag className="w-3 h-3 md:w-3.5 md:h-3.5 text-muted-foreground" />
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
                    Triage:
                  </span>
                  <span className="text-[10px] md:text-xs text-foreground/70 bg-surface-elevated px-2.5 py-1 md:px-3 md:py-1.5 rounded-full">
                    {message.matchedQuestion}
                  </span>
                </div>
              </div>
            )}
          </div>

          <div className="mt-2 pl-2">
            <span className="text-[10px] text-muted-foreground/50 font-medium" suppressHydrationWarning>
              {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}

