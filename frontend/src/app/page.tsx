"use client";

import { useRef, useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { useChat } from '@/hooks/useChat';
import { CommandBar } from '@/components/CommandBar';
import { MessageCard } from '@/components/MessageCard';
import { ThinkingIndicator } from '@/components/ThinkingIndicator';
import { Sidebar } from '@/components/Sidebar';
import { EmptyState } from '@/components/EmptyState';
import { FaqBrowser } from '@/components/FaqBrowser';
import { Menu, X } from 'lucide-react';

export default function Home() {
  const { messages, isThinking, sendMessage, clearMessages } = useChat();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [sidebarExpanded, setSidebarExpanded] = useState(true);
  const [prefillQuery, setPrefillQuery] = useState('');
  const [isBrowserOpen, setIsBrowserOpen] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
    // Auto-collapse sidebar on smaller laptops
    const handleResize = () => {
      if (window.innerWidth < 1200) {
        setSidebarExpanded(false);
      } else {
        setSidebarExpanded(true);
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    if (scrollRef.current && messages.length > 0) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [messages, isThinking]);

  const handleQuerySelect = (query: string) => {
    setPrefillQuery(query);
  };

  const handleGoHome = () => {
    clearMessages();
  };

  const handlePrefillConsumed = () => {
    setPrefillQuery('');
  };

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 -left-40 w-[500px] h-[500px] rounded-full bg-gold/[0.02] blur-3xl" />
        {/* Center subtle gradient */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full bg-gold/[0.01] blur-3xl" />
        {/* Grid overlay */}
        <div
          className="absolute inset-0 opacity-[0.015]"
          style={{
            backgroundImage: `linear-gradient(hsl(0 0% 50% / 0.1) 1px, transparent 1px),
                              linear-gradient(90deg, hsl(0 0% 50% / 0.1) 1px, transparent 1px)`,
            backgroundSize: '60px 60px',
          }}
        />
      </div>

      <div className={`md:block ${isMobileMenuOpen ? 'block' : 'hidden'}`}>
        <Sidebar
          onQuerySelect={(q) => {
            handleQuerySelect(q);
            setIsMobileMenuOpen(false);
          }}
          onGoHome={() => {
            handleGoHome();
            setIsMobileMenuOpen(false);
          }}
          hasMessages={messages.length > 0}
          isExpanded={sidebarExpanded}
          onToggleExpanded={setSidebarExpanded}
          onOpenBrowser={() => {
            setIsBrowserOpen(true);
            setIsMobileMenuOpen(false);
          }}
        />
      </div>

      <button
        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        className="fixed top-4 left-4 z-[51] p-2 bg-surface-2 border border-border rounded-xl md:hidden text-foreground"
      >
        {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
      </button>

      <motion.div
        className="relative z-10 min-h-screen transition-all duration-500 ease-premium"
        initial={false}
        animate={{
          marginLeft: isMounted && window.innerWidth >= 768
            ? (sidebarExpanded ? '256px' : '80px')
            : '0px'
        }}
        transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
      >
        <div
          ref={scrollRef}
          className="h-screen overflow-y-auto pb-44 px-4 sm:px-8 md:px-12"
        >
          <div className="max-w-3xl mx-auto">
            {/* Content */}
            {messages.length === 0 && !isThinking ? (
              <div className="pt-8">
                <EmptyState onQuerySelect={handleQuerySelect} onOpenBrowser={() => setIsBrowserOpen(true)} />
              </div>
            ) : (
              <div className="space-y-8 pt-8">
                <AnimatePresence mode="popLayout">
                  {messages.map((message, index) => (
                    <MessageCard
                      key={message.id}
                      message={message}
                      index={index}
                    />
                  ))}
                </AnimatePresence>

                <AnimatePresence>
                  {isThinking && <ThinkingIndicator />}
                </AnimatePresence>
              </div>
            )}
          </div>
        </div>

        {/* Command Bar */}
        <CommandBar
          onSend={sendMessage}
          isThinking={isThinking}
          prefillValue={prefillQuery}
          onPrefillConsumed={handlePrefillConsumed}
        />
      </motion.div>

      <FaqBrowser
        isOpen={isBrowserOpen}
        onClose={() => setIsBrowserOpen(false)}
        onSelectQuestion={handleQuerySelect}
      />
    </div>
  );
}
