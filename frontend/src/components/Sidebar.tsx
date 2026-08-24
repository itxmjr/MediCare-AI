"use client";

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Heart,
  Stethoscope,
  AlertTriangle,
  Languages,
  TrendingUp,
  Sparkles,
  ChevronLeft,
  ChevronRight,
  Home,
  BookOpen
} from 'lucide-react';

interface SidebarProps {
  onQuerySelect: (query: string) => void;
  onGoHome: () => void;
  hasMessages: boolean;
  isExpanded: boolean;
  onToggleExpanded: (expanded: boolean) => void;
  onOpenBrowser: () => void;
}

const categories = [
  { id: 'symptoms', label: 'Symptoms', icon: Heart, query: 'I have a headache' },
  { id: 'checkup', label: 'Check-up', icon: Stethoscope, query: 'I need a general health check' },
  { id: 'emergency', label: 'Emergency', icon: AlertTriangle, query: 'I am having chest pain' },
  { id: 'languages', label: 'Languages', icon: Languages, query: 'Can you speak Urdu?' },
];

const trendingQueries = [
  { id: 1, query: "I have a fever and body aches", category: "Symptoms" },
  { id: 2, query: "My child has diarrhea", category: "Pediatric" },
  { id: 3, query: "I feel dizzy and nauseous", category: "General" },
];

export function Sidebar({ onQuerySelect, onGoHome, hasMessages, isExpanded, onToggleExpanded, onOpenBrowser }: SidebarProps) {
  const [activeCategory, setActiveCategory] = useState<string | null>(null);

  const handleCategoryClick = (category: typeof categories[0]) => {
    setActiveCategory(activeCategory === category.id ? null : category.id);
    onQuerySelect(category.query);
  };

  return (
    <motion.aside
      initial={{ x: -100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
      className={`fixed left-0 top-0 h-screen z-50 flex transition-all duration-500 ease-premium ${isExpanded ? 'w-64' : 'w-20'
        } ${!isExpanded ? 'hidden md:flex' : 'flex'}`}
    >
      <div className="flex-1 h-full bg-sidebar border-r border-sidebar-border flex flex-col">
        <div className="p-6 border-b border-sidebar-border">
          <button
            onClick={onGoHome}
            className="flex items-center gap-3 w-full group"
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-gold to-gold-dim flex items-center justify-center glow-gold-soft group-hover:scale-105 transition-transform duration-300">
              <Sparkles className="w-5 h-5 text-surface-0" />
            </div>
            <AnimatePresence>
              {isExpanded && (
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.2 }}
                  className="text-left"
                >
                  <h1 className="font-semibold text-foreground group-hover:text-gold transition-colors duration-300">MediCare AI</h1>
                  <p className="text-xs text-muted-foreground">AI</p>
                </motion.div>
              )}
            </AnimatePresence>
          </button>
        </div>

        <AnimatePresence>
          {hasMessages && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="px-4 pt-4"
            >
              <button
                onClick={onGoHome}
                className={`sidebar-item w-full bg-gold/10 border border-gold/20 hover:bg-gold/20 ${!isExpanded ? 'justify-center px-2' : ''
                  }`}
              >
                <Home className="w-5 h-5 flex-shrink-0 text-gold" />
                <AnimatePresence>
                  {isExpanded && (
                    <motion.span
                      initial={{ opacity: 0, width: 0 }}
                      animate={{ opacity: 1, width: 'auto' }}
                      exit={{ opacity: 0, width: 0 }}
                      className="whitespace-nowrap overflow-hidden text-gold"
                    >
                      New Conversation
                    </motion.span>
                  )}
                </AnimatePresence>
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="flex-1 overflow-y-auto p-4">
          <AnimatePresence>
            {isExpanded && (
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground mb-4 px-2"
              >
                Quick Actions
              </motion.p>
            )}
          </AnimatePresence>

          <div className="space-y-1">
            {categories.map((category, index) => (
              <motion.button
                key={category.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 + index * 0.05, ease: [0.4, 0, 0.2, 1] }}
                onClick={() => handleCategoryClick(category)}
                className={`sidebar-item w-full ${activeCategory === category.id ? 'active' : ''} ${!isExpanded ? 'justify-center px-2' : ''
                  }`}
              >
                <category.icon className={`w-5 h-5 flex-shrink-0 ${activeCategory === category.id ? 'text-gold' : 'text-muted-foreground'
                  }`} />
                <AnimatePresence>
                  {isExpanded && (
                    <motion.span
                      initial={{ opacity: 0, width: 0 }}
                      animate={{ opacity: 1, width: 'auto' }}
                      exit={{ opacity: 0, width: 0 }}
                      className="whitespace-nowrap overflow-hidden"
                    >
                      {category.label}
                    </motion.span>
                  )}
                </AnimatePresence>
              </motion.button>
            ))}
          </div>

          <AnimatePresence>
            {isExpanded && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ delay: 0.3 }}
                className="mt-8"
              >
                <div className="flex items-center gap-2 mb-4 px-2">
                  <TrendingUp className="w-4 h-4 text-gold" />
                  <p className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
                    Trending
                  </p>
                </div>

                <div className="space-y-2">
                  {trendingQueries.map((item, index) => (
                    <motion.button
                      key={item.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.4 + index * 0.1 }}
                      onClick={() => onQuerySelect(item.query)}
                      className="trending-card w-full text-left group"
                    >
                      <p className="text-sm text-foreground/90 leading-relaxed mb-2 group-hover:text-foreground transition-colors">
                        {item.query}
                      </p>
                      <span className="text-[10px] text-muted-foreground bg-surface-elevated px-2 py-1 rounded-full">
                        {item.category}
                      </span>
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="p-4 border-t border-sidebar-border space-y-4">
          <button
            onClick={onOpenBrowser}
            className={`sidebar-item w-full bg-surface-elevated hover:bg-surface-2 border border-border/50 ${!isExpanded ? 'justify-center px-2' : ''}`}
          >
            <BookOpen className="w-5 h-5 flex-shrink-0 text-muted-foreground" />
            <AnimatePresence>
              {isExpanded && (
                <motion.span
                  initial={{ opacity: 0, width: 0 }}
                  animate={{ opacity: 1, width: 'auto' }}
                  exit={{ opacity: 0, width: 0 }}
                  className="whitespace-nowrap overflow-hidden text-sm"
                >
                  Browse Health Topics
                </motion.span>
              )}
            </AnimatePresence>
          </button>

          <AnimatePresence>
            {isExpanded && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-2 px-2"
              >
                <div className="w-2 h-2 rounded-full bg-success animate-pulse-soft" />
                <span className="text-xs text-muted-foreground">System Online</span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      <button
        onClick={() => onToggleExpanded(!isExpanded)}
        className="absolute -right-3 top-1/2 -translate-y-1/2 w-6 h-12 bg-surface-2 border border-border rounded-r-lg hidden md:flex items-center justify-center hover:bg-surface-3 transition-colors z-10"
      >
        {isExpanded ? (
          <ChevronLeft className="w-4 h-4 text-muted-foreground" />
        ) : (
          <ChevronRight className="w-4 h-4 text-muted-foreground" />
        )}
      </button>
    </motion.aside>
  );
}
