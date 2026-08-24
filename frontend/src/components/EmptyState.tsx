"use client";

import { motion } from 'framer-motion';
import { Sparkles, Zap, Shield, ArrowRight, Brain, BookOpen } from 'lucide-react';

interface EmptyStateProps {
  onQuerySelect: (query: string) => void;
  onOpenBrowser: () => void;
}

const features = [
  {
    icon: Brain,
    title: 'Smart Triage',
    description: 'AI-powered symptom analysis with emergency detection',
  },
  {
    icon: Zap,
    title: 'Instant Response',
    description: 'Get healthcare guidance in real-time',
  },
  {
    icon: Shield,
    title: 'Medical Knowledge',
    description: 'Backed by healthcare expertise and protocols',
  },
];

const exampleQueries = [
  "I have a headache and fever",
  "My child has been coughing for 3 days",
  "I feel chest pain when breathing",
];

export function EmptyState({ onQuerySelect, onOpenBrowser }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] px-4">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8 }}
        className="text-center max-w-2xl"
      >
        <motion.div
          initial={{ scale: 0, rotate: -180 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{
            type: "spring",
            stiffness: 200,
            damping: 20,
            delay: 0.1
          }}
          className="relative inline-flex mb-10"
        >
          <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-gold via-gold to-gold-dim flex items-center justify-center glow-gold">
            <Sparkles className="w-12 h-12 text-surface-0" />
          </div>
          <motion.div
            className="absolute w-3 h-3 rounded-full bg-gold"
            animate={{
              rotate: 360,
            }}
            transition={{
              duration: 4,
              repeat: Infinity,
              ease: "linear",
            }}
            style={{
              top: '50%',
              left: '50%',
              transformOrigin: '-30px 0px',
            }}
          />
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="text-3xl sm:text-4xl md:text-5xl font-bold mb-4 tracking-tight"
        >
          <span className="text-foreground">MediCare AI</span>{' '}
          <span className="text-gradient-gold">AI</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="text-muted-foreground text-base md:text-lg mb-8 max-w-md mx-auto leading-relaxed"
        >
          Your AI healthcare assistant - describe your symptoms to get guidance
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-8"
        >
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 + index * 0.1 }}
              whileHover={{ y: -4, scale: 1.02 }}
              style={{ transition: 'transform 0.15s ease-out' }}
              className="glass rounded-xl p-4 md:p-5 text-left cursor-default group hover:border-gold/20 hover:shadow-lg"
            >
              <div className="w-10 h-10 rounded-xl bg-surface-elevated flex items-center justify-center mb-3 group-hover:bg-gold/10 transition-colors duration-150">
                <feature.icon className="w-5 h-5 text-muted-foreground group-hover:text-gold transition-colors duration-150" />
              </div>
              <h3 className="text-sm font-semibold text-foreground mb-1.5">
                {feature.title}
              </h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.9 }}
        >
          <p className="text-xs text-muted-foreground/60 mb-4 uppercase tracking-wider">
            Try asking
          </p>
          <div className="flex flex-wrap justify-center gap-2">
            {exampleQueries.map((query, index) => (
              <motion.button
                key={query}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 1 + index * 0.1 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => onQuerySelect(query)}
                className="px-4 py-2.5 rounded-full bg-surface-2 border border-border/50 text-sm text-foreground/80 hover:border-gold/30 hover:text-foreground transition-all duration-300 flex items-center gap-2 group"
              >
                {query}
                <ArrowRight className="w-3.5 h-3.5 text-muted-foreground group-hover:text-gold transition-colors" />
              </motion.button>
            ))}

            <motion.button
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 1.3 }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
              onClick={onOpenBrowser}
              className="px-4 py-2.5 rounded-full bg-gold/10 border border-gold/30 text-sm text-gold hover:bg-gold/20 transition-all duration-300 flex items-center gap-2 group shadow-lg shadow-gold/5"
            >
              <BookOpen className="w-3.5 h-3.5" />
              Browse Health Topics
            </motion.button>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}
