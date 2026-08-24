"use client";

import { motion } from 'framer-motion';

export function ThinkingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -10, scale: 0.98 }}
      transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
      className="w-full"
    >
      <div className="card-bot rounded-2xl p-5 md:p-6 relative overflow-hidden">
        <div className="absolute inset-0 overflow-hidden">
          <motion.div
            className="absolute inset-y-0 w-1/2 bg-gradient-to-r from-transparent via-gold/10 to-transparent skew-x-12"
            animate={{ x: ['-100%', '200%'] }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: 'linear',
            }}
          />
        </div>

        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-5">
            <motion.div
              className="w-8 h-8 rounded-xl bg-gold/20 flex items-center justify-center"
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            >
              <motion.div
                className="w-3 h-3 rounded-full bg-gold"
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
            </motion.div>
            <span className="text-xs uppercase tracking-[0.15em] text-gold font-medium">
              Analyzing Query
            </span>
          </div>

          <div className="space-y-3">
            {[85, 70, 55].map((width, index) => (
              <motion.div
                key={index}
                className="h-3 rounded-full bg-surface-elevated"
                style={{ width: `${width}%` }}
                animate={{ opacity: [0.3, 0.6, 0.3] }}
                transition={{
                  duration: 1.5,
                  repeat: Infinity,
                  delay: index * 0.15,
                }}
              />
            ))}
          </div>
        </div>

        <motion.div
          className="absolute inset-0 rounded-2xl pointer-events-none"
          style={{ border: '1px solid' }}
          animate={{
            borderColor: [
              'hsla(38, 92%, 50%, 0.1)',
              'hsla(38, 92%, 50%, 0.3)',
              'hsla(38, 92%, 50%, 0.1)',
            ],
          }}
          transition={{ duration: 2, repeat: Infinity }}
        />
      </div>
    </motion.div>
  );
}
