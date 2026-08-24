"use client";

import { motion } from 'framer-motion';

interface ConfidenceMeterProps {
  score: number;
  size?: number;
}

export function ConfidenceMeter({ score, size = 36 }: ConfidenceMeterProps) {
  const percentage = Math.round(score * 100);
  const radius = 17;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - score * circumference;

  const getColorClass = () => {
    if (score >= 0.8) return 'confidence-high';
    if (score >= 0.5) return 'confidence-medium';
    return 'confidence-low';
  };

  const getLabel = () => {
    if (score >= 0.8) return 'High';
    if (score >= 0.5) return 'Medium';
    return 'Low';
  };

  const getStrokeColor = () => {
    if (score >= 0.8) return 'hsl(145, 65%, 42%)';
    if (score >= 0.5) return 'hsl(38, 92%, 50%)';
    return 'hsl(0, 70%, 55%)';
  };

  return (
    <div className="flex items-center gap-3">
      <div className="relative" style={{ width: size, height: size }}>
        <svg
          className="transform -rotate-90"
          width={size}
          height={size}
          viewBox="0 0 44 44"
        >
          {/* Background Circle */}
          <circle
            cx="22"
            cy="22"
            r={radius}
            fill="none"
            stroke="hsl(0, 0%, 15%)"
            strokeWidth="3"
          />
          {/* Progress Circle */}
          <motion.circle
            cx="22"
            cy="22"
            r={radius}
            fill="none"
            stroke={getStrokeColor()}
            strokeWidth="3"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 1.2, ease: [0.4, 0, 0.2, 1] }}
            style={{
              filter: `drop-shadow(0 0 6px ${getStrokeColor()})`,
            }}
          />
        </svg>
        {/* Center Text */}
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.span
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5 }}
            className="text-xs font-semibold text-foreground"
          >
            {percentage}
          </motion.span>
        </div>
      </div>

      <div className="flex flex-col">
        <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
          Confidence
        </span>
        <span className={`text-xs font-semibold ${getColorClass()}`}>
          {getLabel()}
        </span>
      </div>
    </div>
  );
}
