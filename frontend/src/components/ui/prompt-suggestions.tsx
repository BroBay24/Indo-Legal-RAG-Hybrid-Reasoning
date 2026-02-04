"use client"

import React from "react"
import { motion } from "framer-motion"
import { Sparkles } from "lucide-react"
import { cn } from "@/lib/utils"

interface PromptSuggestionsProps {
  suggestions: string[]
  onSelect: (suggestion: string) => void
  label?: string
  className?: string
}

export function PromptSuggestions({
  suggestions,
  onSelect,
  label = "Coba pertanyaan berikut",
  className,
}: PromptSuggestionsProps) {
  return (
    <div className={cn("w-full", className)}>
      <p className="mb-3 text-sm font-medium text-muted-foreground flex items-center gap-2">
        <Sparkles className="h-4 w-4" />
        {label}
      </p>
      <div className="grid gap-2 sm:grid-cols-2">
        {suggestions.map((suggestion, idx) => (
          <motion.button
            key={idx}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1, duration: 0.3 }}
            onClick={() => onSelect(suggestion)}
            className={cn(
              "group relative overflow-hidden rounded-xl border bg-card p-4 text-left text-sm",
              "transition-all duration-200",
              "hover:border-primary/50 hover:bg-accent hover:shadow-md",
              "focus:outline-none focus:ring-2 focus:ring-primary/20"
            )}
          >
            <div className="relative z-10 line-clamp-2">{suggestion}</div>
            <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-transparent opacity-0 transition-opacity duration-200 group-hover:opacity-100" />
          </motion.button>
        ))}
      </div>
    </div>
  )
}
