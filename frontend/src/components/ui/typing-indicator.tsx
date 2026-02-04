"use client"

import { cn } from "@/lib/utils"

interface TypingIndicatorProps {
  className?: string
}

export function TypingIndicator({ className }: TypingIndicatorProps) {
  return (
    <div className={cn("flex items-center gap-1", className)}>
      <div className="flex space-x-1">
        <div
          className="h-2 w-2 rounded-full bg-muted-foreground/50 animate-bounce"
          style={{ animationDelay: "0ms", animationDuration: "1s" }}
        />
        <div
          className="h-2 w-2 rounded-full bg-muted-foreground/50 animate-bounce"
          style={{ animationDelay: "150ms", animationDuration: "1s" }}
        />
        <div
          className="h-2 w-2 rounded-full bg-muted-foreground/50 animate-bounce"
          style={{ animationDelay: "300ms", animationDuration: "1s" }}
        />
      </div>
    </div>
  )
}
