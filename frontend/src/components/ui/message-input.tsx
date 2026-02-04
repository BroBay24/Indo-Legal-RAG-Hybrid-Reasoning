"use client"

import React, { useRef, useEffect, useState, useCallback } from "react"
import { ArrowUp, Square, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

interface MessageInputProps {
  value: string
  onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void
  onSubmit: () => void
  placeholder?: string
  className?: string
  disabled?: boolean
  isGenerating?: boolean
  stop?: () => void
}

export function MessageInput({
  value,
  onChange,
  onSubmit,
  placeholder = "Ketik pesan...",
  className,
  disabled = false,
  isGenerating = false,
  stop,
}: MessageInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const [isFocused, setIsFocused] = useState(false)

  // Auto-resize textarea
  const adjustHeight = useCallback(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = "auto"
      const maxHeight = 200
      textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`
    }
  }, [])

  useEffect(() => {
    adjustHeight()
  }, [value, adjustHeight])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      if (!disabled && !isGenerating && value.trim()) {
        onSubmit()
      }
    }
  }

  return (
    <div
      className={cn(
        "relative flex w-full items-end gap-2 rounded-2xl border bg-background p-2 shadow-sm transition-all duration-200",
        isFocused && "border-primary/50 ring-2 ring-primary/20",
        className
      )}
    >
      {/* Textarea */}
      <div className="relative flex-1">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={onChange}
          onKeyDown={handleKeyDown}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder={placeholder}
          disabled={disabled || isGenerating}
          rows={1}
          className={cn(
            "w-full resize-none bg-transparent px-2 py-2 text-sm outline-none",
            "placeholder:text-muted-foreground/60",
            "disabled:cursor-not-allowed disabled:opacity-50",
            "max-h-[200px] min-h-[40px]"
          )}
        />
      </div>

      {/* Submit/Stop Button */}
      <div className="flex shrink-0 items-center gap-1">
        {isGenerating && stop ? (
          <Button
            type="button"
            size="icon"
            variant="destructive"
            className="h-9 w-9 rounded-xl transition-all duration-200 hover:scale-105"
            onClick={stop}
          >
            <Square className="h-4 w-4" fill="currentColor" />
          </Button>
        ) : (
          <Button
            type="button"
            size="icon"
            className={cn(
              "h-9 w-9 rounded-xl transition-all duration-200",
              value.trim() && !disabled && "hover:scale-105"
            )}
            onClick={onSubmit}
            disabled={!value.trim() || disabled || isGenerating}
          >
            {isGenerating ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <ArrowUp className="h-4 w-4" />
            )}
          </Button>
        )}
      </div>
    </div>
  )
}
