"use client"

import React from "react"
import { ChatMessage } from "@/components/ui/chat-message"
import { TypingIndicator } from "@/components/ui/typing-indicator"
import { cn } from "@/lib/utils"
import { Source } from "@/lib/types"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  createdAt?: Date
  sources?: Source[]
}

interface MessageListProps {
  messages: Message[]
  showTimeStamps?: boolean
  isTyping?: boolean
  className?: string
}

export function MessageList({
  messages,
  showTimeStamps = false,
  isTyping = false,
  className,
}: MessageListProps) {
  return (
    <div className={cn("space-y-6", className)}>
      {messages.map((message) => (
        <ChatMessage
          key={message.id}
          id={message.id}
          role={message.role}
          content={message.content}
          createdAt={message.createdAt}
          showTimeStamp={showTimeStamps}
          sources={message.sources}
          animation="scale"
        />
      ))}
      {isTyping && (
        <div className="flex items-start gap-3">
          <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground text-xs font-medium shadow-sm border">
            AI
          </div>
          <div className="bg-muted rounded-2xl px-4 py-3">
            <TypingIndicator />
          </div>
        </div>
      )}
    </div>
  )
}
