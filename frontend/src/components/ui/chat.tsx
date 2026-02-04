"use client"

import React, { forwardRef, useRef, useEffect, useCallback } from "react"
import { ArrowDown } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { MessageList } from "@/components/ui/message-list"
import { MessageInput } from "@/components/ui/message-input"
import { PromptSuggestions } from "@/components/ui/prompt-suggestions"
import { Message } from "@/lib/types"

interface ChatProps {
  messages: Message[]
  input: string
  handleInputChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void
  handleSubmit: () => void
  isGenerating: boolean
  stop?: () => void
  suggestions?: string[]
  onSuggestionSelect?: (suggestion: string) => void
  className?: string
}

export function Chat({
  messages,
  input,
  handleInputChange,
  handleSubmit,
  isGenerating,
  stop,
  suggestions,
  onSuggestionSelect,
  className,
}: ChatProps) {
  const isEmpty = messages.length === 0
  const lastMessage = messages.at(-1)
  const isTyping = isGenerating && lastMessage?.role === "user"

  return (
    <ChatContainer className={className}>
      {isEmpty && suggestions && onSuggestionSelect ? (
        <div className="flex flex-1 items-center justify-center p-4">
          <PromptSuggestions
            suggestions={suggestions}
            onSelect={onSuggestionSelect}
            className="max-w-2xl"
          />
        </div>
      ) : null}

      {messages.length > 0 ? (
        <ChatMessages messages={messages}>
          <MessageList messages={messages} isTyping={isTyping} />
        </ChatMessages>
      ) : null}

      <div className="mt-auto p-4">
        <MessageInput
          value={input}
          onChange={handleInputChange}
          onSubmit={handleSubmit}
          isGenerating={isGenerating}
          stop={stop}
          placeholder="Tanyakan tentang dokumen hukum..."
        />
      </div>
    </ChatContainer>
  )
}

// Chat Container
export const ChatContainer = forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn("grid max-h-full w-full grid-rows-[1fr_auto]", className)}
      {...props}
    />
  )
})
ChatContainer.displayName = "ChatContainer"

// Chat Messages with auto-scroll
interface ChatMessagesProps {
  messages: Message[]
  children: React.ReactNode
}

export function ChatMessages({ messages, children }: ChatMessagesProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [shouldAutoScroll, setShouldAutoScroll] = React.useState(true)

  const scrollToBottom = useCallback(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [])

  // Auto-scroll when new messages arrive
  useEffect(() => {
    if (shouldAutoScroll) {
      scrollToBottom()
    }
  }, [messages, shouldAutoScroll, scrollToBottom])

  const handleScroll = useCallback(() => {
    if (containerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = containerRef.current
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 100
      setShouldAutoScroll(isAtBottom)
    }
  }, [])

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      className="relative grid grid-cols-1 overflow-y-auto px-4 pb-4"
    >
      <div className="max-w-3xl mx-auto w-full py-4">
        {children}
      </div>

      {/* Scroll to bottom button */}
      {!shouldAutoScroll && (
        <div className="pointer-events-none absolute bottom-4 left-0 right-0 flex justify-center">
          <Button
            onClick={() => {
              scrollToBottom()
              setShouldAutoScroll(true)
            }}
            className="pointer-events-auto h-8 w-8 rounded-full shadow-lg animate-in fade-in-0 slide-in-from-bottom-2"
            size="icon"
            variant="secondary"
          >
            <ArrowDown className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  )
}
