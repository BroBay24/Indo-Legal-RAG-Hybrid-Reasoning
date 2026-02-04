"use client"

import React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { motion } from "framer-motion"
import { BookOpen, Copy, Check } from "lucide-react"
import { cn } from "@/lib/utils"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { MarkdownRenderer } from "@/components/ui/markdown-renderer"
import { Button } from "@/components/ui/button"
import { Source } from "@/lib/types"

const chatBubbleVariants = cva(
  "group/message relative break-words rounded-2xl px-4 py-3 text-sm",
  {
    variants: {
      isUser: {
        true: "bg-primary text-primary-foreground ml-auto",
        false: "bg-muted text-foreground mr-auto",
      },
      animation: {
        none: "",
        slide: "duration-300 animate-in fade-in-0",
        scale: "duration-300 animate-in fade-in-0 zoom-in-75",
        fade: "duration-500 animate-in fade-in-0",
      },
    },
    compoundVariants: [
      {
        isUser: true,
        animation: "slide",
        class: "slide-in-from-right",
      },
      {
        isUser: false,
        animation: "slide",
        class: "slide-in-from-left",
      },
      {
        isUser: true,
        animation: "scale",
        class: "origin-bottom-right",
      },
      {
        isUser: false,
        animation: "scale",
        class: "origin-bottom-left",
      },
    ],
  }
)

type Animation = VariantProps<typeof chatBubbleVariants>["animation"]

export interface ChatMessageProps {
  id: string
  role: "user" | "assistant"
  content: string
  createdAt?: Date
  showTimeStamp?: boolean
  animation?: Animation
  actions?: React.ReactNode
  sources?: Source[]
}

export function ChatMessage({
  role,
  content,
  createdAt,
  showTimeStamp = false,
  animation = "scale",
  actions,
  sources,
}: ChatMessageProps) {
  const isUser = role === "user"
  const [copied, setCopied] = React.useState(false)

  const formattedTime = createdAt?.toLocaleTimeString("id-ID", {
    hour: "2-digit",
    minute: "2-digit",
  })

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error("Failed to copy:", err)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={cn("flex gap-3", isUser ? "flex-row-reverse" : "flex-row")}
    >
      {/* Avatar */}
      <Avatar className="h-8 w-8 shrink-0 border shadow-sm">
        <AvatarFallback
          className={cn(
            "text-xs font-medium",
            isUser
              ? "bg-secondary text-secondary-foreground"
              : "bg-primary text-primary-foreground"
          )}
        >
          {isUser ? "U" : "AI"}
        </AvatarFallback>
      </Avatar>

      {/* Message Content */}
      <div
        className={cn(
          "flex flex-col max-w-[80%]",
          isUser ? "items-end" : "items-start"
        )}
      >
        <div className={cn(chatBubbleVariants({ isUser, animation }))}>
          {/* Content */}
          {isUser ? (
            <p className="whitespace-pre-wrap">{content}</p>
          ) : (
            <MarkdownRenderer>{content}</MarkdownRenderer>
          )}

          {/* Sources for assistant */}
          {!isUser && sources && sources.length > 0 && (
            <div className="mt-3 pt-3 border-t border-border/50">
              <p className="text-xs font-medium mb-2 flex items-center gap-1.5 text-muted-foreground">
                <BookOpen className="h-3 w-3" />
                Sumber ({sources.length})
              </p>
              <div className="space-y-1.5">
                {sources.slice(0, 3).map((source, idx) => {
                  const fileName = source.source.split("/").pop() || source.source
                  return (
                    <div
                      key={idx}
                      className="text-xs bg-background/80 rounded-md px-2 py-1.5 truncate border border-border/50"
                    >
                      📄 {fileName} • Hal. {source.page}
                    </div>
                  )
                })}
                {sources.length > 3 && (
                  <p className="text-xs text-muted-foreground/70 pl-2">
                    +{sources.length - 3} sumber lainnya
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Hover Actions for Assistant */}
          {!isUser && (
            <div className="absolute -bottom-3 right-2 flex space-x-1 rounded-lg border bg-background px-1 py-0.5 text-foreground opacity-0 shadow-sm transition-all duration-200 group-hover/message:opacity-100 group-hover/message:-bottom-4">
              <Button
                size="icon"
                variant="ghost"
                className="h-6 w-6"
                onClick={handleCopy}
              >
                {copied ? (
                  <Check className="h-3 w-3 text-green-500" />
                ) : (
                  <Copy className="h-3 w-3" />
                )}
              </Button>
              {actions}
            </div>
          )}
        </div>

        {/* Timestamp */}
        {showTimeStamp && createdAt && (
          <time
            dateTime={createdAt.toISOString()}
            className={cn(
              "mt-1 px-1 text-[10px] text-muted-foreground/60",
              animation !== "none" && "duration-500 animate-in fade-in-0"
            )}
          >
            {formattedTime}
          </time>
        )}
      </div>
    </motion.div>
  )
}
