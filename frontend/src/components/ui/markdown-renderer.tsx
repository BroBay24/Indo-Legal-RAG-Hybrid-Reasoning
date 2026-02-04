"use client"

import React from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { cn } from "@/lib/utils"

interface MarkdownRendererProps {
  children: string
  className?: string
}

export function MarkdownRenderer({ children, className }: MarkdownRendererProps) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      className={cn(
        "prose prose-sm dark:prose-invert max-w-none",
        "prose-p:leading-relaxed prose-p:my-2",
        "prose-headings:my-2 prose-headings:font-semibold",
        "prose-ul:my-2 prose-ol:my-2",
        "prose-li:my-0.5",
        "prose-pre:bg-muted prose-pre:rounded-lg prose-pre:p-3",
        "prose-code:bg-muted prose-code:rounded prose-code:px-1 prose-code:py-0.5 prose-code:text-sm",
        "prose-blockquote:border-l-primary prose-blockquote:bg-muted/50 prose-blockquote:py-1 prose-blockquote:px-4",
        "prose-a:text-primary prose-a:no-underline hover:prose-a:underline",
        "[&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
        className
      )}
      components={{
        p: ({ children }) => (
          <p className="whitespace-pre-wrap">{children}</p>
        ),
        code: ({ className, children, ...props }) => {
          const match = /language-(\w+)/.exec(className || "")
          const isInline = !match && typeof children === "string" && !children.includes("\n")
          
          if (isInline) {
            return (
              <code
                className="bg-muted rounded px-1.5 py-0.5 text-sm font-mono"
                {...props}
              >
                {children}
              </code>
            )
          }
          
          return (
            <code className={cn("block", className)} {...props}>
              {children}
            </code>
          )
        },
        pre: ({ children }) => (
          <pre className="overflow-x-auto rounded-lg bg-muted p-4 text-sm">
            {children}
          </pre>
        ),
        table: ({ children }) => (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-border">
              {children}
            </table>
          </div>
        ),
      }}
    >
      {children}
    </ReactMarkdown>
  )
}
