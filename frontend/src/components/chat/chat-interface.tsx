"use client"

import React, { useState, useRef, useEffect, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Scale, RefreshCw, X, BookOpen } from "lucide-react"
import { Button } from "@/components/ui/button"
import { TooltipProvider } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { Message, Source, ChatResponse } from "@/lib/types"
import { MessageList } from "@/components/ui/message-list"
import { MessageInput } from "@/components/ui/message-input"
import { PromptSuggestions } from "@/components/ui/prompt-suggestions"
import { ThemeToggle } from "@/components/theme-toggle"

// Gunakan proxy Next.js jika tersedia, atau direct URL dari environment
const BACKEND_URL = "/api/chat"

// Log untuk debugging
if (typeof window !== "undefined") {
  console.log("Backend Proxy URL:", BACKEND_URL)
}

const EXAMPLE_PROMPTS = [
  "Jelaskan pertimbangan hukum dalam putusan kasasi",
  "Apa dasar hukum pembatalan jual beli tanah?",
  "Bagaimana prosedur pengajuan kasasi?",
  "Jelaskan tentang wanprestasi dalam hukum perdata",
]

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [currentSources, setCurrentSources] = useState<Source[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true)

  const scrollToBottom = useCallback(() => {
    if (messagesContainerRef.current && shouldAutoScroll) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight
    }
  }, [shouldAutoScroll])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  const handleScroll = useCallback(() => {
    if (messagesContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 100
      setShouldAutoScroll(isAtBottom)
    }
  }, [])

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      createdAt: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)
    setCurrentSources([])
    setShouldAutoScroll(true)

    try {
      console.log("Sending request to:", BACKEND_URL)
      
      const response = await fetch(BACKEND_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          pertanyaan: userMessage.content,
          top_k: 5,
          max_tokens: 400,
          temperature: 0.5,
          include_context: false,
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        console.error("Response error:", errorText)
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: ChatResponse = await response.json()

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.jawaban,
        sources: data.sumber,
        createdAt: new Date(),
      }

      setMessages((prev) => [...prev, assistantMessage])
      setCurrentSources(data.sumber || [])
    } catch (error) {
      console.error("Error sending message:", error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `Maaf, terjadi kesalahan saat menghubungi server. ${error instanceof Error ? error.message : "Silakan coba lagi."}`,
        createdAt: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
  }

  const clearChat = () => {
    setMessages([])
    setCurrentSources([])
    setInput("")
  }

  const handleSuggestionSelect = (suggestion: string) => {
    setInput(suggestion)
  }

  const isEmpty = messages.length === 0

  return (
    <TooltipProvider>
      <div className="flex h-screen bg-gradient-to-b from-background via-background to-muted/20">
        {/* Main Chat Area */}
        <div className="flex flex-1 flex-col min-w-0">
          {/* Header */}
          <motion.header 
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="sticky top-0 z-10 border-b bg-background/80 backdrop-blur-lg"
          >
            <div className="flex h-16 items-center justify-between px-4 md:px-6">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-md">
                  <Scale className="h-5 w-5" />
                </div>
                <div>
                  <h1 className="text-lg font-semibold tracking-tight">JustisiaAi</h1>
                  <p className="text-xs text-muted-foreground">AI Legal Assistant</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {messages.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                  >
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={clearChat}
                      className="gap-2 hover:bg-destructive/10 hover:text-destructive"
                    >
                      <RefreshCw className="h-4 w-4" />
                      <span className="hidden sm:inline">Percakapan Baru</span>
                    </Button>
                  </motion.div>
                )}
                <ThemeToggle />
              </div>
            </div>
          </motion.header>

          {/* Messages Area */}
          <div 
            ref={messagesContainerRef}
            onScroll={handleScroll}
            className="flex-1 overflow-y-auto"
          >
            <div className="mx-auto max-w-3xl px-4 py-6">
              <AnimatePresence mode="wait">
                {isEmpty ? (
                  <motion.div
                    key="welcome"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="flex h-full min-h-[60vh] flex-col items-center justify-center py-12"
                  >
                    {/* Welcome Icon */}
                    <motion.div 
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", stiffness: 200, damping: 15, delay: 0.1 }}
                      className="mb-6 flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-primary/80 text-primary-foreground shadow-xl"
                    >
                      <Scale className="h-10 w-10" />
                    </motion.div>

                    {/* Welcome Text */}
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.2 }}
                      className="text-center mb-8"
                    >
                      <h2 className="text-2xl font-bold tracking-tight mb-2">
                        Selamat Datang di JUSTISIA AI
                      </h2>
                      <p className="text-muted-foreground max-w-md">
                        AI Assistant untuk menganalisis dokumen putusan pengadilan Indonesia.
                        Tanyakan apa saja tentang hukum perdata dan kasasi.
                      </p>
                    </motion.div>

                    {/* Suggestions */}
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                      className="w-full max-w-2xl"
                    >
                      <PromptSuggestions
                        suggestions={EXAMPLE_PROMPTS}
                        onSelect={handleSuggestionSelect}
                        label="Contoh pertanyaan"
                      />
                    </motion.div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="messages"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    <MessageList 
                      messages={messages} 
                      isTyping={isLoading}
                      showTimeStamps
                    />
                    <div ref={messagesEndRef} />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Input Area */}
          <motion.div 
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="border-t bg-background/80 backdrop-blur-lg p-4"
          >
            <div className="mx-auto max-w-3xl">
              <MessageInput
                value={input}
                onChange={handleInputChange}
                onSubmit={sendMessage}
                isGenerating={isLoading}
                placeholder="Tanyakan tentang dokumen hukum... (Enter untuk kirim)"
                disabled={isLoading}
              />
              <p className="mt-2 text-center text-xs text-muted-foreground">
                RAG Hukum Indonesia menggunakan AI untuk menganalisis dokumen putusan pengadilan.
              </p>
            </div>
          </motion.div>
        </div>

        {/* Sources Sidebar */}
        <AnimatePresence>
          {currentSources.length > 0 && (
            <SourcesPanel 
              sources={currentSources} 
              onClose={() => setCurrentSources([])} 
            />
          )}
        </AnimatePresence>
      </div>
    </TooltipProvider>
  )
}

// Sources Panel Component
interface SourcesPanelProps {
  sources: Source[]
  onClose: () => void
}

function SourcesPanel({ sources, onClose }: SourcesPanelProps) {
  return (
    <motion.aside
      initial={{ x: 300, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 300, opacity: 0 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
      className="hidden md:flex w-80 flex-col border-l bg-background/50 backdrop-blur-sm"
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b p-4">
        <div className="flex items-center gap-2">
          <BookOpen className="h-4 w-4 text-primary" />
          <h2 className="font-semibold">Sumber Referensi</h2>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose} className="h-8 w-8">
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Sources List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {sources.map((source, idx) => {
          const fileName = source.source.split("/").pop() || source.source
          return (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="rounded-xl border bg-card p-4 shadow-sm hover:shadow-md transition-shadow"
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-sm truncate">{fileName}</p>
                  <p className="text-xs text-muted-foreground">Halaman {source.page}</p>
                </div>
                {source.relevance_score !== undefined && (
                  <span className="shrink-0 ml-2 text-xs font-medium bg-primary/10 text-primary px-2 py-1 rounded-full">
                    {(source.relevance_score * 100).toFixed(0)}%
                  </span>
                )}
              </div>
              {source.content && (
                <p className="text-xs text-muted-foreground line-clamp-4 mt-2 border-t pt-2">
                  {source.content}
                </p>
              )}
            </motion.div>
          )
        })}
      </div>
    </motion.aside>
  )
}
