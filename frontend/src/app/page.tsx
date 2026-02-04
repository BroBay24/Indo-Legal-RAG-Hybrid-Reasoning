"use client";

import { ChatInterface } from "@/components/chat/chat-interface";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col">
      <ChatInterface />
    </main>
  );
}
