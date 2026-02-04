"use client";

import React from "react";
import { X, FileText, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Source } from "@/lib/types";
import { cn } from "@/lib/utils";

interface SourcesPanelProps {
  sources: Source[];
  onClose: () => void;
}

export function SourcesPanel({ sources, onClose }: SourcesPanelProps) {
  return (
    <aside className="hidden lg:flex w-80 flex-col border-l bg-muted/30">
      <div className="flex h-16 items-center justify-between border-b px-4">
        <h2 className="font-semibold flex items-center gap-2">
          <FileText className="h-4 w-4" />
          Sumber Referensi
        </h2>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-3">
          {sources.map((source, idx) => (
            <SourceCard key={idx} source={source} index={idx + 1} />
          ))}
        </div>
      </div>
    </aside>
  );
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  const fileName = source.source.split("/").pop() || source.source;
  const relevanceScore = Math.min(100, Math.max(0, source.score * 100));
  
  // Color based on score
  const scoreColor = 
    relevanceScore >= 70 ? "text-green-600 bg-green-50" :
    relevanceScore >= 40 ? "text-yellow-600 bg-yellow-50" :
    "text-red-600 bg-red-50";

  return (
    <Card className="overflow-hidden">
      <CardHeader className="p-3 pb-2">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary text-xs font-medium text-primary-foreground">
              {index}
            </span>
            <CardTitle className="text-sm font-medium line-clamp-2">
              {fileName}
            </CardTitle>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-3 pt-0">
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2 text-xs">
            <span className="inline-flex items-center rounded-md bg-secondary px-2 py-1">
              Halaman {source.page}
            </span>
            <span className="inline-flex items-center rounded-md bg-secondary px-2 py-1 capitalize">
              {source.doc_type}
            </span>
            <span className={cn("inline-flex items-center rounded-md px-2 py-1", scoreColor)}>
              {relevanceScore.toFixed(1)}%
            </span>
          </div>
          <div className="text-xs text-muted-foreground">
            <span className="capitalize">{source.retrieval_source}</span> retrieval
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Mobile-friendly sources modal (for smaller screens)
export function SourcesModal({ 
  sources, 
  isOpen, 
  onClose 
}: { 
  sources: Source[]; 
  isOpen: boolean; 
  onClose: () => void;
}) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 lg:hidden">
      <div 
        className="absolute inset-0 bg-background/80 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="absolute bottom-0 left-0 right-0 max-h-[70vh] rounded-t-xl bg-background shadow-lg">
        <div className="flex items-center justify-between border-b p-4">
          <h2 className="font-semibold flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Sumber Referensi ({sources.length})
          </h2>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="overflow-y-auto p-4 max-h-[calc(70vh-60px)]">
          <div className="space-y-3">
            {sources.map((source, idx) => (
              <SourceCard key={idx} source={source} index={idx + 1} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
